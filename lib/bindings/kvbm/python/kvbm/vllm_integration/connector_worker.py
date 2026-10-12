# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Implementation of vLLM KV cache manager protocol.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
from kvbm.utils import is_dyn_runtime_enabled
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.model_executor.models.utils import extract_layer_index
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext


# from kvbm.vllm_integration.kv_cache_utils import KvbmCacheBlocks
# from kvbm.vllm_integration.rust import BlockManager
# from kvbm.vllm_integration.rust import (
#     KvConnectorMetadata as RustKvConnectorMetadata,
#     KvConnectorWorker as RustKvConnectorWorker,
# )

from kvbm.vllm_integration.rust import KvConnectorWorker as RustKvConnectorWorker

from .group_plan import GroupPlan, derive_layer_groups

logger = logging.getLogger(__name__)


@dataclass
class KvTensorLayout:
    """Semantic description of the KV cache tensor layout.

    Python derives this from VllmConfig (which has the model architecture)
    so that Rust does not need to guess from raw tensor shapes.

    For MLA models, outer_dim and inner_dim are set explicitly because Rust's
    shape-based inference cannot distinguish the fused KV latent axis from the
    block/page axis.  For standard attention the fields are None, which tells
    Rust to fall back to its own contiguity-based inference (already correct).
    """

    outer_dim: Optional[int]  # None = let Rust detect; 1 for MLA
    inner_dim: Optional[int]  # None = let Rust detect; head_size for MLA

    @classmethod
    def from_vllm_config(
        cls, vllm_config: "VllmConfig", shape: "torch.Size", use_mla: bool = False
    ) -> "KvTensorLayout":
        if use_mla:
            # MLA tensors are 3D: [num_blocks, page_size, head_size]
            # No outer_dim axis — K and V are fused into a single latent.
            return cls(outer_dim=1, inner_dim=shape[-1])
        else:
            # Standard attention: Rust already infers outer_dim and inner_dim
            # correctly from tensor strides/contiguity.  Don't guess here —
            # the block dimension can be at position 0 or 1 depending on the
            # attention backend, so shape[1] is not reliably outer_dim.
            return cls(outer_dim=None, inner_dim=None)


DistributedRuntime = None
if is_dyn_runtime_enabled():
    from dynamo.runtime import DistributedRuntime


class DynamoConnectorMetadata(KVConnectorMetadata):
    def __init__(self, metadata: bytes):
        assert isinstance(metadata, bytes)
        self.metadata = metadata


def _recover_raw_tensor(state_tensors: "list[torch.Tensor]") -> "torch.Tensor":
    """Recover the flat raw_tensor backing a list of recurrent-state tensors.

    vLLM's hybrid (HMA) mode creates Mamba/GDN state tensors as strided views
    (``torch.as_strided``) into a single flat ``int8`` buffer.  The
    ``raw_tensor`` attribute is not always set on these views, so we
    reconstruct it from the shared ``untyped_storage()``.

    This uses the low-level ``torch.Tensor.set_`` storage pattern (which
    PyTorch discourages) because it is the only way to reach the backing
    buffer from a strided view; it assumes the source tensor's storage is
    valid and its first dimension is the block count.
    """
    raw = getattr(state_tensors[0], "raw_tensor", None)
    if raw is not None:
        return raw
    storage = state_tensors[0].untyped_storage()
    num_blocks = state_tensors[0].shape[0]
    device = state_tensors[0].device
    raw = torch.empty([], dtype=torch.uint8, device=device).set_(storage)
    return raw.view(num_blocks, -1)


class KvConnectorWorker:
    def __init__(self, vllm_config: "VllmConfig", engine_id: str, **kwargs):
        drt: Optional[object] = kwargs.get("drt")

        if drt is None and is_dyn_runtime_enabled():
            drt = DistributedRuntime.detached()
        else:
            drt = None

        self.drt = drt

        self.vllm_config = vllm_config
        self._connector = RustKvConnectorWorker(self.drt, engine_id)
        self.mamba_layer_names: set[str] = set()
        self.group_plans: list[GroupPlan] = []

    # Worker

    def register_kv_caches(
        self, kv_caches: dict[str, torch.Tensor | list[torch.Tensor]]
    ):
        """
        Initialize with the KV caches. Useful for pre-registering the
        KV Caches in the KVConnector (e.g. for NIXL).

        Args: kv_caches:
            dictionary of layer names, kv cache.  Hybrid models (vLLM HMA)
            pass a list[Tensor] for recurrent (Mamba/GDN) layers — strided
            views into one flat backing buffer — instead of a single Tensor.
        """

        cache_config = self.vllm_config.cache_config

        # Create ordered list of (layer_name, tensor) tuples sorted by layer index
        ordered_kv_caches = [
            (layer_name, tensor)
            for layer_name, tensor in sorted(
                kv_caches.items(), key=lambda item: extract_layer_index(item[0])
            )
        ]

        # Partition the layers into HMA groups (vLLM's kv_cache_config) for
        # per-group validation below; a missing config falls back to one
        # group covering every layer.
        group_plans = derive_layer_groups(
            getattr(self.vllm_config, "kv_cache_config", None),
            [name for name, _tensor in ordered_kv_caches],
        )
        self.group_plans = group_plans

        # Normalize recurrent (Mamba/GDN) layers — list[Tensor] strided views —
        # to the flat raw_tensor backing them, so every registered layer is a
        # Tensor with the same [num_blocks, ...] block layout as attention KV.
        attention_layer_count = 0
        recurrent_layer_count = 0
        normalized_kv_caches: list[tuple[str, torch.Tensor]] = []
        mamba_layer_names: set[str] = set()
        for layer_name, value in ordered_kv_caches:
            if isinstance(value, list):
                raw_tensor = _recover_raw_tensor(value)
                mamba_layer_names.add(layer_name)
                recurrent_layer_count += 1
                normalized_kv_caches.append((layer_name, raw_tensor))
                logger.debug(
                    "register_kv_caches: layer=%s list[Tensor] len=%d raw_shape=%s",
                    layer_name,
                    len(value),
                    list(raw_tensor.shape),
                )
            else:
                attention_layer_count += 1
                normalized_kv_caches.append((layer_name, value))
        ordered_kv_caches = normalized_kv_caches
        self.mamba_layer_names = mamba_layer_names

        logger.info(
            "register_kv_caches: %d KV layers (%d attention, %d recurrent/HMA)",
            len(ordered_kv_caches),
            attention_layer_count,
            recurrent_layer_count,
        )
        for group in group_plans:
            logger.info(
                "register_kv_caches: group %d: %d layers (mamba=%s), layers=%s",
                group.group_id,
                len(group.layer_names),
                group.is_mamba,
                group.layer_names[:4],
            )

        events = [
            torch.cuda.Event(enable_timing=False, interprocess=False)
            for _ in range(len(ordered_kv_caches))
        ]

        # events are lazy, if we don't record them once here, the raw handles we pass to rust will be null
        for event in events:
            event.record(torch.cuda.current_stream())

        raw_event_handles = [event.cuda_event for event in events]

        self.events = {
            layer_name: event
            for (layer_name, _tensor), event in zip(ordered_kv_caches, events)
        }

        # num_device_blocks must come from an attention tensor (>= 3 dims):
        # recurrent raw_tensors are 2D [num_blocks, flat_state] where shape[1]
        # is the flattened state size, not the block axis.  Nemotron-style
        # hybrid patterns start with a Mamba layer, so the first tensor cannot
        # be assumed to be attention.
        use_mla = getattr(self.vllm_config.model_config, "use_mla", False)
        attention_tensor = next(
            (t for _n, t in ordered_kv_caches if t.dim() >= 3), None
        )
        if attention_tensor is None:
            ref_tensor = ordered_kv_caches[0][1]
            shape = ref_tensor.shape
            num_device_blocks = shape[0]
        else:
            shape = attention_tensor.shape
            # MLA tensors are [num_blocks, page_size, head_size] — block dim is always axis 0.
            # Standard attention tensors are [outer_dim, num_blocks, ...] or [num_blocks, outer_dim, ...]
            # — block dim is whichever axis is larger, which is unambiguous as long as
            # num_blocks >> outer_dim (always true in practice).
            num_device_blocks = shape[0] if use_mla else max(shape[0], shape[1])

        page_size = cache_config.block_size
        device_id = ordered_kv_caches[0][1].device.index

        # The block manager tracks per-group pools: bytes-per-block must be
        # uniform within each HMA group, not across all layers (attention and
        # Mamba/GDN groups may differ).  With the single-group fallback
        # (kv_cache_config absent) this degrades to the old all-layers check.
        # If a model's recurrent-state config breaks a group's invariant, fail
        # here naming the layer and group instead of corrupting pool metadata
        # downstream.
        def _byte_size_per_block(t: torch.Tensor) -> int:
            return (t.element_size() * t.nelement()) // num_device_blocks

        name_to_tensor = dict(ordered_kv_caches)
        for group in group_plans:
            if not group.layer_names:
                continue
            expected_bspb = _byte_size_per_block(name_to_tensor[group.layer_names[0]])
            for layer_name in group.layer_names:
                actual_bspb = _byte_size_per_block(name_to_tensor[layer_name])
                if actual_bspb != expected_bspb:
                    raise ValueError(
                        f"Layer {layer_name}: byte_size_per_block={actual_bspb} "
                        f"differs from expected={expected_bspb} in group "
                        f"{group.group_id}. All layers in a group must have "
                        "uniform byte-size-per-block for the block manager "
                        "(check the HMA per-group block sizes)."
                    )

        layout = KvTensorLayout.from_vllm_config(self.vllm_config, shape, use_mla)

        # Determine cache dtype
        if cache_config.cache_dtype == "auto":
            kv_cache_dtype = self.vllm_config.model_config.dtype
        else:
            kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        # Register with connector using ordered data
        self._connector.register_kv_caches(
            num_device_blocks,
            page_size,
            device_id,
            kv_cache_dtype.itemsize,
            ordered_kv_caches,
            raw_event_handles,
            outer_dim=layout.outer_dim,
            inner_dim=layout.inner_dim,
        )

    def bind_connector_metadata(self, data: bytes) -> None:
        """Set the connector metadata from the scheduler.

        This function should be called by the model runner every time
        before the model execution. The metadata will be used for runtime
        KV cache loading and saving.

        Args:
            connector_metadata (dict): the connector metadata.
        """
        self._connector.bind_connector_metadata(data)

    def clear_connector_metadata(self) -> None:
        """Clear the connector metadata.

        This function should be called by the model runner every time
        after the model execution.
        """
        self._connector.clear_connector_metadata()

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        """
        Start loading the KV cache from the connector to vLLM's paged
        KV buffer. This is called from the forward context before the
        forward pass to enable async loading during model execution.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be
            the same.

        """
        pass

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        """
        Start saving a layer of KV cache from vLLM's paged buffer
        to the connector. This is called from within attention layer to
        enable async copying during execution.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """
        self.events[layer_name].record(torch.cuda.current_stream())
        self._connector.save_kv_layer(layer_name, kv_layer)

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens on the worker.
        The scheduler process (via the MultiprocExecutor) will use this output
        to track which workers are done.

        Returns:
            ids of requests that have finished asynchronous transfer
            (requests that previously returned True from request_finished()),
            tuple of (sending/saving ids, recving/loading ids).
            The finished saves/sends req ids must belong to a set provided in a
            call to this method (this call or a prior one).
        """
        # finished_ids = [id for id in finished_req_ids]
        # return set(sending_ids), set(receiving_ids)
        return self._connector.get_finished(finished_req_ids)
