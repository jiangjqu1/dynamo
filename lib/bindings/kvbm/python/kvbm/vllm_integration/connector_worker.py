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
    from vllm.v1.kv_cache_interface import KVCacheConfig


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
    def __init__(
        self,
        vllm_config: "VllmConfig",
        engine_id: str,
        kv_cache_config: Optional["KVCacheConfig"] = None,
        **kwargs,
    ):
        drt: Optional[object] = kwargs.get("drt")

        if drt is None and is_dyn_runtime_enabled():
            drt = DistributedRuntime.detached()
        else:
            drt = None

        self.drt = drt

        self.vllm_config = vllm_config
        # vLLM's connector factory hands this to DynamoConnector (the 3-arg
        # contract); the model runner holds it elsewhere on VllmConfig, so
        # this kwarg — not a vllm_config attribute — is the only source.
        self.kv_cache_config = kv_cache_config
        self._connector = RustKvConnectorWorker(self.drt, engine_id)
        self.mamba_layer_names: set[str] = set()
        # Rust handoff (Task 6); fallback: one non-mamba group, page_bytes=None
        self.group_plans: list[GroupPlan] = []

    # Worker

    def register_kv_caches(
        self, kv_caches: dict[str, torch.Tensor | list[torch.Tensor]]
    ):
        """
        Initialize with the KV caches. Useful for pre-registering the
        KV Caches in the KVConnector (e.g. for NIXL).

        Args: kv_caches:
            dictionary of layer names, kv cache.  Hybrid (Mamba/GDN) layers
            arrive either as a plain page tensor ([num_blocks, 1, 1, C]
            int8 — vLLM 0.29) or as a list[Tensor] of strided views into
            one flat backing buffer (vLLM HMA-era); both normalize to the
            flat [num_blocks, C] backing tensor before registration.
        """

        cache_config = self.vllm_config.cache_config

        # Create ordered list of (layer_name, tensor) tuples sorted by layer index
        ordered_kv_caches = [
            (layer_name, tensor)
            for layer_name, tensor in sorted(
                kv_caches.items(), key=lambda item: extract_layer_index(item[0])
            )
        ]

        # Partition the layers into HMA groups (vLLM's kv_cache_config,
        # supplied to the connector by vLLM's factory) for per-group
        # validation below; a missing config falls back to one group
        # covering every layer.  The plan is derived BEFORE normalization:
        # it decides which layers are recurrent.
        kv_cache_config = self.kv_cache_config
        group_plans = derive_layer_groups(
            kv_cache_config, [name for name, _tensor in ordered_kv_caches]
        )
        self.group_plans = group_plans
        if getattr(kv_cache_config, "kv_cache_groups", None):
            config_layer_names = {
                name
                for group in kv_cache_config.kv_cache_groups
                for name in group.layer_names
            }
            unknown_layers = [
                name
                for name, _tensor in ordered_kv_caches
                if name not in config_layer_names
            ]
            if unknown_layers:
                logger.warning(
                    "register_kv_caches: %d layers absent from every "
                    "kv_cache_config group, filed into group 0: %s",
                    len(unknown_layers),
                    unknown_layers[:4],
                )

        # The plans share ONE group_of_layer list parallel to the
        # registration order (GroupPlan docstring); plan[0]'s view is the
        # authoritative mapping.
        group_of_layer = group_plans[0].group_of_layer
        mamba_group_ids = {plan.group_id for plan in group_plans if plan.is_mamba}

        # num_device_blocks must come from an ATTENTION-group tensor, and
        # the plan (not the tensor shape) decides which layers those are:
        # a GDN layer's raw page tensor [num_blocks, 1, 1, C] is 4D and
        # would hijack a shape-based heuristic.  Nemotron-style hybrid
        # patterns can also START with a mamba layer, so "first" alone is
        # not enough.  The isinstance guard covers the single-group
        # fallback (no kv_cache_config) where a list[Tensor] HMA value
        # could come first.
        use_mla = getattr(self.vllm_config.model_config, "use_mla", False)
        attention_tensor = next(
            (
                t
                for (_n, t), gid in zip(ordered_kv_caches, group_of_layer)
                if gid not in mamba_group_ids and isinstance(t, torch.Tensor)
            ),
            None,
        )
        if attention_tensor is None:
            ref_tensor = ordered_kv_caches[0][1]
            if isinstance(ref_tensor, list):
                ref_tensor = _recover_raw_tensor(ref_tensor)
            shape = ref_tensor.shape
            num_device_blocks = shape[0]
        else:
            shape = attention_tensor.shape
            # MLA tensors are [num_blocks, page_size, head_size] — block dim is always axis 0.
            # Standard attention tensors are [outer_dim, num_blocks, ...] or [num_blocks, outer_dim, ...]
            # — block dim is whichever axis is larger, which is unambiguous as long as
            # num_blocks >> outer_dim (always true in practice).
            num_device_blocks = shape[0] if use_mla else max(shape[0], shape[1])

        # Normalize recurrent (Mamba/GDN) layers to a flat 2D
        # [num_blocks, state_bytes] tensor — the same [num_blocks, ...]
        # block layout attention KV registers with.  The PLAN decides
        # which layers are recurrent (not the value type): vLLM 0.29
        # hands GDN layers a plain page tensor while HMA-era vLLM passed
        # list[Tensor] strided views into one flat buffer.
        attention_layer_count = 0
        recurrent_layer_count = 0
        normalized_kv_caches: list[tuple[str, torch.Tensor]] = []
        mamba_layer_names: set[str] = set()
        for layer_pos, (layer_name, value) in enumerate(ordered_kv_caches):
            if group_of_layer[layer_pos] in mamba_group_ids:
                if isinstance(value, list):
                    # HMA-era forward compat: recover the flat backing
                    # raw_tensor from the strided views.
                    raw_tensor = _recover_raw_tensor(value)
                    logger.debug(
                        "register_kv_caches: layer=%s list[Tensor] len=%d raw_shape=%s",
                        layer_name,
                        len(value),
                        list(raw_tensor.shape),
                    )
                else:
                    # Plain page tensor ([num_blocks, 1, 1, C] int8):
                    # flatten every axis after the first.  view() cannot
                    # copy — vLLM allocates the page tensor contiguous
                    # with the block axis outermost, so folding the
                    # trailing dims is stride-preserving and the storage
                    # base (data_ptr) Rust registers stays vLLM's live
                    # buffer.  view() only raises for a non-viewable
                    # layout, where reshape() would ALLOCATE A COPY and
                    # the registered pointer would diverge from vLLM's
                    # cache — that never happens for the raw page tensor,
                    # but warn loudly if it ever does.
                    try:
                        raw_tensor = value.view(value.shape[0], -1)
                    except RuntimeError:
                        logger.warning(
                            "mamba layer %s: shape %s is not viewable as "
                            "[num_blocks, -1]; reshape() may have copied — "
                            "the registered storage base can differ from "
                            "vLLM's cache buffer",
                            layer_name,
                            list(value.shape),
                        )
                        raw_tensor = value.reshape(value.shape[0], -1)
                    # TEMPORARY shape probe (Phase-1-style): confirms the
                    # plain-tensor form from the node log; remove after the
                    # Probe A/B verdicts.
                    logger.warning(
                        "mamba layer %s: value type %s, shape %s, dtype %s "
                        "-> normalized %s",
                        layer_name,
                        type(value).__name__,
                        list(value.shape),
                        value.dtype,
                        list(raw_tensor.shape),
                    )
                mamba_layer_names.add(layer_name)
                recurrent_layer_count += 1
                normalized_kv_caches.append((layer_name, raw_tensor))
            else:
                attention_layer_count += 1
                normalized_kv_caches.append((layer_name, value))
        ordered_kv_caches = normalized_kv_caches
        self.mamba_layer_names = mamba_layer_names

        # TEMPORARY probe level (warning, not info): python logging's default
        # level is WARNING and nothing configures the kvbm loggers — the
        # Phase 1 probe proved INFO lines never reach the worker log.
        logger.warning(
            "register_kv_caches: %d KV layers (%d attention, %d recurrent/HMA)",
            len(ordered_kv_caches),
            attention_layer_count,
            recurrent_layer_count,
        )
        for group in group_plans:
            logger.warning(
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

        # num_device_blocks was derived above from an attention-group
        # tensor (plan-driven, before normalization).
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

        group_of_layer, group_specs, active_save_layers = self._hybrid_handoff(
            name_to_tensor, attention_layer_count, recurrent_layer_count
        )

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
            group_of_layer=group_of_layer,
            group_specs=group_specs,
            active_save_layers=active_save_layers,
        )

    def _hybrid_handoff(
        self,
        name_to_tensor: dict[str, torch.Tensor],
        attention_layer_count: int,
        recurrent_layer_count: int,
    ) -> tuple[
        Optional[list[int]],
        Optional[list[tuple[bool, Optional[int]]]],
        Optional[int],
    ]:
        """Build the hybrid (HMA) registration handoff for the Rust worker.

        Returns ``(group_of_layer, group_specs, active_save_layers)`` so
        the Rust worker derives per-group geometry and the save trigger
        fires at the last ATTENTION layer with the offload enqueue
        deferred to wait_for_save (mamba bytes are only final
        post-forward). Attention-only registrations return ``None`` for
        all three and keep the legacy Rust path byte-identical.
        """
        if recurrent_layer_count <= 0 or not self.group_plans:
            return None, None, None
        # Every plan shares the SAME group_of_layer list (GroupPlan's
        # docstring: one list parallel to the registration order), so
        # plan[0]'s view is the one authoritative mapping.
        group_of_layer: Optional[list[int]] = self.group_plans[0].group_of_layer
        group_specs: list[tuple[bool, Optional[int]]] = []
        for plan in self.group_plans:
            if plan.is_mamba and plan.layer_names:
                first_tensor = name_to_tensor[plan.layer_names[0]]
                # Physical per-block row bytes of the group's NORMALIZED
                # dense [num_blocks, C] state tensor: dtype-folded
                # (element_size * elements per block row) rather than
                # assumed int8 or read off shape[1].  vLLM's spec
                # page_size is padded metadata (Phase 1 probe proved
                # it); the tensor's own layout is ground truth.
                page_bytes: Optional[int] = int(
                    first_tensor.element_size()
                    * (first_tensor.nelement() // first_tensor.shape[0])
                )
            else:
                # Attention groups: Rust infers the geometry from the
                # group's first tensor (page tokens fall back to the
                # registration page_size).
                page_bytes = None
            group_specs.append((plan.is_mamba, page_bytes))
        return group_of_layer, group_specs, attention_layer_count

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

    def wait_for_save(self) -> None:
        """Signal that the step's full forward pass finished saving.

        Hybrid (HMA) models only: vLLM calls this once per step after the
        forward pass. Mamba/GDN layers never call ``save_kv_layer``, so
        their state bytes are only final now — the Rust worker deferred
        the offload enqueue to the fresh event recorded here, which
        covers every registered layer. Attention-only models keep the
        legacy path (trigger and enqueue at the last attention layer)
        and return without touching Rust.
        """
        if not self.mamba_layer_names:
            return
        event = torch.cuda.Event(enable_timing=False, interprocess=False)
        event.record(torch.cuda.current_stream())
        self._connector.wait_for_save(event.cuda_event)

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
