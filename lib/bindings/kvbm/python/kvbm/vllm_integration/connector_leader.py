# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Implementation of vLLM KV cache manager protocol.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Optional

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


# from kvbm.vllm_integration.kv_cache_utils import KvbmCacheBlocks
# from kvbm.vllm_integration.rust import BlockManager, KvbmRequest
# from kvbm.vllm_integration.rust import KvConnectorLeader as RustKvConnectorLeader
# from kvbm.vllm_integration.rust import (
#     KvConnectorMetadata as RustKvConnectorMetadata,
# )
# from kvbm.vllm_integration.rust import SchedulerOutput as RustSchedulerOutput

from kvbm import KvbmLeader
from kvbm.utils import is_dyn_runtime_enabled
from kvbm.vllm_integration.consolidator_config import get_consolidator_mode
from kvbm.vllm_integration.rust import KvbmRequest
from kvbm.vllm_integration.rust import KvConnectorLeader as RustKvConnectorLeader
from kvbm.vllm_integration.rust import SchedulerOutput as RustSchedulerOutput

DistributedRuntime = None
if is_dyn_runtime_enabled():
    from dynamo.runtime import DistributedRuntime

logger = logging.getLogger(__name__)


def _rposition(tracked: list[int], block_id: int) -> Optional[int]:
    """Last index of ``block_id`` in ``tracked``, or None (Rust rposition)."""
    for idx in range(len(tracked) - 1, -1, -1):
        if tracked[idx] == block_id:
            return idx
    return None


def _reconcile_feed(
    current: list[int], feed: list[int], request_id: str, group: int
) -> list[int]:
    """The python mirror of the Rust ``reconcile_device_block_feed``
    decision ladder (keep the two in lockstep — the store must equal what
    the Rust slot holds after the same feed). vLLM's feed is ALWAYS the
    authoritative current list; divergence degrades to replace-with-warn,
    never an error (the engine must not die on HMA hybrid feed shapes)."""
    if not feed or current == feed:
        return current
    if len(current) >= len(feed) and current[len(current) - len(feed) :] == feed:
        return current  # natural suffix — already applied
    if len(feed) > len(current) and feed[: len(current)] == current:
        tail = feed[len(current) :]
        known = set(current)
        for block_id in tail:
            if block_id in known:
                logger.warning(
                    "request %s group %d: feed tail re-tracks id %d "
                    "(vLLM duplicate reporting is legal for HMA mamba groups)",
                    request_id,
                    group,
                    block_id,
                )
        current.extend(tail)
        return current
    pos = _rposition(current, feed[0])
    if pos is None:
        known = set(current)
        for block_id in feed:
            if block_id in known:
                logger.warning(
                    "request %s group %d: feed appends already-tracked id %d "
                    "(vLLM duplicate reporting is legal for HMA mamba groups)",
                    request_id,
                    group,
                    block_id,
                )
        current.extend(feed)
        return current
    suffix_len = len(current) - pos
    if suffix_len <= len(feed) and current[pos:] == feed[:suffix_len]:
        current.extend(feed[suffix_len:])
        return current
    if pos == 0:
        logger.warning(
            "request %s group %d: device blocks reassigned (e.g. "
            "copy-on-write of a prefix-cache-shared block); replacing the "
            "stored list %s with %s",
            request_id,
            group,
            current,
            feed,
        )
        return feed
    # The ladder could not align the feed (HMA hybrid feed shapes) — vLLM's
    # feed is the authoritative current list: replace with a warn.
    logger.warning(
        "request %s group %d: device-block feed could not be aligned (vLLM "
        "HMA hybrid feed shape); replacing the stored list %s with the "
        "authoritative feed %s",
        request_id,
        group,
        current,
        feed,
    )
    return feed


class DynamoConnectorMetadata(KVConnectorMetadata):
    def __init__(self, metadata: bytes):
        assert isinstance(metadata, bytes)
        self.metadata = metadata


class KvConnectorLeader:
    """
    Implements the vLLM KV cache manager protocol.

    This class is a wrapper around the Rust KvbmCacheManager class.
    It is used to convert the Rust KvbmCacheManager into a Python class
    that can be used in the vLLM KV cache manager protocol.
    """

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
        world_size = vllm_config.parallel_config.world_size

        # HMA group count from the factory-supplied kv_cache_config. vLLM's
        # hybrid allocator assigns block ids PER GROUP independently (the
        # HMA-PROBE logged `[[1], [2], [3], [4]]` for one request), so with
        # more than one group the leader must publish per-group id lists to
        # the Rust slot mapping — group 0's flat list is only valid for
        # group 0's pool. Single-group (or unknown) deployments keep the
        # flat group-0-only contract.
        kv_cache_groups = (
            getattr(kv_cache_config, "kv_cache_groups", None)
            if kv_cache_config is not None
            else None
        )
        self._num_groups = len(kv_cache_groups) if kv_cache_groups else 1
        # One-time gate log for the hybrid onboard safety gate (see
        # get_num_new_matched_tokens).
        self._hybrid_onboard_gate_logged = False

        # Cumulative per-group device block lists per request (group-major).
        # This is the leader's authoritative view of what the Rust slot holds:
        # FULL lists at admission/resume, merged with each step's deltas for
        # running requests, published back FULL at every build_connector_meta.
        # Chunked prefill + prefix-cache COW can make vLLM's cached-request
        # ``new_block_ids`` a REPLACEMENT-shaped feed (live cm008 evidence: a
        # non-resumed step delivered ``[0, 6]`` where ``6`` replaced a mid-list
        # id), so a delta alone cannot always reconstruct the full list — the
        # leader tracks the cumulative lists itself and never invents block
        # ids it has not seen vLLM report.
        self._req_full_blocks: dict[str, tuple[list[int], ...]] = {}

        leader = KvbmLeader(world_size, drt=self.drt)

        print(f"KvConnectorLeader initialized with engine_id: {engine_id}")
        # Get kv event consolidator endpoints from vllm_config (pre-computed in main.py)
        consolidator_vllm_endpoint = None
        consolidator_output_endpoint = None
        consolidator_mode = None
        self._consolidator_output_port = None

        _consolidator_eps = vllm_config.additional_config.get("consolidator_endpoints")
        if _consolidator_eps:
            consolidator_mode = get_consolidator_mode()
            # Unpack all three endpoints
            # [0]: vllm_endpoint (for consolidator to subscribe to vLLM)
            # [1]: output_bind_endpoint (for consolidator to bind/publish)
            # [2]: output_connect_endpoint (for clients to connect)
            (
                consolidator_vllm_endpoint,
                consolidator_output_endpoint,
                _consolidator_output_connect_endpoint,  # Not needed here
            ) = _consolidator_eps
            self._consolidator_output_port = int(
                consolidator_output_endpoint.split(":")[-1]
            )

            # Pass endpoints to Rust
            self._connector = RustKvConnectorLeader(
                engine_id,
                self.drt,
                vllm_config.cache_config.block_size,
                leader,
                consolidator_vllm_endpoint=consolidator_vllm_endpoint,
                consolidator_output_endpoint=consolidator_output_endpoint,
                consolidator_mode=consolidator_mode,
            )
        else:
            # No kv event consolidator - pass None to Rust
            self._connector = RustKvConnectorLeader(
                engine_id,
                self.drt,
                vllm_config.cache_config.block_size,
                leader,
                consolidator_vllm_endpoint=None,
                consolidator_output_endpoint=None,
                consolidator_mode=None,
            )

    # KV Connector

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """
        Get number of new tokens that can be loaded from the
        external KV cache beyond the num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            A tuple with the following elements:
                - The number of tokens that can be loaded from the
                  external KV cache beyond what is already computed.
                - `True` if external KV cache tokens will be loaded
                  asynchronously (between scheduler steps).
        """
        self._create_slot(request)
        if (
            self._num_groups > 1
            and not os.environ.get("DYN_KVBM_HYBRID_EXPERIMENTAL_ONBOARD", "").strip()
        ):
            # SAFETY GATE (2026-10-15): the hybrid external ONBOARD is
            # disabled by default. The determinism test on the fp4 proved
            # that onboarding restores content that corrupts generation
            # (temp-0 replay garbles after h2d onboards; coherent without):
            # vLLM 0.29 HMA per-group block lists are NOT position-aligned
            # for cached prefixes (live evidence: group_lengths [4,4,4,10]
            # — the mamba state collapses to checkpoint entries with a
            # different indexing than attention KV blocks), so the tier-2
            # copies restored under positionally-matched hashes carry the
            # wrong content for those positions. Offload (D2H) continues
            # (harmless copies at rest); onboarding requires the
            # per-group semantics from the DEP. Opt back in explicitly
            # with DYN_KVBM_HYBRID_EXPERIMENTAL_ONBOARD=1 for experiments.
            if not self._hybrid_onboard_gate_logged:
                self._hybrid_onboard_gate_logged = True
                logger.warning(
                    "hybrid multi-group model detected: external KV onboarding "
                    "is DISABLED (the HMA per-group block semantics are not "
                    "position-aligned for cached prefixes; onboarding restores "
                    "wrong content — see the phase-2 plan §12). Set "
                    "DYN_KVBM_HYBRID_EXPERIMENTAL_ONBOARD=1 to override."
                )
            return num_computed_tokens, False
        return self._connector.get_num_new_matched_tokens(
            request.request_id,
            request.num_tokens,
            num_computed_tokens,
        )

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        # TEMPORARY HMA-PROBE (remove after Task 9's probe verdicts): verify
        # the shared-index assumption — vLLM HMA hands per-group block-id
        # lists and the multi-group pool design requires them to be identical
        # (group 0 is authoritative here and in build_connector_meta). The
        # probe lives here rather than in get_num_new_matched_tokens because
        # the Request object does not carry block_ids in this vLLM line and
        # the blocks are not allocated yet when that hook runs;
        # update_state_after_alloc receives the allocated KVCacheBlocks,
        # whose get_block_ids() is the per-group tuple.
        group_block_ids = blocks.get_block_ids()
        if len(group_block_ids) > 1 and logger.isEnabledFor(logging.WARNING):
            logger.warning(
                "HMA-PROBE: request %s block_ids per group: %s",
                request.request_id,
                [list(g) for g in group_block_ids[:4]],  # truncate
            )
        block_ids = group_block_ids[0]
        # Multi-group: publish the FULL per-group mapping BEFORE the Rust
        # call — update_state_after_alloc may trigger onboarding inside it
        # (num_external_tokens > 0), and the onboard destinations are
        # resolved per group from this mapping. vLLM hands full per-group
        # lists here, so the mapping is replaced wholesale.
        self.set_per_group_device_blocks(request.request_id, group_block_ids)
        self._req_full_blocks[request.request_id] = tuple(
            list(group) for group in group_block_ids
        )
        self._connector.update_state_after_alloc(
            request.request_id, block_ids, num_external_tokens
        )

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> bytes:
        """
        Build the connector metadata for this step.

        This function should NOT modify fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        output = RustSchedulerOutput()

        for req in scheduler_output.scheduled_new_reqs:
            # NewRequestData.block_ids is the per-group tuple of FULL lists;
            # re-publishing is idempotent (update_state_after_alloc already
            # replaced the mapping for this request this iteration). Record
            # it as the request's cumulative state — the running-step deltas
            # merge into it below.
            self.set_per_group_device_blocks(req.req_id, req.block_ids)
            self._req_full_blocks[req.req_id] = tuple(
                list(group) for group in req.block_ids
            )
            output.add_new_request(
                req.req_id,
                req.prompt_token_ids,
                req.block_ids[0],
                req.num_computed_tokens,
            )

        # In vLLM 0.14.0+, resumed_from_preemption was changed to resumed_req_ids (a set)
        resumed_req_ids = scheduler_output.scheduled_cached_reqs.resumed_req_ids

        # If A == B, and A == C, then B == C
        cached_reqs = scheduler_output.scheduled_cached_reqs
        assert len(cached_reqs.req_ids) == len(
            cached_reqs.new_block_ids
        ), "Number of cached req_ids doesn't match the number of cached new_block_ids"
        assert len(cached_reqs.req_ids) == len(
            cached_reqs.num_computed_tokens
        ), "Number of cached req_ids doesn't match the number of cached num_computed_tokens"

        # In https://github.com/vllm-project/vllm/pull/26388/changes#diff-9eeca590fd99f15621897e559dba39b3ec4e7c2c65ec3c3229711689e008b5f4L732-L736,
        # new_token_ids was changed to return an empty list unless pipeline
        # parallelism is turned on. If needed, which for KVBM it is not needed,
        # KVBM can consult the cached_reqs.all_token_ids to get the token ids
        # for each of the requests. KVBM doesn't consult this field since
        # it holds the token sequence in the _connector.
        for i, req_id in enumerate(cached_reqs.req_ids):
            # new_token_ids may be empty when pipeline parallelism is disabled

            new_token_ids = (
                cached_reqs.new_token_ids[i]
                if i < len(cached_reqs.new_token_ids)
                else []
            )
            new_block_ids = cached_reqs.new_block_ids[i]
            num_computed_tokens = cached_reqs.num_computed_tokens[i]

            resumed_from_preemption = req_id in resumed_req_ids
            if self._num_groups > 1:
                # Publish the CUMULATIVE per-group mapping and feed the
                # cumulative group-0 list. vLLM's CachedRequestData contract:
                # running requests carry this iteration's per-group DELTAS,
                # resumed-from-preemption requests carry FULL per-group lists
                # — but a delta alone is not always reconstructive (prefix
                # -cache COW delivers replacement-shaped feeds; live cm008
                # evidence: a non-resumed step fed [0, 6] where 6 replaced a
                # mid-list id), so the leader maintains the cumulative
                # per-group lists itself and republishes them FULL every
                # step. The Rust reconcile applies a full feed idempotently
                # (no-op / suffix-extend / head-match REPLACE), so this is
                # safe and self-healing, and the offload/onboard consumers
                # index positions that hold vLLM's CURRENT ids.
                if new_block_ids is not None:
                    if resumed_from_preemption:
                        self._store_full_blocks(req_id, new_block_ids)
                    else:
                        self._merge_delta_into_store(req_id, new_block_ids)
                stored = self._req_full_blocks.get(req_id)
                if stored is not None:
                    self.set_per_group_device_blocks(req_id, stored)
                    flat_block_ids = list(stored[0])
                elif new_block_ids is not None:
                    # No cumulative state to republish (slot predates this
                    # leader's store or the delta could not be merged) —
                    # fall back to today's per-step delta behavior; never
                    # invent block ids.
                    self.append_per_group_device_blocks(req_id, new_block_ids)
                    flat_block_ids = new_block_ids[0]
                else:
                    flat_block_ids = []
            elif new_block_ids is not None:
                # Single-group keeps the flat group-0 contract unchanged.
                flat_block_ids = new_block_ids[0]
            else:
                flat_block_ids = []
            output.add_cached_request(
                request_id=req_id,
                resumed_from_preemption=resumed_from_preemption,
                new_token_ids=new_token_ids,
                new_block_ids=flat_block_ids,
                num_computed_tokens=num_computed_tokens,
            )

        output.add_num_scheduled_tokens(scheduler_output.num_scheduled_tokens)

        assert (
            scheduler_output.total_num_scheduled_tokens
            == output.get_num_scheduled_tokens()
        ), "Total number of scheduled tokens does not match"

        return self._connector.build_connector_metadata(output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """
        Called when a request has finished, before its blocks are freed.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        """
        # note our worker can communication with us oob and we can use that to know
        # ahead of time if the request is finished.
        status = self._connector.request_finished(request.request_id, block_ids)
        self._req_full_blocks.pop(request.request_id, None)
        return status, None

    def reset_cache(self) -> bool:
        """Reset KVBM-managed prefix cache state."""
        return self._connector.reset_cache()

    # Utility functions

    def set_per_group_device_blocks(
        self, request_id: str, per_group_block_ids: tuple[list[int], ...]
    ) -> None:
        """Publish the request's FULL per-group device block ids to the Rust
        slot mapping (REPLACE semantics).

        vLLM's hybrid allocator assigns block ids per group independently
        (the HMA-PROBE proved group 0's ids are NOT the other groups' ids),
        and the slot's offload/onboard consumers index the mapping by
        [group][logical block position]. Single-group deployments skip the
        call entirely — the flat group-0 contract applies unchanged.
        """
        if self._num_groups <= 1:
            return
        self._connector.set_per_group_device_blocks(
            request_id, [list(group) for group in per_group_block_ids]
        )

    def append_per_group_device_blocks(
        self, request_id: str, per_group_block_ids: tuple[list[int], ...]
    ) -> None:
        """Publish the request's per-group device block-id DELTAS (this
        iteration's newly allocated ids per group — vLLM's cached-request
        ``new_block_ids``) to the Rust slot mapping (APPEND-at-tail
        semantics). Skipped for single-group deployments.
        """
        if self._num_groups <= 1:
            return
        self._connector.append_per_group_device_blocks(
            request_id, [list(group) for group in per_group_block_ids]
        )

    def _store_full_blocks(
        self, request_id: str, per_group_block_ids: tuple[list[int], ...]
    ) -> None:
        """Record vLLM's FULL per-group lists as the request's cumulative
        state (admission-time and resumed-from-preemption reports — both
        carry the request's complete current lists)."""
        self._req_full_blocks[request_id] = tuple(
            list(group) for group in per_group_block_ids
        )

    def _merge_delta_into_store(
        self, request_id: str, per_group_delta: tuple[list[int], ...]
    ) -> None:
        """Merge a running step's per-group DELTAS into the cumulative store,
        mirroring the Rust ``reconcile_device_block_feed`` ladder per group so
        the store always matches what the Rust slot holds after the feed.

        Per group (the same decision ladder as the Rust, in the same
        order — vLLM's feed is ALWAYS the authoritative current list):

        1. feed == stored → no-op;
        2. feed matches the stored list's natural suffix → no-op;
        3. the stored list is a prefix of the feed → extend with the tail;
        4. feed head not tracked → append (already-tracked ids only warn —
           vLLM's HMA mamba groups legitimately report duplicates);
        5. suffix/prefix overlap → extend with the remainder;
        6. head-match with a divergent tail → REPLACE (the COW shape);
        7. mid-list mismatch → REPLACE with a warn (the HMA hybrid feed
           shapes — replacement-shaped running feeds, cached-mamba
           collapse to a shared id, groups not allocating in step).
        """
        stored = self._req_full_blocks.get(request_id)
        if stored is None:
            logger.warning(
                "request %s has no cumulative block state to merge the delta "
                "into; deferring to the per-step delta feed",
                request_id,
            )
            return
        merged: list[list[int]] = []
        for group, (current, delta) in enumerate(zip(stored, per_group_delta)):
            merged.append(
                _reconcile_feed(list(current), list(delta), request_id, group)
            )
        self._req_full_blocks[request_id] = tuple(merged)

    def _create_slot(self, request: Request) -> None:
        """Create a slot for the request"""

        if self._connector.has_slot(request.request_id):
            return None

        if bool(getattr(request, "mm_features", None)) or bool(
            getattr(request, "mm_positions", None)
        ):
            raise ValueError("Unsupported request - requires mm extra keys")

        all_token_ids = request.all_token_ids

        # extract the critial aspects of the request that effect how the tokens are hashed
        request = KvbmRequest(
            request_id=request.request_id,
            lora_name=request.lora_request.lora_name()
            if request.lora_request
            else None,
            salt_hash=request.cache_salt,
        )

        self._connector.create_slot(request, all_token_ids)
