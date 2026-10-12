# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Derive per-group registration plans from vLLM's HMA kv_cache_config."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig


@dataclass
class GroupPlan:
    """One HMA group's registration plan (order-preserving).

    ``group_of_layer`` is ONE list shared by every plan of a derivation,
    parallel to the caller's ``ordered_layer_names``; each entry indexes the
    plans by ``group_id``.  It is shared intentionally for the Rust handoff
    and should be treated as read-only.
    """

    group_id: int
    layer_names: list[str]
    is_mamba: bool
    page_bytes: Optional[int]  # spec page size if derivable, else None
    group_of_layer: list[int] = field(default_factory=list)


def _spec_is_mamba(spec) -> bool:
    # Keyed on the spec TYPE NAME: that name is the cross-version contract,
    # and an isinstance check would require importing vllm at runtime.
    return type(spec).__name__ == "MambaSpec"


def derive_layer_groups(
    kv_cache_config: Optional["KVCacheConfig"], ordered_layer_names: list[str]
) -> list[GroupPlan]:
    """Partition ordered_layer_names into groups.

    Falls back to a single group when kv_cache_config is None (the
    attention-only path must keep its previous all-layers-uniform behavior).
    """
    if not kv_cache_config or not getattr(kv_cache_config, "kv_cache_groups", None):
        return [
            GroupPlan(
                group_id=0,
                layer_names=list(ordered_layer_names),
                is_mamba=False,
                page_bytes=None,
                group_of_layer=[0] * len(ordered_layer_names),
            )
        ]

    name_to_group: dict[str, int] = {}
    plans: list[GroupPlan] = []
    for gid, group in enumerate(kv_cache_config.kv_cache_groups):
        plans.append(
            GroupPlan(
                group_id=gid,
                layer_names=[],
                is_mamba=_spec_is_mamba(group.kv_cache_spec),
                page_bytes=getattr(group.kv_cache_spec, "page_size", None),
            )
        )
        for name in group.layer_names:
            name_to_group[name] = gid

    group_of_layer: list[int] = []
    for name in ordered_layer_names:
        gid = name_to_group.get(name)
        if gid is None:
            gid = 0  # unregistered layer: keep it in group 0 (warn at caller)
        group_of_layer.append(gid)
        plans[gid].layer_names.append(name)
    for p in plans:
        p.group_of_layer = group_of_layer
    return plans
