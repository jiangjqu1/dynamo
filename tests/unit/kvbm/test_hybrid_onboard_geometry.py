# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hybrid onboard destination geometry (fix 2, HMA multi-group).

The vLLM reference (``SimpleCPUOffloadScheduler.update_state_after_alloc``)
places a request's EXTERNAL blocks at the TAIL of each group's OWN computed
range — per-group arithmetic a flat positional index cannot express for
coarse-granularity groups (the fp4: FA block size 784, the second group
1960; 3920 external tokens are 5 FA blocks but only 2 of the second
group's). The leader must publish the geometry to the Rust slot BEFORE
``update_state_after_alloc`` (the Rust onboard trigger runs inside it)
whenever ``num_external_tokens > 0`` on a multi-group deployment.

The reference arithmetic (node-verified on vLLM 0.29):

- ``num_computed = num_cached_fa_blocks * fa_block_size`` where the FA
  group is the FIRST group whose spec is a ``FullAttentionSpec`` (hashed
  blocks count), falling back to group 0;
- ``total_computed = num_computed + num_external_tokens``;
- per group g (``g_bs = groups[g].kv_cache_spec.block_size``):
  ``n_ext_g = num_external // g_bs``,
  ``n_computed_g = cdiv(total_computed, g_bs)``,
  ``start_g = n_computed_g - n_ext_g`` (ext blocks at the TAIL).

Worked example (the fp4 numbers): groups block sizes [784, 1960], FA
internal 2 hashed blocks x 784 = 1568 computed, external 3920 tokens →
total 5488; group 0: n_ext=5, n_computed=cdiv(5488,784)=7, start=2;
group 1: n_ext=2, n_computed=cdiv(5488,1960)=3, start=1.

Derivation failures (missing block sizes, unaligned external tokens,
blocks structures that differ from the reference's shape) must SKIP the
publish with a warning — the Rust falls back to today's positional
destinations. Single-group deployments publish nothing.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.kvbm,
]

_VLLM_INTEGRATION_DIR = (
    Path(__file__).resolve().parents[3]
    / "lib/bindings/kvbm/python/kvbm/vllm_integration"
)


class FakeKvbmLeader:
    def __init__(self, world_size, drt=None):
        self.world_size = world_size
        self.drt = drt


class FakeRustKvConnectorLeader:
    """Recording stand-in for kvbm.vllm_integration.rust.KvConnectorLeader."""

    def __init__(
        self,
        engine_id,
        drt,
        block_size,
        leader,
        consolidator_vllm_endpoint=None,
        consolidator_output_endpoint=None,
        consolidator_mode=None,
    ):
        self.engine_id = engine_id
        self.drt = drt
        self.block_size = block_size
        self.leader = leader
        self.calls: list[tuple] = []

    def set_per_group_device_blocks(self, request_id, per_group_block_ids):
        self.calls.append(("set", request_id, per_group_block_ids))

    def append_per_group_device_blocks(self, request_id, per_group_block_ids):
        self.calls.append(("append", request_id, per_group_block_ids))

    def set_onboard_geometry(self, request_id, per_group_starts, per_group_ext_counts):
        self.calls.append(
            ("geometry", request_id, per_group_starts, per_group_ext_counts)
        )

    def update_state_after_alloc(self, request_id, block_ids, num_external_tokens):
        self.calls.append(("update", request_id, block_ids, num_external_tokens))

    def build_connector_metadata(self, scheduler_output):
        self.calls.append(("build_meta", scheduler_output))
        return b"{}"

    def request_finished(self, request_id, block_ids):
        self.calls.append(("finish", request_id, block_ids))
        return False, None

    def get_num_new_matched_tokens(self, request_id, num_tokens, num_computed):
        return 0, False

    def has_slot(self, request_id):
        return False

    def create_slot(self, request, tokens):
        pass

    def reset_cache(self):
        return True


# The spec type name is read via type(spec).__name__, so plain classes
# named like vLLM's are the fakes that matter:


class FullAttentionSpec:
    def __init__(self, block_size):
        self.block_size = block_size


class OtherSpec:
    """A coarse-granularity hybrid spec (e.g. a mamba/GDN group)."""

    def __init__(self, block_size):
        self.block_size = block_size


class SpeclessGroup:
    """A group whose spec exposes no block_size (derivation must skip)."""

    class kv_cache_spec:  # noqa: N801 - attribute-style stand-in
        pass


class FakeGroup:
    def __init__(self, spec):
        self.kv_cache_spec = spec


class FakeKVCacheConfig:
    def __init__(self, groups):
        self.kv_cache_groups = groups


class FakeKVCacheBlock:
    """Stands in for vLLM's KVCacheBlock (block_hash presence)."""

    def __init__(self, block_hash):
        self.block_hash = block_hash


class FakeBlocks:
    """Stands in for vLLM's KVCacheBlocks: per-group block lists + ids."""

    def __init__(self, per_group_blocks, per_group_block_ids):
        self.blocks = per_group_blocks
        self._per_group_block_ids = per_group_block_ids

    def get_block_ids(self):
        return self._per_group_block_ids


class FakeRequest:
    def __init__(self, request_id):
        self.request_id = request_id


def _fake_vllm_config():
    cfg = types.SimpleNamespace()
    cfg.parallel_config = types.SimpleNamespace(world_size=1)
    cfg.cache_config = types.SimpleNamespace(block_size=784)
    cfg.additional_config = {}
    return cfg


@pytest.fixture(name="leader_module")
def load_connector_leader_module(monkeypatch):
    """Load connector_leader.py from source with kvbm's heavy imports and
    the Rust connector stubbed. monkeypatch restores sys.modules."""
    kvbm_pkg = types.ModuleType("kvbm")
    kvbm_pkg.__path__ = []
    kvbm_pkg.KvbmLeader = FakeKvbmLeader
    monkeypatch.setitem(sys.modules, "kvbm", kvbm_pkg)

    utils_mod = types.ModuleType("kvbm.utils")
    utils_mod.is_dyn_runtime_enabled = lambda: False
    monkeypatch.setitem(sys.modules, "kvbm.utils", utils_mod)

    vllm_integration = types.ModuleType("kvbm.vllm_integration")
    vllm_integration.__path__ = []
    monkeypatch.setitem(sys.modules, "kvbm.vllm_integration", vllm_integration)

    consolidator_mod = types.ModuleType("kvbm.vllm_integration.consolidator_config")
    consolidator_mod.get_consolidator_mode = lambda: None
    monkeypatch.setitem(sys.modules, consolidator_mod.__name__, consolidator_mod)

    rust_mod = types.ModuleType("kvbm.vllm_integration.rust")
    rust_mod.KvConnectorLeader = FakeRustKvConnectorLeader

    class _StubKvbmRequest:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    rust_mod.KvbmRequest = _StubKvbmRequest
    rust_mod.SchedulerOutput = object
    monkeypatch.setitem(sys.modules, "kvbm.vllm_integration.rust", rust_mod)

    config_mod = types.ModuleType("vllm.config")
    config_mod.VllmConfig = type("VllmConfig", (), {})
    monkeypatch.setitem(sys.modules, "vllm.config", config_mod)

    base_mod = types.ModuleType("vllm.distributed.kv_transfer.kv_connector.v1.base")
    base_mod.KVConnectorMetadata = type("KVConnectorMetadata", (), {})
    monkeypatch.setitem(sys.modules, base_mod.__name__, base_mod)

    manager_mod = types.ModuleType("vllm.v1.core.kv_cache_manager")
    manager_mod.KVCacheBlocks = type("KVCacheBlocks", (), {})
    monkeypatch.setitem(sys.modules, manager_mod.__name__, manager_mod)

    sched_output_mod = types.ModuleType("vllm.v1.core.sched.output")
    sched_output_mod.SchedulerOutput = type("SchedulerOutput", (), {})
    monkeypatch.setitem(sys.modules, sched_output_mod.__name__, sched_output_mod)

    request_mod = types.ModuleType("vllm.v1.request")
    request_mod.Request = type("Request", (), {})
    monkeypatch.setitem(sys.modules, request_mod.__name__, request_mod)

    leader_spec = importlib.util.spec_from_file_location(
        "kvbm.vllm_integration.connector_leader",
        _VLLM_INTEGRATION_DIR / "connector_leader.py",
    )
    leader_mod = importlib.util.module_from_spec(leader_spec)
    monkeypatch.setitem(sys.modules, leader_spec.name, leader_mod)
    leader_spec.loader.exec_module(leader_mod)
    return leader_mod


def _make_leader(leader_module, groups):
    leader = leader_module.KvConnectorLeader(
        vllm_config=_fake_vllm_config(),
        engine_id="engine-0",
        kv_cache_config=FakeKVCacheConfig(groups),
    )
    return leader, leader._connector


def _geometry_calls(rust):
    return [c for c in rust.calls if c[0] == "geometry"]


def test_geometry_matches_the_reference_arithmetic(leader_module):
    """The fp4 worked example: block sizes [784, 1960], FA group 0 with 2
    hashed internal blocks (1568 computed), 3920 external tokens →
    starts [2, 1], ext counts [5, 2] (the vLLM reference's numbers)."""
    groups = [
        FakeGroup(FullAttentionSpec(784)),
        FakeGroup(OtherSpec(1960)),
    ]
    leader, rust = _make_leader(leader_module, groups)

    fa_blocks = [
        FakeKVCacheBlock(1001),
        FakeKVCacheBlock(1002),
    ]
    blocks = FakeBlocks(
        per_group_blocks=[fa_blocks, [FakeKVCacheBlock(None)]],
        per_group_block_ids=([10, 11, 12, 13, 14], [70, 71]),
    )

    leader.update_state_after_alloc(FakeRequest("req-1"), blocks, 3920)

    kinds = [call[0] for call in rust.calls]
    assert kinds.index("geometry") < kinds.index("update"), (
        "the geometry must reach the Rust slot BEFORE update_state_after_alloc: "
        "the Rust onboard trigger runs inside it"
    )
    assert rust.calls[kinds.index("geometry")] == (
        "geometry",
        "req-1",
        [2, 1],
        [5, 2],
    )


def test_geometry_uses_the_first_full_attention_group(leader_module):
    """The FA group may not be group 0: hashed blocks are counted from ITS
    row and its block size derives the computed tokens. Same fp4 shape
    with the groups swapped → starts [1, 2], ext counts [2, 5]."""
    groups = [
        FakeGroup(OtherSpec(1960)),
        FakeGroup(FullAttentionSpec(784)),
    ]
    leader, rust = _make_leader(leader_module, groups)

    fa_blocks = [FakeKVCacheBlock(1001), FakeKVCacheBlock(1002)]
    blocks = FakeBlocks(
        per_group_blocks=[[FakeKVCacheBlock(None)], fa_blocks],
        per_group_block_ids=([70, 71], [10, 11, 12, 13, 14]),
    )

    leader.update_state_after_alloc(FakeRequest("req-2"), blocks, 3920)

    assert _geometry_calls(rust) == [("geometry", "req-2", [1, 2], [2, 5])]


def test_geometry_skipped_without_full_attention_spec_falls_back_to_group0(
    leader_module,
):
    """No FullAttentionSpec → group 0 is the FA stand-in (the fallback)."""
    groups = [FakeGroup(OtherSpec(784)), FakeGroup(OtherSpec(1960))]
    leader, rust = _make_leader(leader_module, groups)

    blocks = FakeBlocks(
        per_group_blocks=[
            [FakeKVCacheBlock(1), FakeKVCacheBlock(2)],
            [FakeKVCacheBlock(None)],
        ],
        per_group_block_ids=([10, 11], [70]),
    )

    leader.update_state_after_alloc(FakeRequest("req-3"), blocks, 3920)

    # Group 0 at 784 behaves like the FA group: starts [2, 1].
    assert _geometry_calls(rust) == [("geometry", "req-3", [2, 1], [5, 2])]


def test_geometry_skipped_when_blocks_shape_differs(leader_module):
    """blocks without the reference's .blocks structure → warn and skip
    (the Rust falls back to positional destinations); the Rust update
    call still runs."""
    groups = [FakeGroup(FullAttentionSpec(784)), FakeGroup(OtherSpec(1960))]
    leader, rust = _make_leader(leader_module, groups)

    class BlocksWithoutRows:
        def get_block_ids(self):
            return ([10, 11], [70])
            # no .blocks attribute

    leader.update_state_after_alloc(FakeRequest("req-4"), BlocksWithoutRows(), 3920)

    assert _geometry_calls(rust) == []
    kinds = [call[0] for call in rust.calls]
    assert "update" in kinds, "the Rust update call must still run"


def test_geometry_skipped_on_unaligned_external_tokens(leader_module):
    """External tokens not whole blocks of EVERY group → the reference's
    arithmetic does not apply; skip publishing (positional fallback)."""
    groups = [FakeGroup(FullAttentionSpec(784)), FakeGroup(OtherSpec(1960))]
    leader, rust = _make_leader(leader_module, groups)

    blocks = FakeBlocks(
        per_group_blocks=[[FakeKVCacheBlock(1)], [FakeKVCacheBlock(None)]],
        per_group_block_ids=([10], [70]),
    )

    # 1000 tokens: whole blocks of 784? no (1000 % 1960 != 0 either).
    leader.update_state_after_alloc(FakeRequest("req-5"), blocks, 1000)

    assert _geometry_calls(rust) == []


def test_geometry_skipped_when_group_block_size_absent(leader_module):
    """A group spec without block_size → geometry cannot be derived; skip
    publishing (positional fallback)."""
    groups = [FakeGroup(FullAttentionSpec(784)), SpeclessGroup()]
    leader, rust = _make_leader(leader_module, groups)

    blocks = FakeBlocks(
        per_group_blocks=[[FakeKVCacheBlock(1)], [FakeKVCacheBlock(None)]],
        per_group_block_ids=([10], [70]),
    )

    leader.update_state_after_alloc(FakeRequest("req-6"), blocks, 3920)

    assert _geometry_calls(rust) == []


def test_geometry_not_published_for_single_group(leader_module):
    """Single-group deployments publish nothing (the legacy flat path)."""
    groups = [FakeGroup(FullAttentionSpec(784))]
    leader, rust = _make_leader(leader_module, groups)

    blocks = FakeBlocks(
        per_group_blocks=[[FakeKVCacheBlock(1)]],
        per_group_block_ids=([10, 11],),
    )

    leader.update_state_after_alloc(FakeRequest("req-7"), blocks, 3920)

    assert _geometry_calls(rust) == []
    kinds = [call[0] for call in rust.calls]
    assert kinds == ["update"]


def test_geometry_not_published_without_external_tokens(leader_module):
    """num_external_tokens == 0 → no onboard will trigger; nothing to
    publish (the plain second update call shape)."""
    groups = [FakeGroup(FullAttentionSpec(784)), FakeGroup(OtherSpec(1960))]
    leader, rust = _make_leader(leader_module, groups)

    blocks = FakeBlocks(
        per_group_blocks=[[FakeKVCacheBlock(1)], [FakeKVCacheBlock(None)]],
        per_group_block_ids=([10, 11], [70, 71]),
    )

    leader.update_state_after_alloc(FakeRequest("req-8"), blocks, 0)

    assert _geometry_calls(rust) == []
