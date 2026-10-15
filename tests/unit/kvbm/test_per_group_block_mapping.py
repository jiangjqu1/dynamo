# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-group block-id mapping on the Python leader (HMA multi-group).

The HMA-PROBE on the fp4 proved vLLM's hybrid allocator assigns block ids
PER GROUP independently (``[[1], [2], [3], [4]]`` for one request), so the
leader must publish per-group id lists to the Rust slot mapping at every
hook that precedes a per-group consumer:

- ``update_state_after_alloc`` — FULL lists, REPLACE, before the Rust call
  (it may trigger onboarding inside, which resolves per-group dst ids).
- ``build_connector_meta`` — new requests re-publish FULL lists; running
  cached requests are merged into the leader's CUMULATIVE per-group store
  and the FULL stored lists are republished every step (a delta alone is
  not always reconstructive — prefix-cache COW delivers replacement-shaped
  feeds, live cm008 evidence ``[0, 6]`` replacing a mid-list id);
  resumed-from-preemption requests REPLACE the store with FULL lists
  (vLLM's CachedRequestData contract).
- single-group deployments must make NONE of these calls and keep the raw
  per-step delta feeds unchanged.

The REAL ``KvConnectorLeader`` is loaded from source with the Rust
connector stubbed (a recording fake), so these tests assert the PUBLISH
DECISIONS and their ordering, not Rust behavior.
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


class FakeRustSchedulerOutput:
    """Stand-in for kvbm.vllm_integration.rust.SchedulerOutput."""

    def __init__(self):
        self.new_requests = []
        self.cached_requests = []
        self.total = None

    def add_new_request(self, *args):
        self.new_requests.append(args)

    def add_cached_request(self, **kwargs):
        self.cached_requests.append(kwargs)

    def add_num_scheduled_tokens(self, num_scheduled_tokens):
        # vLLM's leader passes the per-request dict through to the Rust side.
        self.total = num_scheduled_tokens

    def get_num_scheduled_tokens(self):
        return sum(self.total.values()) if self.total else 0


class FakeKvCacheGroups:
    """Stands in for vLLM's KVCacheConfig (the .kv_cache_groups holder)."""

    def __init__(self, num_groups):
        self.kv_cache_groups = [object() for _ in range(num_groups)]


class FakeBlocks:
    """Stands in for vLLM's KVCacheBlocks (per-group get_block_ids)."""

    def __init__(self, per_group_block_ids):
        self._per_group = per_group_block_ids

    def get_block_ids(self):
        return self._per_group


class FakeRequest:
    def __init__(self, request_id):
        self.request_id = request_id


class FakeNewReq:
    def __init__(self, req_id, block_ids):
        self.req_id = req_id
        self.prompt_token_ids = [1, 2, 3]
        self.block_ids = block_ids
        self.num_computed_tokens = 0


class FakeCachedReqs:
    def __init__(self, req_ids, new_block_ids, resumed_req_ids=()):
        self.req_ids = req_ids
        self.resumed_req_ids = set(resumed_req_ids)
        self.new_token_ids = [[] for _ in req_ids]
        self.new_block_ids = new_block_ids
        self.num_computed_tokens = [0 for _ in req_ids]


def _fake_vllm_config():
    cfg = types.SimpleNamespace()
    cfg.parallel_config = types.SimpleNamespace(world_size=1)
    cfg.cache_config = types.SimpleNamespace(block_size=16)
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
    rust_mod.SchedulerOutput = FakeRustSchedulerOutput
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


def _make_leader(leader_module, num_groups):
    leader = leader_module.KvConnectorLeader(
        vllm_config=_fake_vllm_config(),
        engine_id="engine-0",
        kv_cache_config=FakeKvCacheGroups(num_groups),
    )
    return leader, leader._connector


def test_num_groups_derived_from_kv_cache_config(leader_module):
    _, rust = _make_leader(leader_module, 4)
    assert rust.leader.world_size == 1  # sanity: the ctor ran
    leader_module  # noqa: B018 — fixture loaded


def test_update_state_after_alloc_publishes_full_mapping_before_rust_call(
    leader_module,
):
    leader, rust = _make_leader(leader_module, 4)
    blocks = FakeBlocks(([10, 11], [20, 21], [30, 31], [40, 41]))

    leader.update_state_after_alloc(FakeRequest("req-1"), blocks, 0)

    kinds = [call[0] for call in rust.calls]
    assert kinds.index("set") < kinds.index("update"), (
        "the mapping must reach the Rust slot BEFORE update_state_after_alloc: "
        "it may trigger onboarding inside, which resolves per-group dst ids"
    )
    assert rust.calls[kinds.index("set")] == (
        "set",
        "req-1",
        [[10, 11], [20, 21], [30, 31], [40, 41]],
    )
    # The Rust call itself keeps the flat group-0 contract.
    assert rust.calls[kinds.index("update")] == ("update", "req-1", [10, 11], 0)


def test_update_state_after_alloc_single_group_skips_mapping(leader_module):
    leader, rust = _make_leader(leader_module, 1)
    blocks = FakeBlocks(([10, 11],))

    leader.update_state_after_alloc(FakeRequest("req-1"), blocks, 0)

    assert [call[0] for call in rust.calls] == ["update"]


def _scheduler_output(new_reqs=(), cached=None, total=0):
    output = types.SimpleNamespace()
    output.scheduled_new_reqs = list(new_reqs)
    output.scheduled_cached_reqs = cached or FakeCachedReqs([], [])
    output.num_scheduled_tokens = {}
    output.total_num_scheduled_tokens = total
    return output


def test_build_connector_meta_publishes_new_running_and_resumed(leader_module):
    leader, rust = _make_leader(leader_module, 4)

    new_req = FakeNewReq("req-new", ([10, 11], [20, 21], [30, 31], [40, 41]))
    cached = FakeCachedReqs(
        req_ids=["req-running", "req-resumed", "req-idle"],
        new_block_ids=[
            ([12], [22], [32], [42]),  # running: per-group DELTAS
            ([13, 14], [23, 24], [33, 34], [43, 44]),  # resumed: FULL lists
            None,  # scheduled with no new blocks
        ],
        resumed_req_ids=("req-resumed",),
    )

    leader.build_connector_meta(_scheduler_output(new_reqs=[new_req], cached=cached))

    per_req = {}
    for call in rust.calls:
        if call[0] in ("set", "append"):
            per_req.setdefault(call[1], []).append(call)

    # New request: FULL lists (set).
    assert [c[0] for c in per_req["req-new"]] == ["set"]
    assert per_req["req-new"][0][2] == [[10, 11], [20, 21], [30, 31], [40, 41]]
    # Running request: per-group DELTAS (append).
    assert [c[0] for c in per_req["req-running"]] == ["append"]
    assert per_req["req-running"][0][2] == [[12], [22], [32], [42]]
    # Resumed request: FULL lists (set).
    assert [c[0] for c in per_req["req-resumed"]] == ["set"]
    assert per_req["req-resumed"][0][2] == [
        [13, 14],
        [23, 24],
        [33, 34],
        [43, 44],
    ]
    # No new blocks: no mapping update at all.
    assert "req-idle" not in per_req


def test_build_connector_meta_single_group_skips_mapping(leader_module):
    leader, rust = _make_leader(leader_module, 1)

    new_req = FakeNewReq("req-new", ([10, 11],))
    cached = FakeCachedReqs(req_ids=["req-running"], new_block_ids=[([12],)])

    leader.build_connector_meta(_scheduler_output(new_reqs=[new_req], cached=cached))

    assert not [call for call in rust.calls if call[0] in ("set", "append")]


def _flat_feeds(rust):
    """The per-request flat block-id feeds from the last build_meta call."""
    output = rust.calls[-1][1]
    return {c["request_id"]: c["new_block_ids"] for c in output.cached_requests}


def _mapping_calls(rust, req_id):
    return [c for c in rust.calls if c[0] in ("set", "append") and c[1] == req_id]


def test_chunked_prefill_publishes_cumulative_full_lists(leader_module):
    """The chunked-prefill cadence fix: admission stores the full lists; a
    running step's delta EXTENDS the store; the Rust mapping receives the
    CUMULATIVE lists (set, not append) and the flat feed is the FULL
    group-0 list — so the Rust slot's tracked list grows with every chunk
    and the offload candidates index real positions."""
    leader, rust = _make_leader(leader_module, 4)

    # Chunk 1 (admission): NewRequestData carries FULL per-group lists.
    new_req = FakeNewReq("req-c", ([10, 11], [20, 21], [30, 31], [40, 41]))
    leader.build_connector_meta(_scheduler_output(new_reqs=[new_req]))
    assert "req-c" in leader._req_full_blocks

    # Chunk 2 (running): vLLM's cached-request new_block_ids are per-group
    # DELTAS. The store must absorb them and republish FULL.
    cached = FakeCachedReqs(req_ids=["req-c"], new_block_ids=[([12], [22], [32], [42])])
    leader.build_connector_meta(_scheduler_output(cached=cached))

    calls = _mapping_calls(rust, "req-c")
    assert [c[0] for c in calls] == [
        "set",
        "set",
    ], "the cumulative mapping is REPLACED (not appended) every step"
    assert calls[1][2] == [[10, 11, 12], [20, 21, 22], [30, 31, 32], [40, 41, 42]]
    assert _flat_feeds(rust)["req-c"] == [
        10,
        11,
        12,
    ], "the flat feed must be the FULL cumulative group-0 list, not the delta"


def test_cow_replacement_delta_never_leaves_stale_positions(leader_module):
    """The cm008 COW shape: a NON-resumed step's new_block_ids carry the
    replacement-shaped lists ([0, 6] where 6 replaces mid-list id 1). The
    leader mirrors the Rust reconcile (head-match + divergent tail →
    REPLACE) so every published position holds vLLM's CURRENT id — the
    reassigned-away ids must never remain for the offload consumers."""
    leader, rust = _make_leader(leader_module, 4)

    new_req = FakeNewReq("req-cow", ([0, 1], [0, 2], [0, 3], [4, 5]))
    leader.build_connector_meta(_scheduler_output(new_reqs=[new_req]))

    cached = FakeCachedReqs(
        req_ids=["req-cow"], new_block_ids=[([0, 6], [0, 7], [0, 8], [4, 9])]
    )
    leader.build_connector_meta(_scheduler_output(cached=cached))

    calls = _mapping_calls(rust, "req-cow")
    assert [c[0] for c in calls] == ["set", "set"]
    assert calls[1][2] == [[0, 6], [0, 7], [0, 8], [4, 9]], (
        "the COW replacement must REPLACE the stored lists — no stale "
        "position may survive for the offload consumers"
    )
    assert _flat_feeds(rust)["req-cow"] == [0, 6]


def test_cow_replacement_then_pure_append_extends(leader_module):
    """After the COW step replaced the store, the next step's genuine
    tail-appends extend the replaced lists (the cadence self-heals)."""
    leader, rust = _make_leader(leader_module, 4)

    leader.build_connector_meta(
        _scheduler_output(
            new_reqs=[FakeNewReq("req-h", ([0, 1], [0, 2], [0, 3], [4, 5]))]
        )
    )
    leader.build_connector_meta(
        _scheduler_output(
            cached=FakeCachedReqs(
                req_ids=["req-h"], new_block_ids=[([0, 6], [0, 7], [0, 8], [4, 9])]
            )
        )
    )
    leader.build_connector_meta(
        _scheduler_output(
            cached=FakeCachedReqs(
                req_ids=["req-h"], new_block_ids=[([9], [9], [9], [9])]
            )
        )
    )

    calls = _mapping_calls(rust, "req-h")
    assert [c[0] for c in calls] == ["set", "set", "set"]
    # Groups 0-2 extend with the new id; group 3's delta [9] is already
    # tracked at its tail — the suffix-overlap dedup adds nothing.
    assert calls[2][2] == [[0, 6, 9], [0, 7, 9], [0, 8, 9], [4, 9]]
    assert _flat_feeds(rust)["req-h"] == [0, 6, 9]


def test_resumed_request_replaces_cumulative_store(leader_module):
    """A resumed-from-preemption step carries FULL per-group lists — they
    REPLACE the store (preemption invalidated the old tracking)."""
    leader, rust = _make_leader(leader_module, 4)

    leader.build_connector_meta(
        _scheduler_output(
            new_reqs=[FakeNewReq("req-r", ([10, 11], [20, 21], [30, 31], [40, 41]))]
        )
    )
    cached = FakeCachedReqs(
        req_ids=["req-r"],
        new_block_ids=[([50, 51], [60, 61], [70, 71], [80, 81])],
        resumed_req_ids=("req-r",),
    )
    leader.build_connector_meta(_scheduler_output(cached=cached))

    calls = _mapping_calls(rust, "req-r")
    assert [c[0] for c in calls] == ["set", "set"]
    assert calls[1][2] == [[50, 51], [60, 61], [70, 71], [80, 81]]
    assert _flat_feeds(rust)["req-r"] == [50, 51]


def test_none_delta_republishes_stored_lists(leader_module):
    """A scheduled-with-no-new-blocks step republishes the stored lists
    (state unchanged) and feeds the stored group-0 list."""
    leader, rust = _make_leader(leader_module, 4)

    leader.build_connector_meta(
        _scheduler_output(
            new_reqs=[FakeNewReq("req-i", ([10, 11], [20, 21], [30, 31], [40, 41]))]
        )
    )
    leader.build_connector_meta(
        _scheduler_output(
            cached=FakeCachedReqs(req_ids=["req-i"], new_block_ids=[None])
        )
    )

    calls = _mapping_calls(rust, "req-i")
    assert [c[0] for c in calls] == ["set", "set"]
    assert calls[1][2] == [[10, 11], [20, 21], [30, 31], [40, 41]]
    assert _flat_feeds(rust)["req-i"] == [10, 11]


def test_request_finished_drops_cumulative_store(leader_module):
    leader, rust = _make_leader(leader_module, 4)

    leader.build_connector_meta(
        _scheduler_output(
            new_reqs=[FakeNewReq("req-f", ([10, 11], [20, 21], [30, 31], [40, 41]))]
        )
    )
    assert "req-f" in leader._req_full_blocks

    leader.request_finished(FakeRequest("req-f"), [10, 11])

    assert "req-f" not in leader._req_full_blocks


def test_single_group_feeds_unchanged(leader_module):
    """Single-group: no mapping calls at all and the flat feeds stay the
    raw per-step deltas (the legacy group-0 contract)."""
    leader, rust = _make_leader(leader_module, 1)

    new_req = FakeNewReq("req-new", ([10, 11],))
    cached = FakeCachedReqs(
        req_ids=["req-running", "req-idle"], new_block_ids=[([12],), None]
    )
    leader.build_connector_meta(_scheduler_output(new_reqs=[new_req], cached=cached))

    assert not [c for c in rust.calls if c[0] in ("set", "append")]
    feeds = _flat_feeds(rust)
    assert feeds["req-running"] == [12]
    assert feeds["req-idle"] == []


def test_single_group_config_defaults_to_one(leader_module):
    leader = leader_module.KvConnectorLeader(
        vllm_config=_fake_vllm_config(),
        engine_id="engine-0",
        kv_cache_config=None,
    )
    assert leader._num_groups == 1

    leader, _ = _make_leader(leader_module, 4)
    assert leader._num_groups == 4


# ---------------------------------------------------------------------
# The hybrid onboard SAFETY GATE: multi-group (hybrid) models must not
# receive external KV onboarding by default — the fp4 determinism test
# proved onboarding restores content that corrupts generation (vLLM
# 0.29 HMA per-group lists are not position-aligned for cached prefixes:
# live evidence group_lengths [4,4,4,10]). Offload (D2H) continues;
# onboarding needs the per-group semantics (the DEP).
# ---------------------------------------------------------------------
class _GateProbeRequest:
    request_id = "gate-req-1"
    num_tokens = 100
    cache_salt = None
    lora_request = None
    all_token_ids = [1] * 100


def _gate_probe(leader):
    """Wrap the rust connector's external lookup with a call counter."""
    calls = []

    def _recording(request_id, num_tokens, num_computed):
        calls.append((request_id, num_tokens, num_computed))
        return num_computed, False

    leader._connector.get_num_new_matched_tokens = _recording
    return calls


def test_hybrid_gate_blocks_external_lookup_by_default(leader_module, monkeypatch):
    monkeypatch.delenv("DYN_KVBM_HYBRID_EXPERIMENTAL_ONBOARD", raising=False)
    leader, _ = _make_leader(leader_module, 4)
    calls = _gate_probe(leader)

    num_computed, deferred = leader.get_num_new_matched_tokens(
        _GateProbeRequest(), num_computed_tokens=40
    )

    # The no-hit answer (recompute), and the external lookup never ran.
    assert (num_computed, deferred) == (40, False)
    assert calls == []


def test_hybrid_gate_opens_with_the_experimental_env(leader_module, monkeypatch):
    monkeypatch.setenv("DYN_KVBM_HYBRID_EXPERIMENTAL_ONBOARD", "1")
    leader, _ = _make_leader(leader_module, 4)
    calls = _gate_probe(leader)

    leader.get_num_new_matched_tokens(_GateProbeRequest(), num_computed_tokens=40)

    assert len(calls) == 1


def test_single_group_bypasses_the_gate(leader_module, monkeypatch):
    monkeypatch.delenv("DYN_KVBM_HYBRID_EXPERIMENTAL_ONBOARD", raising=False)
    leader, _ = _make_leader(leader_module, 1)
    calls = _gate_probe(leader)

    leader.get_num_new_matched_tokens(_GateProbeRequest(), num_computed_tokens=40)

    # Attention-only: the external lookup runs (today's behavior).
    assert len(calls) == 1
