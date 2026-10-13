# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hybrid (HMA) registration handoff and wait_for_save wiring.

``KvConnectorWorker.register_kv_caches`` must pass the per-group plan
(``group_of_layer``/``group_specs``) and ``active_save_layers`` to the
Rust PyO3 connector for hybrid models, and pass ``None`` for all three
for attention-only models so the legacy path stays byte-identical. The
mamba ``page_bytes`` must come from the physical tensor shape, not the
vLLM spec's page size. The Rust connector is faked here — these tests
assert the REGISTRATION ARGUMENTS, not Rust behavior.

Like ``test_group_derivation.py``, the module under test is loaded from
source; its heavy ``kvbm`` imports (nixl / the built ``_core``
extension) are stubbed, and the ``torch.cuda`` event API is patched so
the tests run on CPU.
"""

import importlib.util
import sys
import types
from pathlib import Path
from typing import Optional

import pytest
import torch

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

_FAKE_EVENT_HANDLE = 4242


class FakeCudaEvent:
    """Stand-in for torch.cuda.Event (CPU-only test environment)."""

    def __init__(self, enable_timing=False, interprocess=False):
        self.record_count = 0

    def record(self, stream=None):
        self.record_count += 1

    @property
    def cuda_event(self):
        return _FAKE_EVENT_HANDLE


class FakeRustKvConnectorWorker:
    """Recording stand-in for kvbm.vllm_integration.rust.KvConnectorWorker."""

    def __init__(self, drt, engine_id):
        self.drt = drt
        self.engine_id = engine_id
        self.register_calls: list[dict] = []
        self.wait_for_save_handles: list[int] = []

    def register_kv_caches(self, *args, **kwargs):
        self.register_calls.append({"args": args, "kwargs": kwargs})

    def wait_for_save(self, event_handle):
        self.wait_for_save_handles.append(event_handle)

    def bind_connector_metadata(self, data: bytes) -> None:
        pass

    def clear_connector_metadata(self) -> None:
        pass

    def save_kv_layer(self, layer_name: str, kv_layer) -> None:
        pass

    def get_finished(self, finished_req_ids):
        return set(), set()


class MambaSpec:
    """Named after vLLM's hybrid recurrent-state spec class."""

    def __init__(self, page_size):
        self.page_size = page_size


class AttnSpec:
    def __init__(self, page_size):
        self.page_size = page_size


class FakeGroup:
    def __init__(self, layer_names, kind, page_size):
        self.layer_names = layer_names
        self.kv_cache_spec = {"mamba": MambaSpec, "attn": AttnSpec}[kind](page_size)


class FakeKVCacheConfig:
    def __init__(self, groups):
        self.kv_cache_groups = groups


class FakeVllmConfig:
    """Stands in for vllm.config.VllmConfig (attribute surface only)."""

    def __init__(self, kv_cache_config: Optional[FakeKVCacheConfig]):
        cache_config = types.SimpleNamespace(block_size=16, cache_dtype="auto")
        model_config = types.SimpleNamespace(use_mla=False, dtype=torch.uint8)
        self.cache_config = cache_config
        self.model_config = model_config
        self.kv_cache_config = kv_cache_config


@pytest.fixture(name="connector_worker_module")
def load_connector_worker_module(monkeypatch):
    """Load connector_worker.py from source with kvbm's heavy imports stubbed.

    The real ``kvbm`` package import pulls in nixl and the built ``_core``
    extension; a fake package hierarchy lets the module under test resolve
    ``kvbm.utils``, the relative ``.group_plan`` import, and
    ``kvbm.vllm_integration.rust`` without them. monkeypatch restores
    sys.modules after each test.
    """
    kvbm_pkg = types.ModuleType("kvbm")
    kvbm_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "kvbm", kvbm_pkg)

    utils_mod = types.ModuleType("kvbm.utils")
    utils_mod.is_dyn_runtime_enabled = lambda: False
    utils_mod.nvtx_annotate = lambda category: (lambda func: func)
    monkeypatch.setitem(sys.modules, "kvbm.utils", utils_mod)

    vllm_integration = types.ModuleType("kvbm.vllm_integration")
    vllm_integration.__path__ = []
    monkeypatch.setitem(sys.modules, "kvbm.vllm_integration", vllm_integration)

    group_plan_spec = importlib.util.spec_from_file_location(
        "kvbm.vllm_integration.group_plan",
        _VLLM_INTEGRATION_DIR / "group_plan.py",
    )
    group_plan = importlib.util.module_from_spec(group_plan_spec)
    monkeypatch.setitem(sys.modules, group_plan_spec.name, group_plan)
    group_plan_spec.loader.exec_module(group_plan)

    rust_mod = types.ModuleType("kvbm.vllm_integration.rust")
    rust_mod.KvConnectorWorker = FakeRustKvConnectorWorker
    monkeypatch.setitem(sys.modules, "kvbm.vllm_integration.rust", rust_mod)

    worker_spec = importlib.util.spec_from_file_location(
        "kvbm.vllm_integration.connector_worker",
        _VLLM_INTEGRATION_DIR / "connector_worker.py",
    )
    worker_mod = importlib.util.module_from_spec(worker_spec)
    monkeypatch.setitem(sys.modules, worker_spec.name, worker_mod)
    worker_spec.loader.exec_module(worker_mod)

    monkeypatch.setattr(torch.cuda, "Event", FakeCudaEvent)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: None)
    return worker_mod


def _fp4_kv_caches() -> dict:
    """The fp4 hybrid pattern: mamba span with attention layers at 3 and 6.

    Mamba layers arrive as list[Tensor] (one per step-state view); the
    spec's page_size is deliberately NOT the physical row size, so any
    regression that trusts the spec instead of the tensor shape fails
    the page_bytes assertion.
    """
    kv_caches: dict = {}
    for i in (0, 1, 2, 4, 5):
        kv_caches[f"model.layers.{i}.mamba"] = [torch.zeros(4, 96, dtype=torch.uint8)]
    for i in (3, 6):
        kv_caches[f"model.layers.{i}.self_attn"] = torch.zeros(
            2, 4, 16, 64, dtype=torch.uint8
        )
    return kv_caches


def _fp4_config() -> FakeVllmConfig:
    groups = [
        FakeGroup(
            [
                "model.layers.0.mamba",
                "model.layers.1.mamba",
                "model.layers.2.mamba",
                "model.layers.4.mamba",
                "model.layers.5.mamba",
            ],
            "mamba",
            page_size=100,  # padded spec value: NOT the physical row (96)
        ),
        FakeGroup(
            ["model.layers.3.self_attn", "model.layers.6.self_attn"],
            "attn",
            page_size=16,
        ),
    ]
    return FakeVllmConfig(FakeKVCacheConfig(groups))


def _make_worker(module, vllm_config):
    return module.KvConnectorWorker(vllm_config=vllm_config, engine_id="test-engine")


def test_hybrid_registration_passes_groups_and_active_save_layers(
    connector_worker_module,
):
    module = connector_worker_module
    worker = _make_worker(module, _fp4_config())
    worker.register_kv_caches(_fp4_kv_caches())

    rust = worker._connector
    assert len(rust.register_calls) == 1
    call = rust.register_calls[0]
    args, kwargs = call["args"], call["kwargs"]

    # Ordered registration: 7 layers sorted by layer index.
    assert [name for name, _t in args[4]] == [
        "model.layers.0.mamba",
        "model.layers.1.mamba",
        "model.layers.2.mamba",
        "model.layers.3.self_attn",
        "model.layers.4.mamba",
        "model.layers.5.mamba",
        "model.layers.6.self_attn",
    ]
    # Mamba entries are the normalized flat raw tensors ([B, C]).
    assert tuple(args[4][0][1].shape) == (4, 96)

    # Legacy geometry arguments unchanged.
    assert args[0] == 4  # num_device_blocks (max of attention shape axes)
    assert args[1] == 16  # page_size = cache_config.block_size
    assert args[3] == 1  # dtype itemsize (uint8)
    assert kwargs["outer_dim"] is None
    assert kwargs["inner_dim"] is None

    # The hybrid handoff.
    assert kwargs["group_of_layer"] == [0, 0, 0, 1, 0, 0, 1]
    assert kwargs["group_specs"] == [(True, 96), (False, None)]
    assert kwargs["active_save_layers"] == 2


def test_attention_only_registration_keeps_legacy_payload(
    connector_worker_module,
):
    module = connector_worker_module
    kv_caches = {
        "model.layers.0.self_attn": torch.zeros(2, 4, 16, 64),
        "model.layers.1.self_attn": torch.zeros(2, 4, 16, 64),
    }
    worker = _make_worker(module, FakeVllmConfig(kv_cache_config=None))
    worker.register_kv_caches(kv_caches)

    kwargs = worker._connector.register_calls[0]["kwargs"]
    assert kwargs["group_of_layer"] is None
    assert kwargs["group_specs"] is None
    assert kwargs["active_save_layers"] is None


def test_attention_only_with_config_groups_keeps_legacy_payload(
    connector_worker_module,
):
    # An attention-only model whose vLLM config still defines one group
    # must not activate the grouped path: only hybrid registrations do.
    module = connector_worker_module
    groups = [
        FakeGroup(
            ["model.layers.0.self_attn", "model.layers.1.self_attn"],
            "attn",
            page_size=16,
        )
    ]
    kv_caches = {
        "model.layers.0.self_attn": torch.zeros(2, 4, 16, 64),
        "model.layers.1.self_attn": torch.zeros(2, 4, 16, 64),
    }
    worker = _make_worker(module, FakeVllmConfig(FakeKVCacheConfig(groups)))
    worker.register_kv_caches(kv_caches)

    kwargs = worker._connector.register_calls[0]["kwargs"]
    assert kwargs["group_of_layer"] is None
    assert kwargs["group_specs"] is None
    assert kwargs["active_save_layers"] is None


def test_hybrid_wait_for_save_records_fresh_event(connector_worker_module):
    module = connector_worker_module
    worker = _make_worker(module, _fp4_config())
    worker.register_kv_caches(_fp4_kv_caches())

    worker.wait_for_save()
    assert worker._connector.wait_for_save_handles == [_FAKE_EVENT_HANDLE]


def test_attention_only_wait_for_save_is_local_noop(connector_worker_module):
    module = connector_worker_module
    kv_caches = {
        "model.layers.0.self_attn": torch.zeros(2, 4, 16, 64),
    }
    worker = _make_worker(module, FakeVllmConfig(kv_cache_config=None))
    worker.register_kv_caches(kv_caches)

    worker.wait_for_save()
    assert worker._connector.wait_for_save_handles == []
