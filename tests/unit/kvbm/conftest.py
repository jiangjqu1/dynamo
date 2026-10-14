# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fakes and module loader for the kvbm unit tests.

The kvbm package in this environment resolves to a site-packages install
whose built ``_core`` extension is absent from the source tree, so the
modules under test are loaded directly from their source files with the
heavy ``kvbm`` imports (nixl / ``_core``) stubbed. The vLLM spec/config
fakes here stand in for vLLM's hybrid (HMA) classes; the spec TYPE NAME
(e.g. ``MambaSpec``) is the cross-version contract, per group_plan.py.
"""

import importlib.util
import sys
import types
from pathlib import Path
from typing import Optional

import pytest
import torch

_VLLM_INTEGRATION_DIR = (
    Path(__file__).resolve().parents[3]
    / "lib/bindings/kvbm/python/kvbm/vllm_integration"
)

_FAKE_EVENT_HANDLE = 4242


class MambaSpec:
    """Named after vLLM's hybrid recurrent-state spec class (name is the contract)."""

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
    """Stands in for vLLM's KVCacheConfig (the .kv_cache_groups holder)."""

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


class FakeKVConnectorBase_V1:
    """Stand-in for vLLM's KVConnectorBase_V1 (subclass/constructor surface)."""

    def __init__(self, vllm_config=None, role=None, kv_cache_config=None):
        pass


class FakeKVConnectorMetadata:
    """Stand-in for vLLM's KVConnectorMetadata (subclass target)."""


class FakeKVConnectorRole:
    """Stand-in for vLLM's KVConnectorRole enum (scheduler/worker)."""

    SCHEDULER = "scheduler"
    WORKER = "worker"


class FakeSupportsHMA:
    """Stand-in for vLLM's SupportsHMA marker mixin."""


@pytest.fixture(name="connector_worker_module")
def load_connector_worker_module(monkeypatch):
    """Load connector_worker.py from source with kvbm's heavy imports stubbed.

    A fake ``kvbm`` package hierarchy lets the module under test resolve
    ``kvbm.utils``, the relative ``.group_plan`` import (loaded from the
    REAL source file), and ``kvbm.vllm_integration.rust`` (the recording
    fake) without importing nixl or the built ``_core`` extension.
    monkeypatch restores sys.modules after each test.
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


@pytest.fixture(name="connector_module")
def load_connector_module(monkeypatch):
    """Load dynamo_connector.py from source with vllm and kvbm's heavy
    imports stubbed.

    The scheduler-facing delegation methods (``request_finished`` /
    ``request_finished_all_groups``) are pure routing over the leader, so
    vLLM's protocol base classes are faked, the leader/worker modules are
    stubbed, and the REAL ``DynamoConnector`` class is loaded from its
    source file. Tests exercise the methods unbound on caller-provided
    stubs — no vLLM import and no leader construction. monkeypatch
    restores sys.modules after each test.
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

    leader_mod = types.ModuleType("kvbm.vllm_integration.connector_leader")
    leader_mod.KvConnectorLeader = type("KvConnectorLeader", (), {})
    monkeypatch.setitem(sys.modules, leader_mod.__name__, leader_mod)

    worker_mod = types.ModuleType("kvbm.vllm_integration.connector_worker")
    worker_mod.KvConnectorWorker = type("KvConnectorWorker", (), {})
    monkeypatch.setitem(sys.modules, worker_mod.__name__, worker_mod)

    base_mod = types.ModuleType("vllm.distributed.kv_transfer.kv_connector.v1.base")
    base_mod.KVConnectorBase_V1 = FakeKVConnectorBase_V1
    base_mod.KVConnectorMetadata = FakeKVConnectorMetadata
    base_mod.KVConnectorRole = FakeKVConnectorRole
    base_mod.SupportsHMA = FakeSupportsHMA
    monkeypatch.setitem(sys.modules, base_mod.__name__, base_mod)

    output_mod = types.ModuleType("vllm.v1.core.sched.output")
    output_mod.SchedulerOutput = type("SchedulerOutput", (), {})
    monkeypatch.setitem(sys.modules, output_mod.__name__, output_mod)

    connector_spec = importlib.util.spec_from_file_location(
        "kvbm.vllm_integration.connector.dynamo_connector",
        _VLLM_INTEGRATION_DIR / "connector" / "dynamo_connector.py",
    )
    connector_mod = importlib.util.module_from_spec(connector_spec)
    monkeypatch.setitem(sys.modules, connector_spec.name, connector_mod)
    connector_spec.loader.exec_module(connector_mod)
    return connector_mod
