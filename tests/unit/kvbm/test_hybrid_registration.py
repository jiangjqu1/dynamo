# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hybrid (HMA) registration handoff and wait_for_save wiring.

``KvConnectorWorker.register_kv_caches`` must pass the per-group plan
(``group_of_layer``/``group_specs``) and ``active_save_layers`` to the
Rust PyO3 connector for hybrid models, and pass ``None`` for all three
for attention-only models so the legacy path stays byte-identical. The
mamba ``page_bytes`` must come from the physical tensor shape, not the
vLLM spec's page size. The Rust connector is faked here — these tests
assert the REGISTRATION ARGUMENTS, not Rust behavior. Shared fakes and
the source-module loader live in ``conftest.py``.
"""

import pytest
import torch

from tests.unit.kvbm.conftest import (
    _FAKE_EVENT_HANDLE,
    FakeGroup,
    FakeKVCacheConfig,
    FakeVllmConfig,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.kvbm,
]

_MAMBA_LAYERS = (0, 1, 2, 4, 5)
_ATTN_LAYERS = (3, 6)


def _fp4_names() -> list[str]:
    """Registration-order layer names for the fp4 pattern."""
    kinds = {
        0: "mamba",
        1: "mamba",
        2: "mamba",
        3: "self_attn",
        4: "mamba",
        5: "mamba",
        6: "self_attn",
    }
    return [f"model.layers.{i}.{kinds[i]}" for i in sorted(kinds)]


def _fp4_kv_caches() -> dict:
    """The fp4 hybrid pattern: mamba span with attention layers at 3 and 6.

    Mamba layers arrive as list[Tensor] (one per step-state view); the
    spec's page_size is deliberately NOT the physical row size, so any
    regression that trusts the spec instead of the tensor shape fails
    the page_bytes assertion.
    """
    kv_caches: dict = {}
    for i in _MAMBA_LAYERS:
        kv_caches[f"model.layers.{i}.mamba"] = [torch.zeros(4, 96, dtype=torch.uint8)]
    for i in _ATTN_LAYERS:
        kv_caches[f"model.layers.{i}.self_attn"] = torch.zeros(
            2, 4, 16, 64, dtype=torch.uint8
        )
    return kv_caches


def _fp4_config() -> FakeVllmConfig:
    groups = [
        FakeGroup(
            [f"model.layers.{i}.mamba" for i in _MAMBA_LAYERS],
            "mamba",
            page_size=100,  # padded spec value: NOT the physical row (96)
        ),
        FakeGroup(
            [f"model.layers.{i}.self_attn" for i in _ATTN_LAYERS],
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
    assert [name for name, _t in args[4]] == _fp4_names()
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


def test_hybrid_handoff_derives_physical_mamba_page_bytes(
    connector_worker_module,
):
    # Direct unit test of the handoff payload construction (no CUDA, no
    # register flow): the mamba page_bytes must come from the physical
    # tensor shape even when the spec's page_size says otherwise.
    module = connector_worker_module
    worker = _make_worker(module, FakeVllmConfig(kv_cache_config=None))
    worker.group_plans = module.derive_layer_groups(
        FakeKVCacheConfig(_fp4_config().kv_cache_config.kv_cache_groups),
        _fp4_names(),
    )

    name_to_tensor = {
        f"model.layers.{i}.mamba": torch.zeros(4, 96, dtype=torch.uint8)
        for i in _MAMBA_LAYERS
    }
    name_to_tensor.update(
        {f"model.layers.{i}.self_attn": torch.zeros(2, 4, 16, 64) for i in _ATTN_LAYERS}
    )

    group_of_layer, group_specs, active = worker._hybrid_handoff(
        name_to_tensor, attention_layer_count=2, recurrent_layer_count=5
    )
    assert group_of_layer == [0, 0, 0, 1, 0, 0, 1]
    assert group_specs == [(True, 96), (False, None)]
    assert active == 2

    # Attention-only: all three None.
    result = worker._hybrid_handoff(name_to_tensor, 2, recurrent_layer_count=0)
    assert result == (None, None, None)


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
