# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""probe_log_enabled: the DYN_LOG gate for the temporary connector probes.

EngineCore never surfaces kvbm debug/info records (root stays at WARNING
and no Dynamo log bridge runs there), so the Python-side temporary probes
key off DYN_LOG directly. These tests pin the directive semantics the
gate shares with the Rust probes: a global debug/trace level or a
kvbm-targeted debug/trace directive opens the gate; info-level and
other-target directives do not.
"""

import importlib.util
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.kvbm,
]

_UTILS_PATH = (
    Path(__file__).resolve().parents[3] / "lib/bindings/kvbm/python/kvbm/utils.py"
)


@pytest.fixture(name="probe_gate")
def load_probe_gate():
    """Load utils.py fresh from source.

    The kvbm package import needs nixl and the built ``_core`` extension
    (absent from the source tree), so the module under test is loaded
    directly from its file, exactly like the conftest loads group_plan.py.
    A fresh module also hands each test an empty lru_cache.
    """
    spec = importlib.util.spec_from_file_location("kvbm_test_utils_gate", _UTILS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.probe_log_enabled


@pytest.mark.parametrize(
    "dyn_log,expected",
    [
        (None, False),  # unset defaults to info — probes hidden
        ("info", False),
        ("warn", False),
        ("error", False),
        ("debug", True),  # global debug — probes shown
        ("trace", True),
        ("info,kvbm=debug", True),  # kvbm-targeted debug directive
        ("info,kvbm.vllm_integration=trace", True),
        ("info,kvbm=info", False),  # kvbm target at a non-debug level
        ("info,_core=debug", False),  # other targets never open the Python gate
        ("info,dynamo_llm::block_manager=debug", False),
        ("info, _core=debug , kvbm = debug", True),  # tolerant of stray spaces
    ],
)
def test_probe_gate_directives(monkeypatch, probe_gate, dyn_log, expected):
    if dyn_log is None:
        monkeypatch.delenv("DYN_LOG", raising=False)
    else:
        monkeypatch.setenv("DYN_LOG", dyn_log)
    assert probe_gate() is expected
