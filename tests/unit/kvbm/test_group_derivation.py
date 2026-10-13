# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""derive_layer_groups: order-preserving partition of registered layers.

The kvbm package in this environment resolves to a site-packages install
whose built ``_core`` extension is absent from the source tree, so the
module under test is loaded directly from its source file instead of via
a package import. The vLLM spec/config fakes are shared via
``conftest.py``.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

from tests.unit.kvbm.conftest import FakeGroup, FakeKVCacheConfig

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.kvbm,
]

_MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "lib/bindings/kvbm/python/kvbm/vllm_integration/group_plan.py"
)
_spec = importlib.util.spec_from_file_location("_group_plan_under_test", _MODULE_PATH)
group_plan = importlib.util.module_from_spec(_spec)
# Standard module_from_spec recipe: register in sys.modules before
# exec_module so inspect / get_type_hints / pickle can resolve the module.
sys.modules[_spec.name] = group_plan
_spec.loader.exec_module(group_plan)

GroupPlan = group_plan.GroupPlan
derive_layer_groups = group_plan.derive_layer_groups


def test_two_groups_partition_in_layer_order():
    groups = [
        FakeGroup(["m0", "m1", "m2", "m4", "m5"], "mamba", 3207168),
        FakeGroup(["a3", "a6"], "attn", 3211264),
    ]
    plan = derive_layer_groups(
        FakeKVCacheConfig(groups), ["m0", "m1", "m2", "a3", "m4", "m5", "a6"]
    )
    assert [g.group_id for g in plan] == [0, 1]
    assert isinstance(plan[0], GroupPlan)
    assert plan[0].layer_names == ["m0", "m1", "m2", "m4", "m5"]
    assert plan[0].group_of_layer == [0, 0, 0, 1, 0, 0, 1]  # parallel to layer order
    assert plan[0].is_mamba and not plan[1].is_mamba
    assert plan[1].page_bytes == 3211264


def test_missing_layer_names_falls_back_to_single_group():
    plan = derive_layer_groups(None, ["a", "b"])
    assert len(plan) == 1 and plan[0].group_of_layer == [0, 0]


def test_unregistered_group_layer_is_ignored():
    groups = [FakeGroup(["a", "ghost"], "attn", 128)]
    plan = derive_layer_groups(FakeKVCacheConfig(groups), ["a"])
    assert plan[0].layer_names == ["a"]


def test_registered_layer_missing_from_all_groups_files_into_group_zero():
    groups = [FakeGroup(["a"], "attn", 128)]
    plan = derive_layer_groups(FakeKVCacheConfig(groups), ["a", "b"])
    assert plan[0].layer_names == ["a", "b"]
    assert plan[0].group_of_layer == [0, 0]
