# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HMA finish-path delegation (request_finished_all_groups).

Multi-group (hybrid) finishes must run the same leader finish bookkeeping
as single-group ones — group 0's block-id list is authoritative — instead
of returning the Phase-0 ``(False, None)`` stopgap that left slots
unfinishable and never matchable for onboarding. The REAL
``DynamoConnector`` class is loaded from source with vLLM stubbed
(conftest ``connector_module`` fixture) and the method is exercised
unbound on a recording stub, so no vLLM import or leader construction is
needed: these tests assert the DELEGATION DECISION, not leader behavior.
"""

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.kvbm,
]


class _RecordingFinisher:
    """KvConnectorLeader-shaped stub: set_per_group_device_blocks and
    request_finished record calls and echo a canned leader answer."""

    def __init__(self, num_groups=1):
        self._num_groups = num_groups
        # request_finished_all_groups routes through self._scheduler (the
        # leader); the stub stands in for BOTH objects.
        self._scheduler = self
        self.finish_calls: list[tuple[object, list[int]]] = []
        self.per_group_calls: list[tuple[str, tuple[list[int], ...]]] = []
        self.answer = (False, None)

    def set_per_group_device_blocks(self, request_id, per_group_block_ids):
        self.per_group_calls.append((request_id, per_group_block_ids))

    def request_finished(self, request, block_ids):
        self.finish_calls.append((request.request_id, block_ids))
        return self.answer


class _FakeFinishRequest:
    """Stand-in for vLLM's Request (only .request_id is consulted)."""

    def __init__(self, request_id):
        self.request_id = request_id


def _finish_all_groups(module, stub, request, block_ids):
    return module.DynamoConnector.request_finished_all_groups(stub, request, block_ids)


def test_single_group_delegates_to_leader_finish(connector_module):
    stub = _RecordingFinisher()
    result = _finish_all_groups(
        connector_module, stub, _FakeFinishRequest("req-1"), ([11, 12, 13],)
    )
    assert stub.finish_calls == [("req-1", [11, 12, 13])]
    assert result == (False, None)


def test_single_group_does_not_publish_per_group_mapping(connector_module):
    # The per-group mapping only exists in multi-group deployments; the
    # single-group finish path must stay free of the extra Rust round-trip.
    stub = _RecordingFinisher(num_groups=1)
    _finish_all_groups(
        connector_module, stub, _FakeFinishRequest("req-1"), ([11, 12, 13],)
    )
    assert stub.per_group_calls == []


def test_multi_group_delegates_with_group_zero_list(connector_module):
    # The Phase-0 stopgap returned (False, None) for multi-group, so the
    # leader's mark_as_finished never ran and finished slots never became
    # matchable for onboarding.
    stub = _RecordingFinisher(num_groups=4)
    result = _finish_all_groups(
        connector_module,
        stub,
        _FakeFinishRequest("req-2"),
        ([11, 12, 13], [11, 12, 13]),
    )
    assert stub.finish_calls == [("req-2", [11, 12, 13])]
    assert result == (False, None)


def test_multi_group_publishes_divergent_lists_before_delegation(connector_module):
    # The HMA-PROBE proved ids diverge ([[1], [2], [3], [4]]): the FULL
    # per-group lists must reach the slot mapping BEFORE the finish
    # delegation, ordered so any late consumer sees the final mapping.
    stub = _RecordingFinisher(num_groups=4)
    per_group = ([11, 12], [21, 22], [31, 32], [41, 42])
    _finish_all_groups(connector_module, stub, _FakeFinishRequest("req-3"), per_group)
    assert stub.per_group_calls == [("req-3", per_group)]
    assert stub.finish_calls == [("req-3", [11, 12])]


def test_multi_group_divergent_lists_still_delegate_group_zero(connector_module):
    # The finish delegation keeps group 0's list for the slot's flat
    # device-block sequence; the per-group mapping carries the divergence.
    stub = _RecordingFinisher(num_groups=2)
    _finish_all_groups(
        connector_module, stub, _FakeFinishRequest("req-3"), ([11, 12], [99, 98])
    )
    assert stub.finish_calls == [("req-3", [11, 12])]
    assert stub.per_group_calls == [("req-3", ([11, 12], [99, 98]))]


def test_empty_groups_tuple_does_not_delegate(connector_module):
    stub = _RecordingFinisher(num_groups=2)
    result = _finish_all_groups(connector_module, stub, _FakeFinishRequest("req-4"), ())
    assert stub.finish_calls == []
    assert stub.per_group_calls == []
    assert result == (False, None)


def test_leader_answer_passes_through_delegation(connector_module):
    # Whatever the leader's request_finished returns (e.g. a future
    # async-hold answer) must survive the delegation unchanged.
    stub = _RecordingFinisher(num_groups=2)
    stub.answer = (True, {"remote_blocks": [7]})
    result = _finish_all_groups(
        connector_module, stub, _FakeFinishRequest("req-5"), ([5], [5])
    )
    assert result == (True, {"remote_blocks": [7]})
