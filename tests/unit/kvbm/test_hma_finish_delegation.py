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
    """DynamoConnector-shaped stub: request_finished records calls and
    echoes a canned leader answer."""

    def __init__(self):
        self.calls: list[tuple[object, list[int]]] = []
        self.answer = (False, None)

    def request_finished(self, request, block_ids):
        self.calls.append((request, block_ids))
        return self.answer


def _finish_all_groups(module, stub, request, block_ids):
    return module.DynamoConnector.request_finished_all_groups(stub, request, block_ids)


def test_single_group_delegates_to_leader_finish(connector_module):
    stub = _RecordingFinisher()
    result = _finish_all_groups(connector_module, stub, "req-1", ([11, 12, 13],))
    assert stub.calls == [("req-1", [11, 12, 13])]
    assert result == (False, None)


def test_multi_group_delegates_with_group_zero_list(connector_module):
    # The Phase-0 stopgap returned (False, None) for multi-group, so the
    # leader's mark_as_finished never ran and finished slots never became
    # matchable for onboarding.
    stub = _RecordingFinisher()
    result = _finish_all_groups(
        connector_module, stub, "req-2", ([11, 12, 13], [11, 12, 13])
    )
    assert stub.calls == [("req-2", [11, 12, 13])]
    assert result == (False, None)


def test_multi_group_divergent_lists_still_delegate_group_zero(connector_module):
    # Group 0 is authoritative; the HMA-PROBE log in connector_leader.py
    # verifies live on the fp4 that divergence never actually happens.
    stub = _RecordingFinisher()
    _finish_all_groups(connector_module, stub, "req-3", ([11, 12], [99, 98]))
    assert stub.calls == [("req-3", [11, 12])]


def test_empty_groups_tuple_does_not_delegate(connector_module):
    stub = _RecordingFinisher()
    result = _finish_all_groups(connector_module, stub, "req-4", ())
    assert stub.calls == []
    assert result == (False, None)


def test_leader_answer_passes_through_delegation(connector_module):
    # Whatever the leader's request_finished returns (e.g. a future
    # async-hold answer) must survive the delegation unchanged.
    stub = _RecordingFinisher()
    stub.answer = (True, {"remote_blocks": [7]})
    result = _finish_all_groups(connector_module, stub, "req-5", ([5], [5]))
    assert result == (True, {"remote_blocks": [7]})
