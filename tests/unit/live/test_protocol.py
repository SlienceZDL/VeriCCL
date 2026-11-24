import pytest


def test_version_contents_are_immutable_and_all_ready_precedes_commit():
    from vericcl.live.protocol import GroupProtocol, ProtocolError
    group = GroupProtocol('group', (0, 1), 'contract')
    group.prepare('v1', 'xml-hash', 'contract')
    group.ready(0, 'v1', 'xml-hash')
    with pytest.raises(ProtocolError):
        group.commit('v1', request_id=1)
    group.ready(1, 'v1', 'xml-hash')
    decision = group.commit('v1', request_id=1)
    assert decision['version'] == 'v1' and decision['request_id'] == 1
    with pytest.raises(ProtocolError):
        group.prepare('v1', 'different-xml', 'contract')


def test_all_commit_acknowledgements_precede_start_and_old_ack_cannot_vote():
    from vericcl.live.protocol import GroupProtocol, ProtocolError
    group = GroupProtocol('group', (0, 1), 'contract')
    group.prepare('v1', 'xml', 'contract')
    for rank in (0, 1):
        group.ready(rank, 'v1', 'xml')
    decision = group.commit('v1', request_id=1)
    group.acknowledge_commit(0, decision['decision_id'])
    with pytest.raises(ProtocolError):
        group.start(1, attempt=0)
    with pytest.raises(ProtocolError):
        group.acknowledge_commit(1, 'another-decision')
    group.acknowledge_commit(1, decision['decision_id'])
    start = group.start(1, attempt=0)
    assert start['version'] == 'v1'
    assert group.start(1, attempt=0) == start
    with pytest.raises(ProtocolError):
        group.start(2, attempt=0)


def test_unknown_start_result_stops_new_requests_without_reverting_one_rank():
    from vericcl.live.protocol import GroupProtocol, ProtocolError
    group = GroupProtocol('group', (0, 1), 'contract')
    group.prepare('v1', 'xml', 'contract')
    for rank in (0, 1): group.ready(rank, 'v1', 'xml')
    decision = group.commit('v1', request_id=1)
    for rank in (0, 1): group.acknowledge_commit(rank, decision['decision_id'])
    group.start(1, attempt=0)
    group.uncertain('START acknowledgement missing')
    assert group.status()['state'] == 'uncertain'
    assert group.status()['committed_version'] == 'v1'
    with pytest.raises(ProtocolError): group.start(2, attempt=0)


def test_request_is_successful_only_after_all_real_gpu_completion_receipts():
    from vericcl.live.protocol import GroupProtocol, ProtocolError
    group = GroupProtocol('group', (0, 1), 'contract')
    group.prepare('v1', 'xml', 'contract')
    for rank in (0, 1): group.ready(rank, 'v1', 'xml')
    decision = group.commit('v1', request_id=1)
    for rank in (0, 1): group.acknowledge_commit(rank, decision['decision_id'])
    group.start(1, attempt=0)
    with pytest.raises(ProtocolError): group.complete(0, 1, 0, gpu_completed=False, output_valid=True)
    group.complete(0, 1, 0, gpu_completed=True, output_valid=True)
    assert group.status()['unfinished_request'] == 1
    group.complete(1, 1, 0, gpu_completed=True, output_valid=True)
    assert group.status()['unfinished_request'] is None and group.status()['last_successful_request'] == 1
    assert group.complete(1, 1, 0, gpu_completed=True, output_valid=True)['logical_successes'] == 1
