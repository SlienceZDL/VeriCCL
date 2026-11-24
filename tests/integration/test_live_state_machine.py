from dataclasses import replace

import pytest


def summary(request, value=100., *, version='A', qualified=True):
    from vericcl.live.telemetry import RequestSummary
    return RequestSummary('g', request, 0, version, qualified,
        {'request': value, 'service:link:0:1': value / 2}, (), '2026-11-04T00:00:00Z')


def feed(trigger, first, count, value, **kwargs):
    result = None
    for request in range(first, first + count):
        result = trigger.observe_request(summary(request, value, **kwargs))
    return result


def test_two_disjoint_windows_share_a_direction_and_ignore_unqualified_requests():
    from vericcl.live.triggers import LiveTrigger
    trigger = LiveTrigger()
    assert feed(trigger, 1, 8, 100) is None
    assert feed(trigger, 9, 8, 115) is None
    assert feed(trigger, 17, 8, 90) is None  # direction changed
    assert feed(trigger, 25, 8, 90, qualified=False) is None
    event = feed(trigger, 33, 8, 89)
    assert event.kind == 'performance' and event.request_id == 40
    assert event.changes['request'] < -.1
    assert trigger.observe_request(summary(40, 89)) is None  # duplicate receipt


def test_residence_is_counted_by_successful_requests_and_faults_bypass_it():
    from vericcl.live.triggers import LiveTrigger
    trigger = LiveTrigger(window_requests=2, consecutive_windows=2)
    feed(trigger, 1, 2, 100)
    trigger.switched('B', successful_request=2)
    feed(trigger, 3, 2, 100, version='B')
    assert feed(trigger, 5, 4, 120, version='B') is None
    emergency = trigger.emergency('timeout', request_id=9, version='B', evidence={'source': 'runtime'})
    assert emergency.kind == 'timeout' and emergency.invalid_links == ()
    assert feed(trigger, 9, 26, 120, version='B') is not None


def test_backend_tasks_coalesce_and_revalidate_current_version():
    import threading
    from vericcl.live.triggers import TriggerEvent
    from vericcl.live.background import BackgroundWorker
    entered, release = threading.Event(), threading.Event()
    calls = []
    def solve(snapshot, events, lease):
        calls.append(events); entered.set(); release.wait(1); lease.check(); return {'candidate': 'candidate-B'}
    worker = BackgroundWorker(solve, timeout_s=2)
    event = TriggerEvent('performance', 8, 'A', {'request': .2})
    worker.offer({'version': 'A'}, event); assert entered.wait(1)
    worker.offer({'version': 'A'}, replace(event, request_id=9))
    worker.offer({'version': 'A'}, replace(event, request_id=10, kind='fault', invalid_links=((0, 1),)))
    assert len(calls) == 1 and worker.pending_count == 2
    release.set(); worker.join(1)
    result = worker.take_completed()
    assert result.base_version == 'A'
    assert worker.valid_for(result, 'A') and not worker.valid_for(result, 'B')
    worker.close(1)


def test_telemetry_does_not_mark_unused_links_zero_or_treat_retries_as_stable_samples():
    from vericcl.live.telemetry import summarize_request
    record = dict(logical_success=True, request_id=1, attempt=0, version_id='A', ranks=[
        dict(rank=r, request_id=1, attempt=0, version_id='A', gpu_completed=True, correct=True,
             device_time_us=10, telemetry=[dict(kind='s' if r == 0 else 'r', operation_id='op',
                send_peer=1 if r == 0 else -1, recv_peer=0 if r == 1 else -1, channel=0,
                payload_bytes=1024, service_ns=5000, dependency_wait_ns=1000, samples=1, invalid_samples=0,
                source='software_device_counters')]) for r in range(2)])
    value = summarize_request('g', record, ranks=2)
    assert value.qualified and value.metrics['service:link:0:1'] == 5
    assert 'service:link:1:0' not in value.metrics
    record['attempt'] = 1
    for rank in record['ranks']: rank['attempt'] = 1
    assert not summarize_request('g', record, ranks=2).qualified


def test_external_fault_requires_explicit_source_time_scope_and_real_edge():
    from vericcl.live.telemetry import parse_external
    message = dict(schema_version=1, event_id='e1', group_id='g', source='external_health',
        observed_at='2026-11-04T00:00:00Z', kind='link_failed', links=[[0, 1]], metrics={})
    event = parse_external(message, group_id='g', legal_links={(0, 1)}, current_version='A', request_id=4)
    assert event.invalid_links == ((0, 1),) and event.kind == 'fault'
    with pytest.raises(ValueError): parse_external(message | {'links': [[1, 0]]}, group_id='g', legal_links={(0, 1)}, current_version='A', request_id=4)
