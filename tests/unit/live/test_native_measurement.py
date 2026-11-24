import pytest


def sample(rank, times):
    return dict(rank=rank, correct=True, gpu_completed=True, gpu_released=True,
                trace_enabled=False, device_times_us=times, version_id='v', contract_digest='c',
                warmup_iterations=5, timed_iterations=2, runtime_identity={'gpu_uuid': 'gpu-' + str(rank)})


def test_schedule_time_is_mean_of_per_invocation_slowest_rank():
    from vericcl.live.native import combine_schedule_results
    result = combine_schedule_results([sample(0, [10., 20.]), sample(1, [20., 10.])],
                                      rank_count=2, version='v', contract='c', warmup=5, timed=2)
    assert result['time_us'] == 20


def test_missing_rank_or_release_receipt_is_not_a_performance_sample():
    from vericcl.live.native import combine_schedule_results
    for records in ([sample(0, [1., 2.])], [sample(0, [1., 2.]), sample(0, [1., 2.])],
                    [sample(0, [1., 2.]), sample(1, [2., 3.]) | {'gpu_released': False}]):
        with pytest.raises(ValueError):
            combine_schedule_results(records, rank_count=2, version='v', contract='c', warmup=5, timed=2)
def test_explicit_control_bind_is_the_coordinator_endpoint(monkeypatch):
    from types import SimpleNamespace
    from vericcl.live.native import NativeScheduleDriver
    import vericcl.live.native as native
    monkeypatch.setattr(native.socket, 'getaddrinfo', lambda *args: [(2, 1, 6, '', ('192.0.2.1', 0))])
    driver = object.__new__(NativeScheduleDriver)
    driver.transport = SimpleNamespace(_is_local=lambda host: True)
    inputs = SimpleNamespace(request_config={'runtime': {'rank_map': [{'host': 'worker-a'}],
        'control': {'bind_address': 'worker-a', 'port': 24844}}})
    assert driver._control(inputs) == ('worker-a', 'worker-a', 24844)


def test_measurement_sequence_maps_receipts_back_to_launches(monkeypatch, tmp_path):
    import base64, json
    from types import SimpleNamespace
    from vericcl.execution.context import RunContext
    from vericcl.live.native import NativeScheduleDriver
    from vericcl.verification.online.acceptance import LaunchRequest
    context = RunContext.create(tmp_path, 'sequence', 'run')
    driver = object.__new__(NativeScheduleDriver)
    driver.context = context
    inputs = SimpleNamespace(rank_count=2, request_config={'runtime': {'environment': {}, 'rank_map': [{'host': 'h'}]}})
    old, new = SimpleNamespace(xml_sha256='a' * 64), SimpleNamespace(xml_sha256='b' * 64)
    requests = [LaunchRequest('launch-%d' % i, 1, 1, 1 + i // 2, side, 0, 2) for i, side in enumerate('ABBA')]
    items = [(old if r.side == 'A' else new, inputs, r) for r in requests]
    root = context.artifact('jobs/schedule-x'); root.mkdir(parents=True)
    captured = {}
    def launch(bundle, inputs, **kwargs):
        captured.update(kwargs)
        primary = [dict(rank=r, correct=True, gpu_completed=True, gpu_released=True, trace_enabled=False,
                        version_id='a' * 64, contract_digest='c', warmup_iterations=0, timed_iterations=1,
                        device_times_us=[1.], runtime_identity={'gpu_uuid': str(r)}) for r in range(2)]
        for index, (bundle_item, _, request) in enumerate(items):
            for rank in range(2):
                value = 10. if request.side == 'A' else 8.
                record = dict(rank=rank, index=index, label=request.launch_id, version_id=bundle_item.xml_sha256,
                              contract_digest='c', warmup_iterations=0, timed_iterations=2,
                              device_times_us=[value, value + rank], correct=True, gpu_completed=True, trace_enabled=False)
                path = root / 'collected' / 'h' / 'native' / ('rank-%d' % rank) / ('sequence-%d.json' % index)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(record))
        collection = dict(host='h', files={}, prepared=dict(binary_sha256='x', inventory={}))
        evidence = dict(error=None, gpu_released=True, cleanup_confirmed=True)
        return root, dict(version_id='a' * 64, contract_digest='c'), primary, [collection], evidence
    monkeypatch.setattr(driver, '_launch', launch)
    samples = driver.measure_sequence(items, lease=object())
    assert [s.launch_id for s in samples] == [r.launch_id for r in requests]
    assert [s.time_us for s in samples] == [10.5, 8.5, 8.5, 10.5]
    assert [label for label, *_ in captured['sequence']] == [r.launch_id for r in requests]
    assert captured['warmup'] == 0 and captured['timed'] == 1
