import pytest


def sample(rank, times):
    return dict(rank=rank, correct=True, gpu_completed=True, gpu_released=True,
                trace_enabled=False, device_times_us=times, version_id='v', contract_digest='c',
                warmup_iterations=5, timed_iterations=2, runtime_identity={'gpu_uuid': 'gpu-' + str(rank)})


def test_schedule_time_is_mean_of_per_invocation_slowest_rank():
    from vericcl.execution.native import combine_schedule_results
    result = combine_schedule_results([sample(0, [10., 20.]), sample(1, [20., 10.])],
                                      rank_count=2, version='v', contract='c', warmup=5, timed=2)
    assert result['time_us'] == 20


def test_missing_rank_or_release_receipt_is_not_a_performance_sample():
    from vericcl.execution.native import combine_schedule_results
    for records in ([sample(0, [1., 2.])], [sample(0, [1., 2.]), sample(0, [1., 2.])],
                    [sample(0, [1., 2.]), sample(1, [2., 3.]) | {'gpu_released': False}]):
        with pytest.raises(ValueError):
            combine_schedule_results(records, rank_count=2, version='v', contract='c', warmup=5, timed=2)
def test_explicit_control_bind_is_the_coordinator_endpoint(monkeypatch):
    from types import SimpleNamespace
    from vericcl.execution.native import NativeScheduleDriver
    import vericcl.execution.native as native
    monkeypatch.setattr(native.socket, 'getaddrinfo', lambda *args: [(2, 1, 6, '', ('192.0.2.1', 0))])
    driver = object.__new__(NativeScheduleDriver)
    driver.transport = SimpleNamespace(_is_local=lambda host: True)
    inputs = SimpleNamespace(request_config={'runtime': {'rank_map': [{'host': 'worker-a'}],
        'control': {'bind_address': 'worker-a', 'port': 24844}}})
    assert driver._control(inputs) == ('worker-a', 'worker-a', 24844)
