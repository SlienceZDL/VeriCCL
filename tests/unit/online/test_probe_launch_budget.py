from dataclasses import replace
from types import SimpleNamespace

import pytest

from tests.unit.online.test_probe_jobs import probe_context
from vericcl.execution.context import RunContext
from vericcl.verification.online import remote
from vericcl.verification.online.probe_jobs import plan_probes


@pytest.mark.parametrize('budget,expected_launches,expected_status', [(30, 0, 'timeout'), (100, 4, 'unstable')])
def test_probe_launch_admission_preserves_remaining_margin(tmp_path, monkeypatch, budget, expected_launches, expected_status):
    context = probe_context(tmp_path, channels=1, ranks=2)
    context.run_context = RunContext.create(tmp_path, 'probe-budget', 'run')
    context.lease = SimpleNamespace(check=lambda: None, remaining_seconds=lambda: 100)
    job = replace(next(job for job in plan_probes(context) if job.category == 'copy'),
                  timeout_s=budget, launches=4, max_rounds=1)
    clock = [0.0]
    monkeypatch.setattr(remote, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    driver = remote.NativeProbeDriver(project_root=tmp_path)
    calls = []

    def launch(job, artifact, context, round_index, launch_index, timeout):
        calls.append(timeout)
        clock[0] += 8
        sample = {'round': round_index, 'launch': launch_index,
                  'flow_us': {flow.flow_id: float(launch_index) for flow in job.flows}}
        return sample, {'cleanup_confirmed': True, 'gpu_released': True}

    monkeypatch.setattr(driver, '_launch', launch)
    result = driver.execute(job, context)
    assert result.status == expected_status
    assert len(calls) == expected_launches
    assert result.cleanup_confirmed and result.gpu_released
    assert not driver.quarantined
    if expected_status == 'timeout':
        assert 'before starting' in result.message


def test_successful_launch_estimate_survives_the_next_probe_job(tmp_path, monkeypatch):
    context = probe_context(tmp_path, channels=1, ranks=2)
    context.run_context = RunContext.create(tmp_path, 'probe-cross-job', 'run')
    remaining = [100.0]
    context.lease = SimpleNamespace(check=lambda: None, remaining_seconds=lambda: remaining[0])
    job = replace(next(job for job in plan_probes(context) if job.category == 'copy'),
                  timeout_s=600, launches=2, max_rounds=1)
    clock, calls = [0.0], []
    monkeypatch.setattr(remote, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    driver = remote.NativeProbeDriver(project_root=tmp_path)

    def launch(job, artifact, context, round_index, launch_index, timeout):
        calls.append(timeout)
        clock[0] += 17
        return {'round': round_index, 'launch': launch_index,
                'flow_us': {flow.flow_id: float(launch_index) for flow in job.flows}}, {
                    'cleanup_confirmed': True, 'gpu_released': True}

    monkeypatch.setattr(driver, '_launch', launch)
    assert driver.execute(job, context).status == 'unstable'
    assert len(calls) == 2
    remaining[0] = 19.2
    result = driver.execute(replace(job, job_id='next-copy-job'), context)
    assert result.status == 'timeout'
    assert len(calls) == 2
    assert result.provenance['next_launch_admission_s'] == 35.5
    assert remaining[0] == 19.2


def test_failed_launch_does_not_prevent_independent_following_probe(tmp_path, monkeypatch):
    context = probe_context(tmp_path, channels=1, ranks=2)
    context.run_context = RunContext.create(tmp_path, 'probe-failure-isolation', 'run')
    context.lease = SimpleNamespace(check=lambda: None, remaining_seconds=lambda: 1000)
    job = replace(next(job for job in plan_probes(context) if job.category == 'copy'),
                  timeout_s=600, launches=2, max_rounds=1)
    clock, calls = [0.0], []
    monkeypatch.setattr(remote, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    driver = remote.NativeProbeDriver(project_root=tmp_path)

    def launch(job, artifact, context, round_index, launch_index, timeout):
        calls.append(job.job_id)
        evidence = dict(cleanup_confirmed=True, gpu_released=True, error='injected failed launch')
        if len(calls) == 1:
            clock[0] += 600
            return None, evidence
        clock[0] += 8
        return {'round': round_index, 'launch': launch_index,
                'flow_us': {flow.flow_id: float(launch_index) for flow in job.flows}}, evidence

    monkeypatch.setattr(driver, '_launch', launch)
    assert driver.execute(job, context).status == 'failed'
    result = driver.execute(replace(job, job_id='independent-copy-job'), context)
    assert result.status == 'unstable'
    assert len(calls) == 3


@pytest.mark.parametrize('cleanup,released', [(False, True), (True, False)])
def test_valid_sample_with_uncertain_cleanup_cannot_start_another_launch(tmp_path, monkeypatch, cleanup, released):
    context = probe_context(tmp_path, channels=1, ranks=2)
    context.run_context = RunContext.create(tmp_path, 'probe-cleanup-stop', 'run')
    context.lease = SimpleNamespace(check=lambda: None, remaining_seconds=lambda: 1000)
    job = replace(next(job for job in plan_probes(context) if job.category == 'copy'),
                  timeout_s=600, launches=2, max_rounds=1)
    driver = remote.NativeProbeDriver(project_root=tmp_path)
    calls = []

    def launch(*args):
        calls.append(1)
        return {'flow_us': {flow.flow_id: float(len(calls)) for flow in job.flows}}, {
            'cleanup_confirmed': cleanup, 'gpu_released': released, 'error': None}

    monkeypatch.setattr(driver, '_launch', launch)
    result = driver.execute(job, context)
    assert len(calls) == 1 and result.status == 'failed'
    assert result.cleanup_confirmed is cleanup and result.gpu_released is released
    assert not result.samples and 'cleanup' in result.message
