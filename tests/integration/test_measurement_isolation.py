from dataclasses import replace

from tests.unit.online.test_probe_jobs import probe_context


def test_independent_failed_jobs_preserve_successful_link_results(tmp_path):
    from vericcl.verification.online.probe_jobs import plan_probes, execute_campaign, ProbeResult
    context = probe_context(tmp_path, channels=1, ranks=2)
    jobs = list(plan_probes(context))[:6]
    shared = replace(jobs[0], job_id='failed-shared', category='shared')
    inter = replace(jobs[0], job_id='failed-inter', host_scope='inter')
    class ExplicitTestDriver:
        driver_kind = 'test_simulation'
        def execute(self, job, context):
            if job.job_id.startswith('failed'):
                raise RuntimeError('injected independent job failure')
            return ProbeResult(job.job_id, 'completed', self.driver_kind, samples=({'duration_us': 2., 'correct': True},))
    results = list(execute_campaign((shared, jobs[0], inter, jobs[1]), context, ExplicitTestDriver()))
    assert [result.status for result in results] == ['failed', 'completed', 'failed', 'completed']
    assert all(result.driver_kind == 'test_simulation' for result in results)


def test_remote_collection_failure_does_not_erase_prior_results(tmp_path):
    from vericcl.verification.online.probe_jobs import plan_probes, execute_campaign, ProbeResult
    context = probe_context(tmp_path, channels=1, ranks=2)
    jobs = list(plan_probes(context))[:3]
    class ExplicitTestDriver:
        driver_kind = 'test_simulation'
        calls = 0
        def execute(self, job, context):
            self.calls += 1
            if self.calls == 2:
                raise OSError('injected remote collection failure')
            return ProbeResult(job.job_id, 'completed', self.driver_kind, samples=({'duration_us': 2., 'correct': True},))
    results = list(execute_campaign(jobs, context, ExplicitTestDriver()))
    assert [result.status for result in results] == ['completed', 'failed', 'completed']
    assert results[0].samples and results[2].samples


def test_remote_paths_are_confined_after_symlink_resolution(tmp_path):
    import pytest
    from vericcl.verification.online.remote import checked_job_path
    root = tmp_path / 'owned'; root.mkdir()
    outside = tmp_path / 'outside'; outside.mkdir()
    (root / 'escape').symlink_to(outside, target_is_directory=True)
    assert checked_job_path(root, 'jobs/test').is_relative_to(root)
    for unsafe in ('../outside', '/tmp/escape', 'escape/file'):
        with pytest.raises(ValueError):
            checked_job_path(root, unsafe)


def test_remeasurement_keeps_distinct_immutable_attempts(tmp_path):
    from vericcl.execution.context import RunContext
    from vericcl.verification.online.probe_jobs import plan_probes, execute_probe, ProbeResult
    context = probe_context(tmp_path, channels=1, ranks=2)
    context.run_context = RunContext.create(tmp_path, 'repeat-probe', 'run')
    job = next(iter(plan_probes(context)))
    class Simulation:
        driver_kind = 'test_simulation'
        def execute(self, job, context):
            return ProbeResult(job.job_id, 'completed', self.driver_kind)
    first = execute_probe(job, context, Simulation())
    second = execute_probe(job, context, Simulation())
    assert first.provenance['execution_id'] != second.provenance['execution_id']
    directory = context.run_context.artifact('models/probes/' + job.job_id)
    assert len(list(directory.glob('attempt-*/job.json'))) == 2
    assert len(list(directory.glob('attempt-*/result.json'))) == 2
