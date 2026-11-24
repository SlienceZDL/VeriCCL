"""Batched measurement: session drivers measure many jobs in one call."""
from dataclasses import replace

from tests.unit.online.test_probe_jobs import probe_context
from tests.unit.online.test_component_measurements import mathematical_native_contract
from vericcl.execution.context import RunContext
from vericcl.performance.initial import initial_cost_model
from vericcl.verification.online import campaign
from vericcl.verification.online.probe_jobs import execute_jobs, plan_probes


class SessionDriver:
    """Native-contract test results with a session entry point (as the
    campaign tests use mathematical_native_contract for physical costs)."""
    driver_kind = 'native_msccl'

    def __init__(self):
        self.batches, self.singles = [], []

    def execute(self, job, context):
        self.singles.append(job.job_id)
        return mathematical_native_contract(job)

    def execute_batch(self, jobs, context):
        self.batches.append([job.job_id for job in jobs])
        return {job.job_id: mathematical_native_contract(job) for job in jobs}


def test_execute_jobs_uses_one_session_and_records_every_result(tmp_path):
    context = probe_context(tmp_path, channels=1, ranks=2)
    context.run_context = RunContext.create(tmp_path, 'batch', 'run')
    jobs = [job for job in plan_probes(context) if job.category == 'link'][:4]
    driver = SessionDriver()
    pairs = execute_jobs(jobs, context, driver)
    assert driver.batches == [[job.job_id for job in jobs]] and not driver.singles
    assert [job.job_id for job, _ in pairs] == [job.job_id for job in jobs]
    assert all(result.provenance.get('execution_id') for _, result in pairs)
    for job, _ in pairs:
        assert any(path.name == 'result.json' for path in context.run_context.root.rglob('result.json'))
    assert len(context.records) == len(jobs)


def test_single_job_or_driver_without_sessions_measures_job_by_job(tmp_path):
    context = probe_context(tmp_path, channels=1, ranks=2)
    jobs = [job for job in plan_probes(context) if job.category == 'link'][:2]
    driver = SessionDriver()
    execute_jobs(jobs[:1], context, driver)
    assert driver.singles == [jobs[0].job_id] and not driver.batches


def test_campaign_measures_all_groups_in_one_batch(tmp_path):
    context = probe_context(tmp_path, channels=1, ranks=2)
    context.run_context = RunContext.create(tmp_path, 'campaign', 'run')
    jobs = [job for job in plan_probes(context) if job.category == 'link' and job.flows[0].kind == 'send']
    driver = SessionDriver()
    result = campaign.run_calibration(context, initial_cost_model(context.inputs), jobs=jobs, driver=driver)
    assert len(driver.batches) == 1 and sorted(driver.batches[0]) == sorted(job.job_id for job in jobs)
    assert result.status == 'completed'
    assert any(piece.source == 'measured' for pieces in result.model.services.values() for piece in pieces)


def test_session_records_restore_the_job_identity_for_validation():
    from types import SimpleNamespace
    from vericcl.verification.online.remote import session_records
    job = SimpleNamespace(job_id='probe-abc')
    records = [{'job_id': 'j0003', 'rank': 0}, {'job_id': 'j0004', 'rank': 1}]
    mapped = session_records(job, 'j0003', records)
    assert mapped[0]['job_id'] == 'probe-abc' and mapped[1]['job_id'] == 'j0004'


def test_interpreter_passes_merge_into_one_record_per_step():
    from vericcl.verification.online.trace_format import RawStepTraceRecord, merge_passes
    def raw(step, reach, start, end):
        return RawStepTraceRecord(0, 3, step, 1, 2, 0, 7, reach, reach + 5, start, end, 1, 0)
    merged, passes = merge_passes([raw(0, 100, 110, 200), raw(0, 300, 310, 400), raw(1, 120, 130, 220), raw(1, 320, 330, 420)])
    assert passes == {(3, 0, 7): 2, (3, 1, 7): 2}
    first = merged[(3, 0, 7)]
    assert (first.tb_reach, first.transfer_start, first.transfer_end) == (100, 110, 400)


def test_rankfile_places_ranks_without_pinning_them_to_one_core():
    from vericcl.verification.online.remote import physical_cores, unbound_rankfile
    mapping = [{'rank': 0, 'host': 'a'}, {'rank': 1, 'host': 'b'}, {'rank': 2, 'host': 'a'}]
    text = unbound_rankfile(mapping, lambda host: host == 'a', {'a': 36, 'b': 18}).decode()
    assert text == 'rank 0=localhost slot=0-35\nrank 1=b slot=0-17\nrank 2=localhost slot=0-35\n'
    assert 'slot=0\n' not in unbound_rankfile(mapping, lambda host: False, {}).decode().replace('slot=0-0', '')
    assert physical_cores() >= 1
