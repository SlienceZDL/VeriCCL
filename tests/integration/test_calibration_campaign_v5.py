from dataclasses import replace

from tests.unit.online.test_probe_jobs import probe_context
from tests.unit.online.test_component_measurements import mathematical_native_contract
from vericcl.performance.initial import initial_cost_model
from vericcl.verification.online.probe_jobs import plan_probes


def test_incremental_campaign_publishes_intra_before_independent_failed_jobs(tmp_path, monkeypatch):
    from vericcl.verification.online import campaign
    context = probe_context(tmp_path, channels=1, ranks=2)
    jobs = [job for job in plan_probes(context) if job.category == 'link' and job.flows[0].kind == 'send'][:3]
    inter = [replace(job, job_id='inter-' + job.job_id, host_scope='inter') for job in jobs]
    shared = [replace(job, job_id='shared-' + job.job_id, category='shared') for job in jobs]
    calls = []
    def measure(job, *_):
        calls.append((job.category, job.host_scope))
        result = mathematical_native_contract(job)
        return replace(result, status='failed') if job.category == 'shared' or job.host_scope == 'inter' else result
    monkeypatch.setattr(campaign, 'execute_probe', measure)
    result = campaign.run_calibration(context, initial_cost_model(context.inputs), jobs=shared + inter + jobs)
    assert calls[:3] == [('link', 'intra')] * 3
    assert next(piece for piece in result.model.services['link:0:1'] if piece.covers(1024)).source == 'measured'
    assert len(result.observations) == 9 and len(result.issues) == 6


def test_piecewise_campaign_executes_independent_confirmation_payloads(tmp_path, monkeypatch):
    from vericcl.verification.online import campaign
    context = probe_context(tmp_path, channels=1, ranks=2)
    base = next(job for job in plan_probes(context) if job.category == 'copy' and job.rank_ids == (0,))
    jobs = [replace(base, job_id='piece-' + str(size), payload_bytes=size) for size in (100, 200, 300, 1000, 2000, 3000)]
    count = [0]
    def measure(job, *_):
        count[0] += 1
        result = mathematical_native_contract(job)
        duration = 1 + .01 * job.payload_bytes if job.payload_bytes <= 300 else 8 + .04 * job.payload_bytes
        return replace(result, samples=tuple(sample | {'flow_us': {'copy:0': duration}} for sample in result.samples),
            provenance=result.provenance | {'measurement_id': 'fixture-measurement-' + str(count[0])})
    monkeypatch.setattr(campaign, 'execute_probe', measure)
    result = campaign.run_calibration(context, initial_cost_model(context.inputs), jobs=jobs)
    assert count[0] == 12 and len(result.confirmations) == 6
    assert len(result.model.services['copy:0']) == 2


def test_link_component_is_published_before_local_copy_can_exhaust_parent_budget(tmp_path, monkeypatch):
    from vericcl.execution.budget import StandaloneLease
    from vericcl.verification.online import campaign
    context = probe_context(tmp_path, channels=1, ranks=2)
    planned = list(plan_probes(context))
    link = [job for job in planned if job.category == 'link'
            and job.flows[0].kind == 'send' and job.flows[0].src_rank == 0][:3]
    copy = [job for job in planned if job.category == 'copy' and job.rank_ids == (0,)][:3]
    clock, calls = [0.], []
    context.lease = StandaloneLease.after(6, clock=lambda: clock[0])
    def measure(job, *_):
        calls.append(job.category)
        clock[0] += 1 if job.category == 'link' else 3
        result = mathematical_native_contract(job)
        return replace(result, status='unstable') if job.category == 'copy' else result
    monkeypatch.setattr(campaign, 'execute_probe', measure)
    result = campaign.run_calibration(context, initial_cost_model(context.inputs), jobs=link + copy)
    assert calls[:3] == ['link'] * 3
    assert result.status == 'budget_exhausted'
    assert any(piece.source == 'measured' for piece in result.model.services['link:0:1'])
    assert clock[0] == 6 and len(result.observations) == 4


def test_parallel_channel_groups_retain_planner_order_instead_of_job_hash_order(tmp_path, monkeypatch):
    from vericcl.verification.online import campaign
    context = probe_context(tmp_path, channels=2, ranks=2)
    jobs = [replace(job, job_id=('zz-single-' if len(job.flows) == 1 else 'aa-parallel-') + job.job_id)
            for job in plan_probes(context) if job.category == 'link'
            and job.flows[0].kind == 'send' and job.flows[0].src_rank == 0]
    calls = []
    def measure(job, *_):
        calls.append(len(job.flows))
        return mathematical_native_contract(job)
    monkeypatch.setattr(campaign, 'execute_probe', measure)
    result = campaign.run_calibration(context, initial_cost_model(context.inputs), jobs=jobs)
    assert calls == [1, 1, 1, 2, 2, 2]
    assert result.status == 'completed' and not result.issues
