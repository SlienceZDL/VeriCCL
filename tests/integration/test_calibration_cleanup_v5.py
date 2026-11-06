"""Cleanup uncertainty stops launches while clean measurement failures stay isolated."""
from dataclasses import replace
import json

import pytest

from tests.unit.online.test_probe_jobs import probe_context
from tests.unit.online.test_component_measurements import mathematical_native_contract
from vericcl.execution.context import RunContext
from vericcl.performance.initial import initial_cost_model
from vericcl.verification.online import campaign
from vericcl.verification.online.model_publication import ComponentUpdates
from vericcl.verification.online.probe_jobs import plan_probes


@pytest.mark.parametrize('cleanup,released,active', [(False, True, False), (True, False, False), (True, True, True)])
@pytest.mark.parametrize('followup', [None, 'additional_payloads', 'confirmation_payloads'])
def test_campaign_stops_after_unresolved_primary_or_followup(tmp_path, monkeypatch, cleanup, released, active, followup):
    context = probe_context(tmp_path, channels=1, ranks=2)
    context.run_context = RunContext.create(tmp_path, 'campaign', 'run')
    base = next(job for job in plan_probes(context) if job.category == 'copy')
    jobs = [replace(base, job_id='p' + str(size), payload_bytes=size) for size in (4096, 8192, 16384)]
    calls = []
    failing_call = 1 if followup is None else 2

    def measure(job, *_):
        calls.append(job.payload_bytes)
        result = mathematical_native_contract(job)
        if len(calls) == failing_call:
            if active:
                context.run_context.active_jobs.add('unreconciled-launcher')
            return replace(result, cleanup_confirmed=cleanup, gpu_released=released)
        return result

    monkeypatch.setattr(campaign, 'execute_probe', measure)
    if followup:
        jobs = jobs[:1]
        monkeypatch.setattr(campaign, 'fit_components', lambda *a, **k:
            ComponentUpdates(issues=({'reason': 'fixture_followup', followup: [8192, 16384]},)))
    result = campaign.run_calibration(context, initial_cost_model(context.inputs), jobs=jobs)
    assert len(calls) == failing_call
    assert result.status == 'cleanup_unconfirmed'
    assert len(result.observations) + len(result.confirmations) == failing_call
    state = json.loads(context.run_context.artifact('reports/calibration-resource-state.json').read_text())
    assert not state['safe_to_continue']
    assert state['active_jobs'] == (['unreconciled-launcher'] if active else [])


def test_cleanup_stop_preserves_a_previously_published_link_component(tmp_path, monkeypatch):
    context = probe_context(tmp_path, channels=1, ranks=2)
    jobs = [job for job in plan_probes(context) if job.category == 'link' and job.flows[0].kind == 'send']
    calls = []

    def measure(job, *_):
        calls.append(job)
        result = mathematical_native_contract(job)
        return replace(result, cleanup_confirmed=False) if job.flows[0].src_rank == 1 else result

    monkeypatch.setattr(campaign, 'execute_probe', measure)
    result = campaign.run_calibration(context, initial_cost_model(context.inputs), jobs=jobs)
    assert result.status == 'cleanup_unconfirmed' and len(calls) == 4
    assert all(next(piece for piece in result.model.services['link:0:1']
                    if piece.covers(job.payload_bytes)).source == 'measured' for job in calls[:3])


@pytest.mark.parametrize('purpose', ['solve', 'calibrate'])
def test_primary_workflow_stops_before_baseline_when_cleanup_is_unknown(tmp_path, monkeypatch, purpose):
    from tests.integration.test_primary_lifecycle_v5 import request
    from vericcl import workflow_v5
    inputs, context = request(tmp_path, calibration=True)
    monkeypatch.setattr(workflow_v5, 'run_calibration', lambda probes, model, **kwargs:
        campaign.CampaignResult(model, (), (), ({'reason': 'cleanup_unconfirmed'},), 'cleanup_unconfirmed'))
    monkeypatch.setattr(workflow_v5.PrimaryWorkflow, 'baseline', lambda *_: pytest.fail('no new baseline work'))
    result = workflow_v5.run_primary(workflow_v5.PrimaryRequest(inputs, purpose=purpose), context)
    assert not result.contract_met and result.stop_reason == 'cleanup_unconfirmed'
    assert result.failure_category == 'online'
