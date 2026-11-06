"""The public lifecycle can repair a format-invalid but semantic-valid baseline."""
import json

import pytest

from tests.unit.input.v5_support import request_documents, write_request
from vericcl.execution.budget import StandaloneLease
from vericcl.execution.context import RunContext
from vericcl.execution.events import read_events
from vericcl.input.loader import load_request
from vericcl.performance.initial import initial_cost_model
from vericcl.topology import load_topology
from vericcl.workflow_v5 import PrimaryRequest, run_primary
from vericcl.xml.lower import LoweringContext, lower_and_check
from vericcl.xml.v5_model import NetworkCandidate, NetworkOperation


def capacity_input(tmp_path, method, rounds):
    documents = request_documents('broadcast', 3)
    documents['topology']['directed_links'] = [edge for edge in documents['topology']['directed_links']
        if (edge['src'], edge['dst']) in {(0, 1), (1, 2)}]
    documents['topology']['local_costs'] = [dict(rank=r, copy=dict(alpha_us=.1, beta_us_per_byte=.001))
                                         for r in range(3)]
    documents['atom']['synthesis_method'] = method
    documents['sketch']['solver'].update(max_channels=1, per_model_timeout_s=3)
    documents['sketch']['budgets'].update(max_tuning_rounds=rounds, max_run_wall_s=30,
        max_repair_per_round_s=10, max_repair_total_s=20, cleanup_reserve_s=.1)
    documents['runtime']['limits'] = {'steps_per_tb': 3}
    inputs = load_request(write_request(tmp_path, documents))
    operations = []
    for s in range(4):
        first, second = 'slice-%d-first' % s, 'slice-%d-second' % s
        operations.extend((
            NetworkOperation(first, 'send', 0, 1, 0, 0, frozenset({s}), frozenset({s}),
                (0, 1, 2), frozenset(), frozenset(), float(s)),
            NetworkOperation(second, 'send', 1, 2, 0, 0, frozenset({s}), frozenset({s}),
                (0, 1, 2), frozenset({first}), frozenset({first}), float(s)+1),
        ))
    source = NetworkCandidate(3, 4, 1024, tuple(operations), 'capacity-recovery-fixture')
    lowering = LoweringContext(inputs, load_topology(inputs), initial_cost_model(inputs), StandaloneLease.after(10))
    invalid = lower_and_check(source, lowering)
    assert not invalid.can_execute
    assert invalid.checks['semantic'].status.value == 'valid'
    assert invalid.capacity_report and any(v.code == 'steps_per_tb' for v in invalid.capacity_report.violations)
    return inputs, invalid


@pytest.mark.parametrize('method', ('hierarchy', 'constructive_trees'))
def test_public_verify_tune_recovers_capacity_before_first_adoption(tmp_path, method):
    inputs, invalid = capacity_input(tmp_path, method, 1)
    run = RunContext.create(tmp_path, 'capacity-recovery', 'one-round')
    result = run_primary(PrimaryRequest(inputs, purpose='verify', xml_text=invalid.xml_text,
        sidecar=invalid.sidecar, tune=True), run)
    assert result.contract_met, result
    accepted = json.loads((run.artifact('best/accepted.json')).read_text())
    record = accepted['candidate']
    assert record['effective_config']['sketch']['hyperparameters']['slice_size_bytes'] > 1024
    assert record['effective_config']['sketch']['hyperparameters']['total_size_bytes'] == 4096
    latest = json.loads((run.artifact('checkpoints/latest.json')).read_text())
    checkpoint = json.loads((run.root / latest['state']).read_text())
    assert checkpoint['budgets']['started_rounds'] == 1
    assert checkpoint['budgets']['consumed']['repair'] > 0
    assert checkpoint['budgets']['elapsed_wall_s'] < 30
    events = read_events(run.artifact('events.jsonl')).events
    assert any(event['kind'] == 'capacity_recovery_finished' and event['data']['recovered'] for event in events)
    assert not run.active_jobs


def test_capacity_recovery_respects_zero_tuning_rounds(tmp_path):
    inputs, invalid = capacity_input(tmp_path, 'constructive_trees', 0)
    run = RunContext.create(tmp_path, 'capacity-recovery', 'disabled')
    result = run_primary(PrimaryRequest(inputs, purpose='verify', xml_text=invalid.xml_text,
        sidecar=invalid.sidecar, tune=True), run)
    assert not result.contract_met
    assert not (run.artifact('best/accepted.json')).exists()
    assert not any(event['kind'] == 'capacity_recovery_started'
                   for event in read_events(run.artifact('events.jsonl')).events)


def test_capacity_recovery_does_not_accept_an_invalid_xml_identity(tmp_path):
    inputs, invalid = capacity_input(tmp_path, 'constructive_trees', 1)
    run = RunContext.create(tmp_path, 'capacity-recovery', 'bad-xml')
    result = run_primary(PrimaryRequest(inputs, purpose='verify', xml_text=invalid.xml_text + '\n',
        sidecar=invalid.sidecar, tune=True), run)
    assert not result.contract_met
    assert not (run.artifact('best/accepted.json')).exists()
    assert not any(event['kind'] == 'capacity_recovery_started'
                   for event in read_events(run.artifact('events.jsonl')).events)


def recovery_workflow(tmp_path):
    from vericcl.workflow_v5 import PrimaryWorkflow
    from vericcl.verification.online.model_publication import model_json
    inputs, invalid = capacity_input(tmp_path, 'constructive_trees', 1)
    run = RunContext.create(tmp_path, 'capacity-recovery', 'stream-boundary')
    workflow = PrimaryWorkflow(PrimaryRequest(inputs, purpose='verify', tune=True), run)
    run.write_json('models/' + workflow.model.version + '.json', model_json(workflow.model))
    workflow.store(invalid, inputs, 'test_capacity_input')
    return workflow, inputs, invalid


def test_capacity_recovery_returns_to_unfinished_branch_after_first_seed(tmp_path, monkeypatch):
    from vericcl import workflow_v5
    from vericcl.tuning.resynthesis import ResynthesisRequest, ResynthesisCandidate
    workflow, inputs, invalid = recovery_workflow(tmp_path)
    original = workflow_v5.resynthesize_branch
    consumed = []

    def two_candidates(incumbent, branch, context):
        stream = original(incumbent, branch, context)
        try:
            valid = next(stream)
            # Two already validated bundles isolate candidate-stream control:
            # the first exceeds a concrete step limit, the next is executable.
            consumed.append('invalid')
            yield ResynthesisCandidate(incumbent, invalid, context.lowering, branch)
            consumed.append('valid')
            yield valid
        finally:
            stream.close()

    branch = ResynthesisRequest('partition', 'test one recovery branch', slice_bytes=2048, channels=1)
    monkeypatch.setattr(workflow, '_capacity_recovery_requests', lambda *_: iter((branch,)))
    monkeypatch.setattr(workflow_v5, 'resynthesize_branch', two_candidates)
    assert workflow.recover_capacity_baseline()
    assert consumed == ['invalid', 'valid']
    assert not workflow.expanded


def test_capacity_recovery_does_not_mark_expired_empty_stream_complete(tmp_path, monkeypatch):
    from vericcl import workflow_v5
    from vericcl.execution.budget import BudgetExhausted
    from vericcl.tuning.resynthesis import ResynthesisRequest
    workflow, _, _ = recovery_workflow(tmp_path)
    branch = ResynthesisRequest('partition', 'test interrupted branch', slice_bytes=2048, channels=1)

    def expired_stream(incumbent, request, context):
        context.lowering.lease.manager.cancel()
        return
        yield

    monkeypatch.setattr(workflow, '_capacity_recovery_requests', lambda *_: iter((branch,)))
    monkeypatch.setattr(workflow_v5, 'resynthesize_branch', expired_stream)
    with pytest.raises(BudgetExhausted):
        workflow.recover_capacity_baseline()
    assert not workflow.expanded
    latest = json.loads((workflow.context.root / 'checkpoints/latest.json').read_text())
    state = json.loads((workflow.context.root / latest['state']).read_text())
    assert state['search']['expanded'] == []
