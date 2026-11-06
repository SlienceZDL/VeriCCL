from dataclasses import replace
from fractions import Fraction

import pytest

from tests.unit.input.v5_support import request_documents, write_request
from vericcl.execution.context import RunContext
from vericcl.input.loader import load_request


def request(tmp_path, *, rounds=0, calibration=False, measured=False):
    documents = request_documents('broadcast', 2)
    documents['sketch']['solver'].update(max_channels=1, per_model_timeout_s=5)
    documents['sketch']['budgets'].update(max_initial_synthesis_s=15, max_tuning_rounds=rounds,
        max_online_validation_per_round_s=8, max_online_validation_total_s=16,
        max_repair_per_round_s=5, max_repair_total_s=10)
    documents['sketch']['calibration'].update(enabled=calibration, max_relative_fit_error=.05, job_timeout_s=2)
    documents['sketch'].setdefault('trace_and_measure', {}).update(enabled=measured, job_timeout_s=2, trace_capacity_bytes=1048576)
    documents['topology']['local_costs'] = [dict(rank=rank, copy=dict(alpha_us=.1, beta_us_per_byte=.001)) for rank in range(2)]
    inputs = load_request(write_request(tmp_path, documents))
    return inputs, RunContext.create(tmp_path, 'primary', 'run')


def test_primary_initial_synthesis_persists_checked_best_and_zero_round_stop(tmp_path):
    from vericcl.workflow_v5 import run_primary
    inputs, context = request(tmp_path)
    result = run_primary(inputs, context)
    assert result.contract_met and result.stop_reason == 'maximum_rounds'
    assert (context.artifact('best/accepted.json')).is_file()
    assert (context.artifact('checkpoints/latest.json')).is_file()
    assert result.model_coverage != 'unknown'


def test_same_xml_is_reassessed_after_first_calibration_before_baseline(tmp_path, monkeypatch):
    from vericcl import workflow_v5
    from vericcl.verification.online.campaign import CampaignResult
    from vericcl.verification.online.model_publication import ComponentUpdates, publish_model
    inputs, context = request(tmp_path, calibration=True)
    versions = []
    def calibrate(probes, model, **kwargs):
        services = {key: tuple(replace(value, source='measured', alpha_us=value.alpha_us * 2,
                                      beta_us_per_byte=value.beta_us_per_byte * 2, measured_at='2026-11-04T00:00:00Z')
                               for value in values) for key, values in model.services.items()}
        new = publish_model(model, ComponentUpdates(services, signatures={'service:' + key: 'fixture' for key in services}), initial=model)
        versions.extend((model.version, new.version))
        return CampaignResult(new, (), (), (), 'completed')
    monkeypatch.setattr(workflow_v5, 'run_calibration', calibrate)
    result = workflow_v5.run_primary(inputs, context)
    import json
    accepted = json.loads((context.artifact('best/accepted.json')).read_text())
    assert result.contract_met and accepted['model_version'] == versions[1] != versions[0]
    from vericcl.execution.events import read_events
    reevaluated = [item for item in read_events(context.artifact('events.jsonl')).events if item['kind'] == 'candidate_reassessed']
    assert reevaluated and all(item['data']['xml_unchanged'] for item in reevaluated)


def test_candidate_acceptance_requires_same_model_strict_gain_and_shared_gate(tmp_path):
    from tests.integration.test_lowering_v5 import candidate_fixture
    from vericcl.xml.lower import lower_and_check
    from vericcl.workflow_v5 import accept_candidate
    candidate, lowering = candidate_fixture(tmp_path, 'broadcast')
    old = lower_and_check(candidate, lowering)
    faster = replace(old, prediction=replace(old.prediction, t_pred_us=old.prediction.t_pred_us / 2,
        exact_completion_ratio=(old.prediction.exact_completion_ratio[0], old.prediction.exact_completion_ratio[1] * 2)))
    assert accept_candidate(old, faster, {}).accepted
    assert not accept_candidate(old, old, {}).accepted
    assert not accept_candidate(old, replace(faster, prediction=replace(faster.prediction, model_version='other')), {}).accepted
    assert not accept_candidate(old, faster, {'measured_required': True}).accepted


def test_tight_directed_cut_is_rechecked_across_all_repair_families(tmp_path):
    from tests.integration.test_lowering_v5 import candidate_fixture
    from vericcl.xml.lower import lower_and_check
    from vericcl.verification.global_bound import certify_global_cut, check_global_cut
    candidate, lowering = candidate_fixture(tmp_path, 'broadcast', inplace=False)
    model = replace(lowering.model, services={key: tuple(replace(piece, alpha_us=0) for piece in values)
                                             for key, values in lowering.model.services.items()})
    lowering = replace(lowering, model=model)
    bundle = lower_and_check(candidate, lowering)
    certificate = certify_global_cut(bundle, lowering)
    assert certificate and check_global_cut(certificate, bundle, lowering)
    changed = replace(lowering, model=replace(model, resources={key: value * 2 for key, value in model.resources.items()}))
    assert not check_global_cut(certificate, bundle, changed)


def test_finite_exhaustion_expands_and_stops_only_at_round_limit(tmp_path, monkeypatch):
    from vericcl import workflow_v5
    from vericcl.verification.bottlenecks import AnalysisBatch
    from vericcl.execution.events import read_events
    inputs, context = request(tmp_path, rounds=2)
    monkeypatch.setattr(workflow_v5, 'analyze', lambda *args: AnalysisBatch('candidates_exhausted', (), (), {}, {}))
    result = workflow_v5.run_primary(inputs, context)
    assert result.contract_met and result.stop_reason == 'maximum_rounds'
    assert any(event['kind'] == 'search_expanded' for event in read_events(context.artifact('events.jsonl')).events)


def test_failed_local_strategy_does_not_cancel_other_actions(tmp_path, monkeypatch):
    from vericcl import workflow_v5
    from vericcl.errors import SemanticError
    from vericcl.verification.bottlenecks import AnalysisBatch, StrategyRecord
    inputs, context = request(tmp_path, rounds=1)
    actions = (StrategyRecord('path', ('one',)), StrategyRecord('path', ('two',)))
    monkeypatch.setattr(workflow_v5, 'analyze', lambda *args: AnalysisBatch('opportunities', (), actions, {}, {}))
    attempted = []
    def broken(bundle, domain, strategy, repair):
        attempted.append(strategy.operation_ids)
        raise SemanticError('injected invalid local strategy')
    monkeypatch.setattr(workflow_v5, 'solve_repair', broken)
    result = workflow_v5.run_primary(inputs, context)
    assert result.contract_met and attempted == [('one',), ('two',)]


def test_primary_reaches_global_proof_and_time_limit_separately(tmp_path, monkeypatch):
    from vericcl import workflow_v5
    from vericcl.input.schema import _freeze
    inputs, context = request(tmp_path, rounds=1)
    config = workflow_v5.plain(inputs.request_config)
    for edge in config['topology']['directed_links']:
        edge['alpha_us'] = 0
    # Re-load to keep the legacy synthesis topology and new cost model aligned.
    inputs = load_request(write_request(tmp_path, config))
    result = workflow_v5.run_primary(inputs, context)
    assert result.contract_met and result.stop_reason == 'proven_no_improvement'
    assert result.proof_scope['kind'] == 'directed_information_cut_v1'


def test_time_limit_keeps_already_qualified_incumbent(tmp_path, monkeypatch):
    from vericcl import workflow_v5
    from vericcl.execution.budget import BudgetExhausted
    inputs, context = request(tmp_path, rounds=1)
    def expired(self):
        raise BudgetExhausted('injected exhausted cumulative budget')
    monkeypatch.setattr(workflow_v5.PrimaryWorkflow, 'tuning', expired)
    result = workflow_v5.run_primary(inputs, context)
    assert result.contract_met and result.stop_reason == 'time_limit'
    assert (context.artifact('best/accepted.json')).is_file()


def test_analysis_backend_failure_does_not_report_successful_round_end(tmp_path, monkeypatch):
    from vericcl import workflow_v5
    from vericcl.verification.bottlenecks import AnalysisBatch
    inputs, context = request(tmp_path, rounds=1)
    monkeypatch.setattr(workflow_v5, 'analyze', lambda *args: AnalysisBatch('analysis_error', (), (), {}, {'message': 'injected unavailable backend'}))
    result = workflow_v5.run_primary(inputs, context)
    assert not result.contract_met and result.failure_category == 'offline' and result.stop_reason == 'analysis_error'


def test_no_initial_candidate_preserves_the_failure_reason(tmp_path, monkeypatch):
    from vericcl import workflow_v5
    from vericcl.errors import SemanticError
    inputs, context = request(tmp_path)

    def broken(*args, **kwargs):
        raise SemanticError('injected missing receiver input dependency')
        yield  # Make the failure occur while consuming the candidate stream.

    monkeypatch.setattr(workflow_v5, 'plan_hierarchy', broken)
    result = workflow_v5.run_primary(inputs, context)
    assert not result.contract_met and result.stop_reason == 'no_initial_candidate'
    assert any('missing receiver input dependency' in reason for reason in result.optional_failures)


def test_real_owned_scip_repair_adopts_lower_complete_time(tmp_path, monkeypatch):
    from tests.scip.test_joint_repair_v5 import repair_fixture
    from vericcl import workflow_v5
    from vericcl.verification.bottlenecks import AnalysisBatch, StrategyRecord
    import json
    bundle, repair = repair_fixture(tmp_path)
    inputs = repair.lowering.inputs
    context = RunContext.create(tmp_path, 'real-repair', 'run')
    def initial(self):
        with self.stage('initial'):
            self.store(bundle, inputs, 'injected_initial_fixture')
    monkeypatch.setattr(workflow_v5.PrimaryWorkflow, 'initial_candidates', initial)
    ids = ('slice0-to1', 'slice1-to1')
    strategy = StrategyRecord('allocation', ids, paths=tuple((key, (0, 2, 1)) for key in ids))
    monkeypatch.setattr(workflow_v5, 'analyze', lambda *args: AnalysisBatch('opportunities', (), (strategy,), {}, {}))
    result = workflow_v5.run_primary(workflow_v5.PrimaryRequest(inputs, cost_model=repair.lowering.model), context)
    accepted = json.loads((context.artifact('best/accepted.json')).read_text())
    assert result.contract_met and accepted['t_pred_us'] < bundle.prediction.t_pred_us
    assert list((context.artifact('jobs')).glob('scip-*/identity.json'))
