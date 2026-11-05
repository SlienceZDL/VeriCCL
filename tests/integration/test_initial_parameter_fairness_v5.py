import json

import pytest

from tests.unit.input.v5_support import request_documents, write_request
from vericcl.errors import SemanticError, ConstructionInfeasibleError
from vericcl.execution.budget import BudgetExhausted
from vericcl.execution.context import RunContext
from vericcl.input.loader import load_request
from vericcl.verification.online.model_publication import model_json
from vericcl.workflow_v5 import PrimaryWorkflow


def workflow_fixture(tmp_path, method, *, persist_model=True):
    documents = request_documents('broadcast', 3)
    documents['atom']['synthesis_method'] = method
    documents['sketch']['solver'].update(max_channels=2, per_model_timeout_s=1)
    documents['sketch']['budgets'].update(max_initial_synthesis_s=6, max_tuning_rounds=0,
                                        cleanup_reserve_s=.1)
    documents['topology']['local_costs'] = [dict(rank=r, copy=dict(alpha_us=.1, beta_us_per_byte=.001))
                                          for r in range(3)]
    inputs = load_request(write_request(tmp_path, documents))
    run = RunContext.create(tmp_path, 'initial-parameters', 'run')
    workflow = PrimaryWorkflow(inputs, run)
    if persist_model:
        run.write_json('models/' + workflow.model.version + '.json', model_json(workflow.model))
    return workflow


def install_stream(monkeypatch, workflow, generate):
    from vericcl import workflow_v5
    monkeypatch.setattr(workflow_v5, 'plan_hierarchy', lambda inputs, *args: generate(inputs.solver.max_channels))
    monkeypatch.setattr(workflow_v5, 'construct_routes', lambda inputs, *args: inputs)
    monkeypatch.setattr(workflow_v5, 'optimize_fixed_routes', lambda inputs, *args: generate(inputs.solver.max_channels))
    monkeypatch.setattr(workflow_v5, 'lower_and_check', lambda candidate, lowering: candidate)
    saved = []
    monkeypatch.setattr(workflow, 'store', lambda candidate, inputs, origin: saved.append((inputs.solver.max_channels, candidate, origin)))
    return saved


@pytest.mark.parametrize('method', ['hierarchy', 'constructive_trees'])
def test_first_candidate_of_each_parameter_precedes_deeper_enumeration(tmp_path, monkeypatch, method):
    workflow = workflow_fixture(tmp_path, method)
    closed = []
    def generate(channel):
        try:
            yield (channel, 0)
            yield (channel, 1)
            yield (channel, 2)
        finally:
            closed.append(channel)
    saved = install_stream(monkeypatch, workflow, generate)
    workflow.initial_candidates()
    assert [row[1] for row in saved] == [(1, 0), (2, 0), (1, 1), (2, 1), (1, 2), (2, 2)]
    assert sorted(closed) == [1, 2]
    assert not workflow.context.active_jobs and workflow.budgets.cpu_in_use == 0


def test_parent_cancel_closes_other_already_started_parameter_streams(tmp_path, monkeypatch):
    workflow = workflow_fixture(tmp_path, 'hierarchy')
    closed = []
    def generate(channel):
        try:
            yield (channel, 0)
            if channel == 1:
                workflow.budgets.cancel()
                raise BudgetExhausted('parent cancelled during later candidate')
            yield (channel, 1)
        finally:
            closed.append(channel)
    saved = install_stream(monkeypatch, workflow, generate)
    with pytest.raises(BudgetExhausted):
        workflow.initial_candidates()
    assert [row[1] for row in saved] == [(1, 0), (2, 0)]
    assert sorted(closed) == [1, 2]
    assert not workflow.context.active_jobs and workflow.budgets.cpu_in_use == 0


@pytest.mark.parametrize('error', [SemanticError('invalid parameter branch'), BudgetExhausted('one model exhausted'),
                                 ConstructionInfeasibleError('no legal route in this parameter branch')])
def test_one_parameter_failure_retains_candidates_and_other_stream(tmp_path, monkeypatch, error):
    workflow = workflow_fixture(tmp_path, 'constructive_trees')
    closed = []
    def generate(channel):
        try:
            yield (channel, 0)
            if channel == 1:
                raise error
            yield (channel, 1)
        finally:
            closed.append(channel)
    saved = install_stream(monkeypatch, workflow, generate)
    workflow.initial_candidates()
    assert [row[1] for row in saved] == [(1, 0), (2, 0), (2, 1)]
    assert sorted(closed) == [1, 2]
    assert not workflow.context.active_jobs and workflow.budgets.cpu_in_use == 0


@pytest.mark.parametrize('method', ['hierarchy', 'constructive_trees'])
def test_real_public_search_reaches_both_channels_and_releases_model_work(tmp_path, method):
    workflow = workflow_fixture(tmp_path, method, persist_model=False)
    result = workflow.run()
    assert result.contract_met, result
    candidates = [json.loads(path.read_text()) for path in workflow.context.artifact('candidates').glob('*/candidate.json')]
    assert {row['effective_config']['sketch']['solver']['max_channels'] for row in candidates} == {1, 2}
    assert all(row['checks']['semantic']['status'] == 'valid' for row in candidates)
    assert workflow.budgets.cpu_in_use == 0 and not workflow.context.active_jobs
    assert result.stop_reason == 'maximum_rounds'


def test_unavailable_required_solver_is_not_swallowed_as_a_parameter_failure(tmp_path, monkeypatch):
    from vericcl.errors import SolverUnavailableError
    workflow = workflow_fixture(tmp_path, 'hierarchy')
    def unavailable(channel):
        raise SolverUnavailableError('required SCIP unavailable')
        yield
    install_stream(monkeypatch, workflow, unavailable)
    with pytest.raises(SolverUnavailableError):
        workflow.initial_candidates()
    assert not workflow.context.active_jobs and workflow.budgets.cpu_in_use == 0


def test_real_byte_bans_making_large_slice_unroutable_do_not_hide_smaller_partition(tmp_path):
    documents = request_documents('broadcast', 3)
    documents['atom'].update(synthesis_method='constructive_trees',
        forbidden_transfers=[[0, 0, 1, 0], [1, 0, 2, 0]])
    documents['sketch']['hyperparameters'].update(total_size_bytes=128,
        slice_size_bytes=64, reference_slice_size_bytes=64)
    documents['sketch']['solver'].update(max_channels=2, per_model_timeout_s=2,
        max_threads_per_model=1, cpu_budget=1)
    documents['sketch']['budgets'].update(max_initial_synthesis_s=10,
        max_tuning_rounds=0, cleanup_reserve_s=.1)
    documents['runtime']['limits'] = dict(steps_per_tb=1, send_tbs_per_channel=1, recv_tbs_per_channel=1)
    documents['topology']['local_costs'] = [dict(rank=r, copy=dict(alpha_us=.1, beta_us_per_byte=.001))
                                          for r in range(3)]
    run = RunContext.create(tmp_path, 'initial-parameters', 'real-byte-bans')
    workflow = PrimaryWorkflow(load_request(write_request(tmp_path, documents)), run)
    result = workflow.run()
    assert workflow.candidates
    assert all(inputs.solver.max_channels == 2 and inputs.hyperparameters.slice_size_bytes == 64
               for _, inputs, _ in workflow.candidates)
    assert all(bundle.checks['semantic'].status.value == 'valid' for bundle, _, _ in workflow.candidates)
    assert any('complete route-column space' in failure for failure in result.optional_failures)
    # This deliberately tiny TB limit checks reachability, not admission or gain.
    assert not result.contract_met and result.stop_reason == 'no_qualified_baseline'
    assert not workflow.context.active_jobs and workflow.budgets.cpu_in_use == 0
