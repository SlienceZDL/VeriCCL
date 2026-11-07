"""Regression tests for the overhead optimizations and plan-conformance fixes."""
from contextlib import contextmanager
import json
import random
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from vericcl.errors import InputValidationError, SemanticError


def test_solver_package_exports_resolve_lazily_with_the_same_names():
    import vericcl.solver as solver
    from vericcl.solver import ScipAdapter, SolverTemplate, routing_milp, solve_milp

    assert solve_milp.__module__ == 'vericcl.solver.milp'
    assert routing_milp.__name__ == 'vericcl.solver.routing_milp'
    assert ScipAdapter.__name__ == 'ScipAdapter' and SolverTemplate.__name__ == 'SolverTemplate'
    assert set(solver.__all__) <= set(dir(solver))
    with pytest.raises(AttributeError):
        solver.not_an_exported_solver_name


def test_owned_worker_entry_imports_only_the_solver_adapter():
    code = ('import sys, vericcl.solver.worker_scope, vericcl.solver.scip_api, vericcl.execution.budget; '
            'print(",".join(sorted(m for m in sys.modules if m.startswith("vericcl"))))')
    loaded = set(subprocess.run([sys.executable, '-c', code], check=True, capture_output=True,
                                text=True).stdout.strip().split(','))
    assert loaded <= {'vericcl', 'vericcl.errors', 'vericcl.execution', 'vericcl.execution.budget',
                      'vericcl.solver', 'vericcl.solver.scip_api', 'vericcl.solver.worker_scope'}


def _owned_solve(run, build):
    from vericcl.execution.budget import StandaloneLease
    from vericcl.solver.scip_api import ScipAdapter
    from vericcl.solver.worker_scope import isolated_solver_scope
    _, model = ScipAdapter.create_model('owned-reuse')
    value = build(model)
    lease = StandaloneLease.after(8)
    ScipAdapter.configure(model, lease=lease)
    with isolated_solver_scope(run, cleanup_deadline=time.monotonic() + 9):
        outcome = model.optimize()
    solved = model.getVal(value)
    model.freeProb()
    return outcome, solved


def test_identical_proven_owned_models_reuse_one_worker_result(tmp_path):
    from vericcl.execution.context import RunContext
    run = RunContext.create(tmp_path, 'solver', 'run')

    def build(model):
        x = model.addVar('x', vtype='INTEGER', lb=0)
        y = model.addVar('y', vtype='INTEGER', lb=0)
        model.addCons(x + 2 * y >= 7)
        model.setObjective(x + y)
        return x + y

    first, first_value = _owned_solve(run, build)
    second, second_value = _owned_solve(run, build)
    assert first.numerical_optimal and second.numerical_optimal
    assert first.primal == second.primal == first_value == second_value == 4
    assert len(list(run.artifact('jobs').glob('scip-*'))) == 1
    assert len(list(run.artifact('tmp').glob('scip-*/reused.json'))) == 1
    events = [json.loads(line)['kind'] for line in run.artifact('events.jsonl').read_text().splitlines()]
    assert events.count('solver_result_reused') == 1


def test_only_proven_results_are_reusable():
    from vericcl.solver.worker_scope import _reusable
    assert _reusable({'status': 'optimal', 'lexicographic_complete': True})
    assert _reusable({'status': 'infeasible', 'lexicographic_complete': False})
    assert not _reusable({'status': 'optimal', 'lexicographic_complete': False})
    assert not _reusable({'status': 'time_limit', 'lexicographic_complete': False})
    assert not _reusable({'status': 'feasible', 'lexicographic_complete': True})


def test_single_order_cycle_rule_matches_the_full_closure_check():
    from vericcl.verification.bottlenecks import _ancestors
    rng = random.Random(7)
    for _ in range(150):
        keys = ['op%d' % index for index in range(rng.randint(2, 8))]
        dependencies = {key: frozenset(keys[j] for j in range(index) if rng.random() < .35)
                        for index, key in enumerate(keys)}
        ancestors = _ancestors(dependencies)
        for first in keys:
            for second in keys:
                if first == second:
                    continue
                trial = dict(dependencies)
                trial[second] = trial[second] | {first}
                try:
                    _ancestors(trial)
                    acyclic = True
                except SemanticError:
                    acyclic = False
                assert acyclic == (second not in ancestors[first])


def test_checkpoints_share_identical_documents_and_still_verify_them(tmp_path):
    from tests.integration.test_lowering_v5 import candidate_fixture
    from vericcl.execution.context import RunContext
    from vericcl.workflow_state import build_budgets, restore_checkpoint, save_checkpoint
    from vericcl.xml.lower import lower_and_check
    candidate, lowering = candidate_fixture(tmp_path, 'broadcast')
    bundle = lower_and_check(candidate, lowering)
    run = RunContext.create(tmp_path, 'resume', 'r')
    budgets = build_budgets(lowering.inputs)
    first = save_checkpoint(run, lowering.inputs, budgets, lowering.model, bundle, phase='initial')
    second = save_checkpoint(run, lowering.inputs, budgets, lowering.model, bundle, phase='tuning')
    assert first['files'] == second['files'] and first['model'] == second['model']
    _, state, _ = restore_checkpoint(run.root)
    assert state['phase'] == 'tuning'
    (run.root / state['files']['sketch']).write_text('{}')
    with pytest.raises(ValueError, match='checksum'):
        restore_checkpoint(run.root)


def test_simple_chunk_geometry_mirrors_the_pinned_msccl_launch_rule():
    from vericcl.xml.capacity import simple_chunk_geometry
    assert simple_chunk_geometry(2048, 4) == (1024, 1024, True)
    assert simple_chunk_geometry(2048, 1) == (1024, 4096, False)
    assert simple_chunk_geometry(8192, 1) == (4096, 4096, True)
    assert not simple_chunk_geometry(2048, 2)[2]
    assert simple_chunk_geometry(16777216, 8)[2]
    with pytest.raises(SemanticError):
        simple_chunk_geometry(7, 4)


@pytest.mark.parametrize('value,valid', [('7', False), ('8', True), ('2147483647', True), ('2147483648', False)])
def test_nccl_buffsize_input_range_follows_the_plan(tmp_path, value, valid):
    from tests.unit.input.v5_support import request_documents, write_request
    from vericcl.input.loader import load_request
    documents = request_documents('allreduce', 2)
    documents['runtime']['environment']['NCCL_BUFFSIZE'] = value
    if valid:
        load_request(write_request(tmp_path, documents))
    else:
        with pytest.raises(InputValidationError, match='NCCL_BUFFSIZE'):
            load_request(write_request(tmp_path, documents))


def test_input_errors_after_run_creation_exit_with_the_input_code(tmp_path, monkeypatch):
    from vericcl.cli import lifecycle
    from vericcl.execution.context import RunContext
    run = RunContext.create(tmp_path, 'input-error', 'r')

    def raise_input_error(*args, **kwargs):
        raise InputValidationError('manual stage contract is empty')

    monkeypatch.setattr(lifecycle, 'run_primary', raise_input_error)
    assert lifecycle._execute(object(), run) == 2
    summary = json.loads(run.artifact('run-summary.json').read_text())
    assert summary['failure_category'] == 'input' and summary['stop_reason'] == 'input_error'


class _Prediction:
    def __init__(self, version, ratio):
        self.model_version, self.exact_completion_ratio = version, ratio
        self.t_pred_us = ratio[0] / ratio[1]


class _Bundle:
    def __init__(self, name, version, ratio):
        self.name, self.xml_sha256, self.xml_text = name, name, '<algo/>'
        self.prediction, self.can_execute = _Prediction(version, ratio), True
        self.sidecar = {'request_contract': {'ranks': 2, 'slice_bytes': 4}}
        self.checks = {}


def test_compare_reevaluates_a_proposal_made_under_an_earlier_model():
    from vericcl.workflow_v5 import PrimaryWorkflow
    workflow = PrimaryWorkflow.__new__(PrimaryWorkflow)
    workflow.original = SimpleNamespace(request_config={'sketch': {'calibration': {'enabled': False},
                                                                   'trace_and_measure': {'enabled': False}}})
    workflow.model = SimpleNamespace(version='model-2')
    workflow.best, workflow.best_inputs = _Bundle('best', 'model-2', (10, 1)), 'inputs'
    adopted, reassessed = [], []

    @contextmanager
    def stage(name):
        yield None

    def reassess(bundle, inputs, lease):
        reassessed.append(bundle.name)
        return _Bundle(bundle.name + '-reassessed', 'model-2', bundle.prediction.exact_completion_ratio)

    workflow.stage, workflow.reassess = stage, reassess
    workflow.store = lambda bundle, inputs, origin: {'record': origin}
    workflow.publish_best = lambda reason: None
    workflow.measured_evidence = lambda old, new, inputs: {'measured_required': False}
    workflow.adopt = lambda bundle, inputs, record, evidence=None: adopted.append(bundle.name) or True
    assert workflow.compare(_Bundle('proposal', 'model-1', (5, 1)), 'inputs', {'record': 'r'})
    assert reassessed == ['proposal'] and adopted == ['proposal-reassessed']


def test_reduction_reshaping_keeps_allocation_and_shortens_the_estimate(tmp_path):
    from tests.unit.input.v5_support import request_documents, write_request
    from vericcl.execution.budget import StandaloneLease
    from vericcl.input.loader import load_request
    from vericcl.solver.trees import timing
    from vericcl.solver.trees.operators import RouteContext, check_routed_semantics, construct_routes
    from vericcl.topology import load_topology
    documents = request_documents('allreduce', 4)
    documents['atom']['synthesis_method'] = 'constructive_trees'
    documents['sketch']['hyperparameters']['total_size_bytes'] = 4 * 1024
    documents['sketch']['solver'].update(max_channels=1, per_model_timeout_s=5)
    inputs = load_request(write_request(tmp_path, documents))
    context = RouteContext(StandaloneLease.after(60))
    bundle = construct_routes(inputs, load_topology(inputs), None, context)
    variant = timing.reshape_reductions(bundle, context)
    record = context.records[-1]
    assert record['phase'] == 'reduction_tree_reshaping'
    assert variant is not None and record['estimated_after_us'] < record['estimated_before_us']
    assert check_routed_semantics(variant)
    assert [batch.demand_ids for batch in variant.batches] == [batch.demand_ids for batch in bundle.batches]
    for before, after in zip(bundle.batches, variant.batches):
        assert (after.route.root, after.route.send_edges) == (before.route.root, before.route.send_edges)
    assert timing._binomial_reduce_edges(0, (1, 2, 3), {(1, 0), (2, 0)}) is None


def test_release_parser_checks_only_the_selected_nccl_tests_column():
    from vericcl.verification.online.nccl_tests import parse_nccl_tests_output
    row = '        2048           256     float    none      -1    16.61    0.12    0.06      0    16.14    0.13    0.06    N/A\n'
    with pytest.raises(SemanticError, match='wrong count'):
        parse_nccl_tests_output(row, 2048)
    assert parse_nccl_tests_output(row, 2048, checked_column=False)[0].out_of_place.wrong_count == 0
    with pytest.raises(SemanticError, match='wrong count'):
        parse_nccl_tests_output(row, 2048, checked_column=True)
    corrupted = row.replace('    0    16.14', '    3    16.14')
    with pytest.raises(SemanticError, match='correctness'):
        parse_nccl_tests_output(corrupted, 2048, checked_column=False)
