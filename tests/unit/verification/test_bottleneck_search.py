from dataclasses import replace
import importlib

import pytest

from tests.integration.test_lowering_v5 import candidate_fixture
from vericcl.xml.lower import lower_and_check
from vericcl.execution.budget import BudgetExhausted


def setup(tmp_path, *, ranks=3, operator='broadcast', channels=2):
    from vericcl.verification.bottlenecks import AnalysisContext, SearchScope
    candidate, lowering = candidate_fixture(tmp_path, operator, ranks=ranks, channels=channels)
    bundle = lower_and_check(candidate, lowering)
    assert bundle.can_execute
    return bundle, AnalysisContext(lowering), SearchScope('test-search', batch_size=2)


def test_slow_tail_without_wait_has_path_opportunities(tmp_path):
    from vericcl.verification.bottlenecks import analyze
    bundle, context, scope = setup(tmp_path)
    batch = analyze(bundle, context.lowering.model, scope, context)
    assert batch.status == 'opportunities'
    assert batch.causes and batch.strategies
    assert any(cause.evidence['service_us'] > 0 for cause in batch.causes)
    assert any(strategy.paths for strategy in batch.strategies)
    assert batch.coverage['unresolved_branches']
    assert not batch.coverage['proven_no_improvement']


def test_causal_index_keeps_all_equal_tails_and_resource_competitors():
    from vericcl.performance.events import Executable, WorkOperation
    from vericcl.performance.model import CostModel, ServiceCost
    from vericcl.performance.simulation import predict
    from vericcl.verification.flow_index import build_causal_index
    ops = (WorkOperation('a', 'send', 'link', 100, resources=('shared',), flow_id='a'),
           WorkOperation('b', 'send', 'link', 100, resources=('shared',), flow_id='b'),
           WorkOperation('c', 'copy', 'copy', 200))
    model = CostModel('tails', {'link': (ServiceCost(0, .01),), 'copy': (ServiceCost(0, .01),)}, {'shared': 100})
    executable = Executable(ops)
    index = build_causal_index(executable, predict(executable, model, exact=True))
    assert set(index.tails) == {'a', 'b', 'c'}
    assert set(index.related['a']) == {'b'}
    assert index.wait_us['a'] == 0


def test_busy_alternative_is_not_filtered_by_idle_interval(tmp_path):
    from vericcl.verification.bottlenecks import analyze
    bundle, context, scope = setup(tmp_path, operator='allgather')
    candidates = []
    for _ in range(100):
        batch = analyze(bundle, context.lowering.model, scope, context)
        candidates.extend(batch.strategies)
        if batch.status == 'candidates_exhausted':
            break
    assert candidates and any(strategy.paths for strategy in candidates)
    assert any(cause.evidence['resource_competitors'] for cause in batch.causes)
    assert not batch.coverage['proven_no_improvement']


def test_batches_expand_without_duplicate_strategies_or_candidate_cap(tmp_path):
    from vericcl.verification.bottlenecks import analyze
    bundle, context, scope = setup(tmp_path, ranks=4)
    seen = set()
    for _ in range(200):
        batch = analyze(bundle, context.lowering.model, scope, context)
        assert len(batch.strategies) <= 2
        assert not seen.intersection(s.strategy_id for s in batch.strategies)
        seen.update(s.strategy_id for s in batch.strategies)
        if batch.status == 'candidates_exhausted':
            break
    else:
        pytest.fail('finite fixture strategy source failed to exhaust')
    assert len(seen) > scope.batch_size
    assert batch.coverage['unresolved_branches']
    assert batch.coverage['generation_complete']


def test_direct_edge_ban_allows_detour_and_invalid_rows_do_not_fill_batch(tmp_path):
    from vericcl.verification.bottlenecks import analyze, StrategyRecord
    bundle, context, scope = setup(tmp_path)
    network = next(op for op in bundle.storage_plan.instructions if op.kind == 'send')
    src = bundle.storage_plan.values[network.inputs[0]].rank
    dst = network.rank
    middle = next(rank for rank in range(3) if rank not in {src, dst})
    invalid = StrategyRecord('path', (network.operation_id,), paths=((network.operation_id, (src, 99, dst)),))
    valid = StrategyRecord('path', (network.operation_id,), paths=((network.operation_id, (src, middle, dst)),))
    context.strategy_sources = {'path': lambda *args: iter((invalid, invalid, valid, valid))}
    scope = replace(scope, families=('path',), batch_size=1)
    batch = analyze(bundle, context.lowering.model, scope, context)
    assert batch.status == 'opportunities' and batch.strategies == (valid,)
    assert batch.coverage['filtered'] == 2
    assert analyze(bundle, context.lowering.model, scope, context).status == 'candidates_exhausted'


def test_model_change_invalidates_cursor_and_reassesses_same_xml(tmp_path):
    from vericcl.verification.bottlenecks import analyze
    bundle, context, scope = setup(tmp_path)
    first = analyze(bundle, context.lowering.model, scope, context)
    updated, _ = context.lowering.model.with_updates('updated-model')
    second = analyze(bundle, updated, scope, context)
    assert second.strategies[0].strategy_id == first.strategies[0].strategy_id
    assert second.coverage['model_version'] == 'updated-model'
    assert second.evidence['prediction_model'] == 'updated-model'


def test_budget_and_backend_errors_remain_distinct_from_empty_candidates(tmp_path, monkeypatch):
    from vericcl.verification.bottlenecks import analyze
    bundle, context, scope = setup(tmp_path)
    context.lowering.lease.cancel()
    assert analyze(bundle, context.lowering.model, scope, context).status == 'budget_exhausted'
    bundle, context, scope = setup(tmp_path)
    context.backend = 'missing'
    batch = analyze(bundle, context.lowering.model, scope, context)
    assert batch.status == 'analysis_error'
    assert not batch.coverage.get('proven_no_improvement', False)


@pytest.mark.parametrize('backend', ['cudd', 'autoref'])
def test_explicit_backend_and_independent_decision_variables(backend):
    from vericcl.verification.bdd_backend import StrategyDecisionBDD
    relation = StrategyDecisionBDD(3, {'channel': 2, 'order': 2}, backend=backend)
    result = relation.select(((0, 0, 0), (0, 1, 1), (1, 0, 1), (2, 1, 0)), forbidden=((1, 0, 1),))
    assert set(result) == {(0, 0, 0), (0, 1, 1), (2, 1, 0)}
    assert relation.variable_count == 4


def test_missing_cudd_does_not_import_autoref(monkeypatch):
    from vericcl.verification.bdd_backend import CompactBDD, BDDBackendError
    original = importlib.import_module
    seen = []
    def missing(name, *args, **kwargs):
        seen.append(name)
        if name == 'dd.cudd':
            raise ImportError('not installed')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(importlib, 'import_module', missing)
    with pytest.raises(BDDBackendError, match='cudd'):
        CompactBDD({'choice': 2})
    assert 'dd.autoref' not in seen


def test_byte_anchored_direct_ban_allows_alternative_path(tmp_path):
    from vericcl.verification.bottlenecks import StrategyRecord, legal_strategy
    from vericcl.input.models import ForbiddenTransfer
    bundle, context, _ = setup(tmp_path)
    op = next(op for op in bundle.storage_plan.instructions if op.kind == 'send')
    source = bundle.storage_plan.values[op.inputs[0]]
    middle = next(rank for rank in range(3) if rank not in {source.rank, op.rank})
    constraint = ForbiddenTransfer(slice_id=next(iter(source.contributors)), stage_id=op.stage_id, src_rank=source.rank, dst_rank=op.rank)
    inputs = replace(context.lowering.inputs, atom_constraints=replace(context.lowering.inputs.atom_constraints,
                                                                      forbidden_transfers=frozenset({constraint})))
    context.lowering = replace(context.lowering, inputs=inputs)
    direct = StrategyRecord('path', (op.operation_id,), paths=((op.operation_id, (source.rank, op.rank)),))
    detour = replace(direct, paths=((op.operation_id, (source.rank, middle, op.rank)),))
    assert not legal_strategy(direct, bundle, context)
    assert legal_strategy(detour, bundle, context)


def test_group_roles_reject_outside_gpu_relay(tmp_path):
    from vericcl.verification.bottlenecks import StrategyRecord, legal_strategy
    bundle, context, _ = setup(tmp_path)
    op = next(op for op in bundle.storage_plan.instructions if op.kind == 'send')
    source = bundle.storage_plan.values[op.inputs[0]]
    outside = next(rank for rank in range(3) if rank not in {source.rank, op.rank})
    storage = replace(bundle.storage_plan, instructions=tuple(replace(item, participants=(source.rank, op.rank))
                      if item.operation_id == op.operation_id else item for item in bundle.storage_plan.instructions))
    restricted = replace(bundle, storage_plan=storage)
    proposal = StrategyRecord('path', (op.operation_id,), paths=((op.operation_id, (source.rank, outside, op.rank)),))
    assert not legal_strategy(proposal, restricted, context)
