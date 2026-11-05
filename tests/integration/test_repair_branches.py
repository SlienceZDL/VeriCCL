from dataclasses import replace

import pytest

from tests.scip.test_joint_repair_v5 import repair_fixture


def test_alternative_strategy_is_tried_after_invalid_route(tmp_path):
    from vericcl.verification.bottlenecks import StrategyRecord
    from vericcl.tuning.strategies import build_domain, solve_repair
    bundle, context = repair_fixture(tmp_path)
    strategy = StrategyRecord('path', ('slice0-to1',), paths=(('slice0-to1', (0, 2, 1)),))
    bad = replace(strategy, paths=(('slice0-to1', (0, 99, 1)),))
    result = solve_repair(bundle, build_domain(None, strategy), (bad, strategy), context)
    assert result.status == 'improved'
    assert any(record['status'] == 'invalid_strategy' for record in result.attempts)


def test_partition_branch_restarts_initial_synthesis_under_same_parent_budget(tmp_path):
    from vericcl.tuning.resynthesis import resynthesize_branch, ResynthesisRequest
    bundle, context = repair_fixture(tmp_path)
    parent = context.lowering.lease
    request = ResynthesisRequest('partition', slice_bytes=2048, reason='reduce startup and runtime step count')
    stream = resynthesize_branch(bundle, request, context)
    result = next(iter(stream))
    assert result.context.inputs.hyperparameters.slice_size_bytes == 2048
    assert result.context.lease is parent
    assert result.incumbent is bundle
    assert result.candidate.prediction.model_version == context.lowering.model.version
    assert stream.budget_stage == 'repair' and stream.outer_round_reset is False
    stream.close()


def test_partition_branch_preserves_divisibility_and_fixed_external_contract(tmp_path):
    from vericcl.tuning.resynthesis import resynthesize_branch, ResynthesisRequest
    from vericcl.errors import SemanticError
    bundle, context = repair_fixture(tmp_path)
    with pytest.raises(SemanticError):
        next(iter(resynthesize_branch(bundle, ResynthesisRequest('partition', slice_bytes=3000,
                                                               reason='invalid partition'), context)))


def test_structure_branch_keeps_selected_synthesis_method_and_s(tmp_path):
    from vericcl.tuning.resynthesis import resynthesize_branch, ResynthesisRequest
    bundle, context = repair_fixture(tmp_path)
    stream = resynthesize_branch(bundle, ResynthesisRequest('structure', reason='expand route structure'), context)
    result = next(iter(stream))
    assert result.context.inputs.selected_synthesis_method == context.lowering.inputs.selected_synthesis_method
    assert result.context.inputs.hyperparameters.slice_size_bytes == context.lowering.inputs.hyperparameters.slice_size_bytes
    assert result.context.lease is context.lowering.lease
    assert result.incumbent is bundle
    stream.close()


def test_internal_allreduce_root_change_requires_complete_slice_domain(tmp_path):
    from tests.integration.test_lowering_v5 import candidate_fixture
    from vericcl.xml.lower import lower_and_check
    from vericcl.verification.bottlenecks import StrategyRecord
    from vericcl.tuning.strategies import RepairContext, build_domain, implementations
    candidate, lowering = candidate_fixture(tmp_path, 'allreduce', ranks=3)
    bundle = lower_and_check(candidate, lowering)
    context = RepairContext(lowering)
    logical = 0
    ids = tuple(op.operation_id for op in bundle.storage_plan.instructions
                if bundle.storage_plan.values[op.output].logical_position == logical)
    roots = []
    for root in range(3):
        strategy = StrategyRecord('internal_root', ids, parameters=(('root', root),))
        sources = implementations(bundle, build_domain(None, strategy), strategy, context)
        result = lower_and_check(next(sources), lowering)
        sources.close()
        assert result.can_execute
        reductions = [op for op in result.storage_plan.instructions if op.kind == 'rrc'
                      and result.storage_plan.values[op.output].logical_position == logical]
        complete = [op.rank for op in reductions if len(result.storage_plan.values[op.output].contributors) == 3]
        assert complete == [root]
        roots.append(root)
    assert roots == [0, 1, 2]


def test_explicit_root_change_cannot_escape_incomplete_domain(tmp_path):
    from tests.integration.test_lowering_v5 import candidate_fixture
    from vericcl.xml.lower import lower_and_check
    from vericcl.verification.bottlenecks import StrategyRecord
    from vericcl.tuning.strategies import RepairContext, build_domain, solve_repair
    candidate, lowering = candidate_fixture(tmp_path, 'allreduce', ranks=3)
    bundle = lower_and_check(candidate, lowering)
    key = bundle.storage_plan.instructions[0].operation_id
    strategy = StrategyRecord('internal_root', (key,), parameters=(('root', 2),))
    result = solve_repair(bundle, build_domain(None, strategy), strategy, RepairContext(lowering))
    assert result.status == 'invalid_domain'


def test_channel_search_can_use_original_request_cap_after_initial_c_selection(tmp_path):
    from vericcl.planner.parameters import initial_parameter_domain, resolve_parameters
    from vericcl.verification.bottlenecks import StrategyRecord
    from vericcl.tuning.strategies import build_domain, solve_repair
    from vericcl.xml.lower import lower_and_check
    bundle, context = repair_fixture(tmp_path, channels=2)
    original = context.lowering.inputs
    choice = next(p for p in initial_parameter_domain(original) .candidates if p.channels == 1)
    context.lowering = replace(context.lowering, inputs=resolve_parameters(original, choice))
    context.request_inputs = original
    bundle = lower_and_check(bundle.source, context.lowering)
    strategy = StrategyRecord('channel', ('slice0-to1',), channels=(('slice0-to1', 1),))
    result = solve_repair(bundle, build_domain(None, strategy), strategy, context)
    assert result.status == 'improved'
    assert result.context.inputs.solver.max_channels == 2


def test_per_model_expiry_does_not_cancel_remaining_parent_search(tmp_path, monkeypatch):
    from vericcl.execution.budget import BudgetExhausted
    from vericcl.verification.bottlenecks import StrategyRecord
    from vericcl.tuning import strategies
    bundle, context = repair_fixture(tmp_path)
    context.candidate_batch_size = 1
    action = StrategyRecord('path', ('slice0-to1',), paths=(('slice0-to1', (0, 2, 1)),))
    actual, attempts = strategies.solve_event_columns, []
    def expire_first(*args):
        attempts.append(1)
        if len(attempts) == 1:
            raise BudgetExhausted('one native model construction budget')
        return actual(*args)
    monkeypatch.setattr(strategies, 'solve_event_columns', expire_first)
    result = strategies.solve_repair(bundle, strategies.build_domain(None, action), action, context)
    assert result.status == 'improved'
    assert any(item['status'] == 'model_budget_exhausted' for item in result.solver_evidence)
    assert len(attempts) > 1


def test_capacity_invalid_candidate_can_enter_partition_resynthesis(tmp_path):
    from vericcl.tuning.resynthesis import resynthesize_branch, ResynthesisRequest
    from vericcl.xml.capacity import RuntimeLimits
    from vericcl.xml.lower import lower_and_check
    bundle, context = repair_fixture(tmp_path)
    context.lowering = replace(context.lowering, limits=RuntimeLimits(steps_per_tb=3))
    invalid = lower_and_check(bundle.source, context.lowering)
    assert not invalid.can_execute and invalid.can_repair
    stream = resynthesize_branch(invalid, ResynthesisRequest('partition', slice_bytes=2048,
                                                           reason='reduce step count below runtime ceiling'), context)
    candidate = next(stream)
    assert candidate.candidate.can_execute
    assert candidate.incumbent is invalid
    stream.close()
