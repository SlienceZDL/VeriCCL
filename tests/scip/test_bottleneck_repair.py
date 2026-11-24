"""Repair type 3: hotspot without an idle sibling flow -> scoped owner reroute + local MILP."""
from tests.scip.test_joint_repair_v5 import repair_fixture
from vericcl.symbolic.executor import execute_bundle
from vericcl.tuning.allocation import reallocation_strategies
from vericcl.tuning.bottleneck import (BottleneckSettings, bottleneck_proposals, bottleneck_strategies,
                                       related_scope, solve_bottleneck)
from vericcl.tuning.correctness import network_operations, _network_from_storage
from vericcl.verification.hotspots import locate_hotspots


def _report(bundle, lowering):
    execution = execute_bundle(bundle, lowering.inputs, lowering.model)
    return execution, locate_hotspots(execution, bundle, lowering.inputs, lowering.topology, lowering.model)


def test_single_flow_hotspot_is_not_a_reallocation_but_a_bottleneck(tmp_path):
    bundle, context = repair_fixture(tmp_path)
    _, report = _report(bundle, context.lowering)
    assert report.hotspots
    assert not list(reallocation_strategies(report, bundle, context.lowering))
    assert list(bottleneck_strategies(report, bundle, context.lowering))


def test_scope_is_bounded_and_starts_at_the_hotspot(tmp_path):
    bundle, context = repair_fixture(tmp_path)
    execution, report = _report(bundle, context.lowering)
    hotspot = report.hotspots[0]
    ops = _network_from_storage(bundle)
    scope = related_scope(hotspot, execution, ops, bundle.prediction, context.lowering.topology,
                          BottleneckSettings(max_scope_operations=3))
    assert scope.reroutable[0] == hotspot.operation_id
    assert len(scope.reroutable) + len(scope.consumers) <= 3
    assert all('link:0:1' in ('link:%d:%d' % (o.src_rank, o.dst_rank),) for o in ops if o.operation_id in scope.reroutable)


def test_hot_slices_are_forwarded_by_a_rank_that_already_owns_them(tmp_path):
    bundle, context = repair_fixture(tmp_path)
    _, report = _report(bundle, context.lowering)
    strategy = next(record for _, record in bottleneck_strategies(report, bundle, context.lowering))
    proposals, evidence = bottleneck_proposals(bundle, strategy, report, context)
    assert evidence['plans'], evidence
    # Every plan starts at an owner: the root, or rank 2 after it received the slice.
    assert all(path[0] in (0, 2) and path[-1] == 1 for paths in evidence['plans'].values() for path in paths)
    assert evidence['local_milp']['status'] in {'optimal', 'solved', 'gap_limit'} or evidence['local_milp'].get('primal') is not None
    result = solve_bottleneck(bundle, strategy, report, context)
    assert result.status == 'improved', result.attempts
    assert result.candidate.prediction.t_pred_us < bundle.prediction.t_pred_us
    forwarded = [op for op in network_operations(result.candidate, context.lowering)[0] if op.src_rank == 2 and op.dst_rank == 1]
    assert forwarded and all(any(dep.endswith('-to2') for dep in op.dependencies) for op in forwarded)
