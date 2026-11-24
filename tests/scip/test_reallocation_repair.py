"""Repair type 2: re-divide a chunk's slices among its existing flows."""
from tests.scip.test_joint_repair_v5 import repair_fixture
from vericcl.tuning.allocation import chunk_flows, solve_reallocation, units_of, reallocation_strategies
from vericcl.tuning.correctness import network_operations
from vericcl.verification.bottlenecks import StrategyRecord
from vericcl.xml.lower import lower_and_check
from vericcl.xml.v5_model import NetworkCandidate, NetworkOperation


def two_flow_bundle(tmp_path, relayed=(3,)):
    """Root 0 broadcasts 4 slices to ranks 1 and 2. Link 0->1 is 10x slower.

    Flow A sends 0->1 and 0->2 directly; flow B relays 0->2->1. Slices in
    ``relayed`` use flow B, the others flow A.
    """
    bundle, context = repair_fixture(tmp_path)
    inputs = context.lowering.inputs
    ops = []
    for s in range(4):
        ops.append(NetworkOperation('s%d-to2' % s, 'send', 0, 2, 0, 0, frozenset({s}), frozenset({s}), (0, 1, 2),
                                    frozenset(), frozenset(), float(s)))
        if s in relayed:
            ops.append(NetworkOperation('s%d-to1' % s, 'send', 2, 1, 0, 0, frozenset({s}), frozenset({s}), (0, 1, 2),
                                        frozenset({'s%d-to2' % s}), frozenset(), float(s) + .5))
        else:
            ops.append(NetworkOperation('s%d-to1' % s, 'send', 0, 1, 0, 0, frozenset({s}), frozenset({s}), (0, 1, 2),
                                        frozenset(), frozenset(), float(s)))
    candidate = NetworkCandidate(inputs.rank_count, inputs.hyperparameters.slice_count,
                                 inputs.hyperparameters.slice_size_bytes, tuple(ops), 'two-flow')
    bundle = lower_and_check(candidate, context.lowering)
    assert bundle.can_execute, {k: c.message for k, c in bundle.checks.items() if c.status.value == 'invalid'}
    return bundle, context


def _strategy(bundle, context):
    ops = network_operations(bundle, context.lowering)[0]
    units, op_unit = units_of(ops, context.lowering.inputs)
    flows = chunk_flows(units)
    assert len(flows) == 1 and len(next(iter(flows.values()))) == 2
    chunk = next(iter(flows))
    return StrategyRecord('reallocation', tuple(sorted(op.operation_id for op in ops)),
                          parameters=(('chunks', (chunk,)), ('hot_unit', op_unit['s0-to1']),
                                      ('hotspot', 's0-to1'), ('idle_unit', op_unit['s3-to1']),
                                      ('resource', 'link:0:1')))


def test_chunk_flows_group_slices_by_route(tmp_path):
    bundle, context = two_flow_bundle(tmp_path)
    units, _ = units_of(network_operations(bundle, context.lowering)[0], context.lowering.inputs)
    sizes = sorted(len(members) for members in next(iter(chunk_flows(units).values())).values())
    assert sizes == [1, 3]


def test_slices_move_from_the_hot_flow_to_the_idle_flow_and_complete_earlier(tmp_path):
    bundle, context = two_flow_bundle(tmp_path)
    result = solve_reallocation(bundle, _strategy(bundle, context), context)
    assert result.status == 'improved', result.attempts
    assert result.candidate.prediction.t_pred_us < bundle.prediction.t_pred_us
    ops = network_operations(result.candidate, context.lowering)[0]
    units, _ = units_of(ops, context.lowering.inputs)
    sizes = sorted(len(m) for m in next(iter(chunk_flows(units).values())).values())
    assert sizes != [1, 3]                       # 3/1 -> a more even split
    proposals = [a['proposal'] for a in result.attempts if a['status'] == 'proposal']
    assert proposals[0]['kind'] == 'balanced_counts'
    assert any(p['kind'] == 'incremental' for p in proposals)


def test_balanced_chunk_is_not_changed(tmp_path):
    bundle, context = two_flow_bundle(tmp_path, relayed=(0, 1, 2, 3))
    ops = network_operations(bundle, context.lowering)[0]
    units, _ = units_of(ops, context.lowering.inputs)
    assert len(next(iter(chunk_flows(units).values()))) == 1     # one flow: nothing to re-divide


def test_hotspot_with_an_idle_sibling_flow_triggers_reallocation(tmp_path):
    from vericcl.symbolic.executor import execute_bundle
    from vericcl.verification.hotspots import locate_hotspots
    bundle, context = two_flow_bundle(tmp_path)
    lowering = context.lowering
    execution = execute_bundle(bundle, lowering.inputs, lowering.model)
    report = locate_hotspots(execution, bundle, lowering.inputs, lowering.topology, lowering.model)
    assert report.hotspots and report.utilization
    records = [record for _, record in reallocation_strategies(report, bundle, lowering)]
    assert records, [h.operation_id for h in report.hotspots]
    parameters = dict(records[0].parameters)
    assert parameters['resource'] == 'link:0:1'
    assert solve_reallocation(bundle, records[0], context).status == 'improved'
