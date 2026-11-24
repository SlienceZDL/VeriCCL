"""Repair type 1: hard correctness violations are repaired locally, else rejected."""
from dataclasses import replace

from tests.scip.test_joint_repair_v5 import repair_fixture
from vericcl.tuning.correctness import network_operations, repair_correctness, symbolic_ownership
from vericcl.xml.lower import lower_and_check
from vericcl.xml.v5_model import NetworkCandidate


def _broken(bundle, context, ops):
    inputs = context.lowering.inputs
    candidate = NetworkCandidate(inputs.rank_count, inputs.hyperparameters.slice_count,
                                 inputs.hyperparameters.slice_size_bytes, tuple(ops), 'broken')
    return lower_and_check(candidate, context.lowering)


def _ops(bundle):
    return list(bundle.source.operations)


def test_missing_delivery_is_found_by_ownership_execution_and_added_from_an_owner(tmp_path):
    bundle, context = repair_fixture(tmp_path)
    ops = [op for op in _ops(bundle) if op.operation_id != 'slice2-to2']
    broken = _broken(bundle, context, ops)
    assert not broken.can_execute
    found = symbolic_ownership(ops, context.lowering.inputs).violations
    assert [v.kind for v in found] == ['missing'] and found[0].witness['rank'] == 2
    repaired, report = repair_correctness(broken, context.lowering)
    assert repaired is not None and repaired.can_execute, report.as_json()
    assert report.status == 'repaired' and report.steps[0]['action'] == 'add_delivery'
    delivered = [op for op in network_operations(repaired, context.lowering)[0]
                 if op.dst_rank == 2 and op.contributors == frozenset({2})]
    assert len(delivered) == 1


def test_relay_without_its_delivery_dependency_gets_the_dependency(tmp_path):
    bundle, context = repair_fixture(tmp_path)
    ops = [op for op in _ops(bundle) if op.operation_id != 'slice3-to2']
    relay = replace(next(op for op in _ops(bundle) if op.operation_id == 'slice3-to2'),
                    src_rank=1, priority_time=-1.0, dependencies=frozenset())
    ops.append(relay)
    broken = _broken(bundle, context, ops)
    assert not broken.can_execute
    assert symbolic_ownership(ops, context.lowering.inputs).violations[0].kind == 'unavailable'
    repaired, report = repair_correctness(broken, context.lowering)
    assert repaired is not None and repaired.can_execute, report.as_json()
    assert report.steps[0]['action'] == 'add_dependency'
    fixed = next(op for op in network_operations(repaired, context.lowering)[0] if op.operation_id == 'slice3-to2')
    assert 'slice3-to1' in fixed.dependencies


def test_cyclic_order_hints_are_dropped_and_the_network_is_lowered_again(tmp_path):
    bundle, context = repair_fixture(tmp_path)
    ops = _ops(bundle)
    a, b = ops[0], ops[1]
    ops[0] = replace(a, reference_dependencies=frozenset({b.operation_id}))
    ops[1] = replace(b, reference_dependencies=frozenset({a.operation_id}))
    broken = _broken(bundle, context, ops)
    assert not broken.can_execute
    repaired, report = repair_correctness(broken, context.lowering)
    assert repaired is not None and repaired.can_execute, report.as_json()
    assert report.steps[0]['violation'] == 'cycle'


def test_a_valid_candidate_is_returned_unchanged_and_non_correctness_failures_are_not_handled(tmp_path):
    bundle, context = repair_fixture(tmp_path)
    same, report = repair_correctness(bundle, context.lowering)
    assert same is bundle and report.status == 'valid'
    from types import MappingProxyType
    from vericcl.xml.v5_lower import _check
    checks = dict(bundle.checks)
    checks['launch'] = _check('launch', 'invalid', 'runtime capacity')
    capacity = replace(bundle, checks=MappingProxyType(checks))
    none, report = repair_correctness(capacity, context.lowering)
    assert none is None and report.status == 'not_correctness'


def test_forbidden_edge_is_rerouted_over_legal_edges(tmp_path):
    from tests.unit.input.v5_support import request_documents, write_request
    from vericcl.execution.budget import StandaloneLease
    from vericcl.input.loader import load_request
    from vericcl.topology import load_topology
    from vericcl.xml.lower import LoweringContext
    (tmp_path / 'base').mkdir()
    bundle, context = repair_fixture(tmp_path / 'base')
    docs = request_documents('broadcast', 3)
    docs['sketch']['solver'].update(max_channels=1, per_model_timeout_s=5)
    docs['atom']['synthesis_method'] = 'constructive_trees'
    docs['atom']['forbidden_transfers'] = [[1, 0, 2, 0]]       # slice 1 may not go 0 -> 2 in stage 0
    (tmp_path / 'banned').mkdir()
    inputs = load_request(write_request(tmp_path / 'banned', docs))
    lowering = LoweringContext(inputs, load_topology(inputs), context.lowering.model, StandaloneLease.after(30))
    candidate = NetworkCandidate(inputs.rank_count, inputs.hyperparameters.slice_count,
                                 inputs.hyperparameters.slice_size_bytes, tuple(_ops(bundle)), 'broken')
    broken = lower_and_check(candidate, lowering)
    assert not broken.can_execute
    assert symbolic_ownership(_ops(bundle), inputs).violations[0].kind == 'forbidden'
    repaired, report = repair_correctness(broken, lowering)
    assert repaired is not None and repaired.can_execute, report.as_json()
    assert report.steps[0]['action'] == 'reroute_legal_edges'
    hops = [op for op in network_operations(repaired, lowering)[0] if 1 in op.contributors and op.dst_rank == 2]
    assert len(hops) == 1 and hops[0].src_rank == 1
