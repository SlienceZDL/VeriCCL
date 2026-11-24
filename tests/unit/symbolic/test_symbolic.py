"""Symbolic execution, {s, pd, t} atoms, BDD correctness checks and hotspots."""
from vericcl.semantics.atom import symbolic_step
from dataclasses import replace
from types import SimpleNamespace

import pytest

from vericcl.errors import SemanticError
from vericcl.symbolic.atom import Atom, PathDAG, SliceVec, Symbol, TimeVec
from vericcl.symbolic.bdd import BddSpace
from vericcl.symbolic.executor import SymbolicExecutor, execute_bundle
from vericcl.symbolic.terms import TermFactory, critical_branch, evaluate
from vericcl.verification.hotspots import HotspotSettings, hotspot_strategies, locate_hotspots
from vericcl.verification.symbolic_check import CHECKS, check_bundle, check_execution
from tests.scip.test_joint_repair_v5 import repair_fixture


def test_terms_are_interned_and_evaluate_with_attributed_max():
    f = TermFactory()
    a, b = f.param('alpha', 'x'), f.param('beta', 'x')
    assert f.add(a, f.mul(2.0, b)) is f.add(a, f.mul(2.0, b))
    late = f.max((('data:p', f.add(a, f.const(1))), ('tb:q', f.mul(3.0, b))))
    memo = {}
    values = {('alpha', 'x'): 1.0, ('beta', 'x'): 2.0}
    assert evaluate(late, values.__getitem__, memo) == 6.0
    assert critical_branch(late, memo)[0] == 'tb:q'


def test_symbols_exist_only_through_symbolic_execution():
    with pytest.raises(SemanticError):
        Symbol(0, 1, TermFactory().zero)


def test_bdd_space_image_and_closure():
    space = BddSpace({'rank': 4, 'rank2': 4, 'rank3': 4})
    edges = space.relation(('rank', 'rank2'), {(0, 1), (1, 2), (2, 3)})
    reach = space.eq('rank', 0)
    for _ in range(3):
        reach = reach | space.image(reach, edges, present=('rank',), nxt=('rank2',))
    assert space.tuples(reach, ('rank',)) == ((0,), (1,), (2,), (3,))
    closure = space.closure(edges, left='rank', right='rank2', middle='rank3')
    assert (0, 3) in space.tuples(closure, ('rank', 'rank2'))
    assert space.is_empty(closure & space.field_equal('rank', 'rank2'))


@pytest.fixture
def fixture(tmp_path):
    bundle, context = repair_fixture(tmp_path)
    return bundle, context.lowering


def test_symbolic_times_reproduce_the_event_simulator(fixture):
    bundle, lowering = fixture
    execution = execute_bundle(bundle, lowering.inputs, lowering.model)
    times = execution.times()
    for key in execution.order:
        assert times[key][1] == pytest.approx(bundle.prediction.end_times[key], abs=1e-6)
    tail = execution.critical_path()[0]
    assert times[tail][1] == pytest.approx(bundle.prediction.t_pred_us, abs=1e-6)


def test_atoms_have_slice_path_and_time_vectors(fixture):
    bundle, lowering = fixture
    execution = execute_bundle(bundle, lowering.inputs, lowering.model)
    produced = [atom for state, atom in execution.atoms.items() if execution.producer[state] is not None]
    assert produced
    for atom in produced:
        assert isinstance(atom, Atom) and isinstance(atom.s, SliceVec) and isinstance(atom.t, TimeVec)
        assert isinstance(atom.pd, PathDAG) and atom.pd.hop_count >= 1
        assert all(isinstance(symbol, Symbol) for symbol in atom.pd.symbols)
        assert atom.pd.current_rank == atom.rank
        assert {s // execution.slice_count for s in atom.contributors} <= atom.pd.roots
        assert execution.value(atom.t.st_time) <= execution.value(atom.t.ed_time)


def test_all_symbolic_bdd_checks_hold_for_a_lowered_schedule(fixture):
    bundle, lowering = fixture
    result, _ = check_bundle(bundle, lowering.inputs, None)
    assert result.valid and set(result.checks) == set(CHECKS)
    assert bundle.checks['semantic'].evidence['symbolic_bdd'] == dict(result.checks)


def test_missing_transfer_is_a_coverage_violation_with_witness(fixture):
    bundle, lowering = fixture
    storage = bundle.storage_plan
    read = {key for op in storage.instructions for key in op.inputs}
    leaf = next(op for op in storage.instructions if op.output not in read and op.kind == 'send')
    reduced = replace(storage, instructions=tuple(op for op in storage.instructions if op is not leaf))
    program = replace(bundle.program, blocks=tuple(
        replace(block, steps=tuple(step for step in block.steps if step.operation_id != leaf.operation_id))
        for block in bundle.program.blocks))
    execution = SymbolicExecutor(reduced, program, bundle.executable, lowering.inputs).run()
    execution.outputs = dict(storage.outputs)
    result = check_execution(execution, SimpleNamespace(storage_plan=reduced, program=program), lowering.inputs)
    assert not result.valid
    assert result.checks['coverage'] == 'invalid'
    assert result.witnesses['coverage']['rank'] == leaf.rank


def test_cyclic_dependencies_are_rejected_by_symbolic_execution(fixture):
    bundle, lowering = fixture
    executor = SymbolicExecutor(bundle.storage_plan, bundle.program, bundle.executable, lowering.inputs)
    first, second = sorted(executor.dependencies)[:2]
    executor.dependencies = dict(executor.dependencies)
    executor.dependencies[first] = frozenset({second})
    executor.dependencies[second] = frozenset({first})
    with pytest.raises(SemanticError):
        executor.run()


def test_hotspot_is_a_waiting_transfer_with_an_idle_alternative_link(fixture):
    bundle, lowering = fixture
    execution = execute_bundle(bundle, lowering.inputs, lowering.model)
    report = locate_hotspots(execution, bundle, lowering.inputs, lowering.topology, lowering.model)
    assert report.hotspots
    top = report.hotspots[0]
    # slice transfers to rank 1 queue on the slow 0->1 link while 0->2 is idle.
    assert top.congested_resource == 'link:0:1' and (0, 2) in top.alternative_links
    assert top.on_critical_path and top.wait_us > 0
    assert top.related_operations, 'the schedules it waits for are reported'
    strategies = list(hotspot_strategies(report, bundle, lowering))
    # One flow per slice here (no idle sibling flow), so every hotspot gets a
    # scheduling-bottleneck repair (type 3), hotspot first in its domain.
    assert ('bottleneck', top.operation_id) in {(family, dict(s.parameters)['hotspot']) for family, s in strategies}
    assert all(top.operation_id in s.operation_ids for family, s in strategies
               if dict(s.parameters)['hotspot'] == top.operation_id)


def test_no_hotspot_when_alternatives_are_never_idle(fixture):
    bundle, lowering = fixture
    execution = execute_bundle(bundle, lowering.inputs, lowering.model)
    strict = HotspotSettings(idle_utilization_threshold=0.0)
    report = locate_hotspots(execution, bundle, lowering.inputs, lowering.topology, lowering.model, settings=strict)
    assert not report.hotspots


def test_symbolic_bdd_verdict_is_reused_after_a_model_update(fixture):
    from vericcl.performance.model import CostModel
    from vericcl.verification.symbolic_check import clear_check_cache
    from vericcl.xml.parser import check_xml_bundle
    bundle, lowering = fixture
    clear_check_cache()
    first = check_xml_bundle(bundle.xml_text, bundle.sidecar, lowering)
    assert first.checks['semantic'].evidence['symbolic_cache'] == 'computed'
    model = lowering.model
    updated = CostModel(model.version + '-updated', model.services, model.resources, model.profiles, model.context, model.metadata)
    second = check_xml_bundle(bundle.xml_text, bundle.sidecar, replace(lowering, model=updated))
    assert second.checks['semantic'].evidence['symbolic_cache'] == 'hit'
    assert second.checks['semantic'].evidence['symbolic_bdd'] == first.checks['semantic'].evidence['symbolic_bdd']
    assert second.prediction.model_version == updated.version


def test_only_tail_affecting_hotspots_produce_repairs(fixture):
    bundle, lowering = fixture
    execution = execute_bundle(bundle, lowering.inputs, lowering.model)
    report = locate_hotspots(execution, bundle, lowering.inputs, lowering.topology, lowering.model)
    assert report.hotspots and all(hotspot.affects_tail for hotspot in report.hotspots)
    critical = set(execution.critical_path())
    for hotspot in report.hotspots:
        assert hotspot.operation_id in critical or critical & set(hotspot.related_operations)
    everything = locate_hotspots(execution, bundle, lowering.inputs, lowering.topology, lowering.model,
                                 settings=HotspotSettings(tail_only=False))
    assert len(everything.hotspots) >= len(report.hotspots)
    assert report.counters['hotspot_operations'] == everything.counters['hotspot_operations']
