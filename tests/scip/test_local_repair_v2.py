from dataclasses import replace

import pytest

from vericcl.semantics.collective import CollectiveKind
from vericcl.solver.budget import ModelBudget
from tests.scip.helpers import fresh_budget
from vericcl.topology.model import LaneKey
from vericcl.tuning.impact import ImpactClosure
from vericcl.tuning.local_milp import solve_local_repair
from vericcl.tuning.model import RepairStatus
from vericcl.tuning.repair import repair_flow_suffix
from vericcl.verification.bdd_flow import FlowReplacementHint
from vericcl.verification.flow_index import build_flow_index
from vericcl.verification.model import ValidationStatus
from vericcl.verification.constraints import verify_schedule_constraints
from vericcl.verification.semantics import verify_schedule_semantics
from tests.unit.tuning.helpers import overlay
from tests.unit.verification.helpers import inputs, topology
from tests.unit.xml.helpers import (
    two_rank_allreduce_schedule, two_send_same_lane_schedule,
    concurrent_reduce_star_schedule, send_relay_schedule,
)


pytestmark = [pytest.mark.phase05, pytest.mark.scip]


def retime(transfer, start, end):
    return replace(transfer, st_time=start, ed_time=end, atoms=tuple(
        replace(atom, st_time=start, ed_time=end) for atom in transfer.atoms
    ))


def hint_for(schedule, transfer_id, path, channel=0):
    flow = next(flow for flow in build_flow_index(schedule).flows
                if transfer_id in flow.comparison_transfer_ids)
    index = flow.transfer_ids.index(transfer_id)
    return FlowReplacementHint(
        source_flow_id=flow.flow_id, demand_id=flow.demand_id,
        candidate_flow_ids=("selected",), candidate_paths={"selected": path},
        candidate_first_lanes={"selected": LaneKey(path[0], path[1], channel)},
        divergence_rank=path[0], waiting_transfer_id=transfer_id,
        bottleneck_lane=flow.lanes[index], wait_start_us=0.0,
        wait_end_us=1.0, earliest_candidate_start_us=0.0,
    )


def solve(schedule, hint, topo, data, *, ids=None, tuning=None, seconds=10.0):
    ids = frozenset(ids or (hint.waiting_transfer_id,))
    impact = ImpactClosure(ids, ids, {key: frozenset({"seed"}) for key in ids})
    return solve_local_repair(schedule, hint, impact,
                              tuning or replace(overlay(), channel_count=2),
                              topo, data, fresh_budget(seconds))


def broadcast_case():
    schedule = two_send_same_lane_schedule()
    first = retime(schedule.transfers[0], 0.0, 3.0)
    second = schedule.transfers[1]
    atom = second.atoms[0]
    stage = atom.path[0]
    second = replace(second, atoms=(replace(atom, path=(replace(
        stage, symbols=(replace(stage.symbols[0], ready_time=0.0),)),)),))
    second = retime(second, 3.0, 4.0)
    schedule = replace(schedule, transfers=(first, second))
    return schedule, hint_for(schedule, second.transfer_id, (0, 1), 1)


def test_v2_applies_optimized_times_instead_of_greedy_schedule():
    schedule = two_rank_allreduce_schedule()
    schedule = replace(schedule, transfers=(schedule.transfers[0],
                         retime(schedule.transfers[1], 8.0, 9.0)))
    hint = hint_for(schedule, "allreduce-reduce", (1, 0), 1)
    topo, data = topology(), inputs()
    greedy = repair_flow_suffix(schedule, hint, replace(overlay(), channel_count=2), topo, data)
    result = solve(schedule, hint, topo, data, ids={t.transfer_id for t in schedule.transfers})
    assert result.status is RepairStatus.SUCCESS
    assert max(t.ed_time for t in result.schedule.transfers) == pytest.approx(4.0)
    assert max(t.ed_time for t in greedy.schedule.transfers) == pytest.approx(9.0)
    assert result.evidence["objective_us"] == pytest.approx(4.0)
    assert result.evidence["buffer_status"] == "valid"
    assert verify_schedule_semantics(result.schedule, data).status is ValidationStatus.VALID


def test_v2_selected_edge_duration_and_parent_immutability():
    schedule, hint = broadcast_case()
    before = repr(schedule)
    result = solve(schedule, hint, topology(), inputs(CollectiveKind.BROADCAST, slices=2))
    assert result.status is RepairStatus.SUCCESS
    replacement = next(t for t in result.schedule.transfers if t.channel == 1)
    assert replacement.ed_time - replacement.st_time == pytest.approx(3.0)
    assert max(t.ed_time for t in result.schedule.transfers) == pytest.approx(3.0)
    assert result.evidence["objective_us"] == pytest.approx(3.0)
    assert result.schedule.transfers[0] == schedule.transfers[0]
    assert repr(schedule) == before


def test_v2_collective_replay_rejects_lost_broadcast_intermediate():
    schedule = send_relay_schedule()
    schedule = replace(schedule, transfers=(retime(schedule.transfers[0], 1.0, 2.0),
                                            retime(schedule.transfers[1], 2.0, 3.0)))
    hint = hint_for(schedule, "relay-first", (0, 2))
    result = solve(schedule, hint, topology(rank_count=3),
                   inputs(CollectiveKind.BROADCAST, ranks=3))
    assert result.status is RepairStatus.INVALID
    assert result.schedule is None
    assert result.evidence["semantic_status"] == "invalid"


def test_v2_merge_target_is_included_even_when_outside_supplied_impact():
    schedule = concurrent_reduce_star_schedule()
    schedule = replace(schedule, transfers=tuple(retime(t, 0.0, 10.0) for t in schedule.transfers))
    hint = hint_for(schedule, "reduce-star-2", (2, 1, 0))
    result = solve(schedule, hint, topology(rank_count=4),
                   inputs(CollectiveKind.REDUCE, ranks=4),
                   tuning=overlay())
    # A fixed unrelated 10-us transfer prevents a makespan improvement.
    assert result.status is RepairStatus.NOT_RUN
    assert "reduce-star-1" in result.evidence["decision_domain_transfer_ids"]
    assert result.evidence["code"] == "local_repair_no_improvement"


def test_v2_rejects_unsupported_manual_order():
    schedule, hint = broadcast_case()
    tuning = replace(overlay(), channel_count=2,
                     lane_order=(("lane-send-0", "lane-send-1"),))
    result = solve(schedule, hint, topology(), inputs(CollectiveKind.BROADCAST, slices=2), tuning=tuning)
    assert result.status is RepairStatus.NOT_RUN
    assert result.evidence["code"] == "local_repair_unsupported_restrictions"


def test_v2_resource_capacity_and_outside_order_are_preserved():
    schedule, hint = broadcast_case()
    metadata = dict(schedule.metadata)
    metadata["resource_slots"] = {t.transfer_id: {"nic": 0} for t in schedule.transfers}
    schedule = replace(schedule, metadata=metadata)
    topo = topology(shared=True)
    result = solve(schedule, hint, topo, inputs(CollectiveKind.BROADCAST, slices=2))
    assert result.status is RepairStatus.NOT_RUN
    assert result.evidence["code"] == "local_repair_no_improvement"
    assert result.evidence["objective_us"] == pytest.approx(6.0)


def test_v2_preserves_two_outside_transfers_relative_lane_order():
    schedule, hint = broadcast_case()
    first, second = schedule.transfers
    third = replace(second, transfer_id="lane-send-2", member_slice_ids=frozenset({2}),
                    atoms=(replace(second.atoms[0], slice_id=2),))
    third = retime(third, 3.0, 4.0)
    second = retime(second, 4.0, 5.0)
    metadata = dict(schedule.metadata)
    metadata["semantic_predecessors"] = dict(metadata["semantic_predecessors"], **{"lane-send-2": ()})
    outputs = dict(metadata["final_outputs"])
    outputs.update({"r00000000-o00000002": (2,), "r00000001-o00000002": (2,)})
    final = dict(metadata["final_dependencies"])
    final.update({"r00000000-o00000002": (), "r00000001-o00000002": ("lane-send-2",)})
    metadata.update(final_outputs=outputs, final_dependencies=final)
    schedule = replace(schedule, transfers=(first, second, third), metadata=metadata, slice_count=3,
                       final_state_ids=schedule.final_state_ids + ("final-r0-o2", "final-r1-o2"))
    hint = hint_for(schedule, "lane-send-1", (0, 1), 1)
    result = solve(schedule, hint, topology(), inputs(CollectiveKind.BROADCAST, slices=3))
    assert result.status is RepairStatus.SUCCESS
    by_id = {t.transfer_id: t for t in result.schedule.transfers}
    assert by_id["lane-send-0"].ed_time <= by_id["lane-send-2"].st_time
    assert by_id["lane-send-0"] == first
    assert by_id["lane-send-2"] == third
    assert ("lane-send-0", "lane-send-2") in result.evidence["preserved_outside_order_pairs"]


def test_v2_exact_forbidden_tuple_checks_unmodified_transfers():
    from vericcl.input.models import ForbiddenTransfer
    schedule, hint = broadcast_case()
    forbidden = ForbiddenTransfer(0, 0, 1, 0)
    tuning = replace(overlay(), channel_count=2, temporary_forbidden=frozenset({forbidden}))
    result = solve(schedule, hint, topology(), inputs(CollectiveKind.BROADCAST, slices=2), tuning=tuning)
    assert result.status is RepairStatus.INVALID
    assert result.schedule is None
    # Same physical edge but another member is not forbidden by this tuple.
    tuning = replace(tuning, temporary_forbidden=frozenset({replace(forbidden, slice_id=2)}))
    result = solve(schedule, hint, topology(), inputs(CollectiveKind.BROADCAST, slices=2), tuning=tuning)
    assert result.status is RepairStatus.SUCCESS


def test_v2_no_incumbent_retains_parent(monkeypatch):
    from vericcl.solver.scip_api import ScipAdapter, ScipSolveResult
    def no_incumbent(model, lease):
        model.result = ScipSolveResult('time_limit', 'timelimit', None, None, None, (), {}, 0., False, False)
        return model.result
    monkeypatch.setattr(ScipAdapter, 'solve', no_incumbent)
    schedule, hint = broadcast_case()
    before = repr(schedule)
    result = solve(schedule, hint, topology(), inputs(CollectiveKind.BROADCAST, slices=2))
    assert result.status is RepairStatus.TIMEOUT
    assert result.schedule is None
    assert repr(schedule) == before


def test_v2_size_limit_rejects_before_optimization(monkeypatch):
    monkeypatch.setattr("vericcl.tuning.local_milp.MAX_TRANSFERS", 1)
    schedule, hint = broadcast_case()
    result = solve(schedule, hint, topology(), inputs(CollectiveKind.BROADCAST, slices=2))
    assert result.status is RepairStatus.NOT_RUN
    assert result.evidence["code"] == "local_repair_model_size_limit"


def test_v2_rejects_buffer_serialization_that_changes_model_times():
    schedule = concurrent_reduce_star_schedule()
    schedule = replace(schedule, transfers=(
        schedule.transfers[0], retime(schedule.transfers[1], 10.0, 11.0),
        retime(schedule.transfers[2], 2.0, 3.0)))
    hint = hint_for(schedule, "reduce-star-2", (2, 1, 0))
    result = solve(schedule, hint, topology(rank_count=4),
                   inputs(CollectiveKind.REDUCE, ranks=4), tuning=overlay())
    assert result.status is RepairStatus.NOT_RUN
    assert result.schedule is None
    assert result.evidence["code"] == "local_repair_buffer_schedule_mismatch"


def test_v2_preserves_shared_broadcast_prefix_and_other_branch():
    schedule = send_relay_schedule()
    first, second = schedule.transfers
    atom = second.atoms[0]
    stage = atom.path[0]
    branch = replace(second, transfer_id="other-branch", dst_rank=3, atoms=(replace(
        atom, path=(replace(stage, symbols=(stage.symbols[0],
                      replace(stage.symbols[1], dst_rank=3))),)),))
    second = retime(second, 10.0, 11.0)
    metadata = dict(schedule.metadata)
    metadata["semantic_predecessors"] = dict(metadata["semantic_predecessors"], **{"other-branch": ("relay-first",)})
    metadata["final_outputs"] = dict(metadata["final_outputs"], **{"r00000003-o00000000": (0,)})
    metadata["final_dependencies"] = dict(metadata["final_dependencies"], **{"r00000003-o00000000": ("other-branch",)})
    schedule = replace(schedule, transfers=(first, second, branch), rank_count=4,
                       metadata=metadata, final_state_ids=schedule.final_state_ids + ("final-r3-o0",))
    hint = hint_for(schedule, "relay-second", (1, 2), 1)
    data, topo = inputs(CollectiveKind.BROADCAST, ranks=4), topology(rank_count=4)
    result = solve(schedule, hint, topo, data)
    assert result.status is RepairStatus.SUCCESS
    by_id = {t.transfer_id: t for t in result.schedule.transfers}
    assert by_id["relay-first"] == first
    assert by_id["other-branch"] == branch
    assert verify_schedule_constraints(result.schedule, data, topo).status is ValidationStatus.VALID
    assert verify_schedule_semantics(result.schedule, data).status is ValidationStatus.VALID


def observe_solver(monkeypatch, result_attributes=None):
    """Keep real model building/optimization; observe its boundary contract."""
    from dataclasses import replace
    from vericcl.solver.scip_api import ScipAdapter
    observed = {}
    original = ScipAdapter.solve
    attributes = result_attributes or {}
    def solve(model, lease):
        observed["relative_gap_limit"] = model.getParam("limits/gap")
        observed["absolute_gap_limit"] = model.getParam("limits/absgap")
        result = original(model, lease)
        updates = {}
        if "ObjBound" in attributes:
            updates["dual_bound"] = attributes["ObjBound"]
        if "MIPGap" in attributes:
            updates["relative_gap"] = attributes["MIPGap"]
        model.result = replace(result, **updates)
        return model.result
    monkeypatch.setattr(ScipAdapter, "solve", solve)
    return observed


@pytest.mark.parametrize("parameters", [(), (("MIPGap", 0.5), ("MIPGapAbs", 2.0))])
def test_v2_strict_optimality_forces_zero_gaps_after_overlay(monkeypatch, parameters):
    observed = observe_solver(monkeypatch)
    schedule, hint = broadcast_case()
    data = inputs(CollectiveKind.BROADCAST, slices=2)
    data = replace(data, solver=replace(data.solver, require_proven_optimal=True, mip_gap=0.2))
    tuning = replace(overlay(), channel_count=2, milp_parameters=parameters)
    result = solve(schedule, hint, topology(), data, tuning=tuning)
    assert result.status is RepairStatus.SUCCESS
    assert observed["relative_gap_limit"] == 0.0
    assert observed["absolute_gap_limit"] == 0.0
    assert result.evidence["proven_optimal"] is True


@pytest.mark.parametrize("attributes", [
    {"ObjBound": 2.9},
    {"ObjBound": 3.0, "IsMIP": True, "MIPGap": 0.1},
])
@pytest.mark.parametrize("strict", [False, True])
def test_v2_optimal_status_with_unclosed_gap_is_not_proof(monkeypatch, attributes, strict):
    observe_solver(monkeypatch, attributes)
    schedule, hint = broadcast_case()
    data = inputs(CollectiveKind.BROADCAST, slices=2)
    data = replace(data, solver=replace(data.solver, require_proven_optimal=strict))
    result = solve(schedule, hint, topology(), data)
    assert result.evidence["proven_optimal"] is False
    if strict:
        assert result.status is RepairStatus.TIMEOUT
        assert result.schedule is None
        assert result.evidence["code"] == "local_repair_optimality_required"
    else:
        assert result.status is RepairStatus.SUCCESS


@pytest.mark.parametrize("delta, proven", [(5e-10, True), (2e-9, False)])
def test_v2_optimality_bound_roundoff_tolerance_is_explicit(monkeypatch, delta, proven):
    observe_solver(monkeypatch, {"ObjBound": 3.0 - delta})
    schedule, hint = broadcast_case()
    result = solve(schedule, hint, topology(), inputs(CollectiveKind.BROADCAST, slices=2))
    assert result.evidence["proven_optimal"] is proven
    assert result.evidence["proof_absolute_tolerance_us"] == 1e-9
    assert result.evidence["proof_relative_tolerance"] == 1e-9
