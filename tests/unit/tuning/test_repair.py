from dataclasses import replace

import pytest

from vericcl.errors import SemanticError
from vericcl.input.models import ForbiddenTransfer
from vericcl.semantics.collective import CollectiveKind
from vericcl.solver.budget import ModelBudget
from vericcl.topology.model import (
    DirectedLink,
    LaneKey,
    LinkKey,
    PerformanceCurve,
    SharedResource,
    Topology,
)
from vericcl.tuning.impact import ImpactClosure, compute_impact_closure
from vericcl.tuning.local_milp import (
    _ordered_queues,
    _semantic_dependencies,
    solve_local_repair,
)
from vericcl.tuning.model import RepairStatus
from vericcl.tuning.repair import (
    _duration,
    _earliest_start,
    repair_flow_suffix,
)
from vericcl.verification.bdd_flow import FlowReplacementHint
from vericcl.verification.constraints import verify_schedule_constraints
from vericcl.verification.flow_index import build_flow_index
from vericcl.verification.model import ValidationStatus
from vericcl.verification.semantics import verify_schedule_semantics

from tests.unit.tuning.helpers import (
    aggregate_case,
    incomplete_leaf_hint,
    overlay,
    shared_prefix_case,
    waiting_case,
)
from tests.unit.verification.helpers import inputs as verification_inputs
from tests.unit.verification.helpers import topology as verification_topology
from tests.unit.xml.helpers import (
    concurrent_reduce_star_schedule,
    two_rank_allreduce_schedule,
)


pytestmark = pytest.mark.phase05


def test_suffix_repair_preserves_prefix_and_recomputes_deterministic_times():
    schedule, topology, inputs, hint = waiting_case()

    first = repair_flow_suffix(schedule, hint, overlay(), topology, inputs)
    second = repair_flow_suffix(schedule, hint, overlay(), topology, inputs)

    assert first.status is RepairStatus.SUCCESS
    assert first.schedule == second.schedule
    assert first.schedule is not schedule
    by_id = {item.transfer_id: item for item in first.schedule.transfers}
    assert by_id["wait-first"] == schedule.transfers[0]
    assert "wait-middle" not in by_id
    replacements = tuple(
        item
        for item in first.schedule.transfers
        if item.transfer_id.startswith("repair-")
    )
    assert tuple((item.src_rank, item.dst_rank) for item in replacements) == (
        (1, 2),
        (2, 3),
    )
    assert replacements[0].st_time == pytest.approx(1.0)
    assert replacements[1].st_time >= replacements[0].ed_time
    assert replacements[-1].dst_rank == 3
    assert hint.waiting_transfer_id in first.changed_transfer_ids
    assert verify_schedule_constraints(
        first.schedule,
        inputs,
        topology,
    ).status is ValidationStatus.VALID


def test_missing_leaf_delivery_is_completed_with_minimum_legal_suffix():
    schedule, topology, inputs, hint = waiting_case()

    result = repair_flow_suffix(
        schedule,
        incomplete_leaf_hint(hint),
        overlay(),
        topology,
        inputs,
    )

    assert result.status is RepairStatus.SUCCESS
    repaired_edges = tuple(
        (item.src_rank, item.dst_rank)
        for item in result.schedule.transfers
        if item.transfer_id.startswith("repair-")
    )
    assert repaired_edges == ((1, 2), (2, 3))
    assert result.evidence["leaf_repair_hops"] == 1


def test_repair_preserves_semantic_predecessors_and_transfer_metadata():
    schedule, topology, inputs, hint = waiting_case()
    metadata = dict(schedule.metadata)
    semantic = dict(metadata["semantic_predecessors"])
    semantic["wait-middle"] = ("cross-first", "wait-first")
    metadata["semantic_predecessors"] = semantic
    metadata["semantic_contributors"] = {
        transfer.transfer_id: tuple(sorted(transfer.member_slice_ids))
        for transfer in schedule.transfers
    }
    metadata["tree_contributors"] = {
        transfer.transfer_id: ("tree", transfer.transfer_id)
        for transfer in schedule.transfers
    }
    value = replace(schedule, metadata=metadata)

    result = repair_flow_suffix(value, hint, overlay(), topology, inputs)

    assert result.status is RepairStatus.SUCCESS
    new_ids = result.evidence["new_transfer_ids"]
    first_new = new_ids[0]
    assert set(result.schedule.metadata["semantic_predecessors"][first_new]) == {
        "cross-first",
        "wait-first",
    }
    assert "wait-middle" not in result.schedule.metadata["semantic_contributors"]
    assert "wait-middle" not in result.schedule.metadata["tree_contributors"]
    assert result.schedule.metadata["semantic_contributors"][first_new] == (0,)
    assert result.schedule.metadata["tree_contributors"][first_new] == (
        "tree",
        "wait-middle",
    )


def test_forbidden_candidate_is_rejected_without_mutating_parent():
    schedule, topology, inputs, hint = waiting_case()
    candidate_id = hint.candidate_flow_ids[0]
    forbidden = ForbiddenTransfer(0, 1, 2, 0)
    blocked_inputs = replace(
        inputs,
        atom_constraints=replace(
            inputs.atom_constraints,
            forbidden_transfers=(forbidden,),
        ),
    )
    blocked_overlay = replace(
        overlay(),
        temporary_forbidden=frozenset({forbidden}),
    )
    before = repr(schedule)

    user_result = repair_flow_suffix(
        schedule,
        hint,
        overlay(),
        topology,
        blocked_inputs,
    )
    overlay_result = repair_flow_suffix(
        schedule,
        hint,
        blocked_overlay,
        topology,
        inputs,
    )

    assert candidate_id
    assert user_result.status is RepairStatus.INFEASIBLE
    assert overlay_result.status is RepairStatus.INFEASIBLE
    assert user_result.schedule is None
    assert repr(schedule) == before


@pytest.mark.parametrize(
    "path",
    (
        (2, 3),
        (1, 2, 1),
    ),
)
def test_invalid_candidate_path_is_rejected(path):
    schedule, topology, inputs, hint = waiting_case()
    candidate_id = hint.candidate_flow_ids[0]
    invalid = replace(
        hint,
        candidate_paths={candidate_id: path},
    )

    result = repair_flow_suffix(schedule, invalid, overlay(), topology, inputs)

    assert result.status is RepairStatus.INFEASIBLE


def test_unrepairable_leaf_and_invalid_candidate_lane_are_rejected():
    schedule, topology, inputs, hint = waiting_case()
    candidate_id = hint.candidate_flow_ids[0]
    links = dict(topology.links)
    del links[LinkKey(2, 3)]
    disconnected = replace(topology, links=links)
    no_leaf = repair_flow_suffix(
        schedule,
        incomplete_leaf_hint(hint),
        overlay(),
        disconnected,
        inputs,
    )
    bad_lane_hint = replace(
        hint,
        candidate_first_lanes={candidate_id: LaneKey(2, 3, 0)},
    )
    bad_lane = repair_flow_suffix(
        schedule,
        bad_lane_hint,
        overlay(),
        topology,
        inputs,
    )

    assert no_leaf.status is RepairStatus.INFEASIBLE
    assert bad_lane.status is RepairStatus.INFEASIBLE


def test_candidate_must_leave_the_waiting_bottleneck_lane():
    schedule, topology, inputs, hint = waiting_case()
    candidate_id = hint.candidate_flow_ids[0]
    unchanged_lane = replace(
        hint,
        candidate_paths={candidate_id: (1, 3)},
        candidate_first_lanes={candidate_id: hint.bottleneck_lane},
    )

    result = repair_flow_suffix(
        schedule,
        unchanged_lane,
        overlay(),
        topology,
        inputs,
    )

    assert result.status is RepairStatus.INFEASIBLE
    assert result.evidence["code"] == "no_legal_suffix"


def test_repair_cost_uses_calibrated_shared_resource_and_lane_windows():
    curve = PerformanceCurve(1.0, 2.0, {1: 1024.0})
    key = LinkKey(0, 1)
    topology = Topology(
        rank_count=2,
        links={key: DirectedLink(key, 1, curve, ("nic",))},
        shared_resources={
            "nic": SharedResource("nic", (key,), 1, curve),
        },
        node_membership={0: 0, 1: 0},
        gateways=frozenset(),
        warnings=(),
    )

    assert _duration(topology, key, 1024, 1) == pytest.approx(2.0)
    assert _earliest_start(((0.0, 1.0), (3.0, 4.0)), 1.0, 1.0) == 1.0
    assert _earliest_start(((0.0, 2.0),), 1.0, 1.0) == 2.0


def test_aggregate_members_and_shared_suffix_are_repaired_without_duplication():
    schedule, topology, inputs, hint = aggregate_case()

    result = repair_flow_suffix(schedule, hint, overlay(), topology, inputs)

    assert result.status is RepairStatus.SUCCESS
    shared = tuple(
        item for item in result.schedule.transfers if item.transfer_id == "tx-shared"
    )
    assert len(shared) == 1
    assert shared[0].member_slice_ids == frozenset({0, 1})
    member_zero = next(atom for atom in shared[0].atoms if atom.slice_id == 0)
    stage_zero = next(stage for stage in member_zero.path if stage.stage_id == 0)
    assert tuple(
        (symbol.src_rank, symbol.dst_rank) for symbol in stage_zero.symbols
    ) == ((0, 2),)
    index = build_flow_index(result.schedule)
    assert "tx-shared" in index.shared_suffix_transfer_ids


def test_complete_allreduce_repair_replays_exact_collective_semantics():
    schedule = two_rank_allreduce_schedule()
    topology = verification_topology()
    inputs = verification_inputs()
    flow = next(
        item
        for item in build_flow_index(schedule).flows
        if item.stage_id == 0
    )
    candidate_id = "reduce-channel-one"
    hint = FlowReplacementHint(
        source_flow_id=flow.flow_id,
        demand_id=flow.demand_id,
        candidate_flow_ids=(candidate_id,),
        candidate_paths={candidate_id: (1, 0)},
        candidate_first_lanes={candidate_id: LaneKey(1, 0, 1)},
        divergence_rank=1,
        waiting_transfer_id="allreduce-reduce",
        bottleneck_lane=LaneKey(1, 0, 0),
        wait_start_us=0.0,
        wait_end_us=0.0,
        earliest_candidate_start_us=0.0,
    )

    result = repair_flow_suffix(
        schedule,
        hint,
        replace(overlay(), channel_count=2),
        topology,
        inputs,
    )

    assert result.status is RepairStatus.SUCCESS
    assert verify_schedule_semantics(
        result.schedule,
        inputs,
    ).status is ValidationStatus.VALID
    assert result.evidence["semantic_status"] == "valid"


def test_reduce_reroute_merges_into_existing_aggregate_suffix():
    schedule = concurrent_reduce_star_schedule()
    topology = verification_topology(rank_count=4)
    inputs = verification_inputs(
        CollectiveKind.REDUCE,
        ranks=4,
        slices=1,
    )
    flow = next(
        item
        for item in build_flow_index(schedule).flows
        if item.member_slice_ids == frozenset({2})
    )
    candidate_id = "reduce-via-rank-one"
    hint = FlowReplacementHint(
        source_flow_id=flow.flow_id,
        demand_id=flow.demand_id,
        candidate_flow_ids=(candidate_id,),
        candidate_paths={candidate_id: (2, 1, 0)},
        candidate_first_lanes={candidate_id: LaneKey(2, 1, 0)},
        divergence_rank=2,
        waiting_transfer_id="reduce-star-2",
        bottleneck_lane=LaneKey(2, 0, 0),
        wait_start_us=0.0,
        wait_end_us=1.0,
        earliest_candidate_start_us=0.0,
    )

    result = repair_flow_suffix(
        schedule,
        hint,
        overlay(),
        topology,
        inputs,
    )

    assert result.status is RepairStatus.SUCCESS
    by_id = {
        transfer.transfer_id: transfer
        for transfer in result.schedule.transfers
    }
    assert "reduce-star-2" not in by_id
    assert by_id["reduce-star-1"].member_slice_ids == frozenset({1, 2})
    assert {
        atom.slice_id for atom in by_id["reduce-star-1"].atoms
    } == {1, 2}
    new_id = result.evidence["new_transfer_ids"][0]
    assert by_id[new_id].src_rank == 2
    assert by_id[new_id].dst_rank == 1
    assert new_id in result.schedule.metadata["semantic_predecessors"][
        "reduce-star-1"
    ]
    assert verify_schedule_semantics(
        result.schedule,
        inputs,
    ).status is ValidationStatus.VALID


def test_reduce_reroute_without_existing_aggregate_tail_is_rejected():
    schedule = concurrent_reduce_star_schedule()
    topology = verification_topology(rank_count=4)
    inputs = verification_inputs(
        CollectiveKind.REDUCE,
        ranks=4,
        slices=1,
    )
    flow = next(
        item
        for item in build_flow_index(schedule).flows
        if item.member_slice_ids == frozenset({2})
    )
    candidate_id = "unsafe-reduce-tail"
    hint = FlowReplacementHint(
        source_flow_id=flow.flow_id,
        demand_id=flow.demand_id,
        candidate_flow_ids=(candidate_id,),
        candidate_paths={candidate_id: (2, 1, 3, 0)},
        candidate_first_lanes={candidate_id: LaneKey(2, 1, 0)},
        divergence_rank=2,
        waiting_transfer_id="reduce-star-2",
        bottleneck_lane=LaneKey(2, 0, 0),
        wait_start_us=0.0,
        wait_end_us=1.0,
        earliest_candidate_start_us=0.0,
    )

    result = repair_flow_suffix(
        schedule,
        hint,
        overlay(),
        topology,
        inputs,
    )

    assert result.status is RepairStatus.INFEASIBLE
    assert result.evidence["code"] == "no_legal_suffix"


def test_suffix_referenced_by_another_flow_is_preserved():
    schedule, topology, inputs, hint = shared_prefix_case()

    result = repair_flow_suffix(schedule, hint, overlay(), topology, inputs)

    assert result.status is RepairStatus.SUCCESS
    transfer_ids = {
        transfer.transfer_id for transfer in result.schedule.transfers
    }
    assert "wait-middle" in transfer_ids
    assert "wait-branch-5" in transfer_ids
    assert "wait-branch-4" not in transfer_ids
    assert tuple(
        (transfer.src_rank, transfer.dst_rank)
        for transfer in result.schedule.transfers
        if transfer.transfer_id.startswith("repair-")
    ) == ((1, 2), (2, 4))


def test_zero_local_model_budget_returns_timeout_without_global_solve():
    schedule, topology, inputs, hint = waiting_case()
    impact = compute_impact_closure(
        schedule,
        frozenset({hint.waiting_transfer_id}),
        topology,
    )

    result = solve_local_repair(
        schedule,
        hint,
        impact,
        overlay(),
        topology,
        inputs,
        ModelBudget(0.0, 1.0, 1.0),
    )

    assert result.status is RepairStatus.TIMEOUT
    assert result.schedule is None


def test_local_model_rejects_invalid_inputs_and_empty_candidates(monkeypatch):
    schedule, topology, inputs, hint = waiting_case()
    impact = compute_impact_closure(
        schedule,
        frozenset({hint.waiting_transfer_id}),
        topology,
    )
    arguments = (
        schedule,
        hint,
        impact,
        overlay(),
        topology,
        inputs,
        ModelBudget(1.0, 0.0, 1.0),
    )
    for index in range(len(arguments)):
        invalid = list(arguments)
        invalid[index] = object()
        with pytest.raises(SemanticError):
            solve_local_repair(*invalid)

    unknown_impact = ImpactClosure(
        frozenset({"missing"}),
        frozenset({"missing"}),
        {"missing": frozenset({"seed"})},
    )
    with pytest.raises(SemanticError, match="unknown transfer"):
        solve_local_repair(
            schedule,
            hint,
            unknown_impact,
            overlay(),
            topology,
            inputs,
            ModelBudget(1.0, 0.0, 1.0),
        )

    empty = replace(
        hint,
        candidate_flow_ids=(),
        candidate_paths={},
        candidate_first_lanes={},
    )
    result = solve_local_repair(
        schedule,
        empty,
        impact,
        overlay(),
        topology,
        inputs,
        ModelBudget(1.0, 0.0, 1.0),
    )
    assert result.status is RepairStatus.INFEASIBLE

    def unavailable():
        raise RuntimeError("solver unavailable")

    monkeypatch.setattr(
        "vericcl.tuning.local_milp.GurobiAdapter.require",
        unavailable,
    )
    unavailable_result = solve_local_repair(
        schedule,
        hint,
        impact,
        overlay(),
        topology,
        inputs,
        ModelBudget(1.0, 0.0, 1.0),
    )
    assert unavailable_result.status is RepairStatus.NOT_RUN
    assert unavailable_result.evidence["scope"] == "local"


def test_local_model_rejects_malformed_schedule_metadata():
    schedule, topology, inputs, hint = waiting_case()

    with pytest.raises(SemanticError, match="resource_slots"):
        _ordered_queues(
            replace(schedule, metadata={"resource_slots": ()})
        )
    with pytest.raises(SemanticError, match="slot assignment"):
        _ordered_queues(
            replace(
                schedule,
                metadata={
                    "resource_slots": {
                        schedule.transfers[0].transfer_id: (),
                    },
                },
            )
        )
    with pytest.raises(SemanticError, match="semantic_predecessors"):
        _semantic_dependencies(
            replace(schedule, metadata={"semantic_predecessors": ()})
        )
    with pytest.raises(SemanticError, match="must be iterable"):
        _semantic_dependencies(
            replace(
                schedule,
                metadata={
                    "semantic_predecessors": {
                        schedule.transfers[0].transfer_id: 1,
                    },
                },
            )
        )
    with pytest.raises(SemanticError, match="predecessor is missing"):
        _semantic_dependencies(
            replace(
                schedule,
                metadata={
                    "semantic_predecessors": {
                        schedule.transfers[0].transfer_id: ("missing",),
                    },
                },
            )
        )

    malformed = replace(schedule, metadata={"resource_slots": ()})
    impact = compute_impact_closure(
        schedule,
        frozenset({hint.waiting_transfer_id}),
        topology,
    )
    result = solve_local_repair(
        malformed,
        hint,
        impact,
        overlay(),
        topology,
        inputs,
        ModelBudget(1.0, 0.0, 1.0),
    )
    assert result.status is RepairStatus.INVALID
    assert result.evidence["code"] == "local_repair_invalid"
