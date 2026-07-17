from dataclasses import replace
from pathlib import Path

import pytest

from vericcl.errors import ConstructionInfeasibleError, SemanticError
from vericcl.input.loader import resolve_inputs
from vericcl.planner.model import PlanNode, StageInterface
from vericcl.semantics.collective import (
    CollectiveKind,
    CollectiveSpec,
    OutputSlot,
)
from vericcl.solver.constructive import construct_candidate
from vericcl.solver.demands import build_solver_problem
from vericcl.topology.loader import load_topology
from vericcl.topology.model import (
    DirectedLink,
    LinkKey,
    PerformanceCurve,
    SharedResource,
    Topology,
)


pytestmark = pytest.mark.phase03


EXAMPLES = Path(__file__).parents[3] / "vericcl" / "examples"


def base_inputs():
    return resolve_inputs(
        EXAMPLES / "topo" / "two_rank.json",
        EXAMPLES / "sketch" / "allreduce_8m_1m.json",
        EXAMPLES / "atom" / "default.json",
    )


def topology(rank_count, edges):
    curve = PerformanceCurve(
        alpha_us=1.0,
        invbw_us=2.0,
        bandwidth_bytes_per_us={},
    )
    links = {
        LinkKey(src, dst): DirectedLink(
            key=LinkKey(src, dst),
            max_channels=4,
            performance=curve,
            resource_ids=(),
        )
        for src, dst in edges
    }
    return Topology(
        rank_count=rank_count,
        links=links,
        shared_resources={},
        node_membership={rank: 0 for rank in range(rank_count)},
        gateways=frozenset(),
        warnings=(),
    )


def broadcast_node(rank_count, logical_positions, edges):
    group = tuple(range(rank_count))
    logical_input = {}
    logical_output = {}
    for logical_position in logical_positions:
        contributor = logical_position
        logical_input[OutputSlot(0, logical_position)] = frozenset(
            {contributor}
        )
        for rank in group:
            logical_output[OutputSlot(rank, logical_position)] = frozenset(
                {contributor}
            )
    return PlanNode(
        node_id="broadcast",
        stage_id=0,
        local_collective=CollectiveSpec(
            kind=CollectiveKind.BROADCAST,
            datatype="float32",
            root=0,
        ),
        communication_group=group,
        logical_input=StageInterface(logical_input),
        logical_output=StageInterface(logical_output),
        allowed_links=frozenset(LinkKey(*edge) for edge in edges),
        shared_resource_ids=frozenset(),
    )


def problem_for(node, selected_topology):
    inputs = base_inputs()
    inputs = replace(
        inputs,
        rank_count=selected_topology.rank_count,
        hyperparameters=replace(
            inputs.hyperparameters,
            total_size_bytes=8 * inputs.hyperparameters.slice_size_bytes,
        ),
    )
    return build_solver_problem(node, inputs, selected_topology)


def test_complete_broadcast_state_can_branch():
    edges = ((0, 1), (0, 2), (0, 3))
    selected_topology = topology(4, edges)
    node = broadcast_node(4, (0,), edges)

    schedule = construct_candidate(
        problem_for(node, selected_topology),
        channel_count=2,
    )

    assert {
        transfer.dst_rank
        for transfer in schedule.transfers
        if transfer.src_rank == 0
    } == {1, 2, 3}
    assert {transfer.st_time for transfer in schedule.transfers} == {0.0}


def test_constructive_schedule_respects_lane_order():
    edges = ((0, 1),)
    selected_topology = topology(2, edges)
    node = broadcast_node(2, (0, 1), edges)

    schedule = construct_candidate(
        problem_for(node, selected_topology),
        channel_count=1,
    )
    intervals = sorted(
        (transfer.st_time, transfer.ed_time)
        for transfer in schedule.transfers
    )

    assert len(intervals) == 2
    assert intervals[0][1] <= intervals[1][0]
    assert schedule.transfers[0].transfer_id in schedule.transfers[1].predecessor_ids


def test_reduce_dual_does_not_create_root_self_transfer():
    inputs = base_inputs()
    selected_topology = load_topology(inputs)
    from vericcl.planner.build import build_plan

    plan = build_plan(inputs, selected_topology)
    node = next(
        item
        for item in plan.nodes
        if item.node_id == "allreduce-rs-a00000000"
    )
    schedule = construct_candidate(
        build_solver_problem(node, inputs, selected_topology),
        channel_count=1,
    )

    assert schedule.transfers
    assert all(
        transfer.src_rank != transfer.dst_rank
        for transfer in schedule.transfers
    )
    assert schedule.transfers[0].member_slice_ids == frozenset({8})
    assert {atom.slice_id for atom in schedule.transfers[0].atoms} == {8}
    assert schedule.metadata["tree_contributors"][
        schedule.transfers[0].transfer_id
    ] == (0, 8)
    assert schedule.metadata["reduction_dual"]


def test_aggregate_transfer_keeps_one_atom_per_contributor():
    inputs = base_inputs()
    selected_topology = load_topology(inputs)
    from vericcl.planner.build import build_plan

    plan = build_plan(inputs, selected_topology)
    node = next(
        item
        for item in plan.nodes
        if item.node_id == "allreduce-ag-a00000000"
    )
    schedule = construct_candidate(
        build_solver_problem(node, inputs, selected_topology),
        channel_count=1,
    )

    assert schedule.transfers[0].member_slice_ids == frozenset({0, 8})
    assert {atom.slice_id for atom in schedule.transfers[0].atoms} == {0, 8}
    assert schedule.metadata["path_scope"] == "stage_suffix"


def test_constructive_schedule_is_deterministic():
    edges = ((0, 1), (0, 2), (1, 2), (2, 1))
    selected_topology = topology(3, edges)
    node = broadcast_node(3, (0, 1), edges)
    problem = problem_for(node, selected_topology)

    assert construct_candidate(problem, 2) == construct_candidate(problem, 2)


def test_shared_tree_prefix_is_transmitted_once():
    edges = ((0, 1), (1, 2), (1, 3))
    selected_topology = topology(4, edges)
    node = broadcast_node(4, (0,), edges)

    schedule = construct_candidate(
        problem_for(node, selected_topology),
        channel_count=1,
    )

    assert len(schedule.transfers) == 3
    prefix = next(
        transfer
        for transfer in schedule.transfers
        if (transfer.src_rank, transfer.dst_rank) == (0, 1)
    )
    suffixes = [
        transfer
        for transfer in schedule.transfers
        if transfer.src_rank == 1
    ]
    assert all(prefix.transfer_id in item.predecessor_ids for item in suffixes)
    assert all(item.st_time >= prefix.ed_time for item in suffixes)


def test_two_channels_allow_same_link_payloads_to_start_together():
    edges = ((0, 1),)
    selected_topology = topology(2, edges)
    node = broadcast_node(2, (0, 1), edges)

    schedule = construct_candidate(
        problem_for(node, selected_topology),
        channel_count=2,
    )

    assert {transfer.channel for transfer in schedule.transfers} == {0, 1}
    assert {transfer.st_time for transfer in schedule.transfers} == {0.0}


def test_constructive_rejects_infeasible_or_invalid_requests():
    edges = ((0, 1),)
    selected_topology = topology(2, edges)
    node = broadcast_node(2, (0,), edges)
    problem = problem_for(node, selected_topology)

    with pytest.raises(SemanticError):
        construct_candidate(object(), 1)
    with pytest.raises(SemanticError):
        construct_candidate(problem, 0)
    with pytest.raises(SemanticError, match="maximum"):
        construct_candidate(problem, 33)
    with pytest.raises(ConstructionInfeasibleError, match="no legal path"):
        construct_candidate(
            replace(
                problem,
                infeasible_demand_ids=(problem.demands[0].demand_id,),
            ),
            1,
        )
    with pytest.raises(ConstructionInfeasibleError, match="no legal channel"):
        construct_candidate(
            replace(problem, candidate_edges=frozenset()),
            1,
        )


def test_shared_resource_serializes_different_links_in_the_same_slot():
    curve = PerformanceCurve(1.0, 2.0, {})
    keys = (LinkKey(0, 1), LinkKey(0, 2))
    links = {
        key: DirectedLink(
            key=key,
            max_channels=1,
            performance=curve,
            resource_ids=("shared-egress",),
        )
        for key in keys
    }
    resource = SharedResource(
        resource_id="shared-egress",
        member_links=keys,
        max_channels=1,
        performance=curve,
    )
    selected_topology = Topology(
        rank_count=3,
        links=links,
        shared_resources={resource.resource_id: resource},
        node_membership={0: 0, 1: 0, 2: 0},
        gateways=frozenset(),
        warnings=(),
    )
    node = broadcast_node(3, (0,), tuple((key.src_rank, key.dst_rank) for key in keys))

    schedule = construct_candidate(
        problem_for(node, selected_topology),
        channel_count=1,
    )
    transfers = sorted(schedule.transfers, key=lambda item: item.st_time)

    assert transfers[0].ed_time <= transfers[1].st_time
    assert transfers[0].transfer_id in transfers[1].predecessor_ids


def test_batching_reuses_one_tree_for_up_to_channel_count_payloads():
    direct_curve = PerformanceCurve(0.0, 4.0, {})
    alternate_curve = PerformanceCurve(0.0, 1.25, {})
    links = {}
    for key, channels, curve in (
        (LinkKey(0, 2), 1, direct_curve),
        (LinkKey(0, 1), 2, alternate_curve),
        (LinkKey(1, 2), 2, alternate_curve),
    ):
        links[key] = DirectedLink(
            key=key,
            max_channels=channels,
            performance=curve,
            resource_ids=(),
        )
    selected_topology = Topology(
        rank_count=3,
        links=links,
        shared_resources={},
        node_membership={0: 0, 1: 0, 2: 0},
        gateways=frozenset(),
        warnings=(),
    )
    node = broadcast_node(
        3,
        (0, 1),
        tuple((key.src_rank, key.dst_rank) for key in links),
    )
    problem = problem_for(node, selected_topology)
    problem = replace(
        problem,
        inputs=replace(
            problem.inputs,
            strategies=replace(problem.inputs.strategies, batching=True),
        ),
        restrictions=("batching",),
    )

    schedule = construct_candidate(problem, channel_count=2)
    selected_paths = tuple(schedule.metadata["selected_paths"].values())

    assert selected_paths == ((0, 1), (0, 2), (0, 1), (0, 2))
    assert tuple(schedule.metadata["demand_batches"].values()) == (0, 0, 0, 0)
