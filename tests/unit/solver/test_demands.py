from dataclasses import replace
from pathlib import Path

import pytest

from vericcl.input.loader import resolve_inputs
from vericcl.input.models import AtomConstraints, ForbiddenTransfer
from vericcl.errors import SemanticError
from vericcl.planner.build import build_plan
from vericcl.planner.model import PlanNode, StageInterface
from vericcl.semantics.collective import (
    CollectiveKind,
    CollectiveSpec,
    OutputSlot,
)
from vericcl.solver.demands import (
    CandidateEdge,
    SolverProblem,
    build_solver_problem,
)
from vericcl.topology.loader import load_topology
from vericcl.topology.model import (
    DirectedLink,
    LinkKey,
    PerformanceCurve,
    Topology,
)


pytestmark = pytest.mark.phase03


EXAMPLES = Path(__file__).parents[3] / "vericcl" / "examples"


def allreduce_problem(node_prefix, forbidden=()):
    inputs = resolve_inputs(
        EXAMPLES / "topo" / "two_rank.json",
        EXAMPLES / "sketch" / "allreduce_8m_1m.json",
        EXAMPLES / "atom" / "default.json",
    )
    inputs = replace(
        inputs,
        atom_constraints=AtomConstraints(
            stage_num=None,
            forbidden_transfers=tuple(forbidden),
        ),
    )
    topology = load_topology(inputs)
    plan = build_plan(inputs, topology)
    node = next(
        item for item in plan.nodes if item.node_id.startswith(node_prefix)
    )
    return build_solver_problem(node, inputs, topology)


def complete_problem(rank_count):
    inputs = resolve_inputs(
        EXAMPLES / "topo" / "two_rank.json",
        EXAMPLES / "sketch" / "allreduce_8m_1m.json",
        EXAMPLES / "atom" / "default.json",
    )
    inputs = replace(inputs, rank_count=rank_count)
    curve = PerformanceCurve(1.0, 2.0, {})
    links = {
        LinkKey(src, dst): DirectedLink(
            key=LinkKey(src, dst),
            max_channels=2,
            performance=curve,
            resource_ids=(),
        )
        for src in range(rank_count)
        for dst in range(rank_count)
        if src != dst
    }
    topology = Topology(
        rank_count=rank_count,
        links=links,
        shared_resources={},
        node_membership={rank: 0 for rank in range(rank_count)},
        gateways=frozenset(),
        warnings=(),
    )
    contributor = frozenset({0})
    node = PlanNode(
        node_id="bounded-paths",
        stage_id=0,
        local_collective=CollectiveSpec(
            kind=CollectiveKind.BROADCAST,
            datatype="float32",
            root=0,
        ),
        communication_group=tuple(range(rank_count)),
        logical_input=StageInterface({OutputSlot(0, 0): contributor}),
        logical_output=StageInterface(
            {
                OutputSlot(0, 0): contributor,
                OutputSlot(rank_count - 1, 0): contributor,
            }
        ),
        allowed_links=frozenset(links),
        shared_resource_ids=frozenset(),
    )
    return build_solver_problem(node, inputs, topology)


def test_broadcast_demand_excludes_local_root_output():
    problem = allreduce_problem("allreduce-ag-a00000000")

    assert len(problem.demands) == 1
    demand = problem.demands[0]
    assert demand.root_rank == 0
    assert demand.required_leaf_rank == 1
    assert demand.candidate_paths == ((0, 1),)
    assert demand.member_slice_ids == frozenset({0, 8})


def test_shared_transfer_is_removed_if_any_member_is_forbidden():
    forbidden = ForbiddenTransfer(
        slice_id=8,
        src_rank=0,
        dst_rank=1,
        stage_id=1,
    )
    problem = allreduce_problem(
        "allreduce-ag-a00000000",
        forbidden=(forbidden,),
    )

    assert CandidateEdge(0, 1, 0) not in problem.candidate_edges
    assert problem.infeasible_demand_ids == (
        "allreduce-ag-a00000000-a00000000-r00000000-l00000001-m0170e84cea21",
    )


def test_reduce_is_expanded_as_reverse_allgather_demands():
    problem = allreduce_problem("allreduce-rs-a00000000")

    assert problem.reduction_dual
    assert len(problem.demands) == 1
    demand = problem.demands[0]
    assert demand.root_rank == 0
    assert demand.required_leaf_rank == 1
    assert demand.member_slice_ids == frozenset({8})
    assert demand.candidate_paths == ((0, 1),)
    assert demand.physical_link(0, 1) == (1, 0)


def test_shortest_path_restriction_is_recorded():
    problem = allreduce_problem("allreduce-ag-a00000000")
    restricted_inputs = replace(
        problem.inputs,
        strategies=replace(problem.inputs.strategies, shortest_paths=True),
    )

    restricted = build_solver_problem(
        problem.node,
        restricted_inputs,
        problem.topology,
    )

    assert restricted.search_space_restricted
    assert "shortest_paths" in restricted.restrictions


def test_candidate_edges_include_only_legal_channel_indices():
    problem = allreduce_problem("allreduce-ag-a00000000")

    assert CandidateEdge(0, 1, 0) in problem.candidate_edges
    assert CandidateEdge(0, 1, 31) in problem.candidate_edges
    assert CandidateEdge(0, 1, 32) not in problem.candidate_edges


def test_constructive_path_seeds_are_bounded_without_pruning_milp_edges():
    problem = complete_problem(6)

    assert len(problem.demands[0].candidate_paths) <= 32
    assert CandidateEdge(2, 3, 0) in problem.candidate_edges


def test_allgather_and_alltoall_expand_to_source_leaf_chains():
    base = resolve_inputs(
        EXAMPLES / "topo" / "two_rank.json",
        EXAMPLES / "sketch" / "allreduce_8m_1m.json",
        EXAMPLES / "atom" / "default.json",
    )
    topology = load_topology(base)
    allgather_inputs = replace(
        base,
        collective=CollectiveSpec(
            kind=CollectiveKind.ALL_GATHER,
            datatype="float32",
        ),
    )
    allgather_plan = build_plan(allgather_inputs, topology)
    allgather_node = next(
        node
        for node in allgather_plan.nodes
        if node.node_id == "allgather-r00000001-a00000000"
    )
    allgather = build_solver_problem(
        allgather_node,
        allgather_inputs,
        topology,
    )
    assert (
        allgather.demands[0].root_rank,
        allgather.demands[0].required_leaf_rank,
    ) == (1, 0)

    alltoall_inputs = replace(
        base,
        collective=CollectiveSpec(
            kind=CollectiveKind.ALL_TO_ALL,
            datatype="float32",
        ),
    )
    alltoall_plan = build_plan(alltoall_inputs, topology)
    local_node = next(
        node
        for node in alltoall_plan.nodes
        if node.node_id == "alltoall-r00000000-a00000000"
    )
    remote_node = next(
        node
        for node in alltoall_plan.nodes
        if node.node_id == "alltoall-r00000000-a00000004"
    )
    assert not build_solver_problem(
        local_node,
        alltoall_inputs,
        topology,
    ).demands
    remote = build_solver_problem(remote_node, alltoall_inputs, topology)
    assert (
        remote.demands[0].root_rank,
        remote.demands[0].required_leaf_rank,
    ) == (0, 1)


def test_reduce_forbidden_item_uses_physical_reverse_direction():
    forbidden = ForbiddenTransfer(
        slice_id=8,
        src_rank=1,
        dst_rank=0,
        stage_id=0,
    )
    problem = allreduce_problem(
        "allreduce-rs-a00000000",
        forbidden=(forbidden,),
    )

    assert not problem.demands[0].candidate_paths
    assert CandidateEdge(0, 1, 0) not in problem.candidate_edges


@pytest.mark.parametrize(
    "field,value",
    [
        ("demand_id", ""),
        ("required_leaf_rank", 0),
        ("contributors", frozenset()),
        ("allowed_links", frozenset({1})),
        ("legal_links", frozenset({LinkKey(1, 0)})),
        ("forbidden_members", (object(),)),
        ("candidate_paths", ((1, 0),)),
        ("candidate_paths", ((0, 1, 0),)),
        ("reduction_dual", 1),
    ],
)
def test_transfer_demand_rejects_invalid_fields(field, value):
    demand = allreduce_problem("allreduce-ag-a00000000").demands[0]

    with pytest.raises(SemanticError):
        replace(demand, **{field: value})


@pytest.mark.parametrize(
    "arguments",
    [
        (-1, 1, 0),
        (0, 0, 0),
        (0, 1, -1),
        (True, 1, 0),
    ],
)
def test_candidate_edge_rejects_invalid_fields(arguments):
    with pytest.raises(SemanticError):
        CandidateEdge(*arguments)


def test_solver_problem_validates_exact_members():
    problem = allreduce_problem("allreduce-ag-a00000000")
    duplicate = replace(
        problem.demands[0],
        demand_id="duplicate",
    )

    with pytest.raises(SemanticError, match="unique"):
        replace(problem, demands=(duplicate, duplicate))
    with pytest.raises(SemanticError, match="unknown"):
        replace(problem, infeasible_demand_ids=("missing",))
    with pytest.raises(SemanticError, match="CandidateEdge"):
        replace(problem, candidate_edges=frozenset({object()}))


def test_monolithic_allreduce_node_must_be_decomposed():
    problem = allreduce_problem("allreduce-ag-a00000000")
    node = replace(
        problem.node,
        local_collective=CollectiveSpec(
            kind=CollectiveKind.ALL_REDUCE,
            datatype="float32",
            reduction_op="sum",
        ),
    )

    with pytest.raises(SemanticError, match="decomposed"):
        build_solver_problem(node, problem.inputs, problem.topology)
