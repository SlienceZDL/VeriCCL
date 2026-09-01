from dataclasses import replace
from pathlib import Path

import pytest

from vericcl.errors import SemanticError
from vericcl.input.loader import resolve_inputs
from vericcl.input.models import AtomConstraints, ForbiddenTransfer
from vericcl.planner.build import build_plan
from vericcl.planner.model import PlanNode, PlanningMode, StageInterface
from vericcl.semantics.collective import (
    CollectiveKind,
    CollectiveSpec,
    OutputSlot,
)
from vericcl.solver.demands import build_solver_problem
from vericcl.solver.templates import (
    build_solver_templates,
    split_routing_units,
)
from vericcl.topology.loader import topology_from_mapping


pytestmark = pytest.mark.phase03


EXAMPLES = Path(__file__).parents[3] / "vericcl" / "examples"


def _inputs(
    kind,
    *,
    rank_count,
    slice_count,
    root=None,
    forbidden=(),
    symmetry=True,
):
    base = resolve_inputs(
        EXAMPLES / "topo" / "two_rank.json",
        EXAMPLES / "sketch" / "allreduce_8m_1m.json",
        EXAMPLES / "atom" / "default.json",
    )
    hyperparameters = replace(
        base.hyperparameters,
        total_size_bytes=slice_count * 1024,
        slice_size_bytes=1024,
    )
    return replace(
        base,
        collective=CollectiveSpec(
            kind=kind,
            datatype="float32",
            root=root,
            reduction_op=(
                "sum"
                if kind
                in {
                    CollectiveKind.REDUCE,
                    CollectiveKind.REDUCE_SCATTER,
                }
                else None
            ),
        ),
        hyperparameters=hyperparameters,
        rank_count=rank_count,
        strategies=replace(
            base.strategies,
            hierarchy=False,
            symmetry=symmetry,
        ),
        atom_constraints=AtomConstraints(
            stage_num=None,
            forbidden_transfers=tuple(forbidden),
        ),
    )


def _complete_topology(rank_count, *, max_channels=4, invbw=2):
    return topology_from_mapping(
        {
            "ranks": rank_count,
            "nodes": [
                {
                    "id": 0,
                    "ranks": list(range(rank_count)),
                    "gateways": [],
                }
            ],
            "directed_links": [
                {
                    "src": src,
                    "dst": dst,
                    "alpha": 1,
                    "invbw": invbw,
                    "max_channels": max_channels,
                }
                for src in range(rank_count)
                for dst in range(rank_count)
                if src != dst
            ],
            "shared_resources": [],
        }
    )


def _plan_problems(inputs, topology):
    plan = build_plan(inputs, topology)
    return plan, tuple(
        build_solver_problem(node, inputs, topology) for node in plan.nodes
    )


def _paired_topology(change=None):
    links = []
    for group_index, (root, leaf) in enumerate(((0, 1), (2, 3))):
        for src, dst in ((root, leaf), (leaf, root)):
            item = {
                "src": src,
                "dst": dst,
                "alpha": 1,
                "invbw": 2,
                "max_channels": 4,
            }
            if change == "direction" and group_index == 1 and src == leaf:
                continue
            if change == "max_channels" and group_index == 1:
                item["max_channels"] = 2
            if change == "performance" and group_index == 1:
                item["invbw"] = 3
            if change == "shared_members" and (
                (group_index == 0 and src == root)
                or group_index == 1
            ):
                item["resources"] = ["fabric-{}".format(group_index)]
            links.append(item)
    resources = []
    if change == "shared_members":
        resources = [
            {
                "id": "fabric-0",
                "member_links": [[0, 1]],
                "alpha": 1,
                "invbw": 2,
                "max_channels": 4,
            },
            {
                "id": "fabric-1",
                "member_links": [[2, 3], [3, 2]],
                "alpha": 1,
                "invbw": 2,
                "max_channels": 4,
            },
        ]
    return topology_from_mapping(
        {
            "ranks": 4,
            "nodes": [
                {"id": 0, "ranks": [0, 1], "gateways": [0]},
                {"id": 1, "ranks": [2, 3], "gateways": [2]},
            ],
            "directed_links": links,
            "shared_resources": resources,
        }
    )


def _domain_problem(
    topology,
    group,
    *,
    root=None,
    contributors=None,
    reduction_dual=False,
):
    inputs = _inputs(
        CollectiveKind.BROADCAST,
        rank_count=4,
        slice_count=1,
        root=group[0],
    )
    root = group[0] if root is None else root
    leaf = next(rank for rank in group if rank != root)
    contributors = (
        frozenset({root}) if contributors is None else frozenset(contributors)
    )
    node = PlanNode(
        node_id="domain-{}".format(group[0]),
        stage_id=0,
        local_collective=CollectiveSpec(
            kind=CollectiveKind.BROADCAST,
            datatype="float32",
            root=root,
        ),
        communication_group=tuple(group),
        logical_input=StageInterface({OutputSlot(root, 0): contributors}),
        logical_output=StageInterface(
            {
                OutputSlot(root, 0): contributors,
                OutputSlot(leaf, 0): contributors,
            }
        ),
        allowed_links=frozenset(
            key
            for key in topology.links
            if key.src_rank in group and key.dst_rank in group
        ),
        shared_resource_ids=frozenset(
            resource_id
            for key, edge in topology.links.items()
            if key.src_rank in group and key.dst_rank in group
            for resource_id in edge.resource_ids
        ),
    )
    problem = build_solver_problem(node, inputs, topology)
    if reduction_dual:
        demand = replace(problem.demands[0], reduction_dual=True)
        problem = replace(problem, demands=(demand,))
    return problem


def _replace_problem_contributors(problem, contributors):
    contributors = frozenset(contributors)
    node_id = "{}-contributors".format(problem.node.node_id)
    node = replace(
        problem.node,
        node_id=node_id,
        logical_input=StageInterface(
            {
                slot: contributors
                for slot in problem.node.logical_input.values
            }
        ),
        logical_output=StageInterface(
            {
                slot: contributors
                for slot in problem.node.logical_output.values
            }
        ),
    )
    demands = tuple(
        replace(
            demand,
            demand_id="{}-contributors".format(demand.demand_id),
            node_id=node_id,
            contributors=contributors,
            member_slice_ids=contributors,
        )
        for demand in problem.demands
    )
    return replace(problem, node=node, demands=demands)


def test_direct_allgather_units_deduplicate_by_root_not_logical_position():
    inputs = _inputs(
        CollectiveKind.ALL_GATHER,
        rank_count=8,
        slice_count=128,
    )
    topology = _complete_topology(8)
    plan, problems = _plan_problems(inputs, topology)

    units = tuple(
        unit for problem in problems for unit in split_routing_units(problem)
    )
    templates = build_solver_templates(problems, plan.planning_mode)

    assert len(units) == 1024
    assert len(templates) == 8
    assert sum(len(template.members) for template in templates) == 1024
    assert {
        template.representative.demands[0].root_rank
        for template in templates
    } == set(range(8))


def test_same_root_logical_positions_use_identity_rank_mapping():
    inputs = _inputs(
        CollectiveKind.ALL_GATHER,
        rank_count=8,
        slice_count=128,
    )
    topology = _complete_topology(8)
    plan, problems = _plan_problems(inputs, topology)

    template = next(
        item
        for item in build_solver_templates(problems, plan.planning_mode)
        if item.representative.demands[0].root_rank == 0
    )

    assert len(template.members) == 128
    assert all(
        member.rank_map == tuple((rank, rank) for rank in range(8))
        for member in template.members
    )
    assert {member.logical_position_map for member in template.members} == {
        ((template.representative.demands[0].logical_position, position),)
        for position in range(128)
    }


def test_slice_specific_forbidden_transfer_splits_only_the_affected_unit():
    forbidden = ForbiddenTransfer(0, 0, 1, 0)
    inputs = _inputs(
        CollectiveKind.ALL_GATHER,
        rank_count=2,
        slice_count=2,
        forbidden=(forbidden,),
    )
    topology = _complete_topology(2)
    plan, problems = _plan_problems(inputs, topology)

    templates = build_solver_templates(problems, plan.planning_mode)
    affected = [
        template
        for template in templates
        if template.representative.demands[0].forbidden_members
    ]

    assert len(templates) == 3
    assert len(affected) == 1
    assert len(affected[0].members) == 1
    assert sum(len(template.members) for template in templates) == 4


def test_same_group_contributor_change_is_not_a_logical_position_mapping():
    topology = _paired_topology()
    first = _domain_problem(topology, (0, 1))
    second = _replace_problem_contributors(first, {1})

    templates = build_solver_templates(
        (first, second),
        PlanningMode.DIRECT,
    )

    assert len(templates) == 2


def test_external_contributor_change_is_not_an_arbitrary_mapping():
    topology = _paired_topology()
    first = _domain_problem(topology, (0, 1), contributors={2})
    second = _replace_problem_contributors(first, {3})

    templates = build_solver_templates(
        (first, second),
        PlanningMode.DIRECT,
    )

    assert len(templates) == 2


def test_cross_group_contributor_must_follow_verified_rank_bijection():
    topology = _paired_topology()
    first = _domain_problem(topology, (0, 1))
    second = _domain_problem(topology, (2, 3), contributors={3})

    templates = build_solver_templates(
        (first, second),
        PlanningMode.DIRECT,
    )

    assert len(templates) == 2


def test_same_group_owner_change_cannot_reuse_identity_rank_mapping():
    topology = _paired_topology()
    first = _domain_problem(topology, (0, 1))
    second = _domain_problem(topology, (0, 1), root=1)

    templates = build_solver_templates(
        (first, second),
        PlanningMode.DIRECT,
    )

    assert len(templates) == 2


@pytest.mark.parametrize(
    "change",
    ["direction", "max_channels", "performance", "shared_members"],
)
def test_physical_or_resource_difference_prevents_template_reuse(change):
    topology = _paired_topology(change)
    problems = (
        _domain_problem(topology, (0, 1)),
        _domain_problem(topology, (2, 3)),
    )

    assert len(build_solver_templates(problems, PlanningMode.DIRECT)) == 2


@pytest.mark.parametrize("change", ["root", "contributors", "reduction_dual"])
def test_semantic_role_difference_prevents_template_reuse(change):
    topology = _paired_topology()
    first = _domain_problem(topology, (0, 1))
    arguments = {}
    if change == "root":
        arguments["root"] = 3
    elif change == "contributors":
        arguments["contributors"] = {2, 3}
    else:
        arguments["reduction_dual"] = True
    second = _domain_problem(topology, (2, 3), **arguments)

    assert len(
        build_solver_templates((first, second), PlanningMode.DIRECT)
    ) == 2


def test_tree_and_chain_collectives_split_at_semantic_boundaries():
    topology = _complete_topology(3)
    cases = (
        (CollectiveKind.BROADCAST, 2),
        (CollectiveKind.ALL_GATHER, 2),
        (CollectiveKind.REDUCE, 2),
        (CollectiveKind.SCATTER, 2),
        (CollectiveKind.GATHER, 2),
        (CollectiveKind.ALL_TO_ALL, 2),
    )
    for kind, expected in cases:
        reduced = kind is CollectiveKind.REDUCE
        root = 0
        inputs = _inputs(
            kind,
            rank_count=3,
            slice_count=2,
            root=(
                root
                if kind
                in {
                    CollectiveKind.BROADCAST,
                    CollectiveKind.REDUCE,
                    CollectiveKind.SCATTER,
                    CollectiveKind.GATHER,
                }
                else None
            ),
        )
        if reduced:
            input_values = {
                OutputSlot(rank, position): frozenset({rank * 2 + position})
                for rank in range(3)
                for position in range(2)
            }
            output_values = {
                OutputSlot(root, position): frozenset(
                    rank * 2 + position for rank in range(3)
                )
                for position in range(2)
            }
        elif kind in {CollectiveKind.BROADCAST, CollectiveKind.ALL_GATHER}:
            input_values = {
                OutputSlot(root, position): frozenset({position})
                for position in range(2)
            }
            output_values = {
                OutputSlot(rank, position): frozenset({position})
                for rank in range(3)
                for position in range(2)
            }
        else:
            input_values = {
                OutputSlot(
                    position + 1
                    if kind is CollectiveKind.GATHER
                    else root,
                    position,
                ): frozenset({position})
                for position in range(2)
            }
            output_values = {
                OutputSlot(
                    root
                    if kind is CollectiveKind.GATHER
                    else position + 1,
                    position,
                ): frozenset({position})
                for position in range(2)
            }
        node = PlanNode(
            node_id="split-{}".format(kind.value),
            stage_id=0,
            local_collective=CollectiveSpec(
                kind=kind,
                datatype="float32",
                root=(
                    root
                    if kind
                    in {
                        CollectiveKind.BROADCAST,
                        CollectiveKind.REDUCE,
                        CollectiveKind.SCATTER,
                        CollectiveKind.GATHER,
                    }
                    else None
                ),
                reduction_op="sum" if reduced else None,
            ),
            communication_group=(0, 1, 2),
            logical_input=StageInterface(input_values),
            logical_output=StageInterface(output_values),
            allowed_links=frozenset(topology.links),
            shared_resource_ids=frozenset(),
        )
        problem = build_solver_problem(node, inputs, topology)
        units = split_routing_units(problem)

        assert len(units) == expected
        assert sorted(
            demand.demand_id for unit in units for demand in unit.demands
        ) == sorted(demand.demand_id for demand in problem.demands)
        assert len(
            {
                demand.demand_id
                for unit in units
                for demand in unit.demands
            }
        ) == len(problem.demands)
        if kind in {
            CollectiveKind.GATHER,
            CollectiveKind.SCATTER,
            CollectiveKind.ALL_TO_ALL,
        }:
            assert all(len(unit.demands) == 1 for unit in units)


def test_exact_deduplication_remains_enabled_without_symmetry_restriction():
    inputs = _inputs(
        CollectiveKind.ALL_GATHER,
        rank_count=2,
        slice_count=2,
        symmetry=False,
    )
    topology = _complete_topology(2)
    plan, problems = _plan_problems(inputs, topology)

    templates = build_solver_templates(problems, plan.planning_mode)

    assert len(templates) == 2


def test_planning_mode_and_search_restrictions_are_part_of_the_signature():
    topology = _paired_topology()
    first = _domain_problem(topology, (0, 1))
    second = _domain_problem(topology, (2, 3))

    direct = build_solver_templates((first, second), PlanningMode.DIRECT)
    manual = build_solver_templates((first, second), PlanningMode.MANUAL)
    restricted = build_solver_templates(
        (first, replace(second, restrictions=("shortest_paths",))),
        PlanningMode.DIRECT,
    )

    assert len(direct) == 1
    assert len(manual) == 1
    assert direct[0].exact_signature != manual[0].exact_signature
    assert len(restricted) == 2


def test_unproved_domain_isomorphism_keeps_units_independent(monkeypatch):
    topology = _paired_topology()
    problems = (
        _domain_problem(topology, (0, 1)),
        _domain_problem(topology, (2, 3)),
    )

    def bounded_failure(*_arguments):
        raise SemanticError(
            "domain isomorphism canonicalization limit exceeded"
        )

    monkeypatch.setattr(
        "vericcl.solver.templates.exact_domain_signature",
        bounded_failure,
    )

    assert len(build_solver_templates(problems, PlanningMode.DIRECT)) == 2
