from dataclasses import replace
from pathlib import Path

import pytest

from vericcl.errors import InputValidationError, SemanticError
from vericcl.input.loader import resolve_inputs
from vericcl.planner.build import build_plan
from vericcl.planner.hierarchy import (
    build_gateway_allreduce_plan,
    validate_manual_hierarchy,
)
from vericcl.planner.groups import CommunicationGroups
from vericcl.planner.model import PlanningMode
from vericcl.semantics.collective import CollectiveKind, CollectiveSpec
from vericcl.topology.loader import load_topology


pytestmark = pytest.mark.phase02


EXAMPLES = Path(__file__).parents[3] / "vericcl" / "examples"


def two_rank_inputs():
    inputs = resolve_inputs(
        EXAMPLES / "topo" / "two_rank.json",
        EXAMPLES / "sketch" / "allreduce_8m_1m.json",
        EXAMPLES / "atom" / "default.json",
    )
    return replace(
        inputs,
        hyperparameters=replace(
            inputs.hyperparameters,
            total_size_bytes=2,
            slice_size_bytes=1,
        ),
    )


def manual_allreduce():
    reduced = [
        [0, 0, [0, 2]],
        [1, 0, [1, 3]],
    ]
    return (
        {
            "node_id": "manual-rs",
            "stage_id": 0,
            "operator": "reduce_scatter",
            "communication_group": [0, 1],
            "logical_input": [
                [0, 0, [0]],
                [0, 1, [1]],
                [1, 0, [2]],
                [1, 1, [3]],
            ],
            "logical_output": reduced,
            "depends_on": [],
        },
        {
            "node_id": "manual-ag",
            "stage_id": 1,
            "operator": "allgather",
            "communication_group": [0, 1],
            "logical_input": reduced,
            "logical_output": [
                [0, 0, [0, 2]],
                [0, 1, [1, 3]],
                [1, 0, [0, 2]],
                [1, 1, [1, 3]],
            ],
            "depends_on": ["manual-rs"],
        },
    )


def with_manual(inputs, manual):
    return replace(
        inputs,
        strategies=replace(
            inputs.strategies,
            hierarchy=True,
            manual_hierarchy=manual,
        ),
    )


def test_manual_hierarchy_takes_precedence_and_builds_exact_plan():
    inputs = two_rank_inputs()
    topology = load_topology(inputs)
    manual = manual_allreduce()

    validate_manual_hierarchy(manual, topology, inputs.collective)
    plan = build_plan(with_manual(inputs, manual), topology)

    assert [node.node_id for node in plan.nodes] == ["manual-rs", "manual-ag"]
    assert [node.local_collective.kind for node in plan.nodes] == [
        CollectiveKind.REDUCE_SCATTER,
        CollectiveKind.ALL_GATHER,
    ]
    assert plan.planning_mode is PlanningMode.MANUAL
    assert plan.planning_reason == "manual_hierarchy"


def test_manual_hierarchy_rejects_nonexistent_communication_domain():
    inputs = resolve_inputs(
        EXAMPLES / "topo" / "two_node_gateway.json",
        EXAMPLES / "sketch" / "allreduce_8m_1m.json",
        EXAMPLES / "atom" / "default.json",
    )
    topology = load_topology(inputs)
    invalid = (
        {
            "node_id": "invalid-domain",
            "stage_id": 0,
            "operator": "reduce_scatter",
            "communication_group": [1, 5],
            "logical_input": [[1, 0, [0]], [5, 0, [8]]],
            "logical_output": [[1, 0, [0, 8]]],
            "depends_on": [],
        },
    )

    with pytest.raises(InputValidationError, match="connected"):
        validate_manual_hierarchy(invalid, topology, inputs.collective)


@pytest.mark.parametrize(
    "group",
    [[1, 0], [0, 0]],
)
def test_manual_hierarchy_rejects_unsorted_or_duplicate_ranks(group):
    inputs = two_rank_inputs()
    invalid = list(manual_allreduce())
    invalid[0] = dict(invalid[0], communication_group=group)

    with pytest.raises(InputValidationError, match="sorted and unique"):
        validate_manual_hierarchy(
            tuple(invalid),
            load_topology(inputs),
            inputs.collective,
        )


def test_manual_hierarchy_rejects_mismatched_adjacent_contributors():
    inputs = two_rank_inputs()
    invalid = list(manual_allreduce())
    second = dict(invalid[1])
    second["logical_input"] = [
        [0, 0, [0, 3]],
        [1, 0, [1, 3]],
    ]
    invalid[1] = second

    with pytest.raises(InputValidationError, match="contributors"):
        build_plan(
            with_manual(inputs, tuple(invalid)),
            load_topology(inputs),
        )


def test_manual_hierarchy_rejects_incorrect_final_interface():
    inputs = two_rank_inputs()
    invalid = list(manual_allreduce())
    second = dict(invalid[1])
    second["logical_output"] = second["logical_output"][:-1]
    invalid[1] = second

    with pytest.raises(SemanticError, match="final output"):
        build_plan(
            with_manual(inputs, tuple(invalid)),
            load_topology(inputs),
        )


def test_requested_stage_count_must_match_plan():
    inputs = two_rank_inputs()
    inputs = with_manual(inputs, manual_allreduce())
    inputs = replace(
        inputs,
        atom_constraints=replace(inputs.atom_constraints, stage_num=3),
    )

    with pytest.raises(InputValidationError, match="stage_num"):
        build_plan(inputs, load_topology(inputs))


def test_nonhierarchical_build_falls_back_to_direct_plan():
    inputs = two_rank_inputs()

    plan = build_plan(inputs, load_topology(inputs))

    assert len(plan.nodes) == 4
    assert plan.nodes[0].node_id.startswith("allreduce-rs")


def test_hierarchy_without_cross_node_group_falls_back_to_direct_plan():
    inputs = two_rank_inputs()
    inputs = replace(
        inputs,
        strategies=replace(inputs.strategies, hierarchy=True),
    )

    plan = build_plan(inputs, load_topology(inputs))

    assert plan.nodes[0].node_id.startswith("allreduce-rs")
    assert plan.planning_mode is PlanningMode.DIRECT
    assert plan.planning_reason == "no_eligible_gateway_domain"


def test_hierarchy_with_noncovering_gateway_domain_falls_back_to_direct_plan():
    inputs = resolve_inputs(
        EXAMPLES / "topo" / "two_node_gateway.json",
        EXAMPLES / "sketch" / "allreduce_8m_1m.json",
        EXAMPLES / "atom" / "default.json",
    )
    inputs = replace(
        inputs,
        strategies=replace(inputs.strategies, hierarchy=True),
    )
    base_topology = load_topology(inputs)
    topology = replace(
        base_topology,
        links={
            key: link
            for key, link in base_topology.links.items()
            if 7 not in (key.src_rank, key.dst_rank)
        },
        node_membership={
            rank: 0 if rank < 4 else 1 if rank < 7 else 2
            for rank in range(inputs.rank_count)
        },
        gateways=frozenset({0, 4, 7}),
        isomorphism_signature="",
    )

    plan = build_plan(inputs, topology)

    assert plan.planning_mode is PlanningMode.DIRECT
    assert plan.planning_reason == "no_eligible_gateway_domain"


def test_gateway_allreduce_records_eligible_gateway_domain():
    inputs = resolve_inputs(
        EXAMPLES / "topo" / "two_node_gateway.json",
        EXAMPLES / "sketch" / "allreduce_8m_1m.json",
        EXAMPLES / "atom" / "default.json",
    )
    inputs = replace(
        inputs,
        strategies=replace(inputs.strategies, hierarchy=True),
    )

    plan = build_plan(inputs, load_topology(inputs))

    assert plan.planning_mode is PlanningMode.GATEWAY_ALLREDUCE
    assert plan.planning_reason == "eligible_gateway_domain"


def test_build_plan_rejects_invalid_arguments_and_rank_mismatch():
    inputs = two_rank_inputs()
    topology = load_topology(inputs)

    with pytest.raises(InputValidationError, match="ResolvedInput"):
        build_plan({}, topology)
    with pytest.raises(InputValidationError, match="Topology"):
        build_plan(inputs, {})
    with pytest.raises(InputValidationError, match="rank counts"):
        build_plan(replace(inputs, rank_count=3), topology)


def test_manual_stage_ids_must_be_contiguous():
    inputs = two_rank_inputs()
    manual = list(manual_allreduce())
    manual[1] = dict(manual[1], stage_id=2)

    with pytest.raises(InputValidationError, match="contiguous"):
        build_plan(
            with_manual(inputs, tuple(manual)),
            load_topology(inputs),
        )


@pytest.mark.parametrize(
    "manual",
    [
        None,
        (),
        (1,),
        ({**manual_allreduce()[0], "node_id": ""},),
        ({**manual_allreduce()[0], "stage_id": -1},),
        ({**manual_allreduce()[0], "communication_group": []},),
        ({**manual_allreduce()[0], "communication_group": [0, 2]},),
        ({**manual_allreduce()[0], "operator": "scan"},),
        ({**manual_allreduce()[0], "root": 0},),
        ({**manual_allreduce()[0], "unexpected": True},),
    ],
)
def test_manual_hierarchy_rejects_invalid_node_shapes(manual):
    inputs = two_rank_inputs()

    with pytest.raises(InputValidationError):
        validate_manual_hierarchy(
            manual,
            load_topology(inputs),
            inputs.collective,
        )


@pytest.mark.parametrize(
    "logical_input",
    [
        [[0, 0]],
        [[2, 0, [0]]],
        [[0, 0, []]],
        [[0, 0, [0]], [0, 0, [1]]],
    ],
)
def test_manual_hierarchy_rejects_invalid_interfaces(logical_input):
    inputs = two_rank_inputs()
    node = dict(manual_allreduce()[0], logical_input=logical_input)

    with pytest.raises(InputValidationError):
        validate_manual_hierarchy(
            (node,),
            load_topology(inputs),
            inputs.collective,
        )


def test_manual_hierarchy_rejects_duplicate_nodes_and_dependencies():
    inputs = two_rank_inputs()
    first, second = manual_allreduce()
    duplicate_node = (first, dict(second, node_id=first["node_id"]))
    duplicate_dependency = (
        first,
        dict(second, depends_on=["manual-rs", "manual-rs"]),
    )

    with pytest.raises(InputValidationError, match="node IDs"):
        validate_manual_hierarchy(
            duplicate_node,
            load_topology(inputs),
            inputs.collective,
        )
    with pytest.raises(InputValidationError, match="dependencies"):
        validate_manual_hierarchy(
            duplicate_dependency,
            load_topology(inputs),
            inputs.collective,
        )


@pytest.mark.parametrize("dependency", ["missing", "manual-ag"])
def test_manual_hierarchy_rejects_unknown_or_self_dependency(dependency):
    inputs = two_rank_inputs()
    first, second = manual_allreduce()
    invalid = (first, dict(second, depends_on=[dependency]))

    with pytest.raises(InputValidationError, match="unknown|itself"):
        validate_manual_hierarchy(
            invalid,
            load_topology(inputs),
            inputs.collective,
        )


def test_manual_hierarchy_rejects_dependency_without_matching_values():
    inputs = two_rank_inputs()
    first, second = manual_allreduce()
    second = dict(
        second,
        logical_input=[[0, 1, [0, 2]]],
    )

    with pytest.raises(InputValidationError, match="interfaces do not match"):
        validate_manual_hierarchy(
            (first, second),
            load_topology(inputs),
            inputs.collective,
        )


def test_manual_hierarchy_rejects_dependency_cycle():
    inputs = two_rank_inputs()
    interface = [[0, 0, [0]]]
    cycle = (
        {
            "node_id": "a",
            "stage_id": 0,
            "operator": "allgather",
            "communication_group": [0],
            "logical_input": interface,
            "logical_output": interface,
            "depends_on": ["b"],
        },
        {
            "node_id": "b",
            "stage_id": 1,
            "operator": "allgather",
            "communication_group": [0],
            "logical_input": interface,
            "logical_output": interface,
            "depends_on": ["a"],
        },
    )

    with pytest.raises(InputValidationError, match="acyclic"):
        validate_manual_hierarchy(
            cycle,
            load_topology(inputs),
            inputs.collective,
        )


@pytest.mark.parametrize("operator", ["broadcast", "reduce"])
def test_manual_rooted_domains_validate_directional_reachability(operator):
    inputs = two_rank_inputs()
    interface = [[0, 0, [0]], [1, 0, [2]]]
    manual = (
        {
            "node_id": "rooted",
            "stage_id": 0,
            "operator": operator,
            "root": 0,
            "communication_group": [0, 1],
            "logical_input": interface,
            "logical_output": interface,
            "depends_on": [],
        },
    )

    validate_manual_hierarchy(
        manual,
        load_topology(inputs),
        inputs.collective,
    )


def test_manual_reduction_requires_global_reduction_operation():
    inputs = two_rank_inputs()
    non_reduction = CollectiveSpec(
        kind=CollectiveKind.ALL_GATHER,
        datatype="float32",
    )

    with pytest.raises(InputValidationError, match="global reduction"):
        validate_manual_hierarchy(
            (manual_allreduce()[0],),
            load_topology(inputs),
            non_reduction,
        )


def test_validate_manual_hierarchy_rejects_invalid_models():
    inputs = two_rank_inputs()

    with pytest.raises(InputValidationError, match="Topology"):
        validate_manual_hierarchy(manual_allreduce(), {}, inputs.collective)
    with pytest.raises(InputValidationError, match="CollectiveSpec"):
        validate_manual_hierarchy(
            manual_allreduce(),
            load_topology(inputs),
            {},
        )


def test_gateway_builder_rejects_wrong_collective_and_missing_group():
    inputs = two_rank_inputs()
    topology = load_topology(inputs)
    groups = CommunicationGroups(intra_node=((0, 1),), inter_node=())
    wrong = replace(
        inputs,
        collective=CollectiveSpec(
            kind=CollectiveKind.ALL_GATHER,
            datatype="float32",
        ),
    )

    with pytest.raises(InputValidationError, match="requires AllReduce"):
        build_gateway_allreduce_plan(wrong, topology, groups)
    with pytest.raises(InputValidationError, match="covers every node"):
        build_gateway_allreduce_plan(inputs, topology, groups)
