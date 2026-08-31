from dataclasses import replace
from pathlib import Path

import pytest

from vericcl.input.loader import resolve_inputs
from vericcl.planner.build import build_plan
from vericcl.planner.model import PlanningMode
from vericcl.semantics.collective import (
    CollectiveKind,
    CollectiveSpec,
    OutputSlot,
    required_outputs,
)
from vericcl.topology.loader import load_topology, topology_from_mapping
from vericcl.topology.model import LinkKey


pytestmark = pytest.mark.phase02


EXAMPLES = Path(__file__).parents[2] / "vericcl" / "examples"


def hierarchical_allgather_inputs():
    inputs = resolve_inputs(
        EXAMPLES / "topo" / "two_node_gateway.json",
        EXAMPLES / "sketch" / "allreduce_8m_1m.json",
        EXAMPLES / "atom" / "default.json",
    )
    return replace(
        inputs,
        collective=CollectiveSpec(
            kind=CollectiveKind.ALL_GATHER,
            datatype="float32",
        ),
        strategies=replace(inputs.strategies, hierarchy=True),
    )


def four_rail_topology():
    return topology_from_mapping(
        {
            "ranks": 8,
            "nodes": [
                {
                    "id": 0,
                    "ranks": [0, 1, 2, 3],
                    "gateways": [0, 1, 2, 3],
                },
                {
                    "id": 1,
                    "ranks": [4, 5, 6, 7],
                    "gateways": [4, 5, 6, 7],
                },
            ],
            "directed_links": [
                {
                    "src": src,
                    "dst": dst,
                    "alpha": 1,
                    "invbw": 2,
                    "max_channels": 8,
                }
                for base in (0, 4)
                for src in range(base, base + 4)
                for dst in range(base, base + 4)
                if src != dst
            ]
            + [
                {
                    "src": src,
                    "dst": dst,
                    "alpha": 2,
                    "invbw": 3,
                    "max_channels": 4,
                }
                for rail in range(4)
                for src, dst in ((rail, rail + 4), (rail + 4, rail))
            ],
            "shared_resources": [],
        }
    )


def test_single_gateway_allgather_builds_three_stage_dag():
    inputs = hierarchical_allgather_inputs()
    topology = load_topology(inputs)

    plan = build_plan(inputs, topology)

    assert plan.planning_mode is PlanningMode.GATEWAY_ALLGATHER
    assert plan.planning_reason == "eligible_gateway_domain"
    assert [node.node_id for node in plan.nodes] == [
        "local-gather-node-0-rail-0",
        "local-gather-node-1-rail-0",
        "gateway-allgather-rail-0",
        "local-allgather-node-0-rail-0",
        "local-allgather-node-1-rail-0",
    ]
    assert [node.stage_id for node in plan.nodes] == [0, 0, 1, 2, 2]
    assert [node.local_collective.kind for node in plan.nodes] == [
        CollectiveKind.GATHER,
        CollectiveKind.GATHER,
        CollectiveKind.ALL_GATHER,
        CollectiveKind.ALL_GATHER,
        CollectiveKind.ALL_GATHER,
    ]
    assert {
        (edge.producer_id, edge.consumer_id)
        for edge in plan.edges
    } == {
        ("local-gather-node-0-rail-0", "gateway-allgather-rail-0"),
        ("local-gather-node-1-rail-0", "gateway-allgather-rail-0"),
        ("gateway-allgather-rail-0", "local-allgather-node-0-rail-0"),
        ("gateway-allgather-rail-0", "local-allgather-node-1-rail-0"),
    }
    assert all(
        LinkKey(1, 5) not in node.allowed_links
        for node in plan.nodes
    )


def test_four_gateway_rails_partition_every_slice_by_global_id():
    inputs = hierarchical_allgather_inputs()

    plan = build_plan(inputs, four_rail_topology())

    gateway_nodes = {
        int(node.node_id.rsplit("-", 1)[1]): node
        for node in plan.nodes
        if node.node_id.startswith("gateway-allgather-rail-")
    }
    assert tuple(sorted(gateway_nodes)) == (0, 1, 2, 3)
    slices_by_rail = {
        rail: {
            contributor
            for contributors in node.logical_input.values.values()
            for contributor in contributors
        }
        for rail, node in gateway_nodes.items()
    }
    expected_slices = set(
        range(inputs.rank_count * inputs.hyperparameters.slice_count)
    )
    assert set().union(*slices_by_rail.values()) == expected_slices
    assert sum(len(values) for values in slices_by_rail.values()) == len(
        expected_slices
    )
    assert all(
        slice_id % 4 == rail
        for rail, values in slices_by_rail.items()
        for slice_id in values
    )


def test_gateway_allgather_preserves_global_offsets_and_exact_edges():
    inputs = hierarchical_allgather_inputs()

    plan = build_plan(inputs, four_rail_topology())
    by_id = {node.node_id: node for node in plan.nodes}

    assert plan.final_outputs.values == required_outputs(
        inputs.collective,
        inputs.rank_count,
        inputs.hyperparameters.slice_count,
    )
    assert all(
        all(
            by_id[edge.producer_id].logical_output.values.get(slot)
            == contributors
            == by_id[edge.consumer_id].logical_input.values.get(slot)
            for slot, contributors in edge.interface.values.items()
        )
        for edge in plan.edges
    )
    assert all(
        slot.offset == next(iter(contributors))
        for node in plan.nodes
        if node.local_collective.kind is CollectiveKind.ALL_GATHER
        for slot, contributors in node.logical_output.values.items()
    )


@pytest.mark.parametrize(
    "topology",
    [
        {
            "ranks": 4,
            "nodes": [
                {"id": 0, "ranks": [0, 1], "gateways": [0]},
                {"id": 1, "ranks": [2, 3], "gateways": [2, 3]},
            ],
            "directed_links": [
                {"src": 0, "dst": 1, "alpha": 1, "invbw": 2},
                {"src": 1, "dst": 0, "alpha": 1, "invbw": 2},
                {"src": 2, "dst": 3, "alpha": 1, "invbw": 2},
                {"src": 3, "dst": 2, "alpha": 1, "invbw": 2},
                {"src": 0, "dst": 2, "alpha": 2, "invbw": 3},
                {"src": 2, "dst": 0, "alpha": 2, "invbw": 3},
            ],
            "shared_resources": [],
        },
        {
            "ranks": 4,
            "nodes": [
                {"id": 0, "ranks": [0, 1], "gateways": [0]},
                {"id": 1, "ranks": [2, 3], "gateways": [2]},
            ],
            "directed_links": [
                {"src": 0, "dst": 1, "alpha": 1, "invbw": 2},
                {"src": 1, "dst": 0, "alpha": 1, "invbw": 2},
                {"src": 2, "dst": 3, "alpha": 1, "invbw": 2},
                {"src": 3, "dst": 2, "alpha": 1, "invbw": 2},
                {"src": 0, "dst": 2, "alpha": 2, "invbw": 3},
            ],
            "shared_resources": [],
        },
        {
            "ranks": 6,
            "nodes": [
                {"id": 0, "ranks": [0, 1], "gateways": [0]},
                {"id": 1, "ranks": [2, 3], "gateways": [2]},
                {"id": 2, "ranks": [4, 5], "gateways": [4]},
            ],
            "directed_links": [
                {"src": 0, "dst": 1, "alpha": 1, "invbw": 2},
                {"src": 1, "dst": 0, "alpha": 1, "invbw": 2},
                {"src": 2, "dst": 3, "alpha": 1, "invbw": 2},
                {"src": 3, "dst": 2, "alpha": 1, "invbw": 2},
                {"src": 4, "dst": 5, "alpha": 1, "invbw": 2},
                {"src": 5, "dst": 4, "alpha": 1, "invbw": 2},
                {"src": 0, "dst": 2, "alpha": 2, "invbw": 3},
                {"src": 2, "dst": 0, "alpha": 2, "invbw": 3},
            ],
            "shared_resources": [],
        },
        {
            "ranks": 4,
            "nodes": [
                {"id": 0, "ranks": [0, 1], "gateways": [0]},
                {"id": 1, "ranks": [2, 3], "gateways": [2]},
            ],
            "directed_links": [
                {"src": 0, "dst": 1, "alpha": 1, "invbw": 2},
                {"src": 1, "dst": 0, "alpha": 1, "invbw": 2},
                {"src": 2, "dst": 3, "alpha": 1, "invbw": 3},
                {"src": 3, "dst": 2, "alpha": 1, "invbw": 2},
                {"src": 0, "dst": 2, "alpha": 2, "invbw": 3},
                {"src": 2, "dst": 0, "alpha": 2, "invbw": 3},
            ],
            "shared_resources": [],
        },
    ],
    ids=(
        "unequal-gateways",
        "missing-reverse-link",
        "partial-node-coverage",
        "nonisomorphic-local-domain",
    ),
)
def test_invalid_gateway_domains_fall_back_to_direct(topology):
    inputs = hierarchical_allgather_inputs()
    topology = topology_from_mapping(topology)
    inputs = replace(inputs, rank_count=topology.rank_count)

    plan = build_plan(inputs, topology)

    assert plan.planning_mode is PlanningMode.DIRECT
    assert plan.planning_reason == "no_eligible_gateway_domain"
    assert plan.final_outputs.values == required_outputs(
        inputs.collective,
        inputs.rank_count,
        inputs.hyperparameters.slice_count,
    )


def test_each_local_node_uses_only_domain_links_and_resources():
    inputs = hierarchical_allgather_inputs()
    topology = load_topology(inputs)

    plan = build_plan(inputs, topology)

    for node in plan.nodes:
        group = set(node.communication_group)
        expected_links = frozenset(
            key
            for key in topology.links
            if key.src_rank in group and key.dst_rank in group
        )
        expected_resources = frozenset(
            resource_id
            for key in expected_links
            for resource_id in topology.resources_for(key)
        )
        assert node.allowed_links == expected_links
        assert node.shared_resource_ids == expected_resources
