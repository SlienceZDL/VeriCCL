from dataclasses import replace
from pathlib import Path

import pytest

from vericcl.input.loader import resolve_inputs
from vericcl.planner.build import build_plan
from vericcl.planner.model import PlanningMode
from vericcl.semantics.collective import (
    CollectiveKind,
    CollectiveSpec,
    required_outputs,
)
from vericcl.topology.loader import load_topology, topology_from_mapping
from vericcl.topology.model import LinkKey


pytestmark = pytest.mark.phase02


EXAMPLES = Path(__file__).parents[2] / "vericcl" / "examples"


def allgather_inputs(rank_count=8, slice_count=4):
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
        hyperparameters=replace(
            inputs.hyperparameters,
            total_size_bytes=slice_count,
            slice_size_bytes=1,
        ),
        strategies=replace(inputs.strategies, hierarchy=True),
        rank_count=rank_count,
    )


def gateway_topology(
    node_ranks,
    gateways,
    rail_pairs,
    *,
    reverse_rails=True,
    local_invbw=None,
):
    local_invbw = local_invbw or (2,) * len(node_ranks)
    links = []
    for node_index, ranks in enumerate(node_ranks):
        for src in ranks:
            for dst in ranks:
                if src != dst:
                    links.append(
                        {
                            "src": src,
                            "dst": dst,
                            "alpha": 1,
                            "invbw": local_invbw[node_index],
                            "max_channels": 4,
                        }
                    )
    for src, dst in rail_pairs:
        links.append(
            {
                "src": src,
                "dst": dst,
                "alpha": 2,
                "invbw": 5,
                "max_channels": 4,
            }
        )
        if reverse_rails:
            links.append(
                {
                    "src": dst,
                    "dst": src,
                    "alpha": 2,
                    "invbw": 5,
                    "max_channels": 4,
                }
            )
    return topology_from_mapping(
        {
            "ranks": sum(len(ranks) for ranks in node_ranks),
            "nodes": [
                {
                    "id": node_index,
                    "ranks": list(ranks),
                    "gateways": list(gateways[node_index]),
                }
                for node_index, ranks in enumerate(node_ranks)
            ],
            "directed_links": links,
            "shared_resources": [],
        }
    )


def test_one_gateway_builds_three_phase_allgather_dag_on_real_links():
    inputs = allgather_inputs(slice_count=2)
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
    assert [node.local_collective.kind for node in plan.nodes] == [
        CollectiveKind.GATHER,
        CollectiveKind.GATHER,
        CollectiveKind.ALL_GATHER,
        CollectiveKind.ALL_GATHER,
        CollectiveKind.ALL_GATHER,
    ]
    assert [node.stage_id for node in plan.nodes] == [0, 0, 1, 2, 2]
    assert len(plan.edges) == 4
    gateway = next(
        node for node in plan.nodes if node.node_id == "gateway-allgather-rail-0"
    )
    assert gateway.communication_group == (0, 4)
    assert gateway.allowed_links == frozenset({LinkKey(0, 4), LinkKey(4, 0)})
    assert LinkKey(1, 5) not in gateway.allowed_links
    assert plan.final_outputs.values == required_outputs(
        inputs.collective,
        inputs.rank_count,
        inputs.hyperparameters.slice_count,
    )


def test_four_gateways_build_four_independent_slice_partitioned_rails():
    inputs = allgather_inputs(slice_count=4)
    topology = gateway_topology(
        ((0, 1, 2, 3), (4, 5, 6, 7)),
        ((0, 1, 2, 3), (4, 5, 6, 7)),
        ((0, 4), (1, 5), (2, 6), (3, 7)),
    )

    plan = build_plan(inputs, topology)

    gateways = {
        node.node_id: node
        for node in plan.nodes
        if node.node_id.startswith("gateway-allgather")
    }
    assert tuple(gateways) == tuple(
        "gateway-allgather-rail-{}".format(rail) for rail in range(4)
    )
    assert [node.communication_group for node in gateways.values()] == [
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    for rail, node in enumerate(gateways.values()):
        contributors = {
            contributor
            for values in node.logical_input.values.values()
            for contributor in values
        }
        assert contributors
        assert all(contributor % 4 == rail for contributor in contributors)
        assert all(
            contributor % 4 == rail
            for values in node.logical_output.values.values()
            for contributor in values
        )
    for edge in plan.edges:
        producer_rail = int(edge.producer_id.rsplit("-", 1)[-1])
        consumer_rail = int(edge.consumer_id.rsplit("-", 1)[-1])
        assert producer_rail == consumer_rail


@pytest.mark.parametrize(
    "topology",
    [
        gateway_topology(
            ((0, 1), (2, 3)),
            ((0, 1), (2,)),
            ((0, 2),),
        ),
        gateway_topology(
            ((0, 1), (2, 3)),
            ((0,), (2,)),
            ((0, 2),),
            reverse_rails=False,
        ),
        gateway_topology(
            ((0, 1), (2, 3), (4, 5)),
            ((0,), (2,), (4,)),
            ((0, 2),),
        ),
        gateway_topology(
            ((0, 1), (2, 3)),
            ((0,), (2,)),
            ((0, 2),),
            local_invbw=(2, 3),
        ),
    ],
    ids=(
        "unequal-gateway-counts",
        "missing-reverse-link",
        "incomplete-node-coverage",
        "nonisomorphic-local-domains",
    ),
)
def test_ineligible_gateway_domains_fall_back_to_direct(topology):
    inputs = allgather_inputs(
        rank_count=topology.rank_count,
        slice_count=4,
    )

    plan = build_plan(inputs, topology)

    assert plan.planning_mode is PlanningMode.DIRECT
    assert plan.planning_reason == "no_eligible_gateway_domain"
    assert all(node.node_id.startswith("allgather-") for node in plan.nodes)
