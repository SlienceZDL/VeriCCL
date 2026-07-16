from dataclasses import replace
from pathlib import Path

import pytest

from vericcl.input.loader import resolve_inputs
from vericcl.planner.build import build_plan
from vericcl.semantics.collective import (
    CollectiveKind,
    OutputSlot,
    required_outputs,
)
from vericcl.topology.loader import load_topology
from vericcl.topology.model import LinkKey


pytestmark = pytest.mark.phase02


EXAMPLES = Path(__file__).parents[2] / "vericcl" / "examples"


def hierarchical_gateway_inputs():
    inputs = resolve_inputs(
        EXAMPLES / "topo" / "two_node_gateway.json",
        EXAMPLES / "sketch" / "allreduce_8m_1m.json",
        EXAMPLES / "atom" / "default.json",
    )
    return replace(
        inputs,
        strategies=replace(inputs.strategies, hierarchy=True),
    )


def test_gateway_allreduce_uses_only_real_gateways():
    inputs = hierarchical_gateway_inputs()
    topology = load_topology(inputs)

    plan = build_plan(inputs, topology)

    assert [
        (node.local_collective.kind, node.communication_group)
        for node in plan.nodes
    ] == [
        (CollectiveKind.REDUCE, (0, 1, 2, 3)),
        (CollectiveKind.REDUCE, (4, 5, 6, 7)),
        (CollectiveKind.REDUCE_SCATTER, (0, 4)),
        (CollectiveKind.ALL_GATHER, (0, 4)),
        (CollectiveKind.ALL_GATHER, (0, 1, 2, 3)),
        (CollectiveKind.ALL_GATHER, (4, 5, 6, 7)),
    ]
    assert all(
        LinkKey(1, 5) not in node.allowed_links
        for node in plan.nodes
    )


def test_gateway_allreduce_has_exact_global_contributors():
    inputs = hierarchical_gateway_inputs()
    topology = load_topology(inputs)

    plan = build_plan(inputs, topology)

    assert plan.final_outputs.values == required_outputs(
        inputs.collective,
        inputs.rank_count,
        inputs.hyperparameters.slice_count,
    )
    inter_reduce_scatter = plan.nodes[2]
    assert inter_reduce_scatter.logical_output.values[OutputSlot(0, 0)] == (
        frozenset({0, 8, 16, 24, 32, 40, 48, 56})
    )


def test_gateway_stages_encode_only_data_dependencies():
    inputs = hierarchical_gateway_inputs()

    plan = build_plan(inputs, load_topology(inputs))

    assert [node.stage_id for node in plan.nodes] == [0, 0, 1, 2, 3, 3]
    assert {
        (edge.producer_id, edge.consumer_id)
        for edge in plan.edges
    } == {
        ("local-reduce-node-0", "gateway-reduce-scatter"),
        ("local-reduce-node-1", "gateway-reduce-scatter"),
        ("gateway-reduce-scatter", "gateway-allgather"),
        ("gateway-allgather", "local-allgather-node-0"),
        ("gateway-allgather", "local-allgather-node-1"),
    }
    assert not any(
        edge.producer_id == "local-reduce-node-0"
        and edge.consumer_id == "local-reduce-node-1"
        for edge in plan.edges
    )
