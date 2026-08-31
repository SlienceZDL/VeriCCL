from dataclasses import replace
from pathlib import Path

import pytest
from hypothesis import given, strategies as st

from vericcl.input.loader import resolve_inputs
from vericcl.planner.build import build_plan
from vericcl.planner.direct import build_direct_plan
from vericcl.planner.model import PlanningMode
from vericcl.semantics.collective import (
    CollectiveKind,
    CollectiveSpec,
    required_outputs,
)
from vericcl.topology.loader import load_topology


pytestmark = pytest.mark.phase02


EXAMPLES = Path(__file__).parents[2] / "vericcl" / "examples"
DIRECT_KINDS = (
    CollectiveKind.BROADCAST,
    CollectiveKind.REDUCE,
    CollectiveKind.ALL_GATHER,
    CollectiveKind.ALL_REDUCE,
    CollectiveKind.ALL_TO_ALL,
    CollectiveKind.REDUCE_SCATTER,
)


def direct_input(kind, slice_count):
    inputs = resolve_inputs(
        EXAMPLES / "topo" / "two_rank.json",
        EXAMPLES / "sketch" / "allreduce_8m_1m.json",
        EXAMPLES / "atom" / "default.json",
    )
    rooted = kind in {CollectiveKind.BROADCAST, CollectiveKind.REDUCE}
    reduced = kind in {
        CollectiveKind.REDUCE,
        CollectiveKind.ALL_REDUCE,
        CollectiveKind.REDUCE_SCATTER,
    }
    return replace(
        inputs,
        collective=CollectiveSpec(
            kind=kind,
            datatype="float32",
            reduction_op="sum" if reduced else None,
            root=0 if rooted else None,
        ),
        hyperparameters=replace(
            inputs.hyperparameters,
            total_size_bytes=slice_count,
            slice_size_bytes=1,
        ),
    )


@given(
    kind=st.sampled_from(DIRECT_KINDS),
    slice_count=st.sampled_from((2, 4, 6)),
)
def test_direct_plan_interfaces_match_collective_semantics(kind, slice_count):
    inputs = direct_input(kind, slice_count)
    topology = load_topology(inputs)

    plan = build_direct_plan(inputs, topology)

    assert plan.final_outputs.values == required_outputs(
        inputs.collective,
        inputs.rank_count,
        slice_count,
    )
    assert all(
        all(
            link.src_rank in node.communication_group
            and link.dst_rank in node.communication_group
            for link in node.allowed_links
        )
        for node in plan.nodes
    )


@given(slice_count=st.sampled_from((1, 2, 4, 8)))
def test_gateway_allgather_interfaces_match_direct_semantics(slice_count):
    inputs = direct_input(CollectiveKind.ALL_GATHER, slice_count)
    gateway_inputs = resolve_inputs(
        EXAMPLES / "topo" / "two_node_gateway.json",
        EXAMPLES / "sketch" / "allreduce_8m_1m.json",
        EXAMPLES / "atom" / "default.json",
    )
    inputs = replace(
        gateway_inputs,
        collective=inputs.collective,
        hyperparameters=replace(
            gateway_inputs.hyperparameters,
            total_size_bytes=slice_count,
            slice_size_bytes=1,
        ),
        strategies=replace(gateway_inputs.strategies, hierarchy=True),
    )

    plan = build_plan(inputs, load_topology(inputs))

    assert plan.planning_mode is PlanningMode.GATEWAY_ALLGATHER
    assert plan.final_outputs.values == required_outputs(
        inputs.collective,
        inputs.rank_count,
        slice_count,
    )
