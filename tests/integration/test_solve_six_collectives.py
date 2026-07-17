from dataclasses import replace
from pathlib import Path

import pytest

from vericcl.composer import compose
from vericcl.input.loader import resolve_inputs
from vericcl.input.models import ObjectiveMode
from vericcl.planner.build import build_plan
from vericcl.semantics.collective import (
    CollectiveKind,
    CollectiveSpec,
    required_outputs,
)
from vericcl.solver.cache import CandidateCache
from vericcl.solver.model import SolveRequest, SolveStatus
from vericcl.solver.orchestrator import solve
from vericcl.topology.loader import load_topology
from vericcl.topology.model import LaneKey


pytestmark = pytest.mark.phase03


EXAMPLES = Path(__file__).parents[2] / "vericcl" / "examples"


def _spec(kind):
    reduction = kind in {
        CollectiveKind.REDUCE,
        CollectiveKind.ALL_REDUCE,
        CollectiveKind.REDUCE_SCATTER,
    }
    rooted = kind in {CollectiveKind.BROADCAST, CollectiveKind.REDUCE}
    return CollectiveSpec(
        kind=kind,
        datatype="float32",
        reduction_op="sum" if reduction else None,
        root=0 if rooted else None,
        inplace=False,
    )


def _request(kind):
    inputs = resolve_inputs(
        EXAMPLES / "topo" / "two_rank.json",
        EXAMPLES / "sketch" / "allreduce_8m_1m.json",
        EXAMPLES / "atom" / "default.json",
    )
    inputs = replace(
        inputs,
        collective=_spec(kind),
        hyperparameters=replace(
            inputs.hyperparameters,
            total_size_bytes=2 * inputs.hyperparameters.slice_size_bytes,
            objective_mode=ObjectiveMode.LATENCY,
        ),
        solver=replace(
            inputs.solver,
            max_channels=1,
            max_parallel_models=1,
            max_threads_per_model=1,
        ),
        strategies=replace(
            inputs.strategies,
            hierarchy=False,
            constructive_trees=True,
            milp=False,
        ),
    )
    topology = load_topology(inputs)
    return SolveRequest(
        inputs=inputs,
        topology=topology,
        plan=build_plan(inputs, topology),
        solver_version="integration-solver",
        model_version="integration-model",
        environment_signature="integration-environment",
    )


@pytest.mark.parametrize(
    "kind",
    (
        CollectiveKind.BROADCAST,
        CollectiveKind.REDUCE,
        CollectiveKind.ALL_GATHER,
        CollectiveKind.ALL_REDUCE,
        CollectiveKind.ALL_TO_ALL,
        CollectiveKind.REDUCE_SCATTER,
    ),
)
def test_constructive_solve_and_compose_preserve_collective_semantics(kind):
    request = _request(kind)

    result = solve(request, cache=CandidateCache())

    assert result.status is SolveStatus.FEASIBLE
    candidate = result.selected_candidate
    assert candidate is not None
    schedule = compose(
        request.plan,
        {node.node_id: candidate for node in request.plan.nodes},
    )
    actual = {
        (int(key[1:9]), int(key[11:19])): frozenset(contributors)
        for key, contributors in schedule.metadata["final_outputs"].items()
    }
    expected = {
        (slot.rank, slot.offset): contributors
        for slot, contributors in required_outputs(
            request.inputs.collective,
            request.inputs.rank_count,
            request.plan.slice_count,
        ).items()
    }

    assert actual == expected
    lanes = {}
    for transfer in schedule.transfers:
        lane = LaneKey(
            transfer.src_rank,
            transfer.dst_rank,
            transfer.channel,
        )
        lanes.setdefault(lane, []).append(
            (transfer.st_time, transfer.ed_time)
        )
    for intervals in lanes.values():
        ordered = sorted(intervals)
        assert all(
            left[1] <= right[0]
            for left, right in zip(ordered, ordered[1:])
        )
