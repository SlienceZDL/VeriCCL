from dataclasses import replace
from pathlib import Path

import pytest

from vericcl.composer.compose import compose
from vericcl.input.loader import resolve_inputs
from vericcl.input.models import ObjectiveMode
from vericcl.planner.build import build_plan
from vericcl.solver.constructive import construct_candidate
from vericcl.solver.demands import build_solver_problem
from vericcl.solver.model import SolveCandidate, SolveStatus, SolverMetrics
from vericcl.topology.loader import load_topology
from vericcl.topology.model import LaneKey


pytestmark = pytest.mark.phase03


EXAMPLES = Path(__file__).parents[2] / "vericcl" / "examples"


def _candidate(problem, schedule):
    makespan = max(
        (transfer.ed_time for transfer in schedule.transfers),
        default=0.0,
    )
    metrics = SolverMetrics(
        status=SolveStatus.FEASIBLE,
        objective_values=(
            makespan,
            float(len(schedule.transfers)),
            float(len(schedule.transfers)),
        ),
        best_bound=0.0,
        mip_gap=0.0,
        within_requested_gap=True,
        solve_time_s=0.0,
        model_count=1,
        operation_count=len(schedule.transfers),
        hop_count=len(schedule.transfers),
        makespan_us=makespan,
        maximum_normalized_resource_load=makespan,
        solver_name="constructive",
        solver_version="1",
        solver_seed=0,
        thread_count=1,
        termination_reason="complete",
    )
    return SolveCandidate(
        candidate_id="{}-candidate".format(problem.node.node_id),
        node_schedules={problem.node.node_id: schedule},
        objective_mode=ObjectiveMode.LATENCY,
        channel_count=1,
        metrics=metrics,
        selected_best=False,
        proven_optimal=False,
        search_space_restricted=problem.search_space_restricted,
        restrictions=problem.restrictions,
        parent_candidate_id=None,
    )


def test_two_rank_allreduce_composes_dual_and_allgather_semantics():
    inputs = resolve_inputs(
        EXAMPLES / "topo" / "two_rank.json",
        EXAMPLES / "sketch" / "allreduce_8m_1m.json",
        EXAMPLES / "atom" / "default.json",
    )
    inputs = replace(
        inputs,
        hyperparameters=replace(
            inputs.hyperparameters,
            total_size_bytes=2 * inputs.hyperparameters.slice_size_bytes,
        ),
    )
    topology = load_topology(inputs)
    plan = build_plan(inputs, topology)
    candidates = {}
    for node in plan.nodes:
        problem = build_solver_problem(node, inputs, topology)
        schedule = construct_candidate(problem, channel_count=1)
        candidates[node.node_id] = _candidate(problem, schedule)

    schedule = compose(plan, candidates)

    assert schedule.metadata["path_scope"] == "global"
    assert set(schedule.metadata["final_outputs"].values()) == {
        (0, 2),
        (1, 3),
    }
    assert {transfer.kind for transfer in schedule.transfers} == {
        "REDUCE",
        "SEND",
    }
    for transfer in schedule.transfers:
        for atom in transfer.atoms:
            atom.validate_path_prefix(
                current_rank=transfer.dst_rank,
                slice_count=schedule.slice_count,
            )
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
            first[1] <= second[0]
            for first, second in zip(ordered, ordered[1:])
        )
