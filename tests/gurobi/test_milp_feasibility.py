from dataclasses import replace

import pytest

from vericcl.input.models import ObjectiveMode
from vericcl.solver.budget import ModelBudget
from vericcl.solver.constructive import construct_candidate
from vericcl.solver.milp import solve_milp
from vericcl.solver.model import SolveStatus

from tests.gurobi.helpers import (
    broadcast_problem,
    reduction_dual_problem,
    require_gurobi_license,
)


pytestmark = [pytest.mark.phase03, pytest.mark.gurobi]


def test_two_rank_broadcast_satisfies_flow_causality_and_lane_order():
    require_gurobi_license()
    problem = broadcast_problem()

    candidate = solve_milp(
        problem,
        channel_count=1,
        objective=ObjectiveMode.LATENCY,
        budget=ModelBudget(seconds=30, started_at=0, deadline=30),
        warm_start=None,
    )

    assert candidate.metrics.status in {
        SolveStatus.OPTIMAL,
        SolveStatus.FEASIBLE,
    }
    schedule = candidate.node_schedules[problem.node.node_id]
    assert len(schedule.transfers) == 2
    intervals = sorted(
        (transfer.st_time, transfer.ed_time)
        for transfer in schedule.transfers
    )
    assert intervals[0][1] <= intervals[1][0]
    assert all(
        atom.st_time >= atom.current_symbol.ready_time
        for transfer in schedule.transfers
        for atom in transfer.atoms
    )
    assert {
        tuple(value)
        for value in schedule.metadata["semantic_contributors"].values()
    } == {(0,), (1,)}
    assert candidate.metrics.solver_seed == 0
    assert candidate.metrics.thread_count >= 1
    assert candidate.metrics.maximum_normalized_resource_load > 0.0
    assert candidate.metrics.within_requested_gap
    assert not candidate.proven_optimal


def test_zero_gap_optimal_model_retains_global_proof():
    require_gurobi_license()
    problem = broadcast_problem(logical_positions=(0,))
    problem = replace(
        problem,
        inputs=replace(
            problem.inputs,
            solver=replace(
                problem.inputs.solver,
                require_proven_optimal=True,
            ),
        ),
    )

    candidate = solve_milp(
        problem,
        channel_count=1,
        objective=ObjectiveMode.LATENCY,
        budget=ModelBudget(seconds=30, started_at=0, deadline=30),
        warm_start=None,
    )

    assert candidate.metrics.status is SolveStatus.OPTIMAL
    assert candidate.metrics.mip_gap == 0.0
    assert candidate.proven_optimal


def test_reduction_dual_accepts_constructive_virtual_tree_warm_start():
    require_gurobi_license()
    problem = reduction_dual_problem()
    warm_start = construct_candidate(problem, channel_count=1)

    candidate = solve_milp(
        problem,
        channel_count=1,
        objective=ObjectiveMode.LATENCY,
        budget=ModelBudget(seconds=30, started_at=0, deadline=30),
        warm_start=warm_start,
    )

    schedule = candidate.node_schedules[problem.node.node_id]
    assert schedule.metadata["reduction_dual"]
    assert len(schedule.transfers) == 1
    transfer = schedule.transfers[0]
    assert (transfer.src_rank, transfer.dst_rank) == (0, 1)
    assert transfer.member_slice_ids == frozenset({8})
    assert {atom.slice_id for atom in transfer.atoms} == {8}
    symbol = transfer.atoms[0].path[0].symbols[0]
    assert (symbol.src_rank, symbol.dst_rank) == (0, 1)
