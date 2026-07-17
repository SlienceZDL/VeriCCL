from dataclasses import replace

import pytest

from vericcl.errors import SemanticError, SolverUnavailableError
from vericcl.input.models import ObjectiveMode
from vericcl.solver.constructive import construct_candidate
from vericcl.solver.model import (
    SolveCandidate,
    SolveStatus,
    SolverMetrics,
)
from vericcl.solver.search import allocate_model_threads, search_models

from tests.gurobi.helpers import broadcast_problem


pytestmark = pytest.mark.phase03


def test_thread_allocation_never_exceeds_cpu_count():
    allocation = allocate_model_threads(
        model_count=4,
        requested_per_model=12,
        cpu_count=16,
    )

    assert allocation == (4, 4, 4, 4)
    assert sum(allocation) <= 16


@pytest.mark.parametrize("field", ["model_count", "requested", "cpus"])
def test_thread_allocation_rejects_non_positive_inputs(field):
    values = {"model_count": 1, "requested": 1, "cpus": 1}
    values[field] = 0

    with pytest.raises(SemanticError, match="positive integer"):
        allocate_model_threads(
            values["model_count"],
            values["requested"],
            values["cpus"],
        )


def test_search_runs_each_k_and_assigns_bounded_threads(monkeypatch):
    problem = broadcast_problem(logical_positions=(0,))
    config = replace(
        problem.inputs.solver,
        max_channels=4,
        max_parallel_models=2,
        max_threads_per_model=12,
    )
    calls = []

    def fake_solve(current, channel_count, objective, budget, warm_start):
        calls.append(
            (
                channel_count,
                current.inputs.solver.max_threads_per_model,
                objective,
                budget.seconds,
                warm_start,
            )
        )
        return replace(
            _candidate_for_test(channel_count),
            objective_mode=objective,
        )

    monkeypatch.setattr("vericcl.solver.search.solve_milp", fake_solve)
    monkeypatch.setattr("vericcl.solver.search.os.cpu_count", lambda: 4)

    candidates = search_models(
        problem,
        config,
        ObjectiveMode.LATENCY,
        warm_start=None,
    )

    assert [item.channel_count for item in candidates] == [1, 2, 3, 4]
    assert {call[1] for call in calls} == {2}
    assert sum(sorted({call[0]: call[1] for call in calls}.values())[:2]) <= 4
    assert all(call[2] is ObjectiveMode.LATENCY for call in calls)


def test_search_discards_models_without_complete_incumbents(monkeypatch):
    problem = broadcast_problem(logical_positions=(0,))
    config = replace(problem.inputs.solver, max_channels=2)

    def fake_solve(current, channel_count, objective, budget, warm_start):
        candidate = _candidate_for_test(channel_count)
        if channel_count == 2:
            return replace(
                candidate,
                node_schedules={},
                metrics=replace(
                    candidate.metrics,
                    status=SolveStatus.TIME_LIMIT,
                ),
            )
        return candidate

    monkeypatch.setattr("vericcl.solver.search.solve_milp", fake_solve)

    candidates = search_models(
        problem,
        config,
        ObjectiveMode.LATENCY,
        warm_start=None,
    )

    assert [item.channel_count for item in candidates] == [1]


def test_total_budget_stops_launching_new_models(monkeypatch):
    problem = broadcast_problem(logical_positions=(0,))
    config = replace(
        problem.inputs.solver,
        max_channels=4,
        max_parallel_models=1,
        total_solve_timeout_s=10,
        per_model_timeout_s=20,
    )
    times = iter((0.0, 0.0, 11.0))
    calls = []

    def fake_solve(current, channel_count, objective, budget, warm_start):
        calls.append((channel_count, budget.seconds))
        return _candidate_for_test(channel_count)

    monkeypatch.setattr("vericcl.solver.search.solve_milp", fake_solve)
    monkeypatch.setattr("vericcl.solver.search._monotonic", lambda: next(times))
    monkeypatch.setattr("vericcl.solver.search.os.cpu_count", lambda: 1)

    candidates = search_models(
        problem,
        config,
        ObjectiveMode.LATENCY,
        warm_start=None,
    )

    assert calls == [(1, 10.0)]
    assert [item.channel_count for item in candidates] == [1]
    assert candidates[0].metrics.model_count == 1
    assert candidates[0].metrics.model_index == 0


def test_unavailable_backend_stops_additional_model_launches(monkeypatch):
    problem = broadcast_problem(logical_positions=(0,))
    config = replace(
        problem.inputs.solver,
        max_channels=4,
        max_parallel_models=1,
    )
    calls = []

    def unavailable(*args):
        calls.append(args[1])
        raise SolverUnavailableError("missing")

    monkeypatch.setattr("vericcl.solver.search.solve_milp", unavailable)

    candidates = search_models(
        problem,
        config,
        ObjectiveMode.LATENCY,
        warm_start=None,
    )

    assert candidates == ()
    assert calls == [1]


@pytest.mark.parametrize(
    "field,value",
    [
        ("problem", object()),
        ("config", object()),
        ("objective", "latency"),
        ("objective", ObjectiveMode.AUTO),
        ("warm_start", object()),
    ],
)
def test_search_rejects_invalid_api_arguments(field, value):
    problem = broadcast_problem(logical_positions=(0,))
    arguments = {
        "problem": problem,
        "config": problem.inputs.solver,
        "objective": ObjectiveMode.LATENCY,
        "warm_start": None,
    }
    arguments[field] = value

    with pytest.raises(SemanticError):
        search_models(**arguments)


def _candidate_for_test(channel_count):
    problem = broadcast_problem(logical_positions=(0,))
    schedule = construct_candidate(
        problem,
        channel_count=channel_count,
    )
    metrics = SolverMetrics(
        status=SolveStatus.FEASIBLE,
        objective_values=(schedule.transfers[-1].ed_time,),
        best_bound=0.0,
        mip_gap=0.0,
        within_requested_gap=True,
        solve_time_s=0.0,
        model_count=1,
        operation_count=len(schedule.transfers),
        hop_count=len(schedule.transfers),
        makespan_us=schedule.transfers[-1].ed_time,
        maximum_normalized_resource_load=1.0,
        solver_name="test",
        solver_version="1",
        solver_seed=0,
        thread_count=1,
        termination_reason="complete",
    )
    return SolveCandidate(
        candidate_id="candidate-k{:02d}".format(channel_count),
        node_schedules={problem.node.node_id: schedule},
        objective_mode=ObjectiveMode.LATENCY,
        channel_count=channel_count,
        metrics=metrics,
        selected_best=False,
        proven_optimal=False,
        search_space_restricted=problem.search_space_restricted,
        restrictions=problem.restrictions,
        parent_candidate_id=None,
    )
