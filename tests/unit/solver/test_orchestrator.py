from dataclasses import replace
from pathlib import Path

import pytest

from vericcl.input.loader import resolve_inputs
from vericcl.errors import SemanticError, SolverUnavailableError
from vericcl.input.models import ForbiddenTransfer, ObjectiveMode
from vericcl.planner.build import build_plan
from vericcl.solver.cache import CandidateCache
from vericcl.solver.lower_bounds import LowerBound
from vericcl.solver.model import SolveRequest, SolveStatus
from vericcl.solver.orchestrator import solve
from vericcl.topology.loader import load_topology
from vericcl.tuning.model import TuningOverlay

from tests.unit.planner.test_hierarchy import manual_allreduce


pytestmark = pytest.mark.phase03


EXAMPLES = Path(__file__).parents[3] / "vericcl" / "examples"


def _request(
    objective=ObjectiveMode.LATENCY,
    constructive=True,
    milp=False,
    force_resolve=False,
    require_proven_optimal=False,
):
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
            objective_mode=objective,
        ),
        solver=replace(
            inputs.solver,
            max_channels=1,
            max_parallel_models=1,
            max_threads_per_model=1,
            force_resolve=force_resolve,
            require_proven_optimal=require_proven_optimal,
        ),
        strategies=replace(
            inputs.strategies,
            hierarchy=False,
            constructive_trees=constructive,
            milp=milp,
        ),
    )
    topology = load_topology(inputs)
    return SolveRequest(
        inputs=inputs,
        topology=topology,
        plan=build_plan(inputs, topology),
        solver_version="test-solver",
        model_version="test-model",
        environment_signature="test-environment",
    )


def test_constructive_only_returns_complete_global_candidate():
    request = _request()

    result = solve(request, cache=CandidateCache())

    assert result.status is SolveStatus.FEASIBLE
    assert result.selected_candidate is not None
    assert result.selected_candidate.selected_best
    assert set(result.selected_candidate.node_schedules) == {
        node.node_id for node in request.plan.nodes
    }
    assert result.selected_candidate.metrics.solver_name == "constructive"
    assert "independent_node_composition" in (
        result.selected_candidate.restrictions
    )


def test_milp_without_incumbent_falls_back_to_constructive(
    monkeypatch,
):
    request = _request(constructive=True, milp=True)
    calls = []

    def no_incumbent(problem, config, objective, warm_start):
        calls.append((problem.node.node_id, objective, warm_start))
        return ()

    monkeypatch.setattr(
        "vericcl.solver.orchestrator.search_models",
        no_incumbent,
    )

    result = solve(request, cache=CandidateCache())

    assert len(calls) == len(request.plan.nodes)
    assert all(call[2] is not None for call in calls)
    assert result.selected_candidate is not None
    assert result.selected_candidate.metrics.solver_name == "constructive"


def test_unavailable_milp_search_falls_back_to_constructive(monkeypatch):
    request = _request(constructive=True, milp=True)

    def unavailable(problem, config, objective, warm_start):
        raise SolverUnavailableError("unavailable")

    monkeypatch.setattr(
        "vericcl.solver.orchestrator.search_models",
        unavailable,
    )

    result = solve(request, cache=CandidateCache())

    assert result.status is SolveStatus.FEASIBLE
    assert result.selected_candidate.metrics.solver_name == "constructive"


def test_complete_milp_incumbents_are_combined_by_channel(monkeypatch):
    request = _request(constructive=True, milp=True)

    from vericcl.solver import orchestrator

    def time_limited(problem, config, objective, warm_start):
        local = orchestrator._constructive_candidate(
            problem,
            warm_start,
            objective,
            1,
            None,
        )
        return (
            replace(
                local,
                candidate_id="{}-milp".format(problem.node.node_id),
                metrics=replace(
                    local.metrics,
                    status=SolveStatus.TIME_LIMIT,
                    solver_name="gurobi",
                    model_count=1,
                    termination_reason="time_limit",
                ),
            ),
        )

    monkeypatch.setattr(
        "vericcl.solver.orchestrator.search_models",
        time_limited,
    )

    result = solve(request, cache=CandidateCache())

    milp = next(
        candidate
        for candidate in result.candidates
        if "-milp-" in candidate.candidate_id
    )
    assert milp.channel_count == 1
    assert milp.metrics.status is SolveStatus.TIME_LIMIT
    assert milp.metrics.solver_name == "gurobi"


def test_total_budget_is_shared_across_plan_nodes(monkeypatch):
    request = _request(constructive=False, milp=True)
    request = replace(
        request,
        inputs=replace(
            request.inputs,
            solver=replace(
                request.inputs.solver,
                total_solve_timeout_s=10,
                per_model_timeout_s=10,
            ),
        ),
    )
    times = iter((0.0, 0.0, 11.0))
    calls = []

    def no_incumbent(problem, config, objective, warm_start):
        calls.append(
            (
                problem.node.node_id,
                config.total_solve_timeout_s,
                config.per_model_timeout_s,
            )
        )
        return ()

    monkeypatch.setattr(
        "vericcl.solver.orchestrator.GurobiAdapter.available",
        lambda: True,
    )
    monkeypatch.setattr(
        "vericcl.solver.orchestrator.search_models",
        no_incumbent,
    )
    monkeypatch.setattr(
        "vericcl.solver.orchestrator._monotonic",
        lambda: next(times),
    )

    result = solve(request, cache=CandidateCache())

    assert len(calls) == 1
    assert calls[0][1:] == (2, 2)
    assert result.selected_candidate is None


def test_both_disabled_returns_not_run():
    result = solve(
        _request(constructive=False, milp=False),
        cache=CandidateCache(),
    )

    assert result.status is SolveStatus.NOT_RUN
    assert result.candidates == ()
    assert result.selected_candidate is None


def test_milp_only_reports_unavailable_backend(monkeypatch):
    monkeypatch.setattr(
        "vericcl.solver.orchestrator.GurobiAdapter.available",
        lambda: False,
    )

    result = solve(
        _request(constructive=False, milp=True),
        cache=CandidateCache(),
    )

    assert result.status is SolveStatus.UNAVAILABLE
    assert result.candidates == ()
    assert "unavailable" in result.message


def test_overlay_channel_count_and_parent_are_applied():
    request = _request()
    inputs = replace(
        request.inputs,
        solver=replace(request.inputs.solver, max_channels=2),
    )
    request = replace(
        request,
        inputs=inputs,
        overlay=TuningOverlay(
            overlay_id="overlay",
            parent_candidate_id="parent",
            channel_count=2,
        ),
    )

    result = solve(request, cache=CandidateCache())

    assert result.status is SolveStatus.FEASIBLE
    assert result.selected_candidate.channel_count == 2
    assert result.selected_candidate.parent_candidate_id == "parent"


def test_overlay_channel_count_above_solver_limit_is_rejected():
    request = replace(
        _request(),
        overlay=TuningOverlay(
            overlay_id="overlay",
            parent_candidate_id=None,
            channel_count=2,
        ),
    )

    result = solve(request, cache=CandidateCache())

    assert result.status is SolveStatus.ERROR
    assert result.candidates == ()
    assert "channel_count" in result.message


def test_overlay_temporary_forbidden_is_a_hard_constraint():
    request = replace(
        _request(),
        overlay=TuningOverlay(
            overlay_id="overlay",
            parent_candidate_id=None,
            temporary_forbidden=frozenset(
                {
                    ForbiddenTransfer(
                        slice_id=2,
                        src_rank=1,
                        dst_rank=0,
                        stage_id=0,
                    )
                }
            ),
        ),
    )

    result = solve(request, cache=CandidateCache())

    assert result.status is SolveStatus.INFEASIBLE
    assert result.selected_candidate is None


def test_stale_plan_is_rejected_before_backend_execution():
    request = _request()
    nodes = list(request.plan.nodes)
    nodes[0] = replace(nodes[0], allowed_links=frozenset())
    stale = replace(request, plan=replace(request.plan, nodes=tuple(nodes)))

    result = solve(stale, cache=CandidateCache())

    assert result.status is SolveStatus.ERROR
    assert result.candidates == ()
    assert "plan" in result.message


def test_manual_hierarchy_conflict_is_rejected_before_solving():
    request = _request()
    manual_inputs = replace(
        request.inputs,
        strategies=replace(
            request.inputs.strategies,
            hierarchy=True,
            manual_hierarchy=manual_allreduce(),
        ),
    )
    conflict = replace(request, inputs=manual_inputs)

    result = solve(conflict, cache=CandidateCache())

    assert result.status is SolveStatus.ERROR
    assert result.candidates == ()
    assert "plan" in result.message


def test_require_proven_optimal_rejects_constructive_incumbent():
    result = solve(
        _request(require_proven_optimal=True),
        cache=CandidateCache(),
    )

    assert result.status is SolveStatus.ERROR
    assert result.selected_candidate is None
    assert result.candidates == ()
    assert "proof" in result.message


def test_force_resolve_bypasses_complete_candidate_cache(monkeypatch):
    cache = CandidateCache()
    request = _request()
    first = solve(request, cache=cache)

    cached = solve(request, cache=cache)
    assert cached.cache_hit
    assert cached.selected_candidate_id == first.selected_candidate_id

    from vericcl.solver import orchestrator

    original = orchestrator.construct_candidate
    calls = []

    def counted(problem, channel_count):
        calls.append((problem.node.node_id, channel_count))
        return original(problem, channel_count)

    monkeypatch.setattr(orchestrator, "construct_candidate", counted)
    forced_inputs = replace(
        request.inputs,
        solver=replace(request.inputs.solver, force_resolve=True),
    )
    forced = replace(request, inputs=forced_inputs)

    result = solve(forced, cache=cache)

    assert not result.cache_hit
    assert calls


def test_auto_skips_throughput_when_cv_adjusted_gain_is_too_small(
    monkeypatch,
):
    request = _request(
        objective=ObjectiveMode.AUTO,
        constructive=True,
        milp=True,
    )
    objectives = []

    def no_incumbent(problem, config, objective, warm_start):
        objectives.append(objective)
        return ()

    monkeypatch.setattr(
        "vericcl.solver.orchestrator.search_models",
        no_incumbent,
    )
    monkeypatch.setattr(
        "vericcl.solver.orchestrator.throughput_time_lower_bound",
        lambda problem, max_channels: LowerBound(1e9, 1e9),
    )
    monkeypatch.setattr(
        "vericcl.solver.orchestrator._relevant_calibration",
        lambda request, candidate: (0.1, True),
    )

    result = solve(request, cache=CandidateCache())

    assert objectives
    assert set(objectives) == {ObjectiveMode.LATENCY}
    assert "auto_throughput=skipped" in result.message
    assert "threshold=0.2" in result.message


def test_auto_does_not_prune_with_unstable_calibration(monkeypatch):
    request = _request(
        objective=ObjectiveMode.AUTO,
        constructive=True,
        milp=True,
    )
    objectives = []

    def no_incumbent(problem, config, objective, warm_start):
        objectives.append(objective)
        return ()

    monkeypatch.setattr(
        "vericcl.solver.orchestrator.search_models",
        no_incumbent,
    )
    monkeypatch.setattr(
        "vericcl.solver.orchestrator.throughput_time_lower_bound",
        lambda problem, max_channels: LowerBound(1e9, 1e9),
    )
    monkeypatch.setattr(
        "vericcl.solver.orchestrator._relevant_calibration",
        lambda request, candidate: (0.1, False),
    )

    result = solve(request, cache=CandidateCache())

    assert set(objectives) == {
        ObjectiveMode.LATENCY,
        ObjectiveMode.THROUGHPUT,
    }
    assert "unstable_calibration" in result.message


def test_auto_keeps_latency_candidate_when_total_budget_expires(
    monkeypatch,
):
    request = _request(
        objective=ObjectiveMode.AUTO,
        constructive=True,
        milp=False,
    )
    times = iter((0.0, 0.0, 0.0, 0.0, 0.0, 11_000.0))

    monkeypatch.setattr(
        "vericcl.solver.orchestrator._monotonic",
        lambda: next(times),
    )
    monkeypatch.setattr(
        "vericcl.solver.orchestrator.throughput_time_lower_bound",
        lambda *args: pytest.fail("lower bound must respect the total budget"),
    )

    result = solve(request, cache=CandidateCache())

    assert result.status is SolveStatus.FEASIBLE
    assert result.selected_candidate is not None
    assert "budget_exhausted" in result.message


def test_invalid_solver_api_arguments_are_rejected():
    with pytest.raises(SemanticError, match="SolveRequest"):
        solve(object(), cache=CandidateCache())
    with pytest.raises(SemanticError, match="CandidateCache"):
        solve(_request(), cache=object())
