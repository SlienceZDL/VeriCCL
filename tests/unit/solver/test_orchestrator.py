from dataclasses import replace
from pathlib import Path

import pytest

from vericcl.input.loader import resolve_inputs
from vericcl.errors import SemanticError, SolverUnavailableError
from vericcl.input.models import ForbiddenTransfer, ObjectiveMode
from vericcl.planner.build import build_plan
from vericcl.semantics.collective import CollectiveKind, CollectiveSpec
from vericcl.solver.cache import CandidateCache
from vericcl.solver.lower_bounds import LowerBound
from vericcl.solver.model import SolveRequest, SolveStatus
from vericcl.solver.orchestrator import solve
from vericcl.solver.search import RouteSearchResult
import vericcl.solver.orchestrator as orchestrator_module
from vericcl.topology.loader import load_topology
from vericcl.tuning.model import TuningOverlay
from vericcl.verification import (
    ValidationStatus,
    verify_schedule_constraints,
    verify_schedule_semantics,
)

from tests.unit.planner.test_hierarchy import manual_allreduce
from tests.unit.solver.test_instantiate import _route_patterns


pytestmark = pytest.mark.phase03


EXAMPLES = Path(__file__).parents[3] / "vericcl" / "examples"


def _route_result(templates, objective, channel_counts=(1,)):
    patterns_by_channel = {}
    for channel_count in channel_counts:
        patterns = _route_patterns(templates, channel_count)
        patterns_by_channel[channel_count] = {
            template_id: replace(pattern, objective_mode=objective)
            for template_id, pattern in patterns.items()
        }
    model_count = len(templates) * len(channel_counts)
    return RouteSearchResult(
        patterns_by_channel=patterns_by_channel,
        launched_model_count=model_count,
        route_model_build_time_s=float(model_count),
        route_model_optimize_time_s=float(model_count * 2),
        maximum_variable_count=11,
        maximum_constraint_count=13,
        maximum_general_constraint_count=0,
        maximum_thread_count=1,
    )


def _empty_route_result(channel_counts=(1,)):
    return RouteSearchResult(
        patterns_by_channel={value: {} for value in channel_counts},
        launched_model_count=0,
        route_model_build_time_s=0.0,
        route_model_optimize_time_s=0.0,
        maximum_variable_count=0,
        maximum_constraint_count=0,
        maximum_general_constraint_count=0,
        maximum_thread_count=0,
    )


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


def _allgather_request(**kwargs):
    request = _request(**kwargs)
    inputs = replace(
        request.inputs,
        collective=CollectiveSpec(
            kind=CollectiveKind.ALL_GATHER,
            datatype="float32",
            reduction_op=None,
            root=None,
            inplace=False,
        ),
    )
    return replace(
        request,
        inputs=inputs,
        plan=build_plan(inputs, request.topology),
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


def test_explicit_workflow_budget_bounds_solver_deadline(monkeypatch):
    request = replace(
        _request(force_resolve=True),
        wall_clock_budget_s=2.5,
    )
    captured = []

    def no_candidates(request_value, problems, objective, deadline):
        captured.append(deadline)
        return (), orchestrator_module.SearchDiagnostics()

    monkeypatch.setattr(orchestrator_module, "_monotonic", lambda: 100.0)
    monkeypatch.setattr(
        orchestrator_module,
        "_solve_objective",
        no_candidates,
    )

    solve(request, cache=CandidateCache())

    assert captured == [pytest.approx(102.5)]


def test_scalable_route_search_without_incumbent_falls_back_to_constructive(
    monkeypatch,
):
    request = _request(constructive=True, milp=True)
    calls = []

    def no_incumbent(templates, config, objective, deadline, channel_counts=None):
        calls.append((tuple(templates), objective, channel_counts))
        return _empty_route_result(channel_counts or (1,))

    monkeypatch.setattr(
        "vericcl.solver.orchestrator.search_route_models",
        no_incumbent,
    )
    monkeypatch.setattr(
        "vericcl.solver.orchestrator.search_models",
        lambda *args: pytest.fail("non-proof solve used the full timing MILP"),
    )

    result = solve(request, cache=CandidateCache())

    assert len(calls) == 1
    assert calls[0][1] is ObjectiveMode.LATENCY
    assert result.selected_candidate is not None
    assert result.selected_candidate.metrics.solver_name == "constructive"


def test_unavailable_route_search_falls_back_to_constructive(monkeypatch):
    request = _request(constructive=True, milp=True)

    def unavailable(*args, **kwargs):
        raise SolverUnavailableError("unavailable")

    monkeypatch.setattr(
        "vericcl.solver.orchestrator.search_route_models",
        unavailable,
    )

    result = solve(request, cache=CandidateCache())

    assert result.status is SolveStatus.FEASIBLE
    assert result.selected_candidate.metrics.solver_name == "constructive"


def test_complete_route_patterns_create_restricted_non_proven_candidate(
    monkeypatch,
):
    request = _request(constructive=False, milp=True)

    monkeypatch.setattr(
        "vericcl.solver.orchestrator.GurobiAdapter.available",
        lambda: True,
    )
    monkeypatch.setattr(
        "vericcl.solver.orchestrator.search_route_models",
        lambda templates, config, objective, deadline, channel_counts=None: (
            _route_result(templates, objective, channel_counts or (1,))
        ),
    )
    monkeypatch.setattr(
        "vericcl.solver.orchestrator.search_models",
        lambda *args: pytest.fail("non-proof solve used the full timing MILP"),
    )

    result = solve(request, cache=CandidateCache())

    candidate = result.selected_candidate
    assert candidate is not None
    assert candidate.global_schedule is not None
    assert candidate.metrics.solver_name == "gurobi_route"
    assert candidate.metrics.model_count == result.diagnostics.template_count
    assert candidate.search_space_restricted
    assert not candidate.proven_optimal
    assert "template_route_composition" in candidate.restrictions
    assert result.diagnostics.route_model_count == result.diagnostics.template_count


def test_missing_template_invalidates_only_its_channel_count(monkeypatch):
    request = _request(constructive=False, milp=True)
    request = replace(
        request,
        inputs=replace(
            request.inputs,
            solver=replace(request.inputs.solver, max_channels=2),
        ),
    )

    def incomplete(templates, config, objective, deadline, channel_counts=None):
        result = _route_result(templates, objective, channel_counts or (1, 2))
        patterns = {
            channel_count: dict(values)
            for channel_count, values in result.patterns_by_channel.items()
        }
        patterns[1].pop(sorted(patterns[1])[0])
        return replace(result, patterns_by_channel=patterns)

    monkeypatch.setattr(
        "vericcl.solver.orchestrator.GurobiAdapter.available",
        lambda: True,
    )
    monkeypatch.setattr(
        "vericcl.solver.orchestrator.search_route_models",
        incomplete,
    )

    result = solve(request, cache=CandidateCache())

    assert [candidate.channel_count for candidate in result.candidates] == [2]
    assert result.diagnostics.route_model_count == (
        2 * result.diagnostics.template_count
    )


def test_failed_member_mapping_uses_only_one_standalone_fallback(monkeypatch):
    request = _allgather_request(constructive=False, milp=True)
    original_build = orchestrator_module.build_solver_templates

    def stale_member_templates(problems, planning_mode):
        templates = original_build(problems, planning_mode)
        target = next(template for template in templates if len(template.members) > 1)
        member = target.members[1]
        rank_map = list(member.rank_map)
        rank_map[0] = (rank_map[0][0], request.inputs.rank_count + 1)
        changed_member = replace(member, rank_map=tuple(rank_map))
        changed_template = replace(
            target,
            members=tuple(
                changed_member if value == member else value
                for value in target.members
            ),
        )
        return tuple(
            changed_template if value == target else value
            for value in templates
        )

    route_calls = []

    def complete(templates, config, objective, deadline, channel_counts=None):
        route_calls.append(tuple(template.template_id for template in templates))
        return _route_result(templates, objective, channel_counts or (1,))

    monkeypatch.setattr(
        "vericcl.solver.orchestrator.GurobiAdapter.available",
        lambda: True,
    )
    monkeypatch.setattr(
        "vericcl.solver.orchestrator.build_solver_templates",
        stale_member_templates,
    )
    monkeypatch.setattr(
        "vericcl.solver.orchestrator.search_route_models",
        complete,
    )

    result = solve(request, cache=CandidateCache())

    assert result.status is SolveStatus.FEASIBLE
    assert len(route_calls) == 2
    assert len(route_calls[1]) == 1
    assert route_calls[1][0].startswith("standalone-")
    assert result.diagnostics.fallback_member_model_count == 1
    assert result.diagnostics.route_model_count == (
        result.diagnostics.template_count + 1
    )
    schedule = result.selected_candidate.global_schedule
    assert schedule is not None
    assert verify_schedule_semantics(
        schedule,
        request.inputs,
    ).status is ValidationStatus.VALID
    assert verify_schedule_constraints(
        schedule,
        request.inputs,
        request.topology,
    ).status is ValidationStatus.VALID


def test_full_milp_total_budget_is_shared_across_plan_nodes(monkeypatch):
    request = _request(
        constructive=False,
        milp=True,
        require_proven_optimal=True,
    )
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


def test_proof_request_uses_full_timing_milp(monkeypatch):
    request = _request(
        constructive=True,
        milp=True,
        require_proven_optimal=True,
    )
    calls = []

    def no_incumbent(problem, config, objective, warm_start):
        calls.append(problem.node.node_id)
        return ()

    monkeypatch.setattr(
        "vericcl.solver.orchestrator.search_models",
        no_incumbent,
    )
    monkeypatch.setattr(
        "vericcl.solver.orchestrator.search_route_models",
        lambda *args, **kwargs: pytest.fail("proof solve used route-only MILP"),
    )

    result = solve(request, cache=CandidateCache())

    assert calls == [node.node_id for node in request.plan.nodes]
    assert result.status is SolveStatus.ERROR
    assert "proof" in result.message


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

    def no_incumbent(templates, config, objective, deadline, channel_counts=None):
        objectives.append(objective)
        return _empty_route_result(channel_counts or (1,))

    monkeypatch.setattr(
        "vericcl.solver.orchestrator.search_route_models",
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

    def no_incumbent(templates, config, objective, deadline, channel_counts=None):
        objectives.append(objective)
        return _empty_route_result(channel_counts or (1,))

    monkeypatch.setattr(
        "vericcl.solver.orchestrator.search_route_models",
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
