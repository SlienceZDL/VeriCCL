from dataclasses import replace
from pathlib import Path

import pytest

from vericcl.input.loader import resolve_inputs
from vericcl.errors import (
    ConstructionInfeasibleError,
    SemanticError,
    SolverUnavailableError,
)
from vericcl.input.models import ForbiddenTransfer, ObjectiveMode
from vericcl.planner.build import build_plan
from vericcl.solver.cache import CandidateCache, build_cache_signature
from vericcl.solver.demands import build_solver_problem
from vericcl.solver.lower_bounds import LowerBound
from vericcl.solver.model import (
    SearchDiagnostics,
    SolveRequest,
    SolveStatus,
)
from vericcl.solver.orchestrator import solve
from vericcl.solver.template_search import TemplateSearchResult
from vericcl.solver.templates import build_solver_templates
import vericcl.solver.orchestrator as orchestrator_module
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
    assert result.selected_candidate.metrics.solver_name == "constructive-route"
    assert "template_route_composition" in (
        result.selected_candidate.restrictions
    )
    assert "independent_node_composition" in (
        result.selected_candidate.restrictions
    )


def test_explicit_workflow_budget_bounds_solver_deadline(monkeypatch):
    request = replace(
        _request(force_resolve=True),
        wall_clock_budget_s=2.5,
    )
    captured = []

    def no_candidates(
        request_value,
        problems,
        objective,
        deadline,
        *,
        templates,
        cache_key,
    ):
        del templates, cache_key
        captured.append(deadline)
        return TemplateSearchResult((), SearchDiagnostics())

    monkeypatch.setattr(orchestrator_module, "_monotonic", lambda: 100.0)
    monkeypatch.setattr(
        orchestrator_module,
        "_solve_template_objective",
        no_candidates,
    )

    solve(request, cache=CandidateCache())

    assert captured == [pytest.approx(102.5)]


def test_milp_without_incumbent_falls_back_to_constructive(
    monkeypatch,
):
    request = _request(constructive=True, milp=True)
    calls = []

    def no_incumbent(
        template,
        inputs,
        topology,
        channel_count,
        objective,
        budget,
        warm_start,
    ):
        calls.append((template.template_id, objective, warm_start))
        raise ConstructionInfeasibleError("no route incumbent")

    monkeypatch.setattr(
        "vericcl.solver.template_search.solve_route_milp",
        no_incumbent,
    )

    result = solve(request, cache=CandidateCache())

    assert calls
    assert result.diagnostics.route_model_count == len(calls)
    assert result.diagnostics.search_model_count_total == len(calls)
    assert all(call[2] is None for call in calls)
    assert result.selected_candidate is not None
    assert result.selected_candidate.metrics.solver_name == "constructive-route"
    assert "template_route_composition" in (
        result.selected_candidate.restrictions
    )


def test_unavailable_milp_search_falls_back_to_constructive(monkeypatch):
    request = _request(constructive=True, milp=True)

    def unavailable(*args):
        raise SolverUnavailableError("unavailable")

    monkeypatch.setattr(
        "vericcl.solver.template_search.solve_route_milp",
        unavailable,
    )

    result = solve(request, cache=CandidateCache())

    assert result.status is SolveStatus.FEASIBLE
    assert result.selected_candidate.metrics.solver_name == "constructive-route"
    assert result.diagnostics.route_model_count == 1
    assert result.diagnostics.search_model_count_total == 1


def test_complete_milp_incumbents_are_combined_by_channel(monkeypatch):
    request = _request(constructive=True, milp=True)
    from vericcl.solver import template_search

    original = template_search.construct_route_pattern
    calls = []

    def time_limited(
        template,
        inputs,
        topology,
        channel_count,
        objective,
        budget,
        warm_start,
    ):
        del budget, warm_start
        calls.append(template.template_id)
        pattern = original(
            template,
            inputs,
            topology,
            channel_count,
            objective,
        )
        return replace(
            pattern,
            metrics=replace(
                pattern.metrics,
                status=SolveStatus.TIME_LIMIT,
                solver_name="gurobi",
                model_count=1,
                termination_reason="time_limit",
            ),
        )

    monkeypatch.setattr(
        "vericcl.solver.template_search.solve_route_milp",
        time_limited,
    )

    result = solve(request, cache=CandidateCache())

    assert result.selected_candidate is not None
    assert result.selected_candidate.channel_count == 1
    assert result.selected_candidate.metrics.status is SolveStatus.TIME_LIMIT
    assert result.selected_candidate.metrics.solver_name == "gurobi"
    assert result.selected_candidate.metrics.model_count == len(calls)
    assert result.diagnostics.route_model_count == len(calls)


def test_strict_total_budget_is_shared_across_plan_nodes(monkeypatch):
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

    from vericcl.solver import template_search

    original = template_search.construct_route_pattern
    calls = []

    def counted(template, inputs, topology, channel_count, objective):
        calls.append((template.template_id, channel_count, objective))
        return original(
            template,
            inputs,
            topology,
            channel_count,
            objective,
        )

    monkeypatch.setattr(template_search, "construct_route_pattern", counted)
    forced_inputs = replace(
        request.inputs,
        solver=replace(request.inputs.solver, force_resolve=True),
    )
    forced = replace(request, inputs=forced_inputs)

    result = solve(forced, cache=cache)

    assert not result.cache_hit
    assert calls


def test_cache_hit_preserves_structure_and_zeros_execution_diagnostics():
    cache = CandidateCache()
    request = _request()

    cold = solve(request, cache=cache)
    historical = SearchDiagnostics(
        requested_problem_count=cold.diagnostics.requested_problem_count,
        routing_unit_count=cold.diagnostics.routing_unit_count,
        template_count=cold.diagnostics.template_count,
        template_member_count=cold.diagnostics.template_member_count,
        route_model_count=7,
        fallback_member_model_count=2,
        search_model_count_total=11,
        route_model_build_time_s=1.25,
        route_model_optimize_time_s=2.5,
        template_expansion_time_s=3.5,
        global_scheduling_time_s=4.5,
        model_variables_max=13,
        model_constraints_max=17,
        model_general_constraints_max=19,
    )
    cache_key = next(iter(cache._entries))
    cache.put(
        cache_key,
        cold.selected_candidate,
        ttl_seconds=10,
        complete=True,
        diagnostics=historical,
    )
    hot = solve(request, cache=cache)

    assert not cold.cache_hit
    assert hot.cache_hit
    assert hot.selected_candidate is not None
    assert hot.selected_candidate.metrics == cold.selected_candidate.metrics
    for field in (
        "requested_problem_count",
        "routing_unit_count",
        "template_count",
        "template_member_count",
    ):
        assert getattr(hot.diagnostics, field) == getattr(
            historical,
            field,
        )
    assert hot.diagnostics.requested_problem_count > 0
    assert hot.diagnostics.template_count > 0
    assert hot.diagnostics.route_model_build_time_s == 0.0
    assert hot.diagnostics.route_model_optimize_time_s == 0.0
    assert hot.diagnostics.template_expansion_time_s == 0.0
    assert hot.diagnostics.global_scheduling_time_s == 0.0
    assert hot.diagnostics.route_model_count == 0
    assert hot.diagnostics.fallback_member_model_count == 0
    assert hot.diagnostics.search_model_count_total == 0
    assert hot.diagnostics.model_variables_max == 0
    assert hot.diagnostics.model_constraints_max == 0
    assert hot.diagnostics.model_general_constraints_max == 0


def test_auto_skips_throughput_when_cv_adjusted_gain_is_too_small(
    monkeypatch,
):
    request = _request(
        objective=ObjectiveMode.AUTO,
        constructive=True,
        milp=False,
        force_resolve=True,
    )
    objectives = []

    def template_pipeline(
        request_value,
        problems,
        objective,
        deadline,
        *,
        templates,
        cache_key,
    ):
        del templates, cache_key
        del problems, deadline
        objectives.append(objective)
        return TemplateSearchResult(
            (_backend_candidate(request_value, objective, proven=False),),
            SearchDiagnostics(
                requested_problem_count=len(request_value.plan.nodes),
                template_count=2,
                route_model_count=2,
                search_model_count_total=2,
            ),
        )

    monkeypatch.setattr(
        orchestrator_module,
        "_solve_template_objective",
        template_pipeline,
    )
    monkeypatch.setattr(
        orchestrator_module,
        "global_throughput_time_lower_bound",
        lambda problems, max_channels: LowerBound(1e9, 1e9),
    )
    monkeypatch.setattr(
        "vericcl.solver.orchestrator._relevant_calibration",
        lambda request, candidate: (0.1, True),
    )

    result = solve(request, cache=CandidateCache())

    assert objectives
    assert set(objectives) == {ObjectiveMode.LATENCY}
    assert result.diagnostics.search_model_count_total == 2
    assert "auto_throughput=skipped" in result.message
    assert "threshold=0.2" in result.message


def test_auto_does_not_prune_with_unstable_calibration(monkeypatch):
    request = _request(
        objective=ObjectiveMode.AUTO,
        constructive=True,
        milp=False,
        force_resolve=True,
    )
    objectives = []

    def template_pipeline(
        request_value,
        problems,
        objective,
        deadline,
        *,
        templates,
        cache_key,
    ):
        del templates, cache_key
        del problems, deadline
        objectives.append(objective)
        count = 2 if objective is ObjectiveMode.LATENCY else 3
        return TemplateSearchResult(
            (_backend_candidate(request_value, objective, proven=False),),
            SearchDiagnostics(
                requested_problem_count=len(request_value.plan.nodes),
                template_count=2,
                route_model_count=count,
                search_model_count_total=count,
                route_model_build_time_s=float(count),
            ),
        )

    monkeypatch.setattr(
        orchestrator_module,
        "_solve_template_objective",
        template_pipeline,
    )
    monkeypatch.setattr(
        orchestrator_module,
        "global_throughput_time_lower_bound",
        lambda problems, max_channels: LowerBound(1e9, 1e9),
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
    assert result.diagnostics.template_count == 2
    assert result.diagnostics.route_model_count == 5
    assert result.diagnostics.search_model_count_total == 5
    assert result.diagnostics.route_model_build_time_s == 5.0
    assert "unstable_calibration" in result.message


def test_auto_keeps_latency_candidate_when_total_budget_expires(
    monkeypatch,
):
    request = _request(
        objective=ObjectiveMode.AUTO,
        constructive=True,
        milp=False,
        force_resolve=True,
    )
    times = iter((0.0, 11_000.0))

    def latency_only(
        request_value,
        problems,
        objective,
        deadline,
        *,
        templates,
        cache_key,
    ):
        del templates, cache_key
        del problems, deadline
        assert objective is ObjectiveMode.LATENCY
        return TemplateSearchResult(
            (_backend_candidate(request_value, objective, proven=False),),
            SearchDiagnostics(
                requested_problem_count=len(request_value.plan.nodes),
                route_model_count=3,
                search_model_count_total=3,
            ),
        )

    monkeypatch.setattr(
        orchestrator_module,
        "_monotonic",
        lambda: next(times),
    )
    monkeypatch.setattr(
        orchestrator_module,
        "_solve_template_objective",
        latency_only,
    )
    monkeypatch.setattr(
        orchestrator_module,
        "global_throughput_time_lower_bound",
        lambda *args: pytest.fail("lower bound must respect the total budget"),
    )

    result = solve(request, cache=CandidateCache())

    assert result.status is SolveStatus.FEASIBLE
    assert result.selected_candidate is not None
    assert result.diagnostics.search_model_count_total == 3
    assert "budget_exhausted" in result.message


def test_invalid_solver_api_arguments_are_rejected():
    with pytest.raises(SemanticError, match="SolveRequest"):
        solve(object(), cache=CandidateCache())
    with pytest.raises(SemanticError, match="CandidateCache"):
        solve(_request(), cache=object())


def _backend_candidate(request, objective, *, proven):
    problems = tuple(
        orchestrator_module.build_solver_problem(
            node,
            request.inputs,
            request.topology,
        )
        for node in request.plan.nodes
    )
    local = {}
    for problem in problems:
        schedule = orchestrator_module.construct_candidate(problem, 1)
        local[problem.node.node_id] = orchestrator_module._constructive_candidate(
            problem,
            schedule,
            objective,
            1,
            None,
        )
    candidate = orchestrator_module._combine_node_candidates(
        request,
        local,
        "test",
        objective,
        1,
    )
    if proven:
        return replace(
            candidate,
            candidate_id="legacy-proven",
            metrics=replace(candidate.metrics, status=SolveStatus.OPTIMAL),
            proven_optimal=True,
            search_space_restricted=False,
            restrictions=(),
        )
    restrictions = tuple(
        sorted(set(candidate.restrictions) | {"template_route_composition"})
    )
    return replace(
        candidate,
        candidate_id="template-{}-candidate".format(objective.value),
        proven_optimal=False,
        search_space_restricted=True,
        restrictions=restrictions,
    )


def test_default_and_strict_requests_use_disjoint_solver_paths(monkeypatch):
    default = _request(force_resolve=True)
    strict = _request(
        constructive=False,
        milp=True,
        force_resolve=True,
        require_proven_optimal=True,
    )
    calls = []

    def template_pipeline(
        request,
        problems,
        objective,
        deadline,
        *,
        templates,
        cache_key,
    ):
        del templates, cache_key
        calls.append("template_route_pipeline")
        return TemplateSearchResult(
            (_backend_candidate(request, objective, proven=False),),
            SearchDiagnostics(
                requested_problem_count=len(problems),
                route_model_count=1,
                search_model_count_total=1,
            ),
        )

    def legacy_pipeline(
        request,
        problems,
        objective,
        deadline,
        *,
        cache_key,
    ):
        del cache_key
        calls.append("legacy_full_time_milp")
        return TemplateSearchResult(
            (_backend_candidate(request, objective, proven=True),),
            SearchDiagnostics(
                requested_problem_count=len(problems),
                search_model_count_total=1,
            ),
        )

    monkeypatch.setattr(
        orchestrator_module,
        "_solve_template_objective",
        template_pipeline,
    )
    monkeypatch.setattr(
        orchestrator_module,
        "_solve_legacy_objective",
        legacy_pipeline,
    )
    monkeypatch.setattr(
        orchestrator_module.GurobiAdapter,
        "available",
        lambda: True,
    )

    default_result = solve(default, cache=CandidateCache())

    assert calls == ["template_route_pipeline"]
    assert default_result.selected_candidate is not None
    assert "template_route_composition" in (
        default_result.selected_candidate.restrictions
    )
    assert not default_result.selected_candidate.proven_optimal
    assert default_result.diagnostics.search_model_count_total == 1

    calls.clear()
    strict_result = solve(strict, cache=CandidateCache())

    assert calls == ["legacy_full_time_milp"]
    assert strict_result.selected_candidate is not None
    assert strict_result.selected_candidate.proven_optimal
    assert strict_result.selected_candidate.metrics.status is SolveStatus.OPTIMAL
    assert strict_result.selected_candidate.restrictions == ()
    assert strict_result.diagnostics.search_model_count_total == 1


def test_solver_paths_use_distinct_cache_backend_signatures():
    default = _request()
    strict = _request(require_proven_optimal=True)
    default_problems = tuple(
        build_solver_problem(node, default.inputs, default.topology)
        for node in default.plan.nodes
    )
    strict_problems = tuple(
        build_solver_problem(node, strict.inputs, strict.topology)
        for node in strict.plan.nodes
    )
    default_signature = build_cache_signature(
        default,
        default_problems,
        build_solver_templates(
            default_problems,
            default.plan.planning_mode,
        ),
    )
    strict_signature = build_cache_signature(strict, strict_problems, ())

    assert default_signature.backend_type == "template_route"
    assert strict_signature.backend_type == "legacy_full_time_milp"
    assert default_signature != strict_signature


def test_solver_prepares_templates_once_per_solve(monkeypatch):
    from vericcl.solver import template_search

    request = _request(force_resolve=True)
    original = build_solver_templates
    calls = []

    def counted(problems, planning_mode):
        calls.append((tuple(problems), planning_mode))
        return original(problems, planning_mode)

    monkeypatch.setattr(
        orchestrator_module,
        "build_solver_templates",
        counted,
    )
    monkeypatch.setattr(
        template_search,
        "build_solver_templates",
        counted,
    )

    result = solve(request, cache=CandidateCache())

    assert result.selected_candidate is not None
    assert len(calls) == 1


def test_legacy_candidate_id_uses_authoritative_cache_key():
    request = _request(require_proven_optimal=True)
    problems = tuple(
        build_solver_problem(node, request.inputs, request.topology)
        for node in request.plan.nodes
    )
    local = {}
    for problem in problems:
        schedule = orchestrator_module.construct_candidate(problem, 1)
        local[problem.node.node_id] = (
            orchestrator_module._constructive_candidate(
                problem,
                schedule,
                ObjectiveMode.LATENCY,
                1,
                None,
            )
        )
    signature = build_cache_signature(request, problems, ())
    from vericcl.solver.cache import candidate_cache_key

    cache_key = candidate_cache_key(request, signature)

    combined = orchestrator_module._combine_node_candidates(
        request,
        local,
        "milp",
        ObjectiveMode.LATENCY,
        1,
        cache_key=cache_key,
    )

    assert combined.candidate_id.endswith(cache_key[:12])
