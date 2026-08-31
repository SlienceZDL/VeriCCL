from concurrent.futures import ThreadPoolExecutor as RealThreadPoolExecutor
from dataclasses import replace

import pytest

from vericcl.errors import ConstructionInfeasibleError, SemanticError
from vericcl.input.models import ObjectiveMode
from vericcl.solver.demands import build_solver_problem
from vericcl.solver.instantiate import (
    InstantiationFailure,
    InstantiationResult,
    instantiate_route_patterns as real_instantiate_route_patterns,
)
from vericcl.solver.model import SolveRequest
from vericcl.solver.template_search import search_route_models
from vericcl.solver.templates import build_solver_templates

from tests.unit.solver.test_instantiate import _broadcast_fixture, _patterns
from tests.unit.solver.test_orchestrator import _request


pytestmark = pytest.mark.phase03


def _configured_request(*, max_channels, max_parallel_models, cpu_threads=12):
    request = _request(constructive=True, milp=True, force_resolve=True)
    inputs = replace(
        request.inputs,
        solver=replace(
            request.inputs.solver,
            max_channels=max_channels,
            max_parallel_models=max_parallel_models,
            max_threads_per_model=cpu_threads,
            total_solve_timeout_s=100,
            per_model_timeout_s=20,
        ),
    )
    return replace(request, inputs=inputs)


def _pattern(template, inputs, channel_count, objective):
    pattern = _patterns((template,), channel_count)[template.template_id]
    return replace(
        pattern,
        objective_mode=objective,
        metrics=replace(
            pattern.metrics,
            objective_values=(1.0, 1.0, 1.0),
            thread_count=inputs.solver.max_threads_per_model,
        ),
        model_stats=replace(
            pattern.model_stats,
            variable_count=3,
            constraint_count=4,
            general_constraint_count=5,
            build_time_s=0.25,
            optimize_time_s=0.5,
        ),
    )


def test_ready_template_models_share_cpu_and_report_actual_counts(monkeypatch):
    request = _configured_request(
        max_channels=2,
        max_parallel_models=3,
    )
    problems = tuple(
        build_solver_problem(node, request.inputs, request.topology)
        for node in request.plan.nodes
    )
    templates = build_solver_templates(problems, request.plan.planning_mode)
    calls = []
    worker_counts = []

    class RecordingExecutor(RealThreadPoolExecutor):
        def __init__(self, max_workers):
            worker_counts.append(max_workers)
            super().__init__(max_workers=max_workers)

    def fake_solve(
        template,
        inputs,
        topology,
        channel_count,
        objective,
        budget,
        warm_start=None,
    ):
        calls.append(
            (
                template.template_id,
                channel_count,
                objective,
                inputs.solver.max_threads_per_model,
                budget.seconds,
            )
        )
        return _pattern(template, inputs, channel_count, objective)

    monkeypatch.setattr(
        "vericcl.solver.template_search.ThreadPoolExecutor",
        RecordingExecutor,
    )
    monkeypatch.setattr(
        "vericcl.solver.template_search.solve_route_milp",
        fake_solve,
    )
    monkeypatch.setattr(
        "vericcl.solver.template_search.os.cpu_count",
        lambda: 4,
    )
    monkeypatch.setattr(
        "vericcl.solver.template_search._monotonic",
        lambda: 0.0,
    )

    result = search_route_models(
        request,
        problems,
        ObjectiveMode.LATENCY,
        deadline=100.0,
    )

    assert worker_counts == [3]
    assert len(calls) == len(templates) * 2
    assert {
        (template_id, channel_count)
        for template_id, channel_count, _, _, _ in calls
    } == {
        (template.template_id, channel_count)
        for template in templates
        for channel_count in (1, 2)
    }
    assert {call[3] for call in calls} == {1}
    assert all(call[2] is ObjectiveMode.LATENCY for call in calls)
    assert all(call[4] == pytest.approx(20.0) for call in calls)
    assert [candidate.channel_count for candidate in result.candidates] == [1, 2]
    assert {
        candidate.metrics.model_count for candidate in result.candidates
    } == {len(templates)}
    assert all(
        "template_route_composition" in candidate.restrictions
        for candidate in result.candidates
    )
    assert all(
        "independent_node_composition" in candidate.restrictions
        for candidate in result.candidates
    )
    assert result.diagnostics.requested_problem_count == len(problems)
    assert result.diagnostics.routing_unit_count == sum(
        len(template.members) for template in templates
    )
    assert result.diagnostics.template_count == len(templates)
    assert result.diagnostics.route_model_count == len(calls)
    assert result.diagnostics.search_model_count_total == len(calls)
    assert result.diagnostics.route_model_build_time_s == pytest.approx(2.0)
    assert result.diagnostics.route_model_optimize_time_s == pytest.approx(4.0)
    assert result.diagnostics.model_variables_max == 3
    assert result.diagnostics.model_constraints_max == 4
    assert result.diagnostics.model_general_constraints_max == 5


def test_failed_template_member_uses_one_independent_constructive_fallback(
    monkeypatch,
):
    inputs, topology, plan, templates, patterns = _broadcast_fixture()
    inputs = replace(
        inputs,
        solver=replace(
            inputs.solver,
            max_channels=1,
            max_parallel_models=1,
            max_threads_per_model=1,
        ),
    )
    request = SolveRequest(
        inputs=inputs,
        topology=topology,
        plan=plan,
        solver_version="test-solver",
        model_version="test-model",
        environment_signature="test-environment",
    )
    problems = tuple(
        build_solver_problem(node, inputs, topology) for node in plan.nodes
    )
    failed_member = templates[0].members[1]
    patterns = _patterns(templates, channel_count=1)
    original = real_instantiate_route_patterns(
        plan,
        templates,
        patterns,
        inputs,
        topology,
    )
    instantiation_calls = []
    route_calls = []

    def fake_solve(
        template,
        current_inputs,
        current_topology,
        channel_count,
        objective,
        budget,
        warm_start=None,
    ):
        route_calls.append(template.representative.unit_id)
        return _pattern(template, current_inputs, channel_count, objective)

    def fake_instantiate(
        current_plan,
        current_templates,
        current_patterns,
        current_inputs,
        current_topology,
    ):
        instantiation_calls.append(tuple(current_templates))
        if len(instantiation_calls) == 1:
            return InstantiationResult(
                original.node_schedules,
                (
                    InstantiationFailure(
                        unit_id=failed_member.unit_id,
                        node_id=failed_member.node_id,
                        reason="mapped_route_hits_forbidden_transfer",
                    ),
                ),
            )
        return original

    monkeypatch.setattr(
        "vericcl.solver.template_search.solve_route_milp",
        fake_solve,
    )
    monkeypatch.setattr(
        "vericcl.solver.template_search.instantiate_route_patterns",
        fake_instantiate,
    )
    monkeypatch.setattr(
        "vericcl.solver.template_search.construct_route_pattern",
        lambda *args: pytest.fail(
            "an incumbent member model must not use constructive fallback"
        ),
    )
    monkeypatch.setattr(
        "vericcl.solver.template_search._monotonic",
        lambda: 0.0,
    )

    result = search_route_models(
        request,
        problems,
        ObjectiveMode.LATENCY,
        deadline=100.0,
    )

    assert route_calls == [
        templates[0].representative.unit_id,
        failed_member.unit_id,
    ]
    assert len(instantiation_calls) == 2
    assert len(result.candidates) == 1
    assert result.candidates[0].metrics.model_count == 2
    assert result.diagnostics.route_model_count == 1
    assert result.diagnostics.fallback_member_model_count == 1
    assert result.diagnostics.search_model_count_total == 2


def test_incomplete_template_set_does_not_create_a_partial_k_candidate(
    monkeypatch,
):
    request = _configured_request(
        max_channels=2,
        max_parallel_models=2,
    )
    problems = tuple(
        build_solver_problem(node, request.inputs, request.topology)
        for node in request.plan.nodes
    )
    templates = build_solver_templates(problems, request.plan.planning_mode)
    blocked_template_id = templates[-1].template_id

    def fake_solve(
        template,
        inputs,
        topology,
        channel_count,
        objective,
        budget,
        warm_start=None,
    ):
        if channel_count == 2 and template.template_id == blocked_template_id:
            raise ConstructionInfeasibleError("no incumbent")
        return _pattern(template, inputs, channel_count, objective)

    monkeypatch.setattr(
        "vericcl.solver.template_search.solve_route_milp",
        fake_solve,
    )
    monkeypatch.setattr(
        "vericcl.solver.template_search.construct_route_pattern",
        lambda *args: (_ for _ in ()).throw(
            ConstructionInfeasibleError("constructive path unavailable")
        ),
    )
    monkeypatch.setattr(
        "vericcl.solver.template_search._monotonic",
        lambda: 0.0,
    )

    result = search_route_models(
        request,
        problems,
        ObjectiveMode.LATENCY,
        deadline=100.0,
    )

    assert [candidate.channel_count for candidate in result.candidates] == [1]
    assert result.candidates[0].metrics.model_count == len(templates)
    assert result.diagnostics.route_model_count == len(templates) * 2
    assert result.diagnostics.search_model_count_total == len(templates) * 2


def test_shared_deadline_preserves_completed_k_and_stops_new_models(
    monkeypatch,
):
    request = _configured_request(
        max_channels=2,
        max_parallel_models=1,
    )
    problems = tuple(
        build_solver_problem(node, request.inputs, request.topology)
        for node in request.plan.nodes
    )
    templates = build_solver_templates(problems, request.plan.planning_mode)
    calls = []
    clock_calls = 0

    def clock():
        nonlocal clock_calls
        clock_calls += 1
        return 0.0 if clock_calls <= len(templates) else 100.0

    def fake_solve(
        template,
        inputs,
        topology,
        channel_count,
        objective,
        budget,
        warm_start=None,
    ):
        calls.append((template.template_id, channel_count, budget.seconds))
        return _pattern(template, inputs, channel_count, objective)

    monkeypatch.setattr(
        "vericcl.solver.template_search.solve_route_milp",
        fake_solve,
    )
    monkeypatch.setattr(
        "vericcl.solver.template_search.os.cpu_count",
        lambda: 1,
    )
    monkeypatch.setattr(
        "vericcl.solver.template_search._monotonic",
        clock,
    )

    result = search_route_models(
        request,
        problems,
        ObjectiveMode.LATENCY,
        deadline=50.0,
    )

    assert len(calls) == len(templates)
    assert {channel_count for _, channel_count, _ in calls} == {1}
    assert all(seconds == pytest.approx(20.0) for _, _, seconds in calls)
    assert [candidate.channel_count for candidate in result.candidates] == [1]
    assert result.diagnostics.route_model_count == len(templates)
    assert result.diagnostics.search_model_count_total == len(templates)


def test_route_model_without_incumbent_constructs_only_the_representative(
    monkeypatch,
):
    inputs, topology, plan, templates, _ = _broadcast_fixture()
    inputs = replace(
        inputs,
        solver=replace(
            inputs.solver,
            max_channels=1,
            max_parallel_models=1,
            max_threads_per_model=1,
        ),
    )
    request = SolveRequest(
        inputs=inputs,
        topology=topology,
        plan=plan,
        solver_version="test-solver",
        model_version="test-model",
        environment_signature="test-environment",
    )
    problems = tuple(
        build_solver_problem(node, inputs, topology) for node in plan.nodes
    )
    constructed = []

    def no_incumbent(*args):
        raise ConstructionInfeasibleError("no incumbent")

    def fake_construct(
        template,
        current_inputs,
        current_topology,
        channel_count,
        objective,
    ):
        constructed.append(template.representative.unit_id)
        pattern = _pattern(
            template,
            current_inputs,
            channel_count,
            objective,
        )
        return replace(
            pattern,
            metrics=replace(
                pattern.metrics,
                model_count=0,
                solver_name="constructive-route",
            ),
        )

    monkeypatch.setattr(
        "vericcl.solver.template_search.solve_route_milp",
        no_incumbent,
    )
    monkeypatch.setattr(
        "vericcl.solver.template_search.construct_route_pattern",
        fake_construct,
    )
    monkeypatch.setattr(
        "vericcl.solver.template_search._monotonic",
        lambda: 0.0,
    )

    result = search_route_models(
        request,
        problems,
        ObjectiveMode.LATENCY,
        deadline=100.0,
    )

    assert constructed == [templates[0].representative.unit_id]
    assert len(result.candidates) == 1
    assert result.candidates[0].metrics.model_count == 0
    assert result.diagnostics.route_model_count == 1
    assert result.diagnostics.fallback_member_model_count == 0
    assert result.diagnostics.search_model_count_total == 1


def test_route_work_exception_identifies_objective_k_and_template(monkeypatch):
    inputs, topology, plan, templates, _ = _broadcast_fixture()
    inputs = replace(
        inputs,
        solver=replace(
            inputs.solver,
            max_channels=1,
            max_parallel_models=1,
        ),
    )
    request = SolveRequest(
        inputs=inputs,
        topology=topology,
        plan=plan,
        solver_version="test-solver",
        model_version="test-model",
        environment_signature="test-environment",
    )
    problems = tuple(
        build_solver_problem(node, inputs, topology) for node in plan.nodes
    )
    templates = build_solver_templates(problems, plan.planning_mode)
    monkeypatch.setattr(
        "vericcl.solver.template_search.solve_route_milp",
        lambda *args: (_ for _ in ()).throw(RuntimeError("worker exploded")),
    )
    monkeypatch.setattr(
        "vericcl.solver.template_search._monotonic",
        lambda: 0.0,
    )

    with pytest.raises(SemanticError) as captured:
        search_route_models(
            request,
            problems,
            ObjectiveMode.LATENCY,
            deadline=100.0,
        )

    message = str(captured.value)
    assert "objective=latency" in message
    assert "channel_count=1" in message
    assert templates[0].template_id in message
    assert "worker exploded" in message
