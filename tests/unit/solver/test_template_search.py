from concurrent.futures import ThreadPoolExecutor as RealThreadPoolExecutor
from dataclasses import replace
import time
from threading import Barrier, Lock, get_ident

import pytest

from vericcl.artifacts.hashing import candidate_signature
from vericcl.artifacts.writer import _sidecar_payload
from vericcl.composer import compose_routes
from vericcl.errors import (
    ConstructionInfeasibleError,
    SemanticError,
    SolverUnavailableError,
)
from vericcl.input.json_codec import canonical_json
from vericcl.input.models import ObjectiveMode
from vericcl.solver.constructive import construct_route_pattern
from vericcl.solver.demands import build_solver_problem
from vericcl.solver.instantiate import (
    InstantiationFailure,
    InstantiationResult,
    instantiate_route_patterns as real_instantiate_route_patterns,
)
from vericcl.solver.gurobi_api import GurobiAdapter
from vericcl.solver.model import SolveRequest
from vericcl.solver.template_search import (
    _candidate,
    _maximum_resource_load,
    search_route_models,
)
from vericcl.solver.templates import build_solver_templates
from vericcl.workflow import _global_schedule

from tests.unit.solver.test_global_scheduler import (
    _multi_resource_topology,
    _schedule,
    _transfer,
)
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


def test_global_resource_load_normalizes_link_and_shared_resource_capacity():
    topology = _multi_resource_topology(resource_count=2, slot_count=2)
    transfers = tuple(
        _transfer(
            "parallel-{}".format(index),
            slice_id=index,
            slice_count=2,
            path=((0, "SEND", ((0, 1, 0.0),)),),
            st_time=0.0,
            ed_time=1.0,
        )
        for index in range(2)
    )
    slots = {
        transfer.transfer_id: {
            "resource-00": index,
            "resource-01": index,
        }
        for index, transfer in enumerate(transfers)
    }
    schedule = _schedule(
        transfers,
        rank_count=2,
        slice_count=2,
        metadata={"resource_slots": slots},
    )

    assert _maximum_resource_load(schedule, topology, 2) == pytest.approx(1.0)

    constrained = replace(
        topology,
        shared_resources={
            resource_id: (
                replace(resource, max_channels=1)
                if resource_id == "resource-01"
                else resource
            )
            for resource_id, resource in topology.shared_resources.items()
        },
    )
    assert _maximum_resource_load(schedule, constrained, 2) == pytest.approx(2.0)


def test_candidate_hop_count_counts_each_instantiated_shared_tree_path():
    inputs, topology, plan, templates, patterns = _broadcast_fixture()
    patterns = _patterns(templates, channel_count=2)
    instantiated = real_instantiate_route_patterns(
        plan,
        templates,
        patterns,
        inputs,
        topology,
    )
    global_schedule = compose_routes(
        plan,
        instantiated.node_schedules,
        topology,
        2,
    )
    request = SolveRequest(
        inputs=inputs,
        topology=topology,
        plan=plan,
        solver_version="test-solver",
        model_version="test-model",
        environment_signature="test-environment",
    )
    expected_hops = sum(
        len(template.members)
        * sum(len(path) for _, path in patterns[template.template_id].member_paths)
        for template in templates
    )

    candidate = _candidate(
        request,
        ObjectiveMode.LATENCY,
        2,
        tuple(patterns[template.template_id] for template in templates),
        instantiated.node_schedules,
        global_schedule,
        len(templates),
    )

    assert expected_hops == 6
    assert candidate.metrics.operation_count == 4
    assert candidate.metrics.hop_count == expected_hops


def test_multistage_candidate_hop_count_sums_all_instantiated_paths():
    request = _request(
        objective=ObjectiveMode.LATENCY,
        constructive=True,
        milp=False,
        force_resolve=True,
    )
    problems = tuple(
        build_solver_problem(node, request.inputs, request.topology)
        for node in request.plan.nodes
    )
    templates = build_solver_templates(problems, request.plan.planning_mode)
    patterns = {
        template.template_id: construct_route_pattern(
            template,
            request.inputs,
            request.topology,
            1,
            ObjectiveMode.LATENCY,
        )
        for template in templates
    }
    expected_hops = sum(
        len(template.members)
        * sum(len(path) for _, path in patterns[template.template_id].member_paths)
        for template in templates
    )

    result = search_route_models(
        request,
        problems,
        ObjectiveMode.LATENCY,
        deadline=time.monotonic() + 100.0,
    )

    assert {node.stage_id for node in request.plan.nodes} == {0, 1}
    assert len(result.candidates) == 1
    assert result.candidates[0].metrics.hop_count == expected_hops


def test_template_candidate_sidecar_ignores_route_model_wall_time():
    inputs, topology, plan, templates, _ = _broadcast_fixture()
    patterns = _patterns(templates, channel_count=2)
    instantiated = real_instantiate_route_patterns(
        plan,
        templates,
        patterns,
        inputs,
        topology,
    )
    schedule = compose_routes(
        plan,
        instantiated.node_schedules,
        topology,
        2,
    )
    request = SolveRequest(
        inputs=inputs,
        topology=topology,
        plan=plan,
        solver_version="test-solver",
        model_version="test-model",
        environment_signature="test-environment",
    )

    def candidate_with_time(solve_time_s):
        values = tuple(
            replace(
                patterns[template.template_id],
                metrics=replace(
                    patterns[template.template_id].metrics,
                    solve_time_s=solve_time_s,
                ),
            )
            for template in templates
        )
        return _candidate(
            request,
            ObjectiveMode.LATENCY,
            2,
            values,
            instantiated.node_schedules,
            schedule,
            len(values),
        )

    first = candidate_with_time(0.25)
    second = candidate_with_time(91.0)
    signature = candidate_signature(schedule, inputs, topology, None)
    first_text = canonical_json(
        _sidecar_payload(inputs, first, schedule, signature, None, None)
    )
    second_text = canonical_json(
        _sidecar_payload(inputs, second, schedule, signature, None, None)
    )

    assert first.metrics.solve_time_s == 0.0
    assert second.metrics.solve_time_s == 0.0
    assert first_text == second_text


def test_workflow_reuses_template_global_schedule_without_recomposition(
    monkeypatch,
):
    request = _request(
        objective=ObjectiveMode.LATENCY,
        constructive=True,
        milp=False,
        force_resolve=True,
    )
    problems = tuple(
        build_solver_problem(node, request.inputs, request.topology)
        for node in request.plan.nodes
    )
    real_compose_routes = compose_routes
    calls = []

    def recording_compose_routes(*args, **kwargs):
        calls.append((args, kwargs))
        return real_compose_routes(*args, **kwargs)

    monkeypatch.setattr(
        "vericcl.composer.compose_routes",
        recording_compose_routes,
    )
    monkeypatch.setattr(
        "vericcl.workflow.compose_routes",
        recording_compose_routes,
    )

    result = search_route_models(
        request,
        problems,
        ObjectiveMode.LATENCY,
        deadline=time.monotonic() + 100.0,
    )
    candidate = result.candidates[0]
    schedule = _global_schedule(
        request.plan,
        candidate,
        request.topology,
    )

    assert len(calls) == 1
    assert schedule is candidate.global_schedule


def test_workflow_rejects_carried_schedule_for_changed_node_schedule():
    request = _request(
        objective=ObjectiveMode.LATENCY,
        constructive=True,
        milp=False,
        force_resolve=True,
    )
    problems = tuple(
        build_solver_problem(node, request.inputs, request.topology)
        for node in request.plan.nodes
    )
    result = search_route_models(
        request,
        problems,
        ObjectiveMode.LATENCY,
        deadline=time.monotonic() + 100.0,
    )
    candidate = result.candidates[0]
    node_id = next(iter(candidate.node_schedules))
    node_schedule = candidate.node_schedules[node_id]
    tampered = replace(
        candidate,
        node_schedules={
            **candidate.node_schedules,
            node_id: replace(
                node_schedule,
                metadata={
                    **node_schedule.metadata,
                    "backend": "changed-route-backend",
                },
            ),
        },
    )

    with pytest.raises(SemanticError, match="identity"):
        _global_schedule(request.plan, tampered, request.topology)


def test_parallel_route_workers_use_exclusive_explicit_gurobi_environments(
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
    main_thread = get_ident()
    barrier = Barrier(2)
    lock = Lock()
    environments = []
    model_threads = []
    active_environment_ids = set()
    maximum_active = 0

    class FakeGurobiError(Exception):
        pass

    class FakeEnvironment:
        def __init__(self, index):
            self.index = index
            self.created_thread = get_ident()
            self.disposed_thread = None
            self.in_use = False

        def setParam(self, name, value):
            assert (name, value) == ("OutputFlag", 0)

        def start(self):
            assert get_ident() == main_thread

        def dispose(self):
            assert not self.in_use
            self.disposed_thread = get_ident()

    class FakeGp:
        GurobiError = FakeGurobiError

        @staticmethod
        def Env(*, empty):
            assert empty is True
            environment = FakeEnvironment(len(environments))
            environments.append(environment)
            return environment

    class FakeModel:
        def __init__(self, environment):
            nonlocal maximum_active
            with lock:
                assert not environment.in_use
                environment.in_use = True
                active_environment_ids.add(environment.index)
                maximum_active = max(
                    maximum_active,
                    len(active_environment_ids),
                )
                model_threads.append(get_ident())
            self.environment = environment

        def dispose(self):
            with lock:
                assert self.environment.in_use
                self.environment.in_use = False
                active_environment_ids.remove(self.environment.index)

    def fake_solve(
        template,
        inputs,
        topology,
        channel_count,
        objective,
        budget,
        warm_start=None,
        environment=None,
    ):
        assert environment is not None
        model = FakeModel(environment)
        try:
            barrier.wait(timeout=5.0)
            return _pattern(template, inputs, channel_count, objective)
        finally:
            model.dispose()

    fake_solve.requires_explicit_environment = True
    monkeypatch.setattr(
        GurobiAdapter,
        "require",
        classmethod(lambda cls: FakeGp),
    )
    monkeypatch.setattr(
        "vericcl.solver.template_search.solve_route_milp",
        fake_solve,
    )
    monkeypatch.setattr(
        "vericcl.solver.template_search.os.cpu_count",
        lambda: 2,
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

    assert result.candidates
    assert len(environments) == 2
    assert {environment.created_thread for environment in environments} == {
        main_thread
    }
    assert set(model_threads).isdisjoint({main_thread})
    assert maximum_active == 2
    assert not active_environment_ids
    assert {environment.disposed_thread for environment in environments} == {
        main_thread
    }


def test_gurobi_environment_creation_failure_uses_constructive_fallback(
    monkeypatch,
):
    request = _configured_request(
        max_channels=1,
        max_parallel_models=2,
    )
    problems = tuple(
        build_solver_problem(node, request.inputs, request.topology)
        for node in request.plan.nodes
    )

    def forbidden_solve(*args, **kwargs):
        pytest.fail("route model must not start without an environment")

    forbidden_solve.requires_explicit_environment = True
    monkeypatch.setattr(
        "vericcl.solver.template_search.solve_route_milp",
        forbidden_solve,
    )
    monkeypatch.setattr(
        GurobiAdapter,
        "create_environment",
        classmethod(
            lambda cls: (_ for _ in ()).throw(
                SolverUnavailableError("environment unavailable")
            )
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

    assert len(result.candidates) == 1
    assert result.candidates[0].metrics.solver_name == "constructive-route"
    assert result.diagnostics.route_model_count == 0
    assert result.diagnostics.search_model_count_total == 0


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
