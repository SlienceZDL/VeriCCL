from dataclasses import replace
from pathlib import Path

import pytest

from vericcl.input.loader import resolve_inputs
from vericcl.input.models import ObjectiveMode
from vericcl.planner.build import build_plan
from vericcl.semantics.collective import CollectiveKind, CollectiveSpec
from vericcl.solver.cache import CandidateCache
from vericcl.solver.model import SolveRequest, SolveStatus
from vericcl.solver.orchestrator import solve
from vericcl.solver.routing import RoutePattern, RoutingModelStats
from vericcl.solver.search import RouteSearchResult
from vericcl.topology.loader import load_topology
from vericcl.topology.model import LinkKey
from vericcl.verification import (
    ValidationStatus,
    simulate_schedule,
    verify_schedule_constraints,
    verify_schedule_semantics,
)


pytestmark = pytest.mark.phase03


EXAMPLES = Path(__file__).parents[2] / "vericcl" / "examples"


def _request():
    inputs = resolve_inputs(
        EXAMPLES / "topo" / "two_rank.json",
        EXAMPLES / "sketch" / "allreduce_8m_1m.json",
        EXAMPLES / "atom" / "default.json",
    )
    inputs = replace(
        inputs,
        collective=CollectiveSpec(
            kind=CollectiveKind.ALL_GATHER,
            datatype="float32",
            reduction_op=None,
            root=None,
            inplace=False,
        ),
        hyperparameters=replace(
            inputs.hyperparameters,
            total_size_bytes=2 * inputs.hyperparameters.slice_size_bytes,
            objective_mode=ObjectiveMode.LATENCY,
        ),
        solver=replace(
            inputs.solver,
            max_channels=1,
            max_parallel_models=2,
            max_threads_per_model=1,
        ),
        strategies=replace(
            inputs.strategies,
            hierarchy=False,
            constructive_trees=False,
            milp=True,
        ),
    )
    topology = load_topology(inputs)
    return SolveRequest(
        inputs=inputs,
        topology=topology,
        plan=build_plan(inputs, topology),
        solver_version="integration-route-solver",
        model_version="integration-route-model",
        environment_signature="integration-route-environment",
    )


def _complete_route_search(
    templates,
    config,
    objective,
    deadline,
    channel_counts=None,
):
    del deadline
    channels = channel_counts or tuple(range(1, config.max_channels + 1))
    stats = RoutingModelStats(7, 9, 0, 0.01, 0.02)
    patterns_by_channel = {}
    for channel_count in channels:
        patterns = {}
        for template in templates:
            unit = template.representative
            root = unit.demands[0].root_rank
            parents = tuple(
                (root, rank)
                for rank in unit.node.communication_group
                if rank != root
            )
            selected = tuple(
                LinkKey(*unit.demands[0].physical_link(src_rank, dst_rank))
                for src_rank, dst_rank in parents
            )
            patterns[template.template_id] = RoutePattern(
                template_id=template.template_id,
                channel_count=channel_count,
                objective_mode=objective,
                selected_edges=selected,
                parent_edges=parents,
                model_stats=stats,
            )
        patterns_by_channel[channel_count] = patterns
    model_count = len(templates) * len(channels)
    return RouteSearchResult(
        patterns_by_channel=patterns_by_channel,
        launched_model_count=model_count,
        route_model_build_time_s=0.01 * model_count,
        route_model_optimize_time_s=0.02 * model_count,
        maximum_variable_count=7,
        maximum_constraint_count=9,
        maximum_general_constraint_count=0,
        maximum_thread_count=1,
    )


def test_scalable_pipeline_expands_and_schedules_complete_real_work(monkeypatch):
    request = _request()
    monkeypatch.setattr(
        "vericcl.solver.orchestrator.GurobiAdapter.available",
        lambda: True,
    )
    monkeypatch.setattr(
        "vericcl.solver.orchestrator.search_route_models",
        _complete_route_search,
    )
    monkeypatch.setattr(
        "vericcl.solver.orchestrator.search_models",
        lambda *args: pytest.fail("scalable solve used the full timing MILP"),
    )

    result = solve(request, cache=CandidateCache())

    assert result.status is SolveStatus.FEASIBLE
    candidate = result.selected_candidate
    assert candidate is not None
    schedule = candidate.global_schedule
    assert schedule is not None
    assert candidate.search_space_restricted
    assert not candidate.proven_optimal
    assert result.diagnostics.requested_problem_count == len(request.plan.nodes)
    assert result.diagnostics.route_model_count == result.diagnostics.template_count
    assert result.diagnostics.template_member_count > result.diagnostics.template_count
    assert verify_schedule_semantics(
        schedule,
        request.inputs,
    ).status is ValidationStatus.VALID
    assert verify_schedule_constraints(
        schedule,
        request.inputs,
        request.topology,
    ).status is ValidationStatus.VALID
    simulated = simulate_schedule(schedule, request.topology)
    assert simulated.completion_time_us == candidate.metrics.makespan_us
    assert len(schedule.transfers) == candidate.metrics.operation_count
