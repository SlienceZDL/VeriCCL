from dataclasses import replace
from pathlib import Path
import time

import pytest

from vericcl.errors import ConstructionInfeasibleError
from vericcl.input.loader import resolve_inputs
from vericcl.input.models import ObjectiveMode
from vericcl.planner.build import build_plan
from vericcl.semantics.collective import CollectiveKind, CollectiveSpec
from vericcl.solver.budget import ModelBudget
from vericcl.solver.cache import CandidateCache
from vericcl.solver.constructive import construct_route_pattern
from vericcl.solver.demands import build_solver_problem
from vericcl.solver.model import SolveRequest, SolveStatus
from vericcl.solver.orchestrator import solve
from vericcl.solver.routing_milp import _build_route_model
from vericcl.solver.template_search import search_route_models
from vericcl.solver.templates import (
    build_solver_templates,
    split_routing_units,
)
from vericcl.topology.loader import load_topology, topology_from_mapping

from tests.gurobi.helpers import require_gurobi_license


pytestmark = pytest.mark.phase03


EXAMPLES = Path(__file__).parents[2] / "vericcl" / "examples"


def _inputs(
    topology_name: str,
    kind: CollectiveKind,
    slice_count: int,
    *,
    hierarchy: bool,
):
    inputs = resolve_inputs(
        EXAMPLES / "topo" / topology_name,
        EXAMPLES / "sketch" / "allreduce_8m_1m.json",
        EXAMPLES / "atom" / "default.json",
    )
    return replace(
        inputs,
        collective=CollectiveSpec(
            kind=kind,
            datatype="float32",
            reduction_op=(
                "sum" if kind is CollectiveKind.ALL_REDUCE else None
            ),
        ),
        hyperparameters=replace(
            inputs.hyperparameters,
            total_size_bytes=slice_count * 1024,
            slice_size_bytes=1024,
        ),
        strategies=replace(inputs.strategies, hierarchy=hierarchy),
    )


def _complete_topology(rank_count: int):
    return topology_from_mapping(
        {
            "ranks": rank_count,
            "nodes": [
                {
                    "id": 0,
                    "ranks": list(range(rank_count)),
                    "gateways": [],
                }
            ],
            "directed_links": [
                {
                    "src": src,
                    "dst": dst,
                    "alpha": 1,
                    "invbw": 2,
                    "max_channels": 4,
                }
                for src in range(rank_count)
                for dst in range(rank_count)
                if src != dst
            ],
            "shared_resources": [],
        }
    )


def _template_structure(inputs, topology):
    plan = build_plan(inputs, topology)
    problems = tuple(
        build_solver_problem(node, inputs, topology) for node in plan.nodes
    )
    templates = build_solver_templates(problems, plan.planning_mode)
    unit_count = sum(
        len(split_routing_units(problem)) for problem in problems
    )
    return plan, problems, templates, unit_count


def test_direct_allgather_starts_one_route_model_per_root_for_fixed_k():
    inputs = _inputs(
        "two_rank.json",
        CollectiveKind.ALL_GATHER,
        128,
        hierarchy=False,
    )
    inputs = replace(inputs, rank_count=8)
    topology = _complete_topology(8)

    _, _, templates, unit_count = _template_structure(inputs, topology)

    assert unit_count == 8 * 128
    assert len(templates) == 8
    assert {len(template.members) for template in templates} == {128}


def test_direct_allgather_search_starts_eight_route_models_for_fixed_k(
    monkeypatch,
):
    inputs = _inputs(
        "two_rank.json",
        CollectiveKind.ALL_GATHER,
        128,
        hierarchy=False,
    )
    inputs = replace(
        inputs,
        rank_count=8,
        solver=replace(
            inputs.solver,
            max_channels=1,
            max_parallel_models=1,
            max_threads_per_model=1,
            force_resolve=True,
        ),
        strategies=replace(
            inputs.strategies,
            constructive_trees=False,
            milp=True,
        ),
    )
    topology = _complete_topology(8)
    plan, problems, templates, _ = _template_structure(inputs, topology)
    calls = []

    def solve_route(
        template,
        configured_inputs,
        topology_value,
        channel_count,
        objective,
        budget,
        warm_start,
    ):
        del budget, warm_start
        calls.append((template.template_id, channel_count, objective))
        if len(calls) == len(templates):
            raise ConstructionInfeasibleError("intentional incomplete K")
        return construct_route_pattern(
            template,
            configured_inputs,
            topology_value,
            channel_count,
            objective,
        )

    monkeypatch.setattr(
        "vericcl.solver.template_search.solve_route_milp",
        solve_route,
    )
    request = SolveRequest(
        inputs=inputs,
        topology=topology,
        plan=plan,
        solver_version="test-solver",
        model_version="test-model",
        environment_signature="test-environment",
    )

    result = search_route_models(
        request,
        problems,
        ObjectiveMode.LATENCY,
        deadline=time.monotonic() + 120.0,
        templates=templates,
    )

    assert len(calls) == len(templates) == 8
    assert result.candidates == ()
    assert result.diagnostics.route_model_count == 8
    assert result.diagnostics.search_model_count_total == 8


def test_gateway_allgather_template_count_is_slice_invariant():
    counts = []
    for slice_count in (1, 8):
        inputs = _inputs(
            "two_node_gateway.json",
            CollectiveKind.ALL_GATHER,
            slice_count,
            hierarchy=True,
        )
        topology = load_topology(inputs)

        plan, _, templates, unit_count = _template_structure(inputs, topology)

        counts.append((len(plan.nodes), len(templates)))
        assert sum(len(template.members) for template in templates) == (
            unit_count
        )

    assert counts == [counts[0]] * len(counts)


def test_gateway_allreduce_template_count_is_slice_invariant():
    counts = []
    for slice_count in (8, 16, 64, 128):
        inputs = _inputs(
            "two_node_gateway.json",
            CollectiveKind.ALL_REDUCE,
            slice_count,
            hierarchy=True,
        )
        topology = load_topology(inputs)

        plan, _, templates, unit_count = _template_structure(inputs, topology)

        counts.append((len(plan.nodes), len(templates)))
        assert sum(len(template.members) for template in templates) == (
            unit_count
        )

    assert counts == [counts[0]] * len(counts)


@pytest.mark.gurobi
def test_gateway_allreduce_route_model_shape_is_slice_invariant():
    require_gurobi_license()
    structures = []
    for slice_count in (8, 16, 64, 128):
        inputs = _inputs(
            "two_node_gateway.json",
            CollectiveKind.ALL_REDUCE,
            slice_count,
            hierarchy=True,
        )
        topology = load_topology(inputs)
        _, _, templates, _ = _template_structure(inputs, topology)
        current = []
        for template in templates:
            model, _, context = _build_route_model(
                template,
                inputs,
                topology,
                channel_count=1,
                objective=ObjectiveMode.LATENCY,
                budget=ModelBudget(
                    seconds=30.0,
                    started_at=0.0,
                    deadline=30.0,
                ),
                warm_start=None,
            )
            try:
                demand = template.representative.demands[0]
                current.append(
                    (
                        template.representative.node.node_id,
                        demand.root_rank,
                        demand.required_leaf_rank,
                        context.variable_count,
                        context.constraint_count,
                        context.general_constraint_count,
                    )
                )
            finally:
                model.dispose()
        structures.append(tuple(sorted(current)))

    assert structures == [structures[0]] * len(structures)


@pytest.mark.gurobi
def test_strict_optimal_request_uses_only_full_time_backend():
    require_gurobi_license()
    inputs = _inputs(
        "two_rank.json",
        CollectiveKind.ALL_REDUCE,
        2,
        hierarchy=False,
    )
    inputs = replace(
        inputs,
        solver=replace(
            inputs.solver,
            max_channels=1,
            max_parallel_models=1,
            max_threads_per_model=1,
            require_proven_optimal=True,
            force_resolve=True,
        ),
        strategies=replace(
            inputs.strategies,
            constructive_trees=False,
            milp=True,
        ),
    )
    topology = load_topology(inputs)
    plan = build_plan(inputs, topology)
    request = SolveRequest(
        inputs=inputs,
        topology=topology,
        plan=plan,
        solver_version="test-solver",
        model_version="test-model",
        environment_signature="test-environment",
    )

    result = solve(request, cache=CandidateCache())

    assert result.status is SolveStatus.ERROR
    assert result.selected_candidate is None
    assert result.candidates == ()
    assert result.message == "optimality proof was required but not obtained"
    assert result.diagnostics.template_count == 0
    assert result.diagnostics.route_model_count == 0
    assert result.diagnostics.search_model_count_total > 0
