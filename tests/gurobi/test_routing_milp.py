from dataclasses import replace

import pytest

from vericcl.input.models import ForbiddenTransfer, ObjectiveMode
from vericcl.planner.model import PlanningMode
from vericcl.solver.budget import ModelBudget
from vericcl.solver.routing_milp import _build_route_model, solve_route_milp
from vericcl.solver.templates import build_solver_templates
from vericcl.topology.model import LinkKey

from tests.gurobi.helpers import require_gurobi_license
from tests.unit.solver.test_templates import _inputs, _topology, _tree_problem


pytestmark = [pytest.mark.phase03, pytest.mark.gurobi]


def _budget():
    return ModelBudget(seconds=30, started_at=0, deadline=30)


def _template(edges, leaves, *, forbidden=()):
    edge_keys = frozenset(LinkKey(*edge) for edge in edges)
    ranks = tuple(
        sorted(
            {
                rank
                for edge in edge_keys
                for rank in (edge.src_rank, edge.dst_rank)
            }
        )
    )
    inputs = _inputs(
        len(ranks),
        1,
        forbidden=tuple(
            ForbiddenTransfer(0, src, dst, 0) for src, dst in forbidden
        ),
    )
    topology = _topology((ranks,), missing_links=tuple(
        LinkKey(src, dst)
        for src in ranks
        for dst in ranks
        if src != dst and LinkKey(src, dst) not in edge_keys
    ))
    problem = _tree_problem(inputs, topology, ranks, 0, 0)
    demands = tuple(
        demand
        for demand in problem.demands
        if demand.required_leaf_rank in leaves
    )
    problem = replace(
        problem,
        demands=demands,
        infeasible_demand_ids=tuple(
            demand.demand_id for demand in demands if not demand.candidate_paths
        ),
    )
    return build_solver_templates((problem,), PlanningMode.DIRECT)[0]


def _model(template, objective=ObjectiveMode.LATENCY):
    gp = __import__("gurobipy")
    model, variables = _build_route_model(
        gp,
        template.representative,
        channel_count=2,
        objective_mode=objective,
        budget=_budget(),
    )
    return gp, model, variables


def test_route_model_enforces_leaf_reachability_and_flow_conservation():
    require_gurobi_license()
    template = _template({(0, 1), (1, 2)}, {2})
    gp, model, variables = _model(template)
    demand = template.representative.demands[0]
    for link in demand.legal_links:
        if link.src_rank == demand.root_rank:
            model.addConstr(variables.flow_selected[(demand.demand_id, link)] == 0)

    model.optimize()

    assert model.Status == gp.GRB.INFEASIBLE
    model.dispose()


def test_route_model_enforces_one_parent_per_selected_rank():
    require_gurobi_license()
    template = _template(
        {(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)},
        {3, 4},
    )
    gp, model, variables = _model(template)
    model.addConstr(variables.edge_selected[LinkKey(1, 3)] == 1)
    model.addConstr(variables.edge_selected[LinkKey(2, 3)] == 1)

    model.optimize()

    assert model.Status == gp.GRB.INFEASIBLE
    model.dispose()


def test_route_model_uses_levels_to_reject_disconnected_directed_cycles():
    require_gurobi_license()
    template = _template({(0, 1), (2, 3), (3, 2)}, {1})
    gp, model, variables = _model(template)
    model.addConstr(variables.edge_selected[LinkKey(2, 3)] == 1)
    model.addConstr(variables.edge_selected[LinkKey(3, 2)] == 1)

    model.optimize()

    assert model.Status == gp.GRB.INFEASIBLE
    model.dispose()


def test_route_extraction_respects_directed_allowed_and_forbidden_links():
    require_gurobi_license()
    template = _template(
        {(0, 1), (1, 0), (0, 2), (1, 2)},
        {2},
        forbidden={(0, 2)},
    )

    pattern = solve_route_milp(
        template,
        channel_count=2,
        objective_mode=ObjectiveMode.LATENCY,
        budget=_budget(),
    )

    assert pattern.selected_edges == (LinkKey(0, 1), LinkKey(1, 2))
    assert pattern.parent_edges == ((0, 1), (1, 2))
    assert LinkKey(1, 0) not in pattern.selected_edges
    assert LinkKey(0, 2) not in pattern.selected_edges


def test_latency_minimizes_structural_critical_depth_then_route_size():
    require_gurobi_license()
    template = _template(
        {
            (0, 1),
            (1, 2),
            (1, 3),
            (0, 4),
            (4, 2),
            (0, 5),
            (5, 3),
        },
        {2, 3},
    )

    pattern = solve_route_milp(
        template,
        channel_count=2,
        objective_mode=ObjectiveMode.LATENCY,
        budget=_budget(),
    )

    assert pattern.parent_edges == ((0, 1), (1, 2), (1, 3))


def test_throughput_minimizes_maximum_normalized_edge_flow_load():
    require_gurobi_license()
    template = _template(
        {
            (0, 1),
            (1, 2),
            (1, 3),
            (0, 4),
            (4, 2),
            (0, 5),
            (5, 3),
        },
        {2, 3},
    )

    pattern = solve_route_milp(
        template,
        channel_count=2,
        objective_mode=ObjectiveMode.THROUGHPUT,
        budget=_budget(),
    )
    incoming = {dst: src for src, dst in pattern.parent_edges}

    assert len(pattern.parent_edges) == 4
    assert incoming[2] in {1, 4}
    assert incoming[3] in {1, 5}
    assert incoming[2] != incoming[3]


def test_route_model_stats_count_actual_gurobi_model_objects():
    require_gurobi_license()
    template = _template({(0, 1), (1, 2), (0, 2)}, {2})
    _, model, _ = _model(template)
    model.update()
    expected_counts = (
        model.NumVars,
        model.NumConstrs,
        model.NumGenConstrs,
    )
    model.dispose()

    pattern = solve_route_milp(
        template,
        channel_count=2,
        objective_mode=ObjectiveMode.LATENCY,
        budget=_budget(),
    )

    assert (
        pattern.model_stats.variable_count,
        pattern.model_stats.constraint_count,
        pattern.model_stats.general_constraint_count,
    ) == expected_counts
    assert pattern.model_stats.build_time_s >= 0.0
    assert pattern.model_stats.optimize_time_s >= 0.0
