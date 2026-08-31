from dataclasses import replace

import pytest

from vericcl.errors import ConstructionInfeasibleError, SemanticError
from vericcl.input.models import AtomConstraints, ForbiddenTransfer, ObjectiveMode
from vericcl.planner.model import PlanningMode
from vericcl.solver.budget import ModelBudget
from vericcl.solver.demands import build_solver_problem
from vericcl.solver.model import SolveStatus
from vericcl.solver.routing_milp import solve_route_milp
from vericcl.solver.templates import build_solver_templates
from vericcl.topology.model import LinkKey

from tests.gurobi.helpers import (
    multihop_problem,
    reduction_dual_problem,
    require_gurobi_license,
    throughput_tradeoff_problem,
)


pytestmark = [pytest.mark.phase03, pytest.mark.gurobi]


def _template(problem):
    templates = build_solver_templates((problem,), PlanningMode.DIRECT)
    assert len(templates) == 1
    return templates[0]


def _solve(problem, objective, channel_count=2):
    return solve_route_milp(
        _template(problem),
        problem.inputs,
        problem.topology,
        channel_count=channel_count,
        objective=objective,
        budget=ModelBudget(seconds=30, started_at=0, deadline=30),
    )


def _path_nodes(path):
    return (path[0][0],) + tuple(edge[1] for edge in path)


def test_broadcast_route_reaches_each_leaf_with_one_parent_and_legal_flows():
    require_gurobi_license()
    problem = multihop_problem(shared_resource=True)

    pattern = _solve(problem, ObjectiveMode.LATENCY)

    demand_by_id = {
        demand.demand_id: demand
        for demand in _template(problem).representative.demands
    }
    parents = {}
    for src, dst in pattern.selected_edges:
        assert dst not in parents
        parents[dst] = src
    for demand_id, path in pattern.member_paths:
        demand = demand_by_id[demand_id]
        nodes = _path_nodes(path)
        assert nodes[0] == demand.root_rank
        assert nodes[-1] == demand.required_leaf_rank
        assert tuple(nodes) in demand.candidate_paths
        assert all(LinkKey(*edge) in demand.allowed_links for edge in path)
        assert all(LinkKey(*edge) in demand.legal_links for edge in path)
    assert pattern.selected_edges == ((0, 1), (1, 2))
    assert pattern.metrics.status in {SolveStatus.OPTIMAL, SolveStatus.FEASIBLE}
    assert pattern.metrics.makespan_us == 4.0
    assert pattern.metrics.maximum_normalized_resource_load == 4.0
    assert pattern.metrics.objective_values == (4.0, 2.0, 2.0)
    assert pattern.metrics.best_bound == pattern.metrics.objective_values[0]
    assert pattern.metrics.mip_gap >= 0.0
    assert pattern.model_stats.variable_count > 0
    assert pattern.model_stats.constraint_count > 0
    assert pattern.model_stats.general_constraint_count >= 0
    assert pattern.model_stats.build_time_s >= 0.0
    assert pattern.model_stats.optimize_time_s >= 0.0


def test_forbidden_direct_link_is_not_selected_even_when_topology_allows_it():
    require_gurobi_license()
    base = throughput_tradeoff_problem()
    inputs = replace(
        base.inputs,
        atom_constraints=AtomConstraints(
            stage_num=None,
            forbidden_transfers=(ForbiddenTransfer(0, 0, 2, 0),),
        ),
    )
    problem = build_solver_problem(base.node, inputs, base.topology)

    pattern = _solve(problem, ObjectiveMode.LATENCY)

    assert pattern.selected_edges == ((0, 1), (1, 2))
    assert (0, 2) not in pattern.selected_edges


def test_reduction_dual_uses_virtual_root_path_and_physical_link_direction():
    require_gurobi_license()
    problem = reduction_dual_problem()

    pattern = _solve(problem, ObjectiveMode.LATENCY, channel_count=1)

    demand = _template(problem).representative.demands[0]
    assert demand.reduction_dual
    assert pattern.selected_edges == ((0, 1),)
    assert demand.physical_link(0, 1) == (1, 0)
    assert LinkKey(1, 0) in problem.topology.links
    assert pattern.metrics.makespan_us == 2.0


def test_latency_and_throughput_optimize_distinct_route_quantities():
    require_gurobi_license()
    problem = throughput_tradeoff_problem()

    latency = _solve(problem, ObjectiveMode.LATENCY)
    throughput = _solve(problem, ObjectiveMode.THROUGHPUT)

    assert latency.selected_edges == ((0, 2),)
    assert latency.metrics.makespan_us == 4.0
    assert throughput.selected_edges == ((0, 1), (1, 2))
    assert throughput.metrics.maximum_normalized_resource_load == 1.25
    assert throughput.metrics.makespan_us == 5.0
    assert throughput.metrics.objective_values == (1.25, 5.0)


def test_auto_is_rejected_before_model_construction():
    problem = multihop_problem()

    with pytest.raises(SemanticError, match="AUTO"):
        _solve(problem, ObjectiveMode.AUTO)


def test_model_without_any_legal_candidate_path_returns_typed_failure():
    require_gurobi_license()
    base = throughput_tradeoff_problem()
    inputs = replace(
        base.inputs,
        atom_constraints=AtomConstraints(
            stage_num=None,
            forbidden_transfers=(
                ForbiddenTransfer(0, 0, 2, 0),
                ForbiddenTransfer(0, 0, 1, 0),
            ),
        ),
    )
    problem = build_solver_problem(base.node, inputs, base.topology)

    with pytest.raises(ConstructionInfeasibleError, match="candidate path"):
        _solve(problem, ObjectiveMode.LATENCY)
