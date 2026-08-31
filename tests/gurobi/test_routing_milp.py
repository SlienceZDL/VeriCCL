from dataclasses import replace

import pytest
import vericcl.solver.routing_milp as routing_milp

from vericcl.errors import ConstructionInfeasibleError, SemanticError
from vericcl.input.loader import resolve_inputs
from vericcl.input.models import AtomConstraints, ForbiddenTransfer, ObjectiveMode
from vericcl.planner.model import (
    PlanNode,
    PlanningMode,
    StageInterface,
)
from vericcl.semantics.collective import (
    CollectiveKind,
    CollectiveSpec,
    OutputSlot,
)
from vericcl.solver.budget import ModelBudget
from vericcl.solver.demands import build_solver_problem
from vericcl.solver.model import SolveStatus
from vericcl.solver.routing_milp import solve_route_milp
from vericcl.solver.templates import build_solver_templates
from vericcl.topology.loader import topology_from_mapping
from vericcl.topology.model import LinkKey

from tests.gurobi.helpers import (
    EXAMPLES,
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


def _inputs(collective, rank_count):
    base = resolve_inputs(
        EXAMPLES / "topo" / "two_rank.json",
        EXAMPLES / "sketch" / "allreduce_8m_1m.json",
        EXAMPLES / "atom" / "default.json",
    )
    return replace(
        base,
        collective=collective,
        rank_count=rank_count,
        hyperparameters=replace(
            base.hyperparameters,
            total_size_bytes=1024,
            slice_size_bytes=1024,
        ),
        strategies=replace(
            base.strategies,
            shortest_paths=False,
        ),
    )


def _broadcast_problem(
    node_id,
    rank_count,
    link_specs,
    candidate_paths_by_leaf,
):
    collective = CollectiveSpec(
        kind=CollectiveKind.BROADCAST,
        datatype="float32",
        root=0,
    )
    inputs = _inputs(collective, rank_count=rank_count)
    topology = topology_from_mapping(
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
                    "alpha": 0,
                    "invbw": invbw,
                    "max_channels": 2,
                }
                for src, dst, invbw in link_specs
            ],
            "shared_resources": [],
        }
    )
    contributors = frozenset({0})
    node = PlanNode(
        node_id=node_id,
        stage_id=0,
        local_collective=collective,
        communication_group=tuple(range(rank_count)),
        logical_input=StageInterface({OutputSlot(0, 0): contributors}),
        logical_output=StageInterface(
            {OutputSlot(0, 0): contributors}
            | {
                OutputSlot(leaf, 0): contributors
                for leaf in candidate_paths_by_leaf
            }
        ),
        allowed_links=frozenset(topology.links),
        shared_resource_ids=frozenset(),
    )
    problem = build_solver_problem(node, inputs, topology)
    return replace(
        problem,
        demands=tuple(
            replace(
                demand,
                candidate_paths=candidate_paths_by_leaf[
                    demand.required_leaf_rank
                ],
            )
            for demand in problem.demands
        ),
    )


def _single_parent_conflict_problem():
    return _broadcast_problem(
        node_id="route-single-parent-conflict",
        rank_count=7,
        link_specs=(
            (0, 1, 1),
            (0, 2, 0.5),
            (0, 6, 2),
            (1, 3, 1),
            (2, 3, 0.5),
            (3, 4, 0.5),
            (3, 5, 1),
            (6, 4, 2),
        ),
        candidate_paths_by_leaf={
            4: ((0, 1, 3, 4), (0, 6, 4)),
            5: ((0, 1, 3, 5), (0, 2, 3, 5)),
        },
    )


def _equal_latency_operation_problem():
    return _broadcast_problem(
        node_id="route-equal-latency-operation",
        rank_count=6,
        link_specs=(
            (0, 1, 1),
            (0, 2, 1),
            (0, 3, 1),
            (1, 4, 1),
            (1, 5, 1),
            (2, 4, 1),
            (3, 5, 1),
        ),
        candidate_paths_by_leaf={
            4: ((0, 1, 4), (0, 2, 4)),
            5: ((0, 1, 5), (0, 3, 5)),
        },
    )


def _build_test_route_model(problem):
    return routing_milp._build_route_model(
        _template(problem),
        problem.inputs,
        problem.topology,
        channel_count=1,
        objective=ObjectiveMode.LATENCY,
        budget=ModelBudget(seconds=30, started_at=0, deadline=30),
        warm_start=None,
    )


def _selected_candidate_paths(variables, context):
    return {
        demand_id: context.candidate_paths[demand_id][path_index]
        for (demand_id, path_index), variable
        in variables.path_selected.items()
        if variable.X > 0.5
    }


def _fixed_objective_values(problem, paths_by_leaf):
    model, variables, context = _build_test_route_model(problem)
    try:
        demand_by_id = {
            demand.demand_id: demand for demand in context.demands
        }
        for (demand_id, path_index), variable in (
            variables.path_selected.items()
        ):
            selected = (
                context.candidate_paths[demand_id][path_index]
                == paths_by_leaf[
                    demand_by_id[demand_id].required_leaf_rank
                ]
            )
            variable.LB = float(selected)
            variable.UB = float(selected)
        model.optimize()
        assert model.Status == context.gp.GRB.OPTIMAL
        values = []
        for objective_index in range(model.NumObj):
            model.Params.ObjNumber = objective_index
            values.append(float(model.ObjNVal))
        return tuple(values)
    finally:
        model.dispose()


def _branching_reduction_dual_problem():
    collective = CollectiveSpec(
        kind=CollectiveKind.REDUCE,
        datatype="float32",
        reduction_op="sum",
        root=0,
    )
    inputs = _inputs(collective, rank_count=4)
    topology = topology_from_mapping(
        {
            "ranks": 4,
            "nodes": [{"id": 0, "ranks": [0, 1, 2, 3], "gateways": []}],
            "directed_links": [
                {
                    "src": src,
                    "dst": dst,
                    "alpha": 0,
                    "invbw": 1,
                    "max_channels": 2,
                    "resources": ["reduce-fabric"],
                }
                for src, dst in ((1, 0), (2, 1), (3, 1))
            ],
            "shared_resources": [
                {
                    "id": "reduce-fabric",
                    "member_links": [[1, 0], [2, 1], [3, 1]],
                    "alpha": 0,
                    "invbw": 1,
                    "max_channels": 2,
                }
            ],
        }
    )
    node = PlanNode(
        node_id="route-branching-reduction-dual",
        stage_id=0,
        local_collective=collective,
        communication_group=(0, 1, 2, 3),
        logical_input=StageInterface(
            {
                OutputSlot(0, 0): frozenset({0}),
                OutputSlot(1, 0): frozenset({1}),
                OutputSlot(2, 0): frozenset({2}),
                OutputSlot(3, 0): frozenset({3}),
            }
        ),
        logical_output=StageInterface(
            {OutputSlot(0, 0): frozenset({0, 1, 2, 3})}
        ),
        allowed_links=frozenset(topology.links),
        shared_resource_ids=frozenset({"reduce-fabric"}),
    )
    return build_solver_problem(node, inputs, topology)


class _CallbackSnapshot:
    def __init__(self, values):
        self._values = values

    def cbGet(self, key):
        return self._values[key]


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


def test_single_parent_constraint_selects_the_shared_feasible_tree():
    require_gurobi_license()
    problem = _single_parent_conflict_problem()

    pattern = _solve(
        problem,
        ObjectiveMode.LATENCY,
        channel_count=1,
    )

    representative = _template(problem).representative
    demand_by_id = {
        demand.demand_id: demand for demand in representative.demands
    }
    paths_by_leaf = {
        demand_by_id[demand_id].required_leaf_rank: _path_nodes(path)
        for demand_id, path in pattern.member_paths
    }
    assert paths_by_leaf == {
        4: (0, 1, 3, 4),
        5: (0, 1, 3, 5),
    }
    assert pattern.selected_edges == ((0, 1), (1, 3), (3, 4), (3, 5))
    parents = {}
    for src, dst in pattern.selected_edges:
        assert dst not in parents
        parents[dst] = src
    assert parents == {1: 0, 3: 1, 4: 3, 5: 3}
    for _, path in pattern.member_paths:
        nodes = _path_nodes(path)
        assert len(nodes) == len(set(nodes))
        assert all(
            first[1] == second[0]
            for first, second in zip(path, path[1:])
        )
    assert pattern.metrics.objective_values == (3.0, 4.0, 6.0)
    assert pattern.metrics.status is SolveStatus.OPTIMAL
    assert pattern.metrics.best_bound == 3.0
    assert pattern.metrics.mip_gap == 0.0
    assert pattern.metrics.within_requested_gap
    assert pattern.metrics.makespan_us == 3.0
    assert pattern.metrics.operation_count == len(pattern.selected_edges) == 4
    assert pattern.metrics.hop_count == 6
    assert pattern.metrics.maximum_normalized_resource_load == 1.0


def test_single_parent_fixture_exposes_conflict_when_constraint_is_removed():
    require_gurobi_license()
    problem = _single_parent_conflict_problem()
    model, variables, context = _build_test_route_model(problem)
    try:
        constraint = model.getConstrByName(
            "tree-parent-at-most-one-r0003"
        )
        assert constraint is not None
        model.remove(constraint)
        model.update()
        model.optimize()

        assert model.Status == context.gp.GRB.OPTIMAL
        selected = _selected_candidate_paths(variables, context)
        demand_by_id = {
            demand.demand_id: demand for demand in context.demands
        }
        assert {
            demand_by_id[demand_id].required_leaf_rank: path
            for demand_id, path in selected.items()
        } == {
            4: (0, 1, 3, 4),
            5: (0, 2, 3, 5),
        }
        selected_edges = {
            (link.src_rank, link.dst_rank)
            for link, variable in variables.edge_selected.items()
            if variable.X > 0.5
        }
        assert {
            src for src, dst in selected_edges if dst == 3
        } == {1, 2}
        assert variables.route_completion.X == pytest.approx(2.5, abs=1e-3)
    finally:
        model.dispose()


def test_operation_count_breaks_an_equal_latency_tie_with_a_shared_tree():
    require_gurobi_license()
    problem = _equal_latency_operation_problem()

    pattern = _solve(
        problem,
        ObjectiveMode.LATENCY,
        channel_count=1,
    )

    assert pattern.selected_edges == ((0, 1), (1, 4), (1, 5))
    assert pattern.metrics.objective_values == (2.0, 3.0, 4.0)
    assert pattern.metrics.status is SolveStatus.OPTIMAL
    assert pattern.metrics.makespan_us == 2.0
    assert pattern.metrics.operation_count == len(pattern.selected_edges) == 3
    assert pattern.metrics.hop_count == 4


def test_operation_objective_is_the_exact_unique_tree_edge_sum():
    require_gurobi_license()
    problem = _equal_latency_operation_problem()

    assert _fixed_objective_values(
        problem,
        {
            4: (0, 1, 4),
            5: (0, 1, 5),
        },
    ) == (2.0, 3.0, 4.0)
    assert _fixed_objective_values(
        problem,
        {
            4: (0, 2, 4),
            5: (0, 3, 5),
        },
    ) == (2.0, 4.0, 4.0)

    model, variables, _ = _build_test_route_model(problem)
    try:
        model.Params.ObjNumber = 1
        assert model.ObjNName == "operation-count"
        assert model.ObjNPriority == 2
        coefficients = {
            variable.VarName: float(variable.ObjN)
            for variable in model.getVars()
            if abs(variable.ObjN) > 1e-9
        }
        assert coefficients == {
            variable.VarName: 1.0
            for variable in variables.edge_selected.values()
        }
    finally:
        model.dispose()


def test_every_candidate_tree_edge_has_one_named_strict_level_constraint():
    require_gurobi_license()
    problem = _single_parent_conflict_problem()
    model, _, _ = _build_test_route_model(problem)
    try:
        level_names = {
            constraint.ConstrName
            for constraint in model.getConstrs()
            if constraint.ConstrName.startswith("tree-level-increase-")
        }
        assert level_names == {
            "tree-level-increase-r0000-r0001",
            "tree-level-increase-r0000-r0002",
            "tree-level-increase-r0000-r0006",
            "tree-level-increase-r0001-r0003",
            "tree-level-increase-r0002-r0003",
            "tree-level-increase-r0003-r0004",
            "tree-level-increase-r0003-r0005",
            "tree-level-increase-r0006-r0004",
        }
    finally:
        model.dispose()


def test_branching_reduction_dual_shares_virtual_tree_over_physical_reverse_links():
    require_gurobi_license()
    problem = _branching_reduction_dual_problem()

    pattern = _solve(
        problem,
        ObjectiveMode.LATENCY,
        channel_count=1,
    )

    representative = _template(problem).representative
    assert all(demand.reduction_dual for demand in representative.demands)
    assert pattern.selected_edges == ((0, 1), (1, 2), (1, 3))
    assert pattern.member_paths == (
        (representative.demands[0].demand_id, ((0, 1),)),
        (representative.demands[1].demand_id, ((0, 1), (1, 2))),
        (representative.demands[2].demand_id, ((0, 1), (1, 3))),
    )
    demand_by_id = {
        demand.demand_id: demand for demand in representative.demands
    }
    assert {
        demand_by_id[demand_id].physical_link(src, dst)
        for demand_id, path in pattern.member_paths
        for src, dst in path
    } == {(1, 0), (2, 1), (3, 1)}
    assert all(
        "reduce-fabric"
        in problem.topology.link(LinkKey(*physical)).resource_ids
        for physical in ((1, 0), (2, 1), (3, 1))
    )


def test_optimal_route_is_scoped_to_the_candidate_path_domain():
    require_gurobi_license()
    base = throughput_tradeoff_problem()
    demand = replace(
        base.demands[0],
        candidate_paths=((0, 1, 2),),
    )
    problem = replace(base, demands=(demand,))

    pattern = _solve(problem, ObjectiveMode.LATENCY)

    assert LinkKey(0, 2) in demand.legal_links
    assert pattern.metrics.status is SolveStatus.OPTIMAL
    assert pattern.selected_edges == ((0, 1), (1, 2))
    assert pattern.metrics.makespan_us == 5.0


def test_time_limit_incumbent_keeps_primary_callback_bound_and_gap(monkeypatch):
    require_gurobi_license()
    base = multihop_problem(shared_resource=True)
    problem = replace(
        base,
        inputs=replace(
            base.inputs,
            solver=replace(base.inputs.solver, mip_gap=0.3),
        ),
    )

    def controlled_time_limit(model, progress):
        model.optimize()
        callback = progress.gp.GRB.Callback
        progress(
            _CallbackSnapshot(
                {
                    callback.MULTIOBJ_OBJCNT: 1,
                    callback.MULTIOBJ_OBJBST: 4.0,
                    callback.MULTIOBJ_OBJBND: 3.0,
                }
            ),
            callback.MULTIOBJ,
        )
        progress(
            _CallbackSnapshot(
                {
                    callback.MIP_OBJBST: 2.0,
                    callback.MIP_OBJBND: 2.0,
                }
            ),
            callback.MIP,
        )
        return SolveStatus.TIME_LIMIT

    monkeypatch.setattr(
        routing_milp,
        "_optimize_route_model",
        controlled_time_limit,
    )

    pattern = _solve(problem, ObjectiveMode.LATENCY)

    assert pattern.metrics.status is SolveStatus.TIME_LIMIT
    assert pattern.metrics.objective_values[0] == 4.0
    assert pattern.metrics.best_bound == 3.0
    assert pattern.metrics.mip_gap == 0.25
    assert pattern.metrics.within_requested_gap


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
