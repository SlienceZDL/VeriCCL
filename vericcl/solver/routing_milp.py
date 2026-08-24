import time
from dataclasses import dataclass
from typing import Mapping

from vericcl.errors import ConstructionInfeasibleError, SemanticError
from vericcl.input.models import ObjectiveMode
from vericcl.solver.budget import ModelBudget
from vericcl.solver.gurobi_api import GurobiAdapter
from vericcl.solver.routing import RoutePattern, RoutingModelStats
from vericcl.solver.templates import RoutingUnit, SolverTemplate
from vericcl.topology.model import LinkKey


@dataclass(frozen=True)
class _RouteVariables:
    edge_selected: Mapping[LinkKey, object]
    flow_selected: Mapping[tuple[str, LinkKey], object]
    level: Mapping[int, object]
    critical_depth: object | None
    maximum_edge_load: object | None


def _positive_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SemanticError("{} must be a positive integer".format(field))
    return value


def _validate_arguments(
    template: object,
    channel_count: object,
    objective_mode: object,
    budget: object,
    thread_count: object,
) -> tuple[SolverTemplate, int, ObjectiveMode, ModelBudget, int]:
    if not isinstance(template, SolverTemplate):
        raise SemanticError("template must be a SolverTemplate")
    channels = _positive_integer(channel_count, "channel_count")
    if not isinstance(objective_mode, ObjectiveMode):
        raise SemanticError("objective_mode must be an ObjectiveMode")
    if objective_mode is ObjectiveMode.AUTO:
        raise SemanticError("AUTO must be resolved before route-only solving")
    if not isinstance(budget, ModelBudget):
        raise SemanticError("budget must be a ModelBudget")
    if budget.seconds <= 0.0:
        raise SemanticError("route-only solving requires a positive budget")
    threads = _positive_integer(thread_count, "thread_count")
    return template, channels, objective_mode, budget, threads


def _route_domain(
    unit: RoutingUnit,
) -> tuple[tuple[int, ...], int, tuple[LinkKey, ...]]:
    ranks = unit.node.communication_group
    roots = {demand.root_rank for demand in unit.demands}
    reduction_roles = {demand.reduction_dual for demand in unit.demands}
    if len(roots) != 1:
        raise SemanticError("routing unit demands must share one root")
    if len(reduction_roles) != 1:
        raise SemanticError("routing unit demands must share one direction")
    root = next(iter(roots))
    if root not in ranks:
        raise SemanticError("routing unit root leaves its communication group")
    if any(demand.required_leaf_rank not in ranks for demand in unit.demands):
        raise SemanticError("routing unit leaf leaves its communication group")
    edges = tuple(
        sorted({edge for demand in unit.demands for edge in demand.legal_links})
    )
    return ranks, root, edges


def _build_route_model(
    gp,
    unit: RoutingUnit,
    channel_count: int,
    objective_mode: ObjectiveMode,
    budget: ModelBudget,
    thread_count: int = 1,
):
    ranks, root, edges = _route_domain(unit)
    _, model = GurobiAdapter.create_model(
        "vericcl-route-{}-k{:02d}-{}".format(
            unit.unit_id,
            channel_count,
            objective_mode.value,
        )
    )
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.Threads = thread_count
    model.Params.TimeLimit = budget.seconds
    model.Params.MIPGap = 0.0

    edge_selected = {
        edge: model.addVar(
            vtype=gp.GRB.BINARY,
            name="edge-r{:04d}-r{:04d}".format(
                edge.src_rank,
                edge.dst_rank,
            ),
        )
        for edge in edges
    }
    flow_selected = {}
    for demand_index, demand in enumerate(unit.demands):
        for edge_index, edge in enumerate(sorted(demand.legal_links)):
            flow_selected[(demand.demand_id, edge)] = model.addVar(
                vtype=gp.GRB.BINARY,
                name="flow-d{:04d}-e{:04d}".format(
                    demand_index,
                    edge_index,
                ),
            )
    level = {
        rank: model.addVar(
            lb=0.0,
            ub=float(len(ranks) - 1),
            vtype=gp.GRB.CONTINUOUS,
            name="level-r{:04d}".format(rank),
        )
        for rank in ranks
    }
    model.addConstr(level[root] == 0.0, name="root-level")

    for demand_index, demand in enumerate(unit.demands):
        for rank in ranks:
            outgoing = gp.quicksum(
                flow_selected[(demand.demand_id, edge)]
                for edge in demand.legal_links
                if edge.src_rank == rank
            )
            incoming = gp.quicksum(
                flow_selected[(demand.demand_id, edge)]
                for edge in demand.legal_links
                if edge.dst_rank == rank
            )
            target = 0
            if rank == demand.root_rank:
                target = 1
            elif rank == demand.required_leaf_rank:
                target = -1
            model.addConstr(
                outgoing - incoming == target,
                name="flow-balance-d{:04d}-r{:04d}".format(
                    demand_index,
                    rank,
                ),
            )
        for edge in demand.legal_links:
            model.addConstr(
                flow_selected[(demand.demand_id, edge)]
                <= edge_selected[edge]
            )

    for edge in edges:
        edge_flows = [
            flow_selected[(demand.demand_id, edge)]
            for demand in unit.demands
            if edge in demand.legal_links
        ]
        model.addConstr(edge_selected[edge] <= gp.quicksum(edge_flows))

    for rank in ranks:
        incoming_edges = gp.quicksum(
            edge_selected[edge] for edge in edges if edge.dst_rank == rank
        )
        if rank == root:
            model.addConstr(incoming_edges == 0, name="root-parent")
        else:
            model.addConstr(
                incoming_edges <= 1,
                name="one-parent-r{:04d}".format(rank),
            )

    big_m = float(len(ranks))
    for edge in edges:
        model.addConstr(
            level[edge.dst_rank]
            >= level[edge.src_rank] + 1.0 - big_m * (1 - edge_selected[edge])
        )

    route_size = gp.quicksum(edge_selected.values())
    critical_depth = None
    maximum_edge_load = None
    model.ModelSense = gp.GRB.MINIMIZE
    if objective_mode is ObjectiveMode.LATENCY:
        critical_depth = model.addVar(
            lb=0.0,
            ub=float(len(ranks) - 1),
            vtype=gp.GRB.CONTINUOUS,
            name="critical-depth",
        )
        for demand in unit.demands:
            model.addConstr(
                critical_depth >= level[demand.required_leaf_rank]
            )
        model.setObjectiveN(
            critical_depth,
            index=0,
            priority=2,
            weight=1.0,
            name="critical-depth",
        )
    else:
        maximum_edge_load = model.addVar(
            lb=0.0,
            vtype=gp.GRB.CONTINUOUS,
            name="maximum-edge-flow-load",
        )
        for edge in edges:
            model.addConstr(
                maximum_edge_load
                >= gp.quicksum(
                    flow_selected[(demand.demand_id, edge)]
                    for demand in unit.demands
                    if edge in demand.legal_links
                )
                / channel_count
            )
        model.setObjectiveN(
            maximum_edge_load,
            index=0,
            priority=2,
            weight=1.0,
            name="maximum-edge-flow-load",
        )
    model.setObjectiveN(
        route_size,
        index=1,
        priority=1,
        weight=1.0,
        name="route-size",
    )
    model.update()
    return model, _RouteVariables(
        edge_selected=edge_selected,
        flow_selected=flow_selected,
        level=level,
        critical_depth=critical_depth,
        maximum_edge_load=maximum_edge_load,
    )


def _physical_edge(unit: RoutingUnit, edge: LinkKey) -> LinkKey:
    src_rank, dst_rank = unit.demands[0].physical_link(
        edge.src_rank,
        edge.dst_rank,
    )
    return LinkKey(src_rank, dst_rank)


def _validate_extracted_route(
    unit: RoutingUnit,
    selected: set[LinkKey],
    flows: Mapping[str, set[LinkKey]],
) -> None:
    _, root, _ = _route_domain(unit)
    parents = {}
    for edge in selected:
        if edge.dst_rank == root:
            raise SemanticError("route tree enters its root")
        if edge.dst_rank in parents:
            raise SemanticError("route tree gives a rank multiple parents")
        parents[edge.dst_rank] = edge.src_rank

    used = set()
    for demand in unit.demands:
        demand_flow = flows[demand.demand_id]
        if not demand_flow <= demand.legal_links:
            raise SemanticError("route flow uses a forbidden or illegal edge")
        if not demand_flow <= demand.allowed_links:
            raise SemanticError("route flow uses an edge outside the allowed domain")
        outgoing = {}
        for edge in demand_flow:
            outgoing.setdefault(edge.src_rank, []).append(edge.dst_rank)
            used.add(edge)
        rank = demand.root_rank
        visited = {rank}
        while rank != demand.required_leaf_rank:
            destinations = outgoing.get(rank, ())
            if len(destinations) != 1:
                raise SemanticError("route flow does not form one continuous path")
            rank = destinations[0]
            if rank in visited:
                raise SemanticError("route flow contains a cycle")
            visited.add(rank)
        if len(demand_flow) != len(visited) - 1:
            raise SemanticError("route flow contains unused edges")
        physical_flow = {_physical_edge(unit, edge) for edge in demand_flow}
        for forbidden in demand.forbidden_members:
            if forbidden.slice_id not in demand.member_slice_ids:
                continue
            if LinkKey(forbidden.src_rank, forbidden.dst_rank) in physical_flow:
                raise SemanticError("route flow uses a forbidden physical transfer")
    if used != selected:
        raise SemanticError("route tree contains an unused or disconnected edge")

    pending = {root}
    reached = {root}
    while pending:
        source = pending.pop()
        for edge in selected:
            if edge.src_rank != source or edge.dst_rank in reached:
                continue
            reached.add(edge.dst_rank)
            pending.add(edge.dst_rank)
    selected_ranks = {
        rank
        for edge in selected
        for rank in (edge.src_rank, edge.dst_rank)
    }
    if selected_ranks - reached:
        raise SemanticError("route tree contains a disconnected component")


def solve_route_milp(
    template: SolverTemplate,
    channel_count: int,
    objective_mode: ObjectiveMode,
    budget: ModelBudget,
    thread_count: int = 1,
) -> RoutePattern:
    template, channels, objective, budget, threads = _validate_arguments(
        template,
        channel_count,
        objective_mode,
        budget,
        thread_count,
    )
    gp = GurobiAdapter.require()
    build_started = time.monotonic()
    model, variables = _build_route_model(
        gp,
        template.representative,
        channels,
        objective,
        budget,
        threads,
    )
    build_time = time.monotonic() - build_started
    try:
        counts = GurobiAdapter.model_counts(model)
        optimize_started = time.monotonic()
        model.optimize()
        optimize_time = time.monotonic() - optimize_started
        if model.SolCount <= 0:
            raise ConstructionInfeasibleError(
                "representative route MILP has no incumbent (status {})".format(
                    model.Status
                )
            )
        selected = {
            edge
            for edge, variable in variables.edge_selected.items()
            if variable.X > 0.5
        }
        flows = {
            demand.demand_id: {
                edge
                for (demand_id, edge), variable in variables.flow_selected.items()
                if demand_id == demand.demand_id and variable.X > 0.5
            }
            for demand in template.representative.demands
        }
        _validate_extracted_route(template.representative, selected, flows)
        selected_physical = tuple(
            sorted(_physical_edge(template.representative, edge) for edge in selected)
        )
        if len(selected_physical) != len(set(selected_physical)):
            raise SemanticError("route tree maps multiple edges to one physical link")
        return RoutePattern(
            template_id=template.template_id,
            channel_count=channels,
            objective_mode=objective,
            selected_edges=selected_physical,
            parent_edges=tuple(
                sorted((edge.src_rank, edge.dst_rank) for edge in selected)
            ),
            model_stats=RoutingModelStats(
                variable_count=counts[0],
                constraint_count=counts[1],
                general_constraint_count=counts[2],
                build_time_s=build_time,
                optimize_time_s=optimize_time,
            ),
        )
    finally:
        model.dispose()
