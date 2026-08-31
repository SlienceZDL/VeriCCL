import math
import os
import time
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

from vericcl.errors import ConstructionInfeasibleError, SemanticError
from vericcl.input.models import ObjectiveMode, ResolvedInput
from vericcl.solver.budget import ModelBudget
from vericcl.solver.demands import TransferDemand, routing_unit_key
from vericcl.solver.gurobi_api import GurobiAdapter
from vericcl.solver.lower_bounds import route_edge_duration_us
from vericcl.solver.model import SolveStatus, SolverMetrics
from vericcl.solver.objectives import (
    ObjectiveExpressions,
    configure_lexicographic_objective,
)
from vericcl.solver.routing import (
    RoutePattern,
    RoutingModelFailure,
    RoutingModelStats,
)
from vericcl.solver.scheduling import NUMERICAL_TOLERANCE
from vericcl.solver.templates import SolverTemplate
from vericcl.topology.model import LinkKey, Topology


@dataclass(frozen=True)
class _RouteVariables:
    edge_selected: Mapping[LinkKey, object]
    flow_selected: Mapping[Tuple[str, LinkKey], object]
    path_selected: Mapping[Tuple[str, int], object]
    level: Mapping[int, object]
    route_completion: object
    maximum_resource_load: object


@dataclass(frozen=True)
class _RouteContext:
    gp: object
    demands: Tuple[TransferDemand, ...]
    candidate_paths: Mapping[str, Tuple[Tuple[int, ...], ...]]
    edge_durations: Mapping[LinkKey, float]
    physical_edges: Mapping[LinkKey, LinkKey]
    thread_count: int
    variable_count: int
    constraint_count: int
    general_constraint_count: int
    build_time_s: float


@dataclass
class _PrimaryObjectiveProgress:
    gp: object
    best_value: float = math.inf
    best_bound: float = 0.0
    mip_gap: float = math.inf
    finished: bool = False

    def __call__(self, model, where) -> None:
        callback = self.gp.GRB.Callback
        try:
            if where == callback.MIP and not self.finished:
                self.best_value = float(model.cbGet(callback.MIP_OBJBST))
                self.best_bound = float(model.cbGet(callback.MIP_OBJBND))
            elif where == callback.MULTIOBJ and not self.finished:
                if int(model.cbGet(callback.MULTIOBJ_OBJCNT)) != 1:
                    return
                self.best_value = float(
                    model.cbGet(callback.MULTIOBJ_OBJBST)
                )
                self.best_bound = float(
                    model.cbGet(callback.MULTIOBJ_OBJBND)
                )
                denominator = abs(self.best_value)
                if denominator <= NUMERICAL_TOLERANCE:
                    self.mip_gap = (
                        0.0
                        if abs(self.best_bound) <= NUMERICAL_TOLERANCE
                        else 1.0
                    )
                else:
                    self.mip_gap = abs(
                        self.best_value - self.best_bound
                    ) / denominator
                self.finished = True
        except self.gp.GurobiError:
            return


def _positive_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SemanticError("{} must be a positive integer".format(field))
    return value


def _effective_threads(inputs: ResolvedInput) -> int:
    return max(
        1,
        min(inputs.solver.max_threads_per_model, os.cpu_count() or 1),
    )


def _physical_link(demand: TransferDemand, link: LinkKey) -> LinkKey:
    return LinkKey(*demand.physical_link(link.src_rank, link.dst_rank))


def _forbidden_edge(demand: TransferDemand, physical: LinkKey) -> bool:
    return any(
        item.stage_id == demand.stage_id
        and item.slice_id in demand.member_slice_ids
        and item.src_rank == physical.src_rank
        and item.dst_rank == physical.dst_rank
        for item in demand.forbidden_members
    )


def _path_edges(path: Tuple[int, ...]) -> Tuple[LinkKey, ...]:
    return tuple(
        LinkKey(src, dst) for src, dst in zip(path, path[1:])
    )


def _path_is_legal(
    template: SolverTemplate,
    topology: Topology,
    demand: TransferDemand,
    path: Tuple[int, ...],
) -> bool:
    node = template.representative.node
    if (
        len(path) < 2
        or path[0] != demand.root_rank
        or path[-1] != demand.required_leaf_rank
        or len(path) != len(set(path))
        or any(rank not in node.communication_group for rank in path)
    ):
        return False
    for link in _path_edges(path):
        if link not in demand.allowed_links or link not in demand.legal_links:
            return False
        physical = _physical_link(demand, link)
        if physical not in topology.links or physical not in node.allowed_links:
            return False
        if _forbidden_edge(demand, physical):
            return False
        edge = topology.link(physical)
        if any(
            resource_id not in node.shared_resource_ids
            or resource_id not in topology.shared_resources
            or physical
            not in topology.shared_resources[resource_id].member_links
            for resource_id in edge.resource_ids
        ):
            return False
    return True


def _validated_candidate_paths(
    template: SolverTemplate,
    inputs: ResolvedInput,
    topology: Topology,
) -> Tuple[
    Tuple[TransferDemand, ...],
    Mapping[str, Tuple[Tuple[int, ...], ...]],
]:
    unit = template.representative
    node = unit.node
    demands = unit.demands
    if len({routing_unit_key(demand) for demand in demands}) != 1:
        raise SemanticError(
            "solver template representative crosses a routing unit boundary"
        )
    if any(rank >= topology.rank_count for rank in node.communication_group):
        raise SemanticError("representative rank is outside the topology")
    if any(
        resource_id not in topology.shared_resources
        for resource_id in node.shared_resource_ids
    ):
        raise SemanticError("representative references an unknown shared resource")
    global_slice_count = (
        inputs.rank_count * inputs.hyperparameters.slice_count
    )
    result = {}
    for demand in demands:
        if (
            demand.logical_position >= inputs.hyperparameters.slice_count
            or any(
                contributor >= global_slice_count
                for contributor in demand.contributors
            )
        ):
            raise SemanticError("representative demand is outside the input layout")
        paths = tuple(
            path
            for path in demand.candidate_paths
            if _path_is_legal(template, topology, demand, path)
        )
        if not paths:
            raise ConstructionInfeasibleError(
                "representative demand {} has no legal candidate path".format(
                    demand.demand_id
                )
            )
        result[demand.demand_id] = paths
    return demands, result


def _validate_api(
    template: SolverTemplate,
    inputs: ResolvedInput,
    topology: Topology,
    channel_count: int,
    objective: ObjectiveMode,
    budget: ModelBudget,
    warm_start: Optional[RoutePattern],
) -> int:
    if not isinstance(template, SolverTemplate):
        raise SemanticError("template must be a SolverTemplate")
    if not isinstance(inputs, ResolvedInput):
        raise SemanticError("inputs must be a ResolvedInput")
    if not isinstance(topology, Topology):
        raise SemanticError("topology must be a Topology")
    if inputs.rank_count != topology.rank_count:
        raise SemanticError("input and topology rank counts must agree")
    channels = _positive_integer(channel_count, "channel_count")
    if channels > inputs.solver.max_channels:
        raise SemanticError("channel_count exceeds the configured maximum")
    if not isinstance(objective, ObjectiveMode):
        raise SemanticError("objective must be an ObjectiveMode")
    if objective is ObjectiveMode.AUTO:
        raise SemanticError(
            "AUTO must be resolved before building a representative route model"
        )
    if not isinstance(budget, ModelBudget):
        raise SemanticError("budget must be a ModelBudget")
    if budget.seconds <= 0.0:
        raise ConstructionInfeasibleError(
            "representative route model budget is exhausted"
        )
    if warm_start is not None:
        if not isinstance(warm_start, RoutePattern):
            raise SemanticError("warm_start must be a RoutePattern or None")
        if warm_start.template_id != template.template_id:
            raise SemanticError("warm_start template ID does not match")
    return channels


def _resource_expressions(
    gp,
    template: SolverTemplate,
    topology: Topology,
    channel_count: int,
    edges: Tuple[LinkKey, ...],
    edge_selected: Mapping[LinkKey, object],
    edge_durations: Mapping[LinkKey, float],
    physical_edges: Mapping[LinkKey, LinkKey],
) -> Mapping[Tuple[str, object], object]:
    link_terms: Dict[LinkKey, list] = {}
    resource_terms: Dict[str, list] = {}
    for link in edges:
        physical = physical_edges[link]
        duration = edge_durations[link]
        link_terms.setdefault(physical, []).append(
            (duration, edge_selected[link])
        )
        for resource_id in topology.link(physical).resource_ids:
            if resource_id not in template.representative.node.shared_resource_ids:
                raise SemanticError(
                    "route edge uses an undeclared shared resource"
                )
            resource_terms.setdefault(resource_id, []).append(
                (duration, edge_selected[link])
            )
    expressions = {}
    for physical, terms in sorted(link_terms.items()):
        slots = min(channel_count, topology.link(physical).max_channels)
        expressions[("link", physical)] = gp.quicksum(
            duration / slots * selected
            for duration, selected in terms
        )
    for resource_id, terms in sorted(resource_terms.items()):
        slots = min(
            channel_count,
            topology.shared_resources[resource_id].max_channels,
        )
        expressions[("resource", resource_id)] = gp.quicksum(
            duration / slots * selected
            for duration, selected in terms
        )
    return expressions


def _apply_warm_start(
    warm_start: Optional[RoutePattern],
    variables: _RouteVariables,
    candidate_paths: Mapping[str, Tuple[Tuple[int, ...], ...]],
) -> None:
    if warm_start is None:
        return
    selected_edges = {
        LinkKey(src, dst) for src, dst in warm_start.selected_edges
    }
    warm_paths = {
        demand_id: (path[0][0],) + tuple(dst for _, dst in path)
        for demand_id, path in warm_start.member_paths
    }
    for link, variable in variables.edge_selected.items():
        variable.Start = 1.0 if link in selected_edges else 0.0
    for (demand_id, link), variable in variables.flow_selected.items():
        path = warm_paths.get(demand_id, ())
        variable.Start = (
            1.0 if link in _path_edges(path) else 0.0
        )
    for (demand_id, path_index), variable in variables.path_selected.items():
        variable.Start = (
            1.0
            if warm_paths.get(demand_id)
            == candidate_paths[demand_id][path_index]
            else 0.0
        )


def _build_route_model(
    template: SolverTemplate,
    inputs: ResolvedInput,
    topology: Topology,
    channel_count: int,
    objective: ObjectiveMode,
    budget: ModelBudget,
    warm_start: Optional[RoutePattern],
    environment=None,
):
    channels = _validate_api(
        template,
        inputs,
        topology,
        channel_count,
        objective,
        budget,
        warm_start,
    )
    demands, candidate_paths = _validated_candidate_paths(
        template,
        inputs,
        topology,
    )
    started = time.monotonic()
    gp, model = GurobiAdapter.create_model(
        "vericcl-route-{}-k{:02d}-{}".format(
            template.template_id,
            channels,
            objective.value,
        ),
        environment=environment,
    )
    try:
        thread_count = _effective_threads(inputs)
        model.Params.OutputFlag = 0
        model.Params.Seed = inputs.solver.solver_seed
        model.Params.Threads = thread_count
        model.Params.TimeLimit = budget.seconds
        model.Params.MIPGap = (
            0.0
            if inputs.solver.require_proven_optimal
            else inputs.solver.mip_gap
        )
        edges = tuple(
            sorted(
                {
                    link
                    for demand in demands
                    for path in candidate_paths[demand.demand_id]
                    for link in _path_edges(path)
                }
            )
        )
        representative_by_edge = {}
        for demand in demands:
            for path in candidate_paths[demand.demand_id]:
                for link in _path_edges(path):
                    representative_by_edge.setdefault(link, demand)
        edge_durations = {
            link: route_edge_duration_us(
                inputs,
                topology,
                representative_by_edge[link],
                link,
                channels,
            )
            for link in edges
        }
        physical_edges = {
            link: _physical_link(representative_by_edge[link], link)
            for link in edges
        }
        edge_selected = {
            link: model.addVar(
                vtype=gp.GRB.BINARY,
                name="tree-edge-e{:04d}-r{:04d}-r{:04d}".format(
                    index,
                    link.src_rank,
                    link.dst_rank,
                ),
            )
            for index, link in enumerate(edges)
        }
        group = template.representative.node.communication_group
        level = {
            rank: model.addVar(
                lb=0.0,
                ub=float(len(group) - 1),
                vtype=gp.GRB.CONTINUOUS,
                name="level-r{:04d}".format(rank),
            )
            for rank in group
        }
        flow_selected = {}
        path_selected = {}
        for demand_index, demand in enumerate(demands):
            # Model optimality is scoped to this TransferDemand candidate-path
            # domain; it does not assert optimality over every legal graph path.
            paths = candidate_paths[demand.demand_id]
            path_variables = []
            for path_index, _ in enumerate(paths):
                variable = model.addVar(
                    vtype=gp.GRB.BINARY,
                    name="flow-path-d{:04d}-p{:04d}".format(
                        demand_index,
                        path_index,
                    ),
                )
                path_selected[(demand.demand_id, path_index)] = variable
                path_variables.append(variable)
            model.addConstr(gp.quicksum(path_variables) == 1)
            demand_edges = tuple(
                sorted(
                    {
                        link
                        for path in paths
                        for link in _path_edges(path)
                    }
                )
            )
            for edge_index, link in enumerate(demand_edges):
                variable = model.addVar(
                    vtype=gp.GRB.BINARY,
                    name="flow-edge-d{:04d}-e{:04d}".format(
                        demand_index,
                        edge_index,
                    ),
                )
                flow_selected[(demand.demand_id, link)] = variable
                model.addConstr(
                    variable
                    == gp.quicksum(
                        path_selected[(demand.demand_id, path_index)]
                        for path_index, path in enumerate(paths)
                        if link in _path_edges(path)
                    )
                )
                model.addConstr(variable <= edge_selected[link])
            for rank in group:
                outgoing = gp.quicksum(
                    flow_selected[(demand.demand_id, link)]
                    for link in demand_edges
                    if link.src_rank == rank
                )
                incoming = gp.quicksum(
                    flow_selected[(demand.demand_id, link)]
                    for link in demand_edges
                    if link.dst_rank == rank
                )
                target = 0
                if rank == demand.root_rank:
                    target = 1
                elif rank == demand.required_leaf_rank:
                    target = -1
                model.addConstr(outgoing - incoming == target)
                model.addConstr(outgoing <= 1)
                model.addConstr(incoming <= 1)
        for link in edges:
            users = tuple(
                variable
                for (demand_id, candidate), variable in flow_selected.items()
                if candidate == link
            )
            model.addConstr(edge_selected[link] <= gp.quicksum(users))
        root = demands[0].root_rank
        model.addConstr(level[root] == 0.0)
        big_m = float(len(group))
        required_leaves = {
            demand.required_leaf_rank for demand in demands
        }
        for rank in group:
            incoming = gp.quicksum(
                edge_selected[link]
                for link in edges
                if link.dst_rank == rank
            )
            outgoing = gp.quicksum(
                edge_selected[link]
                for link in edges
                if link.src_rank == rank
            )
            if rank == root:
                model.addConstr(incoming == 0)
            else:
                model.addConstr(
                    incoming <= 1,
                    name="tree-parent-at-most-one-r{:04d}".format(rank),
                )
                model.addConstr(outgoing <= len(edges) * incoming)
                model.addConstr(level[rank] <= (len(group) - 1) * incoming)
                if rank in required_leaves:
                    model.addConstr(incoming == 1)
        for link in edges:
            model.addConstr(
                level[link.dst_rank]
                >= level[link.src_rank]
                + 1.0
                - big_m * (1.0 - edge_selected[link]),
                name=(
                    "tree-level-increase-r{:04d}-r{:04d}".format(
                        link.src_rank,
                        link.dst_rank,
                    )
                ),
            )
        route_completion = model.addVar(
            lb=0.0,
            vtype=gp.GRB.CONTINUOUS,
            name="route-completion-us",
        )
        for demand in demands:
            model.addConstr(
                route_completion
                >= gp.quicksum(
                    edge_durations[link]
                    * flow_selected[(demand.demand_id, link)]
                    for link in edges
                    if (demand.demand_id, link) in flow_selected
                )
            )
        maximum_resource_load = model.addVar(
            lb=0.0,
            vtype=gp.GRB.CONTINUOUS,
            name="maximum-resource-load-us",
        )
        resource_expressions = _resource_expressions(
            gp,
            template,
            topology,
            channels,
            edges,
            edge_selected,
            edge_durations,
            physical_edges,
        )
        for expression in resource_expressions.values():
            model.addConstr(maximum_resource_load >= expression)
        configure_lexicographic_objective(
            model,
            gp,
            objective,
            ObjectiveExpressions(
                makespan=route_completion,
                operation_count=gp.quicksum(edge_selected.values()),
                hop_count=gp.quicksum(flow_selected.values()),
                maximum_resource_load=maximum_resource_load,
            ),
        )
        variables = _RouteVariables(
            edge_selected=edge_selected,
            flow_selected=flow_selected,
            path_selected=path_selected,
            level=level,
            route_completion=route_completion,
            maximum_resource_load=maximum_resource_load,
        )
        _apply_warm_start(warm_start, variables, candidate_paths)
        model.update()
        context = _RouteContext(
            gp=gp,
            demands=demands,
            candidate_paths=candidate_paths,
            edge_durations=edge_durations,
            physical_edges=physical_edges,
            thread_count=thread_count,
            variable_count=int(model.NumVars),
            constraint_count=int(model.NumConstrs),
            general_constraint_count=int(model.NumGenConstrs),
            build_time_s=max(0.0, time.monotonic() - started),
        )
        return model, variables, context
    except Exception:
        model.dispose()
        raise


def _status(gp, model) -> SolveStatus:
    if model.Status == gp.GRB.OPTIMAL:
        return SolveStatus.OPTIMAL
    if model.Status == gp.GRB.INFEASIBLE:
        return SolveStatus.INFEASIBLE
    if model.Status == gp.GRB.TIME_LIMIT:
        return SolveStatus.TIME_LIMIT
    if model.SolCount > 0:
        return SolveStatus.FEASIBLE
    return SolveStatus.ERROR


def _optimize_route_model(
    model,
    progress: _PrimaryObjectiveProgress,
) -> SolveStatus:
    model.optimize(progress)
    return _status(progress.gp, model)


def _primary_bound_and_gap(
    model,
    status: SolveStatus,
    primary_value: float,
    progress: _PrimaryObjectiveProgress,
) -> Tuple[float, float]:
    if math.isfinite(progress.best_bound) and math.isfinite(progress.mip_gap):
        return progress.best_bound, progress.mip_gap
    if math.isfinite(progress.best_value) and math.isfinite(
        progress.best_bound
    ):
        denominator = abs(progress.best_value)
        if denominator <= NUMERICAL_TOLERANCE:
            gap = (
                0.0
                if abs(progress.best_bound) <= NUMERICAL_TOLERANCE
                else 1.0
            )
        else:
            gap = abs(
                progress.best_value - progress.best_bound
            ) / denominator
        return progress.best_bound, gap
    try:
        best_bound = float(model.ObjBound)
        mip_gap = float(model.MIPGap)
    except AttributeError:
        if status is SolveStatus.OPTIMAL:
            return primary_value, 0.0
        return 0.0, 1.0 if primary_value > NUMERICAL_TOLERANCE else 0.0
    if not math.isfinite(best_bound):
        best_bound = 0.0
    if not math.isfinite(mip_gap):
        mip_gap = 0.0
    return best_bound, mip_gap


def _selected_path(
    demand: TransferDemand,
    variables: _RouteVariables,
) -> Tuple[Tuple[int, int], ...]:
    outgoing = {}
    selected = set()
    for (demand_id, link), variable in variables.flow_selected.items():
        if demand_id != demand.demand_id or variable.X <= 0.5:
            continue
        if link.src_rank in outgoing:
            raise SemanticError("representative flow branches")
        outgoing[link.src_rank] = link.dst_rank
        selected.add(link)
    rank = demand.root_rank
    visited = {rank}
    path = []
    while rank != demand.required_leaf_rank:
        if rank not in outgoing:
            raise SemanticError("representative flow does not reach its leaf")
        destination = outgoing[rank]
        link = LinkKey(rank, destination)
        path.append((rank, destination))
        rank = destination
        if rank in visited:
            raise SemanticError("representative flow contains a cycle")
        visited.add(rank)
    if set(LinkKey(*edge) for edge in path) != selected:
        raise SemanticError("representative flow contains unused edges")
    return tuple(path)


def _validate_selected_paths(
    template: SolverTemplate,
    topology: Topology,
    selected_edges: Tuple[Tuple[int, int], ...],
    member_paths: Tuple[
        Tuple[str, Tuple[Tuple[int, int], ...]], ...
    ],
) -> None:
    demands = {
        demand.demand_id: demand
        for demand in template.representative.demands
    }
    if set(demands) != {demand_id for demand_id, _ in member_paths}:
        raise SemanticError("representative route omits or invents a demand")
    selected = {LinkKey(*edge) for edge in selected_edges}
    used = set()
    parents = {}
    root = template.representative.demands[0].root_rank
    for link in selected:
        if link.dst_rank == root:
            raise SemanticError("representative tree enters its root")
        if link.dst_rank in parents:
            raise SemanticError("representative tree gives a rank multiple parents")
        parents[link.dst_rank] = link.src_rank
    for demand_id, path in member_paths:
        demand = demands[demand_id]
        ranks = (path[0][0],) + tuple(dst for _, dst in path)
        if (
            ranks[0] != demand.root_rank
            or ranks[-1] != demand.required_leaf_rank
            or ranks not in demand.candidate_paths
        ):
            raise SemanticError("representative route is not a candidate path")
        for edge in path:
            link = LinkKey(*edge)
            if link not in selected:
                raise SemanticError("representative path leaves its tree")
            if not _path_is_legal(template, topology, demand, ranks):
                raise SemanticError("representative path is illegal")
            used.add(link)
    if used != selected:
        raise SemanticError("representative tree contains an unused edge")


def _maximum_normalized_load(
    topology: Topology,
    channel_count: int,
    selected_edges: Tuple[Tuple[int, int], ...],
    context: _RouteContext,
) -> float:
    link_loads: Dict[LinkKey, float] = {}
    resource_loads: Dict[str, float] = {}
    for edge in selected_edges:
        link = LinkKey(*edge)
        physical = context.physical_edges[link]
        duration = context.edge_durations[link]
        link_loads[physical] = link_loads.get(physical, 0.0) + duration / min(
            channel_count,
            topology.link(physical).max_channels,
        )
        for resource_id in topology.link(physical).resource_ids:
            resource_loads[resource_id] = (
                resource_loads.get(resource_id, 0.0)
                + duration
                / min(
                    channel_count,
                    topology.shared_resources[resource_id].max_channels,
                )
            )
    return max(
        tuple(link_loads.values()) + tuple(resource_loads.values()),
        default=0.0,
    )


def solve_route_milp(
    template: SolverTemplate,
    inputs: ResolvedInput,
    topology: Topology,
    channel_count: int,
    objective: ObjectiveMode,
    budget: ModelBudget,
    warm_start: Optional[RoutePattern] = None,
    environment=None,
) -> RoutePattern:
    build_started = time.monotonic()
    try:
        model, variables, context = _build_route_model(
            template,
            inputs,
            topology,
            channel_count,
            objective,
            budget,
            warm_start,
            environment,
        )
    except RoutingModelFailure:
        raise
    except ConstructionInfeasibleError as error:
        raise RoutingModelFailure(
            str(error),
            RoutingModelStats(
                variable_count=0,
                constraint_count=0,
                general_constraint_count=0,
                build_time_s=max(0.0, time.monotonic() - build_started),
                optimize_time_s=0.0,
            ),
        ) from error
    gp = context.gp
    progress = _PrimaryObjectiveProgress(gp)
    optimize_started = time.monotonic()
    try:
        try:
            status = _optimize_route_model(model, progress)
        except RoutingModelFailure:
            raise
        except ConstructionInfeasibleError as error:
            raise RoutingModelFailure(
                str(error),
                RoutingModelStats(
                    variable_count=context.variable_count,
                    constraint_count=context.constraint_count,
                    general_constraint_count=(
                        context.general_constraint_count
                    ),
                    build_time_s=context.build_time_s,
                    optimize_time_s=max(
                        0.0,
                        time.monotonic() - optimize_started,
                    ),
                ),
            ) from error
        optimize_time = max(0.0, time.monotonic() - optimize_started)
        if model.SolCount <= 0:
            raise RoutingModelFailure(
                "representative routing model has no incumbent: status {}".format(
                    model.Status
                ),
                RoutingModelStats(
                    variable_count=context.variable_count,
                    constraint_count=context.constraint_count,
                    general_constraint_count=context.general_constraint_count,
                    build_time_s=context.build_time_s,
                    optimize_time_s=optimize_time,
                ),
            )
        selected_edges = tuple(
            (link.src_rank, link.dst_rank)
            for link, variable in sorted(variables.edge_selected.items())
            if variable.X > 0.5
        )
        member_paths = tuple(
            (demand.demand_id, _selected_path(demand, variables))
            for demand in context.demands
        )
        _validate_selected_paths(
            template,
            topology,
            selected_edges,
            member_paths,
        )
        route_completion = max(
            (
                sum(
                    context.edge_durations[LinkKey(*edge)]
                    for edge in path
                )
                for _, path in member_paths
            ),
            default=0.0,
        )
        operation_count = len(selected_edges)
        hop_count = sum(len(path) for _, path in member_paths)
        maximum_resource_load = _maximum_normalized_load(
            topology,
            channel_count,
            selected_edges,
            context,
        )
        objective_values = (
            (
                route_completion,
                float(operation_count),
                float(hop_count),
            )
            if objective is ObjectiveMode.LATENCY
            else (maximum_resource_load, route_completion)
        )
        best_bound, mip_gap = _primary_bound_and_gap(
            model,
            status,
            objective_values[0],
            progress,
        )
        requested_gap = (
            0.0
            if inputs.solver.require_proven_optimal
            else inputs.solver.mip_gap
        )
        metrics = SolverMetrics(
            status=status,
            objective_values=objective_values,
            best_bound=max(0.0, best_bound),
            mip_gap=max(0.0, mip_gap),
            within_requested_gap=(
                mip_gap <= requested_gap + NUMERICAL_TOLERANCE
            ),
            solve_time_s=context.build_time_s + optimize_time,
            model_count=1,
            operation_count=operation_count,
            hop_count=hop_count,
            makespan_us=route_completion,
            maximum_normalized_resource_load=maximum_resource_load,
            solver_name="gurobi",
            solver_version=GurobiAdapter.version(gp),
            solver_seed=inputs.solver.solver_seed,
            thread_count=context.thread_count,
            termination_reason="gurobi_status_{}".format(model.Status),
        )
        return RoutePattern(
            template_id=template.template_id,
            channel_count=channel_count,
            objective_mode=objective,
            selected_edges=selected_edges,
            member_paths=member_paths,
            metrics=metrics,
            model_stats=RoutingModelStats(
                variable_count=context.variable_count,
                constraint_count=context.constraint_count,
                general_constraint_count=context.general_constraint_count,
                build_time_s=context.build_time_s,
                optimize_time_s=optimize_time,
            ),
        )
    finally:
        model.dispose()


solve_route_milp.requires_explicit_environment = True
