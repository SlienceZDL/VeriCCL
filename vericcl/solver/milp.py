import math
import os
import time
from dataclasses import dataclass
from typing import Dict, FrozenSet, Mapping, Optional, Tuple

from vericcl.errors import (
    SemanticError,
    SolverUnavailableError,
)
from vericcl.input.json_codec import sha256_json
from vericcl.input.models import ObjectiveMode
from vericcl.semantics.atom import Schedule
from vericcl.solver.budget import ModelBudget
from vericcl.solver.demands import (
    RoutingUnitKey,
    SolverProblem,
    TransferDemand,
    routing_unit_key,
)
from vericcl.solver.gurobi_api import GurobiAdapter
from vericcl.solver.model import (
    SolveCandidate,
    SolveStatus,
    SolverMetrics,
)
from vericcl.solver.objectives import (
    ObjectiveExpressions,
    configure_lexicographic_objective,
)
from vericcl.solver.scheduling import (
    NUMERICAL_TOLERANCE,
    RoutedOperation,
    RoutedTree,
    available_channel_count,
    demand_batch_assignments,
    fixed_transfer_duration_us,
    materialize_route_schedule,
    physical_link_key,
)
from vericcl.topology.model import LaneKey, LinkKey


OperationKey = Tuple[int, LinkKey]


@dataclass(frozen=True)
class _Tree:
    index: int
    key: RoutingUnitKey
    root_rank: int
    logical_position: int
    contributors: FrozenSet[int]
    reduction_dual: bool
    demands: Tuple[TransferDemand, ...]
    edges: Tuple[LinkKey, ...]


@dataclass(frozen=True)
class _OperationValue:
    key: OperationKey
    tree: _Tree
    link: LinkKey
    channel: int
    start_time: float
    end_time: float
    duration: float
    resource_slots: Mapping[str, int]


@dataclass(frozen=True)
class _Variables:
    edge_selected: Mapping[OperationKey, object]
    flow_selected: Mapping[Tuple[str, LinkKey], object]
    channel_selected: Mapping[Tuple[OperationKey, int], object]
    resource_selected: Mapping[Tuple[OperationKey, str, int], object]
    start_time: Mapping[OperationKey, object]
    end_time: Mapping[OperationKey, object]
    ready_time: Mapping[Tuple[int, int], object]
    makespan: object
    maximum_resource_load: object


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
                self.best_value = float(
                    model.cbGet(callback.MIP_OBJBST)
                )
                self.best_bound = float(
                    model.cbGet(callback.MIP_OBJBND)
                )
            elif where == callback.MULTIOBJ and not self.finished:
                self.best_value = float(
                    model.cbGet(callback.MULTIOBJ_OBJBST)
                )
                self.best_bound = float(
                    model.cbGet(callback.MULTIOBJ_OBJBND)
                )
                self.mip_gap = float(
                    model.cbGet(callback.MULTIOBJ_MIPGAP)
                )
                self.finished = True
        except self.gp.GurobiError:
            return


def _positive_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SemanticError("{} must be a positive integer".format(field))
    return value


def _trees(problem: SolverProblem) -> Tuple[_Tree, ...]:
    grouped: Dict[RoutingUnitKey, list] = {}
    for demand in problem.demands:
        grouped.setdefault(routing_unit_key(demand), []).append(demand)
    trees = []
    for index, key in enumerate(sorted(grouped)):
        demands = tuple(sorted(grouped[key], key=lambda item: item.demand_id))
        edges = tuple(
            sorted(
                {
                    edge
                    for demand in demands
                    for edge in demand.legal_links
                }
            )
        )
        trees.append(
            _Tree(
                index=index,
                key=key,
                root_rank=key[0],
                logical_position=key[1],
                contributors=frozenset(key[2]),
                reduction_dual=key[3],
                demands=demands,
                edges=edges,
            )
        )
    return tuple(trees)


def _representative_demand(tree: _Tree, link: LinkKey) -> TransferDemand:
    return next(
        demand for demand in tree.demands if link in demand.legal_links
    )


def _effective_threads(problem: SolverProblem) -> int:
    cpu_count = os.cpu_count() or 1
    return max(
        1,
        min(problem.inputs.solver.max_threads_per_model, cpu_count),
    )


def _solver_version(gp) -> str:
    return ".".join(str(value) for value in gp.gurobi.version())


def _candidate_id(
    problem: SolverProblem,
    channel_count: int,
    objective: ObjectiveMode,
) -> str:
    token = sha256_json(
        {
            "backend": "gurobi",
            "node_id": problem.node.node_id,
            "channel_count": channel_count,
            "objective": objective.value,
            "solver_seed": problem.inputs.solver.solver_seed,
        }
    )[:16]
    return "{}-milp-{}".format(problem.node.node_id, token)


def _metrics(
    *,
    gp,
    problem: SolverProblem,
    status: SolveStatus,
    solve_time_s: float,
    thread_count: int,
    termination_reason: str,
    objective_values: Tuple[float, ...] = (),
    best_bound: float = 0.0,
    mip_gap: float = 0.0,
    within_requested_gap: bool = False,
    operation_count: int = 0,
    hop_count: int = 0,
    makespan_us: float = 0.0,
    maximum_normalized_resource_load: float = 0.0,
) -> SolverMetrics:
    return SolverMetrics(
        status=status,
        objective_values=objective_values,
        best_bound=max(0.0, best_bound),
        mip_gap=max(0.0, mip_gap),
        within_requested_gap=within_requested_gap,
        solve_time_s=max(0.0, solve_time_s),
        model_count=1,
        operation_count=operation_count,
        hop_count=hop_count,
        makespan_us=max(0.0, makespan_us),
        maximum_normalized_resource_load=max(
            0.0,
            maximum_normalized_resource_load,
        ),
        solver_name="gurobi",
        solver_version=_solver_version(gp),
        solver_seed=problem.inputs.solver.solver_seed,
        thread_count=thread_count,
        termination_reason=termination_reason,
    )


def _empty_candidate(
    *,
    gp,
    problem: SolverProblem,
    channel_count: int,
    objective: ObjectiveMode,
    status: SolveStatus,
    solve_time_s: float,
    thread_count: int,
    termination_reason: str,
) -> SolveCandidate:
    return SolveCandidate(
        candidate_id=_candidate_id(problem, channel_count, objective),
        node_schedules={},
        objective_mode=objective,
        channel_count=channel_count,
        metrics=_metrics(
            gp=gp,
            problem=problem,
            status=status,
            solve_time_s=solve_time_s,
            thread_count=thread_count,
            termination_reason=termination_reason,
        ),
        selected_best=False,
        proven_optimal=False,
        search_space_restricted=problem.search_space_restricted,
        restrictions=problem.restrictions,
        parent_candidate_id=None,
    )


def _add_disjunction(
    model,
    gp,
    first_active,
    second_active,
    first_start,
    first_end,
    second_start,
    second_end,
    name: str,
) -> None:
    both = model.addVar(vtype=gp.GRB.BINARY, name=name + "-both")
    first_before = model.addVar(
        vtype=gp.GRB.BINARY,
        name=name + "-first",
    )
    second_before = model.addVar(
        vtype=gp.GRB.BINARY,
        name=name + "-second",
    )
    model.addGenConstrAnd(both, [first_active, second_active])
    model.addConstr(first_before + second_before == both)
    model.addGenConstrIndicator(
        first_before,
        True,
        second_start >= first_end,
    )
    model.addGenConstrIndicator(
        second_before,
        True,
        first_start >= second_end,
    )


def _build_model(
    gp,
    problem: SolverProblem,
    trees: Tuple[_Tree, ...],
    channel_count: int,
    objective: ObjectiveMode,
    budget: ModelBudget,
    warm_start: Optional[Schedule],
):
    model_name = "vericcl-{}-k{:02d}-{}".format(
        problem.node.node_id,
        channel_count,
        objective.value,
    )
    try:
        model = gp.Model(model_name)
    except gp.GurobiError as error:
        raise SolverUnavailableError(
            "Gurobi model creation failed: {}".format(error)
        ) from error
    thread_count = _effective_threads(problem)
    config = problem.inputs.solver
    model.Params.OutputFlag = 0
    model.Params.Seed = config.solver_seed
    model.Params.Threads = thread_count
    model.Params.TimeLimit = budget.seconds
    model.Params.MIPGap = 0.0 if config.require_proven_optimal else config.mip_gap
    edge_selected = {}
    flow_selected = {}
    channel_selected = {}
    resource_selected = {}
    start_time = {}
    end_time = {}
    ready_time = {}
    tree_order = {}
    operation_demand = {}
    operation_channels = {}
    operations = []
    for tree in trees:
        for rank in problem.node.communication_group:
            ready_time[(tree.index, rank)] = model.addVar(
                lb=0.0,
                vtype=gp.GRB.CONTINUOUS,
                name="ready-t{:04d}-r{:04d}".format(tree.index, rank),
            )
            tree_order[(tree.index, rank)] = model.addVar(
                lb=0.0,
                ub=float(len(problem.node.communication_group) - 1),
                vtype=gp.GRB.CONTINUOUS,
                name="order-t{:04d}-r{:04d}".format(tree.index, rank),
            )
        model.addConstr(ready_time[(tree.index, tree.root_rank)] == 0.0)
        model.addConstr(tree_order[(tree.index, tree.root_rank)] == 0.0)
        for edge_index, link in enumerate(tree.edges):
            demand = _representative_demand(tree, link)
            usable_channels = available_channel_count(
                problem,
                demand,
                link.src_rank,
                link.dst_rank,
                channel_count,
            )
            if usable_channels < 1:
                continue
            key = (tree.index, link)
            operation_demand[key] = demand
            operation_channels[key] = usable_channels
            operations.append(key)
            selected = model.addVar(
                vtype=gp.GRB.BINARY,
                name="edge-t{:04d}-e{:04d}".format(tree.index, edge_index),
            )
            edge_selected[key] = selected
            start_time[key] = model.addVar(
                lb=0.0,
                vtype=gp.GRB.CONTINUOUS,
                name="start-t{:04d}-e{:04d}".format(tree.index, edge_index),
            )
            end_time[key] = model.addVar(
                lb=0.0,
                vtype=gp.GRB.CONTINUOUS,
                name="end-t{:04d}-e{:04d}".format(tree.index, edge_index),
            )
            duration = fixed_transfer_duration_us(
                problem,
                demand,
                link.src_rank,
                link.dst_rank,
                usable_channels,
            )
            model.addConstr(
                end_time[key]
                == start_time[key] + duration * selected
            )
            model.addGenConstrIndicator(
                selected,
                False,
                start_time[key] == 0.0,
            )
            model.addGenConstrIndicator(
                selected,
                True,
                start_time[key]
                >= ready_time[(tree.index, link.src_rank)],
            )
            model.addGenConstrIndicator(
                selected,
                True,
                ready_time[(tree.index, link.dst_rank)]
                == end_time[key],
            )
            model.addGenConstrIndicator(
                selected,
                True,
                tree_order[(tree.index, link.dst_rank)]
                >= tree_order[(tree.index, link.src_rank)] + 1.0,
            )
            channels = []
            for channel in range(usable_channels):
                variable = model.addVar(
                    vtype=gp.GRB.BINARY,
                    name="channel-t{:04d}-e{:04d}-c{:03d}".format(
                        tree.index,
                        edge_index,
                        channel,
                    ),
                )
                channel_selected[(key, channel)] = variable
                channels.append(variable)
            model.addConstr(gp.quicksum(channels) == selected)
            physical = physical_link_key(
                demand,
                link.src_rank,
                link.dst_rank,
            )
            physical_edge = problem.topology.link(physical)
            for resource_id in physical_edge.resource_ids:
                resource = problem.topology.shared_resources[resource_id]
                slot_count = min(channel_count, resource.max_channels)
                slots = []
                for slot in range(slot_count):
                    variable = model.addVar(
                        vtype=gp.GRB.BINARY,
                        name="resource-t{:04d}-e{:04d}-{}-s{:03d}".format(
                            tree.index,
                            edge_index,
                            resource_id,
                            slot,
                        ),
                    )
                    resource_selected[(key, resource_id, slot)] = variable
                    slots.append(variable)
                model.addConstr(gp.quicksum(slots) == selected)
    operation_set = frozenset(operations)
    if problem.inputs.strategies.batching:
        assignments = demand_batch_assignments(problem, channel_count)
        tree_by_demand_id = {
            demand.demand_id: tree
            for tree in trees
            for demand in tree.demands
        }
        trees_by_batch = {}
        for demand_id, batch_id in assignments.items():
            trees_by_batch.setdefault(batch_id, set()).add(
                tree_by_demand_id[demand_id].index
            )
        for tree_indices in trees_by_batch.values():
            ordered_trees = sorted(tree_indices)
            if len(ordered_trees) < 2:
                continue
            reference = ordered_trees[0]
            reference_edges = {
                key[1] for key in operations if key[0] == reference
            }
            for tree_index in ordered_trees[1:]:
                current_edges = {
                    key[1] for key in operations if key[0] == tree_index
                }
                if current_edges != reference_edges:
                    raise SemanticError(
                        "batch-equivalent trees must expose identical edges"
                    )
                for link in sorted(reference_edges):
                    model.addConstr(
                        edge_selected[(tree_index, link)]
                        == edge_selected[(reference, link)]
                    )
    for tree in trees:
        tree_operations = tuple(
            key for key in operations if key[0] == tree.index
        )
        for demand_index, demand in enumerate(tree.demands):
            demand_edges = tuple(
                link
                for link in sorted(demand.legal_links)
                if (tree.index, link) in operation_set
            )
            for edge_index, link in enumerate(demand_edges):
                variable = model.addVar(
                    vtype=gp.GRB.BINARY,
                    name="flow-d{:04d}-e{:04d}".format(
                        demand_index + tree.index * 10000,
                        edge_index,
                    ),
                )
                flow_selected[(demand.demand_id, link)] = variable
                model.addConstr(variable <= edge_selected[(tree.index, link)])
            for rank in problem.node.communication_group:
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
                if rank == demand.root_rank:
                    model.addConstr(outgoing - incoming == 1)
                elif rank == demand.required_leaf_rank:
                    model.addConstr(incoming - outgoing == 1)
                else:
                    model.addConstr(outgoing - incoming == 0)
                model.addConstr(outgoing <= 1)
                model.addConstr(incoming <= 1)
        for link in tree.edges:
            key = (tree.index, link)
            if key not in edge_selected:
                continue
            users = [
                flow_selected[(demand.demand_id, link)]
                for demand in tree.demands
                if (demand.demand_id, link) in flow_selected
            ]
            model.addConstr(edge_selected[key] <= gp.quicksum(users))
        for rank in problem.node.communication_group:
            incoming = gp.quicksum(
                edge_selected[key]
                for key in tree_operations
                if key[1].dst_rank == rank
            )
            if rank == tree.root_rank:
                model.addConstr(incoming == 0)
            else:
                model.addConstr(incoming <= 1)
    lane_groups = {}
    for key in operations:
        link = key[1]
        for channel in range(operation_channels[key]):
            lane_groups.setdefault(
                LaneKey(link.src_rank, link.dst_rank, channel),
                [],
            ).append((key, channel_selected[(key, channel)]))
    disjunction_index = 0
    for entries in lane_groups.values():
        for first_index, (first_key, first_active) in enumerate(entries):
            for second_key, second_active in entries[first_index + 1 :]:
                _add_disjunction(
                    model,
                    gp,
                    first_active,
                    second_active,
                    start_time[first_key],
                    end_time[first_key],
                    start_time[second_key],
                    end_time[second_key],
                    "lane-order-{:08d}".format(disjunction_index),
                )
                disjunction_index += 1
    resource_groups = {}
    for (key, resource_id, slot), active in resource_selected.items():
        resource_groups.setdefault((resource_id, slot), []).append(
            (key, active)
        )
    for entries in resource_groups.values():
        for first_index, (first_key, first_active) in enumerate(entries):
            for second_key, second_active in entries[first_index + 1 :]:
                _add_disjunction(
                    model,
                    gp,
                    first_active,
                    second_active,
                    start_time[first_key],
                    end_time[first_key],
                    start_time[second_key],
                    end_time[second_key],
                    "resource-order-{:08d}".format(disjunction_index),
                )
                disjunction_index += 1
    resource_work = {}
    for key in operations:
        demand = operation_demand[key]
        link = key[1]
        physical = physical_link_key(
            demand,
            link.src_rank,
            link.dst_rank,
        )
        duration = fixed_transfer_duration_us(
            problem,
            demand,
            link.src_rank,
            link.dst_rank,
            operation_channels[key],
        )
        resource_work.setdefault(("link", physical), []).append(
            (
                duration / operation_channels[key],
                edge_selected[key],
            )
        )
        for resource_id in problem.topology.link(physical).resource_ids:
            slots = min(
                channel_count,
                problem.topology.shared_resources[
                    resource_id
                ].max_channels,
            )
            resource_work.setdefault(("resource", resource_id), []).append(
                (duration / slots, edge_selected[key])
            )
    maximum_resource_load = model.addVar(
        lb=0.0,
        vtype=gp.GRB.CONTINUOUS,
        name="maximum-resource-load-us",
    )
    for entries in resource_work.values():
        model.addConstr(
            maximum_resource_load
            >= gp.quicksum(coefficient * active for coefficient, active in entries)
        )
    makespan = model.addVar(
        lb=0.0,
        vtype=gp.GRB.CONTINUOUS,
        name="makespan-us",
    )
    for tree in trees:
        for demand in tree.demands:
            model.addConstr(
                makespan
                >= ready_time[(tree.index, demand.required_leaf_rank)]
            )
    configure_lexicographic_objective(
        model,
        gp,
        objective,
        ObjectiveExpressions(
            makespan=makespan,
            operation_count=gp.quicksum(edge_selected.values()),
            hop_count=gp.quicksum(flow_selected.values()),
            maximum_resource_load=maximum_resource_load,
        ),
    )
    variables = _Variables(
        edge_selected=edge_selected,
        flow_selected=flow_selected,
        channel_selected=channel_selected,
        resource_selected=resource_selected,
        start_time=start_time,
        end_time=end_time,
        ready_time=ready_time,
        makespan=makespan,
        maximum_resource_load=maximum_resource_load,
    )
    _apply_warm_start(warm_start, trees, variables)
    model.update()
    return model, variables, operation_demand, operation_channels, thread_count


def _apply_warm_start(
    warm_start: Optional[Schedule],
    trees: Tuple[_Tree, ...],
    variables: _Variables,
) -> None:
    if warm_start is None:
        return
    contributors = warm_start.metadata.get(
        "tree_contributors",
        warm_start.metadata.get("semantic_contributors", {}),
    )
    roots = warm_start.metadata.get("path_roots", {})
    tree_by_key = {tree.key: tree for tree in trees}
    for transfer in warm_start.transfers:
        member_values = tuple(
            contributors.get(
                transfer.transfer_id,
                tuple(sorted(transfer.member_slice_ids)),
            )
        )
        if not member_values or transfer.transfer_id not in roots:
            continue
        logical_position = member_values[0] % warm_start.slice_count
        reduction_dual = bool(warm_start.metadata.get("reduction_dual", False))
        key = (
            roots[transfer.transfer_id],
            logical_position,
            tuple(sorted(member_values)),
            reduction_dual,
        )
        tree = tree_by_key.get(key)
        if tree is None:
            continue
        operation_key = (
            tree.index,
            LinkKey(transfer.src_rank, transfer.dst_rank),
        )
        if operation_key not in variables.edge_selected:
            continue
        variables.edge_selected[operation_key].Start = 1.0
        channel_variable = variables.channel_selected.get(
            (operation_key, transfer.channel)
        )
        if channel_variable is not None:
            channel_variable.Start = 1.0
        variables.start_time[operation_key].Start = transfer.st_time
        variables.end_time[operation_key].Start = transfer.ed_time


def _normalized_value(value: float) -> float:
    if abs(value) <= NUMERICAL_TOLERANCE:
        return 0.0
    return float(value)


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
            gap = 0.0 if abs(progress.best_bound) <= NUMERICAL_TOLERANCE else 1.0
        else:
            gap = abs(progress.best_value - progress.best_bound) / denominator
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


def _extract_operations(
    problem: SolverProblem,
    trees: Tuple[_Tree, ...],
    variables: _Variables,
    operation_demand: Mapping[OperationKey, TransferDemand],
    operation_channels: Mapping[OperationKey, int],
) -> Tuple[_OperationValue, ...]:
    tree_by_index = {tree.index: tree for tree in trees}
    operations = []
    for key in sorted(variables.edge_selected, key=lambda value: (value[0], value[1])):
        if variables.edge_selected[key].X <= 0.5:
            continue
        tree = tree_by_index[key[0]]
        link = key[1]
        channels = [
            channel
            for channel in range(operation_channels[key])
            if variables.channel_selected[(key, channel)].X > 0.5
        ]
        if len(channels) != 1:
            raise SemanticError("MILP operation has an invalid channel assignment")
        resource_slots = {}
        physical = physical_link_key(
            operation_demand[key],
            link.src_rank,
            link.dst_rank,
        )
        for resource_id in problem.topology.link(physical).resource_ids:
            slots = [
                slot
                for operation_key, current_resource, slot in variables.resource_selected
                if operation_key == key
                and current_resource == resource_id
                and variables.resource_selected[
                    (operation_key, current_resource, slot)
                ].X
                > 0.5
            ]
            if len(slots) != 1:
                raise SemanticError(
                    "MILP operation has an invalid shared-resource slot"
                )
            resource_slots[resource_id] = slots[0]
        duration = fixed_transfer_duration_us(
            problem,
            operation_demand[key],
            link.src_rank,
            link.dst_rank,
            operation_channels[key],
        )
        operations.append(
            _OperationValue(
                key=key,
                tree=tree,
                link=link,
                channel=channels[0],
                start_time=_normalized_value(variables.start_time[key].X),
                end_time=_normalized_value(variables.end_time[key].X),
                duration=duration,
                resource_slots=resource_slots,
            )
        )
    return tuple(operations)


def _flow_edges(
    trees: Tuple[_Tree, ...],
    variables: _Variables,
) -> Mapping[str, FrozenSet[LinkKey]]:
    result = {}
    for tree in trees:
        for demand in tree.demands:
            result[demand.demand_id] = frozenset(
                link
                for current_demand_id, link in variables.flow_selected
                if current_demand_id == demand.demand_id
                and variables.flow_selected[(current_demand_id, link)].X > 0.5
            )
    return result


def _validate_extracted(
    problem: SolverProblem,
    trees: Tuple[_Tree, ...],
    operations: Tuple[_OperationValue, ...],
    flows: Mapping[str, FrozenSet[LinkKey]],
) -> None:
    operation_by_key = {operation.key: operation for operation in operations}
    for tree in trees:
        selected = {
            operation.link
            for operation in operations
            if operation.tree.index == tree.index
        }
        parents = {}
        for link in selected:
            if link.dst_rank == tree.root_rank:
                raise SemanticError("MILP tree enters its root")
            if link.dst_rank in parents:
                raise SemanticError("MILP tree gives a rank multiple parents")
            parents[link.dst_rank] = link.src_rank
        used = set()
        for demand in tree.demands:
            demand_flow = flows[demand.demand_id]
            if not demand_flow <= demand.legal_links:
                raise SemanticError("MILP flow uses a forbidden or illegal edge")
            outgoing = {}
            for link in demand_flow:
                outgoing.setdefault(link.src_rank, []).append(link.dst_rank)
                used.add(link)
            rank = demand.root_rank
            visited = {rank}
            while rank != demand.required_leaf_rank:
                destinations = outgoing.get(rank, ())
                if len(destinations) != 1:
                    raise SemanticError("MILP flow does not form one chain")
                rank = destinations[0]
                if rank in visited:
                    raise SemanticError("MILP flow contains a cycle")
                visited.add(rank)
            if len(demand_flow) != len(visited) - 1:
                raise SemanticError("MILP flow contains unused edges")
        if used != selected:
            raise SemanticError("MILP tree contains an unused physical edge")
        for operation in operations:
            if operation.tree.index != tree.index:
                continue
            if not math.isclose(
                operation.end_time - operation.start_time,
                operation.duration,
                rel_tol=0.0,
                abs_tol=NUMERICAL_TOLERANCE,
            ):
                raise SemanticError("MILP transfer duration is numerically invalid")
            parent_rank = parents.get(operation.link.src_rank)
            if parent_rank is not None:
                parent = operation_by_key[
                    (tree.index, LinkKey(parent_rank, operation.link.src_rank))
                ]
                if operation.start_time + NUMERICAL_TOLERANCE < parent.end_time:
                    raise SemanticError("MILP transfer violates state readiness")
    lane_groups = {}
    for operation in operations:
        lane_groups.setdefault(
            LaneKey(
                operation.link.src_rank,
                operation.link.dst_rank,
                operation.channel,
            ),
            [],
        ).append(operation)
    for entries in lane_groups.values():
        ordered = sorted(entries, key=lambda item: (item.start_time, item.key))
        for first, second in zip(ordered, ordered[1:]):
            if second.start_time + NUMERICAL_TOLERANCE < first.end_time:
                raise SemanticError("MILP lane intervals overlap")
    resource_groups = {}
    for operation in operations:
        for resource_id, slot in operation.resource_slots.items():
            resource_groups.setdefault((resource_id, slot), []).append(operation)
    for entries in resource_groups.values():
        ordered = sorted(entries, key=lambda item: (item.start_time, item.key))
        for first, second in zip(ordered, ordered[1:]):
            if second.start_time + NUMERICAL_TOLERANCE < first.end_time:
                raise SemanticError("MILP shared-resource intervals overlap")


def _build_schedule(
    problem: SolverProblem,
    trees: Tuple[_Tree, ...],
    operations: Tuple[_OperationValue, ...],
    flows: Mapping[str, FrozenSet[LinkKey]],
    channel_count: int,
) -> Schedule:
    _validate_extracted(problem, trees, operations, flows)
    route_ids = {
        tree.index: "tree-{:08d}".format(tree.index) for tree in trees
    }

    def ordered_path(
        demand: TransferDemand,
        selected: FrozenSet[LinkKey],
    ) -> Tuple[LinkKey, ...]:
        outgoing = {link.src_rank: link for link in selected}
        path = []
        rank = demand.root_rank
        while rank != demand.required_leaf_rank:
            link = outgoing[rank]
            path.append(link)
            rank = link.dst_rank
        return tuple(path)

    routed_trees = tuple(
        RoutedTree(
            route_id=route_ids[tree.index],
            root_rank=tree.root_rank,
            logical_position=tree.logical_position,
            contributors=tree.contributors,
            reduction_dual=tree.reduction_dual,
            demands=tree.demands,
            selected_paths=tuple(
                (
                    demand.demand_id,
                    ordered_path(demand, flows[demand.demand_id]),
                )
                for demand in tree.demands
            ),
        )
        for tree in trees
    )
    routed_operations = tuple(
        RoutedOperation(
            route_id=route_ids[operation.tree.index],
            link=operation.link,
            channel=operation.channel,
            st_time=operation.start_time,
            ed_time=operation.end_time,
            resource_slots=tuple(sorted(operation.resource_slots.items())),
        )
        for operation in operations
    )
    identifiers = {
        (route_ids[operation.tree.index], operation.link): (
            "{}-milp-t{:08d}".format(
                problem.node.node_id,
                index,
            )
        )
        for index, operation in enumerate(
            sorted(operations, key=lambda item: (item.key[0], item.key[1]))
        )
    }
    return materialize_route_schedule(
        node=problem.node,
        trees=routed_trees,
        operations=routed_operations,
        transfer_ids=identifiers,
        schedule_id="{}-milp-k{:02d}".format(
            problem.node.node_id,
            channel_count,
        ),
        rank_count=problem.topology.rank_count,
        slice_count=problem.slice_count,
        slice_size_bytes=problem.slice_size_bytes,
        backend="gurobi",
        channel_count=channel_count,
        restrictions=problem.restrictions,
        routing_only=False,
        include_resource_order=True,
        include_final_metadata=False,
        extra_metadata={
            "selected_flows": {
                demand_id: tuple(
                    (link.src_rank, link.dst_rank)
                    for link in sorted(links)
                )
                for demand_id, links in sorted(flows.items())
            },
            "numerical_tolerance": NUMERICAL_TOLERANCE,
        },
    )


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


def _maximum_normalized_load(
    problem: SolverProblem,
    operations: Tuple[_OperationValue, ...],
    channel_count: int,
) -> float:
    if not operations:
        return 0.0
    link_durations = {}
    link_capacities = {}
    resource_durations = {}
    for operation in operations:
        demand = _representative_demand(operation.tree, operation.link)
        physical = physical_link_key(
            demand,
            operation.link.src_rank,
            operation.link.dst_rank,
        )
        link_durations[physical] = (
            link_durations.get(physical, 0.0) + operation.duration
        )
        link_capacities[physical] = available_channel_count(
            problem,
            demand,
            operation.link.src_rank,
            operation.link.dst_rank,
            channel_count,
        )
        for resource_id in operation.resource_slots:
            resource_durations[resource_id] = (
                resource_durations.get(resource_id, 0.0)
                + operation.duration
            )
    loads = [
        duration / link_capacities[link]
        for link, duration in link_durations.items()
    ]
    loads.extend(
        duration
        / (
            min(
                channel_count,
                problem.topology.shared_resources[
                    resource_id
                ].max_channels,
            )
        )
        for resource_id, duration in resource_durations.items()
    )
    return max(loads, default=0.0)


def solve_milp(
    problem: SolverProblem,
    channel_count: int,
    objective: ObjectiveMode,
    budget: ModelBudget,
    warm_start: Optional[Schedule],
) -> SolveCandidate:
    if not isinstance(problem, SolverProblem):
        raise SemanticError("problem must be a SolverProblem")
    channels = _positive_integer(channel_count, "channel_count")
    if channels > problem.inputs.solver.max_channels:
        raise SemanticError("channel_count exceeds the configured maximum")
    if not isinstance(objective, ObjectiveMode):
        raise SemanticError("objective must be an ObjectiveMode")
    if not isinstance(budget, ModelBudget):
        raise SemanticError("budget must be a ModelBudget")
    if warm_start is not None and not isinstance(warm_start, Schedule):
        raise SemanticError("warm_start must be a Schedule or None")
    gp = GurobiAdapter.require()
    threads = _effective_threads(problem)
    if problem.infeasible_demand_ids:
        return _empty_candidate(
            gp=gp,
            problem=problem,
            channel_count=channels,
            objective=objective,
            status=SolveStatus.INFEASIBLE,
            solve_time_s=0.0,
            thread_count=threads,
            termination_reason="precheck_infeasible",
        )
    if budget.seconds <= 0.0:
        return _empty_candidate(
            gp=gp,
            problem=problem,
            channel_count=channels,
            objective=objective,
            status=SolveStatus.NOT_RUN,
            solve_time_s=0.0,
            thread_count=threads,
            termination_reason="budget_exhausted",
        )
    trees = _trees(problem)
    if not trees:
        schedule = Schedule(
            schedule_id="{}-milp-local".format(problem.node.node_id),
            transfers=(),
            final_state_ids=tuple(
                "{}-r{:08d}-o{:08d}".format(
                    problem.node.node_id,
                    slot.rank,
                    slot.offset,
                )
                for slot in problem.node.logical_output.values
            ),
            rank_count=problem.topology.rank_count,
            slice_count=problem.slice_count,
            slice_size_bytes=problem.slice_size_bytes,
            metadata={
                "backend": "gurobi",
                "channel_count": channels,
                "path_scope": "stage_suffix",
                "path_roots": {},
                "reduction_dual": False,
            },
        )
        return SolveCandidate(
            candidate_id=_candidate_id(problem, channels, objective),
            node_schedules={problem.node.node_id: schedule},
            objective_mode=objective,
            channel_count=channels,
            metrics=_metrics(
                gp=gp,
                problem=problem,
                status=SolveStatus.OPTIMAL,
                solve_time_s=0.0,
                thread_count=threads,
                termination_reason="local_only",
                objective_values=(0.0, 0.0, 0.0)
                if objective is ObjectiveMode.LATENCY
                else (0.0, 0.0),
            ),
            selected_best=False,
            proven_optimal=(
                problem.inputs.solver.require_proven_optimal
                and not problem.search_space_restricted
            ),
            search_space_restricted=problem.search_space_restricted,
            restrictions=problem.restrictions,
            parent_candidate_id=None,
        )
    started = time.monotonic()
    model, variables, operation_demand, operation_channels, threads = _build_model(
        gp,
        problem,
        trees,
        channels,
        objective,
        budget,
        warm_start,
    )
    progress = _PrimaryObjectiveProgress(gp)
    model.optimize(progress)
    elapsed = time.monotonic() - started
    status = _status(gp, model)
    if model.SolCount <= 0:
        candidate = _empty_candidate(
            gp=gp,
            problem=problem,
            channel_count=channels,
            objective=objective,
            status=status,
            solve_time_s=elapsed,
            thread_count=threads,
            termination_reason="gurobi_status_{}".format(model.Status),
        )
        model.dispose()
        return candidate
    try:
        operations = _extract_operations(
            problem,
            trees,
            variables,
            operation_demand,
            operation_channels,
        )
        flows = _flow_edges(trees, variables)
        schedule = _build_schedule(
            problem,
            trees,
            operations,
            flows,
            channels,
        )
    except SemanticError as error:
        candidate = _empty_candidate(
            gp=gp,
            problem=problem,
            channel_count=channels,
            objective=objective,
            status=SolveStatus.ERROR,
            solve_time_s=elapsed,
            thread_count=threads,
            termination_reason="invalid_incumbent: {}".format(error),
        )
        model.dispose()
        return candidate
    makespan = _normalized_value(variables.makespan.X)
    maximum_resource_load = _maximum_normalized_load(
        problem,
        operations,
        channels,
    )
    hop_count = sum(len(value) for value in flows.values())
    objective_values = (
        (makespan, float(len(operations)), float(hop_count))
        if objective is ObjectiveMode.LATENCY
        else (maximum_resource_load, makespan)
    )
    best_bound, mip_gap = _primary_bound_and_gap(
        model,
        status,
        objective_values[0],
        progress,
    )
    requested_gap = (
        0.0
        if problem.inputs.solver.require_proven_optimal
        else problem.inputs.solver.mip_gap
    )
    within_requested_gap = mip_gap <= requested_gap + NUMERICAL_TOLERANCE
    proven_optimal = (
        status is SolveStatus.OPTIMAL
        and requested_gap == 0.0
        and not problem.search_space_restricted
    )
    metrics = _metrics(
        gp=gp,
        problem=problem,
        status=status,
        solve_time_s=elapsed,
        thread_count=threads,
        termination_reason="gurobi_status_{}".format(model.Status),
        objective_values=objective_values,
        best_bound=best_bound,
        mip_gap=mip_gap,
        within_requested_gap=within_requested_gap,
        operation_count=len(operations),
        hop_count=hop_count,
        makespan_us=makespan,
        maximum_normalized_resource_load=maximum_resource_load,
    )
    candidate = SolveCandidate(
        candidate_id=_candidate_id(problem, channels, objective),
        node_schedules={problem.node.node_id: schedule},
        objective_mode=objective,
        channel_count=channels,
        metrics=metrics,
        selected_best=False,
        proven_optimal=proven_optimal,
        search_space_restricted=problem.search_space_restricted,
        restrictions=problem.restrictions,
        parent_candidate_id=None,
    )
    model.dispose()
    return candidate
