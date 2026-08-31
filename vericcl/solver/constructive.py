from dataclasses import dataclass, replace
from typing import Dict, FrozenSet, List, Mapping, Tuple

from vericcl.errors import ConstructionInfeasibleError, SemanticError
from vericcl.input.models import ObjectiveMode, ResolvedInput
from vericcl.semantics.atom import Atom, PathStage, Schedule, Symbol, Transfer
from vericcl.solver.demands import (
    SolverProblem,
    TransferDemand,
    build_solver_problem,
)
from vericcl.solver.model import SolveStatus, SolverMetrics
from vericcl.solver.routing import RoutePattern, RoutingModelStats
from vericcl.solver.scheduling import (
    available_channel_count,
    demand_batch_assignments,
    fixed_transfer_duration_us,
    physical_link_key,
)
from vericcl.topology.model import LaneKey, LinkKey
from vericcl.topology.model import Topology


TreeKey = Tuple[int, int, Tuple[int, ...], bool]


@dataclass(frozen=True)
class _LaneChoice:
    channel: int
    start_time: float
    end_time: float
    duration: float


@dataclass(frozen=True)
class _DraftTransfer:
    transfer_id: str
    tree_key: TreeKey
    src_rank: int
    dst_rank: int
    channel: int
    stage_id: int
    st_time: float
    ed_time: float
    semantic_ready_time: float
    semantic_predecessor_ids: FrozenSet[str]
    predecessor_ids: FrozenSet[str]


def _positive_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SemanticError("{} must be a positive integer".format(field))
    return value


def _tree_key(demand: TransferDemand) -> TreeKey:
    return (
        demand.root_rank,
        demand.logical_position,
        tuple(sorted(demand.contributors)),
        demand.reduction_dual,
    )


def _physical_key(demand: TransferDemand, src: int, dst: int) -> LinkKey:
    return physical_link_key(demand, src, dst)


def _usable_channels(
    problem: SolverProblem,
    demand: TransferDemand,
    src: int,
    dst: int,
    requested: int,
) -> int:
    return available_channel_count(
        problem,
        demand,
        src,
        dst,
        requested,
    )


def _duration(
    problem: SolverProblem,
    demand: TransferDemand,
    src: int,
    dst: int,
    concurrency: int,
) -> float:
    return fixed_transfer_duration_us(
        problem,
        demand,
        src,
        dst,
        concurrency,
    )


def _choose_lane(
    *,
    problem: SolverProblem,
    demand: TransferDemand,
    src: int,
    dst: int,
    ready_time: float,
    channel_count: int,
    lane_ready: Dict[LaneKey, float],
    resource_ready: Dict[Tuple[str, int], float],
) -> _LaneChoice:
    usable = _usable_channels(
        problem,
        demand,
        src,
        dst,
        channel_count,
    )
    if usable < 1:
        raise ConstructionInfeasibleError(
            "no legal channel is available for {} to {}".format(src, dst)
        )
    duration = _duration(problem, demand, src, dst, usable)
    physical = _physical_key(demand, src, dst)
    edge = problem.topology.link(physical)
    choices = []
    for channel in range(usable):
        lane = LaneKey(src, dst, channel)
        start = max(
            [ready_time, lane_ready.get(lane, 0.0)]
            + [
                resource_ready.get((resource_id, channel), 0.0)
                for resource_id in edge.resource_ids
            ]
        )
        choices.append(
            (
                start,
                duration,
                src,
                dst,
                channel,
                _LaneChoice(channel, start, start + duration, duration),
            )
        )
    return min(choices)[-1]


def _path_is_tree_compatible(
    path: Tuple[int, ...],
    parents: Dict[int, int],
) -> bool:
    return all(
        dst not in parents or parents[dst] == src
        for src, dst in zip(path, path[1:])
    )


def _estimate_path(
    *,
    problem: SolverProblem,
    demand: TransferDemand,
    path: Tuple[int, ...],
    tree_key: TreeKey,
    channel_count: int,
    drafts_by_edge: Dict[Tuple[TreeKey, int, int], _DraftTransfer],
    node_ready: Dict[Tuple[TreeKey, int], float],
    lane_ready: Dict[LaneKey, float],
    resource_ready: Dict[Tuple[str, int], float],
) -> Tuple[float, float, int, Tuple[int, ...]]:
    temporary_lane = dict(lane_ready)
    temporary_resource = dict(resource_ready)
    ready = node_ready.get((tree_key, demand.root_rank), 0.0)
    total_new_duration = 0.0
    for src, dst in zip(path, path[1:]):
        existing = drafts_by_edge.get((tree_key, src, dst))
        if existing is not None:
            ready = max(ready, existing.ed_time)
            continue
        choice = _choose_lane(
            problem=problem,
            demand=demand,
            src=src,
            dst=dst,
            ready_time=ready,
            channel_count=channel_count,
            lane_ready=temporary_lane,
            resource_ready=temporary_resource,
        )
        physical = _physical_key(demand, src, dst)
        edge = problem.topology.link(physical)
        temporary_lane[LaneKey(src, dst, choice.channel)] = choice.end_time
        for resource_id in edge.resource_ids:
            temporary_resource[(resource_id, choice.channel)] = choice.end_time
        ready = choice.end_time
        total_new_duration += choice.duration
    return ready, total_new_duration, len(path) - 1, path


def _materialize_path(
    *,
    problem: SolverProblem,
    demand: TransferDemand,
    path: Tuple[int, ...],
    channel_count: int,
    drafts: List[_DraftTransfer],
    drafts_by_edge: Dict[Tuple[TreeKey, int, int], _DraftTransfer],
    parents: Dict[Tuple[TreeKey, int], int],
    node_ready: Dict[Tuple[TreeKey, int], float],
    incoming_transfer: Dict[Tuple[TreeKey, int], str],
    lane_ready: Dict[LaneKey, float],
    lane_last: Dict[LaneKey, str],
    resource_ready: Dict[Tuple[str, int], float],
    resource_last: Dict[Tuple[str, int], str],
) -> None:
    tree_key = _tree_key(demand)
    for src, dst in zip(path, path[1:]):
        existing = drafts_by_edge.get((tree_key, src, dst))
        if existing is not None:
            node_ready[(tree_key, dst)] = existing.ed_time
            incoming_transfer[(tree_key, dst)] = existing.transfer_id
            continue
        ready_time = node_ready.get((tree_key, src), 0.0)
        choice = _choose_lane(
            problem=problem,
            demand=demand,
            src=src,
            dst=dst,
            ready_time=ready_time,
            channel_count=channel_count,
            lane_ready=lane_ready,
            resource_ready=resource_ready,
        )
        transfer_id = "{}-t{:08d}".format(problem.node.node_id, len(drafts))
        lane = LaneKey(src, dst, choice.channel)
        physical = _physical_key(demand, src, dst)
        edge = problem.topology.link(physical)
        predecessors = set()
        dependency = incoming_transfer.get((tree_key, src))
        if dependency is not None:
            predecessors.add(dependency)
        semantic_predecessors = frozenset(predecessors)
        lane_predecessor = lane_last.get(lane)
        if lane_predecessor is not None:
            predecessors.add(lane_predecessor)
        for resource_id in edge.resource_ids:
            resource_predecessor = resource_last.get(
                (resource_id, choice.channel)
            )
            if resource_predecessor is not None:
                predecessors.add(resource_predecessor)
        draft = _DraftTransfer(
            transfer_id=transfer_id,
            tree_key=tree_key,
            src_rank=src,
            dst_rank=dst,
            channel=choice.channel,
            stage_id=demand.stage_id,
            st_time=choice.start_time,
            ed_time=choice.end_time,
            semantic_ready_time=ready_time,
            semantic_predecessor_ids=semantic_predecessors,
            predecessor_ids=frozenset(predecessors),
        )
        drafts.append(draft)
        drafts_by_edge[(tree_key, src, dst)] = draft
        parents[(tree_key, dst)] = src
        node_ready[(tree_key, dst)] = draft.ed_time
        incoming_transfer[(tree_key, dst)] = transfer_id
        lane_ready[lane] = draft.ed_time
        lane_last[lane] = transfer_id
        for resource_id in edge.resource_ids:
            resource_key = (resource_id, choice.channel)
            resource_ready[resource_key] = draft.ed_time
            resource_last[resource_key] = transfer_id


def _tree_path_drafts(
    draft: _DraftTransfer,
    drafts_by_edge: Dict[Tuple[TreeKey, int, int], _DraftTransfer],
    parents: Dict[Tuple[TreeKey, int], int],
) -> Tuple[_DraftTransfer, ...]:
    path = []
    destination = draft.dst_rank
    root = draft.tree_key[0]
    while destination != root:
        source = parents[(draft.tree_key, destination)]
        path.append(drafts_by_edge[(draft.tree_key, source, destination)])
        destination = source
    return tuple(reversed(path))


def _transfer(
    draft: _DraftTransfer,
    drafts_by_edge: Dict[Tuple[TreeKey, int, int], _DraftTransfer],
    parents: Dict[Tuple[TreeKey, int], int],
    slice_size_bytes: int,
    member_slice_ids: FrozenSet[int],
) -> Transfer:
    path = _tree_path_drafts(draft, drafts_by_edge, parents)
    symbols = tuple(
        Symbol(
            src_rank=item.src_rank,
            dst_rank=item.dst_rank,
            ready_time=item.semantic_ready_time,
        )
        for item in path
    )
    atoms = tuple(
        Atom(
            slice_id=slice_id,
            slice_size_bytes=slice_size_bytes,
            path=(
                PathStage(
                    stage_id=draft.stage_id,
                    operator="SEND",
                    symbols=symbols,
                ),
            ),
            st_time=draft.st_time,
            ed_time=draft.ed_time,
        )
        for slice_id in sorted(member_slice_ids)
    )
    return Transfer(
        transfer_id=draft.transfer_id,
        kind="SEND",
        src_rank=draft.src_rank,
        dst_rank=draft.dst_rank,
        channel=draft.channel,
        stage_id=draft.stage_id,
        member_slice_ids=member_slice_ids,
        atoms=atoms,
        st_time=draft.st_time,
        ed_time=draft.ed_time,
        predecessor_ids=draft.predecessor_ids,
    )


def construct_candidate(
    problem: SolverProblem,
    channel_count: int,
) -> Schedule:
    if not isinstance(problem, SolverProblem):
        raise SemanticError("problem must be a SolverProblem")
    channels = _positive_integer(channel_count, "channel_count")
    if channels > problem.inputs.solver.max_channels:
        raise SemanticError("channel_count exceeds the configured maximum")
    if problem.infeasible_demand_ids:
        raise ConstructionInfeasibleError(
            "demands have no legal path: {}".format(
                ", ".join(problem.infeasible_demand_ids)
            )
        )
    drafts: List[_DraftTransfer] = []
    drafts_by_edge = {}
    parents = {}
    node_ready = {}
    incoming_transfer = {}
    lane_ready = {}
    lane_last = {}
    resource_ready = {}
    resource_last = {}
    selected_paths = {}
    demand_batches = demand_batch_assignments(problem, channels)
    batch_path_templates = {}
    demands = sorted(
        problem.demands,
        key=lambda item: (len(item.candidate_paths), item.demand_id),
    )
    for demand in demands:
        tree_key = _tree_key(demand)
        tree_parents = {
            rank: parent
            for (key, rank), parent in parents.items()
            if key == tree_key
        }
        compatible = tuple(
            path
            for path in demand.candidate_paths
            if _path_is_tree_compatible(path, tree_parents)
        )
        if not compatible:
            raise ConstructionInfeasibleError(
                "greedy tree choices block demand {}".format(demand.demand_id)
            )
        template_key = (
            demand_batches[demand.demand_id],
            demand.required_leaf_rank,
        )
        template = batch_path_templates.get(template_key)
        if problem.inputs.strategies.batching and template is not None:
            if template not in compatible:
                raise ConstructionInfeasibleError(
                    "batch tree is incompatible with demand {}".format(
                        demand.demand_id
                    )
                )
            path = template
        else:
            path = min(
                compatible,
                key=lambda candidate: _estimate_path(
                    problem=problem,
                    demand=demand,
                    path=candidate,
                    tree_key=tree_key,
                    channel_count=channels,
                    drafts_by_edge=drafts_by_edge,
                    node_ready=node_ready,
                    lane_ready=lane_ready,
                    resource_ready=resource_ready,
                ),
            )
            if problem.inputs.strategies.batching:
                batch_path_templates[template_key] = path
        selected_paths[demand.demand_id] = path
        _materialize_path(
            problem=problem,
            demand=demand,
            path=path,
            channel_count=channels,
            drafts=drafts,
            drafts_by_edge=drafts_by_edge,
            parents=parents,
            node_ready=node_ready,
            incoming_transfer=incoming_transfer,
            lane_ready=lane_ready,
            lane_last=lane_last,
            resource_ready=resource_ready,
            resource_last=resource_last,
        )
    members_by_edge = {}
    demand_by_tree = {}
    for demand in demands:
        path = selected_paths[demand.demand_id]
        tree_key = _tree_key(demand)
        demand_by_tree.setdefault(tree_key, demand)
        for src, dst in zip(path, path[1:]):
            key = (tree_key, src, dst)
            members_by_edge.setdefault(key, set()).update(
                demand.member_slice_ids
            )
    transfers = tuple(
        _transfer(
            draft,
            drafts_by_edge,
            parents,
            problem.slice_size_bytes,
            frozenset(
                members_by_edge[
                    (draft.tree_key, draft.src_rank, draft.dst_rank)
                ]
            ),
        )
        for draft in drafts
    )
    final_state_ids = tuple(
        "{}-r{:08d}-o{:08d}".format(
            problem.node.node_id,
            slot.rank,
            slot.offset,
        )
        for slot in problem.node.logical_output.values
    )
    return Schedule(
        schedule_id="{}-constructive-k{:02d}".format(
            problem.node.node_id,
            channels,
        ),
        transfers=transfers,
        final_state_ids=final_state_ids,
        rank_count=problem.topology.rank_count,
        slice_count=problem.slice_count,
        slice_size_bytes=problem.slice_size_bytes,
        metadata={
            "backend": "constructive",
            "channel_count": channels,
            "path_scope": "stage_suffix",
            "path_roots": {
                draft.transfer_id: draft.tree_key[0]
                for draft in drafts
            },
            "reduction_dual": problem.reduction_dual,
            "restrictions": problem.restrictions,
            "selected_paths": {
                demand_id: path
                for demand_id, path in sorted(selected_paths.items())
            },
            "demand_batches": {
                demand_id: batch_id
                for demand_id, batch_id in sorted(demand_batches.items())
            },
            "semantic_contributors": {
                draft.transfer_id: tuple(
                    sorted(
                        members_by_edge[
                            (
                                draft.tree_key,
                                draft.src_rank,
                                draft.dst_rank,
                            )
                        ]
                    )
                )
                for draft in drafts
            },
            "semantic_predecessors": {
                draft.transfer_id: tuple(
                    sorted(draft.semantic_predecessor_ids)
                )
                for draft in drafts
            },
            "resource_slots": {
                draft.transfer_id: {
                    resource_id: draft.channel
                    for resource_id in problem.topology.link(
                        _physical_key(
                            demand_by_tree[draft.tree_key],
                            draft.src_rank,
                            draft.dst_rank,
                        )
                    ).resource_ids
                }
                for draft in drafts
            },
            "tree_contributors": {
                draft.transfer_id: draft.tree_key[2]
                for draft in drafts
            },
        },
    )


def _constructive_resource_load(schedule: Schedule) -> float:
    raw_slots = schedule.metadata.get("resource_slots", {})
    if not isinstance(raw_slots, Mapping):
        raise SemanticError("resource_slots metadata must be a mapping")
    loads = {}
    for transfer in schedule.transfers:
        duration = transfer.ed_time - transfer.st_time
        link = ("link", transfer.src_rank, transfer.dst_rank)
        loads[link] = loads.get(link, 0.0) + duration
        slots = raw_slots.get(transfer.transfer_id, {})
        if not isinstance(slots, Mapping):
            raise SemanticError("transfer resource slots must be a mapping")
        for resource_id in slots:
            key = ("resource", resource_id)
            loads[key] = loads.get(key, 0.0) + duration
    return max(loads.values(), default=0.0)


def construct_route_pattern(
    template,
    inputs: ResolvedInput,
    topology: Topology,
    channel_count: int,
    objective: ObjectiveMode,
) -> RoutePattern:
    from vericcl.solver.templates import SolverTemplate

    if not isinstance(template, SolverTemplate):
        raise SemanticError("template must be a SolverTemplate")
    if not isinstance(inputs, ResolvedInput):
        raise SemanticError("inputs must be a ResolvedInput")
    if not isinstance(topology, Topology):
        raise SemanticError("topology must be a Topology")
    channels = _positive_integer(channel_count, "channel_count")
    if not isinstance(objective, ObjectiveMode):
        raise SemanticError("objective must be an ObjectiveMode")
    if objective is ObjectiveMode.AUTO:
        raise SemanticError("AUTO must be resolved before route construction")
    base = build_solver_problem(
        template.representative.node,
        inputs,
        topology,
    )
    demand_ids = {
        demand.demand_id for demand in template.representative.demands
    }
    problem = replace(
        base,
        demands=template.representative.demands,
        infeasible_demand_ids=tuple(
            demand_id
            for demand_id in base.infeasible_demand_ids
            if demand_id in demand_ids
        ),
    )
    schedule = construct_candidate(problem, channels)
    raw_paths = schedule.metadata.get("selected_paths")
    if not isinstance(raw_paths, Mapping):
        raise SemanticError("constructive schedule omits selected paths")
    member_paths = tuple(
        (
            demand.demand_id,
            tuple(zip(raw_paths[demand.demand_id], raw_paths[demand.demand_id][1:])),
        )
        for demand in template.representative.demands
    )
    selected_edges = tuple(
        sorted({edge for _, path in member_paths for edge in path})
    )
    makespan = max(
        (transfer.ed_time for transfer in schedule.transfers),
        default=0.0,
    )
    resource_load = _constructive_resource_load(schedule)
    operation_count = len(selected_edges)
    hop_count = sum(len(path) for _, path in member_paths)
    objective_values = (
        (makespan, float(operation_count), float(hop_count))
        if objective is ObjectiveMode.LATENCY
        else (resource_load, makespan)
    )
    return RoutePattern(
        template_id=template.template_id,
        channel_count=channels,
        objective_mode=objective,
        selected_edges=selected_edges,
        member_paths=member_paths,
        metrics=SolverMetrics(
            status=SolveStatus.FEASIBLE,
            objective_values=objective_values,
            best_bound=0.0,
            mip_gap=0.0,
            within_requested_gap=False,
            solve_time_s=0.0,
            model_count=0,
            operation_count=operation_count,
            hop_count=hop_count,
            makespan_us=makespan,
            maximum_normalized_resource_load=resource_load,
            solver_name="constructive-route",
            solver_version="1",
            solver_seed=inputs.solver.solver_seed,
            thread_count=1,
            termination_reason="constructive_route_complete",
        ),
        model_stats=RoutingModelStats(0, 0, 0, 0.0, 0.0),
    )
