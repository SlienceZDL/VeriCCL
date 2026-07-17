from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Mapping, Optional, Tuple

from vericcl.errors import SemanticError
from vericcl.input.models import ResolvedInput
from vericcl.semantics.atom import (
    Atom,
    PathStage,
    Schedule,
    Symbol,
    Transfer,
)
from vericcl.solver.pruning import ranked_simple_paths
from vericcl.topology.model import LaneKey, LinkKey, PerformanceCurve, Topology
from vericcl.topology.performance import (
    safe_per_channel_bandwidth,
    transfer_duration_us,
)
from vericcl.tuning.model import RepairResult, RepairStatus, TuningOverlay
from vericcl.verification.bdd_flow import FlowReplacementHint
from vericcl.verification.constraints import verify_schedule_constraints
from vericcl.verification.flow_index import FlowRecord, build_flow_index
from vericcl.verification.model import ValidationStatus
from vericcl.verification.semantics import verify_schedule_semantics


@dataclass(frozen=True)
class _SelectedPath:
    candidate_id: str
    path: Tuple[int, ...]
    first_lane: LaneKey
    merge_transfer_ids: Tuple[Optional[str], ...]
    leaf_repair_hops: int
    cost: Tuple[object, ...]


def _failure(
    status: RepairStatus,
    code: str,
    message: str,
    **evidence
) -> RepairResult:
    values = dict(evidence)
    values["code"] = code
    values["message"] = message
    return RepairResult(
        status=status,
        schedule=None,
        changed_transfer_ids=frozenset(),
        selected_candidate_flow_id=None,
        method="greedy",
        evidence=values,
    )


def _curve_duration(
    curve: PerformanceCurve,
    slice_size_bytes: int,
    concurrency: int,
) -> float:
    if curve.is_calibrated:
        return curve.alpha_us + slice_size_bytes / safe_per_channel_bandwidth(
            curve,
            concurrency,
        )
    return curve.alpha_us + concurrency * curve.beta_effective_us


def _duration(
    topology: Topology,
    link: LinkKey,
    slice_size_bytes: int,
    channel_count: int,
) -> float:
    edge = topology.link(link)
    link_concurrency = min(channel_count, edge.max_channels)
    values = [
        transfer_duration_us(edge, slice_size_bytes, link_concurrency)
    ]
    for resource_id in edge.resource_ids:
        resource = topology.shared_resources[resource_id]
        resource_concurrency = min(channel_count, resource.max_channels)
        values.append(
            _curve_duration(
                resource.performance,
                slice_size_bytes,
                resource_concurrency,
            )
        )
    return max(values)


def _forbidden_edges(
    flow: FlowRecord,
    inputs: ResolvedInput,
    overlay: TuningOverlay,
) -> frozenset[LinkKey]:
    values = tuple(inputs.atom_constraints.forbidden_transfers) + tuple(
        overlay.temporary_forbidden
    )
    return frozenset(
        LinkKey(item.src_rank, item.dst_rank)
        for item in values
        if item.stage_id == flow.stage_id
        and item.slice_id in flow.member_slice_ids
    )


def _complete_path(
    path: Tuple[int, ...],
    flow: FlowRecord,
    topology: Topology,
    forbidden: frozenset[LinkKey],
) -> Tuple[Tuple[int, ...], int]:
    if len(path) < 2 or path[0] not in flow.ranks:
        raise SemanticError("candidate path does not start at the divergence rank")
    if len(path) != len(set(path)):
        raise SemanticError("candidate path must be simple")
    for src_rank, dst_rank in zip(path, path[1:]):
        key = LinkKey(src_rank, dst_rank)
        if key not in topology.links or key in forbidden:
            raise SemanticError("candidate path uses a forbidden or missing link")
    if path[-1] == flow.leaf_rank:
        return path, 0
    legal = frozenset(topology.links) - forbidden

    def edge_cost(src_rank: int, dst_rank: int) -> float:
        return topology.link(LinkKey(src_rank, dst_rank)).performance.invbw_us

    suffixes = ranked_simple_paths(
        legal,
        path[-1],
        flow.leaf_rank,
        edge_cost,
        limit=32,
    )
    suffixes = tuple(
        suffix
        for suffix in suffixes
        if not set(path[:-1]).intersection(suffix[1:])
    )
    if not suffixes:
        raise SemanticError("candidate path cannot repair the target leaf")
    completed = path[:-1] + suffixes[0]
    return completed, len(suffixes[0]) - 1


def _reduce_merge_transfer_ids(
    flow: FlowRecord,
    path: Tuple[int, ...],
    divergence_index: int,
    schedule: Schedule,
) -> Optional[Tuple[Optional[str], ...]]:
    original_suffix = flow.ranks[divergence_index:]
    if path == original_suffix:
        return tuple(None for _ in zip(path, path[1:]))
    tail = path[1:]
    anchors = tuple(
        other
        for other in build_flow_index(schedule).flows
        if other.flow_id != flow.flow_id
        and other.stage_id == flow.stage_id
        and other.operator == "REDUCE"
        and other.logical_slice_index == flow.logical_slice_index
        and other.leaf_rank == flow.leaf_rank
        and other.ranks == tail
        and not other.member_slice_ids.intersection(flow.member_slice_ids)
    )
    if not anchors:
        return None
    anchor = min(anchors, key=lambda item: item.flow_id)
    return (None,) + tuple(anchor.transfer_ids)


def _legal_options(
    flow: FlowRecord,
    hint: FlowReplacementHint,
    overlay: TuningOverlay,
    topology: Topology,
    inputs: ResolvedInput,
    schedule: Schedule,
) -> Tuple[_SelectedPath, ...]:
    forbidden = _forbidden_edges(flow, inputs, overlay)
    channel_count = overlay.channel_count or int(
        schedule.metadata.get(
            "channel_count",
            max((transfer.channel for transfer in schedule.transfers), default=0)
            + 1,
        )
    )
    weights = dict(overlay.path_weights)
    divergence_index = flow.ranks.index(hint.divergence_rank)
    options = []
    for candidate_id in hint.candidate_flow_ids:
        raw_path = tuple(hint.candidate_paths[candidate_id])
        try:
            path, repair_hops = _complete_path(
                raw_path,
                flow,
                topology,
                forbidden,
            )
        except SemanticError:
            continue
        first_lane = hint.candidate_first_lanes[candidate_id]
        if (
            not isinstance(first_lane, LaneKey)
            or first_lane == hint.bottleneck_lane
            or (first_lane.src_rank, first_lane.dst_rank)
            != (path[0], path[1])
            or first_lane.channel
            >= topology.link(LinkKey(path[0], path[1])).max_channels
        ):
            continue
        merge_transfer_ids = tuple(None for _ in zip(path, path[1:]))
        if flow.operator == "REDUCE":
            reduce_targets = _reduce_merge_transfer_ids(
                flow,
                path,
                divergence_index,
                schedule,
            )
            if reduce_targets is None:
                continue
            merge_transfer_ids = reduce_targets
        added_time = sum(
            _duration(
                topology,
                LinkKey(src_rank, dst_rank),
                schedule.slice_size_bytes,
                channel_count,
            )
            for src_rank, dst_rank in zip(path, path[1:])
        ) + weights.get(candidate_id, 0.0)
        lane_wait = max(
            0.0,
            hint.earliest_candidate_start_us - hint.wait_start_us,
        )
        resource_load = sum(
            len(topology.resources_for(LinkKey(src_rank, dst_rank)))
            for src_rank, dst_rank in zip(path, path[1:])
        )
        options.append(
            _SelectedPath(
                candidate_id=candidate_id,
                path=path,
                first_lane=first_lane,
                merge_transfer_ids=merge_transfer_ids,
                leaf_repair_hops=repair_hops,
                cost=(
                    added_time,
                    len(path) - 1,
                    lane_wait,
                    resource_load,
                    repair_hops,
                    candidate_id,
                ),
            )
        )
    return tuple(sorted(options, key=lambda item: item.cost))


def _earliest_start(
    intervals: Tuple[Tuple[float, float], ...],
    ready_time: float,
    duration: float,
) -> float:
    cursor = ready_time
    for start, end in sorted(intervals):
        if end <= cursor:
            continue
        if cursor + duration <= start:
            return cursor
        cursor = max(cursor, end)
    return cursor


def _stage_symbols(
    ranks: Tuple[int, ...],
    ready_times: Tuple[float, ...],
) -> Tuple[Symbol, ...]:
    return tuple(
        Symbol(src_rank, dst_rank, ready_time)
        for (src_rank, dst_rank), ready_time in zip(
            zip(ranks, ranks[1:]),
            ready_times,
        )
    )


def _replace_later_stage_path(
    atom: Atom,
    stage_id: int,
    symbols: Tuple[Symbol, ...],
    terminal_ready: float,
    st_time: float,
    ed_time: float,
) -> Atom:
    path = []
    replaced_stage = False
    for stage in atom.path:
        if stage.stage_id == stage_id:
            path.append(PathStage(stage_id, stage.operator, symbols))
            replaced_stage = True
        elif replaced_stage:
            path.append(
                PathStage(
                    stage.stage_id,
                    stage.operator,
                    tuple(
                        Symbol(
                            symbol.src_rank,
                            symbol.dst_rank,
                            max(symbol.ready_time, terminal_ready),
                        )
                        for symbol in stage.symbols
                    ),
                )
            )
        else:
            path.append(stage)
    return Atom(
        atom.slice_id,
        atom.slice_size_bytes,
        tuple(path),
        st_time,
        ed_time,
    )


def _operation_key(
    slice_id: int,
    stage_id: int,
    operator: str,
    src_rank: int,
    dst_rank: int,
) -> tuple:
    return slice_id, stage_id, operator, src_rank, dst_rank


def _reschedule(
    transfers: Tuple[Transfer, ...],
    semantic: Mapping[str, frozenset[str]],
    resource_slots: Mapping[str, Mapping[str, int]],
) -> Tuple[Transfer, ...]:
    by_id = {transfer.transfer_id: transfer for transfer in transfers}
    operations = {}
    for transfer in transfers:
        for atom in transfer.atoms:
            key = _operation_key(
                atom.slice_id,
                transfer.stage_id,
                transfer.kind,
                transfer.src_rank,
                transfer.dst_rank,
            )
            if key in operations and operations[key] != transfer.transfer_id:
                raise SemanticError("repair creates a duplicate path operation")
            operations[key] = transfer.transfer_id
    dependencies = {}
    for transfer in transfers:
        values = set(transfer.predecessor_ids) | set(
            semantic.get(transfer.transfer_id, ())
        )
        for atom in transfer.atoms:
            flattened = tuple(
                (stage, symbol)
                for stage in atom.path
                for symbol in stage.symbols
            )
            if len(flattened) > 1:
                stage, symbol = flattened[-2]
                key = _operation_key(
                    atom.slice_id,
                    stage.stage_id,
                    stage.operator,
                    symbol.src_rank,
                    symbol.dst_rank,
                )
                predecessor = operations.get(key)
                if predecessor is None:
                    raise SemanticError("repair path predecessor is missing")
                values.add(predecessor)
        values.discard(transfer.transfer_id)
        if not values <= set(by_id):
            raise SemanticError("repair dependency is missing")
        dependencies[transfer.transfer_id] = frozenset(values)

    lane_queues: Dict[LaneKey, list] = {}
    resource_queues: Dict[tuple, list] = {}
    order_key = lambda transfer_id: (
        by_id[transfer_id].st_time,
        by_id[transfer_id].ed_time,
        transfer_id,
    )
    for transfer in transfers:
        lane = LaneKey(
            transfer.src_rank,
            transfer.dst_rank,
            transfer.channel,
        )
        lane_queues.setdefault(lane, []).append(transfer.transfer_id)
        for resource_id, slot in resource_slots.get(
            transfer.transfer_id,
            {},
        ).items():
            resource_queues.setdefault((resource_id, slot), []).append(
                transfer.transfer_id
            )
    for queue in tuple(lane_queues.values()) + tuple(resource_queues.values()):
        queue.sort(key=order_key)

    completed = set()
    end_times = {}
    start_times = {}
    lane_available: Dict[LaneKey, float] = {}
    resource_available: Dict[tuple, float] = {}
    pending = set(by_id)
    while pending:
        candidates = []
        for transfer_id in pending:
            transfer = by_id[transfer_id]
            if not dependencies[transfer_id] <= completed:
                continue
            lane = LaneKey(
                transfer.src_rank,
                transfer.dst_rank,
                transfer.channel,
            )
            next_lane_transfer = next(
                item for item in lane_queues[lane] if item in pending
            )
            if next_lane_transfer != transfer_id:
                continue
            slots = resource_slots.get(transfer_id, {})
            if any(
                next(
                    item
                    for item in resource_queues[(resource_id, slot)]
                    if item in pending
                )
                != transfer_id
                for resource_id, slot in slots.items()
            ):
                continue
            candidates.append(transfer_id)
        if not candidates:
            raise SemanticError("repair scheduling cannot make progress")
        transfer_id = min(candidates, key=order_key)
        transfer = by_id[transfer_id]
        lane = LaneKey(
            transfer.src_rank,
            transfer.dst_rank,
            transfer.channel,
        )
        slots = resource_slots.get(transfer_id, {})
        start = max(
            [transfer.st_time, lane_available.get(lane, 0.0)]
            + [end_times[item] for item in dependencies[transfer_id]]
            + [
                resource_available.get((resource_id, slot), 0.0)
                for resource_id, slot in slots.items()
            ]
        )
        duration = transfer.ed_time - transfer.st_time
        end = start + duration
        start_times[transfer_id] = start
        end_times[transfer_id] = end
        lane_available[lane] = end
        for resource_id, slot in slots.items():
            resource_available[(resource_id, slot)] = end
        pending.remove(transfer_id)
        completed.add(transfer_id)

    operation_ends = {
        key: end_times[transfer_id] for key, transfer_id in operations.items()
    }
    rebuilt = []
    for transfer in transfers:
        atoms = []
        for atom in transfer.atoms:
            path = []
            previous_key = None
            previous_ready = 0.0
            for stage in atom.path:
                symbols = []
                for symbol in stage.symbols:
                    ready = (
                        symbol.ready_time
                        if previous_key is None
                        else operation_ends[previous_key]
                    )
                    ready = max(ready, previous_ready)
                    symbols.append(
                        Symbol(symbol.src_rank, symbol.dst_rank, ready)
                    )
                    previous_ready = ready
                    previous_key = _operation_key(
                        atom.slice_id,
                        stage.stage_id,
                        stage.operator,
                        symbol.src_rank,
                        symbol.dst_rank,
                    )
                path.append(
                    PathStage(stage.stage_id, stage.operator, tuple(symbols))
                )
            atoms.append(
                Atom(
                    atom.slice_id,
                    atom.slice_size_bytes,
                    tuple(path),
                    start_times[transfer.transfer_id],
                    end_times[transfer.transfer_id],
                )
            )
        rebuilt.append(
            replace(
                transfer,
                atoms=tuple(atoms),
                st_time=start_times[transfer.transfer_id],
                ed_time=end_times[transfer.transfer_id],
            )
        )
    return tuple(rebuilt)


def _replace_dependency_values(
    values: object,
    removed: frozenset[str],
    terminal_id: str,
) -> Tuple[str, ...]:
    normalized = set(values)
    if normalized.intersection(removed):
        normalized.difference_update(removed)
        normalized.add(terminal_id)
    return tuple(sorted(normalized))


def repair_flow_suffix(
    schedule: Schedule,
    hint: FlowReplacementHint,
    overlay: TuningOverlay,
    topology: Topology,
    inputs: ResolvedInput,
) -> RepairResult:
    try:
        overlay.validate_against(inputs, schedule, topology)
        if not isinstance(hint, FlowReplacementHint):
            raise SemanticError("hint must be a FlowReplacementHint")
        flow_index = build_flow_index(schedule)
        flow = flow_index.flow(hint.source_flow_id)
        if flow.demand_id != hint.demand_id:
            raise SemanticError("hint demand does not match source flow")
        if hint.divergence_rank not in flow.ranks:
            raise SemanticError("hint divergence rank is outside the flow")
        divergence_index = flow.ranks.index(hint.divergence_rank)
        if (
            divergence_index >= flow.comparison_end
            or flow.transfer_ids[divergence_index]
            != hint.waiting_transfer_id
        ):
            raise SemanticError("hint waiting transfer does not match source flow")
        options = _legal_options(
            flow,
            hint,
            overlay,
            topology,
            inputs,
            schedule,
        )
        if not options:
            return _failure(
                RepairStatus.INFEASIBLE,
                "no_legal_suffix",
                "no candidate suffix preserves repair boundaries",
            )
        selected = options[0]
        full_ranks = flow.ranks[:divergence_index] + selected.path
        first_difference = next(
            (
                index
                for index, (left, right) in enumerate(
                    zip(
                        zip(flow.ranks, flow.ranks[1:]),
                        zip(full_ranks, full_ranks[1:]),
                    )
                )
                if left != right
            ),
            divergence_index,
        )
        if first_difference >= flow.comparison_end:
            raise SemanticError("candidate changes only the aggregate suffix")
        old_suffix_ids = frozenset(
            flow.transfer_ids[first_difference : flow.comparison_end]
        )
        protected_ids = frozenset(
            transfer_id
            for other in flow_index.flows
            if other.flow_id != flow.flow_id
            and other.member_slice_ids.intersection(flow.member_slice_ids)
            for transfer_id in old_suffix_ids.intersection(
                other.comparison_transfer_ids
            )
        )
        by_id = {
            transfer.transfer_id: transfer for transfer in schedule.transfers
        }
        first_old = by_id[flow.transfer_ids[first_difference]]
        members = flow.member_slice_ids
        remaining = []
        removed = set()
        for transfer in schedule.transfers:
            if transfer.transfer_id not in old_suffix_ids:
                remaining.append(transfer)
                continue
            if transfer.transfer_id in protected_ids:
                remaining.append(transfer)
                continue
            remaining_atoms = tuple(
                atom for atom in transfer.atoms if atom.slice_id not in members
            )
            if remaining_atoms:
                remaining.append(
                    replace(
                        transfer,
                        member_slice_ids=frozenset(
                            atom.slice_id for atom in remaining_atoms
                        ),
                        atoms=remaining_atoms,
                    )
                )
            else:
                removed.add(transfer.transfer_id)
        removed_ids = frozenset(removed)
        if not removed_ids:
            raise SemanticError("candidate suffix is shared by another flow")

        metadata = dict(schedule.metadata)
        raw_semantic = metadata.get("semantic_predecessors", {})
        if not isinstance(raw_semantic, Mapping):
            raise SemanticError("semantic_predecessors must be a mapping")
        semantic = {
            transfer.transfer_id: frozenset(
                raw_semantic.get(
                    transfer.transfer_id,
                    transfer.predecessor_ids,
                )
            )
            for transfer in remaining
        }
        raw_slots = metadata.get("resource_slots", {})
        if not isinstance(raw_slots, Mapping):
            raise SemanticError("resource_slots must be a mapping")
        resource_slots = {
            transfer.transfer_id: dict(raw_slots.get(transfer.transfer_id, {}))
            for transfer in remaining
        }
        remaining_by_id = {
            transfer.transfer_id: transfer for transfer in remaining
        }
        channel_count = overlay.channel_count or int(
            metadata.get(
                "channel_count",
                max((transfer.channel for transfer in schedule.transfers), default=0)
                + 1,
            )
        )
        lane_intervals: Dict[LaneKey, list] = {}
        for transfer in remaining:
            lane = LaneKey(
                transfer.src_rank,
                transfer.dst_rank,
                transfer.channel,
            )
            lane_intervals.setdefault(lane, []).append(
                (transfer.st_time, transfer.ed_time)
            )

        prefix_ready = flow.ready_times[:first_difference]
        replacement_ranks = full_ranks[first_difference:]
        ready_values = list(prefix_ready)
        new_transfers = []
        merged_transfer_ids = set()
        previous_id = (
            flow.transfer_ids[first_difference - 1]
            if first_difference > 0
            else None
        )
        ready_time = flow.ready_times[first_difference]
        base_atoms = {
            atom.slice_id: atom for atom in first_old.atoms if atom.slice_id in members
        }
        if set(base_atoms) != set(members):
            raise SemanticError("repair source atoms are incomplete")
        for edge_index, (src_rank, dst_rank) in enumerate(
            zip(replacement_ranks, replacement_ranks[1:])
        ):
            key = LinkKey(src_rank, dst_rank)
            edge = topology.link(key)
            merge_transfer_id = selected.merge_transfer_ids[edge_index]
            merge_target = (
                remaining_by_id.get(merge_transfer_id)
                if merge_transfer_id is not None
                else None
            )
            if merge_transfer_id is not None and merge_target is None:
                raise SemanticError("aggregate merge target is unavailable")
            if merge_target is not None and (
                merge_target.kind != "REDUCE"
                or merge_target.stage_id != flow.stage_id
                or merge_target.src_rank != src_rank
                or merge_target.dst_rank != dst_rank
                or merge_target.member_slice_ids.intersection(members)
            ):
                raise SemanticError("aggregate merge target is incompatible")
            if merge_target is not None:
                channel = merge_target.channel
            elif edge_index == 0:
                channel = selected.first_lane.channel
            else:
                channel = min(
                    range(min(channel_count, edge.max_channels)),
                    key=lambda value: (
                        _earliest_start(
                            tuple(
                                lane_intervals.get(
                                    LaneKey(src_rank, dst_rank, value),
                                    (),
                                )
                            ),
                            ready_time,
                            _duration(
                                topology,
                                key,
                                schedule.slice_size_bytes,
                                channel_count,
                            ),
                        ),
                        value,
                    ),
                )
            duration = _duration(
                topology,
                key,
                schedule.slice_size_bytes,
                channel_count,
            )
            lane = LaneKey(src_rank, dst_rank, channel)
            if merge_target is not None:
                lane_intervals[lane].remove(
                    (merge_target.st_time, merge_target.ed_time)
                )
                start = max(merge_target.st_time, ready_time)
            else:
                start = _earliest_start(
                    tuple(lane_intervals.get(lane, ())),
                    ready_time,
                    duration,
                )
            end = start + duration
            ready_values.append(ready_time)
            stage_ranks = full_ranks[: first_difference + edge_index + 2]
            stage_ready = tuple(ready_values)
            atoms = []
            for member in sorted(members):
                base = base_atoms[member]
                earlier = tuple(
                    stage for stage in base.path if stage.stage_id < flow.stage_id
                )
                current = PathStage(
                    flow.stage_id,
                    flow.operator,
                    _stage_symbols(stage_ranks, stage_ready),
                )
                atoms.append(
                    Atom(
                        member,
                        schedule.slice_size_bytes,
                        earlier + (current,),
                        start,
                        end,
                    )
                )
            predecessor_ids = frozenset(
                {previous_id} if previous_id is not None else ()
            )
            if merge_target is not None:
                transfer_id = merge_target.transfer_id
                predecessor_ids |= frozenset(
                    item
                    for item in merge_target.predecessor_ids
                    if item not in removed_ids
                )
                transfer = replace(
                    merge_target,
                    member_slice_ids=(
                        merge_target.member_slice_ids | members
                    ),
                    atoms=tuple(
                        replace(
                            atom,
                            st_time=start,
                            ed_time=end,
                        )
                        for atom in merge_target.atoms
                    )
                    + tuple(atoms),
                    st_time=start,
                    ed_time=end,
                    predecessor_ids=predecessor_ids,
                )
                remaining_by_id[transfer_id] = transfer
                semantic[transfer_id] = (
                    semantic[transfer_id] - removed_ids
                ) | predecessor_ids
                merged_transfer_ids.add(transfer_id)
            else:
                transfer_id = "repair-{}-{}-e{:04d}".format(
                    overlay.overlay_id,
                    flow.flow_id,
                    edge_index,
                )
                if previous_id is None:
                    predecessor_ids = frozenset(
                        item
                        for item in first_old.predecessor_ids
                        if item not in removed_ids
                    )
                transfer = Transfer(
                    transfer_id,
                    flow.operator,
                    src_rank,
                    dst_rank,
                    channel,
                    flow.stage_id,
                    members,
                    tuple(atoms),
                    start,
                    end,
                    predecessor_ids,
                )
                new_transfers.append(transfer)
            if edge_index == 0 and merge_target is None:
                semantic_values = set(
                    raw_semantic.get(
                        first_old.transfer_id,
                        first_old.predecessor_ids,
                    )
                )
                semantic_values.difference_update(removed_ids)
                semantic_values.update(predecessor_ids)
                semantic[transfer_id] = frozenset(semantic_values)
            elif merge_target is None:
                semantic[transfer_id] = predecessor_ids
            if merge_target is None:
                resource_slots[transfer_id] = {
                    resource_id: 0
                    for resource_id in topology.resources_for(key)
                }
            lane_intervals.setdefault(lane, []).append((start, end))
            previous_id = transfer_id
            ready_time = end
        if not new_transfers:
            raise SemanticError("repair candidate contains no replacement edge")
        remaining = [
            remaining_by_id[transfer.transfer_id]
            for transfer in remaining
        ]
        terminal_id = previous_id
        if terminal_id is None:
            raise SemanticError("repair candidate has no terminal transfer")
        full_symbols = _stage_symbols(full_ranks, tuple(ready_values))

        updated_remaining = []
        for transfer in remaining:
            predecessor_ids = frozenset(
                _replace_dependency_values(
                    transfer.predecessor_ids,
                    removed_ids,
                    terminal_id,
                )
            )
            semantic[transfer.transfer_id] = frozenset(
                _replace_dependency_values(
                    semantic[transfer.transfer_id],
                    removed_ids,
                    terminal_id,
                )
            )
            modified = (
                transfer.stage_id > flow.stage_id
                and any(
                    atom.slice_id in members
                    and any(stage.stage_id == flow.stage_id for stage in atom.path)
                    for atom in transfer.atoms
                )
            )
            start = max(transfer.st_time, ready_time) if modified else transfer.st_time
            end = start + (transfer.ed_time - transfer.st_time)
            atoms = tuple(
                (
                    _replace_later_stage_path(
                        atom,
                        flow.stage_id,
                        full_symbols,
                        ready_time,
                        start,
                        end,
                    )
                    if modified and atom.slice_id in members
                    else replace(atom, st_time=start, ed_time=end)
                )
                for atom in transfer.atoms
            )
            updated_remaining.append(
                replace(
                    transfer,
                    atoms=atoms,
                    st_time=start,
                    ed_time=end,
                    predecessor_ids=predecessor_ids,
                )
            )

        transfers = tuple(updated_remaining + new_transfers)
        transfers = _reschedule(transfers, semantic, resource_slots)
        transfer_ids = {transfer.transfer_id for transfer in transfers}
        semantic = {
            transfer_id: frozenset(values)
            for transfer_id, values in semantic.items()
            if transfer_id in transfer_ids
        }
        metadata["semantic_predecessors"] = {
            transfer_id: tuple(sorted(values))
            for transfer_id, values in semantic.items()
        }
        metadata["resource_slots"] = {
            transfer_id: dict(resource_slots.get(transfer_id, {}))
            for transfer_id in transfer_ids
        }
        for field in ("semantic_contributors", "tree_contributors"):
            if field not in metadata:
                continue
            raw_values = metadata[field]
            if not isinstance(raw_values, Mapping):
                raise SemanticError("{} must be a mapping".format(field))
            values = {
                transfer_id: value
                for transfer_id, value in raw_values.items()
                if transfer_id in transfer_ids
            }
            template = raw_values.get(first_old.transfer_id)
            for transfer in new_transfers:
                values[transfer.transfer_id] = (
                    tuple(sorted(transfer.member_slice_ids))
                    if field == "semantic_contributors"
                    else template
                )
            if field == "semantic_contributors":
                repaired_by_id = {
                    transfer.transfer_id: transfer
                    for transfer in transfers
                }
                for transfer_id in merged_transfer_ids:
                    values[transfer_id] = tuple(
                        sorted(
                            repaired_by_id[transfer_id].member_slice_ids
                        )
                    )
            metadata[field] = values
        if "final_dependencies" in metadata:
            metadata["final_dependencies"] = {
                key: _replace_dependency_values(
                    values,
                    removed_ids,
                    terminal_id,
                )
                for key, values in metadata["final_dependencies"].items()
            }
        if metadata.get("path_scope") == "stage_suffix":
            roots = dict(metadata.get("path_roots", {}))
            root_value = roots.get(first_old.transfer_id, flow.root_rank)
            for transfer_id in removed_ids:
                roots.pop(transfer_id, None)
            for transfer in new_transfers:
                roots[transfer.transfer_id] = root_value
            metadata["path_roots"] = roots
        repaired = Schedule(
            schedule_id="{}-{}".format(schedule.schedule_id, overlay.overlay_id),
            transfers=transfers,
            final_state_ids=schedule.final_state_ids,
            rank_count=schedule.rank_count,
            slice_count=schedule.slice_count,
            slice_size_bytes=schedule.slice_size_bytes,
            metadata=metadata,
        )
        constraint_result = verify_schedule_constraints(
            repaired,
            inputs,
            topology,
        )
        if constraint_result.status is not ValidationStatus.VALID:
            return _failure(
                RepairStatus.INVALID,
                constraint_result.code,
                constraint_result.message,
                constraint_evidence=dict(constraint_result.evidence),
            )
        semantic_status = "not_run"
        if "final_outputs" in repaired.metadata:
            semantic_result = verify_schedule_semantics(repaired, inputs)
            semantic_status = semantic_result.status.value
            if semantic_result.status is not ValidationStatus.VALID:
                return _failure(
                    RepairStatus.INVALID,
                    semantic_result.code,
                    semantic_result.message,
                    semantic_evidence=dict(semantic_result.evidence),
                )
        parent_by_id = {
            transfer.transfer_id: transfer for transfer in schedule.transfers
        }
        changed = set(removed_ids) | {
            transfer.transfer_id
            for transfer in repaired.transfers
            if parent_by_id.get(transfer.transfer_id) != transfer
        }
        return RepairResult(
            status=RepairStatus.SUCCESS,
            schedule=repaired,
            changed_transfer_ids=frozenset(changed),
            selected_candidate_flow_id=selected.candidate_id,
            method="greedy",
            evidence={
                "cost": selected.cost,
                "leaf_repair_hops": selected.leaf_repair_hops,
                "removed_transfer_ids": tuple(sorted(removed_ids)),
                "new_transfer_ids": tuple(
                    transfer.transfer_id for transfer in new_transfers
                ),
                "merged_transfer_ids": tuple(
                    sorted(merged_transfer_ids)
                ),
                "semantic_status": semantic_status,
            },
        )
    except SemanticError as error:
        return _failure(
            RepairStatus.INVALID,
            "repair_invalid",
            str(error),
        )
