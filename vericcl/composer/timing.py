from typing import Mapping, Optional

from vericcl.errors import SemanticError
from vericcl.semantics.atom import Atom, PathStage, Schedule, Symbol, Transfer
from vericcl.topology.model import LaneKey, LinkKey, Topology


def _semantic_predecessors(schedule: Schedule) -> Mapping[str, frozenset]:
    raw = schedule.metadata.get("semantic_predecessors")
    if raw is None:
        return {
            transfer.transfer_id: transfer.predecessor_ids
            for transfer in schedule.transfers
        }
    if not isinstance(raw, Mapping):
        raise SemanticError("semantic_predecessors metadata must be a mapping")
    if set(raw) != {transfer.transfer_id for transfer in schedule.transfers}:
        raise SemanticError(
            "semantic_predecessors must cover every transfer exactly"
        )
    result = {key: frozenset(value) for key, value in raw.items()}
    transfer_ids = set(result)
    if any(not values <= transfer_ids for values in result.values()):
        raise SemanticError("semantic predecessor is missing from the schedule")
    return result


def _resource_slots(schedule: Schedule) -> Mapping[str, Mapping[str, int]]:
    raw = schedule.metadata.get("resource_slots", {})
    if not isinstance(raw, Mapping):
        raise SemanticError("resource_slots metadata must be a mapping")
    result = {}
    for transfer in schedule.transfers:
        values = raw.get(transfer.transfer_id, {})
        if not isinstance(values, Mapping):
            raise SemanticError("transfer resource slots must be a mapping")
        result[transfer.transfer_id] = dict(values)
    return result


def _validate_topology(
    transfer: Transfer,
    slots: Mapping[str, int],
    topology: Topology,
) -> None:
    link = LinkKey(transfer.src_rank, transfer.dst_rank)
    if link not in topology.links:
        raise SemanticError("schedule transfer is absent from the topology")
    edge = topology.link(link)
    if transfer.channel >= edge.max_channels:
        raise SemanticError("schedule channel exceeds the topology limit")
    if not set(slots) <= set(edge.resource_ids):
        raise SemanticError("schedule uses an unrelated shared resource")
    for resource_id, slot in slots.items():
        resource = topology.shared_resources[resource_id]
        if slot < 0 or slot >= resource.max_channels:
            raise SemanticError("schedule resource slot exceeds its limit")


def _retime(
    schedule: Schedule,
    topology: Optional[Topology],
) -> Schedule:
    if not isinstance(schedule, Schedule):
        raise SemanticError("schedule must be a Schedule")
    if topology is not None:
        if not isinstance(topology, Topology):
            raise SemanticError("topology must be a Topology")
        if topology.rank_count != schedule.rank_count:
            raise SemanticError("schedule and topology rank counts must agree")
    semantic = _semantic_predecessors(schedule)
    resource_slots = _resource_slots(schedule)
    by_id = {transfer.transfer_id: transfer for transfer in schedule.transfers}
    if topology is not None:
        for transfer in schedule.transfers:
            _validate_topology(
                transfer,
                resource_slots[transfer.transfer_id],
                topology,
            )
    pending = dict(by_id)
    times = {}
    all_predecessors = {}
    lane_ready = {}
    lane_last = {}
    resource_ready = {}
    resource_last = {}
    while pending:
        ready = [
            transfer
            for transfer in pending.values()
            if semantic[transfer.transfer_id] <= set(times)
        ]
        if not ready:
            raise SemanticError("schedule semantic dependencies contain a cycle")
        choices = []
        for transfer in ready:
            transfer_id = transfer.transfer_id
            semantic_ready = max(
                (
                    times[predecessor][1]
                    for predecessor in semantic[transfer_id]
                ),
                default=0.0,
            )
            lane = LaneKey(
                transfer.src_rank,
                transfer.dst_rank,
                transfer.channel,
            )
            start = max(
                [semantic_ready, lane_ready.get(lane, 0.0)]
                + [
                    resource_ready.get((resource_id, slot), 0.0)
                    for resource_id, slot in resource_slots[
                        transfer_id
                    ].items()
                ]
            )
            choices.append(
                (
                    start,
                    transfer.st_time,
                    transfer_id,
                    semantic_ready,
                    lane,
                    transfer,
                )
            )
        start, _, transfer_id, semantic_ready, lane, transfer = min(choices)
        duration = transfer.ed_time - transfer.st_time
        end = start + duration
        predecessors = set(semantic[transfer_id])
        if lane in lane_last:
            predecessors.add(lane_last[lane])
        for resource_id, slot in resource_slots[transfer_id].items():
            key = (resource_id, slot)
            if key in resource_last:
                predecessors.add(resource_last[key])
        times[transfer_id] = (start, end, semantic_ready)
        all_predecessors[transfer_id] = frozenset(predecessors)
        del pending[transfer_id]
        lane_ready[lane] = end
        lane_last[lane] = transfer_id
        for resource_id, slot in resource_slots[transfer_id].items():
            key = (resource_id, slot)
            resource_ready[key] = end
            resource_last[key] = transfer_id
    operation_ids = {}
    for transfer in schedule.transfers:
        for member in transfer.member_slice_ids:
            key = (
                transfer.stage_id,
                transfer.src_rank,
                transfer.dst_rank,
                member,
            )
            if key in operation_ids and operation_ids[key] != transfer.transfer_id:
                raise SemanticError("slice operation is duplicated in the schedule")
            operation_ids[key] = transfer.transfer_id
    rebuilt = []
    for transfer in schedule.transfers:
        start, end, _ = times[transfer.transfer_id]
        atoms = []
        for atom in transfer.atoms:
            path = []
            for stage in atom.path:
                symbols = []
                for symbol in stage.symbols:
                    key = (
                        stage.stage_id,
                        symbol.src_rank,
                        symbol.dst_rank,
                        atom.slice_id,
                    )
                    operation_id = operation_ids.get(key)
                    if operation_id is None:
                        raise SemanticError(
                            "atom path operation is missing from the schedule"
                        )
                    symbols.append(
                        Symbol(
                            symbol.src_rank,
                            symbol.dst_rank,
                            times[operation_id][2],
                        )
                    )
                path.append(PathStage(stage.stage_id, stage.operator, tuple(symbols)))
            atoms.append(
                Atom(
                    slice_id=atom.slice_id,
                    slice_size_bytes=atom.slice_size_bytes,
                    path=tuple(path),
                    st_time=start,
                    ed_time=end,
                )
            )
        rebuilt.append(
            Transfer(
                transfer_id=transfer.transfer_id,
                kind=transfer.kind,
                src_rank=transfer.src_rank,
                dst_rank=transfer.dst_rank,
                channel=transfer.channel,
                stage_id=transfer.stage_id,
                member_slice_ids=transfer.member_slice_ids,
                atoms=tuple(atoms),
                st_time=start,
                ed_time=end,
                predecessor_ids=all_predecessors[transfer.transfer_id],
            )
        )
    metadata = dict(schedule.metadata)
    metadata["semantic_predecessors"] = {
        key: tuple(sorted(values)) for key, values in semantic.items()
    }
    metadata["resource_slots"] = {
        key: dict(values) for key, values in resource_slots.items()
    }
    metadata["timing_recomputed"] = True
    return Schedule(
        schedule_id=schedule.schedule_id,
        transfers=tuple(sorted(rebuilt, key=lambda item: item.transfer_id)),
        final_state_ids=schedule.final_state_ids,
        rank_count=schedule.rank_count,
        slice_count=schedule.slice_count,
        slice_size_bytes=schedule.slice_size_bytes,
        metadata=metadata,
    )


def recompute_earliest_times(
    schedule: Schedule,
    topology: Topology,
) -> Schedule:
    return _retime(schedule, topology)
