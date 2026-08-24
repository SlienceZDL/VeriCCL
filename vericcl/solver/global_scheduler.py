from dataclasses import dataclass
from itertools import product
from typing import FrozenSet, Mapping

from vericcl.errors import SemanticError
from vericcl.semantics.atom import Schedule
from vericcl.solver.scheduling import (
    conservative_transfer_duration_us,
    rebuild_scheduled_transfers,
)
from vericcl.topology.model import LaneKey, LinkKey, Topology


@dataclass(frozen=True)
class _Assignment:
    st_time: float
    ed_time: float
    semantic_ready_time: float
    channel: int
    resource_slots: Mapping[str, int]
    predecessor_ids: FrozenSet[str]


def _channel_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SemanticError("channel_count must be a positive integer")
    return value


def _semantic_predecessors(
    schedule: Schedule,
) -> Mapping[str, FrozenSet[str]]:
    transfers = {
        transfer.transfer_id: transfer for transfer in schedule.transfers
    }
    raw = schedule.metadata.get("semantic_predecessors")
    if raw is None:
        return {
            transfer_id: transfer.predecessor_ids
            for transfer_id, transfer in transfers.items()
        }
    if not isinstance(raw, Mapping):
        raise SemanticError("semantic_predecessors metadata must be a mapping")
    if set(raw) != set(transfers):
        raise SemanticError(
            "semantic_predecessors must cover every transfer exactly"
        )
    result = {
        transfer_id: frozenset(values)
        for transfer_id, values in raw.items()
    }
    if any(
        not predecessors <= set(transfers)
        for predecessors in result.values()
    ):
        raise SemanticError("semantic predecessor is missing from the schedule")
    return result


def _resource_options(
    topology: Topology,
    link: LinkKey,
    channel_count: int,
) -> tuple[tuple[tuple[str, int], ...], ...]:
    resource_ids = topology.link(link).resource_ids
    if not resource_ids:
        return ((),)
    slots = tuple(
        tuple(
            range(
                min(
                    channel_count,
                    topology.shared_resources[resource_id].max_channels,
                )
            )
        )
        for resource_id in resource_ids
    )
    if any(not values for values in slots):
        raise SemanticError("shared resource has no available capacity")
    return tuple(
        tuple(zip(resource_ids, values)) for values in product(*slots)
    )


def _rebuild(
    schedule: Schedule,
    assignments: Mapping[str, _Assignment],
    semantic: Mapping[str, FrozenSet[str]],
    channel_count: int,
) -> Schedule:
    return rebuild_scheduled_transfers(
        schedule,
        timings={
            transfer_id: (
                assignment.st_time,
                assignment.ed_time,
                assignment.semantic_ready_time,
            )
            for transfer_id, assignment in assignments.items()
        },
        channels={
            transfer_id: assignment.channel
            for transfer_id, assignment in assignments.items()
        },
        semantic_predecessors=semantic,
        predecessor_ids={
            transfer_id: assignment.predecessor_ids
            for transfer_id, assignment in assignments.items()
        },
        resource_slots={
            transfer_id: assignment.resource_slots
            for transfer_id, assignment in assignments.items()
        },
        metadata_updates={
            "routing_only": False,
            "channel_count": channel_count,
            "global_resources_assigned": True,
        },
    )


def assign_global_resources(
    schedule: Schedule,
    topology: Topology,
    channel_count: int,
) -> Schedule:
    if not isinstance(schedule, Schedule):
        raise SemanticError("schedule must be a Schedule")
    if not isinstance(topology, Topology):
        raise SemanticError("topology must be a Topology")
    if schedule.rank_count != topology.rank_count:
        raise SemanticError("schedule and topology rank counts must agree")
    channels = _channel_count(channel_count)
    semantic = _semantic_predecessors(schedule)
    transfers = {
        transfer.transfer_id: transfer for transfer in schedule.transfers
    }
    for transfer in schedule.transfers:
        if LinkKey(transfer.src_rank, transfer.dst_rank) not in topology.links:
            raise SemanticError("schedule transfer is absent from the topology")
    links = {
        LinkKey(transfer.src_rank, transfer.dst_rank)
        for transfer in schedule.transfers
    }
    durations = {
        link: conservative_transfer_duration_us(
            topology,
            link,
            schedule.slice_size_bytes,
            channels,
        )
        for link in links
    }
    resource_options = {
        link: _resource_options(topology, link, channels)
        for link in links
    }
    pending = dict(transfers)
    assignments = {}
    lane_ready = {}
    lane_last = {}
    resource_ready = {}
    resource_last = {}
    while pending:
        completed = set(assignments)
        ready = tuple(
            transfer
            for transfer_id, transfer in sorted(pending.items())
            if semantic[transfer_id] <= completed
        )
        if not ready:
            raise SemanticError("schedule semantic dependencies contain a cycle")
        best_choice = None
        for transfer in ready:
            transfer_id = transfer.transfer_id
            link = LinkKey(transfer.src_rank, transfer.dst_rank)
            edge = topology.link(link)
            semantic_ready = max(
                (
                    assignments[predecessor].ed_time
                    for predecessor in semantic[transfer_id]
                ),
                default=0.0,
            )
            for channel in range(min(channels, edge.max_channels)):
                lane = LaneKey(link.src_rank, link.dst_rank, channel)
                for resource_tuple in resource_options[link]:
                    slots = dict(resource_tuple)
                    start = max(
                        [semantic_ready, lane_ready.get(lane, 0.0)]
                        + [
                            resource_ready.get((resource_id, slot), 0.0)
                            for resource_id, slot in resource_tuple
                        ]
                    )
                    end = start + durations[link]
                    topology_key = (
                        link.src_rank,
                        link.dst_rank,
                        channel,
                        resource_tuple,
                    )
                    candidate = (
                        (end, start, topology_key, transfer_id),
                        transfer,
                        semantic_ready,
                        lane,
                        slots,
                    )
                    if best_choice is None or candidate[0] < best_choice[0]:
                        best_choice = candidate
        if best_choice is None:
            raise SemanticError(
                "transfer has no available channel or resource capacity"
            )
        choice, transfer, semantic_ready, lane, slots = best_choice
        end, start, _, transfer_id = choice
        predecessors = set(semantic[transfer_id])
        if lane in lane_last:
            predecessors.add(lane_last[lane])
        for resource_id, slot in slots.items():
            resource_key = (resource_id, slot)
            if resource_key in resource_last:
                predecessors.add(resource_last[resource_key])
        assignment = _Assignment(
            st_time=start,
            ed_time=end,
            semantic_ready_time=semantic_ready,
            channel=lane.channel,
            resource_slots=slots,
            predecessor_ids=frozenset(predecessors),
        )
        assignments[transfer_id] = assignment
        del pending[transfer_id]
        lane_ready[lane] = end
        lane_last[lane] = transfer_id
        for resource_id, slot in slots.items():
            resource_key = (resource_id, slot)
            resource_ready[resource_key] = end
            resource_last[resource_key] = transfer_id
    return _rebuild(schedule, assignments, semantic, channels)
