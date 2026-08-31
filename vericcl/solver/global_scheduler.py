import math
from dataclasses import dataclass
from itertools import product
from typing import Mapping, Tuple

from vericcl.composer.timing import _retime, _semantic_predecessors
from vericcl.errors import SemanticError
from vericcl.semantics.atom import Atom, PathStage, Schedule, Symbol, Transfer
from vericcl.solver.scheduling import fixed_topology_transfer_duration_us
from vericcl.topology.model import LaneKey, LinkKey, Topology


GLOBAL_SCHEDULER_VERSION = "1"


class GlobalSchedulingError(SemanticError):
    """Raised when a complete deterministic global schedule cannot be built."""


@dataclass(frozen=True)
class _Assignment:
    channel: int
    resource_slots: Tuple[Tuple[str, int], ...]
    st_time: float
    ed_time: float
    predecessor_ids: frozenset


def _positive_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise GlobalSchedulingError("{} must be a positive integer".format(field))
    return value


def _route_priorities(schedule: Schedule) -> Mapping[str, int]:
    raw = schedule.metadata.get("route_priorities", {})
    if not isinstance(raw, Mapping):
        raise GlobalSchedulingError("route_priorities must be a mapping")
    transfer_ids = {transfer.transfer_id for transfer in schedule.transfers}
    if not set(raw) <= transfer_ids:
        raise GlobalSchedulingError(
            "route priority references a missing transfer"
        )
    result = {}
    for transfer in schedule.transfers:
        value = raw.get(transfer.transfer_id, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise GlobalSchedulingError(
                "route priority must be a non-negative integer"
            )
        result[transfer.transfer_id] = value
    return result


def _logical_position(transfer: Transfer, slice_count: int) -> int:
    positions = {
        slice_id % slice_count for slice_id in transfer.member_slice_ids
    }
    if len(positions) != 1:
        raise GlobalSchedulingError(
            "transfer members must share one logical position"
        )
    return next(iter(positions))


def _resource_slot_options(
    topology: Topology,
    edge,
    channel_count: int,
) -> Tuple[Tuple[Tuple[str, int], ...], ...]:
    resource_ids = tuple(sorted(edge.resource_ids))
    ranges = []
    for resource_id in resource_ids:
        resource = topology.shared_resources.get(resource_id)
        if resource is None:
            raise GlobalSchedulingError(
                "topology link references a missing shared resource"
            )
        slot_count = min(channel_count, resource.max_channels)
        if slot_count < 1:
            raise GlobalSchedulingError(
                "no shared resource slot is available"
            )
        ranges.append(range(slot_count))
    if not ranges:
        return ((),)
    return tuple(
        tuple(zip(resource_ids, slots))
        for slots in product(*ranges)
    )


def _zero_ready_path(atom: Atom) -> Tuple[PathStage, ...]:
    return tuple(
        PathStage(
            stage.stage_id,
            stage.operator,
            tuple(
                Symbol(symbol.src_rank, symbol.dst_rank, 0.0)
                for symbol in stage.symbols
            ),
        )
        for stage in atom.path
    )


def _assigned_transfer(
    transfer: Transfer,
    assignment: _Assignment,
) -> Transfer:
    atoms = tuple(
        Atom(
            slice_id=atom.slice_id,
            slice_size_bytes=atom.slice_size_bytes,
            path=_zero_ready_path(atom),
            st_time=assignment.st_time,
            ed_time=assignment.ed_time,
        )
        for atom in transfer.atoms
    )
    return Transfer(
        transfer_id=transfer.transfer_id,
        kind=transfer.kind,
        src_rank=transfer.src_rank,
        dst_rank=transfer.dst_rank,
        channel=assignment.channel,
        stage_id=transfer.stage_id,
        member_slice_ids=transfer.member_slice_ids,
        atoms=atoms,
        st_time=assignment.st_time,
        ed_time=assignment.ed_time,
        predecessor_ids=assignment.predecessor_ids,
    )


def _validate_inputs(
    schedule: Schedule,
    topology: Topology,
    channel_count: int,
) -> int:
    if not isinstance(schedule, Schedule):
        raise GlobalSchedulingError("schedule must be a Schedule")
    if not isinstance(topology, Topology):
        raise GlobalSchedulingError("topology must be a Topology")
    channels = _positive_integer(channel_count, "channel_count")
    if schedule.rank_count != topology.rank_count:
        raise GlobalSchedulingError(
            "schedule and topology rank counts must agree"
        )
    if "semantic_predecessors" not in schedule.metadata:
        raise GlobalSchedulingError(
            "semantic_predecessors metadata is required"
        )
    return channels


def assign_global_resources(
    schedule: Schedule,
    topology: Topology,
    channel_count: int,
) -> Schedule:
    """Assign fixed-K lanes, shared-resource slots, and global times."""
    channels = _validate_inputs(schedule, topology, channel_count)
    semantic = _semantic_predecessors(schedule)
    priorities = _route_priorities(schedule)
    by_id = {
        transfer.transfer_id: transfer for transfer in schedule.transfers
    }
    links = {}
    durations = {}
    slot_options = {}
    logical_positions = {}
    for transfer_id in sorted(by_id):
        transfer = by_id[transfer_id]
        link = LinkKey(transfer.src_rank, transfer.dst_rank)
        edge = topology.links.get(link)
        if edge is None:
            raise GlobalSchedulingError(
                "schedule transfer is absent from the topology"
            )
        lane_count = min(channels, edge.max_channels)
        if lane_count < 1:
            raise GlobalSchedulingError("no directed-link channel is available")
        links[transfer_id] = link
        durations[transfer_id] = fixed_topology_transfer_duration_us(
            topology,
            link,
            transfer.physical_bytes,
            channels,
        )
        slot_options[transfer_id] = _resource_slot_options(
            topology,
            edge,
            channels,
        )
        logical_positions[transfer_id] = _logical_position(
            transfer,
            schedule.slice_count,
        )

    pending = set(by_id)
    assignments = {}
    lane_ready = {}
    lane_last = {}
    resource_ready = {}
    resource_last = {}
    while pending:
        ready = tuple(
            transfer_id
            for transfer_id in sorted(pending)
            if semantic[transfer_id] <= set(assignments)
        )
        if not ready:
            raise GlobalSchedulingError(
                "schedule semantic dependencies contain a cycle"
            )
        choices = []
        for transfer_id in ready:
            transfer = by_id[transfer_id]
            semantic_ready = max(
                (
                    assignments[predecessor].ed_time
                    for predecessor in semantic[transfer_id]
                ),
                default=0.0,
            )
            edge = topology.link(links[transfer_id])
            for channel in range(min(channels, edge.max_channels)):
                lane = LaneKey(
                    transfer.src_rank,
                    transfer.dst_rank,
                    channel,
                )
                for slots in slot_options[transfer_id]:
                    start = max(
                        [semantic_ready, lane_ready.get(lane, 0.0)]
                        + [
                            resource_ready.get(item, 0.0)
                            for item in slots
                        ]
                    )
                    end = start + durations[transfer_id]
                    key = (
                        end,
                        semantic_ready,
                        priorities[transfer_id],
                        transfer.stage_id,
                        logical_positions[transfer_id],
                        transfer.src_rank,
                        transfer.dst_rank,
                        transfer_id,
                        channel,
                        slots,
                    )
                    choices.append((key, lane, slots, start, end))
        if not choices:
            raise GlobalSchedulingError(
                "no legal global resource assignment is available"
            )
        (
            key,
            lane,
            slots,
            start,
            end,
        ) = min(choices, key=lambda item: item[0])
        transfer_id = key[7]
        predecessors = set(semantic[transfer_id])
        if lane in lane_last:
            predecessors.add(lane_last[lane])
        for item in slots:
            if item in resource_last:
                predecessors.add(resource_last[item])
        assignments[transfer_id] = _Assignment(
            channel=key[8],
            resource_slots=slots,
            st_time=start,
            ed_time=end,
            predecessor_ids=frozenset(predecessors),
        )
        pending.remove(transfer_id)
        lane_ready[lane] = end
        lane_last[lane] = transfer_id
        for item in slots:
            resource_ready[item] = end
            resource_last[item] = transfer_id
    if set(assignments) != set(by_id):
        raise GlobalSchedulingError("global schedule is incomplete")

    metadata = dict(schedule.metadata)
    metadata.pop("routing_only", None)
    if "final_dependencies" not in metadata:
        metadata.pop("final_ready_times", None)
    metadata.update(
        {
            "channel_count": channels,
            "global_resources_assigned": True,
            "global_scheduler_version": GLOBAL_SCHEDULER_VERSION,
            "semantic_predecessors": {
                transfer_id: tuple(sorted(semantic[transfer_id]))
                for transfer_id in sorted(semantic)
            },
            "resource_slots": {
                transfer_id: dict(assignments[transfer_id].resource_slots)
                for transfer_id in sorted(assignments)
            },
        }
    )
    if "route_priorities" in metadata:
        metadata["route_priorities"] = {
            transfer_id: priorities[transfer_id]
            for transfer_id in sorted(priorities)
        }
    provisional = Schedule(
        schedule_id=schedule.schedule_id,
        transfers=tuple(
            _assigned_transfer(by_id[transfer_id], assignments[transfer_id])
            for transfer_id in sorted(by_id)
        ),
        final_state_ids=schedule.final_state_ids,
        rank_count=schedule.rank_count,
        slice_count=schedule.slice_count,
        slice_size_bytes=schedule.slice_size_bytes,
        metadata=metadata,
    )
    result = _retime(provisional, topology)
    result_by_id = {
        transfer.transfer_id: transfer for transfer in result.transfers
    }
    for transfer_id, assignment in assignments.items():
        transfer = result_by_id.get(transfer_id)
        if transfer is None:
            raise GlobalSchedulingError("global retiming omitted a transfer")
        if (
            transfer.channel != assignment.channel
            or not math.isclose(
                transfer.st_time,
                assignment.st_time,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                transfer.ed_time,
                assignment.ed_time,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or transfer.predecessor_ids != assignment.predecessor_ids
        ):
            raise GlobalSchedulingError(
                "global retiming changed the resource assignment"
            )
    return result


__all__ = [
    "GLOBAL_SCHEDULER_VERSION",
    "GlobalSchedulingError",
    "assign_global_resources",
]
