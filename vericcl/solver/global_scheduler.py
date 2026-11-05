import heapq
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Mapping, Tuple

from vericcl.composer.timing import _retime, _semantic_predecessors
from vericcl.errors import SemanticError
from vericcl.semantics.atom import Atom, PathStage, Schedule, Symbol, Transfer
from vericcl.solver.scheduling import fixed_topology_transfer_duration_us
from vericcl.topology.model import LaneKey, LinkKey, Topology


GLOBAL_SCHEDULER_VERSION = "1"
MAX_GLOBAL_CHANNEL_COUNT = 32


class GlobalSchedulingError(SemanticError):
    """Raised when a complete deterministic global schedule cannot be built."""


@dataclass(frozen=True)
class _Assignment:
    channel: int
    resource_slots: Tuple[Tuple[str, int], ...]
    st_time: float
    ed_time: float
    predecessor_ids: frozenset


@dataclass
class _ReadyGroup:
    """Transfers sharing one lane, resource domain, and duration."""
    lane: LaneKey
    slot_counts: tuple
    duration: float
    hierarchy: bool
    releases: list = field(default_factory=list)
    eligible: list = field(default_factory=list)
    version: int = 0

    def candidate(self, assignments, lane_ready, resource_ready):
        frontier, _ = _earliest_resource_slots(
            self.slot_counts, resource_ready, lane_ready.get(self.lane, 0.0))
        while self.releases and self.releases[0][0] <= frontier:
            _, tail, transfer_id = heapq.heappop(self.releases)
            if transfer_id not in assignments:
                heapq.heappush(self.eligible, (tail, transfer_id))
        while self.eligible and self.eligible[0][1] in assignments:
            heapq.heappop(self.eligible)
        if self.eligible:
            tail, transfer_id = self.eligible[0]
            start = frontier
        else:
            while self.releases and self.releases[0][2] in assignments:
                heapq.heappop(self.releases)
            if not self.releases:
                return None
            start, tail, transfer_id = self.releases[0]
        end = start + self.duration
        # transfer_id and channel already uniquely identify a choice, so the
        # resource-slot tie-breaker cannot distinguish two queue entries.
        key = ((start if self.hierarchy else end),) + tail + (self.lane.channel,)
        return key, start, end, transfer_id


def _validate_channel_count(value: object) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
        or value > MAX_GLOBAL_CHANNEL_COUNT
    ):
        raise GlobalSchedulingError(
            "channel_count must be an integer between 1 and {}".format(
                MAX_GLOBAL_CHANNEL_COUNT
            )
        )
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


def _resource_slot_counts(
    topology: Topology,
    edge,
    channel_count: int,
) -> Tuple[Tuple[str, int], ...]:
    counts = []
    for resource_id in sorted(edge.resource_ids):
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
        counts.append((resource_id, slot_count))
    return tuple(counts)


def _earliest_resource_slots(
    slot_counts: Tuple[Tuple[str, int], ...],
    resource_ready: Mapping[Tuple[str, int], float],
    base_ready: float,
) -> Tuple[float, Tuple[Tuple[str, int], ...]]:
    start = base_ready
    for resource_id, slot_count in slot_counts:
        if slot_count < 1:
            raise GlobalSchedulingError(
                "no shared resource slot is available"
            )
        start = max(
            start,
            min(
                resource_ready.get((resource_id, slot), 0.0)
                for slot in range(slot_count)
            ),
        )
    slots = []
    for resource_id, slot_count in slot_counts:
        for slot in range(slot_count):
            if resource_ready.get((resource_id, slot), 0.0) <= start:
                slots.append((resource_id, slot))
                break
        else:
            raise GlobalSchedulingError(
                "no shared resource slot is available"
            )
    return start, tuple(slots)


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
    channels = _validate_channel_count(channel_count)
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
    *, lease=None,
) -> Schedule:
    """Assign fixed-K lanes, shared-resource slots, and global times."""
    def check_budget():
        if lease is not None:
            lease.check()

    check_budget()
    channels = _validate_inputs(schedule, topology, channel_count)
    try:
        semantic = _semantic_predecessors(schedule)
    except SemanticError as error:
        raise GlobalSchedulingError(str(error)) from error
    priorities = _route_priorities(schedule)
    by_id = {
        transfer.transfer_id: transfer for transfer in schedule.transfers
    }
    links = {}
    durations = {}
    slot_counts = {}
    logical_positions = {}
    for index, transfer_id in enumerate(sorted(by_id)):
        if index % 128 == 0:
            check_budget()
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
        slot_counts[transfer_id] = _resource_slot_counts(
            topology,
            edge,
            channels,
        )
        logical_positions[transfer_id] = _logical_position(
            transfer,
            schedule.slice_count,
        )

    successors = {key: [] for key in by_id}
    degrees = {key: len(values) for key, values in semantic.items()}
    for key, dependencies in semantic.items():
        for dependency in dependencies:
            successors[dependency].append(key)
    hierarchy = bool(schedule.metadata.get('hierarchy_event_policy'))
    remaining_path = {}
    if hierarchy:
        unresolved = {key: len(values) for key, values in successors.items()}
        sinks = deque(key for key, degree in unresolved.items() if degree == 0)
        while sinks:
            check_budget()
            key = sinks.popleft()
            remaining_path[key] = durations[key] + max(
                (remaining_path[s] for s in successors[key]), default=0.)
            for predecessor in semantic[key]:
                unresolved[predecessor] -= 1
                if unresolved[predecessor] == 0:
                    sinks.append(predecessor)
        if len(remaining_path) != len(by_id):
            raise GlobalSchedulingError('semantic dependencies contain a cycle')

    assignments = {}
    lane_ready = {}
    lane_last = {}
    resource_ready = {}
    resource_last = {}
    groups, group_ids, queue = [], {}, []

    def add_ready(transfer_id):
        transfer = by_id[transfer_id]
        release = max((assignments[p].ed_time for p in semantic[transfer_id]), default=0.)
        if hierarchy:
            tail = (priorities[transfer_id], -remaining_path[transfer_id], release,
                    transfer.stage_id, logical_positions[transfer_id], transfer_id)
        else:
            tail = (release, priorities[transfer_id], transfer.stage_id,
                    logical_positions[transfer_id], transfer.src_rank,
                    transfer.dst_rank, transfer_id)
        edge = topology.link(links[transfer_id])
        changed = set()
        for channel in range(min(channels, edge.max_channels)):
            lane = LaneKey(transfer.src_rank, transfer.dst_rank, channel)
            signature = (lane, slot_counts[transfer_id], durations[transfer_id])
            if signature not in group_ids:
                group_ids[signature] = len(groups)
                groups.append(_ReadyGroup(lane, slot_counts[transfer_id],
                                          durations[transfer_id], hierarchy))
            index = group_ids[signature]
            heapq.heappush(groups[index].releases, (release, tail, transfer_id))
            changed.add(index)
        return changed

    def enqueue(index):
        group = groups[index]
        group.version += 1
        candidate = group.candidate(assignments, lane_ready, resource_ready)
        if candidate is not None:
            heapq.heappush(queue, (candidate[0], index, group.version))

    changed = set()
    for transfer_id, degree in degrees.items():
        check_budget()
        if degree == 0:
            changed.update(add_ready(transfer_id))
    for index in changed:
        enqueue(index)
    while queue:
        check_budget()
        previous_key, index, version = heapq.heappop(queue)
        group = groups[index]
        if version != group.version:
            continue
        candidate = group.candidate(assignments, lane_ready, resource_ready)
        if candidate is None:
            continue
        key, start, end, transfer_id = candidate
        # Increasing resource frontiers and removing a chosen transfer only
        # increase a group's key. Newly ready transfers may decrease it and
        # are therefore enqueued immediately, invalidating its prior entry.
        if key != previous_key:
            enqueue(index)
            continue
        lane = group.lane
        _, slots = _earliest_resource_slots(group.slot_counts, resource_ready, start)
        predecessors = set(semantic[transfer_id])
        if lane in lane_last:
            predecessors.add(lane_last[lane])
        for item in slots:
            if item in resource_last:
                predecessors.add(resource_last[item])
        assignments[transfer_id] = _Assignment(
            channel=lane.channel,
            resource_slots=slots,
            st_time=start,
            ed_time=end,
            predecessor_ids=frozenset(predecessors),
        )
        lane_ready[lane] = end
        lane_last[lane] = transfer_id
        for item in slots:
            resource_ready[item] = end
            resource_last[item] = transfer_id
        changed = {index}
        for follower in successors[transfer_id]:
            degrees[follower] -= 1
            if degrees[follower] == 0:
                changed.update(add_ready(follower))
        for changed_index in changed:
            enqueue(changed_index)
    if set(assignments) != set(by_id):
        raise GlobalSchedulingError("schedule semantic dependencies contain a cycle")
    check_budget()

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
    result = _retime(
        provisional,
        topology,
        preserve_predecessor_order=True,
        lease=lease,
    )
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
