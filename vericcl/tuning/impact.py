from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, Mapping

from vericcl.errors import SemanticError
from vericcl.semantics.atom import Schedule
from vericcl.topology.model import LaneKey, LinkKey, Topology


@dataclass(frozen=True)
class ImpactClosure:
    seed_transfer_ids: frozenset[str]
    transfer_ids: frozenset[str]
    reasons: Mapping[str, frozenset[str]]

    def __post_init__(self) -> None:
        seeds = frozenset(self.seed_transfer_ids)
        transfers = frozenset(self.transfer_ids)
        reasons = {
            transfer_id: frozenset(values)
            for transfer_id, values in self.reasons.items()
        }
        if not seeds <= transfers or set(reasons) != set(transfers):
            raise SemanticError("impact closure metadata does not match transfers")
        object.__setattr__(self, "seed_transfer_ids", seeds)
        object.__setattr__(self, "transfer_ids", transfers)
        object.__setattr__(self, "reasons", MappingProxyType(reasons))


def _semantic_predecessors(schedule: Schedule) -> Mapping[str, frozenset[str]]:
    raw = schedule.metadata.get("semantic_predecessors", {})
    if not isinstance(raw, Mapping):
        raise SemanticError("semantic_predecessors must be a mapping")
    transfer_ids = {transfer.transfer_id for transfer in schedule.transfers}
    result = {}
    for transfer in schedule.transfers:
        try:
            semantic = frozenset(raw.get(transfer.transfer_id, ()))
        except TypeError as error:
            raise SemanticError("semantic predecessor IDs must be iterable") from error
        values = transfer.predecessor_ids | semantic
        if not values <= transfer_ids:
            raise SemanticError("semantic predecessor is missing")
        result[transfer.transfer_id] = values
    return result


def compute_impact_closure(
    schedule: Schedule,
    changed_transfer_ids: frozenset[str],
    topology: Topology,
) -> ImpactClosure:
    if not isinstance(schedule, Schedule):
        raise SemanticError("schedule must be a Schedule")
    if not isinstance(topology, Topology):
        raise SemanticError("topology must be a Topology")
    if schedule.rank_count != topology.rank_count:
        raise SemanticError("schedule and topology rank counts must agree")
    seeds = frozenset(changed_transfer_ids)
    transfers = {
        transfer.transfer_id: transfer for transfer in schedule.transfers
    }
    unknown = seeds - set(transfers)
    if unknown:
        raise SemanticError("impact closure contains an unknown transfer")
    if not seeds:
        return ImpactClosure(frozenset(), frozenset(), {})

    predecessors = _semantic_predecessors(schedule)
    consumers: Dict[str, set] = {transfer_id: set() for transfer_id in transfers}
    for consumer_id, values in predecessors.items():
        for predecessor_id in values:
            consumers[predecessor_id].add(consumer_id)
    lanes: Dict[LaneKey, list] = {}
    links: Dict[LinkKey, set] = {}
    resource_transfers: Dict[str, set] = {
        resource_id: set() for resource_id in topology.shared_resources
    }
    for transfer in schedule.transfers:
        lane = LaneKey(
            transfer.src_rank,
            transfer.dst_rank,
            transfer.channel,
        )
        lanes.setdefault(lane, []).append(transfer)
        link = LinkKey(transfer.src_rank, transfer.dst_rank)
        links.setdefault(link, set()).add(transfer.transfer_id)
        for resource_id in topology.resources_for(link):
            resource_transfers[resource_id].add(transfer.transfer_id)
    for values in lanes.values():
        values.sort(
            key=lambda item: (
                item.st_time,
                item.ed_time,
                item.transfer_id,
            )
        )

    reasons: Dict[str, set] = {
        transfer_id: {"seed"} for transfer_id in seeds
    }

    def include(transfer_id: str, reason: str) -> bool:
        values = reasons.setdefault(transfer_id, set())
        previous = len(values)
        values.add(reason)
        return len(values) != previous

    changed = True
    while changed:
        changed = False
        for transfer_id in tuple(sorted(reasons)):
            transfer = transfers[transfer_id]
            for consumer_id in consumers[transfer_id]:
                changed |= include(consumer_id, "dependency")
            lane = LaneKey(
                transfer.src_rank,
                transfer.dst_rank,
                transfer.channel,
            )
            lane_values = lanes[lane]
            position = next(
                index
                for index, value in enumerate(lane_values)
                if value.transfer_id == transfer_id
            )
            for successor in lane_values[position + 1 :]:
                changed |= include(
                    successor.transfer_id,
                    "same_lane_successor",
                )
            link = LinkKey(transfer.src_rank, transfer.dst_rank)
            for related_id in links[link]:
                if related_id != transfer_id:
                    changed |= include(
                        related_id,
                        "directed_link_concurrency",
                    )
            for resource_id in topology.resources_for(link):
                for related_id in resource_transfers[resource_id]:
                    if related_id != transfer_id:
                        changed |= include(
                            related_id,
                            "shared_resource:{}".format(resource_id),
                        )
    frozen_reasons = {
        transfer_id: frozenset(values)
        for transfer_id, values in reasons.items()
    }
    return ImpactClosure(
        seeds,
        frozenset(reasons),
        frozen_reasons,
    )
