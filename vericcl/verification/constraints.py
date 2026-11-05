from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Tuple

from vericcl.errors import SemanticError
from vericcl.input.models import ResolvedInput
from vericcl.semantics.atom import Schedule
from vericcl.topology.model import LaneKey, LinkKey, Topology
from vericcl.verification.model import CheckResult, ValidationStatus
from vericcl.verification.semantics import semantic_check_results


def _valid(dimension: str, code: str, evidence=None) -> CheckResult:
    return CheckResult(
        dimension=dimension,
        status=ValidationStatus.VALID,
        code=code,
        message="{} validation passed".format(dimension),
        evidence={} if evidence is None else evidence,
    )


def _invalid(
    dimension: str,
    code: str,
    message: str,
    transfer_ids=(),
    **evidence
) -> CheckResult:
    values = dict(evidence)
    values["transfer_ids"] = tuple(sorted(set(transfer_ids)))
    return CheckResult(
        dimension=dimension,
        status=ValidationStatus.INVALID,
        code=code,
        message=message,
        evidence=values,
    )


def _endpoint_metadata_result(schedule: Schedule) -> CheckResult:
    operation_ids = {}
    for transfer in sorted(
        schedule.transfers,
        key=lambda item: item.transfer_id,
    ):
        atoms = tuple(transfer.atoms)
        atom_ids = tuple(atom.slice_id for atom in atoms)
        if (
            not atoms
            or len(atom_ids) != len(set(atom_ids))
            or frozenset(atom_ids) != transfer.member_slice_ids
        ):
            return _invalid(
                "endpoint",
                "paired_endpoint_metadata_missing",
                "transfer cannot derive one complete endpoint pair",
                (transfer.transfer_id,),
            )
        for atom in atoms:
            current = atom.current_symbol
            stage = atom.path[-1]
            if (
                stage.stage_id != transfer.stage_id
                or stage.operator != transfer.kind
                or current.src_rank != transfer.src_rank
                or current.dst_rank != transfer.dst_rank
            ):
                return _invalid(
                    "endpoint",
                    "paired_endpoint_metadata_mismatch",
                    "transfer endpoint member metadata differs",
                    (transfer.transfer_id,),
                )
            key = (
                transfer.stage_id,
                transfer.src_rank,
                transfer.dst_rank,
                atom.slice_id,
            )
            previous = operation_ids.get(key)
            if previous is not None and previous != transfer.transfer_id:
                return _invalid(
                    "endpoint",
                    "duplicate_physical_operation",
                    "member atom has duplicate physical operations",
                    (previous, transfer.transfer_id),
                    slice_ids=(atom.slice_id,),
                )
            operation_ids[key] = transfer.transfer_id
    for transfer in schedule.transfers:
        for atom in transfer.atoms:
            for stage in atom.path:
                for symbol in stage.symbols:
                    key = (
                        stage.stage_id,
                        symbol.src_rank,
                        symbol.dst_rank,
                        atom.slice_id,
                    )
                    if key not in operation_ids:
                        return _invalid(
                            "endpoint",
                            "path_operation_missing",
                            "atom path operation is missing from the schedule",
                            (transfer.transfer_id,),
                            slice_ids=(atom.slice_id,),
                        )
    return _valid(
        "endpoint",
        "endpoint_metadata_valid",
        {"physical_operation_count": len(operation_ids)},
    )


def _topology_result(
    schedule: Schedule,
    inputs: ResolvedInput,
    topology: Topology,
) -> CheckResult:
    if (
        schedule.rank_count != inputs.rank_count
        or topology.rank_count != inputs.rank_count
    ):
        return _invalid(
            "topology",
            "rank_count_mismatch",
            "schedule, input, and topology rank counts differ",
        )
    forbidden = inputs.atom_constraints.forbidden_transfers
    for transfer in sorted(
        schedule.transfers,
        key=lambda item: item.transfer_id,
    ):
        key = LinkKey(transfer.src_rank, transfer.dst_rank)
        edge = topology.links.get(key)
        if edge is None:
            return _invalid(
                "topology",
                "missing_directed_link",
                "schedule transfer uses an absent directed link",
                (transfer.transfer_id,),
                src_rank=transfer.src_rank,
                dst_rank=transfer.dst_rank,
            )
        if transfer.channel >= edge.max_channels:
            return _invalid(
                "topology",
                "channel_limit_exceeded",
                "schedule transfer exceeds the directed link channel limit",
                (transfer.transfer_id,),
                channel=transfer.channel,
                limit=edge.max_channels,
            )
        matches = tuple(
            sorted(
                item.slice_id
                for item in forbidden
                if item.slice_id in transfer.member_slice_ids
                and item.src_rank == transfer.src_rank
                and item.dst_rank == transfer.dst_rank
                and item.stage_id == transfer.stage_id
            )
        )
        if matches:
            return _invalid(
                "topology",
                "forbidden_transfer",
                "shared physical transfer contains a forbidden member",
                (transfer.transfer_id,),
                slice_ids=matches,
            )
    return _valid(
        "topology",
        "topology_constraints_valid",
        {"transfer_count": len(schedule.transfers)},
    )


def _timing_result(schedule: Schedule) -> CheckResult:
    by_id = {
        transfer.transfer_id: transfer for transfer in schedule.transfers
    }
    operations = {
        (
            transfer.stage_id,
            transfer.src_rank,
            transfer.dst_rank,
            atom.slice_id,
        ): transfer
        for transfer in schedule.transfers
        for atom in transfer.atoms
    }
    for transfer in sorted(
        schedule.transfers,
        key=lambda item: item.transfer_id,
    ):
        for predecessor_id in sorted(transfer.predecessor_ids):
            predecessor = by_id[predecessor_id]
            if predecessor.ed_time > transfer.st_time:
                return _invalid(
                    "timing",
                    "predecessor_not_ready",
                    "transfer starts before a predecessor is ready",
                    (transfer.transfer_id,),
                    predecessor_id=predecessor_id,
                    predecessor_end_time=predecessor.ed_time,
                    start_time=transfer.st_time,
                )
        for atom in transfer.atoms:
            if atom.current_symbol.ready_time > transfer.st_time:
                return _invalid(
                    "timing",
                    "atom_not_ready",
                    "transfer starts before an atom is ready",
                    (transfer.transfer_id,),
                    slice_ids=(atom.slice_id,),
                )
            path_operations = tuple(
                (stage, symbol)
                for stage in atom.path
                for symbol in stage.symbols
            )
            for (previous_stage, previous), (_, current) in zip(
                path_operations,
                path_operations[1:],
            ):
                previous_transfer = operations.get(
                    (
                        previous_stage.stage_id,
                        previous.src_rank,
                        previous.dst_rank,
                        atom.slice_id,
                    )
                )
                if previous_transfer is None:
                    continue
                if previous_transfer.ed_time > current.ready_time:
                    return _invalid(
                        "timing",
                        "path_ready_time_invalid",
                        "path ready time precedes its physical operation",
                        (transfer.transfer_id,),
                        path_transfer_id=previous_transfer.transfer_id,
                        slice_ids=(atom.slice_id,),
                        ready_time=current.ready_time,
                        path_end_time=previous_transfer.ed_time,
                    )
    lanes = defaultdict(list)
    for transfer in schedule.transfers:
        lanes[
            LaneKey(
                transfer.src_rank,
                transfer.dst_rank,
                transfer.channel,
            )
        ].append(transfer)
    for lane, transfers in sorted(lanes.items()):
        ordered = sorted(
            transfers,
            key=lambda item: (
                item.st_time,
                item.ed_time,
                item.transfer_id,
            ),
        )
        for previous, current in zip(ordered, ordered[1:]):
            if current.st_time < previous.ed_time:
                return _invalid(
                    "timing",
                    "lane_overlap",
                    "same-lane transfer intervals overlap",
                    (current.transfer_id,),
                    conflicting_transfer_id=previous.transfer_id,
                    lane=(
                        lane.src_rank,
                        lane.dst_rank,
                        lane.channel,
                    ),
                )
    return _valid(
        "timing",
        "timing_constraints_valid",
        {"lane_count": len(lanes)},
    )


def _resource_result(
    schedule: Schedule,
    topology: Topology,
) -> CheckResult:
    raw_slots = schedule.metadata.get("resource_slots", {})
    if not isinstance(raw_slots, Mapping):
        return _invalid(
            "resource",
            "resource_slots_invalid",
            "resource_slots metadata must be a mapping",
        )
    intervals = defaultdict(list)
    for transfer in sorted(
        schedule.transfers,
        key=lambda item: item.transfer_id,
    ):
        slots = raw_slots.get(transfer.transfer_id, {})
        if not isinstance(slots, Mapping):
            return _invalid(
                "resource",
                "resource_slots_invalid",
                "transfer resource slots must be a mapping",
                (transfer.transfer_id,),
            )
        key = LinkKey(transfer.src_rank, transfer.dst_rank)
        for resource_id in topology.resources_for(key):
            if resource_id not in slots:
                return _invalid(
                    "resource",
                    "shared_resource_slot_missing",
                    "shared resource transfer has no fixed slot",
                    (transfer.transfer_id,),
                    resource_id=resource_id,
                )
            slot = slots[resource_id]
            resource = topology.shared_resources[resource_id]
            if (
                isinstance(slot, bool)
                or not isinstance(slot, int)
                or slot < 0
                or slot >= resource.max_channels
            ):
                return _invalid(
                    "resource",
                    "shared_resource_capacity_exceeded",
                    "shared resource slot exceeds fixed capacity",
                    (transfer.transfer_id,),
                    resource_id=resource_id,
                    slot=slot,
                    limit=resource.max_channels,
                )
            intervals[(resource_id, slot)].append(transfer)
    for (resource_id, slot), transfers in sorted(intervals.items()):
        ordered = sorted(
            transfers,
            key=lambda item: (
                item.st_time,
                item.ed_time,
                item.transfer_id,
            ),
        )
        for previous, current in zip(ordered, ordered[1:]):
            if current.st_time < previous.ed_time:
                return _invalid(
                    "resource",
                    "shared_resource_capacity_exceeded",
                    "shared resource fixed-slot intervals overlap",
                    (current.transfer_id,),
                    conflicting_transfer_id=previous.transfer_id,
                    resource_id=resource_id,
                    slot=slot,
                )
    return _valid(
        "resource",
        "resource_constraints_valid",
        {"resource_lane_count": len(intervals)},
    )


def constraint_check_results(
    schedule: Schedule,
    inputs: ResolvedInput,
    topology: Topology,
) -> Tuple[CheckResult, ...]:
    if not isinstance(schedule, Schedule):
        raise SemanticError("schedule must be a Schedule")
    if not isinstance(inputs, ResolvedInput):
        raise SemanticError("inputs must be a ResolvedInput")
    if not isinstance(topology, Topology):
        raise SemanticError("topology must be a Topology")
    endpoint = _endpoint_metadata_result(schedule)
    topology_result = _topology_result(schedule, inputs, topology)
    timing = _timing_result(schedule)
    resource = (
        _resource_result(schedule, topology)
        if topology_result.status is ValidationStatus.VALID
        else CheckResult(
            dimension="resource",
            status=ValidationStatus.NOT_RUN,
            code="topology_prerequisite_failed",
            message="resource validation requires a valid topology",
            evidence={"topology_code": topology_result.code},
        )
    )
    return endpoint, topology_result, timing, resource


def verify_schedule_constraints(
    schedule: Schedule,
    inputs: ResolvedInput,
    topology: Topology,
) -> CheckResult:
    results = constraint_check_results(schedule, inputs, topology)
    for result in results:
        if result.status is ValidationStatus.INVALID:
            return CheckResult(
                dimension="constraints",
                status=result.status,
                code=result.code,
                message=result.message,
                evidence=result.evidence,
            )
    return _valid(
        "constraints",
        "schedule_constraints_valid",
        {"dimensions": tuple(result.dimension for result in results)},
    )


def verify_schedule_pre_lowering(
    schedule: Schedule,
    inputs: ResolvedInput,
    topology: Topology,
) -> Tuple[CheckResult, ...]:
    semantic = semantic_check_results(schedule, inputs)
    constraints = constraint_check_results(schedule, inputs, topology)
    return semantic + constraints
