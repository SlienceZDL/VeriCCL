from __future__ import annotations

import re
from dataclasses import replace
from typing import Mapping, Tuple

from vericcl.errors import SemanticError
from vericcl.input.models import ResolvedInput
from vericcl.semantics.atom import Schedule, Transfer
from vericcl.semantics.checker import check_final_states
from vericcl.semantics.collective import OutputSlot, required_outputs
from vericcl.semantics.state import (
    PayloadLedger,
    PayloadState,
    initial_payload_states,
)
from vericcl.verification.model import CheckResult, ValidationStatus


_OUTPUT_KEY = re.compile(r"^r([0-9]{8})-o([0-9]{8})$")


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
    evidence=None,
) -> CheckResult:
    return CheckResult(
        dimension=dimension,
        status=ValidationStatus.INVALID,
        code=code,
        message=message,
        evidence={} if evidence is None else evidence,
    )


def _alignment_result(
    schedule: Schedule,
    inputs: ResolvedInput,
) -> CheckResult:
    if schedule.rank_count != inputs.rank_count:
        return _invalid(
            "semantic",
            "rank_count_mismatch",
            "schedule and input rank counts differ",
            {
                "schedule_rank_count": schedule.rank_count,
                "input_rank_count": inputs.rank_count,
            },
        )
    expected_slices = inputs.hyperparameters.slice_count
    if schedule.slice_count != expected_slices:
        return _invalid(
            "semantic",
            "slice_count_mismatch",
            "schedule and input slice counts differ",
            {
                "schedule_slice_count": schedule.slice_count,
                "input_slice_count": expected_slices,
            },
        )
    if schedule.slice_size_bytes != inputs.hyperparameters.slice_size_bytes:
        return _invalid(
            "semantic",
            "slice_size_mismatch",
            "schedule and input slice sizes differ",
            {
                "schedule_slice_size_bytes": schedule.slice_size_bytes,
                "input_slice_size_bytes": (
                    inputs.hyperparameters.slice_size_bytes
                ),
            },
        )
    return _valid("semantic", "dimensions_valid")


def _expected_output_metadata(
    schedule: Schedule,
    inputs: ResolvedInput,
) -> Mapping[str, Tuple[int, ...]]:
    return {
        "r{:08d}-o{:08d}".format(slot.rank, slot.offset): tuple(
            sorted(contributors)
        )
        for slot, contributors in required_outputs(
            inputs.collective,
            schedule.rank_count,
            schedule.slice_count,
        ).items()
    }


def _metadata_result(
    schedule: Schedule,
    inputs: ResolvedInput,
) -> CheckResult:
    raw = schedule.metadata.get("final_outputs")
    if not isinstance(raw, Mapping):
        return _invalid(
            "semantic",
            "final_output_metadata_missing",
            "schedule final_outputs metadata is missing",
        )
    try:
        actual = {
            key: tuple(sorted(value)) for key, value in raw.items()
        }
    except TypeError:
        return _invalid(
            "semantic",
            "final_output_metadata_invalid",
            "schedule final_outputs metadata is invalid",
        )
    expected = _expected_output_metadata(schedule, inputs)
    if actual == expected and len(schedule.final_state_ids) == len(expected):
        return _valid(
            "semantic",
            "final_output_metadata_valid",
            {"final_output_count": len(expected)},
        )
    for key, contributors in sorted(actual.items()):
        if key in expected:
            continue
        match = _OUTPUT_KEY.fullmatch(key) if isinstance(key, str) else None
        if match is None:
            continue
        rank, offset = (int(part) for part in match.groups())
        if any(
            expected_key.startswith("r{:08d}-".format(rank))
            and expected_contributors == contributors
            for expected_key, expected_contributors in expected.items()
        ):
            return _invalid(
                "semantic",
                "wrong_logical_address",
                "final output uses an incorrect logical address",
                {"rank": rank, "offset": offset},
            )
    return _invalid(
        "semantic",
        "final_output_metadata_mismatch",
        "schedule final_outputs do not match collective semantics",
        {
            "actual_keys": tuple(sorted(actual)),
            "expected_keys": tuple(sorted(expected)),
        },
    )


def _ordered_transfers(schedule: Schedule) -> Tuple[Transfer, ...]:
    pending = {
        transfer.transfer_id: transfer for transfer in schedule.transfers
    }
    completed = set()
    ordered = []
    while pending:
        ready = [
            transfer
            for transfer in pending.values()
            if transfer.predecessor_ids <= completed
        ]
        if not ready:
            raise SemanticError("schedule dependencies contain a cycle")
        transfer = min(
            ready,
            key=lambda item: (
                item.st_time,
                item.ed_time,
                item.transfer_id,
            ),
        )
        ordered.append(transfer)
        completed.add(transfer.transfer_id)
        del pending[transfer.transfer_id]
    return tuple(ordered)


def _source_state(
    ledger: PayloadLedger,
    transfer: Transfer,
) -> PayloadState:
    matches = [
        state
        for state in ledger.states
        if state.rank == transfer.src_rank
        and state.contributors == transfer.member_slice_ids
    ]
    active = [state for state in matches if state.active]
    if len(active) == 1:
        return active[0]
    if len(active) > 1:
        raise SemanticError("source payload state is ambiguous")
    if matches:
        return max(matches, key=lambda state: state.version)
    raise SemanticError("source payload state is missing")


def _destination_state(
    ledger: PayloadLedger,
    transfer: Transfer,
    source: PayloadState,
) -> PayloadState:
    address = source.logical_address
    candidates = [
        state
        for state in ledger.states
        if state.active
        and state.rank == transfer.dst_rank
        and state.logical_address == address
    ]
    disjoint = [
        state
        for state in candidates
        if not state.contributors & source.contributors
    ]
    if not disjoint:
        if candidates:
            raise SemanticError("REDUCE contributors must be disjoint")
        raise SemanticError("REDUCE destination state is missing")
    return min(
        disjoint,
        key=lambda state: (-len(state.contributors), state.state_id),
    )


def _transfer_address(transfer: Transfer, slice_count: int) -> int:
    addresses = {
        slice_id % slice_count for slice_id in transfer.member_slice_ids
    }
    if len(addresses) != 1:
        raise SemanticError(
            "transfer contributors have different logical addresses"
        )
    return addresses.pop()


def _concurrent_reduce_group(
    first: Transfer,
    candidates: Tuple[Transfer, ...],
    slice_count: int,
    completed_ids: frozenset,
) -> Tuple[Transfer, ...]:
    address = _transfer_address(first, slice_count)
    group = [first]
    group_end = first.ed_time
    for candidate in candidates:
        if candidate.kind != "REDUCE":
            continue
        if candidate.dst_rank != first.dst_rank:
            continue
        if _transfer_address(candidate, slice_count) != address:
            continue
        if not candidate.predecessor_ids <= completed_ids:
            continue
        if candidate.st_time >= group_end:
            continue
        group.append(candidate)
        group_end = max(group_end, candidate.ed_time)
    return tuple(group)


def _apply_reduce_group(
    ledger: PayloadLedger,
    group: Tuple[Transfer, ...],
) -> None:
    sources = tuple(_source_state(ledger, transfer) for transfer in group)
    for transfer, source in zip(group, sources):
        if source.ready_time > transfer.st_time:
            raise SemanticError(
                "transfer starts before its source state is ready"
            )
    contributor_union = frozenset()
    for source in sources:
        if contributor_union & source.contributors:
            raise SemanticError("REDUCE contributors must be disjoint")
        contributor_union = contributor_union | source.contributors
    destination = _destination_state(ledger, group[0], sources[0])
    if destination.contributors & contributor_union:
        raise SemanticError("REDUCE contributors must be disjoint")
    if any(destination.ready_time > transfer.st_time for transfer in group):
        raise SemanticError(
            "transfer starts before its destination is ready"
        )
    group_ready_time = max(transfer.ed_time for transfer in group)
    current = destination
    for transfer, source in zip(group, sources):
        current = ledger.reduce(
            source.state_id,
            current.state_id,
            transfer.dst_rank,
            group_ready_time,
        )


def _required_contributors(
    transfer: Transfer,
    outputs: Mapping[OutputSlot, frozenset],
) -> frozenset:
    candidates = {
        contributors
        for contributors in outputs.values()
        if transfer.member_slice_ids <= contributors
    }
    if not candidates:
        raise SemanticError(
            "transfer contributors do not belong to a required output"
        )
    return min(
        candidates,
        key=lambda value: (len(value), tuple(sorted(value))),
    )


def _failure_code(message: str) -> str:
    if "contributors must be disjoint" in message:
        return "duplicate_reduction_contributor"
    if (
        "state version is inactive" in message
        or "incomplete state already sent" in message
    ):
        return "inactive_state_reuse"
    if "logical address" in message:
        return "wrong_logical_address"
    return "state_replay_failed"


def _state_result(
    schedule: Schedule,
    inputs: ResolvedInput,
) -> CheckResult:
    outputs = required_outputs(
        inputs.collective,
        schedule.rank_count,
        schedule.slice_count,
    )
    ledger = PayloadLedger(
        initial_payload_states(schedule.rank_count, schedule.slice_count)
    )
    try:
        ordered = _ordered_transfers(schedule)
    except SemanticError as error:
        return _invalid("state", "state_replay_failed", str(error))
    remaining = list(ordered)
    completed_ids = set()
    while remaining:
        transfer = remaining.pop(0)
        active_transfer_ids = (transfer.transfer_id,)
        try:
            if transfer.kind == "SEND":
                source = _source_state(ledger, transfer)
                if source.ready_time > transfer.st_time:
                    raise SemanticError(
                        "transfer starts before its source state is ready"
                    )
                ledger.send(
                    source.state_id,
                    transfer.dst_rank,
                    transfer.ed_time,
                    _required_contributors(transfer, outputs),
                )
            else:
                group = _concurrent_reduce_group(
                    transfer,
                    tuple(remaining),
                    schedule.slice_count,
                    frozenset(completed_ids),
                )
                active_transfer_ids = tuple(
                    item.transfer_id for item in group
                )
                _apply_reduce_group(ledger, group)
                grouped_ids = set(active_transfer_ids[1:])
                remaining = [
                    item
                    for item in remaining
                    if item.transfer_id not in grouped_ids
                ]
            completed_ids.update(active_transfer_ids)
        except SemanticError as error:
            message = str(error)
            return _invalid(
                "state",
                _failure_code(message),
                message,
                {"transfer_ids": active_transfer_ids},
            )
    final_states = []
    for slot, contributors in sorted(outputs.items()):
        matches = [
            state
            for state in ledger.states
            if state.active
            and state.rank == slot.rank
            and state.contributors == contributors
        ]
        if not matches:
            return _invalid(
                "state",
                "missing_final_contributor",
                "missing final output at rank {} offset {}".format(
                    slot.rank,
                    slot.offset,
                ),
                {
                    "rank": slot.rank,
                    "offset": slot.offset,
                    "contributors": tuple(sorted(contributors)),
                },
            )
        if len(matches) > 1:
            return _invalid(
                "state",
                "duplicate_final_state",
                "multiple final states match one output slot",
                {"rank": slot.rank, "offset": slot.offset},
            )
        final_states.append(matches[0])
    try:
        check_final_states(
            inputs.collective,
            schedule.rank_count,
            schedule.slice_count,
            final_states,
        )
    except SemanticError as error:
        message = str(error)
        return _invalid(
            "state",
            _failure_code(message),
            message,
        )
    return _valid(
        "state",
        "state_replay_valid",
        {
            "final_state_count": len(final_states),
            "ledger_state_count": len(ledger.states),
        },
    )


def semantic_check_results(
    schedule: Schedule,
    inputs: ResolvedInput,
) -> Tuple[CheckResult, CheckResult]:
    if not isinstance(schedule, Schedule):
        raise SemanticError("schedule must be a Schedule")
    if not isinstance(inputs, ResolvedInput):
        raise SemanticError("inputs must be a ResolvedInput")
    semantic = _alignment_result(schedule, inputs)
    if semantic.status is ValidationStatus.VALID:
        semantic = _metadata_result(schedule, inputs)
    state = (
        _state_result(schedule, inputs)
        if semantic.status is ValidationStatus.VALID
        else CheckResult(
            dimension="state",
            status=ValidationStatus.NOT_RUN,
            code="semantic_prerequisite_failed",
            message="state replay requires valid collective semantics",
            evidence={"semantic_code": semantic.code},
        )
    )
    return semantic, state


def verify_schedule_semantics(
    schedule: Schedule,
    inputs: ResolvedInput,
) -> CheckResult:
    semantic, state = semantic_check_results(schedule, inputs)
    if semantic.status is not ValidationStatus.VALID:
        return semantic
    if state.status is not ValidationStatus.VALID:
        return replace(state, dimension="semantic")
    return _valid(
        "semantic",
        "schedule_semantics_valid",
        {
            "final_state_count": state.evidence["final_state_count"],
            "transfer_count": len(schedule.transfers),
        },
    )
