from dataclasses import replace

import pytest

from vericcl.errors import SemanticError
from vericcl.semantics.collective import CollectiveKind
from vericcl.verification.model import ValidationStatus
from vericcl.verification.semantics import verify_schedule_semantics

from tests.unit.verification.helpers import (
    duplicate_reduction_schedule,
    inactive_reuse_schedule,
    inputs,
    interleaved_concurrent_reduce_schedule,
    reduce_scatter_schedule,
)
from tests.unit.xml.helpers import (
    allreduce_star_schedule,
    inplace_alltoall_overwrite_schedule,
    reduce_chain_schedule,
    send_relay_schedule,
    two_rank_allgather_schedule,
    two_rank_allreduce_schedule,
)


pytestmark = pytest.mark.phase05


def test_valid_schedule_replays_to_exact_required_outputs():
    result = verify_schedule_semantics(
        two_rank_allreduce_schedule(),
        inputs(),
    )

    assert result.status is ValidationStatus.VALID
    assert result.evidence["final_state_count"] == 2


def test_reduction_send_carries_the_complete_real_contributor_set():
    schedule = two_rank_allreduce_schedule()
    reduce_transfer, send_transfer = schedule.transfers

    assert reduce_transfer.member_slice_ids == frozenset({1})
    assert send_transfer.member_slice_ids == frozenset({0, 1})
    assert send_transfer.predecessor_ids == frozenset(
        {reduce_transfer.transfer_id}
    )
    assert {
        atom.slice_id for atom in send_transfer.atoms
    } == {0, 1}
    assert schedule.metadata["final_outputs"] == {
        "r00000000-o00000000": (0, 1),
        "r00000001-o00000000": (0, 1),
    }
    assert verify_schedule_semantics(
        schedule,
        inputs(),
    ).status is ValidationStatus.VALID


@pytest.mark.parametrize(
    "schedule,input_value",
    [
        (
            send_relay_schedule(),
            inputs(CollectiveKind.BROADCAST, ranks=3),
        ),
        (
            reduce_chain_schedule(slices=2),
            inputs(CollectiveKind.REDUCE, ranks=3, slices=2),
        ),
        (
            two_rank_allgather_schedule(),
            inputs(CollectiveKind.ALL_GATHER),
        ),
        (
            allreduce_star_schedule(),
            inputs(CollectiveKind.ALL_REDUCE, ranks=4),
        ),
        (
            inplace_alltoall_overwrite_schedule(),
            inputs(CollectiveKind.ALL_TO_ALL, slices=2),
        ),
        (
            reduce_scatter_schedule(),
            inputs(CollectiveKind.REDUCE_SCATTER, slices=2),
        ),
    ],
)
def test_six_direct_collectives_replay_exactly(schedule, input_value):
    result = verify_schedule_semantics(schedule, input_value)

    assert result.status is ValidationStatus.VALID


def test_concurrent_reduce_groups_are_independent_of_transfer_id_order():
    result = verify_schedule_semantics(
        interleaved_concurrent_reduce_schedule(),
        inputs(CollectiveKind.REDUCE, ranks=3, slices=2),
    )

    assert result.status is ValidationStatus.VALID


def test_missing_final_contributor_is_invalid():
    schedule = two_rank_allreduce_schedule()
    metadata = dict(schedule.metadata)
    metadata["semantic_predecessors"] = {
        "allreduce-reduce": (),
    }
    incomplete = replace(
        schedule,
        transfers=(schedule.transfers[0],),
        metadata=metadata,
    )

    result = verify_schedule_semantics(incomplete, inputs())

    assert result.status is ValidationStatus.INVALID
    assert result.code == "missing_final_contributor"
    assert "rank 1" in result.message


def test_duplicate_reduction_contributor_is_invalid():
    result = verify_schedule_semantics(
        duplicate_reduction_schedule(),
        inputs(),
    )

    assert result.status is ValidationStatus.INVALID
    assert result.code == "duplicate_reduction_contributor"
    assert "duplicate-reduce" in result.evidence["transfer_ids"]


def test_inactive_state_cannot_be_reused():
    result = verify_schedule_semantics(
        inactive_reuse_schedule(),
        inputs(CollectiveKind.REDUCE, ranks=3, slices=1),
    )

    assert result.status is ValidationStatus.INVALID
    assert result.code == "inactive_state_reuse"
    assert "incomplete-send-2" in result.evidence["transfer_ids"]


def test_wrong_final_logical_address_is_invalid():
    schedule = two_rank_allreduce_schedule()
    metadata = dict(schedule.metadata)
    outputs = dict(metadata["final_outputs"])
    contributors = outputs.pop("r00000001-o00000000")
    outputs["r00000001-o00000001"] = contributors
    metadata["final_outputs"] = outputs
    invalid = replace(schedule, metadata=metadata)

    result = verify_schedule_semantics(invalid, inputs())

    assert result.status is ValidationStatus.INVALID
    assert result.code == "wrong_logical_address"
    assert result.evidence["rank"] == 1
    assert result.evidence["offset"] == 1


def test_schedule_and_input_dimensions_must_match():
    schedule = two_rank_allreduce_schedule()
    value = inputs()

    rank_result = verify_schedule_semantics(
        replace(schedule, rank_count=3),
        value,
    )
    slice_result = verify_schedule_semantics(
        schedule,
        replace(
            value,
            hyperparameters=replace(
                value.hyperparameters,
                total_size_bytes=2048,
            ),
        ),
    )
    size_result = verify_schedule_semantics(
        schedule,
        replace(
            value,
            hyperparameters=replace(
                value.hyperparameters,
                total_size_bytes=2048,
                slice_size_bytes=2048,
            ),
        ),
    )

    assert rank_result.code == "rank_count_mismatch"
    assert slice_result.code == "slice_count_mismatch"
    assert size_result.code == "slice_size_mismatch"


@pytest.mark.parametrize(
    "final_outputs,final_state_ids,code",
    [
        (None, None, "final_output_metadata_missing"),
        ({"invalid": None}, None, "final_output_metadata_invalid"),
        ({}, None, "final_output_metadata_mismatch"),
        (None, ("final-r0-o0",), "final_output_metadata_mismatch"),
    ],
)
def test_invalid_final_output_metadata_is_reported(
    final_outputs,
    final_state_ids,
    code,
):
    schedule = two_rank_allreduce_schedule()
    metadata = dict(schedule.metadata)
    if final_outputs is None:
        if final_state_ids is None:
            metadata.pop("final_outputs")
    else:
        metadata["final_outputs"] = final_outputs
    invalid = replace(
        schedule,
        metadata=metadata,
        final_state_ids=(
            schedule.final_state_ids
            if final_state_ids is None
            else final_state_ids
        ),
    )

    result = verify_schedule_semantics(invalid, inputs())

    assert result.code == code


def test_semantic_verifier_rejects_invalid_arguments():
    with pytest.raises(SemanticError, match="Schedule"):
        verify_schedule_semantics(None, inputs())
    with pytest.raises(SemanticError, match="ResolvedInput"):
        verify_schedule_semantics(two_rank_allreduce_schedule(), None)


def test_state_replay_rejects_cyclic_transfer_dependencies():
    schedule = two_rank_allgather_schedule()
    first = replace(
        schedule.transfers[0],
        predecessor_ids=frozenset({schedule.transfers[1].transfer_id}),
    )
    second = replace(
        schedule.transfers[1],
        predecessor_ids=frozenset({schedule.transfers[0].transfer_id}),
    )
    cyclic = replace(schedule, transfers=(first, second))

    result = verify_schedule_semantics(
        cyclic,
        inputs(CollectiveKind.ALL_GATHER),
    )

    assert result.status is ValidationStatus.INVALID
    assert result.code == "state_replay_failed"
    assert "cycle" in result.message
