import copy
from dataclasses import replace

import pytest

from vericcl.errors import SemanticError
from vericcl.semantics.atom import Atom, PathStage, Symbol
from vericcl.semantics.collective import CollectiveKind
from vericcl.verification.constraints import (
    verify_schedule_constraints,
    verify_schedule_pre_lowering,
)
from vericcl.verification.model import ValidationStatus

from tests.unit.verification.helpers import (
    forbidden_shared_transfer_inputs,
    inputs,
    topology,
)
from tests.unit.xml.helpers import (
    send_relay_schedule,
    two_rank_allgather_schedule,
    two_rank_allreduce_schedule,
    two_send_same_lane_schedule,
)


pytestmark = pytest.mark.phase05


def _assert_invalid(result, code, transfer_id):
    assert result.status is ValidationStatus.INVALID
    assert result.code == code
    assert transfer_id in result.evidence["transfer_ids"]


def test_forbidden_member_in_shared_transfer_is_rejected():
    result = verify_schedule_constraints(
        two_rank_allreduce_schedule(),
        forbidden_shared_transfer_inputs(),
        topology(),
    )

    _assert_invalid(result, "forbidden_transfer", "allreduce-send")
    assert result.evidence["slice_ids"] == (1,)


def test_absent_directed_link_is_rejected():
    result = verify_schedule_constraints(
        two_rank_allreduce_schedule(),
        inputs(),
        topology(links=((1, 0),)),
    )

    _assert_invalid(result, "missing_directed_link", "allreduce-send")


def test_start_before_predecessor_ready_time_is_rejected():
    schedule = two_rank_allreduce_schedule()
    send = schedule.transfers[1]
    send_stage = PathStage(1, "SEND", (Symbol(0, 1, 0.5),))
    reduce_stage = PathStage(0, "REDUCE", (Symbol(1, 0, 0.0),))
    early = replace(
        send,
        atoms=(
            Atom(0, 1024, (send_stage,), 0.5, 1.5),
            Atom(1, 1024, (reduce_stage, send_stage), 0.5, 1.5),
        ),
        st_time=0.5,
        ed_time=1.5,
    )
    invalid = replace(
        schedule,
        transfers=(schedule.transfers[0], early),
    )

    result = verify_schedule_constraints(invalid, inputs(), topology())

    _assert_invalid(result, "predecessor_not_ready", "allreduce-send")


def test_path_ready_time_must_follow_previous_physical_operation():
    schedule = send_relay_schedule()
    second = schedule.transfers[1]
    path = PathStage(
        0,
        "SEND",
        (
            Symbol(0, 1, 0.0),
            Symbol(1, 2, 0.5),
        ),
    )
    invalid_second = replace(
        second,
        atoms=(Atom(0, 1024, (path,), 1.0, 2.0),),
        predecessor_ids=frozenset(),
    )
    invalid = replace(
        schedule,
        transfers=(schedule.transfers[0], invalid_second),
    )

    result = verify_schedule_constraints(
        invalid,
        inputs(CollectiveKind.BROADCAST, ranks=3),
        topology(rank_count=3),
    )

    _assert_invalid(result, "path_ready_time_invalid", "relay-second")
    assert result.evidence["path_transfer_id"] == "relay-first"


def test_same_lane_overlap_is_rejected():
    schedule = two_send_same_lane_schedule()
    second = schedule.transfers[1]
    stage = PathStage(0, "SEND", (Symbol(0, 1, 0.0),))
    overlapping = replace(
        second,
        atoms=(Atom(1, 1024, (stage,), 0.5, 1.5),),
        st_time=0.5,
        ed_time=1.5,
    )
    invalid = replace(
        schedule,
        transfers=(schedule.transfers[0], overlapping),
    )

    result = verify_schedule_constraints(
        invalid,
        inputs(CollectiveKind.BROADCAST, slices=2),
        topology(),
    )

    _assert_invalid(result, "lane_overlap", "lane-send-1")
    assert result.evidence["conflicting_transfer_id"] == "lane-send-0"


def test_fixed_shared_resource_capacity_is_enforced():
    schedule = two_rank_allgather_schedule()
    metadata = dict(schedule.metadata)
    metadata["resource_slots"] = {
        transfer.transfer_id: {"nic": 0}
        for transfer in schedule.transfers
    }
    schedule = replace(schedule, metadata=metadata)

    result = verify_schedule_constraints(
        schedule,
        inputs(CollectiveKind.ALL_GATHER),
        topology(shared=True, resource_channels=1),
    )

    _assert_invalid(
        result,
        "shared_resource_capacity_exceeded",
        "allgather-send-1",
    )
    assert result.evidence["resource_id"] == "nic"


def test_missing_paired_endpoint_member_metadata_is_rejected():
    schedule = two_rank_allreduce_schedule()
    broken = copy.copy(schedule.transfers[1])
    object.__setattr__(broken, "atoms", ())
    invalid = replace(
        schedule,
        transfers=(schedule.transfers[0], broken),
    )

    result = verify_schedule_constraints(invalid, inputs(), topology())

    _assert_invalid(
        result,
        "paired_endpoint_metadata_missing",
        "allreduce-send",
    )


def test_valid_schedule_constraints_pass():
    result = verify_schedule_constraints(
        two_rank_allreduce_schedule(),
        inputs(),
        topology(),
    )

    assert result.status is ValidationStatus.VALID


def test_rank_and_channel_limits_are_checked():
    schedule = two_rank_allreduce_schedule()
    rank_result = verify_schedule_constraints(
        schedule,
        inputs(),
        topology(rank_count=3),
    )
    send = replace(schedule.transfers[1], channel=2)
    channel_result = verify_schedule_constraints(
        replace(schedule, transfers=(schedule.transfers[0], send)),
        inputs(),
        topology(),
    )

    assert rank_result.code == "rank_count_mismatch"
    assert channel_result.code == "channel_limit_exceeded"


def test_shared_resource_slots_must_be_present_and_well_formed():
    schedule = two_rank_allgather_schedule()
    missing = verify_schedule_constraints(
        schedule,
        inputs(CollectiveKind.ALL_GATHER),
        topology(shared=True),
    )
    metadata = dict(schedule.metadata)
    metadata["resource_slots"] = "invalid"
    invalid = verify_schedule_constraints(
        replace(schedule, metadata=metadata),
        inputs(CollectiveKind.ALL_GATHER),
        topology(shared=True),
    )

    assert missing.code == "shared_resource_slot_missing"
    assert invalid.code == "resource_slots_invalid"


def test_pre_lowering_preserves_independent_dimensions():
    results = verify_schedule_pre_lowering(
        two_rank_allreduce_schedule(),
        inputs(),
        topology(),
    )

    assert tuple(result.dimension for result in results) == (
        "semantic",
        "state",
        "endpoint",
        "topology",
        "timing",
        "resource",
    )
    assert all(result.status is ValidationStatus.VALID for result in results)


def test_constraint_verifier_rejects_invalid_arguments():
    schedule = two_rank_allreduce_schedule()
    with pytest.raises(SemanticError, match="Schedule"):
        verify_schedule_constraints(None, inputs(), topology())
    with pytest.raises(SemanticError, match="ResolvedInput"):
        verify_schedule_constraints(schedule, None, topology())
    with pytest.raises(SemanticError, match="Topology"):
        verify_schedule_constraints(schedule, inputs(), None)
