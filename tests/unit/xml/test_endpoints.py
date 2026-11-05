from dataclasses import replace

import pytest

from vericcl.errors import SemanticError
from vericcl.semantics.collective import CollectiveKind
from vericcl.xml.buffers import build_buffer_plan
from vericcl.xml.endpoints import EndpointProgram, EndpointType, lower_endpoints

from tests.unit.xml.helpers import (
    reduce_chain_schedule,
    resolved,
    send_relay_schedule,
)


pytestmark = pytest.mark.phase04


def test_send_creates_exact_s_and_r_pair():
    schedule = send_relay_schedule()
    buffers = build_buffer_plan(
        schedule,
        resolved(CollectiveKind.BROADCAST, ranks=3, slices=1),
    )

    program = lower_endpoints(schedule, buffers)

    endpoints = program.by_transfer_id["relay-first"]
    assert {endpoint.xml_type for endpoint in endpoints} == {
        EndpointType.SEND,
        EndpointType.RECV,
    }
    assert {endpoint.rank for endpoint in endpoints} == {0, 1}
    assert all(endpoint.member_slice_ids == frozenset({0}) for endpoint in endpoints)


def test_reduce_creates_exact_s_and_rrc_pair():
    schedule = reduce_chain_schedule()
    buffers = build_buffer_plan(
        schedule,
        resolved(CollectiveKind.REDUCE, ranks=3, slices=1),
    )

    program = lower_endpoints(schedule, buffers)

    endpoints = program.by_transfer_id["reduce-a0-first"]
    assert {endpoint.xml_type for endpoint in endpoints} == {
        EndpointType.SEND,
        EndpointType.RECV_REDUCE_COPY,
    }
    recv = next(
        endpoint
        for endpoint in endpoints
        if endpoint.xml_type is EndpointType.RECV_REDUCE_COPY
    )
    assert recv.dst_ref == buffers.transfer_dst_refs["reduce-a0-first"]


def test_local_copies_are_single_local_endpoints():
    schedule = reduce_chain_schedule()
    buffers = build_buffer_plan(
        schedule,
        resolved(CollectiveKind.REDUCE, ranks=3, slices=1),
    )

    program = lower_endpoints(schedule, buffers)

    assert set(program.local_endpoints) == {
        copy.copy_id for copy in buffers.local_copies
    }
    assert all(
        endpoint.xml_type is EndpointType.COPY
        and endpoint.peer == -1
        and endpoint.channel == -1
        for endpoint in program.local_endpoints.values()
    )


def test_lowering_rejects_missing_physical_reference():
    schedule = send_relay_schedule()
    buffers = build_buffer_plan(
        schedule,
        resolved(CollectiveKind.BROADCAST, ranks=3, slices=1),
    )
    src_refs = dict(buffers.transfer_src_refs)
    del src_refs["relay-first"]

    with pytest.raises(SemanticError, match="source reference"):
        lower_endpoints(schedule, replace(buffers, transfer_src_refs=src_refs))


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda send, recv, copy: replace(send, xml_type="s"), "EndpointType"),
        (
            lambda send, recv, copy: replace(send, st_time=2.0, ed_time=1.0),
            "start must not exceed",
        ),
        (
            lambda send, recv, copy: replace(copy, atom=send.atom),
            "must not contain",
        ),
        (
            lambda send, recv, copy: replace(copy, channel=0),
            "sentinel",
        ),
        (
            lambda send, recv, copy: replace(copy, src_ref=None),
            "both references",
        ),
        (
            lambda send, recv, copy: replace(
                copy,
                src_ref=replace(copy.src_ref, rank=copy.rank + 1),
            ),
            "incorrect rank",
        ),
        (
            lambda send, recv, copy: replace(send, transfer_kind="COPY"),
            "kind is invalid",
        ),
        (
            lambda send, recv, copy: replace(send, peer=send.rank),
            "peer equals",
        ),
        (
            lambda send, recv, copy: replace(send, member_atoms=()),
            "canonical atoms",
        ),
        (
            lambda send, recv, copy: replace(
                send,
                member_slice_ids=frozenset({99}),
            ),
            "member slices",
        ),
        (
            lambda send, recv, copy: replace(send, dst_ref=recv.dst_ref),
            "send endpoint references",
        ),
        (
            lambda send, recv, copy: replace(
                send,
                src_ref=replace(send.src_ref, rank=recv.rank),
            ),
            "incorrect rank",
        ),
        (
            lambda send, recv, copy: replace(send, xml_type=EndpointType.NOP),
            "unsupported",
        ),
        (
            lambda send, recv, copy: replace(recv, src_ref=send.src_ref),
            "receive endpoint references",
        ),
        (
            lambda send, recv, copy: replace(
                recv,
                dst_ref=replace(recv.dst_ref, rank=send.rank),
            ),
            "incorrect rank",
        ),
    ],
)
def test_endpoint_models_reject_invalid_records(mutation, match):
    relay_schedule = send_relay_schedule()
    relay_buffers = build_buffer_plan(
        relay_schedule,
        resolved(CollectiveKind.BROADCAST, ranks=3, slices=1),
    )
    relay_program = lower_endpoints(relay_schedule, relay_buffers)
    send, recv = relay_program.by_transfer_id["relay-first"]
    reduce_schedule = reduce_chain_schedule()
    reduce_buffers = build_buffer_plan(
        reduce_schedule,
        resolved(CollectiveKind.REDUCE, ranks=3, slices=1),
    )
    copy = lower_endpoints(
        reduce_schedule,
        reduce_buffers,
    ).local_endpoints[reduce_buffers.local_copies[0].copy_id]

    with pytest.raises(SemanticError, match=match):
        mutation(send, recv, copy)


def test_endpoint_program_rejects_incomplete_pair():
    schedule = send_relay_schedule()
    buffers = build_buffer_plan(
        schedule,
        resolved(CollectiveKind.BROADCAST, ranks=3, slices=1),
    )
    program = lower_endpoints(schedule, buffers)
    send = program.by_transfer_id["relay-first"][0]

    with pytest.raises(SemanticError, match="exactly two"):
        EndpointProgram(
            endpoints=(send,),
            by_transfer_id={"relay-first": (send,)},
            local_endpoints={},
        )


def test_endpoint_program_rejects_non_endpoint_member():
    with pytest.raises(SemanticError, match="invalid endpoint"):
        EndpointProgram(
            endpoints=(object(),),
            by_transfer_id={},
            local_endpoints={},
        )


def test_lowering_rejects_missing_destination_reference():
    schedule = send_relay_schedule()
    buffers = build_buffer_plan(
        schedule,
        resolved(CollectiveKind.BROADCAST, ranks=3, slices=1),
    )
    dst_refs = dict(buffers.transfer_dst_refs)
    del dst_refs["relay-first"]

    with pytest.raises(SemanticError, match="destination reference"):
        lower_endpoints(schedule, replace(buffers, transfer_dst_refs=dst_refs))
