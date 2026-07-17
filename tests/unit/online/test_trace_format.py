from dataclasses import replace

import pytest

from vericcl.errors import SemanticError
from vericcl.semantics.collective import CollectiveKind
from vericcl.topology import load_topology
from vericcl.topology.model import (
    DirectedLink,
    LinkKey,
    PerformanceCurve,
    Topology,
)
from vericcl.verification.online.trace_format import (
    RAW_HEADER_STRUCT,
    RAW_RECORD_STRUCT,
    RawStepTraceRecord,
    encode_raw_trace,
    parse_trace,
)
from vericcl.xml.endpoints import EndpointType
from vericcl.xml.lower import lower_to_xml
from vericcl.xml.trace_sidecar import (
    TraceSidecar,
    build_trace_sidecar,
    load_trace_sidecar,
    write_trace_sidecar,
)

from tests.unit.xml.helpers import (
    allreduce_star_schedule,
    resolved,
    two_rank_allreduce_schedule,
)


pytestmark = pytest.mark.phase06


def _two_rank_sidecar():
    schedule = two_rank_allreduce_schedule()
    inputs = resolved(CollectiveKind.ALL_REDUCE, ranks=2, slices=1)
    artifact = lower_to_xml(schedule, inputs, load_topology(inputs))
    return build_trace_sidecar(artifact, schedule)


def _complete_topology(schedule):
    curve = PerformanceCurve(1.0, 2.0, {})
    keys = {
        LinkKey(transfer.src_rank, transfer.dst_rank)
        for transfer in schedule.transfers
    }
    return Topology(
        rank_count=schedule.rank_count,
        links={key: DirectedLink(key, 4, curve, ()) for key in keys},
        shared_resources={},
        node_membership={rank: 0 for rank in range(schedule.rank_count)},
        gateways=frozenset(),
        warnings=(),
    )


def _entry(sidecar, transfer_id, endpoint_type):
    return next(
        entry
        for entry in sidecar.entries.values()
        if entry.transfer_id == transfer_id
        and entry.endpoint_type is endpoint_type
    )


def _raw(entry, *, iteration=0, start=30, end=40):
    return RawStepTraceRecord(
        rank=entry.rank,
        tb_id=entry.tb_id,
        step_index=entry.step_index,
        endpoint_type=entry.runtime_endpoint_type,
        peer=entry.peer,
        channel=entry.runtime_channel,
        iteration=iteration,
        tb_reach=10,
        dependency_done=20,
        transfer_start=start,
        transfer_end=end,
        flags=0,
        reserved=0,
    )


def _rank_records(sidecar, rank):
    return tuple(
        _raw(entry)
        for entry in sorted(sidecar.entries.values(), key=lambda value: value.key)
        if entry.rank == rank
    )


def test_sidecar_maps_runtime_step_to_full_semantic_metadata():
    sidecar = _two_rank_sidecar()
    entry = _entry(sidecar, "allreduce-send", EndpointType.SEND)

    assert sidecar.xml_sha256
    assert sidecar.entry(entry.rank, entry.tb_id, entry.step_index) == entry
    assert entry.atom_ids == (
        "allreduce-send:atom-s00000000",
        "allreduce-send:atom-s00000001",
    )
    assert len(entry.flow_ids) == 2
    assert entry.lane.src_rank == 0
    assert entry.lane.dst_rank == 1
    assert entry.lane.channel == 0
    assert entry.semantic_predecessor_ids == ("allreduce-reduce",)
    assert TraceSidecar.from_json_text(sidecar.to_json_text()) == sidecar
    assert RAW_HEADER_STRUCT.size == 40
    assert RAW_RECORD_STRUCT.size == 64


def test_sidecar_file_round_trip(tmp_path):
    sidecar = _two_rank_sidecar()
    path = tmp_path / "schedule.trace.json"

    write_trace_sidecar(sidecar, path)

    assert load_trace_sidecar(path) == sidecar


def test_sidecar_accounts_for_nops_removed_by_msccl_runtime():
    schedule = allreduce_star_schedule()
    inputs = resolved(CollectiveKind.ALL_REDUCE, ranks=4, slices=1)
    artifact = lower_to_xml(schedule, inputs, _complete_topology(schedule))
    sidecar = build_trace_sidecar(artifact, schedule)
    entry = _entry(sidecar, "allreduce-send-1", EndpointType.SEND)

    assert entry.xml_step_index == 2
    assert entry.step_index == 0
    for threadblock in artifact.tb_program.threadblocks:
        entries = sorted(
            (
                value
                for value in sidecar.entries.values()
                if value.rank == threadblock.key.rank
                and value.tb_id == threadblock.tb_id
            ),
            key=lambda value: value.step_index,
        )
        assert [value.step_index for value in entries] == list(
            range(len(entries))
        )


def test_binary_trace_round_trip_joins_sidecar_metadata(tmp_path):
    sidecar = _two_rank_sidecar()
    entry = _entry(sidecar, "allreduce-send", EndpointType.SEND)
    raw_records = _rank_records(sidecar, entry.rank)
    path = tmp_path / "trace.bin"
    path.write_bytes(encode_raw_trace(raw_records, rank=entry.rank))

    records = parse_trace(path, sidecar)

    assert len(records) == len(raw_records)
    record = next(
        value for value in records if value.transfer_id == "allreduce-send"
    )
    assert record.raw == _raw(entry)
    assert record.atom_ids == entry.atom_ids
    assert record.flow_ids == entry.flow_ids
    assert record.semantic_predecessor_ids == (
        "allreduce-reduce",
    )


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda data: b"BAD!" + data[4:], "magic"),
        (
            lambda data: data[:4] + b"\x02\x00" + data[6:],
            "version",
        ),
        (lambda data: data[:-1], "length"),
    ],
)
def test_parser_rejects_bad_or_truncated_binary(tmp_path, mutation, match):
    sidecar = _two_rank_sidecar()
    entry = _entry(sidecar, "allreduce-send", EndpointType.SEND)
    encoded = encode_raw_trace(
        _rank_records(sidecar, entry.rank),
        rank=entry.rank,
    )
    path = tmp_path / "trace.bin"
    path.write_bytes(mutation(encoded))

    with pytest.raises(SemanticError, match=match):
        parse_trace(path, sidecar)


def test_parser_rejects_overflow_and_missing_sidecar_entry(tmp_path):
    sidecar = _two_rank_sidecar()
    entry = _entry(sidecar, "allreduce-send", EndpointType.SEND)
    path = tmp_path / "trace.bin"
    path.write_bytes(
        encode_raw_trace((_raw(entry),), rank=entry.rank, overflow=True)
    )
    with pytest.raises(SemanticError, match="overflow"):
        parse_trace(path, sidecar)

    missing = replace(_raw(entry), tb_id=entry.tb_id + 100)
    path.write_bytes(encode_raw_trace((missing,), rank=entry.rank))
    with pytest.raises(SemanticError, match="sidecar"):
        parse_trace(path, sidecar)


def test_parser_rejects_a_well_formed_but_incomplete_iteration(tmp_path):
    sidecar = _two_rank_sidecar()
    entry = _entry(sidecar, "allreduce-send", EndpointType.SEND)
    path = tmp_path / "trace.bin"
    path.write_bytes(encode_raw_trace((_raw(entry),), rank=entry.rank))

    with pytest.raises(SemanticError, match="incomplete"):
        parse_trace(path, sidecar)


def test_parser_rejects_endpoint_identity_mismatch(tmp_path):
    sidecar = _two_rank_sidecar()
    entry = _entry(sidecar, "allreduce-send", EndpointType.SEND)
    path = tmp_path / "trace.bin"
    path.write_bytes(
        encode_raw_trace(
            (
                replace(_rank_records(sidecar, entry.rank)[0], endpoint_type=1),
            )
            + _rank_records(sidecar, entry.rank)[1:],
            rank=entry.rank,
        )
    )

    with pytest.raises(SemanticError, match="endpoint type"):
        parse_trace(path, sidecar)
