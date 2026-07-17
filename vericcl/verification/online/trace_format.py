from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Iterable, Optional, Tuple

from vericcl.errors import SemanticError
from vericcl.topology.model import LaneKey
from vericcl.xml.endpoints import EndpointType
from vericcl.xml.trace_sidecar import TraceSidecar, TraceStepMetadata


TRACE_MAGIC = 0x5643434C
TRACE_VERSION = 1
TRACE_OVERFLOW_FLAG = 0x1

RAW_HEADER_STRUCT = struct.Struct("<IHHIIQQII")
RAW_RECORD_STRUCT = struct.Struct("<IHHHhH2xI4xQQQQII")


def _unsigned(value: object, field: str, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > maximum
    ):
        raise SemanticError(
            "{} must be an unsigned integer no greater than {}".format(
                field,
                maximum,
            )
        )
    return value


@dataclass(frozen=True)
class RawStepTraceRecord:
    rank: int
    tb_id: int
    step_index: int
    endpoint_type: int
    peer: int
    channel: int
    iteration: int
    tb_reach: int
    dependency_done: int
    transfer_start: int
    transfer_end: int
    flags: int
    reserved: int

    def __post_init__(self) -> None:
        for value, field, maximum in (
            (self.rank, "raw_trace.rank", 0xFFFFFFFF),
            (self.tb_id, "raw_trace.tb_id", 0xFFFF),
            (self.step_index, "raw_trace.step_index", 0xFFFF),
            (self.endpoint_type, "raw_trace.endpoint_type", 0xFFFF),
            (self.channel, "raw_trace.channel", 0xFFFF),
            (self.iteration, "raw_trace.iteration", 0xFFFFFFFF),
            (self.tb_reach, "raw_trace.tb_reach", 0xFFFFFFFFFFFFFFFF),
            (
                self.dependency_done,
                "raw_trace.dependency_done",
                0xFFFFFFFFFFFFFFFF,
            ),
            (
                self.transfer_start,
                "raw_trace.transfer_start",
                0xFFFFFFFFFFFFFFFF,
            ),
            (
                self.transfer_end,
                "raw_trace.transfer_end",
                0xFFFFFFFFFFFFFFFF,
            ),
            (self.flags, "raw_trace.flags", 0xFFFFFFFF),
            (self.reserved, "raw_trace.reserved", 0xFFFFFFFF),
        ):
            _unsigned(value, field, maximum)
        if (
            isinstance(self.peer, bool)
            or not isinstance(self.peer, int)
            or self.peer < -0x8000
            or self.peer > 0x7FFF
        ):
            raise SemanticError("raw_trace.peer must fit int16")
        if self.reserved != 0:
            raise SemanticError("raw trace reserved field must be zero")
        if not (
            self.tb_reach
            <= self.dependency_done
            <= self.transfer_start
            <= self.transfer_end
        ):
            raise SemanticError("raw trace timestamps are not monotonic")


@dataclass(frozen=True)
class StepTraceRecord:
    raw: RawStepTraceRecord
    metadata: TraceStepMetadata

    def __post_init__(self) -> None:
        if not isinstance(self.raw, RawStepTraceRecord):
            raise SemanticError("step trace raw record is invalid")
        if not isinstance(self.metadata, TraceStepMetadata):
            raise SemanticError("step trace metadata is invalid")
        if (
            self.raw.rank,
            self.raw.tb_id,
            self.raw.step_index,
        ) != self.metadata.key:
            raise SemanticError("step trace raw and sidecar keys differ")
        if self.raw.endpoint_type != self.metadata.runtime_endpoint_type:
            raise SemanticError("step trace endpoint type differs from sidecar")
        if self.raw.peer != self.metadata.peer:
            raise SemanticError("step trace peer differs from sidecar")
        if self.raw.channel != self.metadata.runtime_channel:
            raise SemanticError("step trace channel differs from sidecar")

    @property
    def rank(self) -> int:
        return self.raw.rank

    @property
    def tb_id(self) -> int:
        return self.raw.tb_id

    @property
    def step_index(self) -> int:
        return self.raw.step_index

    @property
    def iteration(self) -> int:
        return self.raw.iteration

    @property
    def endpoint_type(self) -> EndpointType:
        return self.metadata.endpoint_type

    @property
    def transfer_id(self) -> str:
        return self.metadata.transfer_id

    @property
    def atom_ids(self) -> Tuple[str, ...]:
        return self.metadata.atom_ids

    @property
    def flow_ids(self) -> Tuple[str, ...]:
        return self.metadata.flow_ids

    @property
    def lane(self) -> Optional[LaneKey]:
        return self.metadata.lane

    @property
    def semantic_predecessor_ids(self) -> Tuple[str, ...]:
        return self.metadata.semantic_predecessor_ids


def _pack_record(record: RawStepTraceRecord) -> bytes:
    return RAW_RECORD_STRUCT.pack(
        record.rank,
        record.tb_id,
        record.step_index,
        record.endpoint_type,
        record.peer,
        record.channel,
        record.iteration,
        record.tb_reach,
        record.dependency_done,
        record.transfer_start,
        record.transfer_end,
        record.flags,
        record.reserved,
    )


def encode_raw_trace(
    records: Iterable[RawStepTraceRecord],
    *,
    rank: int,
    capacity: Optional[int] = None,
    overflow: bool = False,
) -> bytes:
    normalized = tuple(records)
    _unsigned(rank, "raw_trace_header.rank", 0xFFFFFFFF)
    if not all(isinstance(record, RawStepTraceRecord) for record in normalized):
        raise SemanticError("raw trace records are invalid")
    if any(record.rank != rank for record in normalized):
        raise SemanticError("raw trace record rank differs from header")
    if capacity is None:
        capacity = len(normalized)
    _unsigned(capacity, "raw_trace_header.capacity", 0xFFFFFFFFFFFFFFFF)
    if capacity < len(normalized):
        raise SemanticError("raw trace capacity is below record count")
    if not isinstance(overflow, bool):
        raise SemanticError("raw trace overflow must be a boolean")
    header = RAW_HEADER_STRUCT.pack(
        TRACE_MAGIC,
        TRACE_VERSION,
        RAW_HEADER_STRUCT.size,
        RAW_RECORD_STRUCT.size,
        rank,
        len(normalized),
        capacity,
        TRACE_OVERFLOW_FLAG if overflow else 0,
        0,
    )
    return header + b"".join(_pack_record(record) for record in normalized)


def _decode_record(data: bytes, offset: int) -> RawStepTraceRecord:
    return RawStepTraceRecord(*RAW_RECORD_STRUCT.unpack_from(data, offset))


def parse_trace(path: Path, sidecar: TraceSidecar) -> Tuple[StepTraceRecord, ...]:
    if not isinstance(sidecar, TraceSidecar):
        raise SemanticError("trace parser requires a TraceSidecar")
    try:
        trace_path = Path(path)
    except TypeError as error:
        raise SemanticError("trace path is invalid") from error
    data = trace_path.read_bytes()
    if len(data) < RAW_HEADER_STRUCT.size:
        raise SemanticError("trace file length is below the header size")
    (
        magic,
        version,
        header_size,
        record_size,
        rank,
        count,
        capacity,
        overflow,
        reserved,
    ) = RAW_HEADER_STRUCT.unpack_from(data)
    if magic != TRACE_MAGIC:
        raise SemanticError("trace file magic is invalid")
    if version != TRACE_VERSION:
        raise SemanticError("trace file version is unsupported")
    if header_size != RAW_HEADER_STRUCT.size:
        raise SemanticError("trace header size is invalid")
    if record_size != RAW_RECORD_STRUCT.size:
        raise SemanticError("trace record size is invalid")
    if count > capacity:
        raise SemanticError("trace count exceeds capacity")
    if overflow & TRACE_OVERFLOW_FLAG:
        raise SemanticError("trace buffer overflow invalidates the trace")
    if overflow != 0 or reserved != 0:
        raise SemanticError("trace header flags are invalid")
    expected_size = header_size + count * record_size
    if len(data) != expected_size:
        raise SemanticError("trace file length does not match its header")

    result = []
    identities = set()
    for index in range(count):
        raw = _decode_record(data, header_size + index * record_size)
        if raw.rank != rank:
            raise SemanticError("trace record rank differs from header")
        metadata = sidecar.entry(raw.rank, raw.tb_id, raw.step_index)
        if raw.endpoint_type != metadata.runtime_endpoint_type:
            raise SemanticError("trace endpoint type differs from sidecar")
        if raw.peer != metadata.peer:
            raise SemanticError("trace peer differs from sidecar")
        if raw.channel != metadata.runtime_channel:
            raise SemanticError("trace channel differs from sidecar")
        identity = (
            raw.rank,
            raw.tb_id,
            raw.step_index,
            raw.iteration,
        )
        if identity in identities:
            raise SemanticError("trace contains a duplicate step record")
        identities.add(identity)
        result.append(StepTraceRecord(raw, metadata))
    expected_steps = {
        (entry.tb_id, entry.step_index)
        for entry in sidecar.entries.values()
        if entry.rank == rank
    }
    actual_iterations = {record.iteration for record in result}
    if expected_steps and not actual_iterations:
        raise SemanticError("trace is incomplete for its sidecar rank")
    for iteration in actual_iterations:
        actual_steps = {
            (record.tb_id, record.step_index)
            for record in result
            if record.iteration == iteration
        }
        if actual_steps != expected_steps:
            raise SemanticError("trace iteration is incomplete for its sidecar")
    return tuple(result)
