from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Optional, Tuple

from vericcl.errors import SemanticError
from vericcl.semantics.atom import Schedule
from vericcl.topology.model import LaneKey
from vericcl.verification.flow_index import build_flow_index
from vericcl.xml.endpoints import EndpointType


TRACE_SIDECAR_VERSION = 1

_RUNTIME_ENDPOINT_TYPES = {
    EndpointType.SEND: 0,
    EndpointType.RECV: 1,
    EndpointType.RECV_REDUCE_COPY: 4,
    EndpointType.COPY: 6,
}


def _integer(value: object, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SemanticError(
            "{} must be an integer of at least {}".format(field, minimum)
        )
    return value


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise SemanticError("{} must be a non-empty string".format(field))
    return value


@dataclass(frozen=True)
class TraceStepMetadata:
    rank: int
    tb_id: int
    step_index: int
    xml_step_index: int
    step_id: str
    transfer_id: str
    endpoint_type: EndpointType
    runtime_endpoint_type: int
    peer: int
    runtime_channel: int
    stage_id: int
    atom_ids: Tuple[str, ...]
    flow_ids: Tuple[str, ...]
    lane: Optional[LaneKey]
    semantic_predecessor_ids: Tuple[str, ...]
    member_slice_ids: frozenset[int]

    def __post_init__(self) -> None:
        for value, field in (
            (self.rank, "trace_step.rank"),
            (self.tb_id, "trace_step.tb_id"),
            (self.step_index, "trace_step.step_index"),
            (self.xml_step_index, "trace_step.xml_step_index"),
            (self.runtime_endpoint_type, "trace_step.runtime_endpoint_type"),
            (self.runtime_channel, "trace_step.runtime_channel"),
        ):
            _integer(value, field)
        for value, field in (
            (self.step_id, "trace_step.step_id"),
            (self.transfer_id, "trace_step.transfer_id"),
        ):
            _identifier(value, field)
        if self.endpoint_type not in _RUNTIME_ENDPOINT_TYPES:
            raise SemanticError("trace_step endpoint type is not executable")
        if (
            self.runtime_endpoint_type
            != _RUNTIME_ENDPOINT_TYPES[self.endpoint_type]
        ):
            raise SemanticError("trace_step runtime endpoint type is incorrect")
        if isinstance(self.peer, bool) or not isinstance(self.peer, int):
            raise SemanticError("trace_step.peer must be an integer")
        if isinstance(self.stage_id, bool) or not isinstance(self.stage_id, int):
            raise SemanticError("trace_step.stage_id must be an integer")
        atom_ids = tuple(self.atom_ids)
        flow_ids = tuple(self.flow_ids)
        predecessors = tuple(self.semantic_predecessor_ids)
        if any(not isinstance(value, str) or not value for value in atom_ids):
            raise SemanticError("trace_step atom IDs are invalid")
        if any(not isinstance(value, str) or not value for value in flow_ids):
            raise SemanticError("trace_step flow IDs are invalid")
        if any(
            not isinstance(value, str) or not value for value in predecessors
        ):
            raise SemanticError("trace_step predecessor IDs are invalid")
        if len(atom_ids) != len(set(atom_ids)):
            raise SemanticError("trace_step atom IDs must be unique")
        if len(flow_ids) != len(set(flow_ids)):
            raise SemanticError("trace_step flow IDs must be unique")
        if len(predecessors) != len(set(predecessors)):
            raise SemanticError("trace_step predecessor IDs must be unique")
        object.__setattr__(self, "atom_ids", atom_ids)
        object.__setattr__(self, "flow_ids", flow_ids)
        object.__setattr__(self, "semantic_predecessor_ids", predecessors)
        members = frozenset(self.member_slice_ids)
        for member in members:
            _integer(member, "trace_step.member_slice_ids")
        object.__setattr__(self, "member_slice_ids", members)
        if self.endpoint_type is EndpointType.COPY:
            if (
                self.peer != -1
                or self.stage_id != -1
                or self.lane is not None
                or atom_ids
                or flow_ids
            ):
                raise SemanticError("copy trace metadata is inconsistent")
        elif (
            self.peer < 0
            or not isinstance(self.lane, LaneKey)
            or not atom_ids
            or not flow_ids
        ):
            raise SemanticError("communication trace metadata is incomplete")

    @property
    def key(self) -> Tuple[int, int, int]:
        return self.rank, self.tb_id, self.step_index


@dataclass(frozen=True)
class TraceSidecar:
    xml_sha256: str
    schedule_id: str
    rank_count: int
    entries: Mapping[Tuple[int, int, int], TraceStepMetadata]
    schema_version: int = TRACE_SIDECAR_VERSION

    def __post_init__(self) -> None:
        if (
            not isinstance(self.xml_sha256, str)
            or len(self.xml_sha256) != 64
            or any(value not in "0123456789abcdef" for value in self.xml_sha256)
        ):
            raise SemanticError("trace sidecar XML hash is invalid")
        _identifier(self.schedule_id, "trace_sidecar.schedule_id")
        _integer(self.rank_count, "trace_sidecar.rank_count", minimum=1)
        if self.schema_version != TRACE_SIDECAR_VERSION:
            raise SemanticError("trace sidecar schema version is unsupported")
        try:
            entries = dict(self.entries)
        except (TypeError, ValueError) as error:
            raise SemanticError("trace sidecar entries must be a mapping") from error
        if not entries:
            raise SemanticError("trace sidecar must contain at least one entry")
        for key, entry in entries.items():
            if not isinstance(entry, TraceStepMetadata) or key != entry.key:
                raise SemanticError("trace sidecar entry key is inconsistent")
            if entry.rank >= self.rank_count:
                raise SemanticError("trace sidecar entry rank is out of range")
        object.__setattr__(self, "entries", MappingProxyType(entries))

    def entry(self, rank: int, tb_id: int, step_index: int) -> TraceStepMetadata:
        key = (rank, tb_id, step_index)
        try:
            return self.entries[key]
        except KeyError as error:
            raise SemanticError(
                "trace record has no matching sidecar entry"
            ) from error

    def to_json_text(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "xml_sha256": self.xml_sha256,
            "schedule_id": self.schedule_id,
            "rank_count": self.rank_count,
            "entries": [
                _entry_payload(entry)
                for entry in sorted(self.entries.values(), key=lambda item: item.key)
            ],
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json_text(cls, text: str) -> "TraceSidecar":
        if not isinstance(text, str):
            raise SemanticError("trace sidecar JSON must be text")
        try:
            payload = json.loads(text)
        except (TypeError, ValueError) as error:
            raise SemanticError("trace sidecar JSON is invalid") from error
        if not isinstance(payload, dict) or not isinstance(
            payload.get("entries"), list
        ):
            raise SemanticError("trace sidecar JSON structure is invalid")
        entries = tuple(_entry_from_payload(value) for value in payload["entries"])
        entries_by_key = {}
        for entry in entries:
            if entry.key in entries_by_key:
                raise SemanticError("trace sidecar JSON contains duplicate keys")
            entries_by_key[entry.key] = entry
        return cls(
            xml_sha256=payload.get("xml_sha256"),
            schedule_id=payload.get("schedule_id"),
            rank_count=payload.get("rank_count"),
            entries=entries_by_key,
            schema_version=payload.get("schema_version"),
        )


def _entry_payload(entry: TraceStepMetadata) -> dict:
    return {
        "rank": entry.rank,
        "tb_id": entry.tb_id,
        "step_index": entry.step_index,
        "xml_step_index": entry.xml_step_index,
        "step_id": entry.step_id,
        "transfer_id": entry.transfer_id,
        "endpoint_type": entry.endpoint_type.value,
        "runtime_endpoint_type": entry.runtime_endpoint_type,
        "peer": entry.peer,
        "runtime_channel": entry.runtime_channel,
        "stage_id": entry.stage_id,
        "atom_ids": list(entry.atom_ids),
        "flow_ids": list(entry.flow_ids),
        "lane": (
            None
            if entry.lane is None
            else {
                "src_rank": entry.lane.src_rank,
                "dst_rank": entry.lane.dst_rank,
                "channel": entry.lane.channel,
            }
        ),
        "semantic_predecessor_ids": list(
            entry.semantic_predecessor_ids
        ),
        "member_slice_ids": sorted(entry.member_slice_ids),
    }


def _entry_from_payload(payload: object) -> TraceStepMetadata:
    if not isinstance(payload, dict):
        raise SemanticError("trace sidecar entry JSON is invalid")
    lane_payload = payload.get("lane")
    if lane_payload is None:
        lane = None
    elif isinstance(lane_payload, dict):
        try:
            lane = LaneKey(
                lane_payload["src_rank"],
                lane_payload["dst_rank"],
                lane_payload["channel"],
            )
        except KeyError as error:
            raise SemanticError("trace sidecar lane JSON is incomplete") from error
    else:
        raise SemanticError("trace sidecar lane JSON is invalid")
    try:
        endpoint_type = EndpointType(payload.get("endpoint_type"))
        return TraceStepMetadata(
            rank=payload.get("rank"),
            tb_id=payload.get("tb_id"),
            step_index=payload.get("step_index"),
            xml_step_index=payload.get("xml_step_index"),
            step_id=payload.get("step_id"),
            transfer_id=payload.get("transfer_id"),
            endpoint_type=endpoint_type,
            runtime_endpoint_type=payload.get("runtime_endpoint_type"),
            peer=payload.get("peer"),
            runtime_channel=payload.get("runtime_channel"),
            stage_id=payload.get("stage_id"),
            atom_ids=tuple(payload.get("atom_ids", ())),
            flow_ids=tuple(payload.get("flow_ids", ())),
            lane=lane,
            semantic_predecessor_ids=tuple(
                payload.get("semantic_predecessor_ids", ())
            ),
            member_slice_ids=frozenset(
                payload.get("member_slice_ids", ())
            ),
        )
    except (TypeError, ValueError) as error:
        raise SemanticError("trace sidecar entry JSON is invalid") from error


def build_trace_sidecar(artifact, schedule: Schedule) -> TraceSidecar:
    from vericcl.xml.lower import XmlArtifact

    if not isinstance(artifact, XmlArtifact):
        raise SemanticError("artifact must be an XmlArtifact")
    if not isinstance(schedule, Schedule):
        raise SemanticError("schedule must be a Schedule")
    if artifact.runtime_compatible is not True:
        raise SemanticError("trace sidecar requires a runtime-compatible XML")
    if set(artifact.endpoint_program.by_transfer_id) != {
        transfer.transfer_id for transfer in schedule.transfers
    }:
        raise SemanticError("artifact and schedule transfers differ")

    transfers = {
        transfer.transfer_id: transfer for transfer in schedule.transfers
    }
    flows_by_transfer = {transfer_id: [] for transfer_id in transfers}
    for flow in build_flow_index(schedule).flows:
        for transfer_id in flow.transfer_ids:
            flows_by_transfer[transfer_id].append(flow.flow_id)
    endpoints = {
        endpoint.endpoint_id: endpoint
        for endpoint in artifact.endpoint_program.endpoints
    }
    entries = {}
    for threadblock in artifact.tb_program.threadblocks:
        runtime_step_index = 0
        for xml_step_index, step in enumerate(threadblock.steps):
            if step.endpoint_id is None:
                continue
            endpoint = endpoints.get(step.endpoint_id)
            if endpoint is None:
                raise SemanticError("trace sidecar endpoint is missing")
            if step.xml_type is EndpointType.COPY:
                lane = None
                atom_ids = ()
                flow_ids = ()
                stage_id = -1
                runtime_channel = 0
            else:
                transfer = transfers.get(step.transfer_id)
                if transfer is None:
                    raise SemanticError("trace sidecar transfer is missing")
                lane = LaneKey(
                    transfer.src_rank,
                    transfer.dst_rank,
                    transfer.channel,
                )
                atom_ids = tuple(
                    "{}:atom-s{:08d}".format(
                        transfer.transfer_id,
                        atom.slice_id,
                    )
                    for atom in transfer.atoms
                )
                flow_ids = tuple(sorted(set(flows_by_transfer[step.transfer_id])))
                if not flow_ids:
                    raise SemanticError("trace sidecar transfer has no flow")
                stage_id = transfer.stage_id
                runtime_channel = transfer.channel
            entry = TraceStepMetadata(
                rank=step.rank,
                tb_id=threadblock.tb_id,
                step_index=runtime_step_index,
                xml_step_index=xml_step_index,
                step_id=step.step_id,
                transfer_id=step.transfer_id,
                endpoint_type=step.xml_type,
                runtime_endpoint_type=_RUNTIME_ENDPOINT_TYPES[step.xml_type],
                peer=step.peer,
                runtime_channel=runtime_channel,
                stage_id=stage_id,
                atom_ids=atom_ids,
                flow_ids=flow_ids,
                lane=lane,
                semantic_predecessor_ids=tuple(
                    step.semantic_predecessor_node_ids
                ),
                member_slice_ids=step.member_slice_ids,
            )
            if entry.key in entries:
                raise SemanticError("trace sidecar runtime key is duplicated")
            entries[entry.key] = entry
            runtime_step_index += 1
    return TraceSidecar(
        xml_sha256=artifact.sha256,
        schedule_id=schedule.schedule_id,
        rank_count=schedule.rank_count,
        entries=entries,
    )


def write_trace_sidecar(sidecar: TraceSidecar, path: Path) -> None:
    if not isinstance(sidecar, TraceSidecar):
        raise SemanticError("sidecar must be a TraceSidecar")
    try:
        output = Path(path)
    except TypeError as error:
        raise SemanticError("trace sidecar path is invalid") from error
    output.write_text(sidecar.to_json_text() + "\n", encoding="utf-8")


def load_trace_sidecar(path: Path) -> TraceSidecar:
    try:
        source = Path(path)
    except TypeError as error:
        raise SemanticError("trace sidecar path is invalid") from error
    return TraceSidecar.from_json_text(source.read_text(encoding="utf-8"))
