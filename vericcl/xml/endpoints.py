from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Optional, Tuple

from vericcl.errors import SemanticError
from vericcl.semantics.atom import Atom, Schedule
from vericcl.xml.model import BufferPlan, PhysicalRef


class EndpointType(str, Enum):
    SEND = "s"
    RECV = "r"
    RECV_REDUCE_COPY = "rrc"
    COPY = "cpy"
    NOP = "nop"


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise SemanticError("{} must be a non-empty string".format(field))
    return value


def _integer(value: object, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SemanticError("{} must be an integer".format(field))
    if value < minimum:
        raise SemanticError("{} must be at least {}".format(field, minimum))
    return value


def _time(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SemanticError("{} must be a number".format(field))
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise SemanticError("{} must be finite and non-negative".format(field))
    return normalized


@dataclass(frozen=True)
class EndpointAtom:
    endpoint_id: str
    atom: Optional[Atom]
    member_atoms: Tuple[Atom, ...]
    transfer_id: str
    transfer_kind: str
    xml_type: EndpointType
    rank: int
    peer: int
    channel: int
    stage_id: int
    member_slice_ids: frozenset[int]
    src_ref: Optional[PhysicalRef]
    dst_ref: Optional[PhysicalRef]
    st_time: float
    ed_time: float

    def __post_init__(self) -> None:
        _identifier(self.endpoint_id, "endpoint.endpoint_id")
        _identifier(self.transfer_id, "endpoint.transfer_id")
        if not isinstance(self.xml_type, EndpointType):
            raise SemanticError("endpoint.xml_type must be an EndpointType")
        _integer(self.rank, "endpoint.rank")
        object.__setattr__(
            self,
            "st_time",
            _time(self.st_time, "endpoint.st_time"),
        )
        object.__setattr__(
            self,
            "ed_time",
            _time(self.ed_time, "endpoint.ed_time"),
        )
        if self.st_time > self.ed_time:
            raise SemanticError("endpoint start must not exceed end")
        atoms = tuple(self.member_atoms)
        members = frozenset(self.member_slice_ids)
        object.__setattr__(self, "member_atoms", atoms)
        object.__setattr__(self, "member_slice_ids", members)
        if self.xml_type is EndpointType.COPY:
            if self.atom is not None or atoms or members:
                raise SemanticError("copy endpoint must not contain transfer atoms")
            if self.peer != -1 or self.channel != -1 or self.stage_id != -1:
                raise SemanticError("copy endpoint must use local sentinel fields")
            if self.src_ref is None or self.dst_ref is None:
                raise SemanticError("copy endpoint requires both references")
            if self.src_ref.rank != self.rank or self.dst_ref.rank != self.rank:
                raise SemanticError("copy endpoint references use an incorrect rank")
            return
        if self.transfer_kind not in {"SEND", "REDUCE"}:
            raise SemanticError("communication endpoint kind is invalid")
        _integer(self.peer, "endpoint.peer")
        _integer(self.channel, "endpoint.channel")
        _integer(self.stage_id, "endpoint.stage_id")
        if self.peer == self.rank:
            raise SemanticError("communication endpoint peer equals its rank")
        if not atoms or self.atom != atoms[0]:
            raise SemanticError("communication endpoint requires canonical atoms")
        if frozenset(atom.slice_id for atom in atoms) != members:
            raise SemanticError("endpoint atoms do not match member slices")
        if self.xml_type is EndpointType.SEND:
            if self.src_ref is None or self.dst_ref is not None:
                raise SemanticError("send endpoint references are invalid")
            if self.src_ref.rank != self.rank:
                raise SemanticError("send source reference uses an incorrect rank")
        elif self.xml_type in {
            EndpointType.RECV,
            EndpointType.RECV_REDUCE_COPY,
        }:
            if self.src_ref is not None or self.dst_ref is None:
                raise SemanticError("receive endpoint references are invalid")
            if self.dst_ref.rank != self.rank:
                raise SemanticError(
                    "receive destination reference uses an incorrect rank"
                )
        else:
            raise SemanticError("unsupported communication endpoint type")


@dataclass(frozen=True)
class EndpointProgram:
    endpoints: Tuple[EndpointAtom, ...]
    by_transfer_id: Mapping[str, Tuple[EndpointAtom, EndpointAtom]]
    local_endpoints: Mapping[str, EndpointAtom]

    def __post_init__(self) -> None:
        endpoints = tuple(self.endpoints)
        if not all(isinstance(endpoint, EndpointAtom) for endpoint in endpoints):
            raise SemanticError("endpoint program contains an invalid endpoint")
        endpoint_ids = tuple(endpoint.endpoint_id for endpoint in endpoints)
        if len(endpoint_ids) != len(set(endpoint_ids)):
            raise SemanticError("endpoint IDs must be unique")
        object.__setattr__(self, "endpoints", endpoints)
        pairs = {
            transfer_id: tuple(pair)
            for transfer_id, pair in self.by_transfer_id.items()
        }
        for transfer_id, pair in pairs.items():
            _validate_pair(transfer_id, pair)
        object.__setattr__(self, "by_transfer_id", MappingProxyType(pairs))
        local = dict(self.local_endpoints)
        if set(pairs).intersection(local):
            raise SemanticError("transfer and local endpoint IDs must be disjoint")
        for copy_id, endpoint in local.items():
            if (
                endpoint.transfer_id != copy_id
                or endpoint.xml_type is not EndpointType.COPY
            ):
                raise SemanticError("local endpoint mapping is invalid")
        object.__setattr__(self, "local_endpoints", MappingProxyType(local))
        expected = {
            endpoint.endpoint_id
            for pair in pairs.values()
            for endpoint in pair
        } | {endpoint.endpoint_id for endpoint in local.values()}
        if set(endpoint_ids) != expected:
            raise SemanticError("endpoint program mappings do not cover endpoints")


def _validate_pair(
    transfer_id: str,
    pair: Tuple[EndpointAtom, ...],
) -> None:
    if len(pair) != 2:
        raise SemanticError("physical transfer must have exactly two endpoints")
    send = next(
        (endpoint for endpoint in pair if endpoint.xml_type is EndpointType.SEND),
        None,
    )
    recv = next(
        (
            endpoint
            for endpoint in pair
            if endpoint.xml_type
            in {EndpointType.RECV, EndpointType.RECV_REDUCE_COPY}
        ),
        None,
    )
    if send is None or recv is None:
        raise SemanticError("physical transfer endpoint types are incompatible")
    if send.transfer_id != transfer_id or recv.transfer_id != transfer_id:
        raise SemanticError("endpoint pair transfer IDs differ")
    if send.rank != recv.peer or recv.rank != send.peer:
        raise SemanticError("endpoint pair ranks are not opposite")
    comparable = (
        send.transfer_kind,
        send.channel,
        send.stage_id,
        send.member_slice_ids,
        send.member_atoms,
        send.st_time,
        send.ed_time,
    )
    if comparable != (
        recv.transfer_kind,
        recv.channel,
        recv.stage_id,
        recv.member_slice_ids,
        recv.member_atoms,
        recv.st_time,
        recv.ed_time,
    ):
        raise SemanticError("endpoint pair metadata differs")
    expected_recv = (
        EndpointType.RECV_REDUCE_COPY
        if send.transfer_kind == "REDUCE"
        else EndpointType.RECV
    )
    if recv.xml_type is not expected_recv:
        raise SemanticError("endpoint pair receive type is incorrect")


def lower_endpoints(
    schedule: Schedule,
    buffers: BufferPlan,
) -> EndpointProgram:
    if not isinstance(schedule, Schedule):
        raise SemanticError("schedule must be a Schedule")
    if not isinstance(buffers, BufferPlan):
        raise SemanticError("buffers must be a BufferPlan")
    pairs = {}
    endpoints = []
    for transfer in schedule.transfers:
        src_ref = buffers.transfer_src_refs.get(transfer.transfer_id)
        if src_ref is None:
            raise SemanticError("transfer source reference is missing")
        dst_ref = buffers.transfer_dst_refs.get(transfer.transfer_id)
        if dst_ref is None:
            raise SemanticError("transfer destination reference is missing")
        atoms = tuple(transfer.atoms)
        send = EndpointAtom(
            endpoint_id="{}:s".format(transfer.transfer_id),
            atom=atoms[0],
            member_atoms=atoms,
            transfer_id=transfer.transfer_id,
            transfer_kind=transfer.kind,
            xml_type=EndpointType.SEND,
            rank=transfer.src_rank,
            peer=transfer.dst_rank,
            channel=transfer.channel,
            stage_id=transfer.stage_id,
            member_slice_ids=transfer.member_slice_ids,
            src_ref=src_ref,
            dst_ref=None,
            st_time=transfer.st_time,
            ed_time=transfer.ed_time,
        )
        recv = EndpointAtom(
            endpoint_id="{}:recv".format(transfer.transfer_id),
            atom=atoms[0],
            member_atoms=atoms,
            transfer_id=transfer.transfer_id,
            transfer_kind=transfer.kind,
            xml_type=(
                EndpointType.RECV_REDUCE_COPY
                if transfer.kind == "REDUCE"
                else EndpointType.RECV
            ),
            rank=transfer.dst_rank,
            peer=transfer.src_rank,
            channel=transfer.channel,
            stage_id=transfer.stage_id,
            member_slice_ids=transfer.member_slice_ids,
            src_ref=None,
            dst_ref=dst_ref,
            st_time=transfer.st_time,
            ed_time=transfer.ed_time,
        )
        pair = (send, recv)
        _validate_pair(transfer.transfer_id, pair)
        pairs[transfer.transfer_id] = pair
        endpoints.extend(pair)
    local = {}
    for copy in buffers.local_copies:
        endpoint = EndpointAtom(
            endpoint_id="{}:copy".format(copy.copy_id),
            atom=None,
            member_atoms=(),
            transfer_id=copy.copy_id,
            transfer_kind="COPY",
            xml_type=EndpointType.COPY,
            rank=copy.rank,
            peer=-1,
            channel=-1,
            stage_id=-1,
            member_slice_ids=frozenset(),
            src_ref=copy.src_ref,
            dst_ref=copy.dst_ref,
            st_time=copy.st_time,
            ed_time=copy.ed_time,
        )
        local[copy.copy_id] = endpoint
        endpoints.append(endpoint)
    return EndpointProgram(tuple(endpoints), pairs, local)
