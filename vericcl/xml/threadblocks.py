from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional, Tuple

from vericcl.errors import SemanticError
from vericcl.xml.endpoints import EndpointType
from vericcl.xml.model import PhysicalRef


@dataclass(frozen=True, order=True)
class ThreadblockKey:
    rank: int
    direction: str
    peer: int
    channel: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.rank, bool)
            or not isinstance(self.rank, int)
            or self.rank < 0
        ):
            raise SemanticError("threadblock rank must be a non-negative integer")
        if self.direction not in {"send", "recv", "copy"}:
            raise SemanticError("threadblock direction is invalid")
        if self.direction == "copy":
            if self.peer != -1 or self.channel != -1:
                raise SemanticError("copy threadblock must use local sentinels")
        elif self.peer < 0 or self.channel < 0 or self.peer == self.rank:
            raise SemanticError("communication threadblock key is invalid")


@dataclass(frozen=True)
class XmlStep:
    step_id: str
    node_id: str
    transfer_id: str
    endpoint_id: Optional[str]
    xml_type: EndpointType
    rank: int
    peer: int
    channel: int
    src_ref: Optional[PhysicalRef]
    dst_ref: Optional[PhysicalRef]
    dependency_step_id: Optional[str]
    has_dependence: bool
    semantic_predecessor_node_ids: Tuple[str, ...]
    member_slice_ids: frozenset[int]
    solver_st_time: float
    solver_ed_time: float
    effective_st_time: float
    effective_ed_time: float

    def __post_init__(self) -> None:
        for value, field in (
            (self.step_id, "step_id"),
            (self.node_id, "node_id"),
            (self.transfer_id, "transfer_id"),
        ):
            if not isinstance(value, str) or not value:
                raise SemanticError("{} must be a non-empty string".format(field))
        if not isinstance(self.xml_type, EndpointType):
            raise SemanticError("step xml_type must be an EndpointType")
        if not isinstance(self.has_dependence, bool):
            raise SemanticError("step has_dependence must be a boolean")
        predecessors = tuple(self.semantic_predecessor_node_ids)
        object.__setattr__(self, "semantic_predecessor_node_ids", predecessors)
        object.__setattr__(self, "member_slice_ids", frozenset(self.member_slice_ids))
        if self.xml_type is EndpointType.NOP:
            if (
                self.endpoint_id is not None
                or self.src_ref is not None
                or self.dst_ref is not None
            ):
                raise SemanticError(
                    "NOP must not contain an endpoint or buffer reference"
                )
        elif self.endpoint_id is None:
            raise SemanticError("non-NOP step requires an endpoint ID")


@dataclass(frozen=True)
class Threadblock:
    key: ThreadblockKey
    tb_id: int
    steps: Tuple[XmlStep, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.key, ThreadblockKey):
            raise SemanticError("threadblock key is invalid")
        if (
            isinstance(self.tb_id, bool)
            or not isinstance(self.tb_id, int)
            or self.tb_id < 0
        ):
            raise SemanticError("threadblock ID must be a non-negative integer")
        steps = tuple(self.steps)
        if not all(isinstance(step, XmlStep) for step in steps):
            raise SemanticError("threadblock contains an invalid step")
        if any(step.rank != self.key.rank for step in steps):
            raise SemanticError("threadblock step rank is incorrect")
        if any(
            step.peer != self.key.peer or step.channel != self.key.channel
            for step in steps
        ):
            raise SemanticError("threadblock step lane is incorrect")
        allowed = {
            "copy": {EndpointType.COPY, EndpointType.NOP},
            "send": {EndpointType.SEND, EndpointType.NOP},
            "recv": {
                EndpointType.RECV,
                EndpointType.RECV_REDUCE_COPY,
                EndpointType.NOP,
            },
        }[self.key.direction]
        if any(step.xml_type not in allowed for step in steps):
            raise SemanticError("threadblock step direction is incorrect")
        object.__setattr__(self, "steps", steps)

    @property
    def send_peer(self) -> int:
        return self.key.peer if self.key.direction == "send" else -1

    @property
    def recv_peer(self) -> int:
        return self.key.peer if self.key.direction == "recv" else -1


@dataclass(frozen=True)
class ThreadblockProgram:
    threadblocks: Tuple[Threadblock, ...]
    steps_by_id: Mapping[str, XmlStep]
    transfer_steps: Mapping[str, Tuple[str, str]]
    node_steps: Mapping[str, Tuple[str, ...]]
    referenced_step_ids: frozenset[str]
    inversion_count: int

    def __post_init__(self) -> None:
        threadblocks = tuple(self.threadblocks)
        if not all(isinstance(tb, Threadblock) for tb in threadblocks):
            raise SemanticError("threadblock program contains an invalid block")
        block_ids = [(tb.key.rank, tb.tb_id) for tb in threadblocks]
        if len(block_ids) != len(set(block_ids)):
            raise SemanticError("threadblock IDs must be unique per rank")
        object.__setattr__(self, "threadblocks", threadblocks)
        actual_steps = {
            step.step_id: step for tb in threadblocks for step in tb.steps
        }
        if len(actual_steps) != sum(len(tb.steps) for tb in threadblocks):
            raise SemanticError("step IDs must be unique")
        if dict(self.steps_by_id) != actual_steps:
            raise SemanticError("steps_by_id does not match threadblocks")
        object.__setattr__(self, "steps_by_id", MappingProxyType(actual_steps))
        transfer_steps = {
            transfer_id: tuple(step_ids)
            for transfer_id, step_ids in self.transfer_steps.items()
        }
        for step_ids in transfer_steps.values():
            if len(step_ids) != 2 or not set(step_ids) <= set(actual_steps):
                raise SemanticError("physical transfer must map to two steps")
        object.__setattr__(
            self,
            "transfer_steps",
            MappingProxyType(transfer_steps),
        )
        node_steps = {
            node_id: tuple(step_ids) for node_id, step_ids in self.node_steps.items()
        }
        if any(
            not set(step_ids) <= set(actual_steps)
            for step_ids in node_steps.values()
        ):
            raise SemanticError("node step mapping references a missing step")
        object.__setattr__(self, "node_steps", MappingProxyType(node_steps))
        referenced = frozenset(self.referenced_step_ids)
        if not referenced <= set(actual_steps):
            raise SemanticError("referenced step is missing")
        for step in actual_steps.values():
            if step.dependency_step_id is not None:
                dependency = actual_steps.get(step.dependency_step_id)
                if dependency is None:
                    raise SemanticError("step dependency is missing")
                if dependency.rank != step.rank:
                    raise SemanticError("XML dependency crosses ranks")
            if step.has_dependence != (step.step_id in referenced):
                raise SemanticError("step completion flag metadata is inconsistent")
        object.__setattr__(self, "referenced_step_ids", referenced)
        if (
            isinstance(self.inversion_count, bool)
            or not isinstance(self.inversion_count, int)
            or self.inversion_count < 0
        ):
            raise SemanticError("inversion_count must be a non-negative integer")

    def threadblock_for_step(self, step_id: str) -> Threadblock:
        if step_id not in self.steps_by_id:
            raise SemanticError("unknown step ID")
        return next(
            tb
            for tb in self.threadblocks
            if any(step.step_id == step_id for step in tb.steps)
        )
