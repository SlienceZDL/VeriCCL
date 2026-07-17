from __future__ import annotations

import hashlib
from dataclasses import dataclass

from vericcl.errors import SemanticError
from vericcl.input.models import ResolvedInput
from vericcl.semantics.atom import Schedule
from vericcl.topology.model import LinkKey, Topology
from vericcl.xml.buffers import build_buffer_plan
from vericcl.xml.deadlock import simulate_endpoint_execution
from vericcl.xml.dependencies import build_transfer_dag
from vericcl.xml.emitter import emit_xml
from vericcl.xml.endpoints import EndpointProgram, lower_endpoints
from vericcl.xml.liveness import verify_buffer_liveness
from vericcl.xml.list_scheduler import schedule_threadblocks
from vericcl.xml.model import BufferPlan
from vericcl.xml.parser import validate_xml
from vericcl.xml.threadblocks import ThreadblockProgram


@dataclass(frozen=True)
class XmlArtifact:
    xml_text: str
    buffer_plan: BufferPlan
    endpoint_program: EndpointProgram
    tb_program: ThreadblockProgram
    sha256: str
    runtime_compatible: bool

    def __post_init__(self) -> None:
        if not isinstance(self.xml_text, str) or not self.xml_text:
            raise SemanticError("artifact xml_text must be a non-empty string")
        if not isinstance(self.buffer_plan, BufferPlan):
            raise SemanticError("artifact buffer_plan is invalid")
        if not isinstance(self.endpoint_program, EndpointProgram):
            raise SemanticError("artifact endpoint_program is invalid")
        if not isinstance(self.tb_program, ThreadblockProgram):
            raise SemanticError("artifact tb_program is invalid")
        expected = hashlib.sha256(self.xml_text.encode("utf-8")).hexdigest()
        if self.sha256 != expected:
            raise SemanticError("artifact sha256 does not match xml_text")
        if not isinstance(self.runtime_compatible, bool):
            raise SemanticError("artifact runtime_compatible must be a boolean")


def lower_to_xml(
    schedule: Schedule,
    inputs: ResolvedInput,
    topology: Topology,
) -> XmlArtifact:
    if not isinstance(schedule, Schedule):
        raise SemanticError("schedule must be a Schedule")
    if not isinstance(inputs, ResolvedInput):
        raise SemanticError("inputs must be a ResolvedInput")
    if not isinstance(topology, Topology):
        raise SemanticError("topology must be a Topology")
    if (
        schedule.rank_count != inputs.rank_count
        or topology.rank_count != inputs.rank_count
    ):
        raise SemanticError("schedule, inputs, and topology rank counts differ")
    if schedule.slice_count != inputs.hyperparameters.slice_count:
        raise SemanticError("schedule and input slice counts differ")
    for transfer in schedule.transfers:
        link = topology.links.get(LinkKey(transfer.src_rank, transfer.dst_rank))
        if link is None:
            raise SemanticError("schedule transfer uses a missing topology link")
        if transfer.channel >= link.max_channels:
            raise SemanticError("schedule transfer channel exceeds its topology link")

    buffers = build_buffer_plan(schedule, inputs)
    verify_buffer_liveness(schedule, buffers, inputs)
    endpoints = lower_endpoints(schedule, buffers)
    dag = build_transfer_dag(endpoints, schedule, buffers)
    threadblocks = schedule_threadblocks(endpoints, dag)
    deadlock = simulate_endpoint_execution(threadblocks)
    if deadlock.deadlocked:
        raise SemanticError("threadblock program is deadlocked")
    xml_text = emit_xml(threadblocks, buffers, inputs)
    validate_xml(xml_text, threadblocks, buffers, inputs)
    return XmlArtifact(
        xml_text=xml_text,
        buffer_plan=buffers,
        endpoint_program=endpoints,
        tb_program=threadblocks,
        sha256=hashlib.sha256(xml_text.encode("utf-8")).hexdigest(),
        runtime_compatible=True,
    )
