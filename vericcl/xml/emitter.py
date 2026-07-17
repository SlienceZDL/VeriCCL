from __future__ import annotations

from typing import Dict, Tuple

from lxml import etree

from vericcl.errors import SemanticError
from vericcl.input.models import ResolvedInput
from vericcl.semantics.collective import CollectiveKind
from vericcl.xml.endpoints import EndpointType
from vericcl.xml.model import BufferPlan, PhysicalRef
from vericcl.xml.threadblocks import ThreadblockProgram, XmlStep


def _geometry(inputs: ResolvedInput) -> Tuple[int, int]:
    rank_count = inputs.rank_count
    slice_count = inputs.hyperparameters.slice_count
    if inputs.collective.kind is CollectiveKind.ALL_GATHER:
        return rank_count * slice_count, (
            rank_count * inputs.hyperparameters.total_size_bytes
        )
    return slice_count, inputs.hyperparameters.total_size_bytes


def _communication_refs(
    program: ThreadblockProgram,
) -> Dict[str, Tuple[PhysicalRef, PhysicalRef]]:
    result = {}
    for transfer_id, step_ids in program.transfer_steps.items():
        steps = tuple(program.steps_by_id[step_id] for step_id in step_ids)
        source = next((step.src_ref for step in steps if step.src_ref), None)
        destination = next((step.dst_ref for step in steps if step.dst_ref), None)
        if source is None or destination is None:
            raise SemanticError("communication transfer references are incomplete")
        result[transfer_id] = source, destination
    return result


def _step_refs(
    step: XmlStep,
    communication_refs: Dict[str, Tuple[PhysicalRef, PhysicalRef]],
    buffers: BufferPlan,
) -> Tuple[str, int, str, int]:
    if step.xml_type is EndpointType.NOP:
        return "i", -1, "o", -1
    if step.xml_type is EndpointType.COPY:
        if step.src_ref is None or step.dst_ref is None:
            raise SemanticError("copy step references are incomplete")
        return (
            step.src_ref.buffer,
            step.src_ref.offset,
            step.dst_ref.buffer,
            step.dst_ref.offset,
        )
    source, destination = communication_refs[step.transfer_id]
    if step.xml_type is EndpointType.RECV_REDUCE_COPY:
        source = buffers.transfer_accumulator_refs.get(step.transfer_id)
        if source is None:
            raise SemanticError("reduction accumulator reference is missing")
    return (
        source.buffer,
        source.offset,
        destination.buffer,
        destination.offset,
    )


def emit_xml(
    program: ThreadblockProgram,
    buffers: BufferPlan,
    inputs: ResolvedInput,
) -> str:
    if not isinstance(program, ThreadblockProgram):
        raise SemanticError("program must be a ThreadblockProgram")
    if not isinstance(buffers, BufferPlan):
        raise SemanticError("buffers must be a BufferPlan")
    if not isinstance(inputs, ResolvedInput):
        raise SemanticError("inputs must be a ResolvedInput")
    rank_count = inputs.rank_count
    expected_ranks = set(range(rank_count))
    for field in (buffers.i_chunks, buffers.o_chunks, buffers.s_chunks):
        if set(field) != expected_ranks:
            raise SemanticError("buffer chunk declarations must cover every rank")
    if any(tb.key.rank >= rank_count for tb in program.threadblocks):
        raise SemanticError("threadblock rank is outside the input rank range")

    nchunks_per_loop, runtime_bytes = _geometry(inputs)
    communication_channels = [
        tb.key.channel
        for tb in program.threadblocks
        if tb.key.direction != "copy"
    ]
    nchannels = max(communication_channels, default=0) + 1
    root = etree.Element("algo")
    root.attrib.update(
        {
            "name": "vericcl",
            "nchannels": str(nchannels),
            "nchunksperloop": str(nchunks_per_loop),
            "proto": "Simple",
            "coll": inputs.collective.kind.value,
            "inplace": "1" if inputs.collective.inplace else "0",
            "redop": "nop",
            "ngpus": str(rank_count),
            "minBytes": str(runtime_bytes),
            "maxBytes": str(runtime_bytes + 1),
        }
    )
    step_locations = {
        step.step_id: (threadblock.tb_id, index)
        for threadblock in program.threadblocks
        for index, step in enumerate(threadblock.steps)
    }
    communication_refs = _communication_refs(program)
    blocks_by_rank = {
        rank: sorted(
            (
                tb
                for tb in program.threadblocks
                if tb.key.rank == rank
            ),
            key=lambda tb: tb.tb_id,
        )
        for rank in range(rank_count)
    }
    for rank in range(rank_count):
        gpu = etree.SubElement(
            root,
            "gpu",
            id=str(rank),
            i_chunks=str(buffers.i_chunks[rank]),
            o_chunks=str(buffers.o_chunks[rank]),
            s_chunks=str(buffers.s_chunks[rank]),
        )
        for threadblock in blocks_by_rank[rank]:
            serialized_channel = (
                0
                if threadblock.key.direction == "copy"
                else threadblock.key.channel
            )
            tb = etree.SubElement(
                gpu,
                "tb",
                id=str(threadblock.tb_id),
                send=str(threadblock.send_peer),
                recv=str(threadblock.recv_peer),
                chan=str(serialized_channel),
            )
            for index, step in enumerate(threadblock.steps):
                srcbuf, srcoff, dstbuf, dstoff = _step_refs(
                    step,
                    communication_refs,
                    buffers,
                )
                depid, deps = (-1, -1)
                if step.dependency_step_id is not None:
                    depid, deps = step_locations[step.dependency_step_id]
                etree.SubElement(
                    tb,
                    "step",
                    s=str(index),
                    type=step.xml_type.value,
                    srcbuf=srcbuf,
                    srcoff=str(srcoff),
                    dstbuf=dstbuf,
                    dstoff=str(dstoff),
                    cnt="1",
                    depid=str(depid),
                    deps=str(deps),
                    hasdep="1" if step.has_dependence else "0",
                )
    etree.indent(root, space="  ")
    return etree.tostring(root, encoding="unicode") + "\n"
