from __future__ import annotations

from typing import Dict, Tuple

from lxml import etree

from vericcl.errors import SemanticError
from vericcl.input.models import ResolvedInput
from vericcl.semantics.collective import CollectiveKind
from vericcl.xml.endpoints import EndpointType
from vericcl.xml.model import BufferPlan, PhysicalRef
from vericcl.xml.threadblocks import ThreadblockProgram


_STEP_ATTRIBUTES = {
    "s",
    "type",
    "srcbuf",
    "srcoff",
    "dstbuf",
    "dstoff",
    "cnt",
    "depid",
    "deps",
    "hasdep",
}
_KNOWN_TYPES = {value.value for value in EndpointType}


def _parse(xml_text: str) -> etree._Element:
    if not isinstance(xml_text, str) or not xml_text:
        raise SemanticError("xml_text must be a non-empty string")
    parser = etree.XMLParser(
        resolve_entities=False,
        no_network=True,
        remove_blank_text=True,
    )
    try:
        root = etree.fromstring(xml_text.encode("utf-8"), parser=parser)
    except (etree.XMLSyntaxError, ValueError) as error:
        raise SemanticError("XML is not well formed") from error
    if root.tag != "algo":
        raise SemanticError("XML root must be algo")
    return root


def _integer(element, name: str) -> int:
    value = element.attrib.get(name)
    if value is None:
        raise SemanticError("XML attribute {} is missing".format(name))
    try:
        return int(value)
    except ValueError as error:
        raise SemanticError(
            "XML attribute {} must be an integer".format(name)
        ) from error


def _expected_geometry(inputs: ResolvedInput) -> Tuple[int, int]:
    slices = inputs.hyperparameters.slice_count
    if inputs.collective.kind is CollectiveKind.ALL_GATHER:
        return (
            inputs.rank_count * slices,
            inputs.rank_count * inputs.hyperparameters.total_size_bytes,
        )
    return slices, inputs.hyperparameters.total_size_bytes


def _buffer_limit(buffers: BufferPlan, rank: int, buffer: str) -> int:
    values = {
        "i": buffers.i_chunks,
        "o": buffers.o_chunks,
        "s": buffers.s_chunks,
    }
    if buffer not in values:
        raise SemanticError("XML step uses an unknown buffer")
    return values[buffer][rank]


def _sidecar_refs(
    program: ThreadblockProgram,
) -> Dict[str, Tuple[PhysicalRef, PhysicalRef]]:
    references = {}
    for transfer_id, step_ids in program.transfer_steps.items():
        steps = tuple(program.steps_by_id[step_id] for step_id in step_ids)
        source = next((step.src_ref for step in steps if step.src_ref), None)
        destination = next((step.dst_ref for step in steps if step.dst_ref), None)
        if source is None or destination is None:
            raise SemanticError("XML transfer sidecar references are incomplete")
        references[transfer_id] = source, destination
    return references


def _expected_step_refs(
    step,
    sidecar_refs,
    buffers: BufferPlan,
) -> Tuple[str, int, str, int]:
    if step.xml_type is EndpointType.NOP:
        return "i", -1, "o", -1
    if step.xml_type is EndpointType.COPY:
        if step.src_ref is None or step.dst_ref is None:
            raise SemanticError("XML copy sidecar references are incomplete")
        source, destination = step.src_ref, step.dst_ref
    else:
        source, destination = sidecar_refs[step.transfer_id]
        if step.xml_type is EndpointType.RECV_REDUCE_COPY:
            source = buffers.transfer_accumulator_refs.get(step.transfer_id)
            if source is None:
                raise SemanticError(
                    "XML reduction accumulator sidecar reference is missing"
                )
    return source.buffer, source.offset, destination.buffer, destination.offset


def validate_xml(
    xml_text: str,
    program: ThreadblockProgram,
    buffers: BufferPlan,
    inputs: ResolvedInput,
) -> etree._Element:
    if not isinstance(program, ThreadblockProgram):
        raise SemanticError("program must be a ThreadblockProgram")
    if not isinstance(buffers, BufferPlan):
        raise SemanticError("buffers must be a BufferPlan")
    if not isinstance(inputs, ResolvedInput):
        raise SemanticError("inputs must be a ResolvedInput")
    root = _parse(xml_text)
    expected_chunks, runtime_bytes = _expected_geometry(inputs)
    expected_root = {
        "name": "vericcl",
        "nchunksperloop": str(expected_chunks),
        "proto": "Simple",
        "coll": inputs.collective.kind.value,
        "inplace": "1" if inputs.collective.inplace else "0",
        "redop": "nop",
        "ngpus": str(inputs.rank_count),
        "minBytes": str(runtime_bytes),
        "maxBytes": str(runtime_bytes + 1),
    }
    for name, value in expected_root.items():
        if root.attrib.get(name) != value:
            raise SemanticError("XML root attribute {} is incorrect".format(name))
    nchannels = _integer(root, "nchannels")
    if nchannels < 1:
        raise SemanticError("XML nchannels must be positive")
    if root.xpath(".//copy"):
        raise SemanticError("XML must not contain a top-level copy")
    if any(child.tag != "gpu" for child in root):
        raise SemanticError("XML algo contains an unsupported child")

    gpu_nodes = root.xpath("./gpu")
    gpu_ids = [_integer(gpu, "id") for gpu in gpu_nodes]
    if gpu_ids != list(range(inputs.rank_count)):
        raise SemanticError("XML GPU IDs must be sorted and contiguous")
    program_blocks = {
        (tb.key.rank, tb.tb_id): tb for tb in program.threadblocks
    }
    represented_steps: Dict[str, Tuple[int, int, int]] = {}
    coordinates = {}
    sidecar_refs = _sidecar_refs(program)
    maximum_channel = 0
    for gpu in gpu_nodes:
        rank = _integer(gpu, "id")
        expected_counts = {
            "i_chunks": buffers.i_chunks[rank],
            "o_chunks": buffers.o_chunks[rank],
            "s_chunks": buffers.s_chunks[rank],
        }
        if any(
            _integer(gpu, name) != value
            for name, value in expected_counts.items()
        ):
            raise SemanticError("XML GPU chunk declaration is incorrect")
        if any(child.tag != "tb" for child in gpu):
            raise SemanticError("XML GPU contains an unsupported child")
        tb_nodes = gpu.xpath("./tb")
        tb_ids = [_integer(tb, "id") for tb in tb_nodes]
        if tb_ids != list(range(len(tb_ids))):
            raise SemanticError("XML TB IDs must be contiguous per rank")
        for tb_node in tb_nodes:
            tb_id = _integer(tb_node, "id")
            threadblock = program_blocks.get((rank, tb_id))
            if threadblock is None:
                raise SemanticError("XML contains an unknown threadblock")
            send = _integer(tb_node, "send")
            recv = _integer(tb_node, "recv")
            channel = _integer(tb_node, "chan")
            expected_channel = (
                0
                if threadblock.key.direction == "copy"
                else threadblock.key.channel
            )
            if (
                send != threadblock.send_peer
                or recv != threadblock.recv_peer
                or channel != expected_channel
            ):
                raise SemanticError("XML threadblock lane metadata is incorrect")
            if channel < 0 or channel >= nchannels:
                raise SemanticError("XML threadblock channel is out of range")
            maximum_channel = max(maximum_channel, channel)
            step_nodes = tb_node.xpath("./step")
            if len(step_nodes) != len(threadblock.steps):
                raise SemanticError("XML threadblock step count is incorrect")
            for index, (node, step) in enumerate(
                zip(step_nodes, threadblock.steps)
            ):
                if set(node.attrib) != _STEP_ATTRIBUTES:
                    raise SemanticError("XML step attributes are incomplete")
                if _integer(node, "s") != index:
                    raise SemanticError("XML step IDs must be contiguous")
                if node.attrib["type"] not in _KNOWN_TYPES:
                    raise SemanticError("XML step type is unsupported")
                if node.attrib["type"] != step.xml_type.value:
                    raise SemanticError("XML step type differs from the sidecar")
                if _integer(node, "cnt") != 1:
                    raise SemanticError("XML step cnt must equal one")
                srcbuf = node.attrib["srcbuf"]
                dstbuf = node.attrib["dstbuf"]
                srcoff = _integer(node, "srcoff")
                dstoff = _integer(node, "dstoff")
                _buffer_limit(buffers, rank, srcbuf)
                _buffer_limit(buffers, rank, dstbuf)
                if (srcbuf, srcoff, dstbuf, dstoff) != _expected_step_refs(
                    step,
                    sidecar_refs,
                    buffers,
                ):
                    raise SemanticError("XML step address differs from the sidecar")
                if step.xml_type is EndpointType.NOP:
                    if srcoff != -1 or dstoff != -1:
                        raise SemanticError("XML NOP offsets must equal minus one")
                else:
                    check_source = step.xml_type in {
                        EndpointType.SEND,
                        EndpointType.RECV_REDUCE_COPY,
                        EndpointType.COPY,
                    }
                    check_destination = step.xml_type in {
                        EndpointType.RECV,
                        EndpointType.RECV_REDUCE_COPY,
                        EndpointType.COPY,
                    }
                    if check_source and not 0 <= srcoff < _buffer_limit(
                        buffers,
                        rank,
                        srcbuf,
                    ):
                        raise SemanticError("XML source offset is out of bounds")
                    if check_destination and not 0 <= dstoff < _buffer_limit(
                        buffers,
                        rank,
                        dstbuf,
                    ):
                        raise SemanticError("XML destination offset is out of bounds")
                depid = _integer(node, "depid")
                deps = _integer(node, "deps")
                if (depid < 0) != (deps < 0):
                    raise SemanticError("XML dependency coordinates are incomplete")
                coordinates[(rank, tb_id, index)] = (depid, deps, step)
                represented_steps[step.step_id] = (rank, tb_id, index)
                if _integer(node, "hasdep") != int(step.has_dependence):
                    raise SemanticError("XML completion flag differs from the sidecar")
    if maximum_channel + 1 != nchannels:
        raise SemanticError("XML nchannels does not match its threadblocks")
    if set(represented_steps) != set(program.steps_by_id):
        raise SemanticError("XML steps do not match the sidecar")
    for coordinate, (depid, deps, step) in coordinates.items():
        expected = (-1, -1)
        if step.dependency_step_id is not None:
            dependency = represented_steps[step.dependency_step_id]
            expected = dependency[1], dependency[2]
            if dependency[0] != coordinate[0]:
                raise SemanticError("XML dependency crosses ranks")
        if (depid, deps) != expected:
            raise SemanticError("XML dependency differs from the sidecar")
        if depid >= 0 and (coordinate[0], depid, deps) not in coordinates:
            raise SemanticError("XML dependency references a missing step")
    for transfer_id, step_ids in program.transfer_steps.items():
        if len(step_ids) != 2 or not set(step_ids) <= set(represented_steps):
            raise SemanticError("XML transfer sidecar is incomplete")
        types = {program.steps_by_id[step_id].xml_type for step_id in step_ids}
        if EndpointType.SEND not in types or not types.intersection(
            {EndpointType.RECV, EndpointType.RECV_REDUCE_COPY}
        ):
            raise SemanticError(
                "XML transfer {} has incompatible endpoints".format(transfer_id)
            )
    return root


def normalize_xml(xml_text: str) -> str:
    root = _parse(xml_text)
    return etree.tostring(root, encoding="unicode")
