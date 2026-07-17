from dataclasses import replace

import pytest
from lxml import etree

from vericcl.errors import SemanticError
from vericcl.semantics.collective import CollectiveKind
from vericcl.topology import Topology, load_topology
from vericcl.xml.buffers import build_buffer_plan
from vericcl.xml.dependencies import build_transfer_dag
from vericcl.xml.emitter import emit_xml
from vericcl.xml.endpoints import lower_endpoints
from vericcl.xml.granularity import verify_atom_granularity
from vericcl.xml.list_scheduler import schedule_threadblocks
from vericcl.xml.lower import lower_to_xml
from vericcl.xml.parser import validate_xml

from tests.unit.xml.helpers import (
    allreduce_star_schedule,
    final_schedule,
    resolved,
    two_rank_allreduce_schedule,
)


pytestmark = pytest.mark.phase04


def _emitted(schedule=None, inputs=None):
    schedule = schedule or two_rank_allreduce_schedule()
    inputs = inputs or resolved(
        CollectiveKind.ALL_REDUCE,
        ranks=2,
        slices=1,
    )
    buffers = build_buffer_plan(schedule, inputs)
    endpoints = lower_endpoints(schedule, buffers)
    dag = build_transfer_dag(endpoints, schedule, buffers)
    program = schedule_threadblocks(endpoints, dag)
    return emit_xml(program, buffers, inputs), program, buffers, inputs


def test_emitter_uses_exact_atom_granularity_and_allowed_operations():
    xml_text, _, _, _ = _emitted()
    root = etree.fromstring(xml_text.encode("utf-8"))

    assert root.attrib["name"] == "vericcl"
    assert root.attrib["proto"] == "Simple"
    assert not root.xpath(".//copy")
    steps = root.xpath(".//step")
    assert steps
    assert all(step.attrib["cnt"] == "1" for step in steps)
    assert {step.attrib["type"] for step in steps} <= {
        "s",
        "r",
        "rrc",
        "cpy",
        "nop",
    }


@pytest.mark.parametrize(
    "kind,inplace,expected_chunks,runtime_multiplier",
    [
        (CollectiveKind.BROADCAST, False, 2, 1),
        (CollectiveKind.REDUCE, False, 2, 1),
        (CollectiveKind.ALL_REDUCE, False, 2, 1),
        (CollectiveKind.ALL_TO_ALL, False, 2, 1),
        (CollectiveKind.REDUCE_SCATTER, False, 2, 1),
        (CollectiveKind.ALL_GATHER, True, 4, 2),
    ],
)
def test_operator_geometry_uses_exact_chunk_counts_and_byte_range(
    kind,
    inplace,
    expected_chunks,
    runtime_multiplier,
):
    inputs = resolved(kind, ranks=2, slices=2, inplace=inplace)
    xml_text, _, _, _ = _emitted(
        final_schedule(kind, ranks=2, slices=2, inplace=inplace),
        inputs,
    )
    root = etree.fromstring(xml_text.encode("utf-8"))
    runtime_bytes = inputs.hyperparameters.total_size_bytes * runtime_multiplier

    assert root.attrib["coll"] == kind.value
    assert root.attrib["inplace"] == ("1" if inplace else "0")
    assert int(root.attrib["nchunksperloop"]) == expected_chunks
    assert int(root.attrib["minBytes"]) == runtime_bytes
    assert int(root.attrib["maxBytes"]) == runtime_bytes + 1


def test_step_fields_encode_local_dependency_coordinates():
    xml_text, program, buffers, inputs = _emitted()
    root = etree.fromstring(xml_text.encode("utf-8"))

    for gpu in root.xpath("./gpu"):
        rank = int(gpu.attrib["id"])
        for tb in gpu.xpath("./tb"):
            tb_id = int(tb.attrib["id"])
            for index, step in enumerate(tb.xpath("./step")):
                assert int(step.attrib["s"]) == index
                assert set(step.attrib) == {
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
                if int(step.attrib["depid"]) >= 0:
                    assert int(step.attrib["depid"]) != tb_id
                    assert rank == int(gpu.attrib["id"])

    validate_xml(xml_text, program, buffers, inputs)


def test_rrc_source_address_is_the_receiver_local_accumulator():
    xml_text, _, buffers, _ = _emitted()
    root = etree.fromstring(xml_text.encode("utf-8"))
    rrc = root.xpath(".//step[@type='rrc']")[0]
    accumulator = buffers.transfer_accumulator_refs["allreduce-reduce"]

    assert (rrc.attrib["srcbuf"], int(rrc.attrib["srcoff"])) == (
        accumulator.buffer,
        accumulator.offset,
    )


def test_runtime_geometry_preserves_one_atom_per_step():
    verify_atom_granularity(
        runtime_count=67_108_864,
        size_multiplier=1,
        datatype_size_bytes=4,
        nchunks_per_loop=256,
        slice_size_bytes=1_048_576,
        nccl_buffsize_bytes=2_097_152,
    )


@pytest.mark.parametrize(
    "mutation,match",
    [
        ({"runtime_count": 67_108_863}, "divisible"),
        ({"slice_size_bytes": 1_048_575}, "slice size"),
        ({"nccl_buffsize_bytes": 4_194_304}, "NCCL_BUFFSIZE"),
    ],
)
def test_runtime_geometry_rejects_split_or_mismatched_atoms(mutation, match):
    arguments = {
        "runtime_count": 67_108_864,
        "size_multiplier": 1,
        "datatype_size_bytes": 4,
        "nchunks_per_loop": 256,
        "slice_size_bytes": 1_048_576,
        "nccl_buffsize_bytes": 2_097_152,
    }
    arguments.update(mutation)

    with pytest.raises(SemanticError, match=match):
        verify_atom_granularity(**arguments)


def test_parser_rejects_non_unit_count_and_top_level_copy():
    xml_text, program, buffers, inputs = _emitted()
    root = etree.fromstring(xml_text.encode("utf-8"))
    root.xpath(".//step")[0].attrib["cnt"] = "2"

    with pytest.raises(SemanticError, match="cnt"):
        validate_xml(
            etree.tostring(root, encoding="unicode"),
            program,
            buffers,
            inputs,
        )

    root = etree.fromstring(xml_text.encode("utf-8"))
    etree.SubElement(root.xpath("./gpu")[0], "copy")
    with pytest.raises(SemanticError, match="top-level copy"):
        validate_xml(
            etree.tostring(root, encoding="unicode"),
            program,
            buffers,
            inputs,
        )


def test_parser_rejects_address_or_dependency_sidecar_mismatch():
    xml_text, program, buffers, inputs = _emitted()
    root = etree.fromstring(xml_text.encode("utf-8"))
    send = next(
        step
        for step in root.xpath(".//step")
        if step.attrib["type"] == "s" and step.attrib["srcbuf"] == "o"
    )
    send.attrib["srcbuf"] = "i"

    with pytest.raises(SemanticError, match="address"):
        validate_xml(
            etree.tostring(root, encoding="unicode"),
            program,
            buffers,
            inputs,
        )

    root = etree.fromstring(xml_text.encode("utf-8"))
    dependent = next(
        step
        for step in root.xpath(".//step")
        if int(step.attrib["depid"]) >= 0
    )
    dependent.attrib["depid"] = "-1"
    with pytest.raises(SemanticError, match="coordinates"):
        validate_xml(
            etree.tostring(root, encoding="unicode"),
            program,
            buffers,
            inputs,
        )


def test_emitter_and_parser_support_nop_joins():
    schedule = allreduce_star_schedule()
    inputs = resolved(
        CollectiveKind.ALL_REDUCE,
        ranks=4,
        slices=1,
    )
    xml_text, program, buffers, _ = _emitted(schedule, inputs)

    assert etree.fromstring(xml_text.encode("utf-8")).xpath(
        ".//step[@type='nop']"
    )
    validate_xml(xml_text, program, buffers, inputs)


@pytest.mark.parametrize("value", [None, True, 0])
def test_granularity_rejects_non_positive_integer_inputs(value):
    with pytest.raises(SemanticError, match="positive integer"):
        verify_atom_granularity(
            runtime_count=value,
            size_multiplier=1,
            datatype_size_bytes=4,
            nchunks_per_loop=1,
            slice_size_bytes=4,
            nccl_buffsize_bytes=8,
        )


def test_emitter_rejects_invalid_inputs():
    _, program, buffers, inputs = _emitted()

    with pytest.raises(SemanticError, match="ThreadblockProgram"):
        emit_xml(None, buffers, inputs)
    with pytest.raises(SemanticError, match="BufferPlan"):
        emit_xml(program, None, inputs)
    with pytest.raises(SemanticError, match="ResolvedInput"):
        emit_xml(program, buffers, None)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("integer", "must be an integer"),
        ("root", "root must be algo"),
        ("root_attribute", "root attribute"),
        ("nchannels", "must be positive"),
        ("root_child", "unsupported child"),
        ("gpu_order", "GPU IDs"),
        ("gpu_chunks", "chunk declaration"),
        ("gpu_child", "unsupported child"),
        ("tb_ids", "TB IDs"),
        ("lane", "lane metadata"),
        ("step_count", "step count"),
        ("step_attributes", "attributes are incomplete"),
        ("step_id", "step IDs"),
        ("unknown_type", "type is unsupported"),
        ("sidecar_type", "type differs"),
        ("unknown_buffer", "unknown buffer"),
        ("completion", "completion flag"),
        ("channel_count", "nchannels does not match"),
    ],
)
def test_parser_rejects_structural_mutations(mutation, match):
    xml_text, program, buffers, inputs = _emitted()
    root = etree.fromstring(xml_text.encode("utf-8"))
    if mutation == "integer":
        root.attrib["nchannels"] = "x"
    elif mutation == "root":
        root.tag = "algorithm"
    elif mutation == "root_attribute":
        root.attrib["name"] = "other"
    elif mutation == "nchannels":
        root.attrib["nchannels"] = "0"
    elif mutation == "root_child":
        etree.SubElement(root, "tb")
    elif mutation == "gpu_order":
        root.append(root[0])
    elif mutation == "gpu_chunks":
        root.xpath("./gpu")[0].attrib["i_chunks"] = "2"
    elif mutation == "gpu_child":
        etree.SubElement(root.xpath("./gpu")[0], "step")
    elif mutation == "tb_ids":
        root.xpath("./gpu/tb")[-1].attrib["id"] = "9"
    elif mutation == "lane":
        root.xpath("./gpu/tb")[0].attrib["send"] = "1"
    elif mutation == "step_count":
        step = root.xpath(".//step")[0]
        step.getparent().remove(step)
    elif mutation == "step_attributes":
        del root.xpath(".//step")[0].attrib["cnt"]
    elif mutation == "step_id":
        root.xpath(".//step")[0].attrib["s"] = "1"
    elif mutation == "unknown_type":
        root.xpath(".//step")[0].attrib["type"] = "rcs"
    elif mutation == "sidecar_type":
        root.xpath(".//step")[0].attrib["type"] = "nop"
    elif mutation == "unknown_buffer":
        root.xpath(".//step")[0].attrib["srcbuf"] = "x"
    elif mutation == "completion":
        step = root.xpath(".//step")[0]
        step.attrib["hasdep"] = "0" if step.attrib["hasdep"] == "1" else "1"
    elif mutation == "channel_count":
        root.attrib["nchannels"] = "2"

    with pytest.raises(SemanticError, match=match):
        validate_xml(
            etree.tostring(root, encoding="unicode"),
            program,
            buffers,
            inputs,
        )


def test_parser_rejects_malformed_xml_and_invalid_model_inputs():
    xml_text, program, buffers, inputs = _emitted()

    with pytest.raises(SemanticError, match="well formed"):
        validate_xml("<algo>", program, buffers, inputs)
    with pytest.raises(SemanticError, match="ThreadblockProgram"):
        validate_xml(xml_text, None, buffers, inputs)
    with pytest.raises(SemanticError, match="BufferPlan"):
        validate_xml(xml_text, program, None, inputs)
    with pytest.raises(SemanticError, match="ResolvedInput"):
        validate_xml(xml_text, program, buffers, None)


def test_lowering_and_artifact_models_reject_invalid_inputs():
    schedule = two_rank_allreduce_schedule()
    inputs = resolved(CollectiveKind.ALL_REDUCE, ranks=2, slices=1)
    topology = load_topology(inputs)
    artifact = lower_to_xml(schedule, inputs, topology)

    with pytest.raises(SemanticError, match="Schedule"):
        lower_to_xml(None, inputs, topology)
    with pytest.raises(SemanticError, match="ResolvedInput"):
        lower_to_xml(schedule, None, topology)
    with pytest.raises(SemanticError, match="Topology"):
        lower_to_xml(schedule, inputs, None)
    with pytest.raises(SemanticError, match="rank counts"):
        lower_to_xml(schedule, replace(inputs, rank_count=3), topology)
    different_slices = replace(
        inputs,
        hyperparameters=replace(
            inputs.hyperparameters,
            total_size_bytes=2048,
        ),
    )
    with pytest.raises(SemanticError, match="slice counts"):
        lower_to_xml(schedule, different_slices, topology)
    empty_topology = Topology(
        rank_count=2,
        links={},
        shared_resources={},
        node_membership={0: 0, 1: 0},
        gateways=frozenset(),
        warnings=(),
    )
    with pytest.raises(SemanticError, match="missing topology link"):
        lower_to_xml(schedule, inputs, empty_topology)
    wide_channel = replace(
        schedule,
        transfers=tuple(
            replace(transfer, channel=99) for transfer in schedule.transfers
        ),
    )
    with pytest.raises(SemanticError, match="channel exceeds"):
        lower_to_xml(wide_channel, inputs, topology)

    for field, value, match in (
        ("xml_text", "", "xml_text"),
        ("buffer_plan", None, "buffer_plan"),
        ("endpoint_program", None, "endpoint_program"),
        ("tb_program", None, "tb_program"),
        ("sha256", "bad", "sha256"),
        ("runtime_compatible", 1, "runtime_compatible"),
    ):
        with pytest.raises(SemanticError, match=match):
            replace(artifact, **{field: value})
