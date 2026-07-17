import hashlib
from copy import deepcopy
from dataclasses import replace

import pytest
from lxml import etree

from vericcl.errors import SemanticError
from vericcl.semantics.collective import CollectiveKind
from vericcl.topology import load_topology
from vericcl.xml.compatibility import (
    CompatibilityIssue,
    CompatibilityReport,
    check_msccl_compatibility,
    renumber_dependent_threadblocks,
)
from vericcl.xml.lower import lower_to_xml

from tests.unit.xml.helpers import resolved, two_rank_allreduce_schedule


pytestmark = pytest.mark.phase04


def _artifact():
    inputs = resolved(CollectiveKind.ALL_REDUCE, ranks=2, slices=1)
    return lower_to_xml(
        two_rank_allreduce_schedule(),
        inputs,
        load_topology(inputs),
    )


def _changed(artifact, mutation):
    root = etree.fromstring(artifact.xml_text.encode("utf-8"))
    mutation(root)
    xml_text = etree.tostring(root, encoding="unicode")
    return replace(
        artifact,
        xml_text=xml_text,
        sha256=hashlib.sha256(xml_text.encode("utf-8")).hexdigest(),
    )


def _steps(root):
    tb = root.xpath("./gpu/tb")[0]
    template = tb.xpath("./step")[0]
    for index in range(1, 257):
        step = deepcopy(template)
        step.attrib["s"] = str(index)
        tb.append(step)


def _directional_tbs(root, direction):
    gpu = root.xpath("./gpu")[1]
    attribute = "send" if direction == "send" else "recv"
    template = next(
        tb for tb in gpu.xpath("./tb") if int(tb.attrib[attribute]) >= 0
    )
    for index in range(32):
        tb = deepcopy(template)
        tb.attrib["id"] = str(10 + index)
        gpu.append(tb)


def _rank_tbs(root):
    gpu = root.xpath("./gpu")[0]
    current = len(gpu.xpath("./tb"))
    for index in range(current, 217):
        etree.SubElement(
            gpu,
            "tb",
            id=str(index),
            send="-1",
            recv="-1",
            chan="0",
        )


def _channels(root):
    root.attrib["nchannels"] = "33"
    root.xpath("./gpu/tb")[0].attrib["chan"] = "32"


def _offset(root):
    root.xpath(".//step")[0].attrib["srcoff"] = "32768"


def _dependent_tbs(root):
    gpu = root.xpath("./gpu")[0]
    template = gpu.xpath("./tb")[0]
    for child in tuple(gpu):
        gpu.remove(child)
    for tb_id in range(129):
        tb = deepcopy(template)
        tb.attrib["id"] = str(tb_id)
        step = tb.xpath("./step")[0]
        step.attrib.update(
            {
                "s": "0",
                "depid": str(tb_id),
                "deps": "0",
                "hasdep": "1",
            }
        )
        gpu.append(tb)


@pytest.mark.parametrize(
    "mutation,code,current,limit,location",
    [
        (_steps, "steps_per_tb", 257, 256, "tb"),
        (
            lambda root: _directional_tbs(root, "send"),
            "send_tbs_per_channel",
            33,
            32,
            "channel",
        ),
        (
            lambda root: _directional_tbs(root, "recv"),
            "recv_tbs_per_channel",
            33,
            32,
            "channel",
        ),
        (_rank_tbs, "tbs_per_rank", 217, 216, "rank"),
        (_channels, "channels", 33, 32, "channel"),
        (_offset, "buffer_offset", 32768, 32767, "tb"),
        (_dependent_tbs, "dependent_tb_id", 128, 127, "tb"),
    ],
)
def test_each_confirmed_msccl_limit_is_reported(
    mutation,
    code,
    current,
    limit,
    location,
):
    report = check_msccl_compatibility(_changed(_artifact(), mutation))
    issue = next(issue for issue in report.issues if issue.code == code)

    assert report.runtime_compatible is False
    assert issue.rank >= 0
    assert issue.current_value == current
    assert issue.limit == limit
    assert issue.transfer_ids
    if location == "tb":
        assert issue.tb_id is not None
    elif location == "channel":
        assert issue.channel is not None


def test_compatible_artifact_remains_runtime_compatible():
    artifact = _artifact()
    report = check_msccl_compatibility(artifact)

    assert report.runtime_compatible is True
    assert report.issues == ()
    assert report.apply(artifact).runtime_compatible is True


def test_dependent_threadblock_renumbering_preserves_steps_and_dependencies():
    artifact = _artifact()
    program = artifact.tb_program

    renumbered = renumber_dependent_threadblocks(program)

    assert renumbered.steps_by_id == program.steps_by_id
    assert renumbered.node_steps == program.node_steps
    assert renumbered.transfer_steps == program.transfer_steps
    assert {
        step.step_id: step.dependency_step_id
        for step in renumbered.steps_by_id.values()
    } == {
        step.step_id: step.dependency_step_id
        for step in program.steps_by_id.values()
    }
    for rank in range(2):
        dependent_ids = {
            tb.tb_id
            for tb in renumbered.threadblocks
            if tb.key.rank == rank
            and any(
                step.step_id in renumbered.referenced_step_ids
                for step in tb.steps
            )
        }
        assert not dependent_ids or max(dependent_ids) <= 127


def _issue():
    return CompatibilityIssue(
        code="steps_per_tb",
        message="limit exceeded",
        rank=0,
        tb_id=0,
        channel=0,
        current_value=257,
        limit=256,
        transfer_ids=("transfer-0",),
    )


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("code", "", "code"),
        ("message", "", "message"),
        ("rank", -1, "rank"),
        ("tb_id", -1, "tb_id"),
        ("channel", True, "channel"),
        ("current_value", True, "current_value"),
        ("limit", True, "limit"),
        ("limit", -1, "limit"),
        ("transfer_ids", (), "transfer_ids"),
    ],
)
def test_compatibility_issue_rejects_invalid_fields(field, value, message):
    with pytest.raises(SemanticError, match=message):
        replace(_issue(), **{field: value})


def test_compatibility_report_rejects_invalid_issue():
    with pytest.raises(SemanticError, match="invalid issue"):
        CompatibilityReport((object(),))


def test_compatibility_operations_reject_invalid_inputs():
    with pytest.raises(SemanticError, match="ThreadblockProgram"):
        renumber_dependent_threadblocks(None)
    with pytest.raises(SemanticError, match="XmlArtifact"):
        check_msccl_compatibility(None)


def test_compatibility_check_rejects_malformed_xml_and_channel_count():
    artifact = _artifact()
    malformed_text = "<algo>"
    malformed = replace(
        artifact,
        xml_text=malformed_text,
        sha256=hashlib.sha256(malformed_text.encode("utf-8")).hexdigest(),
    )
    with pytest.raises(SemanticError, match="well formed"):
        check_msccl_compatibility(malformed)

    invalid_channels = _changed(
        artifact,
        lambda root: root.attrib.update({"nchannels": "invalid"}),
    )
    with pytest.raises(SemanticError, match="nchannels"):
        check_msccl_compatibility(invalid_channels)
