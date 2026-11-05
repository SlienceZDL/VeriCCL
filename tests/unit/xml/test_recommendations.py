import hashlib
from copy import deepcopy
from dataclasses import replace

import pytest
from lxml import etree

from vericcl.errors import SemanticError
from vericcl.semantics.collective import CollectiveKind
from vericcl.topology import load_topology
from vericcl.xml.compatibility import check_msccl_compatibility
from vericcl.xml.lower import lower_to_xml
from vericcl.xml.recommendations import (
    Recommendation,
    artifact_xml_filename,
    recommend_runtime_compatible_inputs,
)

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
    changed = replace(
        artifact,
        xml_text=xml_text,
        sha256=hashlib.sha256(xml_text.encode("utf-8")).hexdigest(),
    )
    return check_msccl_compatibility(changed).apply(changed)


def test_filename_distinguishes_executable_and_offline_candidate_xml():
    compatible = _artifact()
    incompatible = _changed(
        compatible,
        lambda root: root.xpath(".//step")[0].attrib.update(
            {"srcoff": "32768"}
        ),
    )

    assert artifact_xml_filename("schedule", compatible) == "schedule.xml"
    assert (
        artifact_xml_filename("schedule", incompatible)
        == "schedule.candidate.xml"
    )
    assert incompatible.xml_text
    assert incompatible.runtime_compatible is False


def test_recommendations_prioritize_renumber_then_channels_then_slice_size():
    artifact = _artifact()

    def mutate(root):
        gpu = root.xpath("./gpu")[0]
        template = gpu.xpath("./tb")[0]
        source = deepcopy(template)
        source.attrib["id"] = "200"
        gpu.append(source)
        consumer = gpu.xpath("./tb")[1].xpath("./step")[0]
        consumer.attrib["depid"] = "200"
        consumer.attrib["deps"] = "0"
        tb = gpu.xpath("./tb")[0]
        step = tb.xpath("./step")[0]
        for index in range(1, 257):
            clone = deepcopy(step)
            clone.attrib["s"] = str(index)
            tb.append(clone)

    incompatible = _changed(artifact, mutate)
    inputs = resolved(CollectiveKind.ALL_REDUCE, ranks=2, slices=8)

    recommendations = recommend_runtime_compatible_inputs(
        inputs,
        incompatible,
    )

    assert [item.kind for item in recommendations[:3]] == [
        "renumber_threadblocks",
        "increase_channels",
        "increase_slice_size",
    ]
    assert recommendations[1].parameters["max_channels"] == 2
    assert recommendations[2].parameters["slice_size_bytes"] == 2048
    assert inputs.hyperparameters.slice_size_bytes == 1024


@pytest.mark.parametrize(
    "kind",
    [CollectiveKind.ALL_TO_ALL, CollectiveKind.REDUCE_SCATTER],
)
def test_larger_slice_recommendation_preserves_partition_divisibility(kind):
    artifact = _changed(
        _artifact(),
        lambda root: root.xpath(".//step")[0].attrib.update(
            {"srcoff": "32768"}
        ),
    )
    inputs = resolved(kind, ranks=2, slices=8)

    recommendation = next(
        item
        for item in recommend_runtime_compatible_inputs(inputs, artifact)
        if item.kind == "increase_slice_size"
    )
    candidate = recommendation.parameters["slice_size_bytes"]

    assert inputs.hyperparameters.total_size_bytes % candidate == 0
    assert (
        inputs.hyperparameters.total_size_bytes // candidate
    ) % inputs.rank_count == 0


def _recommendation():
    return Recommendation(
        kind="increase_channels",
        priority=1,
        parameters={"max_channels": 2},
        reason_codes=("steps_per_tb",),
        requires_resolve=True,
    )


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("kind", "", "kind"),
        ("priority", True, "priority"),
        ("priority", -1, "priority"),
        ("parameters", {"max_channels": 0}, "parameters"),
        ("reason_codes", (), "reason_codes"),
        ("reason_codes", ("",), "reason_codes"),
        ("requires_resolve", 1, "requires_resolve"),
    ],
)
def test_recommendation_rejects_invalid_fields(field, value, message):
    with pytest.raises(SemanticError, match=message):
        replace(_recommendation(), **{field: value})


def test_recommendation_helpers_reject_invalid_inputs():
    artifact = _artifact()
    with pytest.raises(SemanticError, match="schedule_name"):
        artifact_xml_filename("", artifact)
    with pytest.raises(SemanticError, match="ResolvedInput"):
        recommend_runtime_compatible_inputs(None, artifact)


def test_compatible_artifact_has_no_runtime_recommendations():
    inputs = resolved(CollectiveKind.ALL_REDUCE, ranks=2, slices=1)

    assert recommend_runtime_compatible_inputs(inputs, _artifact()) == ()
