from collections import Counter

import pytest
from lxml import etree

from vericcl.errors import SemanticError
from vericcl.topology.model import (
    DirectedLink,
    LinkKey,
    PerformanceCurve,
    Topology,
)
from vericcl.verification.online.calibration import CalibrationRequest
from vericcl.verification.online.calibration_xml import (
    build_calibration_artifact,
    build_calibration_artifacts,
    build_calibration_benchmark,
)
from vericcl.xml.endpoints import EndpointType


pytestmark = pytest.mark.phase06


MIB = 1024 * 1024


def _topology(*, max_channels=4, inter_node=False):
    key = LinkKey(0, 1)
    curve = PerformanceCurve(2.0, 5.0, {})
    return Topology(
        rank_count=2,
        links={
            key: DirectedLink(key, max_channels, curve, ()),
        },
        shared_resources={},
        node_membership={0: 0, 1: 1 if inter_node else 0},
        gateways=frozenset({0, 1}) if inter_node else frozenset(),
        warnings=(),
    )


def _request(**changes):
    values = {
        "link_class": "intra_node",
        "slice_size_bytes": MIB,
        "max_calibration_channels": 4,
        "datatype": "float",
    }
    values.update(changes)
    return CalibrationRequest(**values)


def _send_endpoints(artifact):
    return tuple(
        next(
            endpoint
            for endpoint in artifact.endpoint_program.by_transfer_id[
                transfer_id
            ]
            if endpoint.xml_type is EndpointType.SEND
        )
        for transfer_id in sorted(artifact.endpoint_program.by_transfer_id)
    )


def test_calibration_assigns_round_robin_channels_and_full_waves():
    artifact = build_calibration_artifact(
        _request(),
        _topology(),
        concurrency=3,
    )
    sends = _send_endpoints(artifact)
    assignments = tuple(endpoint.channel for endpoint in sends)
    waves = Counter(endpoint.st_time for endpoint in sends)

    assert assignments[:6] == (0, 1, 2, 0, 1, 2)
    assert len([count for count in waves.values() if count == 3]) == 128 // 3
    assert waves[max(waves)] == 128 % 3
    assert all(
        atom.slice_size_bytes == MIB
        for endpoint in sends
        for atom in endpoint.member_atoms
    )

    root = etree.fromstring(artifact.xml_text.encode("utf-8"))
    assert root.attrib["nchannels"] == "3"
    assert root.attrib["nchunksperloop"] == "128"
    assert root.attrib["inplace"] == "1"
    assert not root.xpath(".//step[@type='cpy']")
    assert root.attrib["minBytes"] == str(128 * MIB)
    assert root.attrib["maxBytes"] == str(128 * MIB + 1)


def test_calibration_builds_every_integer_concurrency_without_interpolation():
    artifacts = build_calibration_artifacts(_request(), _topology())

    assert len(artifacts) == 4
    assert tuple(
        etree.fromstring(artifact.xml_text.encode("utf-8")).attrib[
            "nchannels"
        ]
        for artifact in artifacts
    ) == ("1", "2", "3", "4")


def test_calibration_respects_global_software_concurrency_limit():
    artifacts = build_calibration_artifacts(
        _request(max_calibration_channels=32),
        _topology(max_channels=32),
    )

    assert len(artifacts) == 16


def test_calibration_benchmark_exposes_matching_schedule_and_inputs():
    benchmark = build_calibration_benchmark(
        _request(),
        _topology(),
        concurrency=2,
    )

    assert benchmark.schedule.metadata["calibration_concurrency"] == 2
    assert benchmark.inputs.rank_count == 2
    assert benchmark.inputs.collective.inplace is True
    assert benchmark.inputs.hyperparameters.total_size_bytes == 128 * MIB
    assert benchmark.artifact == build_calibration_artifact(
        _request(),
        _topology(),
        concurrency=2,
    )


def test_nondivisible_benchmark_skips_without_changing_slice_size():
    request = _request(slice_size_bytes=3 * MIB)

    artifacts = build_calibration_artifacts(request, _topology())

    assert artifacts == ()
    assert request.slice_size_bytes == 3 * MIB


def test_calibration_respects_link_limit_and_link_class():
    artifacts = build_calibration_artifacts(
        _request(max_calibration_channels=32),
        _topology(max_channels=2),
    )
    assert len(artifacts) == 2

    inter_request = _request(link_class="inter_node")
    assert len(
        build_calibration_artifacts(
            inter_request,
            _topology(inter_node=True),
        )
    ) == 4
    with pytest.raises(SemanticError, match="link class"):
        build_calibration_artifacts(inter_request, _topology())
    with pytest.raises(SemanticError, match="concurrency"):
        build_calibration_artifact(
            _request(),
            _topology(max_channels=2),
            concurrency=3,
        )
