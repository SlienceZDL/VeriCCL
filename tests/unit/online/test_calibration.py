from dataclasses import replace

import pytest

from vericcl.errors import SemanticError
from vericcl.verification.online.calibration import (
    CalibrationPoint,
    CalibrationRequest,
    CalibrationResult,
    apply_calibration_to_topology,
    calibration_point_from_trace,
    derive_calibrated_curve,
)
from vericcl.verification.online.clock_sync import AlignedTimestamp
from vericcl.verification.online.trace_analysis import (
    PhysicalTransferInterval,
    TraceAnalysis,
)
from vericcl.verification.online.statistics import summarize_runs
from vericcl.topology.model import (
    DirectedLink,
    LinkKey,
    PerformanceCurve,
    SharedResource,
    Topology,
)


pytestmark = pytest.mark.phase06


def _point(concurrency, duration_us, *, stable=True):
    samples = (
        (duration_us,) * 20
        if stable
        else (duration_us,) * 19 + (duration_us * 3.0,)
    )
    return CalibrationPoint(
        concurrency=concurrency,
        duration_statistics=summarize_runs(samples),
        full_wave_count=4,
        tail_transfer_count=0,
    )


def test_calibration_curve_preserves_alpha_and_uses_safe_p95_formula():
    curve = derive_calibrated_curve(
        alpha_us=2.0,
        slice_size_bytes=1000,
        points=(
            _point(1, 12.0),
            _point(2, 8.0),
            _point(3, 7.0),
        ),
    )

    assert curve.alpha_us == pytest.approx(2.0)
    assert curve.invbw_us == pytest.approx(12.0)
    assert curve.beta_effective_us == pytest.approx(10.0)
    assert curve.bandwidth_bytes_per_us[1] == pytest.approx(100.0)
    assert curve.bandwidth_bytes_per_us[2] == pytest.approx(2000.0 / 6.0)
    assert curve.bandwidth_bytes_per_us[3] == pytest.approx(600.0)


def test_curve_rejects_invalid_incomplete_or_unstable_points():
    with pytest.raises(SemanticError, match="above alpha"):
        derive_calibrated_curve(2.0, 1000, (_point(1, 2.0),))
    with pytest.raises(SemanticError, match="contiguous"):
        derive_calibrated_curve(
            2.0,
            1000,
            (_point(1, 12.0), _point(3, 7.0)),
        )
    with pytest.raises(SemanticError, match="stable"):
        derive_calibrated_curve(
            2.0,
            1000,
            (_point(1, 12.0, stable=False),),
        )
    with pytest.raises(SemanticError, match="point"):
        derive_calibrated_curve(2.0, 1000, ())


def test_stable_calibration_updates_only_matching_links_and_resources():
    intra = LinkKey(0, 1)
    inter = LinkKey(0, 2)
    intra_curve = PerformanceCurve(2.0, 8.0, {})
    inter_curve = PerformanceCurve(4.0, 12.0, {})
    topology = Topology(
        rank_count=3,
        links={
            intra: DirectedLink(intra, 4, intra_curve, ("pcie",)),
            inter: DirectedLink(inter, 2, inter_curve, ("nic",)),
        },
        shared_resources={
            "pcie": SharedResource("pcie", (intra,), 4, intra_curve),
            "nic": SharedResource("nic", (inter,), 2, inter_curve),
        },
        node_membership={0: 0, 1: 0, 2: 1},
        gateways=frozenset({0, 2}),
        warnings=(),
    )
    points = (_point(1, 12.0), _point(2, 8.0))
    result = CalibrationResult(
        request=CalibrationRequest("intra_node", 1000, 2, "float"),
        points=points,
        curve=derive_calibrated_curve(2.0, 1000, points),
        skipped_reason=None,
    )

    updated = apply_calibration_to_topology(topology, result)

    assert updated.links[intra].performance.is_calibrated is True
    assert updated.links[intra].max_channels == 2
    assert updated.links[intra].performance.alpha_us == 2.0
    assert updated.links[inter].performance == inter_curve
    assert updated.shared_resources["pcie"].performance.is_calibrated is True
    assert updated.shared_resources["pcie"].max_channels == 2
    assert updated.shared_resources["nic"].performance == inter_curve
    assert updated.isomorphism_signature != topology.isomorphism_signature


def test_topology_update_rejects_skipped_calibration():
    key = LinkKey(0, 1)
    curve = PerformanceCurve(2.0, 8.0, {})
    topology = Topology(
        rank_count=2,
        links={key: DirectedLink(key, 1, curve, ())},
        shared_resources={},
        node_membership={0: 0, 1: 0},
        gateways=frozenset(),
        warnings=(),
    )
    skipped = CalibrationResult(
        request=CalibrationRequest("intra_node", 1000, 1, "float"),
        points=(),
        curve=None,
        skipped_reason="not_divisible",
    )

    with pytest.raises(SemanticError, match="stable"):
        apply_calibration_to_topology(topology, skipped)


def test_topology_update_rejects_nonisomorphic_links_in_one_class():
    first = LinkKey(0, 1)
    second = LinkKey(2, 3)
    curve = PerformanceCurve(2.0, 8.0, {})
    topology = Topology(
        rank_count=4,
        links={
            first: DirectedLink(first, 2, curve, ()),
            second: DirectedLink(second, 1, curve, ()),
        },
        shared_resources={},
        node_membership={0: 0, 1: 0, 2: 1, 3: 1},
        gateways=frozenset(),
        warnings=(),
    )
    points = (_point(1, 12.0), _point(2, 8.0))
    result = CalibrationResult(
        request=CalibrationRequest("intra_node", 1000, 2, "float"),
        points=points,
        curve=derive_calibrated_curve(2.0, 1000, points),
        skipped_reason=None,
    )

    with pytest.raises(SemanticError, match="isomorphic"):
        apply_calibration_to_topology(topology, result)


def test_topology_update_rejects_different_communication_domain_sizes():
    first = LinkKey(0, 1)
    second = LinkKey(3, 4)
    curve = PerformanceCurve(2.0, 8.0, {})
    topology = Topology(
        rank_count=5,
        links={
            first: DirectedLink(first, 2, curve, ()),
            second: DirectedLink(second, 2, curve, ()),
        },
        shared_resources={},
        node_membership={0: 0, 1: 0, 2: 0, 3: 1, 4: 1},
        gateways=frozenset(),
        warnings=(),
    )
    points = (_point(1, 12.0), _point(2, 8.0))
    result = CalibrationResult(
        request=CalibrationRequest("intra_node", 1000, 2, "float"),
        points=points,
        curve=derive_calibrated_curve(2.0, 1000, points),
        skipped_reason=None,
    )

    with pytest.raises(SemanticError, match="isomorphic"):
        apply_calibration_to_topology(topology, result)


def test_calibration_point_normalizes_full_wave_aggregate_elapsed_time():
    intervals = []
    for iteration in range(20):
        for logical, (start, end) in enumerate(
            ((0.0, 5.0), (0.5, 4.5), (10.0, 17.0), (10.5, 16.0))
        ):
            intervals.append(
                PhysicalTransferInterval(
                    transfer_id="calibration-send-{:08d}".format(logical),
                    iteration=iteration,
                    send=None,
                    receive=None,
                    local=None,
                    physical_start=AlignedTimestamp(start, 0.0),
                    physical_end=AlignedTimestamp(end, 0.0),
                    endpoint_order_uncertain=False,
                    sender_start=AlignedTimestamp(start, 0.0),
                    sender_end=AlignedTimestamp(end, 0.0),
                )
            )
        intervals.append(
            PhysicalTransferInterval(
                transfer_id="copy-{:08d}".format(iteration),
                iteration=iteration,
                send=None,
                receive=None,
                local=object(),
                physical_start=AlignedTimestamp(0.0, 0.0),
                physical_end=AlignedTimestamp(1.0, 0.0),
                endpoint_order_uncertain=False,
            )
        )
    analysis = TraceAnalysis(tuple(intervals), (), (), (), True)
    request = CalibrationRequest("intra_node", 32 * 1024 * 1024, 2, "float")

    point = calibration_point_from_trace(request, 2, analysis)

    assert point.full_wave_count == 2
    assert point.tail_transfer_count == 0
    assert point.duration_statistics.p95_us == 8.5


def test_calibration_point_does_not_require_channels_to_advance_in_lockstep():
    intervals = []
    times = ((0.0, 5.0), (2.0, 7.0), (5.0, 10.0), (7.0, 12.0))
    for iteration in range(20):
        for logical, (start, end) in enumerate(times):
            intervals.append(
                PhysicalTransferInterval(
                    transfer_id="calibration-send-{:08d}".format(logical),
                    iteration=iteration,
                    send=None,
                    receive=None,
                    local=None,
                    physical_start=AlignedTimestamp(start, 0.0),
                    physical_end=AlignedTimestamp(end, 0.0),
                    endpoint_order_uncertain=False,
                    sender_start=AlignedTimestamp(start, 0.0),
                    sender_end=AlignedTimestamp(end, 0.0),
                )
            )
    analysis = TraceAnalysis(tuple(intervals), (), (), (), True)
    request = CalibrationRequest("intra_node", 32 * 1024 * 1024, 2, "float")

    point = calibration_point_from_trace(request, 2, analysis)

    assert point.duration_statistics.p95_us == 6.0


def test_calibration_uses_sender_local_wave_when_endpoints_are_uncertain():
    intervals = []
    for iteration in range(20):
        for logical, (start, end) in enumerate(
            ((0.0, 5.0), (0.5, 4.5))
        ):
            intervals.append(
                PhysicalTransferInterval(
                    transfer_id="calibration-send-{:08d}".format(logical),
                    iteration=iteration,
                    send=object(),
                    receive=object(),
                    local=None,
                    physical_start=AlignedTimestamp(start + 100.0, 50.0),
                    physical_end=AlignedTimestamp(end + 200.0, 50.0),
                    endpoint_order_uncertain=True,
                    sender_start=AlignedTimestamp(start, 50.0),
                    sender_end=AlignedTimestamp(end, 50.0),
                )
            )
    analysis = TraceAnalysis(tuple(intervals), (), (), (), False)
    request = CalibrationRequest("inter_node", 64 * 1024 * 1024, 2, "float")

    point = calibration_point_from_trace(request, 2, analysis)

    assert point.duration_statistics.p95_us == 5.0


def test_calibration_point_rejects_unknown_nonlocal_transfer():
    intervals = [
        PhysicalTransferInterval(
            transfer_id="calibration-send-00000000",
            iteration=iteration,
            send=None,
            receive=None,
            local=None,
            physical_start=AlignedTimestamp(0.0, 0.0),
            physical_end=AlignedTimestamp(5.0, 0.0),
            endpoint_order_uncertain=False,
            sender_start=AlignedTimestamp(0.0, 0.0),
            sender_end=AlignedTimestamp(5.0, 0.0),
        )
        for iteration in range(20)
    ]
    intervals.append(
        PhysicalTransferInterval(
            transfer_id="unexpected-network-transfer",
            iteration=0,
            send=object(),
            receive=object(),
            local=None,
            physical_start=AlignedTimestamp(0.0, 0.0),
            physical_end=AlignedTimestamp(1.0, 0.0),
            endpoint_order_uncertain=False,
            sender_start=AlignedTimestamp(0.0, 0.0),
            sender_end=AlignedTimestamp(1.0, 0.0),
        )
    )
    analysis = TraceAnalysis(tuple(intervals), (), (), (), True)
    request = CalibrationRequest("intra_node", 128 * 1024 * 1024, 1, "float")

    with pytest.raises(SemanticError, match="unknown transfer"):
        calibration_point_from_trace(request, 1, analysis)


def test_calibration_point_rejects_missing_full_wave_interval():
    analysis = TraceAnalysis((), (), (), (), True)
    request = CalibrationRequest("intra_node", 32 * 1024 * 1024, 2, "float")

    with pytest.raises(SemanticError, match="trace"):
        calibration_point_from_trace(request, 2, analysis)


def test_calibration_models_reject_inconsistent_boundaries():
    request = CalibrationRequest("intra_node", 1024, 4, "float")
    for changes in (
        {"link_class": "invalid"},
        {"slice_size_bytes": 0},
        {"max_calibration_channels": 0},
        {"datatype": ""},
    ):
        with pytest.raises(SemanticError):
            replace(request, **changes)

    point = _point(1, 12.0)
    for changes in (
        {"concurrency": 0},
        {"duration_statistics": object()},
        {"full_wave_count": 0},
        {"tail_transfer_count": -1},
    ):
        with pytest.raises(SemanticError):
            replace(point, **changes)

    curve = derive_calibrated_curve(2.0, 1000, (point,))
    valid = CalibrationResult(
        request=request,
        points=(point,),
        curve=curve,
        skipped_reason=None,
    )
    assert valid.stable is True
    with pytest.raises(SemanticError, match="request"):
        replace(valid, request=object())
    with pytest.raises(SemanticError, match="curve"):
        replace(valid, curve=None)

    skipped = CalibrationResult(
        request=request,
        points=(),
        curve=None,
        skipped_reason="slice_size_not_divisible",
    )
    assert skipped.stable is False
    with pytest.raises(SemanticError, match="skipped"):
        replace(skipped, points=(point,))
