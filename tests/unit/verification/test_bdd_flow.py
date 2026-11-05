from dataclasses import replace

import pytest

from vericcl.topology.model import LaneKey
from vericcl.errors import SemanticError
from vericcl.semantics.atom import PathStage, Symbol
from vericcl.verification.bdd_backend import (
    BDDAnalysisResult,
    BDDBackendError,
    CompactBDD,
)
from vericcl.verification.bdd_flow import analyze_flow_congestion
from vericcl.verification.model import ValidationStatus

from tests.unit.verification.bdd_helpers import (
    _transfer,
    crossing_flows_with_ready_wait,
    crossing_inputs,
    crossing_topology,
)


pytestmark = pytest.mark.phase05


def test_compact_bdd_supports_required_set_operations():
    backend = CompactBDD({"flow_id": 3, "lane_id": 2})
    left = backend.relation(((0, 0), (1, 0)))
    right = backend.relation(((1, 0), (2, 1)))

    assert left.union(right).tuples() == ((0, 0), (1, 0), (2, 1))
    assert left.intersection(right).tuples() == ((1, 0),)
    assert left.difference(right).tuples() == ((0, 0),)
    assert left.complement().tuples() == (
        (0, 1),
        (1, 1),
        (2, 0),
        (2, 1),
    )
    assert backend.fields == ("flow_id", "lane_id")
    with pytest.raises(BDDBackendError, match="unknown"):
        left._binary(right, "unknown")


@pytest.mark.parametrize(
    "domains",
    (
        {},
        {"": 1},
        {"flow_id": 0},
        {"flow_id": True},
    ),
)
def test_compact_bdd_rejects_invalid_domains(domains):
    with pytest.raises(BDDBackendError):
        CompactBDD(domains)


@pytest.mark.parametrize(
    "rows",
    (
        ((0,),),
        ((0, 2),),
        ((False, 0),),
        None,
    ),
)
def test_compact_bdd_rejects_invalid_relation_rows(rows):
    backend = CompactBDD({"flow_id": 2, "lane_id": 2})
    with pytest.raises(BDDBackendError):
        backend.relation(rows)


def test_relations_from_different_backends_cannot_be_combined():
    left = CompactBDD({"flow_id": 1}).relation(((0,),))
    right = CompactBDD({"flow_id": 1}).relation(((0,),))

    with pytest.raises(BDDBackendError, match="same backend"):
        left.union(right)


def test_bdd_analysis_result_rejects_non_analysis_status_and_evidence():
    with pytest.raises(SemanticError, match="status"):
        BDDAnalysisResult(
            ValidationStatus.INVALID,
            "invalid",
            "invalid",
            (),
            {},
        )
    with pytest.raises(SemanticError, match="evidence"):
        BDDAnalysisResult(
            ValidationStatus.VALID,
            "valid",
            "valid",
            (),
            object(),
        )


def test_ready_wait_with_earlier_idle_route_produces_replacement_hint():
    result = analyze_flow_congestion(
        crossing_flows_with_ready_wait(),
        crossing_topology(),
        crossing_inputs(),
    )

    assert result.status is ValidationStatus.VALID
    assert len(result.hints) == 1
    hint = result.hints[0]
    assert hint.source_flow_id
    assert hint.candidate_flow_ids
    assert hint.divergence_rank == 1
    assert hint.waiting_transfer_id == "wait-middle"
    assert hint.bottleneck_lane == LaneKey(1, 3, 0)
    assert hint.wait_interval_us == pytest.approx((1.0, 5.0))
    assert hint.earliest_candidate_start_us == pytest.approx(1.0)
    assert set(hint.candidate_paths.values()) == {(1, 2, 3)}


def test_bdd_flow_identity_comes_from_instantiated_schedule_not_template():
    schedule = crossing_flows_with_ready_wait()
    schedule = replace(
        schedule,
        metadata={
            **schedule.metadata,
            "template_id": "template-abstract-route",
            "template_member_id": "unit-abstract-member",
        },
    )

    result = analyze_flow_congestion(
        schedule,
        crossing_topology(),
        crossing_inputs(),
    )

    hint = next(
        value
        for value in result.hints
        if value.waiting_transfer_id == "wait-middle"
    )
    assert hint.source_flow_id == (
        "flow-s00000000-a00000000-m00000000-r00000000-l00000003-p0000"
    )
    assert hint.demand_id == (
        "demand-s00000000-send-a00000000-r00000000-l00000003"
    )
    assert hint.divergence_rank == 1
    assert hint.bottleneck_lane == LaneKey(1, 3, 0)
    assert hint.wait_interval_us == pytest.approx((1.0, 5.0))
    assert all(
        "template" not in value
        for value in (
            hint.source_flow_id,
            hint.demand_id,
            *hint.candidate_flow_ids,
        )
    )


def test_forbidden_member_filters_the_alternative_route():
    result = analyze_flow_congestion(
        crossing_flows_with_ready_wait(),
        crossing_topology(),
        crossing_inputs(forbid_alternative=True),
    )

    assert result.status is ValidationStatus.VALID
    assert result.hints == ()


def test_compatible_route_without_idle_window_is_removed_by_bdd_intersection():
    schedule = crossing_flows_with_ready_wait()
    blocker = _transfer(
        "alternative-blocker",
        1,
        2,
        0,
        {1: (PathStage(0, "SEND", (Symbol(1, 2, 0.0),)),)},
        1.0,
        5.0,
    )
    metadata = dict(schedule.metadata)
    semantic = dict(metadata["semantic_predecessors"])
    semantic[blocker.transfer_id] = ()
    metadata["semantic_predecessors"] = semantic
    blocked = replace(
        schedule,
        transfers=schedule.transfers + (blocker,),
        metadata=metadata,
    )

    result = analyze_flow_congestion(
        blocked,
        crossing_topology(),
        crossing_inputs(),
    )

    assert result.status is ValidationStatus.VALID
    assert result.hints == ()
    assert result.evidence["candidate_count"] == 1
    assert result.evidence["relation_count"] == 0


def test_root_ready_delay_is_reported_when_alternative_flow_can_start_earlier():
    schedule = crossing_flows_with_ready_wait()
    first = schedule.transfers[0]
    delayed_atom = replace(first.atoms[0], st_time=2.0, ed_time=3.0)
    delayed = replace(
        first,
        atoms=(delayed_atom,),
        st_time=2.0,
        ed_time=3.0,
    )
    value = replace(
        schedule,
        transfers=(delayed,) + schedule.transfers[1:],
    )

    result = analyze_flow_congestion(
        value,
        crossing_topology(),
        crossing_inputs(),
    )

    root_hint = next(
        hint for hint in result.hints if hint.waiting_transfer_id == "wait-first"
    )
    assert root_hint.divergence_rank == 0
    assert root_hint.wait_interval_us == pytest.approx((0.0, 2.0))
    assert root_hint.earliest_candidate_start_us == pytest.approx(0.0)
    assert all(
        lane != root_hint.bottleneck_lane
        for lane in root_hint.candidate_first_lanes.values()
    )


def test_backend_exception_becomes_analysis_error(monkeypatch):
    def fail_backend(*args, **kwargs):
        raise RuntimeError("backend failed")

    monkeypatch.setattr(
        "vericcl.verification.bdd_flow.CompactBDD",
        fail_backend,
    )
    result = analyze_flow_congestion(
        crossing_flows_with_ready_wait(),
        crossing_topology(),
        crossing_inputs(),
    )

    assert result.status is ValidationStatus.ANALYSIS_ERROR
    assert result.hints == ()
    assert result.code == "flow_bdd_analysis_error"
