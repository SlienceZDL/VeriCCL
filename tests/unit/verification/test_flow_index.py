import pytest

from vericcl.errors import SemanticError
from vericcl.semantics.atom import Atom, PathStage, Schedule, Symbol, Transfer
from vericcl.topology.model import LaneKey
from vericcl.verification.flow_index import (
    FlowIndex,
    LaneInterval,
    LaneState,
    build_flow_index,
)

from tests.unit.verification.bdd_helpers import (
    crossing_flows_with_ready_wait,
    two_members_with_shared_suffix,
)


pytestmark = pytest.mark.phase05


def test_bdd_interfaces_are_public_verification_exports():
    from vericcl.verification import (
        BDDAnalysisResult,
        FlowRecord,
        FlowReplacementHint,
        LaneState,
        TBOrderHint,
        analyze_flow_congestion,
        analyze_tb_order,
        build_flow_index,
    )

    assert all(
        value is not None
        for value in (
            BDDAnalysisResult,
            FlowRecord,
            FlowReplacementHint,
            LaneState,
            TBOrderHint,
            analyze_flow_congestion,
            analyze_tb_order,
            build_flow_index,
        )
    )


def test_member_flows_stop_comparing_after_first_aggregate_merge():
    index = build_flow_index(two_members_with_shared_suffix())
    merged = tuple(
        flow for flow in index.flows if flow.stage_id == 1
    )

    assert len(merged) == 2
    assert merged[0].comparison_end == merged[1].comparison_end == 0
    assert index.shared_suffix_transfer_ids == frozenset({"tx-shared"})


def test_different_root_leaf_flows_share_one_global_lane_state():
    index = build_flow_index(crossing_flows_with_ready_wait())
    lane = index.lane(LaneKey(1, 3, 0))

    assert lane.transfer_ids == ("cross-middle", "wait-middle")
    assert {
        (flow.root_rank, flow.leaf_rank)
        for flow in index.flows
        if LaneKey(1, 3, 0) in flow.lanes
    } == {(0, 3), (4, 5)}


def test_lane_state_finds_only_windows_large_enough_for_transfer():
    lane = build_flow_index(crossing_flows_with_ready_wait()).lane(
        LaneKey(1, 3, 0)
    )

    assert lane.earliest_start(1.0, 5.0, 1.0) is None
    assert lane.earliest_start(0.0, 1.0, 1.0) == pytest.approx(0.0)


def test_lane_state_handles_empty_lanes_and_invalid_intervals():
    lane = LaneKey(0, 1, 0)
    state = LaneState(lane, ())

    assert state.earliest_start(2.0, 1.0, 0.0) is None
    assert state.earliest_start(1.0, 3.0, 2.0) == pytest.approx(1.0)
    with pytest.raises(SemanticError, match="start"):
        LaneInterval(2.0, 1.0, "tx")
    with pytest.raises(SemanticError, match="transfer ID"):
        LaneInterval(0.0, 1.0, "")
    with pytest.raises(SemanticError, match="LaneKey"):
        LaneState(object(), ())
    with pytest.raises(SemanticError, match="intervals"):
        LaneState(lane, (object(),))


def test_flow_index_rejects_unknown_and_duplicate_flow_ids():
    index = build_flow_index(crossing_flows_with_ready_wait())

    assert index.lane(LaneKey(2, 4, 0)).intervals == ()
    with pytest.raises(SemanticError, match="unknown flow"):
        index.flow("missing")
    with pytest.raises(SemanticError, match="unique"):
        FlowIndex((index.flows[0], index.flows[0]), {}, frozenset())


def test_flow_index_rejects_missing_path_operation():
    path = PathStage(
        0,
        "SEND",
        (Symbol(0, 1, 0.0), Symbol(1, 2, 1.0)),
    )
    atom = Atom(0, 1024, (path,), 1.0, 2.0)
    transfer = Transfer(
        "only-second",
        "SEND",
        1,
        2,
        0,
        0,
        frozenset({0}),
        (atom,),
        1.0,
        2.0,
        frozenset(),
    )
    schedule = Schedule(
        "missing-path-operation",
        (transfer,),
        (),
        3,
        1,
        1024,
        {"path_scope": "global"},
    )

    with pytest.raises(SemanticError, match="missing"):
        build_flow_index(schedule)


def test_flow_index_requires_schedule_instance():
    with pytest.raises(SemanticError, match="Schedule"):
        build_flow_index(object())
