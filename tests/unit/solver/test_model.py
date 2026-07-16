from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import pytest

from vericcl.errors import SemanticError
from vericcl.input.loader import resolve_inputs
from vericcl.input.models import ObjectiveMode
from vericcl.input.models import ForbiddenTransfer
from vericcl.planner.build import build_plan
from vericcl.semantics.atom import Schedule
from vericcl.solver.model import (
    SolveCandidate,
    SolveRequest,
    SolveResult,
    SolveStatus,
    SolverMetrics,
)
from vericcl.topology.loader import load_topology
from vericcl.tuning.model import TuningOverlay


pytestmark = pytest.mark.phase03


EXAMPLES = Path(__file__).parents[3] / "vericcl" / "examples"


def request():
    inputs = resolve_inputs(
        EXAMPLES / "topo" / "two_rank.json",
        EXAMPLES / "sketch" / "allreduce_8m_1m.json",
        EXAMPLES / "atom" / "default.json",
    )
    topology = load_topology(inputs)
    return SolveRequest(
        inputs=inputs,
        topology=topology,
        plan=build_plan(inputs, topology),
        solver_version="test-solver-1",
        model_version="test-model-1",
        environment_signature="test-environment",
    )


def metrics(status=SolveStatus.FEASIBLE):
    return SolverMetrics(
        status=status,
        objective_values=(10.0, 3.0, 4.0),
        best_bound=9.0,
        mip_gap=0.1,
        within_requested_gap=False,
        solve_time_s=1.5,
        model_count=1,
        operation_count=3,
        hop_count=4,
        makespan_us=10.0,
        maximum_normalized_resource_load=0.5,
        solver_name="fake",
        solver_version="1",
        solver_seed=0,
        thread_count=1,
        termination_reason="complete",
    )


def schedule():
    return Schedule(
        schedule_id="node-schedule",
        transfers=(),
        final_state_ids=(),
        rank_count=2,
        slice_count=8,
        slice_size_bytes=1_048_576,
        metadata={},
    )


def candidate(
    *,
    selected_best=True,
    proven_optimal=False,
    status=SolveStatus.FEASIBLE,
):
    return SolveCandidate(
        candidate_id="candidate",
        node_schedules={"node": schedule()},
        objective_mode=ObjectiveMode.LATENCY,
        channel_count=1,
        metrics=metrics(status),
        selected_best=selected_best,
        proven_optimal=proven_optimal,
        search_space_restricted=False,
        restrictions=(),
        parent_candidate_id=None,
    )


def test_selected_best_is_distinct_from_proven_optimal():
    value = candidate(selected_best=True, proven_optimal=False)

    assert value.selected_best
    assert not value.proven_optimal


def test_candidate_schedule_mapping_is_immutable():
    value = candidate()

    assert isinstance(value.node_schedules, MappingProxyType)
    with pytest.raises(TypeError):
        value.node_schedules["other"] = schedule()


def test_proven_optimal_requires_optimal_solver_status():
    with pytest.raises(SemanticError, match="OPTIMAL"):
        candidate(proven_optimal=True, status=SolveStatus.FEASIBLE)

    value = candidate(proven_optimal=True, status=SolveStatus.OPTIMAL)
    assert value.proven_optimal


def test_restriction_flag_and_descriptions_must_agree():
    value = candidate()

    with pytest.raises(SemanticError, match="restrictions"):
        replace(value, search_space_restricted=True)
    with pytest.raises(SemanticError, match="restrictions"):
        replace(value, restrictions=("shortest_paths",))


def test_solve_result_identifies_selected_candidate():
    selected = candidate()
    other = replace(selected, candidate_id="other", selected_best=False)

    result = SolveResult(
        status=SolveStatus.FEASIBLE,
        candidates=(other, selected),
        selected_candidate_id="candidate",
        cache_hit=False,
        message="complete",
    )

    assert result.selected_candidate == selected


def test_solve_result_rejects_unknown_selection():
    with pytest.raises(SemanticError, match="selected"):
        SolveResult(
            status=SolveStatus.FEASIBLE,
            candidates=(candidate(),),
            selected_candidate_id="missing",
            cache_hit=False,
            message="invalid",
        )


def test_solve_request_requires_exact_rank_compatible_models():
    value = request()

    with pytest.raises(SemanticError, match="rank"):
        replace(value, inputs=replace(value.inputs, rank_count=3))


def test_tuning_overlay_normalizes_deterministic_fields():
    overlay = TuningOverlay(
        overlay_id="overlay",
        parent_candidate_id=None,
        channel_count=2,
        path_weights=(("b", 2.0), ("a", 1.0)),
        resolve_scope=("node-b", "node-a"),
    )

    assert overlay.path_weights == (("a", 1.0), ("b", 2.0))
    assert overlay.resolve_scope == ("node-a", "node-b")


def test_tuning_overlay_normalizes_all_structural_fields():
    forbidden = ForbiddenTransfer(
        slice_id=0,
        src_rank=0,
        dst_rank=1,
        stage_id=0,
    )
    overlay = TuningOverlay(
        overlay_id="overlay",
        parent_candidate_id="parent",
        channel_count=2,
        path_weights=(("path-b", 2), ("path-a", 1)),
        temporary_forbidden=frozenset({forbidden}),
        batch_size=4,
        tree_roots=((1, 0), (0, 1)),
        tree_edges=((1, 0, 2), (0, 1, 2)),
        lane_order=(("b", "c"), ("a", "b")),
        milp_parameters=(("TimeLimit", 10),),
        warm_start_candidate_id="warm",
        resolve_scope=("node-b", "node-a"),
        hierarchy_template="gateway",
    )

    assert overlay.tree_roots == ((0, 1), (1, 0))
    assert overlay.tree_edges == ((0, 1, 2), (1, 0, 2))
    assert overlay.lane_order == (("a", "b"), ("b", "c"))
    assert overlay.temporary_forbidden == frozenset({forbidden})


@pytest.mark.parametrize(
    "field,value",
    [
        ("overlay_id", ""),
        ("channel_count", 0),
        ("batch_size", 0),
        ("path_weights", (("path", float("nan")),)),
        ("resolve_scope", ("node", "node")),
    ],
)
def test_tuning_overlay_rejects_invalid_fields(field, value):
    overlay = TuningOverlay(overlay_id="overlay", parent_candidate_id=None)

    with pytest.raises(SemanticError):
        replace(overlay, **{field: value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("parent_candidate_id", ""),
        ("temporary_forbidden", frozenset({1})),
        ("tree_roots", None),
        ("tree_roots", ((0,),)),
        ("tree_roots", ((0, 1), (0, 1))),
        ("tree_edges", ((0, True, 1),)),
        ("lane_order", None),
        ("lane_order", (("a",),)),
        ("lane_order", (("a", "a"),)),
        ("lane_order", (("a", "b"), ("a", "b"))),
        ("path_weights", None),
        ("path_weights", (("path",),)),
        ("path_weights", (("path", 1), ("path", 2))),
        ("warm_start_candidate_id", ""),
        ("hierarchy_template", ""),
    ],
)
def test_tuning_overlay_rejects_invalid_structures(field, value):
    overlay = TuningOverlay(overlay_id="overlay", parent_candidate_id=None)

    with pytest.raises(SemanticError):
        replace(overlay, **{field: value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "feasible"),
        ("objective_values", (float("nan"),)),
        ("within_requested_gap", 1),
        ("operation_count", -1),
        ("solver_name", ""),
    ],
)
def test_solver_metrics_reject_invalid_fields(field, value):
    with pytest.raises(SemanticError):
        replace(metrics(), **{field: value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("candidate_id", ""),
        ("node_schedules", 1),
        ("node_schedules", {"node": object()}),
        ("objective_mode", "latency"),
        ("channel_count", 0),
        ("metrics", object()),
        ("selected_best", 1),
        ("restrictions", ("same", "same")),
        ("parent_candidate_id", ""),
    ],
)
def test_candidate_rejects_invalid_fields(field, value):
    with pytest.raises(SemanticError):
        replace(candidate(), **{field: value})


def test_restricted_candidate_cannot_claim_global_proof():
    with pytest.raises(SemanticError, match="restricted"):
        replace(
            candidate(proven_optimal=True, status=SolveStatus.OPTIMAL),
            search_space_restricted=True,
            restrictions=("shortest_paths",),
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "feasible"),
        ("candidates", (object(),)),
        ("cache_hit", 1),
        ("message", None),
    ],
)
def test_solve_result_rejects_invalid_fields(field, value):
    result = SolveResult(
        status=SolveStatus.FEASIBLE,
        candidates=(candidate(),),
        selected_candidate_id="candidate",
        cache_hit=False,
        message="complete",
    )

    with pytest.raises(SemanticError):
        replace(result, **{field: value})


def test_solve_result_rejects_duplicate_candidate_ids():
    value = candidate()

    with pytest.raises(SemanticError, match="unique"):
        SolveResult(
            status=SolveStatus.FEASIBLE,
            candidates=(value, value),
            selected_candidate_id=None,
            cache_hit=False,
            message="duplicate",
        )
