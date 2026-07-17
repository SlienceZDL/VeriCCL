import json
from dataclasses import replace

import pytest

from vericcl.artifacts.hashing import (
    artifact_binding_sha256,
    candidate_signature,
    verify_artifact_binding,
)
from vericcl.artifacts.reports import (
    build_candidate_report,
    build_validation_json,
)
from vericcl.errors import SemanticError
from vericcl.verification.pipeline import VerificationOutcome
from vericcl.input.models import ObjectiveMode
from vericcl.solver.model import (
    SolveCandidate,
    SolveStatus,
    SolverMetrics,
)
from vericcl.verification.pipeline import validate_and_lower_candidate

from tests.unit.tuning.helpers import overlay
from tests.unit.verification.helpers import inputs, topology
from tests.unit.xml.helpers import two_rank_allreduce_schedule


pytestmark = pytest.mark.phase05


def _candidate(schedule):
    return SolveCandidate(
        candidate_id="report-candidate",
        node_schedules={"global": schedule},
        objective_mode=ObjectiveMode.LATENCY,
        channel_count=1,
        metrics=SolverMetrics(
            status=SolveStatus.FEASIBLE,
            objective_values=(2.0, 2.0, 2.0),
            best_bound=1.0,
            mip_gap=0.5,
            within_requested_gap=False,
            solve_time_s=1.0,
            model_count=1,
            operation_count=2,
            hop_count=2,
            makespan_us=2.0,
            maximum_normalized_resource_load=2.0,
            solver_name="test-solver",
            solver_version="1",
            solver_seed=0,
            thread_count=1,
            termination_reason="test_complete",
        ),
        selected_best=False,
        proven_optimal=False,
        search_space_restricted=True,
        restrictions=("shortest_paths",),
        parent_candidate_id="parent",
    )


def test_report_contains_reproducibility_validation_and_xml_binding():
    schedule = two_rank_allreduce_schedule()
    input_value = inputs()
    topology_value = topology()
    outcome = validate_and_lower_candidate(
        schedule,
        input_value,
        topology_value,
    )
    candidate = _candidate(schedule)
    value = build_candidate_report(
        candidate,
        input_value,
        topology_value,
        outcome,
        overlay=overlay(),
        applied_strategies={"milp": False, "constructive_trees": True},
        hierarchy_plan={"template": "direct", "groups": ((0, 1),)},
        rejection_reason=None,
        selected_best=True,
        tuning_strategy={"kind": "flow_suffix", "hint_id": "hint-0"},
    )
    decoded = json.loads(build_validation_json(value))

    assert decoded["normalized_input_sha256"] == input_value.input_sha256
    assert decoded["requested_strategies"]["milp"] is True
    assert decoded["applied_strategies"]["milp"] is False
    assert decoded["strategy_parameters"]["solver_seed"] == 0
    assert decoded["overlay"]["overlay_id"] == "repair-overlay"
    assert decoded["hierarchy_plan"]["template"] == "direct"
    assert decoded["channel_count"] == 1
    assert decoded["buffer_plan"]["slice_count"] == 1
    assert decoded["solver_metrics"]["status"] == "feasible"
    assert set(decoded["validation"]) == {
        "input",
        "semantic",
        "state",
        "topology",
        "timing",
        "resource",
        "buffer",
        "endpoint",
        "deadlock",
        "xml",
        "bdd",
        "simulation",
        "runtime",
        "online",
    }
    assert decoded["lineage"]["parent_candidate_id"] == "parent"
    assert decoded["rejection_reason"] is None
    assert decoded["selected_best"] is True
    assert decoded["proven_optimal"] is False
    assert decoded["search_space_restricted"] is True
    assert decoded["runtime_compatible"] is True
    assert decoded["xml_sha256"] == outcome.artifact.sha256
    assert decoded["bdd_evidence"]
    assert decoded["simulation_evidence"]
    assert decoded["tuning_strategy"]["kind"] == "flow_suffix"
    assert decoded["runtime_recommendations"] == []
    assert decoded["reproducibility"] == {
        "deterministic_artifacts": True,
        "limits": [
            "environment_signature",
            "hardware_measurement",
            "parallel_solver_execution",
            "solver_version",
        ],
        "solver_name": "test-solver",
        "solver_seed": 0,
        "solver_version": "1",
        "thread_count": 1,
    }
    assert verify_artifact_binding(
        value.artifact_binding_sha256,
        input_value.input_sha256,
        value.candidate_signature,
        outcome.artifact.sha256,
    )


def test_candidate_signature_and_binding_are_exact_and_deterministic():
    schedule = two_rank_allreduce_schedule()
    input_value = inputs()
    topology_value = topology()

    first = candidate_signature(
        schedule,
        input_value,
        topology_value,
        overlay(),
    )
    second = candidate_signature(
        schedule,
        input_value,
        topology_value,
        overlay(),
    )
    changed = candidate_signature(
        schedule,
        input_value,
        topology_value,
        None,
    )
    binding = artifact_binding_sha256(
        input_value.input_sha256,
        first,
        "0" * 64,
    )

    assert first == second
    assert first != changed
    renamed = candidate_signature(
        replace(schedule, schedule_id="renamed-schedule"),
        input_value,
        topology_value,
        replace(
            overlay(),
            overlay_id="renamed-overlay",
            parent_candidate_id="other-parent",
        ),
    )
    assert renamed == first
    assert verify_artifact_binding(
        binding,
        input_value.input_sha256,
        first,
        "0" * 64,
    )
    assert not verify_artifact_binding(
        binding,
        input_value.input_sha256,
        first,
        "1" * 64,
    )


def test_report_and_hashing_reject_invalid_boundaries():
    schedule = two_rank_allreduce_schedule()
    input_value = inputs()
    topology_value = topology()
    arguments = (schedule, input_value, topology_value, overlay())
    for index in range(len(arguments)):
        invalid = list(arguments)
        invalid[index] = object()
        with pytest.raises(SemanticError):
            candidate_signature(*invalid)

    with pytest.raises(SemanticError, match="SHA-256"):
        artifact_binding_sha256("short", "0" * 64, "0" * 64)
    with pytest.raises(SemanticError, match="SHA-256"):
        artifact_binding_sha256("z" * 64, "0" * 64, "0" * 64)
    assert not verify_artifact_binding("invalid", "invalid", "invalid", "invalid")
    with pytest.raises(SemanticError, match="report"):
        build_validation_json(object())

    outcome = validate_and_lower_candidate(
        schedule,
        input_value,
        topology_value,
    )
    candidate = _candidate(schedule)
    with pytest.raises(SemanticError, match="unknown field"):
        build_candidate_report(
            candidate,
            input_value,
            topology_value,
            outcome,
            overlay=None,
            applied_strategies={"unknown": True},
            hierarchy_plan={},
            rejection_reason=None,
            selected_best=False,
            tuning_strategy={},
        )
    without_artifact = VerificationOutcome(
        outcome.report,
        None,
        outcome.simulation,
        outcome.flow_bdd,
        outcome.order_bdd,
    )
    with pytest.raises(SemanticError, match="XML artifact"):
        build_candidate_report(
            candidate,
            input_value,
            topology_value,
            without_artifact,
            overlay=None,
            applied_strategies={},
            hierarchy_plan={},
            rejection_reason="invalid",
            selected_best=False,
            tuning_strategy={},
        )

    value = build_candidate_report(
        candidate,
        input_value,
        topology_value,
        outcome,
        overlay=None,
        applied_strategies={},
        hierarchy_plan={},
        rejection_reason=None,
        selected_best=False,
        tuning_strategy={},
    )
    with pytest.raises(SemanticError, match="validation"):
        replace(value, validation=object())


def test_hierarchical_candidate_report_binds_explicit_global_schedule():
    schedule = two_rank_allreduce_schedule()
    input_value = inputs()
    topology_value = topology()
    outcome = validate_and_lower_candidate(
        schedule,
        input_value,
        topology_value,
    )
    candidate = replace(
        _candidate(schedule),
        node_schedules={"local-a": schedule, "local-b": schedule},
    )

    with pytest.raises(SemanticError, match="global schedule"):
        build_candidate_report(
            candidate,
            input_value,
            topology_value,
            outcome,
            overlay=None,
            applied_strategies={},
            hierarchy_plan={"template": "hierarchical"},
            rejection_reason=None,
            selected_best=False,
            tuning_strategy={},
        )

    report = build_candidate_report(
        candidate,
        input_value,
        topology_value,
        outcome,
        global_schedule=schedule,
        overlay=None,
        applied_strategies={},
        hierarchy_plan={"template": "hierarchical"},
        rejection_reason=None,
        selected_best=True,
        tuning_strategy={},
    )

    assert report.candidate_signature == candidate_signature(
        schedule,
        input_value,
        topology_value,
        None,
    )


def test_report_uses_candidate_overlay_channel_configuration():
    schedule = two_rank_allreduce_schedule()
    input_value = inputs()
    topology_value = topology()
    outcome = validate_and_lower_candidate(
        schedule,
        input_value,
        topology_value,
    )

    report = build_candidate_report(
        _candidate(schedule),
        input_value,
        topology_value,
        outcome,
        overlay=replace(overlay(), channel_count=2),
        applied_strategies={},
        hierarchy_plan={},
        rejection_reason=None,
        selected_best=False,
        tuning_strategy={},
    )

    assert report.channel_count == 2
