import hashlib
import json
from dataclasses import replace
import time

import pytest

from vericcl.artifacts.layout import create_run_layout
from vericcl.artifacts.writer import (
    atomic_write_bytes,
    atomic_write_text,
    load_schedule_sidecar,
    read_schedule_sidecar,
    write_candidate_artifact,
    write_final_alias,
    write_resolved_input,
)
from vericcl.errors import SemanticError
from vericcl.input.models import ObjectiveMode
from vericcl.solver.model import (
    SearchDiagnostics,
    SolveCandidate,
    SolveStatus,
    SolverMetrics,
)
from vericcl.verification.model import CheckResult, ValidationStatus
from vericcl.verification.pipeline import validate_and_lower_candidate
from vericcl.verification.online.pipeline import (
    OnlineStageStatus,
    OnlineTuningEvidence,
    OnlineValidationResult,
)
from vericcl.verification.online.statistics import (
    PerformanceHistory,
    summarize_runs,
)
from vericcl.topology.model import LaneKey
from vericcl.verification.online.trace_analysis import (
    BottleneckRecord,
    TraceAnalysis,
    WaitClass,
)
from vericcl.xml.endpoints import EndpointType
from vericcl.tuning.engine import (
    CandidateProposal,
    TuningHistoryEntry,
    TuningResult,
)
from vericcl.tuning.model import TuningOverlay
import vericcl.workflow as workflow_module
from vericcl.workflow import (
    _Deadline,
    _run_online_candidate,
    _tuned_candidate,
    _tuning_records,
    _with_online_result,
    _online_evidence,
)

from tests.unit.verification.helpers import inputs, topology
from tests.unit.tuning.helpers import overlay
from tests.unit.xml.helpers import two_rank_allreduce_schedule


pytestmark = pytest.mark.phase07


def _candidate(schedule):
    return SolveCandidate(
        candidate_id="candidate-0",
        node_schedules={"global": schedule},
        objective_mode=ObjectiveMode.LATENCY,
        channel_count=1,
        metrics=SolverMetrics(
            status=SolveStatus.FEASIBLE,
            objective_values=(2.0, 2.0, 2.0),
            best_bound=1.0,
            mip_gap=0.5,
            within_requested_gap=False,
            solve_time_s=0.25,
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
        selected_best=True,
        proven_optimal=False,
        search_space_restricted=True,
        restrictions=("constructive_candidate",),
        parent_candidate_id=None,
    )


def _runtime_incompatible(outcome):
    runtime = CheckResult(
        dimension="runtime",
        status=ValidationStatus.WARNING,
        code="msccl_runtime_incompatible",
        message="MSCCL execution compatibility limits were exceeded",
        evidence={"issues": ({"code": "tb_step_limit"},)},
    )
    return replace(
        outcome,
        report=replace(outcome.report, runtime=runtime),
        artifact=replace(outcome.artifact, runtime_compatible=False),
    )


def test_writer_emits_bound_xml_report_and_schedule_sidecar(tmp_path):
    input_value = inputs()
    topology_value = topology()
    schedule = two_rank_allreduce_schedule()
    outcome = validate_and_lower_candidate(
        schedule,
        input_value,
        topology_value,
    )
    layout = create_run_layout(tmp_path, input_value, run_id="writer")
    write_resolved_input(layout, input_value)

    artifact = write_candidate_artifact(
        layout,
        input_value,
        topology_value,
        _candidate(schedule),
        schedule,
        outcome,
        iteration=0,
        selected_best=True,
        accepted=True,
        rejection_reason=None,
    )

    assert artifact.xml_path.suffix == ".xml"
    assert artifact.xml_path.name.endswith("selected-best-true.xml")
    assert "optimal" not in artifact.xml_path.name
    assert artifact.report_path.name.endswith(".validation.json")
    assert artifact.schedule_path.name.endswith(".schedule.json")
    assert load_schedule_sidecar(artifact.schedule_path) == schedule
    assert artifact.xml_sha256 == hashlib.sha256(
        artifact.xml_path.read_bytes()
    ).hexdigest()
    assert artifact.report_sha256 == hashlib.sha256(
        artifact.report_path.read_bytes()
    ).hexdigest()
    report = json.loads(artifact.report_path.read_text(encoding="utf-8"))
    assert report["xml_sha256"] == artifact.xml_sha256
    assert report["candidate_signature"] == artifact.candidate_signature
    assert report["artifact_binding_sha256"] == artifact.binding_sha256
    assert report["accepted"] is True
    assert report["iteration"] == 0
    assert json.loads(layout.resolved_input.read_text(encoding="utf-8"))[
        "input_sha256"
    ] == input_value.input_sha256


def test_writer_uses_candidate_suffix_and_final_alias_is_exact(tmp_path):
    input_value = inputs()
    topology_value = topology()
    schedule = two_rank_allreduce_schedule()
    outcome = _runtime_incompatible(
        validate_and_lower_candidate(schedule, input_value, topology_value)
    )
    layout = create_run_layout(tmp_path, input_value, run_id="candidate")

    artifact = write_candidate_artifact(
        layout,
        input_value,
        topology_value,
        _candidate(schedule),
        schedule,
        outcome,
        iteration=7,
        selected_best=False,
        accepted=False,
        rejection_reason="runtime_incompatible",
    )
    final_xml, final_report = write_final_alias(layout, artifact)

    assert artifact.xml_path.name.endswith(".candidate.xml")
    assert final_xml.name.endswith("_final.candidate.xml")
    assert final_xml.read_bytes() == artifact.xml_path.read_bytes()
    assert final_report.read_bytes() == artifact.report_path.read_bytes()
    final_schedule = layout.root / "vericcl_allreduce_1KiB_final.schedule.json"
    assert final_schedule.read_bytes() == artifact.schedule_path.read_bytes()
    assert artifact.runtime_compatible is False


def test_writer_never_overwrites_an_existing_iteration(tmp_path):
    input_value = inputs()
    topology_value = topology()
    schedule = two_rank_allreduce_schedule()
    outcome = validate_and_lower_candidate(
        schedule,
        input_value,
        topology_value,
    )
    layout = create_run_layout(tmp_path, input_value, run_id="collision")
    arguments = (
        layout,
        input_value,
        topology_value,
        _candidate(schedule),
        schedule,
        outcome,
    )
    write_candidate_artifact(
        *arguments,
        iteration=0,
        selected_best=True,
        accepted=True,
        rejection_reason=None,
    )

    with pytest.raises(FileExistsError, match="already exists"):
        write_candidate_artifact(
            *arguments,
            iteration=0,
            selected_best=True,
            accepted=True,
            rejection_reason=None,
        )


def test_schedule_sidecar_preserves_tuning_overlay_identity(tmp_path):
    input_value = inputs()
    topology_value = topology()
    schedule = two_rank_allreduce_schedule()
    outcome = validate_and_lower_candidate(
        schedule,
        input_value,
        topology_value,
    )
    layout = create_run_layout(tmp_path, input_value, run_id="overlay")
    value = overlay()

    artifact = write_candidate_artifact(
        layout,
        input_value,
        topology_value,
        _candidate(schedule),
        schedule,
        outcome,
        iteration=1,
        selected_best=True,
        accepted=True,
        rejection_reason=None,
        overlay=value,
    )
    sidecar = read_schedule_sidecar(artifact.schedule_path)

    assert sidecar.overlay == value
    assert sidecar.candidate_signature == artifact.candidate_signature


def test_writer_round_trips_diagnostics_and_old_sidecar_defaults(tmp_path):
    input_value = inputs()
    topology_value = topology()
    schedule = two_rank_allreduce_schedule()
    outcome = validate_and_lower_candidate(
        schedule,
        input_value,
        topology_value,
    )
    layout = create_run_layout(tmp_path, input_value, run_id="diagnostics")
    diagnostics = SearchDiagnostics(
        requested_problem_count=8,
        template_count=2,
        template_member_count=8,
        route_model_count=4,
        fallback_member_model_count=1,
        route_model_build_time_s=0.1,
        route_model_optimize_time_s=0.2,
        expansion_time_s=0.3,
        scheduling_time_s=0.4,
        maximum_variable_count=10,
        maximum_constraint_count=20,
        maximum_general_constraint_count=1,
    )

    artifact = write_candidate_artifact(
        layout,
        input_value,
        topology_value,
        _candidate(schedule),
        schedule,
        outcome,
        iteration=0,
        selected_best=True,
        accepted=True,
        rejection_reason=None,
        search_diagnostics=diagnostics,
    )

    report = json.loads(artifact.report_path.read_text(encoding="utf-8"))
    assert report["search_diagnostics"]["route_model_count"] == 4
    assert artifact.search_diagnostics == diagnostics
    assert read_schedule_sidecar(artifact.schedule_path).diagnostics == diagnostics

    payload = json.loads(artifact.schedule_path.read_text(encoding="utf-8"))
    del payload["search_diagnostics"]
    artifact.schedule_path.write_text(json.dumps(payload), encoding="utf-8")
    assert read_schedule_sidecar(artifact.schedule_path).diagnostics == (
        SearchDiagnostics()
    )


def test_atomic_writer_rejects_invalid_values_and_existing_target(tmp_path):
    target = tmp_path / "artifact.txt"
    with pytest.raises(SemanticError, match="path"):
        atomic_write_bytes("artifact.txt", b"data")
    with pytest.raises(SemanticError, match="data"):
        atomic_write_bytes(target, "data")
    with pytest.raises(SemanticError, match="text"):
        atomic_write_text(target, b"data")

    atomic_write_text(target, "preserve")
    with pytest.raises(FileExistsError, match="already exists"):
        atomic_write_text(target, "replace")
    assert target.read_text(encoding="utf-8") == "preserve"


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda value: value.update(schema_version=2), "schema version"),
        (lambda value: value.update(schedule="invalid"), "schedule is invalid"),
        (
            lambda value: value["schedule"].update(transfers=["invalid"]),
            "transfer is invalid",
        ),
        (
            lambda value: value["schedule"]["transfers"][0]["atoms"].__setitem__(
                0,
                "invalid",
            ),
            "atom is invalid",
        ),
        (
            lambda value: value["schedule"]["transfers"][0]["atoms"][0][
                "path"
            ].__setitem__(0, "invalid"),
            "path stage is invalid",
        ),
        (
            lambda value: value["schedule"]["transfers"][0]["atoms"][0][
                "path"
            ][0]["symbols"].__setitem__(0, "invalid"),
            "symbol is invalid",
        ),
        (
            lambda value: value.update(candidate="invalid"),
            "candidate is invalid",
        ),
        (
            lambda value: value["candidate"].update(objective_mode="invalid"),
            "objective mode",
        ),
        (
            lambda value: value["candidate"].update(metrics="invalid"),
            "solver metrics",
        ),
        (
            lambda value: value["candidate"]["metrics"].update(
                status="invalid"
            ),
            "solver status",
        ),
        (lambda value: value.update(overlay="invalid"), "overlay is invalid"),
    ),
)
def test_schedule_sidecar_rejects_corrupt_typed_fields(
    tmp_path,
    mutation,
    message,
):
    input_value = inputs()
    topology_value = topology()
    schedule = two_rank_allreduce_schedule()
    outcome = validate_and_lower_candidate(
        schedule,
        input_value,
        topology_value,
    )
    layout = create_run_layout(tmp_path, input_value, run_id="corrupt")
    artifact = write_candidate_artifact(
        layout,
        input_value,
        topology_value,
        _candidate(schedule),
        schedule,
        outcome,
        iteration=0,
        selected_best=True,
        accepted=True,
        rejection_reason=None,
    )
    payload = json.loads(artifact.schedule_path.read_text(encoding="utf-8"))
    mutation(payload)
    artifact.schedule_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SemanticError, match=message):
        read_schedule_sidecar(artifact.schedule_path)


def test_tuned_candidate_records_overlay_channel_and_restricted_lineage():
    schedule = two_rank_allreduce_schedule()
    input_value = inputs()
    topology_value = topology()
    outcome = validate_and_lower_candidate(
        schedule,
        input_value,
        topology_value,
    )
    initial = _candidate(schedule)
    value = overlay()
    entry = TuningHistoryEntry(
        candidate_id="candidate-tuned",
        parent_candidate_id="parent",
        schedule=schedule,
        overlay=value,
        tuning_strategy={"kind": "flow_suffix"},
        candidate_signature="0" * 64,
        report=outcome.report,
        artifact=outcome.artifact,
        simulation_time_us=1.5,
        online_performance=None,
        accepted=True,
        rejection_reason=None,
        selected_best=True,
        offline_analysis_only=False,
        actual_improvement=0.25,
        required_improvement=0.01,
        outcome=outcome,
    )

    candidate = _tuned_candidate(initial, entry)

    assert candidate.candidate_id == "candidate-tuned"
    assert candidate.channel_count == 1
    assert candidate.parent_candidate_id == "parent"
    assert candidate.metrics.solver_name == "vericcl-tuner"
    assert candidate.search_space_restricted is True


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"candidate_id": ""}, "ID"),
        ({"iteration": True}, "iteration must be an integer"),
        ({"iteration": -1}, "iteration must be non-negative"),
        ({"xml_path": "invalid"}, "XML path"),
        ({"report_path": "invalid"}, "paths are invalid"),
        ({"validation": []}, "validation is invalid"),
    ),
)
def test_candidate_artifact_rejects_invalid_public_fields(
    tmp_path,
    changes,
    message,
):
    input_value = inputs()
    topology_value = topology()
    schedule = two_rank_allreduce_schedule()
    outcome = validate_and_lower_candidate(
        schedule,
        input_value,
        topology_value,
    )
    layout = create_run_layout(tmp_path, input_value, run_id="fields")
    artifact = write_candidate_artifact(
        layout,
        input_value,
        topology_value,
        _candidate(schedule),
        schedule,
        outcome,
        iteration=0,
        selected_best=True,
        accepted=True,
        rejection_reason=None,
    )

    with pytest.raises(SemanticError, match=message):
        replace(artifact, **changes)


def test_tuning_record_adapter_assesses_and_serializes_derived_candidate(
    monkeypatch,
):
    schedule = two_rank_allreduce_schedule()
    input_value = inputs()
    topology_value = topology()
    outcome = validate_and_lower_candidate(
        schedule,
        input_value,
        topology_value,
    )
    initial = _candidate(schedule)
    value = TuningOverlay(
        overlay_id="derived-overlay",
        parent_candidate_id=initial.candidate_id,
        channel_count=1,
    )

    def fake_tune(candidate, context):
        initial_proposal = CandidateProposal(
            candidate_id=candidate.candidate_id,
            schedule=schedule,
            overlay=None,
            parent_candidate_id=candidate.parent_candidate_id,
            tuning_strategy={"kind": "initial"},
        )
        derived_proposal = CandidateProposal(
            candidate_id="candidate-derived",
            schedule=schedule,
            overlay=value,
            parent_candidate_id=candidate.candidate_id,
            tuning_strategy={"kind": "flow_suffix"},
        )
        initial_assessment = context.assess(initial_proposal)
        context.assess(derived_proposal)
        history = (
            TuningHistoryEntry(
                candidate_id=candidate.candidate_id,
                parent_candidate_id=candidate.parent_candidate_id,
                schedule=schedule,
                overlay=None,
                tuning_strategy={"kind": "initial"},
                candidate_signature="0" * 64,
                report=initial_assessment.report,
                artifact=initial_assessment.artifact,
                simulation_time_us=initial_assessment.simulation_time_us,
                online_performance=None,
                accepted=True,
                rejection_reason=None,
                selected_best=True,
                offline_analysis_only=False,
                actual_improvement=None,
                required_improvement=None,
                outcome=initial_assessment.outcome,
            ),
            TuningHistoryEntry(
                candidate_id="candidate-derived",
                parent_candidate_id=candidate.candidate_id,
                schedule=schedule,
                overlay=value,
                tuning_strategy={"kind": "flow_suffix"},
                candidate_signature="1" * 64,
                report=None,
                artifact=None,
                simulation_time_us=1.0,
                online_performance=None,
                accepted=False,
                rejection_reason="no_improvement",
                selected_best=False,
                offline_analysis_only=False,
                actual_improvement=0.0,
                required_improvement=0.01,
                outcome=None,
            ),
        )
        return TuningResult(
            selected_candidate_id=candidate.candidate_id,
            selected_schedule=schedule,
            selected_artifact=initial_assessment.artifact,
            history=history,
            stop_reason="candidate_space_exhausted",
            iterations=1,
        )

    monkeypatch.setattr(workflow_module, "tune", fake_tune)
    result, records = _tuning_records(
        initial,
        schedule,
        outcome,
        input_value,
        topology_value,
        _Deadline(10.0, time.monotonic()),
    )

    assert result.selected_candidate_id == initial.candidate_id
    assert len(records) == 1
    assert records[0].candidate.candidate_id == "candidate-derived"
    assert records[0].overlay == value
    assert records[0].rejection_reason == "no_improvement"


def test_tuning_record_adapter_attaches_stable_online_evidence(
    tmp_path,
    monkeypatch,
):
    schedule = two_rank_allreduce_schedule()
    input_value = inputs()
    topology_value = topology()
    outcome = validate_and_lower_candidate(
        schedule,
        input_value,
        topology_value,
    )
    initial = _candidate(schedule)
    history = PerformanceHistory((summarize_runs((10.0,) * 20),))
    online = OnlineValidationResult(
        context_schedule=schedule,
        preflight_status=OnlineStageStatus.PASSED,
        calibration_status=OnlineStageStatus.NOT_RUN,
        release_status=OnlineStageStatus.PASSED,
        online_operator_validation=OnlineStageStatus.PASSED,
        failure_code=None,
        failure_message=None,
        runtime_environment={},
        release_history=history,
        calibration=None,
        trace_analysis=None,
        trace_rank_files=(),
        trace_clock_uncertainty_us=1.0,
        requires_resolve=False,
        online_tuning_allowed=True,
        tuning_evidence=OnlineTuningEvidence({}, ()),
    )
    online_outcome = _with_online_result(outcome, online)

    def fake_tune(candidate, context):
        proposal = CandidateProposal(
            candidate_id=candidate.candidate_id,
            schedule=schedule,
            overlay=None,
            parent_candidate_id=candidate.parent_candidate_id,
            tuning_strategy={"kind": "initial"},
        )
        assessment = context.assess(proposal)
        assert assessment.online_performance.median_time_us == 10.0
        entry = TuningHistoryEntry(
            candidate_id=candidate.candidate_id,
            parent_candidate_id=candidate.parent_candidate_id,
            schedule=schedule,
            overlay=None,
            tuning_strategy={"kind": "initial"},
            candidate_signature="0" * 64,
            report=assessment.report,
            artifact=assessment.artifact,
            simulation_time_us=assessment.simulation_time_us,
            online_performance=assessment.online_performance,
            accepted=True,
            rejection_reason=None,
            selected_best=True,
            offline_analysis_only=False,
            actual_improvement=None,
            required_improvement=None,
            outcome=assessment.outcome,
        )
        return TuningResult(
            selected_candidate_id=candidate.candidate_id,
            selected_schedule=schedule,
            selected_artifact=assessment.artifact,
            history=(entry,),
            stop_reason="candidate_space_exhausted",
            iterations=0,
        )

    monkeypatch.setattr(workflow_module, "tune", fake_tune)
    layout = create_run_layout(tmp_path, input_value, run_id="online-tune")
    result, records = _tuning_records(
        initial,
        schedule,
        online_outcome,
        input_value,
        topology_value,
        _Deadline(10.0, time.monotonic()),
        online_result=online,
        online_factory=lambda *args: None,
        layout=layout,
    )

    assert result.selected_candidate_id == initial.candidate_id
    assert records == ()


def test_online_failure_code_is_not_reported_as_valid():
    schedule = two_rank_allreduce_schedule()
    outcome = validate_and_lower_candidate(schedule, inputs(), topology())
    online = OnlineValidationResult(
        context_schedule=schedule,
        preflight_status=OnlineStageStatus.PASSED,
        calibration_status=OnlineStageStatus.FAILED,
        release_status=OnlineStageStatus.PASSED,
        online_operator_validation=OnlineStageStatus.PASSED,
        failure_code="calibration_failed",
        failure_message="calibration failed",
        runtime_environment={},
        release_history=PerformanceHistory(
            (summarize_runs((10.0,) * 20),)
        ),
        calibration=None,
        trace_analysis=TraceAnalysis((), (), (), (), True),
        trace_rank_files=(),
        trace_clock_uncertainty_us=1.0,
        requires_resolve=False,
        online_tuning_allowed=False,
        tuning_evidence=None,
    )

    updated = _with_online_result(outcome, online)

    assert updated.report.online.status is ValidationStatus.FAILED
    assert updated.report.online.code == "calibration_failed"


def test_online_evidence_contains_full_statistics_and_bottleneck_identity():
    schedule = two_rank_allreduce_schedule()
    bottleneck = BottleneckRecord(
        transfer_id="transfer-0",
        stage_id=1,
        endpoint_type=EndpointType.SEND,
        atom_ids=("atom-0",),
        flow_ids=("flow-0",),
        rank=0,
        tb_id=1,
        step_index=2,
        iteration=3,
        lane=LaneKey(0, 1, 0),
        wait_class=WaitClass.HEAD_OF_LINE,
        duration_us=4.0,
        ordering_confident=True,
    )
    result = OnlineValidationResult(
        context_schedule=schedule,
        preflight_status=OnlineStageStatus.PASSED,
        calibration_status=OnlineStageStatus.NOT_RUN,
        release_status=OnlineStageStatus.PASSED,
        online_operator_validation=OnlineStageStatus.PASSED,
        failure_code=None,
        failure_message=None,
        runtime_environment={},
        release_history=PerformanceHistory(
            (summarize_runs(tuple(float(value) for value in range(1, 21))),)
        ),
        calibration=None,
        trace_analysis=TraceAnalysis((), (), (bottleneck,), (), True),
        trace_rank_files=(),
        trace_clock_uncertainty_us=1.0,
        requires_resolve=False,
        online_tuning_allowed=True,
        tuning_evidence=OnlineTuningEvidence(
            {"transfer-0": 4.0},
            (bottleneck,),
        ),
    )

    evidence = _online_evidence(result)

    assert evidence["release_rounds"][0]["mean_us"] == pytest.approx(10.5)
    assert "population_standard_deviation_us" in evidence[
        "release_rounds"
    ][0]
    record = evidence["trace_analysis"]["bottlenecks"][0]
    assert record["flow_ids"] == ("flow-0",)
    assert record["rank"] == 0
    assert record["tb_id"] == 1
    assert record["step_index"] == 2
    assert record["lane"] == {
        "src_rank": 0,
        "dst_rank": 1,
        "channel": 0,
    }
    assert evidence["tuning_evidence"]["wait_us_by_transfer"] == {
        "transfer-0": 4.0
    }


def test_online_runtime_incompatibility_is_reported_without_launch(tmp_path):
    schedule = two_rank_allreduce_schedule()
    input_value = inputs()
    outcome = _runtime_incompatible(
        validate_and_lower_candidate(schedule, input_value, topology())
    )
    layout = create_run_layout(tmp_path, input_value, run_id="online-warning")

    updated, result = _run_online_candidate(
        candidate_id="candidate-warning",
        schedule=schedule,
        outcome=outcome,
        inputs=input_value,
        layout=layout,
        factory=lambda *args: pytest.fail("runtime launch must not run"),
        tuning_requested=False,
        deadline=_Deadline(10.0, time.monotonic()),
    )

    assert result is None
    assert updated.report.online.status is ValidationStatus.FAILED
    assert updated.report.online.code == "online_runtime_incompatible"
