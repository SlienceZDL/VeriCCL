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
from vericcl.tuning.model import TuningOverlay

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


def test_writer_reports_diagnostics_without_changing_stable_sidecar(tmp_path):
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
        route_model_count=4,
        fallback_member_model_count=1,
        search_model_count_total=6,
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
        hierarchy_plan={"planning_mode": "direct"},
        diagnostics=diagnostics,
        verification_time_s=0.75,
    )
    report = json.loads(artifact.report_path.read_text(encoding="utf-8"))
    sidecar = json.loads(artifact.schedule_path.read_text(encoding="utf-8"))

    assert report["route_model_count"] == 4
    assert report["fallback_member_model_count"] == 1
    assert report["search_model_count_total"] == 6
    assert report["verification_time_s"] == 0.75
    assert "diagnostics" not in sidecar
    assert "global_schedule" not in sidecar["candidate"]


def test_sidecar_bytes_ignore_solver_and_diagnostic_wall_clocks(tmp_path):
    input_value = inputs()
    topology_value = topology()
    schedule = two_rank_allreduce_schedule()
    outcome = validate_and_lower_candidate(
        schedule,
        input_value,
        topology_value,
    )
    first_layout = create_run_layout(
        tmp_path / "first",
        input_value,
        run_id="first",
    )
    second_layout = create_run_layout(
        tmp_path / "second",
        input_value,
        run_id="second",
    )
    first_candidate = _candidate(schedule)
    second_candidate = replace(
        first_candidate,
        metrics=replace(first_candidate.metrics, solve_time_s=91.0),
    )
    first = write_candidate_artifact(
        first_layout,
        input_value,
        topology_value,
        first_candidate,
        schedule,
        outcome,
        iteration=0,
        selected_best=True,
        accepted=True,
        rejection_reason=None,
        diagnostics=SearchDiagnostics(
            route_model_build_time_s=0.25,
            route_model_optimize_time_s=0.5,
        ),
        verification_time_s=0.75,
    )
    second = write_candidate_artifact(
        second_layout,
        input_value,
        topology_value,
        second_candidate,
        schedule,
        outcome,
        iteration=0,
        selected_best=True,
        accepted=True,
        rejection_reason=None,
        diagnostics=SearchDiagnostics(
            route_model_build_time_s=21.0,
            route_model_optimize_time_s=34.0,
        ),
        verification_time_s=55.0,
    )

    assert first.schedule_path.read_bytes() == second.schedule_path.read_bytes()
    first_report = json.loads(first.report_path.read_text(encoding="utf-8"))
    second_report = json.loads(second.report_path.read_text(encoding="utf-8"))
    assert first_report["solver_metrics"]["solve_time_s"] == 0.25
    assert second_report["solver_metrics"]["solve_time_s"] == 91.0
    assert first_report["verification_time_s"] == 0.75
    assert second_report["verification_time_s"] == 55.0


def test_sidecar_missing_solve_time_reads_as_zero(tmp_path):
    input_value = inputs()
    topology_value = topology()
    schedule = two_rank_allreduce_schedule()
    outcome = validate_and_lower_candidate(
        schedule,
        input_value,
        topology_value,
    )
    layout = create_run_layout(tmp_path, input_value, run_id="no-solve-time")
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
    assert "solve_time_s" not in payload["candidate"]["metrics"]
    artifact.schedule_path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    sidecar = read_schedule_sidecar(artifact.schedule_path)

    assert sidecar.candidate.metrics.solve_time_s == 0.0

    payload["candidate"]["metrics"]["solve_time_s"] = 4.5
    artifact.schedule_path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    legacy = read_schedule_sidecar(artifact.schedule_path)

    assert legacy.candidate.metrics.solve_time_s == 4.5


@pytest.mark.parametrize("has_xml", (True, False))
def test_writer_rejects_malformed_hierarchy_before_report_branch(
    tmp_path,
    has_xml,
):
    input_value = inputs()
    topology_value = topology()
    schedule = two_rank_allreduce_schedule()
    outcome = validate_and_lower_candidate(
        schedule,
        input_value,
        topology_value,
    )
    if not has_xml:
        outcome = replace(outcome, artifact=None)
    layout = create_run_layout(
        tmp_path,
        input_value,
        run_id="hierarchy-{}".format(has_xml),
    )

    with pytest.raises(SemanticError, match="hierarchy_plan"):
        write_candidate_artifact(
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
            hierarchy_plan=object(),
        )


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
            lambda value: value["schedule"]["transfers"][0]["atoms"][0].__setitem__(
                "pd", "invalid"
            ),
            "atom path is invalid",
        ),
        (
            lambda value: value["schedule"]["transfers"][0]["atoms"][0][
                "pd"
            ]["symbols"].__setitem__(0, "invalid"),
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



def test_schedule_sidecar_records_slice_path_time_vectors(tmp_path):
    input_value = inputs()
    schedule = two_rank_allreduce_schedule()
    topology_value = topology()
    outcome = validate_and_lower_candidate(schedule, input_value, topology_value)
    layout = create_run_layout(tmp_path, input_value, run_id="spt")
    artifact = write_candidate_artifact(layout, input_value, topology_value, _candidate(schedule), schedule, outcome,
                                        iteration=0, selected_best=True, accepted=True, rejection_reason=None)
    payload = json.loads(artifact.schedule_path.read_text(encoding="utf-8"))
    atom = payload["schedule"]["transfers"][0]["atoms"][0]
    assert {"s", "pd", "t"} <= set(atom)
    assert load_schedule_sidecar(artifact.schedule_path) == schedule
