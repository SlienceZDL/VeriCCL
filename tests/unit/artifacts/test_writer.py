import hashlib
import json
from dataclasses import replace

import pytest

from vericcl.artifacts.layout import create_run_layout
from vericcl.artifacts.writer import (
    load_schedule_sidecar,
    write_candidate_artifact,
    write_final_alias,
    write_resolved_input,
)
from vericcl.input.models import ObjectiveMode
from vericcl.solver.model import SolveCandidate, SolveStatus, SolverMetrics
from vericcl.verification.model import CheckResult, ValidationStatus
from vericcl.verification.pipeline import validate_and_lower_candidate

from tests.unit.verification.helpers import inputs, topology
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
