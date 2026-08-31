import hashlib
import json
import re
import shlex
from dataclasses import replace
from pathlib import Path

import pytest

import vericcl.workflow as workflow_module
from vericcl.workflow import RunContext, execute_solve, execute_verify
from vericcl.errors import SemanticError
from vericcl.verification.model import CheckResult, ValidationStatus
from vericcl.verification.online.pipeline import (
    OnlineCalibrationOutcome,
    OnlineStageStatus,
    OnlineValidationResult,
)
from vericcl.verification.online.calibration import (
    CalibrationPoint,
    CalibrationRequest,
    derive_calibrated_curve,
)
from vericcl.verification.online.statistics import summarize_runs


pytestmark = pytest.mark.phase07


PROJECT_ROOT = Path(__file__).parents[2]
EXAMPLES = PROJECT_ROOT / "vericcl" / "examples"
README = PROJECT_ROOT / "README.md"
DOCUMENTED_SOLVE_PATTERN = re.compile(
    r"<!-- vericcl-doc-test: solve -->\s*"
    r"```bash\s*\n([^\n]+)\n```"
)


def _documented_solve_run_id():
    match = DOCUMENTED_SOLVE_PATTERN.search(README.read_text(encoding="utf-8"))
    assert match is not None
    arguments = shlex.split(match.group(1))
    return arguments[arguments.index("--run-id") + 1]


def _write_constructive_inputs(tmp_path, hierarchy=False):
    topology = EXAMPLES / "topo" / "two_rank.json"
    atom_payload = json.loads(
        (EXAMPLES / "atom" / "default.json").read_text(encoding="utf-8")
    )
    atom_payload["strategies"]["milp"] = False
    atom_payload["strategies"]["hierarchy"] = hierarchy
    atom = tmp_path / "atom.json"
    atom.write_text(json.dumps(atom_payload), encoding="utf-8")
    sketch_payload = json.loads(
        (EXAMPLES / "sketch" / "allreduce_8m_1m.json").read_text(
            encoding="utf-8"
        )
    )
    sketch_payload["hyperparameters"]["total_size_bytes"] = 2 * 1024 * 1024
    sketch_payload["hyperparameters"]["input_chunkup"] = 2
    sketch_payload["hyperparameters"]["objective_mode"] = "latency"
    sketch_payload["solver"]["max_channels"] = 1
    sketch_payload["solver"]["max_parallel_models"] = 1
    sketch_payload["solver"]["max_threads_per_model"] = 1
    sketch = tmp_path / "sketch.json"
    sketch.write_text(json.dumps(sketch_payload), encoding="utf-8")
    return topology, sketch, atom


def test_documented_smoke_artifact_geometry_matches_workflow(tmp_path):
    run_id = _documented_solve_run_id()
    result = execute_solve(
        RunContext(
            topology_path=EXAMPLES / "topo" / "two_rank.json",
            sketch_path=EXAMPLES / "sketch" / "allreduce_8m_1m.json",
            atom_path=EXAMPLES / "atom" / "constructive.json",
            output_base=tmp_path / "documented-smoke",
            run_id=run_id,
        )
    )
    expected_root = (
        tmp_path
        / "documented-smoke"
        / "vericcl_allreduce_8MiB_{}".format(run_id)
    )

    assert result.layout.root == expected_root
    assert result.layout.resolved_input == expected_root / "resolved-input.json"
    assert result.layout.summary == expected_root / "run-summary.json"
    assert result.layout.schedules == expected_root / "schedules"
    assert result.layout.reports == expected_root / "reports"
    assert result.layout.traces == expected_root / "traces"
    assert result.final_xml == (
        expected_root / "vericcl_allreduce_8MiB_final.xml"
    )
    assert result.final_report == (
        expected_root / "vericcl_allreduce_8MiB_final.validation.json"
    )
    assert (
        expected_root / "vericcl_allreduce_8MiB_final.schedule.json"
    ).is_file()


def test_solve_workflow_writes_complete_lineage_and_final_alias(tmp_path):
    topology, sketch, atom = _write_constructive_inputs(tmp_path)
    result = execute_solve(
        RunContext(
            topology_path=topology,
            sketch_path=sketch,
            atom_path=atom,
            output_base=tmp_path / "runs",
            run_id="solve-0001",
            solver_version="test-solver",
            model_version="test-model",
            environment_signature="test-environment",
        )
    )

    assert result.final_candidate_id is not None
    assert result.final_xml is not None and result.final_xml.is_file()
    assert result.final_report is not None and result.final_report.is_file()
    assert result.layout.resolved_input.is_file()
    assert result.layout.summary.is_file()
    summary = json.loads(result.layout.summary.read_text(encoding="utf-8"))
    assert summary["final_selection"]["candidate_id"] == (
        result.final_candidate_id
    )
    assert summary["final_selection"]["xml_path"] == result.final_xml.name
    assert len(summary["candidates"]) == len(result.candidates)
    assert all("optimal" not in item["xml_path"] for item in summary["candidates"])
    for item in summary["candidates"]:
        if item["xml_path"] is None:
            continue
        xml_path = result.layout.root / item["xml_path"]
        report_path = result.layout.root / item["report_path"]
        assert hashlib.sha256(xml_path.read_bytes()).hexdigest() == item[
            "xml_sha256"
        ]
        assert hashlib.sha256(report_path.read_bytes()).hexdigest() == item[
            "report_sha256"
        ]


def test_solve_reports_direct_fallback_for_requested_hierarchy(tmp_path):
    topology, sketch, atom = _write_constructive_inputs(tmp_path, hierarchy=True)
    result = execute_solve(
        RunContext(
            topology_path=topology,
            sketch_path=sketch,
            atom_path=atom,
            output_base=tmp_path / "runs",
            run_id="hierarchy-fallback",
        )
    )

    report = json.loads(result.final_report.read_text(encoding="utf-8"))

    assert report["requested_strategies"]["hierarchy"] is True
    assert report["applied_strategies"]["hierarchy"] is False
    assert report["hierarchy_plan"]["planning_mode"] == "direct"
    assert (
        report["hierarchy_plan"]["planning_reason"]
        == "no_eligible_gateway_domain"
    )


def test_verify_workflow_reconstructs_schedule_and_uses_same_validation(tmp_path):
    topology, sketch, atom = _write_constructive_inputs(tmp_path)
    solved = execute_solve(
        RunContext(
            topology_path=topology,
            sketch_path=sketch,
            atom_path=atom,
            output_base=tmp_path / "runs",
            run_id="source",
        )
    )
    selected = next(
        item
        for item in solved.candidates
        if item.candidate_id == solved.final_candidate_id
    )

    verified = execute_verify(
        RunContext(
            topology_path=topology,
            sketch_path=sketch,
            atom_path=atom,
            output_base=tmp_path / "runs",
            run_id="verify",
            xml_path=selected.xml_path,
            sidecar_path=selected.schedule_path,
        )
    )

    assert verified.final_candidate_id == selected.candidate_id
    assert verified.final_xml is not None
    summary = json.loads(verified.layout.summary.read_text(encoding="utf-8"))
    assert summary["mode"] == "verify"
    assert summary["candidates"][0]["validation"]["semantic"] == "valid"
    assert summary["candidates"][0]["validation"]["xml"] == "valid"


def test_workflow_refuses_to_overwrite_a_previous_run(tmp_path):
    topology, sketch, atom = _write_constructive_inputs(tmp_path)
    context = RunContext(
        topology_path=topology,
        sketch_path=sketch,
        atom_path=atom,
        output_base=tmp_path / "runs",
        run_id="same",
    )
    execute_solve(context)

    with pytest.raises(FileExistsError, match="non-empty"):
        execute_solve(context)


def test_invalid_candidate_writes_report_and_sidecar_but_no_xml(
    tmp_path,
    monkeypatch,
):
    topology, sketch, atom = _write_constructive_inputs(tmp_path)
    original = workflow_module.validate_and_lower_candidate

    def force_invalid(schedule, inputs, topology_value):
        return original(
            schedule.__class__(
                schedule.schedule_id,
                schedule.transfers,
                schedule.final_state_ids,
                schedule.rank_count + 1,
                schedule.slice_count,
                schedule.slice_size_bytes,
                schedule.metadata,
            ),
            inputs,
            topology_value,
        )

    monkeypatch.setattr(
        workflow_module,
        "validate_and_lower_candidate",
        force_invalid,
    )
    result = execute_solve(
        RunContext(
            topology_path=topology,
            sketch_path=sketch,
            atom_path=atom,
            output_base=tmp_path / "runs",
            run_id="invalid",
        )
    )

    assert result.final_candidate_id is None
    assert result.final_xml is None
    assert result.candidates
    assert all(item.xml_path is None for item in result.candidates)
    assert all(item.report_path.is_file() for item in result.candidates)
    assert all(item.schedule_path.is_file() for item in result.candidates)
    summary = json.loads(result.layout.summary.read_text(encoding="utf-8"))
    assert summary["final_selection"] is None
    assert all(item["xml_path"] is None for item in summary["candidates"])


def test_workflow_wall_clock_budget_includes_input_resolution(
    tmp_path,
    monkeypatch,
):
    topology, sketch, atom = _write_constructive_inputs(tmp_path)
    timestamps = iter((0.0, 2.0))
    monkeypatch.setattr(workflow_module, "_monotonic", lambda: next(timestamps))

    with pytest.raises(TimeoutError, match="input resolution"):
        execute_solve(
            RunContext(
                topology_path=topology,
                sketch_path=sketch,
                atom_path=atom,
                output_base=tmp_path / "runs",
                run_id="timeout",
                timeout_s=1.0,
            )
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"run_id": ""}, "run_id"),
        ({"xml_path": Path("only.xml")}, "provided together"),
        ({"online": "yes"}, "online must be a boolean"),
        ({"timeout_s": 0}, "timeout_s"),
        ({"solver_version": ""}, "solver_version"),
    ),
)
def test_run_context_rejects_invalid_public_contract(changes, message):
    values = {
        "topology_path": Path("topology.json"),
        "sketch_path": Path("sketch.json"),
        "atom_path": Path("atom.json"),
        "output_base": Path("runs"),
        "run_id": "run",
    }
    values.update(changes)

    with pytest.raises(SemanticError, match=message):
        RunContext(**values)


def test_workflow_entrypoints_reject_invalid_modes_before_file_access():
    with pytest.raises(SemanticError, match="RunContext"):
        execute_solve(object())
    with pytest.raises(SemanticError, match="RunContext"):
        execute_verify(object())

    common = {
        "topology_path": Path("topology.json"),
        "sketch_path": Path("sketch.json"),
        "atom_path": Path("atom.json"),
        "output_base": Path("runs"),
        "run_id": "run",
    }
    with pytest.raises(SemanticError, match="does not accept"):
        execute_solve(
            RunContext(
                **common,
                xml_path=Path("schedule.xml"),
                sidecar_path=Path("schedule.schedule.json"),
            )
        )
    with pytest.raises(SemanticError, match="runtime configuration"):
        execute_solve(RunContext(**common, online=True))


@pytest.mark.parametrize(
    ("operator_status", "expected"),
    (
        (OnlineStageStatus.PASSED, "valid"),
        (OnlineStageStatus.FAILED, "failed"),
    ),
)
def test_online_result_is_written_without_discarding_offline_xml(
    tmp_path,
    monkeypatch,
    operator_status,
    expected,
):
    topology, sketch, atom = _write_constructive_inputs(tmp_path)

    def online_result(context):
        passed = operator_status is OnlineStageStatus.PASSED
        return OnlineValidationResult(
            context_schedule=context.schedule,
            preflight_status=OnlineStageStatus.PASSED,
            calibration_status=OnlineStageStatus.NOT_RUN,
            release_status=OnlineStageStatus.PASSED,
            online_operator_validation=operator_status,
            failure_code=None if passed else "forced_online_failure",
            failure_message=None if passed else "forced online failure",
            runtime_environment={},
            release_history=None,
            calibration=None,
            trace_analysis=None,
            trace_rank_files=(),
            trace_clock_uncertainty_us=1.0,
            requires_resolve=False,
            online_tuning_allowed=False,
            tuning_evidence=None,
        )

    monkeypatch.setattr(workflow_module, "run_online_validation", online_result)
    result = execute_solve(
        RunContext(
            topology_path=topology,
            sketch_path=sketch,
            atom_path=atom,
            output_base=tmp_path / "runs",
            run_id="online-{}".format(expected),
            online=True,
            online_context_factory=lambda artifact, schedule, *args: (
                type(
                    "FakeOnlineContext",
                    (),
                    {"artifact": artifact, "schedule": schedule},
                )()
            ),
        )
    )

    selected = next(
        item
        for item in result.candidates
        if item.candidate_id == result.final_candidate_id
    )
    assert result.final_xml is not None
    assert selected.validation["online"] == expected
    report = json.loads(selected.report_path.read_text(encoding="utf-8"))
    assert report["validation"]["online"]["status"] == expected
    assert any(result.layout.traces.glob("candidate-*/online-input.xml"))


def test_online_runtime_warning_writes_failure_without_launch_crash(
    tmp_path,
    monkeypatch,
):
    topology, sketch, atom = _write_constructive_inputs(tmp_path)
    original = workflow_module.validate_and_lower_candidate

    def runtime_warning(schedule, inputs, topology_value):
        outcome = original(schedule, inputs, topology_value)
        return replace(
            outcome,
            report=replace(
                outcome.report,
                runtime=CheckResult(
                    dimension="runtime",
                    status=ValidationStatus.WARNING,
                    code="msccl_runtime_incompatible",
                    message="runtime incompatible",
                    evidence={},
                ),
            ),
            artifact=replace(outcome.artifact, runtime_compatible=False),
        )

    monkeypatch.setattr(
        workflow_module,
        "validate_and_lower_candidate",
        runtime_warning,
    )
    result = execute_solve(
        RunContext(
            topology_path=topology,
            sketch_path=sketch,
            atom_path=atom,
            output_base=tmp_path / "runs",
            run_id="online-runtime-warning",
            online=True,
            online_context_factory=lambda *args: pytest.fail(
                "runtime launch must not run"
            ),
        )
    )

    selected = next(
        item
        for item in result.candidates
        if item.candidate_id == result.final_candidate_id
    )
    assert selected.validation["online"] == "failed"
    assert selected.runtime_compatible is False


def test_online_verify_handles_missing_runtime_result_without_crash(
    tmp_path,
    monkeypatch,
):
    topology, sketch, atom = _write_constructive_inputs(tmp_path)
    solved = execute_solve(
        RunContext(
            topology_path=topology,
            sketch_path=sketch,
            atom_path=atom,
            output_base=tmp_path / "runs",
            run_id="verify-none-source",
        )
    )
    selected = next(
        item
        for item in solved.candidates
        if item.candidate_id == solved.final_candidate_id
    )

    def unavailable(**kwargs):
        return (
            workflow_module._with_online_failure(
                kwargs["outcome"],
                "online_runtime_incompatible",
                "runtime incompatible",
            ),
            None,
        )

    monkeypatch.setattr(workflow_module, "_run_online_candidate", unavailable)
    verified = execute_verify(
        RunContext(
            topology_path=topology,
            sketch_path=sketch,
            atom_path=atom,
            output_base=tmp_path / "runs",
            run_id="verify-none",
            xml_path=selected.xml_path,
            sidecar_path=selected.schedule_path,
            online=True,
            online_context_factory=lambda *args: None,
        )
    )

    assert verified.candidates[0].validation["online"] == "failed"


def test_stable_online_calibration_updates_topology_and_resolves_again(
    tmp_path,
    monkeypatch,
):
    topology, sketch, atom = _write_constructive_inputs(tmp_path)
    request = CalibrationRequest(
        "intra_node",
        1024 * 1024,
        1,
        "float",
    )
    point = CalibrationPoint(
        1,
        summarize_runs((10.0,) * 20),
        full_wave_count=128,
        tail_transfer_count=0,
    )
    calibration = OnlineCalibrationOutcome(
        request=request,
        points=(point,),
        curve=derive_calibrated_curve(1.0, 1024 * 1024, (point,)),
        cache_hit_concurrencies=(),
        stable=True,
    )
    online_calls = []

    def online_result(context):
        online_calls.append(context)
        if len(online_calls) == 1:
            return OnlineValidationResult(
                context_schedule=context.schedule,
                preflight_status=OnlineStageStatus.PASSED,
                calibration_status=OnlineStageStatus.REQUIRES_RESOLVE,
                release_status=OnlineStageStatus.NOT_RUN,
                online_operator_validation=OnlineStageStatus.NOT_RUN,
                failure_code=None,
                failure_message=None,
                runtime_environment={},
                release_history=None,
                calibration=calibration,
                trace_analysis=None,
                trace_rank_files=(),
                trace_clock_uncertainty_us=None,
                requires_resolve=True,
                online_tuning_allowed=False,
                tuning_evidence=None,
            )
        return OnlineValidationResult(
            context_schedule=context.schedule,
            preflight_status=OnlineStageStatus.PASSED,
            calibration_status=OnlineStageStatus.NOT_RUN,
            release_status=OnlineStageStatus.PASSED,
            online_operator_validation=OnlineStageStatus.PASSED,
            failure_code=None,
            failure_message=None,
            runtime_environment={},
            release_history=None,
            calibration=None,
            trace_analysis=None,
            trace_rank_files=(),
            trace_clock_uncertainty_us=1.0,
            requires_resolve=False,
            online_tuning_allowed=False,
            tuning_evidence=None,
        )

    solve_calls = []
    original_solve = workflow_module.solve

    def counted_solve(request_value):
        solve_calls.append(request_value)
        return original_solve(request_value)

    monkeypatch.setattr(workflow_module, "run_online_validation", online_result)
    monkeypatch.setattr(workflow_module, "solve", counted_solve)
    result = execute_solve(
        RunContext(
            topology_path=topology,
            sketch_path=sketch,
            atom_path=atom,
            output_base=tmp_path / "runs",
            run_id="online-calibration",
            online=True,
            online_context_factory=lambda artifact, schedule, *args: type(
                "FakeOnlineContext",
                (),
                {"artifact": artifact, "schedule": schedule},
            )(),
        )
    )

    assert len(solve_calls) == 2
    calibrated_topology = solve_calls[1].topology
    calibrated_link = calibrated_topology.links[
        next(iter(calibrated_topology.links))
    ]
    assert calibrated_link.performance.is_calibrated
    assert all(call.wall_clock_budget_s is not None for call in solve_calls)
    assert solve_calls[1].wall_clock_budget_s <= (
        solve_calls[0].wall_clock_budget_s
    )
    assert len(online_calls) == 2
    assert len(result.candidates) == 2
    strategies = {
        json.loads(item.report_path.read_text(encoding="utf-8"))[
            "tuning_strategy"
        ]["kind"]
        for item in result.candidates
    }
    assert strategies == {"initial_solve", "calibrated_resolve"}
    initial_ids = {
        item.candidate_id
        for item in result.candidates
        if json.loads(item.report_path.read_text(encoding="utf-8"))[
            "tuning_strategy"
        ]["kind"] == "initial_solve"
    }
    selected_artifact = next(
        item
        for item in result.candidates
        if item.candidate_id == result.final_candidate_id
    )
    assert selected_artifact.parent_candidate_id in initial_ids
    selected = next(
        item
        for item in result.candidates
        if item.candidate_id == result.final_candidate_id
    )
    report = json.loads(selected.report_path.read_text(encoding="utf-8"))
    evidence = report["validation"]["online"]["evidence"]
    assert evidence["calibration_status"] == "passed"
    assert evidence["requires_resolve"] is False
    assert evidence["calibration"]["link_class"] == "intra_node"
    assert evidence["calibration"]["points"][0]["p95_us"] == 10.0
    assert "mean_us" in evidence["calibration"]["points"][0]
    assert "population_standard_deviation_us" in evidence[
        "calibration"
    ]["points"][0]


def test_online_verify_runs_operator_after_calibration_and_requires_resolve(
    tmp_path,
    monkeypatch,
):
    topology, sketch, atom = _write_constructive_inputs(tmp_path)
    solved = execute_solve(
        RunContext(
            topology_path=topology,
            sketch_path=sketch,
            atom_path=atom,
            output_base=tmp_path / "runs",
            run_id="verify-calibration-source",
        )
    )
    selected = next(
        item
        for item in solved.candidates
        if item.candidate_id == solved.final_candidate_id
    )
    request = CalibrationRequest("intra_node", 1024 * 1024, 1, "float")
    point = CalibrationPoint(
        1,
        summarize_runs((10.0,) * 20),
        full_wave_count=128,
        tail_transfer_count=0,
    )
    calibration = OnlineCalibrationOutcome(
        request=request,
        points=(point,),
        curve=derive_calibrated_curve(1.0, 1024 * 1024, (point,)),
        cache_hit_concurrencies=(1,),
        stable=True,
    )
    calls = []

    def online_result(context):
        calls.append(context)
        return OnlineValidationResult(
            context_schedule=context.schedule,
            preflight_status=OnlineStageStatus.PASSED,
            calibration_status=(
                OnlineStageStatus.REQUIRES_RESOLVE
                if len(calls) == 1
                else OnlineStageStatus.NOT_RUN
            ),
            release_status=(
                OnlineStageStatus.NOT_RUN
                if len(calls) == 1
                else OnlineStageStatus.PASSED
            ),
            online_operator_validation=(
                OnlineStageStatus.NOT_RUN
                if len(calls) == 1
                else OnlineStageStatus.PASSED
            ),
            failure_code=None,
            failure_message=None,
            runtime_environment={},
            release_history=None,
            calibration=calibration if len(calls) == 1 else None,
            trace_analysis=None,
            trace_rank_files=(),
            trace_clock_uncertainty_us=(
                None if len(calls) == 1 else 1.0
            ),
            requires_resolve=len(calls) == 1,
            online_tuning_allowed=False,
            tuning_evidence=None,
        )

    monkeypatch.setattr(workflow_module, "run_online_validation", online_result)
    verified = execute_verify(
        RunContext(
            topology_path=topology,
            sketch_path=sketch,
            atom_path=atom,
            output_base=tmp_path / "runs",
            run_id="verify-calibration",
            xml_path=selected.xml_path,
            sidecar_path=selected.schedule_path,
            online=True,
            online_context_factory=lambda artifact, schedule, *args: type(
                "FakeOnlineContext",
                (),
                {"artifact": artifact, "schedule": schedule},
            )(),
        )
    )

    assert len(calls) == 2
    report = json.loads(verified.final_report.read_text(encoding="utf-8"))
    online = report["validation"]["online"]
    assert online["status"] == "valid"
    assert online["evidence"]["requires_resolve"] is True
    assert online["evidence"]["calibration"][
        "cache_hit_concurrencies"
    ] == [1]
