import json
from argparse import Namespace
from pathlib import Path
import subprocess

import pytest

from tests.unit.input.v5_support import request_documents, write_request
from vericcl.verification.online.runner import ProcessRequest, ProcessResult


def test_logging_executor_retains_both_output_streams(tmp_path):
    from vericcl.experiments.overhead_jobs import LoggingExecutor
    from vericcl.experiments.overhead import PhaseTimer

    class Delegate:
        def run(self, request):
            return ProcessResult(7, "table\n", "diagnostic\n")

    runner = LoggingExecutor(Delegate(), tmp_path, PhaseTimer())
    request = ProcessRequest(("test",), {"NCCL_PROTO": "Simple", "UNRELATED_SECRET": "private"}, "release")
    assert runner.run(request).returncode == 7
    assert (tmp_path / "000.stdout.log").read_text() == "table\n"
    assert (tmp_path / "000.stderr.log").read_text() == "diagnostic\n"
    assert (tmp_path / "000.request.json").is_file()
    recorded = json.loads((tmp_path / "000.request.json").read_text())
    assert recorded["environment"] == {"NCCL_PROTO": "Simple"}


def test_logging_executor_preserves_timeout_output_and_reraises(tmp_path):
    from vericcl.experiments.overhead_jobs import LoggingExecutor
    from vericcl.experiments.overhead import PhaseTimer

    class Delegate:
        def run(self, request):
            try:
                raise subprocess.TimeoutExpired("test", 1, b"partial table", b"partial error")
            except subprocess.TimeoutExpired as error:
                raise RuntimeError("failed") from error

    runner = LoggingExecutor(Delegate(), tmp_path, PhaseTimer())
    with pytest.raises(RuntimeError, match="failed"):
        runner.run(ProcessRequest(("test",), {}, "timeout"))
    assert (tmp_path / "000.stdout.log").read_text() == "partial table"
    assert (tmp_path / "000.stderr.log").read_text() == "partial error"
    assert json.loads((tmp_path / "000.result.json").read_text())["error"] == "RuntimeError"


def test_micro_cases_start_from_the_state_each_repair_type_handles(tmp_path):
    from vericcl.experiments.overhead_jobs import REPAIR_METHODS, micro_case

    for method in REPAIR_METHODS:
        directory = tmp_path / method
        directory.mkdir()
        case = micro_case(method, directory)
        assert case.bundle.can_execute is (method != "correctness"), method
        assert case.lowering.inputs.rank_count == 3 and callable(case.repair)
        if method in {"reallocation", "bottleneck"}:
            assert case.strategy is not None and case.setup["hotspot_count"] >= 1


def test_correctness_repair_micro_benchmark_records_repeated_samples(tmp_path, capsys):
    from vericcl.experiments.overhead_jobs import main
    from vericcl.experiments.overhead_summary import build_summary

    output = tmp_path / "raw" / "micro" / "repair-correctness.json"
    assert main(["repair", "--method", "correctness", "--repeats", "2", "--output", str(output)]) == 0
    payload = json.loads(output.read_text())
    assert payload["complete"] and payload["pipeline"] == "v5"
    assert payload["incumbent"]["can_execute"] is False
    assert len(payload["records"]) == len(payload["repeated"]["samples_s"]) == 2
    for record in payload["records"]:
        assert record["status"] == "repaired" and record["candidate"]["can_execute"]
        assert record["evidence"]["steps"][0]["action"] == "add_delivery"
        assert {"repair_correctness", "lowering", "symbolic_check"} <= set(record["timing"]["exclusive_s"])
    summary = build_summary(tmp_path / "raw")["records"]["micro/repair-correctness.json"]
    assert [item["status"] for item in summary["records"]] == ["repaired", "repaired"]


@pytest.mark.scip
@pytest.mark.parametrize("method,label", [("reallocation", "repair_reallocation"),
                                          ("bottleneck", "repair_bottleneck"),
                                          ("strategy", "repair_strategy")])
def test_solver_repair_micro_benchmarks_improve_the_incumbent(tmp_path, capsys, method, label):
    pytest.importorskip("pyscipopt")
    from vericcl.experiments.overhead_jobs import main

    output = tmp_path / ("repair-%s.json" % method)
    assert main(["repair", "--method", method, "--repeats", "1", "--output", str(output)]) == 0
    payload = json.loads(output.read_text())
    record = payload["records"][0]
    assert record["status"] == "improved" and record["model_calls"] >= 1
    assert record["completion_time_us"] < payload["parent_completion_time_us"]
    assert {label, "model_optimize"} <= set(record["timing"]["exclusive_s"])


def test_online_repeats_use_unique_directories_and_retain_failures(tmp_path, monkeypatch):
    import vericcl.experiments.overhead_jobs as jobs
    from vericcl.experiments.overhead import write_json

    calls = []

    def run_once(args):
        calls.append(args.output)
        write_json(Path(args.output) / "overhead.json", {"status": "failed"})
        return 1

    monkeypatch.setattr(jobs, "run_online_once", run_once)
    monkeypatch.setattr(jobs, "online_output_root", lambda args: Path(args.output))
    assert jobs.run_online(Namespace(repeats=3, output=str(tmp_path))) == 1
    payload = json.loads((tmp_path / "repeats.json").read_text())
    assert payload["complete"] is True
    assert len(set(calls)) == len(payload["records"]) == 3


def test_repeated_online_validates_path_before_any_write(tmp_path, monkeypatch):
    import vericcl.experiments.overhead_jobs as jobs

    def reject(args):
        raise ValueError("outside experiment root")

    monkeypatch.setattr(jobs, "online_output_root", reject, raising=False)
    output = tmp_path / "not-created"
    with pytest.raises(ValueError, match="outside experiment root"):
        jobs.run_online(Namespace(repeats=2, output=str(output)))
    assert not output.exists()


def test_input_identity_includes_normalized_and_raw_hashes(tmp_path):
    from types import SimpleNamespace
    from vericcl.experiments.overhead import input_identity

    path = tmp_path / "topology.json"
    path.write_text("{}")
    result = input_identity(Namespace(topology=str(path), sketch=None, atom=None, runtime=None),
                            SimpleNamespace(input_sha256="normalized"))
    assert result["normalized_input_sha256"] == "normalized"
    assert len(result["raw_sha256"]["topology"]) == 64


@pytest.mark.parametrize("argv", [
    ["clock", "--config", "config.json", "--repeats", "2"],
    ["clock"],
    ["calibration", "--config", "config.json"],
    ["online", "--effective-request", "run.json"],
    ["calibration", "--effective-request", "run.json", "--xml", "schedule.xml"],
    ["repair"],
    ["repair", "--method", "correctness", "--effective-request", "run.json"],
])
def test_incomplete_or_ignored_options_are_rejected(tmp_path, argv):
    from vericcl.experiments.overhead_jobs import main

    with pytest.raises(SystemExit) as error:
        main(argv + ["--output", str(tmp_path / "unused")])
    assert error.value.code == 2


def _calibration_documents():
    documents = request_documents("broadcast", 2)
    documents["sketch"]["solver"].update(max_channels=1, per_model_timeout_s=5)
    documents["sketch"]["calibration"].update(enabled=True, max_relative_fit_error=.05, job_timeout_s=30)
    documents["sketch"]["budgets"].update(max_online_validation_per_round_s=60, max_online_validation_total_s=120)
    documents["topology"]["local_costs"] = [dict(rank=rank, copy=dict(alpha_us=.1, beta_us_per_byte=.001))
                                            for rank in range(2)]
    return documents


def test_calibration_run_is_instrumented_v5_calibrate_purpose(tmp_path, capsys):
    """A mathematical probe driver stands in for GPUs; no process is launched."""
    from tests.unit.online.test_component_measurements import mathematical_native_contract
    from vericcl.experiments.overhead_jobs import run_online_once
    from vericcl.workflow_v5 import WorkflowServices

    class FixtureProbeDriver:
        driver_kind = "native_msccl"

        def __init__(self):
            self.jobs = []

        def execute(self, job, context):
            self.jobs.append(job.job_id)
            return mathematical_native_contract(job)

    (tmp_path / "input").mkdir()
    paths = write_request(tmp_path / "input", _calibration_documents())
    driver = FixtureProbeDriver()
    args = Namespace(mode="calibration", output=str(tmp_path / "calibration"), config=None, repeats=1,
                     set=[], effective_request=None, xml=None, sidecar=None, model=None,
                     **{name: str(path) for name, path in paths.items()})
    assert run_online_once(args, services=WorkflowServices(probe_driver=driver)) == 0
    record = json.loads((tmp_path / "calibration" / "overhead.json").read_text())
    assert driver.jobs and record["purpose"] == "calibrate" and record["exit_code"] == 0
    assert record["result"]["stop_reason"] == "calibration_finished"
    assert record["phases"]["online_measurement_s"] > 0
    exclusive = record["timing"]["exclusive_s"]
    assert {"calibration_plan", "probe_job", "calibration_fit", "model_publish"} <= set(exclusive)
    assert record["timing"]["calls"]["probe_job"] == len(driver.jobs)


def test_online_mode_enables_trace_measurement_by_default():
    from vericcl.experiments.overhead_jobs import online_overrides

    assert online_overrides(Namespace(mode="online", set=[])) == {"sketch.trace_and_measure.enabled": True}
    assert online_overrides(Namespace(mode="online", set=["sketch.trace_and_measure.enabled=false"])) == {
        "sketch.trace_and_measure.enabled": False}
    assert online_overrides(Namespace(mode="calibration", set=[])) == {}
