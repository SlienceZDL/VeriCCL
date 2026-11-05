import json
from pathlib import Path
import subprocess

import pytest

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


def test_micro_cases_have_complete_semantics():
    from vericcl.experiments.overhead_jobs import micro_case
    from vericcl.verification.constraints import verify_schedule_constraints
    from vericcl.verification.semantics import verify_schedule_semantics
    from vericcl.verification.model import ValidationStatus

    for name in ("broadcast", "allreduce"):
        schedule, topo, data, hint = micro_case(name)
        assert hint.waiting_transfer_id in {t.transfer_id for t in schedule.transfers}
        assert verify_schedule_semantics(schedule, data).status is ValidationStatus.VALID
        assert verify_schedule_constraints(schedule, data, topo).status is ValidationStatus.VALID


def test_online_repeats_use_unique_directories_and_retain_failures(tmp_path, monkeypatch):
    from argparse import Namespace
    import vericcl.experiments.overhead_jobs as jobs
    from vericcl.experiments.overhead import write_json

    calls = []

    def run_once(args):
        calls.append(args.output)
        write_json(Path(args.output) / "overhead.json", {"status": "failed"})
        return 1

    from pathlib import Path
    monkeypatch.setattr(jobs, "run_online_once", run_once)
    monkeypatch.setattr(jobs, "online_output_root", lambda args: Path(args.output))
    assert jobs.run_online(Namespace(repeats=3, output=str(tmp_path))) == 1
    payload = json.loads((tmp_path / "repeats.json").read_text())
    assert payload["complete"] is True
    assert len(set(calls)) == len(payload["records"]) == 3


def test_online_gate_rejects_bdd_error_even_with_compatible_xml():
    from types import SimpleNamespace
    from vericcl.experiments.overhead_jobs import online_eligible
    from vericcl.verification.model import ValidationStatus as Status

    report = SimpleNamespace(overall_status=Status.VALID,
        bdd=SimpleNamespace(status=Status.ANALYSIS_ERROR),
        simulation=SimpleNamespace(status=Status.VALID))
    outcome = SimpleNamespace(report=report,
                              artifact=SimpleNamespace(runtime_compatible=True))
    assert not online_eligible(outcome)
    report.bdd.status = Status.VALID
    assert online_eligible(outcome)


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


def test_input_identity_includes_normalized_and_raw_hashes(tmp_path):
    from argparse import Namespace
    from types import SimpleNamespace
    from vericcl.experiments.overhead import input_identity

    path = tmp_path / "topology.json"
    path.write_text("{}")
    result = input_identity(Namespace(topology=str(path), sketch=None, atoms=None),
                            SimpleNamespace(input_sha256="normalized"))
    assert result["normalized_input_sha256"] == "normalized"
    assert len(result["raw_sha256"]["topology"]) == 64


def test_repeated_online_validates_path_before_any_write(tmp_path, monkeypatch):
    from argparse import Namespace
    import vericcl.experiments.overhead_jobs as jobs

    def reject(args):
        raise ValueError("outside experiment root")

    monkeypatch.setattr(jobs, "online_output_root", reject, raising=False)
    output = tmp_path / "not-created"
    with pytest.raises(ValueError, match="outside experiment root"):
        jobs.run_online(Namespace(repeats=2, output=str(output)))
    assert not output.exists()


@pytest.mark.parametrize("argv", [
    ["clock", "--config", "config.json", "--repeats", "2"],
    ["calibration", "--config", "config.json"],
])
def test_incomplete_or_ignored_options_are_rejected(tmp_path, argv):
    from vericcl.experiments.overhead_jobs import main

    with pytest.raises(SystemExit) as error:
        main(argv + ["--output", str(tmp_path / "unused")])
    assert error.value.code == 2


def test_calibration_case_uses_complete_existing_benchmark():
    from argparse import Namespace
    from vericcl.experiments.overhead_jobs import load_calibration_case
    from vericcl.verification.pipeline import validate_and_lower_candidate

    root = Path(__file__).resolve().parents[3]
    args = Namespace(topology=str(root / "tests/fixtures/legacy-experiments/topo/v100-n2g4.json"),
        sketch=str(root / "tests/fixtures/legacy-experiments/sketch/v100-n2g4/ag/ag-4m.json"),
        atoms=str(root / "tests/fixtures/legacy-catalog/atom/default.json"), channels=1)
    schedule, topo, data = load_calibration_case(args)
    assert data.rank_count == 2
    assert data.hyperparameters.total_size_bytes == 128 * 1024 * 1024
    outcome = validate_and_lower_candidate(schedule, data, topo)
    from vericcl.experiments.overhead_jobs import online_eligible
    assert online_eligible(outcome)
