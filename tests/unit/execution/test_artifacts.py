import json
from pathlib import Path
import sys

import pytest


def make_context(tmp_path, **kwargs):
    from vericcl.execution.context import RunContext
    return RunContext.create(tmp_path, "demo", "run-1", **kwargs)


def test_run_layout_refuses_to_reuse_a_nonempty_run(tmp_path):
    from vericcl.execution.context import RunContext
    context = make_context(tmp_path)
    assert context.root == tmp_path / "examples/demo/output/run-1"
    expected = {"input_snapshot", "models", "candidates", "jobs", "live",
                "statistics", "checkpoints", "reports", "best", "tmp", "rebuildable-cache"}
    assert {p.name for p in context.root.iterdir() if p.is_dir()} == {'process'}
    assert expected <= {p.name for p in (context.root / 'process').iterdir()}
    with pytest.raises(FileExistsError):
        RunContext.create(tmp_path, "demo", "run-1")


@pytest.mark.parametrize("case,run", [("../outside", "r"), ("case", "../r"),
                                    ("a/b", "r"), ("c", "/absolute")])
def test_identifiers_cannot_escape_the_case_directory(tmp_path, case, run):
    from vericcl.execution.context import RunContext
    with pytest.raises(ValueError, match="identifier"):
        RunContext.create(tmp_path, case, run)
    assert not (tmp_path / "outside").exists()


def test_case_and_artifact_symlinks_cannot_escape_the_project(tmp_path):
    from vericcl.execution.context import RunContext
    outside = tmp_path / "outside"
    outside.mkdir()
    project = tmp_path / "project"
    (project / "examples").mkdir(parents=True)
    (project / "examples/bad").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="boundary"):
        RunContext.create(project, "bad", "r")
    context = RunContext.create(project, "good", "r")
    (context.artifact("models/escape")).symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="boundary"):
        context.write_json("models/escape/model.json", {"alpha_us": 1})
    assert not list(outside.iterdir())


def test_snapshot_is_immutable_and_not_a_link_to_the_original(tmp_path):
    source = tmp_path / "request.json"
    original = b'{"size_bytes": 4096}\n'
    source.write_bytes(original)
    context = make_context(tmp_path, input_files={"request.json": source})
    source.write_text("changed")
    assert (context.artifact("input_snapshot/request.json")).read_bytes() == original
    with pytest.raises(FileExistsError):
        context.write_json("input_snapshot/request.json", {"size_bytes": 1})
    with pytest.raises(ValueError, match="immutable"):
        context.write_json("input_snapshot/request.json", {}, replace=True)


def test_a_user_input_named_manifest_is_preserved(tmp_path):
    source = tmp_path / "manifest.json"
    source.write_text('{"topologies": ["t1"]}')
    context = make_context(tmp_path, input_files={"manifest.json": source})
    assert (context.artifact("input_snapshot/manifest.json")).read_bytes() == source.read_bytes()


def test_secrets_and_unknown_environment_are_not_persisted(tmp_path):
    from vericcl.execution.events import emit_event, read_events
    token = "secret-value-that-must-stay-private"
    context = make_context(tmp_path, environment={"PATH": "/bin", "CUDA_VISIBLE_DEVICES": "0",
                                                "API_TOKEN": token, "OTHER_SETTING": "hidden"})
    emit_event(context, "diagnostic", {"message": "token=" + token,
               "password": "another-private-value", "environment": {"OTHER_SETTING": "hidden"}})
    text = "\n".join(p.read_text() for p in context.root.rglob("*") if p.is_file())
    assert token not in text
    assert "another-private-value" not in text
    assert "hidden" not in text
    assert '"CUDA_VISIBLE_DEVICES": "0"' in text
    assert read_events(context.artifact("events.jsonl")).events[-1]["kind"] == "diagnostic"


def test_secret_in_a_snapshot_is_rejected_before_copying(tmp_path):
    source = tmp_path / "sensitive.json"
    source.write_text('{"password": "do-not-copy"}')
    with pytest.raises(ValueError, match="credential"):
        make_context(tmp_path, input_files={"request.json": source})
    assert not (tmp_path / "examples/demo/output/run-1").exists()


def test_truncated_event_tail_does_not_invent_a_completed_run(tmp_path):
    from vericcl.execution.events import emit_event, read_events
    context = make_context(tmp_path)
    emit_event(context, "candidate_started", {"candidate_id": "c1"}, rank=2)
    path = context.artifact("rank-2.events.jsonl")
    with path.open("ab") as stream:
        stream.write(b'{"kind":"finished"')
    scanned = read_events(path)
    assert scanned.truncated_tail
    assert [event["kind"] for event in scanned.events] == ["candidate_started"]
    assert not (context.artifact("run-summary.json")).exists()


def test_failed_candidate_evidence_survives_finalization_and_cache_cleanup(tmp_path):
    from vericcl.execution.context import finalize_run
    from vericcl.execution.status import RunResult
    context = make_context(tmp_path)
    context.write_json("candidates/rejected/checks.json", {"semantic": "invalid"})
    context.write_json("tmp/transient.json", {"temporary": True})
    context.cleanup_rebuildable()
    result = finalize_run(context, RunResult(contract_met=True, execution="completed",
                          checks={"best.semantic": "valid"}, model_coverage="partial",
                          acceptance="accepted", stop_reason="max_rounds",
                          proof_scope=None, optional_failures=("shared_probe",)))
    assert result.exit_code == 0 and result.complete
    summary = json.loads((context.artifact("run-summary.json")).read_text())
    assert summary["model_coverage"] == "partial"
    assert summary["proof_scope"] is None
    assert summary["optional_failures"] == ["shared_probe"]
    assert (context.artifact("candidates/rejected/checks.json")).is_file()
    assert not (context.artifact("tmp/transient.json")).exists()
    assert "max_rounds" in (context.artifact("reports/summary.md")).read_text()


def test_required_storage_failure_blocks_new_unrecorded_processes(tmp_path, monkeypatch):
    from vericcl.execution.context import StorageFailure
    from vericcl.execution.processes import ProcessLease, run_owned_process
    context = make_context(tmp_path)
    def disk_full(*args, **kwargs):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr("vericcl.execution.context.atomic_write_bytes", disk_full)
    with pytest.raises(StorageFailure):
        context.write_json("models/new.json", {})
    marker = tmp_path / "must-not-run"
    with pytest.raises(StorageFailure):
        run_owned_process(context, [sys.executable, "-c", f"open({str(marker)!r},'w').close()"],
                          ProcessLease.after(2, cleanup_reserve_s=0.2))
    assert not marker.exists()


@pytest.mark.parametrize("field,code", [("input", 2), ("offline", 3), ("online", 4),
                                      ("internal", 5), ("invalid", 6), ("storage", 7),
                                      ("backend", 8), ("cancelled", 130)])
def test_required_failures_have_distinct_public_exit_codes(field, code):
    from vericcl.execution.status import RunResult
    result = RunResult(contract_met=False, execution="failed", failure_category=field)
    assert result.exit_code == code


def test_native_events_use_the_same_rank_and_task_identity(tmp_path):
    from vericcl.execution.events import emit_native_event, read_events
    context = make_context(tmp_path)
    emit_native_event(context, b'{"schema_version":1,"kind":"gpu_completed","data":{"request_id":"r1"}}', rank=0)
    event = read_events(context.artifact("rank-0.events.jsonl")).events[0]
    assert event["task_id"] == context.task_id and event["rank"] == 0
    with pytest.raises(ValueError):
        emit_native_event(context, b'{"schema_version":1,"kind":"x","task_id":"forged"}', rank=0)


def test_atomic_evidence_creation_never_replaces_an_existing_symlink(tmp_path):
    from vericcl.artifacts.writer import atomic_write_bytes
    outside = tmp_path / "not-yet-created"
    destination = tmp_path / "evidence.json"
    destination.symlink_to(outside)
    with pytest.raises(FileExistsError):
        atomic_write_bytes(destination, b"replacement")
    assert destination.is_symlink() and not outside.exists()


def test_finalization_reports_disk_failure_without_a_success_exit(tmp_path, monkeypatch):
    from vericcl.execution.context import finalize_run
    from vericcl.execution.status import RunResult
    context = make_context(tmp_path)
    def disk_full(*args, **kwargs):
        raise OSError(28, "full")
    monkeypatch.setattr("vericcl.execution.context.atomic_write_bytes", disk_full)
    result = finalize_run(context, RunResult(True, "completed"))
    assert result.exit_code == 7 and not result.complete
    assert not (context.artifact("run-summary.json")).exists()
