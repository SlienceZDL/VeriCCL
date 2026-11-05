from dataclasses import replace

import pytest

from vericcl.errors import SemanticError
from vericcl.experiments.state import (
    ExperimentStateStore,
    TaskRecord,
    TaskStatus,
    atomic_replace_text,
)


def _record(**changes):
    values = {
        "task_id": "v100-n2g4-ag-4m",
        "status": TaskStatus.PASSED,
        "input_sha256": "a" * 64,
        "output_sha256": "b" * 64,
        "command": ("vericcl", "solve"),
        "returncode": 0,
        "finished_at_utc": "2026-09-02T00:00:00Z",
    }
    values.update(changes)
    return TaskRecord(**values)


def test_state_store_resumes_only_matching_completed_task(tmp_path):
    store = ExperimentStateStore(tmp_path / "state.json")
    record = _record()

    store.put(record)

    assert store.reusable(record.task_id, "a" * 64, "b" * 64) is True
    assert store.reusable(record.task_id, "c" * 64, "b" * 64) is False
    assert store.reusable(record.task_id, "a" * 64, "c" * 64) is False
    assert store.load()[record.task_id] == record


def test_running_record_is_not_reusable_and_can_complete(tmp_path):
    store = ExperimentStateStore(tmp_path / "state.json")
    running = _record(
        status=TaskStatus.RUNNING,
        output_sha256=None,
        returncode=None,
        finished_at_utc=None,
        started_at_utc="2026-09-02T00:00:00Z",
    )

    store.put(running)
    assert store.reusable(running.task_id, "a" * 64, "b" * 64) is False
    store.put(_record(started_at_utc=running.started_at_utc))

    assert store.load()[running.task_id].status is TaskStatus.PASSED


def test_state_store_rejects_illegal_status_transition(tmp_path):
    store = ExperimentStateStore(tmp_path / "state.json")
    store.put(_record())

    with pytest.raises(SemanticError, match="transition"):
        store.put(
            _record(
                status=TaskStatus.FAILED,
                output_sha256=None,
                returncode=1,
                failure_code="runtime_failed",
            )
        )


def test_atomic_replace_text_replaces_complete_document(tmp_path):
    path = tmp_path / "nested" / "state.json"

    atomic_replace_text(path, "first\n")
    atomic_replace_text(path, "second\n")

    assert path.read_text(encoding="utf-8") == "second\n"
    assert tuple(path.parent.glob(".*.tmp")) == ()


def test_task_record_rejects_invalid_boundaries():
    valid = _record()

    for changes in (
        {"task_id": ""},
        {"input_sha256": "bad"},
        {"output_sha256": "bad"},
        {"command": ()},
        {"returncode": -1},
        {"status": "passed"},
        {"status": TaskStatus.PASSED, "returncode": 1},
        {"status": TaskStatus.PASSED, "output_sha256": None},
    ):
        with pytest.raises(SemanticError):
            replace(valid, **changes)


def test_state_store_rejects_corrupted_document(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("not-json", encoding="ascii")

    with pytest.raises(SemanticError, match="state"):
        ExperimentStateStore(path).load()
