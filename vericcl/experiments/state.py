from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import os
from pathlib import Path
import uuid
from typing import Mapping, Optional, Tuple

from vericcl.errors import SemanticError


_HEX = frozenset("0123456789abcdef")
_STATE_FIELDS = frozenset({"schema_version", "tasks"})
_RECORD_FIELDS = frozenset(
    {
        "task_id",
        "status",
        "input_sha256",
        "output_sha256",
        "command",
        "returncode",
        "failure_code",
        "log_path",
        "started_at_utc",
        "finished_at_utc",
    }
)


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise SemanticError("{} must be a non-empty string".format(field))
    return value


def _optional_string(value: object, field: str) -> Optional[str]:
    if value is None:
        return None
    return _identifier(value, field)


def _sha256(value: object, field: str, *, optional: bool) -> Optional[str]:
    if value is None and optional:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise SemanticError("{} must be a SHA-256 digest".format(field))
    return value


def _returncode(value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SemanticError("task returncode must be non-negative")
    return value


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    status: TaskStatus
    input_sha256: str
    output_sha256: Optional[str]
    command: Tuple[str, ...]
    returncode: Optional[int]
    failure_code: Optional[str] = None
    log_path: Optional[str] = None
    started_at_utc: Optional[str] = None
    finished_at_utc: Optional[str] = None

    def __post_init__(self) -> None:
        _identifier(self.task_id, "task_id")
        if not isinstance(self.status, TaskStatus):
            raise SemanticError("task status is invalid")
        _sha256(self.input_sha256, "task input_sha256", optional=False)
        _sha256(self.output_sha256, "task output_sha256", optional=True)
        try:
            command = tuple(self.command)
        except TypeError as error:
            raise SemanticError("task command must be iterable") from error
        if not command or not all(
            isinstance(value, str) and value and "\x00" not in value
            for value in command
        ):
            raise SemanticError("task command is invalid")
        object.__setattr__(self, "command", command)
        _returncode(self.returncode)
        _optional_string(self.failure_code, "task failure_code")
        _optional_string(self.log_path, "task log_path")
        _optional_string(self.started_at_utc, "task started_at_utc")
        _optional_string(self.finished_at_utc, "task finished_at_utc")
        if self.status in {TaskStatus.PENDING, TaskStatus.RUNNING}:
            if (
                self.output_sha256 is not None
                or self.returncode is not None
                or self.failure_code is not None
                or self.finished_at_utc is not None
            ):
                raise SemanticError("incomplete task record contains final data")
        elif self.status is TaskStatus.PASSED:
            if (
                self.output_sha256 is None
                or self.returncode != 0
                or self.failure_code is not None
            ):
                raise SemanticError("passed task record is inconsistent")
        elif self.status is TaskStatus.FAILED:
            if self.output_sha256 is not None or self.failure_code is None:
                raise SemanticError("failed task record is inconsistent")
        elif self.status is TaskStatus.SKIPPED:
            if self.output_sha256 is not None or self.returncode is not None:
                raise SemanticError("skipped task record is inconsistent")


def atomic_replace_text(path: Path, text: str) -> None:
    destination = Path(path)
    if not isinstance(text, str):
        raise SemanticError("atomic replacement content must be text")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        ".{}.{}.tmp".format(destination.name, uuid.uuid4().hex)
    )
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        descriptor = os.open(str(destination.parent), os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as error:
        raise SemanticError("atomic state replacement failed") from error
    finally:
        if temporary.exists():
            temporary.unlink()


def _payload(record: TaskRecord) -> dict:
    return {
        "task_id": record.task_id,
        "status": record.status.value,
        "input_sha256": record.input_sha256,
        "output_sha256": record.output_sha256,
        "command": list(record.command),
        "returncode": record.returncode,
        "failure_code": record.failure_code,
        "log_path": record.log_path,
        "started_at_utc": record.started_at_utc,
        "finished_at_utc": record.finished_at_utc,
    }


def _record(payload: object) -> TaskRecord:
    if not isinstance(payload, dict) or set(payload) != _RECORD_FIELDS:
        raise SemanticError("experiment state task fields are invalid")
    try:
        status = TaskStatus(payload["status"])
    except (TypeError, ValueError) as error:
        raise SemanticError("experiment state task status is invalid") from error
    command = payload["command"]
    if not isinstance(command, list):
        raise SemanticError("experiment state task command is invalid")
    return TaskRecord(
        task_id=payload["task_id"],
        status=status,
        input_sha256=payload["input_sha256"],
        output_sha256=payload["output_sha256"],
        command=tuple(command),
        returncode=payload["returncode"],
        failure_code=payload["failure_code"],
        log_path=payload["log_path"],
        started_at_utc=payload["started_at_utc"],
        finished_at_utc=payload["finished_at_utc"],
    )


_TRANSITIONS = {
    TaskStatus.PENDING: frozenset(
        {TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.SKIPPED}
    ),
    TaskStatus.RUNNING: frozenset(
        {
            TaskStatus.RUNNING,
            TaskStatus.PASSED,
            TaskStatus.FAILED,
            TaskStatus.SKIPPED,
        }
    ),
    TaskStatus.PASSED: frozenset({TaskStatus.PASSED, TaskStatus.RUNNING}),
    TaskStatus.FAILED: frozenset(
        {TaskStatus.FAILED, TaskStatus.RUNNING, TaskStatus.SKIPPED}
    ),
    TaskStatus.SKIPPED: frozenset({TaskStatus.SKIPPED, TaskStatus.RUNNING}),
}


class ExperimentStateStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def load(self) -> Mapping[str, TaskRecord]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as error:
            raise SemanticError("experiment state is unreadable") from error
        if not isinstance(payload, dict) or set(payload) != _STATE_FIELDS:
            raise SemanticError("experiment state fields are invalid")
        if payload["schema_version"] != 1 or not isinstance(
            payload["tasks"],
            list,
        ):
            raise SemanticError("experiment state schema is invalid")
        records = {}
        for value in payload["tasks"]:
            record = _record(value)
            if record.task_id in records:
                raise SemanticError("experiment state contains duplicate tasks")
            records[record.task_id] = record
        return records

    def put(self, record: TaskRecord) -> None:
        if not isinstance(record, TaskRecord):
            raise SemanticError("state store requires a TaskRecord")
        records = dict(self.load())
        previous = records.get(record.task_id)
        if previous is not None and record.status not in _TRANSITIONS[
            previous.status
        ]:
            raise SemanticError("task status transition is invalid")
        records[record.task_id] = record
        payload = {
            "schema_version": 1,
            "tasks": [
                _payload(records[task_id]) for task_id in sorted(records)
            ],
        }
        text = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ) + "\n"
        atomic_replace_text(self.path, text)

    def reusable(
        self,
        task_id: str,
        input_sha256: str,
        output_sha256: str,
    ) -> bool:
        _identifier(task_id, "task_id")
        expected_input = _sha256(
            input_sha256,
            "task input_sha256",
            optional=False,
        )
        expected_output = _sha256(
            output_sha256,
            "task output_sha256",
            optional=False,
        )
        record = self.load().get(task_id)
        return (
            record is not None
            and record.status is TaskStatus.PASSED
            and record.input_sha256 == expected_input
            and record.output_sha256 == expected_output
        )
