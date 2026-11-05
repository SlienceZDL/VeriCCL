"""One owned case run and the evidence required to start more work."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import threading
import uuid

from vericcl.artifacts.layout import resolve_case_paths
from vericcl.artifacts.writer import atomic_write_bytes
from vericcl.execution.security import ENVIRONMENT_ALLOWLIST, Redactor
from vericcl.execution.status import ExitCode, FinalizationResult, RunResult


DIRECTORIES = ("input_snapshot", "models", "candidates", "jobs", "live", "statistics",
               "checkpoints", "reports", "best", "tmp", "rebuildable-cache")


class StorageFailure(RuntimeError):
    """Essential evidence could not be retained; no new job may start."""


class RunContext:
    def __init__(self, root: Path, *, task_id: str, run_id: str, redactor: Redactor):
        self.root = root
        self.task_id = task_id
        self.run_id = run_id
        self.redactor = redactor
        self.node = socket.gethostname()
        self.lock = threading.RLock()
        self.sequence = 0
        self.storage_error = None
        self.finalized = False
        self.active_jobs = set()

    @classmethod
    def create(cls, project_root: Path, case_id: str, run_id: str, *, input_files=None,
               environment=None, secret_values=(), task_id=None):
        paths = resolve_case_paths(project_root, case_id, run_id)
        redactor = Redactor(secret_values, environment)
        snapshots = {}
        for name, source in (input_files or {}).items():
            if not isinstance(name, str) or Path(name).name != name or name in {"", ".", ".."} or "\\" in name:
                raise ValueError("snapshot name must be a single filename")
            data = Path(source).read_bytes()
            redactor.check_snapshot(data)
            snapshots[name] = data
        # Resolve before any mutation, including each existing symlink component.
        paths.input.mkdir(parents=True, exist_ok=True)
        paths.output.mkdir(parents=True, exist_ok=True)
        if paths.run.exists() and (not paths.run.is_dir() or any(paths.run.iterdir())):
            raise FileExistsError("refusing a nonempty run directory: " + str(paths.run))
        paths.run.mkdir(exist_ok=True)
        context = cls(paths.run, task_id=task_id or uuid.uuid4().hex, run_id=run_id, redactor=redactor)
        for name in DIRECTORIES:
            context.artifact(name).mkdir()
        snapshot_hashes = {}
        for name, data in snapshots.items():
            context.write_bytes("input_snapshot/" + name, data)
            snapshot_hashes[name] = hashlib.sha256(data).hexdigest()
        context.write_json("input-snapshot.json", {"files": snapshot_hashes})
        context.write_json("run.json", {"schema_version": 1, "task_id": context.task_id,
                           "run_id": run_id, "case_id": case_id, "root": str(context.root),
                           "node": context.node, "environment": environment or {}})
        from vericcl.execution.events import emit_event
        emit_event(context, "run_started")
        return context

    def ensure_writable(self):
        if self.storage_error is not None:
            raise StorageFailure(self.storage_error)
        if self.finalized:
            raise RuntimeError("run is already finalized")

    def fail_storage(self, error):
        self.storage_error = "essential evidence storage failed: " + self.redactor.text(str(error))
        raise StorageFailure(self.storage_error) from error

    def artifact(self, relative) -> Path:
        path = Path(relative)
        if path.is_absolute() or not path.parts or ".." in path.parts or "\\" in str(relative):
            raise ValueError("artifact path violates run boundary")
        target = self.root / path
        if not target.resolve().is_relative_to(self.root):
            raise ValueError("artifact symlink violates run boundary")
        return target

    def write_bytes(self, relative, data: bytes, *, replace=False) -> Path:
        with self.lock:
            self.ensure_writable()
            path = self.artifact(relative)
            if replace and Path(relative).parts[0] == "input_snapshot":
                raise ValueError("input snapshot is immutable")
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                if not replace:
                    atomic_write_bytes(path, data)
                else:
                    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
                    try:
                        atomic_write_bytes(temporary, data)
                        os.replace(temporary, path)
                        descriptor = os.open(path.parent, os.O_RDONLY)
                        try:
                            os.fsync(descriptor)
                        finally:
                            os.close(descriptor)
                    finally:
                        temporary.unlink(missing_ok=True)
            except FileExistsError:
                raise
            except OSError as error:
                self.fail_storage(error)
            return path

    def write_json(self, relative, value, *, replace=False) -> Path:
        text = json.dumps(self.redactor.value(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        return self.write_bytes(relative, text.encode("utf-8"), replace=replace)

    def cleanup_rebuildable(self):
        with self.lock:
            if self.active_jobs:
                raise RuntimeError("cannot reclaim cache while owned jobs are active")
            for name in ("tmp", "rebuildable-cache"):
                directory = self.artifact(name)
                for child in directory.iterdir():
                    # Refuse rather than follow an unexpected outside reference.
                    self.artifact(child.relative_to(self.root))
                    if child.is_dir() and not child.is_symlink():
                        shutil.rmtree(child)
                    else:
                        child.unlink()


def finalize_run(context: RunContext, result: RunResult) -> FinalizationResult:
    from vericcl.execution.events import emit_event
    if context.active_jobs:
        return FinalizationResult(int(ExitCode.STORAGE), False, ("owned jobs remain unresolved",))
    payload = asdict(result)
    payload.update(schema_version=1, task_id=context.task_id, run_id=context.run_id, exit_code=result.exit_code)
    try:
        if getattr(context, 'resumed', False):
            for relative in ('run-summary.json', 'reports/summary.md'):
                previous = context.artifact(relative)
                if previous.is_file():
                    context.write_bytes('reports/previous-' + uuid.uuid4().hex + '-' + previous.name, previous.read_bytes())
        context.write_json("run-summary.json", payload, replace=getattr(context, 'resumed', False))
        text = ("本次运行结果\n\n任务：" + context.task_id + "\n\n执行状态：" + result.execution
                + "\n\n模型覆盖：" + result.model_coverage + "\n\n候选采用：" + result.acceptance
                + "\n\n结束原因：" + result.stop_reason + "\n\n退出码：" + str(result.exit_code) + "\n")
        context.write_bytes("reports/summary.md", context.redactor.text(text).encode("utf-8"), replace=getattr(context, "resumed", False))
        emit_event(context, "run_finalized", payload)
        context.finalized = True
        return FinalizationResult(result.exit_code, True)
    except (StorageFailure, OSError) as error:
        return FinalizationResult(int(ExitCode.STORAGE), False, (context.redactor.text(str(error)),))
