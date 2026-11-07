"""Bounded local POSIX jobs, with streaming logs and PID birth identities.

The ownership scope is the newly created process group. Remote helpers and GPU
resource release have separate acknowledgements; a local exit does not certify
either of them. Managed programs must keep their children in this group.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import codecs
import hashlib
import math
import os
from pathlib import Path
import selectors
import shutil
import signal
import subprocess
import time
import uuid

from vericcl.execution.context import RunContext, StorageFailure
from vericcl.execution.events import emit_event
from vericcl.execution.security import Redactor, StreamRedactor, sensitive_key


def _proc_stat(pid: int):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[0], int(fields[2]), int(fields[19])
    except (FileNotFoundError, ProcessLookupError):
        return None


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    pgid: int
    start_ticks: int
    boot_id: str

    @classmethod
    def capture(cls, pid: int):
        state = _proc_stat(pid)
        if state is None:
            raise ProcessLookupError(pid)
        return cls(pid, state[1], state[2], Path("/proc/sys/kernel/random/boot_id").read_text().strip())

    def matches(self) -> bool:
        current = _proc_stat(self.pid)
        return (current is not None and current[1:] == (self.pgid, self.start_ticks)
                and Path("/proc/sys/kernel/random/boot_id").read_text().strip() == self.boot_id)


def signal_owned_process_group(identity: ProcessIdentity, sig: int) -> bool:
    # The group leader remains unreaped while descendants are managed, so its
    # PID cannot be recycled between this check and killpg.
    if identity.pid != identity.pgid or not identity.matches():
        return False
    try:
        os.killpg(identity.pgid, sig)
        return True
    except ProcessLookupError:
        return False


def _active_group_members(identity: ProcessIdentity) -> tuple[int, ...]:
    # Same predicate as _proc_stat (state, pgrp), using raw descriptors: this
    # scan visits every /proc entry after each owned job, and pathlib/text
    # decoding dominated its cost on hosts with ~1000 processes.
    active = []
    pgid = identity.pgid
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            descriptor = os.open("/proc/" + name + "/stat", os.O_RDONLY)
        except (FileNotFoundError, ProcessLookupError):
            continue
        try:
            raw = os.read(descriptor, 4096)
        except (FileNotFoundError, ProcessLookupError):
            continue
        finally:
            os.close(descriptor)
        fields = raw.rsplit(b")", 1)[-1].split()
        if len(fields) > 2 and int(fields[2]) == pgid and fields[0] not in (b"Z", b"X"):
            active.append(int(name))
    return tuple(active)


def reconcile_owned_process(context, result) -> bool:
    """Recheck late OS cleanup without changing execution or GPU receipts."""
    if result.cleanup_complete is True:
        return True
    identity = result.identity
    if identity is None or identity.pid != identity.pgid:
        return False
    if Path('/proc/sys/kernel/random/boot_id').read_text().strip() != identity.boot_id:
        return False
    relative = 'jobs/' + result.job_id + '/cleanup-reconciliation.json'
    with context.lock:
        existing = context.artifact(relative)
        if existing.is_file():
            import json
            previous = json.loads(existing.read_text())
            if previous.get('identity') != asdict(identity) or previous.get('cleanup_complete') is not True:
                raise RuntimeError('owned process cleanup receipt differs')
            context.active_jobs.discard(result.job_id)
            return True
        if _active_group_members(identity):
            return False
        # Only reap our exact zombie leader, if it remains a child. Never send
        # signals or treat disappearance as evidence of GPU completion.
        if identity.matches():
            try:
                os.waitpid(identity.pid, os.WNOHANG)
            except ChildProcessError:
                pass
        receipt = dict(job_id=result.job_id, identity=asdict(identity), cleanup_complete=True,
                       scope='owned_local_process_group', original_execution_unchanged=True)
        context.write_json(relative, receipt)
        context.active_jobs.discard(result.job_id)
        emit_event(context, 'job_cleanup_reconciled', receipt)
        return True


@dataclass(frozen=True)
class ProcessLease:
    deadline_monotonic: float
    cleanup_reserve_s: float

    def __post_init__(self):
        if (not math.isfinite(self.deadline_monotonic) or not math.isfinite(self.cleanup_reserve_s)
                or self.cleanup_reserve_s <= 0):
            raise ValueError("process lease needs a finite deadline and positive cleanup reserve")

    @classmethod
    def after(cls, timeout_s: float, *, cleanup_reserve_s: float):
        if not math.isfinite(timeout_s) or timeout_s <= cleanup_reserve_s:
            raise ValueError("timeout must exceed its cleanup reserve")
        return cls(time.monotonic() + timeout_s, cleanup_reserve_s)


@dataclass(frozen=True)
class ProcessResult:
    job_id: str
    status: str
    returncode: int | None
    identity: ProcessIdentity | None
    stdout_path: Path
    stderr_path: Path
    stdout_bytes: int
    stderr_bytes: int
    stdout_truncated: bool
    stderr_truncated: bool
    cleanup_complete: bool
    elapsed_s: float
    capture_complete: bool
    cleanup_scope: str = "owned_local_process_group"

    def as_json(self):
        result = asdict(self)
        result["stdout_path"] = str(self.stdout_path)
        result["stderr_path"] = str(self.stderr_path)
        return result


# Executable digests keyed by the file's full stat identity. Any replacement or
# in-place modification changes the inode, size, mtime or (kernel-maintained)
# ctime, so a hit always describes the same bytes that a fresh hash would read.
_EXECUTABLE_DIGESTS: dict = {}


def _executable_identity(argv, cwd, environment):
    executable = shutil.which(argv[0], path=environment.get("PATH"))
    if executable is None:
        return {"path": None, "sha256": None}
    path = Path(executable)
    if not path.is_absolute():
        path = cwd / path
    path = path.resolve()
    try:
        status = path.stat()
        key = (str(path), status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns, status.st_ctime_ns)
        digest = _EXECUTABLE_DIGESTS.get(key)
        if digest is None:
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if path.stat().st_ctime_ns == status.st_ctime_ns:
                _EXECUTABLE_DIGESTS[key] = digest
    except OSError:
        digest = None
    return {"path": str(path), "sha256": digest}


def run_owned_process(context: RunContext, argv, lease: ProcessLease, *, cwd=None,
                      environment=None, job_id=None, rank=None, max_output_bytes=None, cancel_check=None,
                      durable_records=True) -> ProcessResult:
    """Run one owned local process group.

    ``durable_records=False`` is for CPU-only helpers (owned SCIP workers):
    their request, identity and result are published atomically under
    jobs/<id>/ without fsync. Durable job_requested/job_started/job_finished
    events retain the command, process identity and outcome; the full request
    metadata remains in request.json.
    """
    def record(relative, value):
        if durable_records:
            context.write_json(relative, value)
        else:
            context.write_unsynced_bytes(relative, context.encode_json(value))

    command = tuple(argv)
    if not command or any(not isinstance(a, str) or "\x00" in a for a in command):
        raise ValueError("argv must contain valid string arguments")
    if max_output_bytes is not None and (isinstance(max_output_bytes, bool)
                                         or not isinstance(max_output_bytes, int) or max_output_bytes < 0):
        raise ValueError("output byte limit must be a nonnegative integer")
    if not Path("/proc/sys/kernel/random/boot_id").is_file():
        raise RuntimeError("owned process identities require Linux /proc")
    context.ensure_writable()
    started = time.monotonic()
    job_id = job_id or uuid.uuid4().hex
    if not isinstance(job_id, str) or not job_id or any(not(c.isalnum() or c in "_.-") for c in job_id) or job_id in {".", ".."}:
        raise ValueError("invalid job identifier")
    directory = context.artifact("jobs/" + job_id)
    directory.mkdir()
    paths = {name: directory / (name + ".log") for name in ("stdout", "stderr")}
    actual_cwd = Path(cwd or context.root).resolve()
    env = dict(os.environ)
    env.update(environment or {})
    argument_secrets = []
    for index, argument in enumerate(command):
        if "=" in argument and sensitive_key(argument.split("=", 1)[0]):
            argument_secrets.append(argument.split("=", 1)[1])
        elif sensitive_key(argument) and index + 1 < len(command):
            argument_secrets.append(command[index + 1])
    context.redactor = Redactor(context.redactor.secrets + tuple(argument_secrets), env)
    record("jobs/" + job_id + "/request.json", {
        "job_id": job_id, "argv": command, "cwd": str(actual_cwd), "environment": env,
        "program": _executable_identity(command, actual_cwd, env),
        "deadline_monotonic": lease.deadline_monotonic, "rank": rank,
    })
    emit_event(context, "job_requested", {"job_id": job_id, "argv": command}, rank=rank)
    counts = {"stdout": 0, "stderr": 0}
    written = {"stdout": 0, "stderr": 0}
    truncated = {"stdout": False, "stderr": False}
    process, identity, code = None, None, None
    status, cleanup = "not_started", True
    files, selector = {}, selectors.DefaultSelector()
    filters = {name: StreamRedactor(context.redactor.secrets) for name in paths}
    decoders = {name: codecs.getincrementaldecoder("utf-8")("replace") for name in paths}
    capture_complete = True
    storage_failure = None
    cancelled = False
    term_sent = kill_sent = False
    cleanup_deadline = lease.deadline_monotonic
    execution_deadline = lease.deadline_monotonic - lease.cleanup_reserve_s
    kill_deadline = lease.deadline_monotonic - lease.cleanup_reserve_s / 2

    def safe_emit(kind, data):
        nonlocal storage_failure
        if storage_failure is None:
            try:
                emit_event(context, kind, data, rank=rank)
            except StorageFailure as error:
                storage_failure = str(error)

    def save(name, raw, final=False):
        nonlocal storage_failure
        counts[name] += len(raw)
        redacted = filters[name].feed(raw, final=final)
        data = decoders[name].decode(redacted, final=final).encode("utf-8")
        allowed = len(data) if max_output_bytes is None else max(0, max_output_bytes - written[name])
        if len(data) > allowed and not truncated[name]:
            truncated[name] = True
            safe_emit("output_truncated", {"job_id": job_id, "stream": name, "limit_bytes": max_output_bytes})
        if storage_failure is None:
            try:
                retained = data[:allowed].decode("utf-8", errors="ignore").encode("utf-8")
                files[name].write(retained)
                files[name].flush()
                written[name] += len(retained)
            except OSError as error:
                storage_failure = context.redactor.text(str(error))
                context.storage_error = storage_failure

    try:
        for name, path in paths.items():
            files[name] = path.open("xb")
        if time.monotonic() < execution_deadline and not (cancel_check is not None and cancel_check()):
            process = subprocess.Popen(command, cwd=actual_cwd, env=env,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
            identity = ProcessIdentity.capture(process.pid)
            context.active_jobs.add(job_id)
            record("jobs/" + job_id + "/identity.json", asdict(identity))
            safe_emit("job_started", {"job_id": job_id, "identity": asdict(identity)})
            status, cleanup = "running", False
            for name, pipe in (("stdout", process.stdout), ("stderr", process.stderr)):
                os.set_blocking(pipe.fileno(), False)
                selector.register(pipe, selectors.EVENT_READ, name)
            while True:
                now = time.monotonic()
                if cancel_check is not None and cancel_check():
                    cancelled = True
                if now >= execution_deadline and status == "running":
                    status = "timeout"
                if storage_failure is not None:
                    status = "storage_failure"
                if cancelled:
                    status = "cancelled"
                if status != "running" and not term_sent:
                    term_sent = signal_owned_process_group(identity, signal.SIGTERM)
                if now >= kill_deadline and status != "running" and not kill_sent:
                    kill_sent = signal_owned_process_group(identity, signal.SIGKILL)
                for key, _ in selector.select(timeout=max(0, min(0.02, lease.deadline_monotonic - now))):
                    try:
                        raw = os.read(key.fileobj.fileno(), 65536)
                    except BlockingIOError:
                        continue
                    if raw:
                        save(key.data, raw)
                    else:
                        save(key.data, b"", final=True)
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                # WNOWAIT preserves the leader's birth identity until group
                # cleanup is checked. Popen.poll would reap it too early.
                exited = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                if exited is not None:
                    members = _active_group_members(identity)
                    if not members and not selector.get_map():
                        cleanup = True
                        code = process.wait(timeout=0)
                        if status == "running":
                            status = "completed" if code == 0 else "failed"
                        break
                if time.monotonic() >= lease.deadline_monotonic:
                    cleanup = not _active_group_members(identity)
                    if exited is not None:
                        code = process.wait(timeout=0)
                    break
    except KeyboardInterrupt:
        # A synchronous caller cancellation still has a bounded cleanup path.
        cancelled, status = True, "cancelled"
        cleanup_deadline = min(cleanup_deadline, time.monotonic() + lease.cleanup_reserve_s)
        if identity is not None:
            signal_owned_process_group(identity, signal.SIGKILL)
    except StorageFailure as error:
        storage_failure, status = str(error), "storage_failure"
        cleanup_deadline = min(cleanup_deadline, time.monotonic() + lease.cleanup_reserve_s)
        if identity is not None:
            signal_owned_process_group(identity, signal.SIGKILL)
    except OSError as error:
        status = "spawn_failed" if process is None else "failed"
        cleanup_deadline = min(cleanup_deadline, time.monotonic() + lease.cleanup_reserve_s)
        safe_emit("job_error", {"job_id": job_id, "error": str(error)})
        if identity is not None:
            signal_owned_process_group(identity, signal.SIGKILL)
    finally:
        capture_complete = not bool(selector.get_map())
        selector.close()
        if process is not None:
            for pipe in (process.stdout, process.stderr):
                if pipe is not None and not pipe.closed:
                    pipe.close()
            if code is None:
                remaining = max(0, cleanup_deadline - time.monotonic())
                try:
                    code = process.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    pass
                cleanup = identity is not None and not _active_group_members(identity)
            if cleanup:
                context.active_jobs.discard(job_id)
        for name, stream in files.items():
            if not stream.closed:
                save(name, b"", final=True)
                try:
                    # An empty capture has no bytes to make durable; its
                    # existence is already recorded by the job request/result.
                    if written[name] or counts[name]:
                        os.fsync(stream.fileno())
                    stream.close()
                except OSError as error:
                    context.storage_error = storage_failure = context.redactor.text(str(error))
    if storage_failure is not None:
        status = "storage_failure"
    result = ProcessResult(job_id, status, code, identity, paths["stdout"], paths["stderr"],
                           counts["stdout"], counts["stderr"], truncated["stdout"], truncated["stderr"],
                           cleanup, time.monotonic() - started, capture_complete)
    if storage_failure is None:
        record("jobs/" + job_id + "/result.json", result.as_json())
        emit_event(context, "job_finished", result.as_json(), rank=rank)
    return result
