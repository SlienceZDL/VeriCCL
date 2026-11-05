from dataclasses import replace
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def _run(tmp_path, script, *, timeout=4, **options):
    from vericcl.execution.context import RunContext
    from vericcl.execution.processes import ProcessLease, run_owned_process
    context = RunContext.create(tmp_path, "process", "r", environment=options.pop("environment", None))
    result = run_owned_process(context, [sys.executable, "-u", "-c", script],
                               ProcessLease.after(timeout, cleanup_reserve_s=0.4), **options)
    return context, result


def test_both_streams_are_drained_without_pipe_deadlock(tmp_path):
    _, result = _run(tmp_path, "import os\nfor i in range(128):\n os.write(1,b'a'*16384)\n os.write(2,b'b'*16384)")
    assert result.status == "completed" and result.returncode == 0
    assert result.cleanup_complete
    assert result.stdout_path.stat().st_size == 128 * 16384
    assert result.stderr_path.stat().st_size == 128 * 16384
    assert result.stdout_bytes == result.stderr_bytes == 128 * 16384


def test_timeout_keeps_partial_output_and_kills_an_ignoring_child(tmp_path):
    script = """import os, signal, time
child = os.fork()
if child == 0:
 signal.signal(signal.SIGTERM, signal.SIG_IGN)
 print('child=' + str(os.getpid()), flush=True)
 while True: time.sleep(1)
else:
 print('parent-started', flush=True)
 while True: time.sleep(1)
"""
    started = time.monotonic()
    _, result = _run(tmp_path, script, timeout=1.2)
    assert time.monotonic() - started < 2.0
    assert result.status == "timeout" and result.cleanup_complete
    output = result.stdout_path.read_text()
    assert "parent-started" in output and "child=" in output
    pid = int(next(line[6:] for line in output.splitlines() if line.startswith("child=")))
    stat = Path(f"/proc/{pid}/stat")
    assert not stat.exists() or stat.read_text().rsplit(')', 1)[1].split()[0] == "Z"


def test_identity_mismatch_never_signals_a_process(tmp_path):
    from vericcl.execution.processes import ProcessIdentity, signal_owned_process_group
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"], start_new_session=True)
    try:
        identity = ProcessIdentity.capture(process.pid)
        assert not signal_owned_process_group(replace(identity, start_ticks=identity.start_ticks + 1), signal.SIGTERM)
        assert process.poll() is None
    finally:
        process.terminate()
        process.wait(timeout=2)


def test_capped_output_is_drained_and_truncation_is_reported(tmp_path):
    context, result = _run(tmp_path, "import os; os.write(1,b'x'*100000)", max_output_bytes=100)
    assert result.status == "completed"
    assert result.stdout_path.stat().st_size <= 100
    assert result.stdout_bytes == 100000 and result.stdout_truncated
    events = [json.loads(line) for line in (context.root / "events.jsonl").read_text().splitlines()]
    assert any(e["kind"] == "output_truncated" for e in events)


def test_secret_split_across_stream_chunks_never_reaches_disk(tmp_path):
    token = "private-token-over-multiple-writes"
    script = "import os,time\nos.write(1,b'private-token-')\ntime.sleep(.05)\nos.write(1,b'over-multiple-writes\\n')"
    context, result = _run(tmp_path, script, environment={"API_TOKEN": token})
    assert result.status == "completed"
    for path in context.root.rglob("*"):
        if path.is_file():
            assert token not in path.read_text()
    assert "[REDACTED]" in result.stdout_path.read_text()


def test_expired_lease_records_no_launch(tmp_path):
    from vericcl.execution.context import RunContext
    from vericcl.execution.processes import ProcessLease, run_owned_process
    context = RunContext.create(tmp_path, "expired", "r")
    marker = tmp_path / "not-started"
    result = run_owned_process(context, [sys.executable, "-c", f"open({str(marker)!r},'w').close()"],
                               ProcessLease(time.monotonic() - 1, 0.1))
    assert result.status == "not_started" and result.identity is None
    assert not marker.exists()


def test_a_detached_writer_is_reported_as_unresolved_at_the_deadline(tmp_path):
    # Setsid leaves the explicitly owned group. The runner must neither claim
    # its log stream completed nor kill a process outside that ownership scope.
    script = """import os,time
if os.fork() == 0:
 os.setsid()
 print('detached=' + str(os.getpid()), flush=True)
 time.sleep(3)
else:
 time.sleep(.05)
"""
    context, result = _run(tmp_path, script, timeout=0.8)
    pid = int(result.stdout_path.read_text().strip().split("=")[1])
    try:
        assert result.status == "timeout"
        assert not result.capture_complete
        assert result.cleanup_complete  # Its local process group has ended.
    finally:
        os.kill(pid, signal.SIGTERM)


def test_process_logs_are_valid_utf8_even_for_malformed_output(tmp_path):
    _, result = _run(tmp_path, "import os; os.write(1,b'hello\\xff\\n')")
    assert result.stdout_path.read_text(encoding="utf-8") == "hello\ufffd\n"


def test_runtime_compiler_uses_owned_process_logs_and_group_cleanup(tmp_path, monkeypatch):
    from runtime.build_runtime import _compile
    tools = tmp_path / "bin"
    tools.mkdir()
    make = tools / "make"
    make.write_text("#!/bin/sh\necho build-started\nsleep 5\n")
    make.chmod(0o755)
    monkeypatch.setenv("PATH", str(tools) + os.pathsep + os.environ["PATH"])
    output = tmp_path / "build-run"
    output.mkdir()
    result = _compile(tmp_path, output, tmp_path, 70, 1, 1)
    assert result["status"] == "timeout" and not result["compiled"]
    assert result["cleanup_complete"]
    assert list((output / "jobs").glob("*/identity.json"))
    assert "build-started" in Path(result["stdout_path"]).read_text()


def test_caller_cancellation_stops_owned_work_and_reports_cancellation(tmp_path, monkeypatch):
    import selectors
    original = selectors.EpollSelector.select
    raised = False
    def interrupt_once(self, timeout=None):
        nonlocal raised
        if not raised:
            raised = True
            raise KeyboardInterrupt
        return original(self, timeout)
    monkeypatch.setattr(selectors.EpollSelector, "select", interrupt_once)
    started = time.monotonic()
    _, result = _run(tmp_path, "import time; time.sleep(10)", timeout=5)
    assert result.status == "cancelled" and result.cleanup_complete
    assert time.monotonic() - started < 1
