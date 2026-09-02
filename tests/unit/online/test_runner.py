from dataclasses import replace
import os
from pathlib import Path
import sys

import pytest
import vericcl.verification.online.runner as runner_module

from vericcl.errors import SemanticError
from vericcl.semantics.collective import CollectiveKind
from vericcl.topology import load_topology
from vericcl.verification.online.model import NcclTestRequest
from vericcl.verification.online.runner import (
    NcclTestsRunner,
    ProcessRequest,
    ProcessResult,
    SubprocessCommandExecutor,
    TraceCollectionRequest,
    TraceCollectionResult,
    collect_trace_files,
    process_environment,
)
from vericcl.verification.online.trace_format import (
    RawStepTraceRecord,
    encode_raw_trace,
)
from vericcl.xml.lower import lower_to_xml
from vericcl.xml.trace_sidecar import build_trace_sidecar

from tests.unit.xml.helpers import resolved, two_rank_allreduce_schedule


pytestmark = pytest.mark.phase06


def _request():
    return NcclTestRequest(
        kind=CollectiveKind.ALL_REDUCE,
        message_size_bytes=1024,
        datatype="float",
        reduction_op="sum",
        root=None,
        inplace=False,
        binary_directory="/tmp/nccl-tests",
    )


def _row(value):
    return (
        "1024 256 float sum -1 "
        "{} 100.0 150.0 0 {} 90.0 140.0 0"
    ).format(value, value + 1.0)


class SequenceExecutor:
    def __init__(self, values):
        self.values = iter(values)
        self.calls = []

    def run(self, request):
        self.calls.append(request)
        if request.command[-1:] == ("--help",):
            return ProcessResult(
                0,
                "usage: perf -b -e -w -n -c -g -d -o -r",
                "",
            )
        value = next(self.values)
        return ProcessResult(0, _row(value), "")


def test_runner_retries_three_independent_unstable_rounds():
    values = tuple(
        1.0 if index % 2 else 20.0
        for index in range(60)
    )
    executor = SequenceExecutor(values)
    runner = NcclTestsRunner(
        executor,
        environment={"A": "B"},
        launcher_prefix=("mpirun", "-np", "2"),
    )

    history = runner.measure(_request())

    assert len(history.rounds) == 3
    assert history.stable is False
    assert len(history.all_samples_us) == 60
    assert executor.calls[0].command[-1] == "--help"
    assert executor.calls[1].command[:3] == ("mpirun", "-np", "2")


def test_runner_validates_release_with_one_correctness_process():
    executor = SequenceExecutor((10.0,))
    runner = NcclTestsRunner(executor, environment={"A": "B"})

    runner.validate_release(_request())

    assert len(executor.calls) == 2
    command = executor.calls[1].command
    assert command[command.index("-w") + 1] == "5"
    assert command[command.index("-n") + 1] == "20"
    assert command[command.index("-c") + 1] == "1"


def test_runner_rejects_multiple_rows_and_invalid_executor_results():
    class MultipleRows:
        def run(self, request):
            if request.command[-1:] == ("--help",):
                return ProcessResult(
                    0,
                    "usage: perf -b -e -w -n -c -g -d -o -r",
                    "",
                )
            return ProcessResult(0, _row(1.0) + "\n" + _row(2.0), "")

    with pytest.raises(SemanticError, match="one performance row"):
        NcclTestsRunner(MultipleRows(), environment={}).measure(_request())

    class InvalidResult:
        def run(self, request):
            return object()

    with pytest.raises(SemanticError, match="invalid process result"):
        NcclTestsRunner(InvalidResult(), environment={}).validate_help(
            _request()
        )
    with pytest.raises(SemanticError, match="executor"):
        NcclTestsRunner(object(), environment={})


def test_subprocess_executor_runs_with_exact_environment_and_reports_errors():
    environment = process_environment({"VERICCL_RUNNER_TEST": "visible"})
    request = ProcessRequest(
        command=(
            sys.executable,
            "-c",
            "import os; print(os.environ['VERICCL_RUNNER_TEST'])",
        ),
        environment=environment,
        label="python smoke",
        cwd=Path.cwd(),
        timeout_s=5,
    )

    result = SubprocessCommandExecutor().run(request)

    assert result.returncode == 0
    assert result.stdout.strip() == "visible"
    with pytest.raises(SemanticError, match="ProcessRequest"):
        SubprocessCommandExecutor().run(object())
    with pytest.raises(SemanticError, match="could not be executed"):
        SubprocessCommandExecutor().run(
            ProcessRequest(
                ("/path/that/does/not/exist",),
                os.environ,
                "missing",
            )
        )


def test_runner_shares_one_declining_wall_clock_budget(monkeypatch):
    class Executor:
        def __init__(self):
            self.calls = []

        def run(self, request):
            self.calls.append(request)
            return ProcessResult(0, "ok", "")

    ticks = iter((100.0, 101.0, 104.0, 106.0))
    monkeypatch.setattr(runner_module, "_monotonic", lambda: next(ticks))
    executor = Executor()
    runner = NcclTestsRunner(executor, environment={}, timeout_s=5.0)

    runner.run_auxiliary(("first",), "first", {})
    runner.run_auxiliary(("second",), "second", {})

    assert executor.calls[0].timeout_s == pytest.approx(4.0)
    assert executor.calls[1].timeout_s == pytest.approx(1.0)
    with pytest.raises(SemanticError, match="wall-clock budget"):
        runner.run_auxiliary(("third",), "third", {})


@pytest.mark.parametrize(
    "factory",
    (
        lambda: ProcessRequest(object(), {}, "label"),
        lambda: ProcessRequest((), {}, "label"),
        lambda: ProcessRequest(("ok",), object(), "label"),
        lambda: ProcessRequest(("ok",), {"A": object()}, "label"),
        lambda: ProcessRequest(("ok",), {}, ""),
        lambda: ProcessRequest(("ok",), {}, "label", timeout_s=0),
        lambda: ProcessResult(True, "", ""),
        lambda: ProcessResult(0, object(), ""),
    ),
)
def test_process_models_reject_invalid_values(factory):
    with pytest.raises(SemanticError):
        factory()


def _clock_output():
    rows = []
    for rank in (0, 1):
        for index in range(3):
            ticks = 1000 + index * 100
            host = 1000000 + index * 100
            rows.append(
                "VERICCL_CLOCK_SYNC {} {} {} {} 0 0".format(
                    rank,
                    ticks,
                    host * 1000,
                    host * 1000,
                )
            )
    return "\n".join(rows)


def _trace_fixture(tmp_path, *, iterations=(0,)):
    schedule = two_rank_allreduce_schedule()
    inputs = resolved(CollectiveKind.ALL_REDUCE, ranks=2, slices=1)
    artifact = lower_to_xml(schedule, inputs, load_topology(inputs))
    sidecar = build_trace_sidecar(artifact, schedule)
    prefix = tmp_path / "trace"
    for rank in range(2):
        records = []
        for iteration in iterations:
            for entry in sorted(
                sidecar.entries.values(), key=lambda item: item.key
            ):
                if entry.rank != rank:
                    continue
                records.append(
                    RawStepTraceRecord(
                        rank=rank,
                        tb_id=entry.tb_id,
                        step_index=entry.step_index,
                        endpoint_type=entry.runtime_endpoint_type,
                        peer=entry.peer,
                        channel=entry.runtime_channel,
                        iteration=iteration,
                        tb_reach=10,
                        dependency_done=20,
                        transfer_start=30,
                        transfer_end=40,
                        flags=0,
                        reserved=0,
                    )
                )
        Path("{}.rank-{}.bin".format(prefix, rank)).write_bytes(
            encode_raw_trace(records, rank=rank)
        )
    return TraceCollectionRequest(sidecar, prefix, 2, _clock_output(), 10.0)


def test_default_trace_collector_parses_every_rank_and_analyzes(tmp_path):
    request = _trace_fixture(tmp_path)

    result = collect_trace_files(request)

    assert isinstance(result, TraceCollectionResult)
    assert result.complete is True
    assert len(result.rank_files) == 2
    assert result.analysis.intervals


def test_trace_collector_selects_only_twenty_measured_invocations(tmp_path):
    request = replace(
        _trace_fixture(tmp_path, iterations=tuple(range(21))),
        measured_iterations=20,
    )

    result = collect_trace_files(request)

    assert {
        interval.iteration for interval in result.analysis.intervals
    } == set(range(1, 21))


def test_trace_collector_selects_requested_inplace_invocation_block(tmp_path):
    request = replace(
        _trace_fixture(tmp_path, iterations=tuple(range(42))),
        measured_iterations=20,
        inplace=True,
    )

    result = collect_trace_files(request)

    assert {
        interval.iteration for interval in result.analysis.intervals
    } == set(range(22, 42))


def test_trace_collector_rejects_unexpected_invocation_count(tmp_path):
    request = replace(
        _trace_fixture(tmp_path, iterations=tuple(range(20))),
        measured_iterations=20,
    )

    with pytest.raises(SemanticError, match="invocation count"):
        collect_trace_files(request)


def test_default_trace_collector_rejects_missing_rank_file(tmp_path):
    request = _trace_fixture(tmp_path)
    request.file_prefix.with_name(
        request.file_prefix.name + ".rank-1.bin"
    ).unlink()

    with pytest.raises(SemanticError, match="missing"):
        collect_trace_files(request)
    with pytest.raises(SemanticError, match="trace request"):
        collect_trace_files(object())


def test_default_trace_collector_enforces_clock_uncertainty_limit(tmp_path):
    request = _trace_fixture(tmp_path)
    uncertain_output = request.clock_sync_output.replace(
        "1000000000 1000000000",
        "999990000 1000010000",
    ).replace(
        "1000100000 1000100000",
        "1000090000 1000110000",
    ).replace(
        "1000200000 1000200000",
        "1000190000 1000210000",
    )

    with pytest.raises(SemanticError, match="configured maximum"):
        collect_trace_files(
            replace(
                request,
                clock_sync_output=uncertain_output,
                max_clock_uncertainty_us=1.0,
            )
        )


def test_trace_collection_models_reject_inconsistent_values(tmp_path):
    request = _trace_fixture(tmp_path)
    with pytest.raises(SemanticError, match="rank count differs"):
        TraceCollectionRequest(
            request.sidecar,
            request.file_prefix,
            1,
            request.clock_sync_output,
            request.max_clock_uncertainty_us,
        )
    with pytest.raises(SemanticError, match="clock output"):
        TraceCollectionRequest(
            request.sidecar,
            request.file_prefix,
            2,
            "",
            10.0,
        )
    with pytest.raises(SemanticError, match="rank files"):
        TraceCollectionResult(
            collect_trace_files(request).analysis,
            (),
            True,
        )
