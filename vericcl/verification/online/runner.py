from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import subprocess
import time
from types import MappingProxyType
from typing import Mapping, Optional, Protocol, Tuple

from vericcl.errors import SemanticError
from vericcl.verification.online.clock_sync import (
    align_clocks,
    parse_clock_sync_output,
)
from vericcl.verification.online.model import NcclTestRequest
from vericcl.verification.online.nccl_tests import (
    NcclTestsHelpValidator,
    build_nccl_tests_command,
    build_nccl_tests_trace_command,
    parse_nccl_tests_output,
)
from vericcl.verification.online.statistics import (
    MEASUREMENT_SAMPLE_COUNT,
    PerformanceHistory,
)
from vericcl.verification.online.trace_analysis import (
    TraceAnalysis,
    analyze_trace,
)
from vericcl.verification.online.trace_format import parse_trace
from vericcl.xml.trace_sidecar import TraceSidecar


_monotonic = time.monotonic


def _command(value: object) -> Tuple[str, ...]:
    try:
        command = tuple(value)
    except TypeError as error:
        raise SemanticError("process command must be iterable") from error
    if not command or not all(
        isinstance(item, str) and item and "\x00" not in item
        for item in command
    ):
        raise SemanticError("process command contains an invalid argument")
    return command


def _environment(value: object) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise SemanticError("process environment must be a mapping")
    normalized = dict(value)
    if not all(
        isinstance(key, str)
        and key
        and isinstance(item, str)
        and "\x00" not in key
        and "\x00" not in item
        for key, item in normalized.items()
    ):
        raise SemanticError("process environment contains an invalid entry")
    return MappingProxyType(normalized)


@dataclass(frozen=True)
class ProcessRequest:
    command: Tuple[str, ...]
    environment: Mapping[str, str]
    label: str
    cwd: Optional[Path] = None
    timeout_s: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "command", _command(self.command))
        object.__setattr__(
            self,
            "environment",
            _environment(self.environment),
        )
        if not isinstance(self.label, str) or not self.label:
            raise SemanticError("process label must be a non-empty string")
        if self.cwd is not None:
            object.__setattr__(self, "cwd", Path(self.cwd))
        if self.timeout_s is not None:
            if (
                isinstance(self.timeout_s, bool)
                or not isinstance(self.timeout_s, (int, float))
                or not math.isfinite(float(self.timeout_s))
                or self.timeout_s <= 0
            ):
                raise SemanticError("process timeout must be positive")
            object.__setattr__(self, "timeout_s", float(self.timeout_s))


@dataclass(frozen=True)
class ProcessResult:
    returncode: int
    stdout: str
    stderr: str

    def __post_init__(self) -> None:
        if isinstance(self.returncode, bool) or not isinstance(
            self.returncode,
            int,
        ):
            raise SemanticError("process return code must be an integer")
        if not isinstance(self.stdout, str) or not isinstance(
            self.stderr,
            str,
        ):
            raise SemanticError("process output must be text")


class CommandExecutor(Protocol):
    def run(self, request: ProcessRequest) -> ProcessResult:
        ...


class SubprocessCommandExecutor:
    def run(self, request: ProcessRequest) -> ProcessResult:
        if not isinstance(request, ProcessRequest):
            raise SemanticError("executor request must be a ProcessRequest")
        try:
            completed = subprocess.run(
                request.command,
                cwd=request.cwd,
                env=dict(request.environment),
                capture_output=True,
                text=True,
                timeout=request.timeout_s,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise SemanticError(
                "{} process could not be executed: {}".format(
                    request.label,
                    error,
                )
            ) from error
        return ProcessResult(
            completed.returncode,
            completed.stdout,
            completed.stderr,
        )


def _run_checked(
    executor: CommandExecutor,
    request: ProcessRequest,
) -> ProcessResult:
    try:
        result = executor.run(request)
    except AttributeError as error:
        raise SemanticError("executor must provide run(request)") from error
    if not isinstance(result, ProcessResult):
        raise SemanticError("executor returned an invalid process result")
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no output"
        raise SemanticError(
            "{} process failed with status {}: {}".format(
                request.label,
                result.returncode,
                detail,
            )
        )
    return result


class NcclTestsRunner:
    def __init__(
        self,
        executor: CommandExecutor,
        *,
        environment: Mapping[str, str],
        launcher_prefix: Tuple[str, ...] = (),
        cwd: Optional[Path] = None,
        timeout_s: Optional[float] = None,
    ) -> None:
        if not callable(getattr(executor, "run", None)):
            raise SemanticError("executor must provide run(request)")
        self._executor = executor
        self._environment = _environment(environment)
        self._launcher_prefix = _command(launcher_prefix) if launcher_prefix else ()
        self._cwd = None if cwd is None else Path(cwd)
        if timeout_s is not None and (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or not math.isfinite(float(timeout_s))
            or timeout_s <= 0.0
        ):
            raise SemanticError("runner wall-clock budget must be positive")
        self._deadline = (
            None
            if timeout_s is None
            else _monotonic() + float(timeout_s)
        )
        self._help = NcclTestsHelpValidator()

    def _remaining_timeout(self) -> Optional[float]:
        if self._deadline is None:
            return None
        remaining = self._deadline - _monotonic()
        if remaining <= 0.0:
            raise SemanticError("runner wall-clock budget expired")
        return remaining

    def _request(
        self,
        command: Tuple[str, ...],
        label: str,
        *,
        environment: Optional[Mapping[str, str]] = None,
        launcher: bool = True,
    ) -> ProcessRequest:
        return ProcessRequest(
            command=(self._launcher_prefix if launcher else ()) + command,
            environment=(
                self._environment if environment is None else environment
            ),
            label=label,
            cwd=self._cwd,
            timeout_s=self._remaining_timeout(),
        )

    def validate_help(self, request: NcclTestRequest) -> None:
        def load_help(command: Tuple[str, ...]) -> str:
            process = self._request(
                command,
                "nccl-tests help",
                launcher=False,
            )
            return _run_checked(self._executor, process).stdout

        self._help.validate(request, load_help)

    def measure(self, request: NcclTestRequest) -> PerformanceHistory:
        self.validate_help(request)
        command = build_nccl_tests_command(request)
        history = PerformanceHistory()
        while (
            not history.rounds
            or (not history.stable and history.retry_required)
        ):
            samples = []
            for index in range(MEASUREMENT_SAMPLE_COUNT):
                process = self._request(
                    command,
                    "nccl-tests release sample {}".format(index),
                )
                output = _run_checked(self._executor, process).stdout
                rows = parse_nccl_tests_output(
                    output,
                    request.message_size_bytes,
                )
                if len(rows) != 1:
                    raise SemanticError(
                        "each nccl-tests process must produce one performance row"
                    )
                samples.append(rows[0].selected_time_us(inplace=request.inplace))
            history = history.add_round(samples)
        return history

    def validate_release(self, request: NcclTestRequest) -> None:
        self.validate_help(request)
        process = self._request(
            build_nccl_tests_command(request),
            "nccl-tests release validation",
        )
        output = _run_checked(self._executor, process).stdout
        rows = parse_nccl_tests_output(
            output,
            request.message_size_bytes,
        )
        if len(rows) != 1:
            raise SemanticError(
                "release validation must produce one performance row"
            )

    def diagnostic(
        self,
        request: NcclTestRequest,
        environment: Mapping[str, str],
    ) -> ProcessResult:
        process = self._request(
            build_nccl_tests_trace_command(request),
            "nccl-tests trace diagnostic",
            environment=environment,
        )
        return _run_checked(self._executor, process)

    def run_auxiliary(
        self,
        command: Tuple[str, ...],
        label: str,
        environment: Mapping[str, str],
    ) -> ProcessResult:
        process = self._request(
            command,
            label,
            environment=environment,
        )
        return _run_checked(self._executor, process)


@dataclass(frozen=True)
class TraceCollectionRequest:
    sidecar: TraceSidecar
    file_prefix: Path
    rank_count: int
    clock_sync_output: str
    max_clock_uncertainty_us: float
    measured_iterations: Optional[int] = None
    inplace: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.sidecar, TraceSidecar):
            raise SemanticError("trace collection sidecar is invalid")
        object.__setattr__(self, "file_prefix", Path(self.file_prefix))
        if (
            isinstance(self.rank_count, bool)
            or not isinstance(self.rank_count, int)
            or self.rank_count < 1
        ):
            raise SemanticError("trace collection rank count is invalid")
        if self.rank_count != self.sidecar.rank_count:
            raise SemanticError("trace collection rank count differs from sidecar")
        if not isinstance(self.clock_sync_output, str) or not self.clock_sync_output:
            raise SemanticError("trace collection clock output is empty")
        if (
            isinstance(self.max_clock_uncertainty_us, bool)
            or not isinstance(self.max_clock_uncertainty_us, (int, float))
            or not math.isfinite(float(self.max_clock_uncertainty_us))
            or self.max_clock_uncertainty_us < 0
        ):
            raise SemanticError("maximum clock uncertainty is invalid")
        object.__setattr__(
            self,
            "max_clock_uncertainty_us",
            float(self.max_clock_uncertainty_us),
        )
        if self.measured_iterations is not None and (
            isinstance(self.measured_iterations, bool)
            or not isinstance(self.measured_iterations, int)
            or self.measured_iterations < 1
        ):
            raise SemanticError("measured trace iterations must be positive")
        if not isinstance(self.inplace, bool):
            raise SemanticError("trace inplace selector must be boolean")


@dataclass(frozen=True)
class TraceCollectionResult:
    analysis: TraceAnalysis
    rank_files: Tuple[Path, ...]
    complete: bool
    clock_uncertainty_us: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.analysis, TraceAnalysis):
            raise SemanticError("trace collection analysis is invalid")
        files = tuple(Path(value) for value in self.rank_files)
        if not files:
            raise SemanticError("trace collection rank files are empty")
        object.__setattr__(self, "rank_files", files)
        if not isinstance(self.complete, bool):
            raise SemanticError("trace collection completeness must be boolean")
        if (
            isinstance(self.clock_uncertainty_us, bool)
            or not isinstance(self.clock_uncertainty_us, (int, float))
            or not math.isfinite(float(self.clock_uncertainty_us))
            or self.clock_uncertainty_us < 0
        ):
            raise SemanticError("trace clock uncertainty is invalid")
        object.__setattr__(
            self,
            "clock_uncertainty_us",
            float(self.clock_uncertainty_us),
        )


def collect_trace_files(
    request: TraceCollectionRequest,
) -> TraceCollectionResult:
    if not isinstance(request, TraceCollectionRequest):
        raise SemanticError("trace request is invalid")
    paths = tuple(
        Path("{}.rank-{}.bin".format(request.file_prefix, rank))
        for rank in range(request.rank_count)
    )
    missing = tuple(str(path) for path in paths if not path.is_file())
    if missing:
        raise SemanticError(
            "trace files are missing: {}".format(", ".join(missing))
        )
    records_by_rank = {
        rank: parse_trace(paths[rank], request.sidecar)
        for rank in range(request.rank_count)
    }
    if any(not records for records in records_by_rank.values()):
        raise SemanticError("trace rank contains no records")
    if request.measured_iterations is not None:
        invocation_sets = tuple(
            frozenset(record.iteration for record in records_by_rank[rank])
            for rank in range(request.rank_count)
        )
        if any(values != invocation_sets[0] for values in invocation_sets[1:]):
            raise SemanticError("trace invocation identifiers differ by rank")
        invocation_ids = tuple(sorted(invocation_sets[0]))
        block_size = request.measured_iterations + 1
        expected_counts = (block_size, 2 * block_size)
        if len(invocation_ids) not in expected_counts:
            raise SemanticError(
                "trace invocation count must be {} or {}, got {}".format(
                    expected_counts[0],
                    expected_counts[1],
                    len(invocation_ids),
                )
            )
        block_offset = (
            block_size
            if request.inplace and len(invocation_ids) == 2 * block_size
            else 0
        )
        selected = frozenset(
            invocation_ids[
                block_offset + 1:
                block_offset + block_size
            ]
        )
        records_by_rank = {
            rank: tuple(
                record
                for record in records_by_rank[rank]
                if record.iteration in selected
            )
            for rank in range(request.rank_count)
        }
    samples = parse_clock_sync_output(request.clock_sync_output)
    alignment = align_clocks(records_by_rank, samples)
    uncertainty = max(
        transform.uncertainty_us
        for transform in alignment.transforms.values()
    )
    if uncertainty > request.max_clock_uncertainty_us:
        raise SemanticError(
            "clock uncertainty exceeds the configured maximum"
        )
    records = tuple(
        record
        for rank in range(request.rank_count)
        for record in records_by_rank[rank]
    )
    analysis = analyze_trace(records, request.sidecar, alignment)
    return TraceCollectionResult(analysis, paths, True, uncertainty)


def process_environment(
    additions: Mapping[str, str],
) -> Mapping[str, str]:
    normalized = dict(os.environ)
    normalized.update(_environment(additions))
    return MappingProxyType(normalized)
