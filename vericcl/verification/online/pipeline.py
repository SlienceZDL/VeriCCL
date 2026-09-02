from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
import os
from pathlib import Path
import shutil
import time
from types import MappingProxyType
from typing import Callable, Mapping, Optional, Tuple

from lxml import etree

from vericcl.constants import SOFTWARE_MAX_CONCURRENCY
from vericcl.errors import SemanticError
from vericcl.input.models import ResolvedInput
from vericcl.semantics.atom import Schedule
from vericcl.topology.model import PerformanceCurve
from vericcl.verification.online.cache import (
    CalibrationCache,
    EnvironmentSignature,
)
from vericcl.verification.online.calibration import (
    CalibrationPoint,
    CalibrationRequest,
    derive_calibrated_curve,
)
from vericcl.verification.online.model import NcclTestRequest
from vericcl.verification.online.nccl_tests import (
    build_nccl_tests_command,
    parse_nccl_tests_output,
)
from vericcl.verification.online.runner import (
    CommandExecutor,
    NcclTestsRunner,
    TraceCollectionRequest,
    TraceCollectionResult,
    collect_trace_files,
    process_environment,
)
from vericcl.verification.online.statistics import (
    MEASUREMENT_SAMPLE_COUNT,
    PerformanceHistory,
)
from vericcl.verification.online.trace_analysis import (
    BottleneckRecord,
    TraceAnalysis,
)
from vericcl.xml.lower import XmlArtifact
from vericcl.xml.trace_sidecar import TraceSidecar, build_trace_sidecar


EXPECTED_MSCCL_CHUNK_STEPS = 4
EXPECTED_MSCCL_SLICE_STEPS = 4
_MPI_EXPORTED_NAMES = frozenset(
    {"PATH", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES"}
)
_MPI_EXPORTED_PREFIXES = ("NCCL_", "MSCCL_", "VERICCL_")
_monotonic = time.monotonic


class OnlineStageStatus(str, Enum):
    NOT_RUN = "not_run"
    PASSED = "passed"
    FAILED = "failed"
    UNSTABLE = "unstable"
    REQUIRES_RESOLVE = "requires_resolve"


class _PreflightFailure(Exception):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


MeasureCalibrationPoint = Callable[[EnvironmentSignature], CalibrationPoint]
TraceCollector = Callable[[TraceCollectionRequest], TraceCollectionResult]


def _tuple(value: object, field: str) -> tuple:
    try:
        return tuple(value)
    except TypeError as error:
        raise SemanticError("{} must be iterable".format(field)) from error


def _path(value: object, field: str) -> Path:
    try:
        return Path(value)
    except TypeError as error:
        raise SemanticError("{} is invalid".format(field)) from error


def _positive_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SemanticError("{} must be a positive integer".format(field))
    return value


def _string_mapping(value: object, field: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise SemanticError("{} must be a mapping".format(field))
    normalized = dict(value)
    if not all(
        isinstance(key, str)
        and key
        and isinstance(item, str)
        and "\x00" not in key
        and "\x00" not in item
        for key, item in normalized.items()
    ):
        raise SemanticError("{} contains an invalid entry".format(field))
    return MappingProxyType(normalized)


@dataclass(frozen=True)
class CalibrationPlan:
    request: CalibrationRequest
    alpha_us: float
    signatures: Tuple[EnvironmentSignature, ...]
    cache: CalibrationCache
    measure_point: MeasureCalibrationPoint
    force_recalibrate: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.request, CalibrationRequest):
            raise SemanticError("calibration plan request is invalid")
        if (
            isinstance(self.alpha_us, bool)
            or not isinstance(self.alpha_us, (int, float))
            or not math.isfinite(float(self.alpha_us))
            or self.alpha_us < 0
        ):
            raise SemanticError("calibration plan alpha_us is invalid")
        object.__setattr__(self, "alpha_us", float(self.alpha_us))
        signatures = _tuple(self.signatures, "calibration plan signatures")
        if not all(
            isinstance(value, EnvironmentSignature) for value in signatures
        ):
            raise SemanticError("calibration plan signatures are invalid")
        expected = tuple(range(1, len(signatures) + 1))
        actual = tuple(value.concurrency for value in signatures)
        if actual != expected:
            raise SemanticError(
                "calibration signatures must be contiguous from one"
            )
        benchmark_slices = self.request.benchmark_slice_count
        expected_count = (
            0
            if benchmark_slices is None
            else min(
                self.request.max_calibration_channels,
                SOFTWARE_MAX_CONCURRENCY,
                benchmark_slices,
            )
        )
        if len(signatures) != expected_count:
            raise SemanticError(
                "calibration signatures do not cover every concurrency"
            )
        for signature in signatures:
            if (
                signature.link_class != self.request.link_class
                or signature.slice_size_bytes
                != self.request.slice_size_bytes
                or signature.benchmark_size_bytes
                != self.request.benchmark_size_bytes
                or signature.protocol != "Simple"
                or signature.nccl_buffsize_bytes
                != 2 * self.request.slice_size_bytes
                or signature.chunk_steps != EXPECTED_MSCCL_CHUNK_STEPS
                or signature.slice_steps != EXPECTED_MSCCL_SLICE_STEPS
            ):
                raise SemanticError(
                    "calibration signature differs from its request"
                )
        object.__setattr__(self, "signatures", signatures)
        if not isinstance(self.cache, CalibrationCache):
            raise SemanticError("calibration plan cache is invalid")
        if not callable(self.measure_point):
            raise SemanticError("calibration measure_point must be callable")
        if not isinstance(self.force_recalibrate, bool):
            raise SemanticError("force_recalibrate must be a boolean")


@dataclass(frozen=True)
class OnlineCalibrationOutcome:
    request: CalibrationRequest
    points: Tuple[CalibrationPoint, ...]
    curve: Optional[PerformanceCurve]
    cache_hit_concurrencies: Tuple[int, ...]
    stable: bool
    skipped_reason: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.request, CalibrationRequest):
            raise SemanticError("online calibration request is invalid")
        points = tuple(self.points)
        if not all(isinstance(point, CalibrationPoint) for point in points):
            raise SemanticError("online calibration points are invalid")
        object.__setattr__(self, "points", points)
        hits = tuple(self.cache_hit_concurrencies)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in hits
        ):
            raise SemanticError("calibration cache hits are invalid")
        object.__setattr__(self, "cache_hit_concurrencies", hits)
        if not isinstance(self.stable, bool):
            raise SemanticError("online calibration stability is invalid")
        if self.curve is not None and not isinstance(
            self.curve,
            PerformanceCurve,
        ):
            raise SemanticError("online calibration curve is invalid")
        if self.stable != (bool(points) and all(point.stable for point in points)):
            raise SemanticError("online calibration stability is inconsistent")
        if self.stable and self.curve is None:
            raise SemanticError("stable calibration requires a curve")
        if not self.stable and self.curve is not None:
            raise SemanticError("unstable calibration must not contain a curve")
        if self.skipped_reason is not None:
            if not isinstance(self.skipped_reason, str) or not self.skipped_reason:
                raise SemanticError("calibration skipped reason is invalid")
            if points or self.curve is not None or self.stable:
                raise SemanticError("skipped calibration contains results")


@dataclass(frozen=True)
class OnlineTuningEvidence:
    wait_us_by_transfer: Mapping[str, float]
    bottleneck_priorities: Tuple[BottleneckRecord, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.wait_us_by_transfer, Mapping):
            raise SemanticError("online wait evidence must be a mapping")
        waits = dict(self.wait_us_by_transfer)
        if not all(
            isinstance(key, str)
            and key
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and value >= 0
            for key, value in waits.items()
        ):
            raise SemanticError("online wait evidence is invalid")
        object.__setattr__(
            self,
            "wait_us_by_transfer",
            MappingProxyType(dict(sorted(waits.items()))),
        )
        priorities = tuple(self.bottleneck_priorities)
        if not all(isinstance(item, BottleneckRecord) for item in priorities):
            raise SemanticError("online bottleneck priorities are invalid")
        object.__setattr__(self, "bottleneck_priorities", priorities)


@dataclass(frozen=True)
class OnlineContext:
    artifact: XmlArtifact
    schedule: Schedule
    inputs: ResolvedInput
    request: NcclTestRequest
    xml_paths: Tuple[Path, ...]
    msccl_library_path: Path
    executor: CommandExecutor
    environment: Mapping[str, str]
    inter_node: bool
    mpi_launcher: Optional[Path]
    mpi_hostfile: Optional[Path]
    trace_file_prefix: Optional[Path]
    clock_sync_binary: Optional[Path]
    max_clock_uncertainty_us: float
    trace_collector: TraceCollector = collect_trace_files
    calibration_plan: Optional[CalibrationPlan] = None
    online_tuning_requested: bool = False
    single_process_release_validation: bool = False
    chunk_steps: int = EXPECTED_MSCCL_CHUNK_STEPS
    slice_steps: int = EXPECTED_MSCCL_SLICE_STEPS
    trace_record_capacity: int = 1048576
    clock_sync_samples: int = 16
    timeout_s: Optional[float] = 3600.0
    cwd: Optional[Path] = None

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, XmlArtifact):
            raise SemanticError("online context artifact is invalid")
        if not isinstance(self.schedule, Schedule):
            raise SemanticError("online context schedule is invalid")
        if not isinstance(self.inputs, ResolvedInput):
            raise SemanticError("online context inputs are invalid")
        if not isinstance(self.request, NcclTestRequest):
            raise SemanticError("online context request is invalid")
        paths = tuple(
            _path(value, "online context XML path")
            for value in _tuple(self.xml_paths, "online context XML paths")
        )
        object.__setattr__(self, "xml_paths", paths)
        object.__setattr__(
            self,
            "msccl_library_path",
            _path(self.msccl_library_path, "MSCCL library path"),
        )
        if not callable(getattr(self.executor, "run", None)):
            raise SemanticError("online context executor must provide run(request)")
        object.__setattr__(
            self,
            "environment",
            _string_mapping(self.environment, "online context environment"),
        )
        if not isinstance(self.inter_node, bool):
            raise SemanticError("online context inter_node must be boolean")
        for field in (
            "mpi_launcher",
            "mpi_hostfile",
            "trace_file_prefix",
            "clock_sync_binary",
            "cwd",
        ):
            value = getattr(self, field)
            if value is not None:
                object.__setattr__(self, field, _path(value, field))
        if not callable(self.trace_collector):
            raise SemanticError("online context trace_collector must be callable")
        if self.calibration_plan is not None and not isinstance(
            self.calibration_plan,
            CalibrationPlan,
        ):
            raise SemanticError("online context calibration plan is invalid")
        if not isinstance(self.online_tuning_requested, bool):
            raise SemanticError("online_tuning_requested must be boolean")
        if not isinstance(self.single_process_release_validation, bool):
            raise SemanticError(
                "single_process_release_validation must be boolean"
            )
        if self.chunk_steps != EXPECTED_MSCCL_CHUNK_STEPS:
            raise SemanticError("chunk_steps must equal four")
        if self.slice_steps != EXPECTED_MSCCL_SLICE_STEPS:
            raise SemanticError("slice_steps must equal four")
        _positive_integer(
            self.trace_record_capacity,
            "trace_record_capacity",
        )
        samples = _positive_integer(
            self.clock_sync_samples,
            "clock_sync_samples",
        )
        if samples < 2:
            raise SemanticError("clock_sync_samples must be at least two")
        if (
            isinstance(self.max_clock_uncertainty_us, bool)
            or not isinstance(self.max_clock_uncertainty_us, (int, float))
            or not math.isfinite(float(self.max_clock_uncertainty_us))
            or self.max_clock_uncertainty_us < 0
        ):
            raise SemanticError("max_clock_uncertainty_us is invalid")
        object.__setattr__(
            self,
            "max_clock_uncertainty_us",
            float(self.max_clock_uncertainty_us),
        )
        if self.timeout_s is not None:
            if (
                isinstance(self.timeout_s, bool)
                or not isinstance(self.timeout_s, (int, float))
                or not math.isfinite(float(self.timeout_s))
                or self.timeout_s <= 0
            ):
                raise SemanticError("online context timeout must be positive")
            object.__setattr__(self, "timeout_s", float(self.timeout_s))


@dataclass(frozen=True)
class OnlineValidationResult:
    context_schedule: Schedule
    preflight_status: OnlineStageStatus
    calibration_status: OnlineStageStatus
    release_status: OnlineStageStatus
    online_operator_validation: OnlineStageStatus
    failure_code: Optional[str]
    failure_message: Optional[str]
    runtime_environment: Mapping[str, str]
    release_history: Optional[PerformanceHistory]
    calibration: Optional[OnlineCalibrationOutcome]
    trace_analysis: Optional[TraceAnalysis]
    trace_rank_files: Tuple[Path, ...]
    trace_clock_uncertainty_us: Optional[float]
    requires_resolve: bool
    online_tuning_allowed: bool
    tuning_evidence: Optional[OnlineTuningEvidence]
    single_process_release_validation: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.context_schedule, Schedule):
            raise SemanticError("online result schedule is invalid")
        for field in (
            "preflight_status",
            "calibration_status",
            "release_status",
            "online_operator_validation",
        ):
            if not isinstance(getattr(self, field), OnlineStageStatus):
                raise SemanticError("online result status is invalid")
        for value, field in (
            (self.failure_code, "failure_code"),
            (self.failure_message, "failure_message"),
        ):
            if value is not None and (not isinstance(value, str) or not value):
                raise SemanticError("online result {} is invalid".format(field))
        object.__setattr__(
            self,
            "runtime_environment",
            _string_mapping(
                self.runtime_environment,
                "online result environment",
            ),
        )
        if self.release_history is not None and not isinstance(
            self.release_history,
            PerformanceHistory,
        ):
            raise SemanticError("online result release history is invalid")
        if self.calibration is not None and not isinstance(
            self.calibration,
            OnlineCalibrationOutcome,
        ):
            raise SemanticError("online result calibration is invalid")
        if self.trace_analysis is not None and not isinstance(
            self.trace_analysis,
            TraceAnalysis,
        ):
            raise SemanticError("online result trace analysis is invalid")
        object.__setattr__(
            self,
            "trace_rank_files",
            tuple(Path(value) for value in self.trace_rank_files),
        )
        if self.trace_clock_uncertainty_us is not None:
            if (
                isinstance(self.trace_clock_uncertainty_us, bool)
                or not isinstance(self.trace_clock_uncertainty_us, (int, float))
                or not math.isfinite(float(self.trace_clock_uncertainty_us))
                or self.trace_clock_uncertainty_us < 0
            ):
                raise SemanticError("online result clock uncertainty is invalid")
            object.__setattr__(
                self,
                "trace_clock_uncertainty_us",
                float(self.trace_clock_uncertainty_us),
            )
        if not isinstance(self.requires_resolve, bool) or not isinstance(
            self.online_tuning_allowed,
            bool,
        ):
            raise SemanticError("online result boolean field is invalid")
        if not isinstance(self.single_process_release_validation, bool):
            raise SemanticError("online result release mode is invalid")
        if self.tuning_evidence is not None and not isinstance(
            self.tuning_evidence,
            OnlineTuningEvidence,
        ):
            raise SemanticError("online result tuning evidence is invalid")
        if self.online_tuning_allowed != (self.tuning_evidence is not None):
            raise SemanticError("online tuning evidence is inconsistent")


def _result(
    context: OnlineContext,
    **changes,
) -> OnlineValidationResult:
    values = {
        "context_schedule": context.schedule,
        "preflight_status": OnlineStageStatus.NOT_RUN,
        "calibration_status": OnlineStageStatus.NOT_RUN,
        "release_status": OnlineStageStatus.NOT_RUN,
        "online_operator_validation": OnlineStageStatus.NOT_RUN,
        "failure_code": None,
        "failure_message": None,
        "runtime_environment": {},
        "release_history": None,
        "calibration": None,
        "trace_analysis": None,
        "trace_rank_files": (),
        "trace_clock_uncertainty_us": None,
        "requires_resolve": False,
        "online_tuning_allowed": False,
        "tuning_evidence": None,
        "single_process_release_validation": (
            context.single_process_release_validation
        ),
    }
    values.update(changes)
    return OnlineValidationResult(**values)


_DATATYPE_TO_NCCL_TESTS = {
    "float16": "half",
    "float32": "float",
    "float64": "double",
    "int8": "int8",
    "uint8": "uint8",
    "int32": "int32",
    "uint32": "uint32",
    "int64": "int64",
    "uint64": "uint64",
    "bfloat16": "bfloat16",
}


def _fail(code: str, message: str) -> None:
    raise _PreflightFailure(code, message)


def _executable(path: Path, code: str, description: str) -> None:
    if not path.is_file() or not os.access(path, os.X_OK):
        _fail(code, "{} is missing or not executable".format(description))


def _validate_request(context: OnlineContext, root) -> None:
    request = context.request
    spec = context.inputs.collective
    if request.kind is not spec.kind:
        _fail("collective_kind_mismatch", "request collective kind differs")
    try:
        expected_size = int(root.attrib["minBytes"])
    except (KeyError, TypeError, ValueError):
        _fail("xml_size_range_invalid", "XML minBytes is invalid")
    if request.message_size_bytes != expected_size:
        _fail("message_size_mismatch", "request message size differs from XML")
    expected_datatype = _DATATYPE_TO_NCCL_TESTS.get(
        spec.datatype,
        spec.datatype,
    )
    if request.datatype != expected_datatype:
        _fail("datatype_mismatch", "request datatype differs from input")
    if request.reduction_op != spec.reduction_op:
        _fail("reduction_op_mismatch", "request reduction operation differs")
    if request.root != spec.root:
        _fail("root_mismatch", "request root differs from input")
    if request.inplace != spec.inplace:
        _fail("inplace_mismatch", "request inplace mode differs from input")
    if root.attrib.get("coll") != spec.kind.value:
        _fail("xml_collective_mismatch", "XML collective kind differs")
    if root.attrib.get("inplace") != ("1" if spec.inplace else "0"):
        _fail("xml_inplace_mismatch", "XML inplace mode differs")
    if root.attrib.get("proto") != "Simple":
        _fail("xml_protocol_mismatch", "XML protocol must be Simple")
    if root.attrib.get("ngpus") != str(context.inputs.rank_count):
        _fail("xml_rank_count_mismatch", "XML rank count differs")
    if root.attrib.get("maxBytes") != str(expected_size + 1):
        _fail("xml_size_range_mismatch", "XML size range is not exact")


def _runtime_values(
    context: OnlineContext,
    xml_path: Path,
    sidecar: TraceSidecar,
) -> Mapping[str, str]:
    prefix = str(context.trace_file_prefix)
    entries_per_rank = max(
        sum(1 for entry in sidecar.entries.values() if entry.rank == rank)
        for rank in range(context.inputs.rank_count)
    )
    trace_capacity = max(
        context.trace_record_capacity,
        2 * (MEASUREMENT_SAMPLE_COUNT + 1) * entries_per_rank,
    )
    expected = {
        "NCCL_ALGO": "MSCCL,RING",
        "NCCL_BUFFSIZE": str(2 * context.schedule.slice_size_bytes),
        "NCCL_PROTO": "Simple",
        "MSCCL_XML_FILES": str(xml_path),
        "VERICCL_EXPECTED_MSCCL_CHUNKSTEPS": str(context.chunk_steps),
        "VERICCL_EXPECTED_MSCCL_SLICESTEPS": str(context.slice_steps),
        "VERICCL_TRACE_RECORDS": str(trace_capacity),
        "VERICCL_TRACE_FILE_PREFIX": prefix,
    }
    for key, value in expected.items():
        if (
            key != "VERICCL_TRACE_RECORDS"
            and key in context.environment
            and context.environment[key] != value
        ):
            _fail(
                "runtime_environment_conflict",
                "runtime environment conflicts with {}".format(key),
            )
    if (
        "VERICCL_TRACE_ENABLE" in context.environment
        and context.environment["VERICCL_TRACE_ENABLE"] not in {"0", "1"}
    ):
        _fail(
            "runtime_environment_conflict",
            "runtime trace enable value is invalid",
        )
    values = dict(context.environment)
    values.update(expected)
    library = str(context.msccl_library_path)
    inherited = values.get("LD_LIBRARY_PATH", os.environ.get("LD_LIBRARY_PATH", ""))
    values["LD_LIBRARY_PATH"] = (
        library if not inherited else library + os.pathsep + inherited
    )
    return MappingProxyType(values)


def _launcher_prefix(
    context: OnlineContext,
    exported_keys: Tuple[str, ...],
) -> Tuple[str, ...]:
    if context.mpi_launcher is None:
        return ()
    command = [
        str(context.mpi_launcher),
        "-np",
        str(context.inputs.rank_count),
    ]
    if context.inter_node and context.inputs.rank_count == 2:
        command.extend(("-N", "1"))
    if context.mpi_hostfile is not None:
        command.extend(("--hostfile", str(context.mpi_hostfile)))
    if context.inter_node:
        command.extend(
            (
                "-mca",
                "pml",
                "ob1",
                "-mca",
                "btl",
                "tcp,self,vader",
                "-mca",
                "btl_vader_single_copy_mechanism",
                "none",
            )
        )
        tcp_interface = context.environment.get(
            "VERICCL_MPI_TCP_IF_INCLUDE"
        )
        if tcp_interface:
            command.extend(
                ("-mca", "btl_tcp_if_include", tcp_interface)
            )
    for key in exported_keys:
        command.extend(("-x", key))
    return tuple(command)


@dataclass(frozen=True)
class _Prepared:
    release_environment: Mapping[str, str]
    trace_environment: Mapping[str, str]
    launcher_prefix: Tuple[str, ...]
    sidecar: TraceSidecar


def _preflight(context: OnlineContext) -> _Prepared:
    if not context.artifact.runtime_compatible:
        _fail("runtime_incompatible", "XML is not MSCCL runtime compatible")
    if (
        context.schedule.rank_count != context.inputs.rank_count
        or context.schedule.slice_count
        != context.inputs.hyperparameters.slice_count
        or context.schedule.slice_size_bytes
        != context.inputs.hyperparameters.slice_size_bytes
    ):
        _fail("schedule_input_mismatch", "schedule and input geometry differ")
    if len(context.xml_paths) != 1:
        _fail("xml_path_count_invalid", "exactly one XML path is required")
    xml_path = context.xml_paths[0]
    if not xml_path.is_file():
        _fail("xml_path_missing", "XML path is missing")
    try:
        xml_text = xml_path.read_text(encoding="utf-8")
    except OSError as error:
        _fail("xml_path_unreadable", "XML path cannot be read: {}".format(error))
    if xml_text != context.artifact.xml_text:
        _fail("xml_artifact_mismatch", "XML file differs from its artifact")
    try:
        root = etree.fromstring(xml_text.encode("utf-8"))
    except (etree.XMLSyntaxError, ValueError) as error:
        _fail("xml_parse_failed", "XML cannot be parsed: {}".format(error))
    _validate_request(context, root)
    if not context.msccl_library_path.is_dir():
        _fail("msccl_library_missing", "MSCCL library path is missing")
    binary_value = build_nccl_tests_command(context.request)[0]
    binary = Path(binary_value)
    if binary.parent == Path("."):
        located = shutil.which(binary_value)
        if located is not None:
            binary = Path(located)
    _executable(binary, "nccl_tests_binary_missing", "nccl-tests binary")
    if context.mpi_launcher is None:
        _fail(
            "mpi_launcher_missing",
            "online validation requires one MPI process per GPU",
        )
    if context.mpi_launcher is not None:
        _executable(context.mpi_launcher, "mpi_launcher_missing", "MPI launcher")
    if context.mpi_hostfile is not None and not context.mpi_hostfile.is_file():
        _fail("mpi_hostfile_missing", "MPI hostfile is missing")
    if context.trace_file_prefix is None:
        _fail("trace_prefix_missing", "trace file prefix is required")
    if context.clock_sync_binary is None:
        _fail("clock_sync_binary_missing", "clock sync binary is required")
    _executable(
        context.clock_sync_binary,
        "clock_sync_binary_missing",
        "clock sync binary",
    )
    try:
        sidecar = build_trace_sidecar(context.artifact, context.schedule)
    except SemanticError as error:
        _fail(
            "trace_sidecar_failed",
            "trace sidecar could not be built: {}".format(error),
        )
    additions = _runtime_values(context, xml_path, sidecar)
    release_additions = dict(additions)
    release_additions["VERICCL_TRACE_ENABLE"] = "0"
    trace_additions = dict(additions)
    trace_additions["VERICCL_TRACE_ENABLE"] = "1"
    exported_keys = tuple(
        sorted(
            key
            for key in trace_additions
            if key in _MPI_EXPORTED_NAMES
            or key.startswith(_MPI_EXPORTED_PREFIXES)
        )
    )
    return _Prepared(
        release_environment=process_environment(release_additions),
        trace_environment=process_environment(trace_additions),
        launcher_prefix=_launcher_prefix(context, exported_keys),
        sidecar=sidecar,
    )


def _calibrate(plan: CalibrationPlan) -> OnlineCalibrationOutcome:
    if not plan.signatures:
        return OnlineCalibrationOutcome(
            request=plan.request,
            points=(),
            curve=None,
            cache_hit_concurrencies=(),
            stable=False,
            skipped_reason="slice_size_does_not_divide_128_mib",
        )
    points = []
    hits = []
    for signature in plan.signatures:
        point = plan.cache.get(
            signature,
            force_recalibrate=plan.force_recalibrate,
        )
        if point is None:
            point = plan.measure_point(signature)
            if not isinstance(point, CalibrationPoint):
                raise SemanticError("calibration callback returned an invalid point")
            if point.concurrency != signature.concurrency:
                raise SemanticError("calibration point concurrency differs")
        else:
            hits.append(signature.concurrency)
        benchmark_slices = plan.request.benchmark_slice_count
        assert benchmark_slices is not None
        if (
            point.full_wave_count
            != benchmark_slices // signature.concurrency
            or point.tail_transfer_count
            != benchmark_slices % signature.concurrency
        ):
            raise SemanticError(
                "calibration point wave geometry differs from the request"
            )
        if signature.concurrency not in hits:
            plan.cache.put(signature, point)
        points.append(point)
    normalized = tuple(points)
    stable = all(point.stable for point in normalized)
    curve = (
        derive_calibrated_curve(
            plan.alpha_us,
            plan.request.slice_size_bytes,
            normalized,
        )
        if stable
        else None
    )
    return OnlineCalibrationOutcome(
        request=plan.request,
        points=normalized,
        curve=curve,
        cache_hit_concurrencies=tuple(hits),
        stable=stable,
    )


def _tuning_evidence(analysis: TraceAnalysis) -> OnlineTuningEvidence:
    waits = {}
    for step in analysis.step_waits:
        value = (
            step.waits.head_of_line_wait_us
            + step.waits.dependency_wait_us
            + step.waits.peer_resource_wait_us
        )
        waits[step.transfer_id] = waits.get(step.transfer_id, 0.0) + value
    priorities = tuple(
        sorted(
            analysis.bottlenecks,
            key=lambda item: (
                -item.duration_us,
                item.transfer_id,
                item.rank,
                item.tb_id,
                item.step_index,
                item.wait_class.value,
            ),
        )
    )
    return OnlineTuningEvidence(waits, priorities)


def run_online_validation(context: OnlineContext) -> OnlineValidationResult:
    if not isinstance(context, OnlineContext):
        raise SemanticError("online validation requires an OnlineContext")
    online_started = _monotonic()
    try:
        prepared = _preflight(context)
    except _PreflightFailure as error:
        return _result(
            context,
            preflight_status=OnlineStageStatus.FAILED,
            failure_code=error.code,
            failure_message=error.message,
        )

    calibration = None
    calibration_status = OnlineStageStatus.NOT_RUN
    calibration_blocks_tuning = False
    calibration_error = None
    if context.calibration_plan is not None:
        try:
            calibration = _calibrate(context.calibration_plan)
        except SemanticError as error:
            calibration_status = OnlineStageStatus.FAILED
            calibration_blocks_tuning = True
            calibration_error = str(error)
        else:
            if (
                context.timeout_s is not None
                and _monotonic() - online_started >= context.timeout_s
            ):
                return _result(
                    context,
                    preflight_status=OnlineStageStatus.PASSED,
                    calibration_status=OnlineStageStatus.FAILED,
                    release_status=OnlineStageStatus.FAILED,
                    failure_code="online_timeout",
                    failure_message="online wall-clock budget expired",
                    runtime_environment=prepared.release_environment,
                    calibration=calibration,
                )
            if calibration.skipped_reason is not None:
                calibration_status = OnlineStageStatus.NOT_RUN
                calibration_blocks_tuning = True
            elif calibration.stable:
                return _result(
                    context,
                    preflight_status=OnlineStageStatus.PASSED,
                    calibration_status=OnlineStageStatus.REQUIRES_RESOLVE,
                    runtime_environment=prepared.release_environment,
                    calibration=calibration,
                    requires_resolve=True,
                )
            else:
                calibration_status = OnlineStageStatus.UNSTABLE
                calibration_blocks_tuning = True

    operator_timeout = context.timeout_s
    if operator_timeout is not None:
        operator_timeout -= _monotonic() - online_started
        if operator_timeout <= 0.0:
            return _result(
                context,
                preflight_status=OnlineStageStatus.PASSED,
                calibration_status=calibration_status,
                release_status=OnlineStageStatus.FAILED,
                failure_code="online_timeout",
                failure_message="online wall-clock budget expired",
                runtime_environment=prepared.release_environment,
                calibration=calibration,
            )
    runner = NcclTestsRunner(
        context.executor,
        environment=prepared.release_environment,
        launcher_prefix=prepared.launcher_prefix,
        cwd=context.cwd,
        timeout_s=operator_timeout,
    )
    try:
        history = None
        if context.single_process_release_validation:
            runner.validate_release(context.request)
        else:
            history = runner.measure(context.request)
    except SemanticError as error:
        return _result(
            context,
            preflight_status=OnlineStageStatus.PASSED,
            calibration_status=calibration_status,
            release_status=OnlineStageStatus.FAILED,
            failure_code="release_measurement_failed",
            failure_message=str(error),
            runtime_environment=prepared.release_environment,
            calibration=calibration,
        )
    release_status = OnlineStageStatus.PASSED
    if history is not None and not history.stable:
        release_status = OnlineStageStatus.UNSTABLE

    assert context.clock_sync_binary is not None
    assert context.trace_file_prefix is not None
    context.trace_file_prefix.parent.mkdir(parents=True, exist_ok=True)
    try:
        clock_result = runner.run_auxiliary(
            (
                str(context.clock_sync_binary),
                str(context.clock_sync_samples),
            ),
            "VeriCCL clock synchronization",
            prepared.release_environment,
        )
    except SemanticError as error:
        return _result(
            context,
            preflight_status=OnlineStageStatus.PASSED,
            calibration_status=calibration_status,
            release_status=release_status,
            online_operator_validation=OnlineStageStatus.FAILED,
            failure_code="clock_sync_failed",
            failure_message=str(error),
            runtime_environment=prepared.release_environment,
            release_history=history,
            calibration=calibration,
        )
    try:
        diagnostic = runner.diagnostic(
            context.request,
            prepared.trace_environment,
        )
        rows = parse_nccl_tests_output(
            diagnostic.stdout,
            context.request.message_size_bytes,
            allow_unchecked=True,
        )
        if len(rows) != 1:
            raise SemanticError(
                "trace diagnostic must produce one performance row"
            )
    except SemanticError as error:
        return _result(
            context,
            preflight_status=OnlineStageStatus.PASSED,
            calibration_status=calibration_status,
            release_status=release_status,
            online_operator_validation=OnlineStageStatus.FAILED,
            failure_code="trace_run_failed",
            failure_message=str(error),
            runtime_environment=prepared.release_environment,
            release_history=history,
            calibration=calibration,
        )
    try:
        trace = context.trace_collector(
            TraceCollectionRequest(
                sidecar=prepared.sidecar,
                file_prefix=context.trace_file_prefix,
                rank_count=context.inputs.rank_count,
                clock_sync_output=clock_result.stdout,
                max_clock_uncertainty_us=context.max_clock_uncertainty_us,
                measured_iterations=20,
                inplace=context.request.inplace,
            )
        )
        if not isinstance(trace, TraceCollectionResult):
            raise SemanticError("trace collector returned an invalid result")
        if trace.clock_uncertainty_us > context.max_clock_uncertainty_us:
            raise SemanticError(
                "clock uncertainty exceeds the configured maximum"
            )
        if not trace.complete:
            return _result(
                context,
                preflight_status=OnlineStageStatus.PASSED,
                calibration_status=calibration_status,
                release_status=release_status,
                online_operator_validation=OnlineStageStatus.FAILED,
                failure_code="trace_incomplete",
                failure_message="step trace is incomplete",
                runtime_environment=prepared.release_environment,
                release_history=history,
                calibration=calibration,
                trace_analysis=trace.analysis,
                trace_rank_files=trace.rank_files,
                trace_clock_uncertainty_us=trace.clock_uncertainty_us,
            )
    except (OSError, SemanticError) as error:
        return _result(
            context,
            preflight_status=OnlineStageStatus.PASSED,
            calibration_status=calibration_status,
            release_status=release_status,
            online_operator_validation=OnlineStageStatus.FAILED,
            failure_code="trace_collection_failed",
            failure_message=str(error),
            runtime_environment=prepared.release_environment,
            release_history=history,
            calibration=calibration,
        )

    tuning_allowed = (
        context.online_tuning_requested
        and not calibration_blocks_tuning
        and history is not None
        and history.stable
        and trace.analysis.tuning_eligible
    )
    evidence = _tuning_evidence(trace.analysis) if tuning_allowed else None
    return _result(
        context,
        preflight_status=OnlineStageStatus.PASSED,
        calibration_status=calibration_status,
        release_status=release_status,
        online_operator_validation=OnlineStageStatus.PASSED,
        failure_code=(
            "calibration_failed" if calibration_error is not None else None
        ),
        failure_message=calibration_error,
        runtime_environment=prepared.release_environment,
        release_history=history,
        calibration=calibration,
        trace_analysis=trace.analysis,
        trace_rank_files=trace.rank_files,
        trace_clock_uncertainty_us=trace.clock_uncertainty_us,
        online_tuning_allowed=tuning_allowed,
        tuning_evidence=evidence,
    )


def attach_online_result_to_tuning_context(
    context,
    candidate_id: str,
    result: OnlineValidationResult,
):
    from vericcl.tuning.engine import (
        OnlinePerformance,
        TuningContext,
    )

    if not isinstance(context, TuningContext):
        raise SemanticError("context must be a TuningContext")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise SemanticError("candidate_id must be a non-empty string")
    if not isinstance(result, OnlineValidationResult):
        raise SemanticError("result must be an OnlineValidationResult")
    if (
        not result.online_tuning_allowed
        or result.online_operator_validation is not OnlineStageStatus.PASSED
        or result.release_status is not OnlineStageStatus.PASSED
        or result.release_history is None
        or result.tuning_evidence is None
    ):
        raise SemanticError("online result is not eligible for tuning")
    statistics = result.release_history.rounds[-1]
    performance = dict(context.online_performance)
    performance[candidate_id] = OnlinePerformance(
        statistics.median_us,
        statistics.coefficient_of_variation,
    )
    trace_evidence = dict(context.online_trace_evidence)
    trace_evidence[candidate_id] = result.tuning_evidence
    return replace(
        context,
        online_validation=True,
        online_performance=performance,
        online_trace_evidence=trace_evidence,
    )


__all__ = [
    "CalibrationPlan",
    "OnlineCalibrationOutcome",
    "OnlineContext",
    "OnlineStageStatus",
    "OnlineTuningEvidence",
    "OnlineValidationResult",
    "TraceCollectionRequest",
    "TraceCollectionResult",
    "attach_online_result_to_tuning_context",
    "run_online_validation",
]
