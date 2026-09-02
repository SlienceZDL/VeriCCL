from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping, Optional, Sequence, Tuple

from vericcl.cli.online import build_online_context_factory
from vericcl.artifacts.writer import atomic_write_text
from vericcl.errors import SemanticError, VeriCCLError
from vericcl.experiments.model import (
    ExperimentCase,
    ExperimentManifest,
    load_experiment_manifest,
)
from vericcl.experiments.remote import (
    ExperimentPathPolicy,
    RemoteTraceCollector,
    SshFileStager,
    SshStagingCommandExecutor,
)
from vericcl.experiments.state import (
    ExperimentStateStore,
    TaskRecord,
    TaskStatus,
    atomic_replace_text,
)
from vericcl.verification.online.runner import (
    ProcessRequest,
    SubprocessCommandExecutor,
)
from vericcl.verification.online.calibration import CalibrationRequest
from vericcl.verification.online.calibration_xml import (
    build_calibration_benchmark,
)
from vericcl.verification.online.pipeline import (
    OnlineStageStatus,
    run_online_validation,
)
from vericcl.cli.online import representative_calibration_topology
from vericcl.input.loader import resolve_inputs
from vericcl.topology.loader import load_topology
from vericcl.workflow import RunContext, execute_solve


HOSTS = {
    2: (("10.0.0.104", 1), ("10.0.0.102", 1)),
    8: (("10.0.0.104", 4), ("10.0.0.102", 4)),
    16: (("10.0.0.104", 8), ("10.0.0.102", 8)),
}

_CONFIG_FIELDS = frozenset(
    {
        "schema_version",
        "experiment_root",
        "repo_root",
        "manifest_path",
        "baseline_source",
        "remote_host",
        "mpi_launcher",
        "msccl_library_path",
        "nccl_tests_binary_directory",
        "clock_sync_binary",
        "calibration_cache_path",
        "hostfiles",
        "max_clock_uncertainty_us",
        "environment",
    }
)
_HOSTFILE_FIELDS = frozenset({"8", "16"})
_REQUIRED_ENVIRONMENT = frozenset(
    {
        "NCCL_ALGO",
        "NCCL_IB_DISABLE",
        "NCCL_PROTO",
        "VERICCL_CALIBRATION_LINK_CLASS",
        "VERICCL_CUDA_VERSION",
        "VERICCL_FORCE_RECALIBRATE",
        "VERICCL_GPU_MODEL",
        "VERICCL_MSCCL_VERSION",
        "VERICCL_NCCL_VERSION",
        "VERICCL_NIC_MODEL",
        "VERICCL_ONLINE_INTER_NODE",
    }
)


def _absolute_path(value: object, field: str) -> Path:
    try:
        path = Path(value)
    except TypeError as error:
        raise SemanticError("{} is invalid".format(field)) from error
    if not path.is_absolute():
        raise SemanticError("{} must be absolute".format(field))
    resolved = path.resolve()
    forbidden = Path("/home/cc")
    if resolved == forbidden or resolved.is_relative_to(forbidden):
        raise SemanticError("{} uses a forbidden path".format(field))
    return resolved


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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            while True:
                block = stream.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
    except OSError as error:
        raise SemanticError("experiment artifact is unreadable") from error
    return digest.hexdigest()


def _case_input_sha256(case: ExperimentCase, atom_path: Path) -> str:
    digest = hashlib.sha256()
    for label, path in (
        ("topology", case.topology_path),
        ("sketch", case.sketch_path),
        ("atom", atom_path),
    ):
        digest.update(label.encode("ascii"))
        digest.update(b"\x00")
        digest.update(bytes.fromhex(_file_sha256(path)))
    return digest.hexdigest()


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SemanticError("V100 config contains duplicate fields")
        result[key] = value
    return result


@dataclass(frozen=True)
class V100ExperimentConfig:
    experiment_root: Path
    repo_root: Path
    manifest_path: Path
    baseline_source: Path
    remote_host: str
    mpi_launcher: Path
    msccl_library_path: Path
    nccl_tests_binary_directory: Path
    clock_sync_binary: Path
    calibration_cache_path: Path
    hostfile_8: Path
    hostfile_16: Path
    environment: Mapping[str, str]
    max_clock_uncertainty_us: float = 50.0

    def __post_init__(self) -> None:
        for field in (
            "experiment_root",
            "repo_root",
            "manifest_path",
            "baseline_source",
            "mpi_launcher",
            "msccl_library_path",
            "nccl_tests_binary_directory",
            "clock_sync_binary",
            "calibration_cache_path",
            "hostfile_8",
            "hostfile_16",
        ):
            object.__setattr__(
                self,
                field,
                _absolute_path(getattr(self, field), field),
            )
        if not isinstance(self.remote_host, str) or not self.remote_host:
            raise SemanticError("remote_host must be a non-empty string")
        root = self.experiment_root
        for field in (
            "repo_root",
            "manifest_path",
            "msccl_library_path",
            "nccl_tests_binary_directory",
            "clock_sync_binary",
            "calibration_cache_path",
            "hostfile_8",
            "hostfile_16",
        ):
            if not getattr(self, field).is_relative_to(root):
                raise SemanticError(
                    "{} is outside the experiment root".format(field)
                )
        if not self.manifest_path.is_relative_to(self.repo_root):
            raise SemanticError("manifest_path is outside repo_root")
        environment = _string_mapping(self.environment, "environment")
        missing = sorted(_REQUIRED_ENVIRONMENT.difference(environment))
        if missing:
            raise SemanticError(
                "environment is missing {}".format(", ".join(missing))
            )
        object.__setattr__(self, "environment", environment)
        threshold = self.max_clock_uncertainty_us
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
            or float(threshold) <= 0.0
        ):
            raise SemanticError(
                "max_clock_uncertainty_us must be positive"
            )
        object.__setattr__(
            self,
            "max_clock_uncertainty_us",
            float(threshold),
        )

    def hostfile(self, rank_count: int) -> Path:
        if rank_count == 8:
            return self.hostfile_8
        if rank_count == 16:
            return self.hostfile_16
        if rank_count == 2:
            return self.experiment_root / "hostfile-2x1"
        raise SemanticError("unsupported V100 rank count")

    @property
    def state_path(self) -> Path:
        return self.experiment_root / "state.json"


def load_v100_config(path: Path) -> V100ExperimentConfig:
    config_path = _absolute_path(path, "V100 config path")
    try:
        payload = json.loads(
            config_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except SemanticError:
        raise
    except (OSError, UnicodeError, ValueError) as error:
        raise SemanticError("V100 config is unreadable") from error
    if not isinstance(payload, dict) or set(payload) != _CONFIG_FIELDS:
        raise SemanticError("V100 config fields are invalid")
    if payload["schema_version"] != 1:
        raise SemanticError("V100 config schema is unsupported")
    hostfiles = payload["hostfiles"]
    if not isinstance(hostfiles, dict) or set(hostfiles) != _HOSTFILE_FIELDS:
        raise SemanticError("V100 config hostfile fields are invalid")
    return V100ExperimentConfig(
        experiment_root=payload["experiment_root"],
        repo_root=payload["repo_root"],
        manifest_path=payload["manifest_path"],
        baseline_source=payload["baseline_source"],
        remote_host=payload["remote_host"],
        mpi_launcher=payload["mpi_launcher"],
        msccl_library_path=payload["msccl_library_path"],
        nccl_tests_binary_directory=payload[
            "nccl_tests_binary_directory"
        ],
        clock_sync_binary=payload["clock_sync_binary"],
        calibration_cache_path=payload["calibration_cache_path"],
        hostfile_8=hostfiles["8"],
        hostfile_16=hostfiles["16"],
        environment=payload["environment"],
        max_clock_uncertainty_us=payload["max_clock_uncertainty_us"],
    )


def write_hostfile(path: Path, rank_count: int) -> None:
    try:
        hosts = HOSTS[rank_count]
    except KeyError as error:
        raise SemanticError("unsupported V100 rank count") from error
    lines = tuple(
        "{} slots={}".format(host, slots) for host, slots in hosts
    )
    atomic_replace_text(Path(path), "\n".join(lines) + "\n")


def _expected_hostfile(rank_count: int) -> str:
    return "\n".join(
        "{} slots={}".format(host, slots)
        for host, slots in HOSTS[rank_count]
    ) + "\n"


def preflight(
    config: V100ExperimentConfig,
    *,
    executor=None,
) -> None:
    if not isinstance(config, V100ExperimentConfig):
        raise SemanticError("preflight requires V100ExperimentConfig")
    files = (
        config.manifest_path,
        config.mpi_launcher,
        config.clock_sync_binary,
        config.hostfile_8,
        config.hostfile_16,
    )
    if not all(path.is_file() for path in files):
        raise SemanticError("V100 preflight required file is missing")
    directories = (
        config.repo_root,
        config.baseline_source,
        config.msccl_library_path,
        config.nccl_tests_binary_directory,
    )
    if not all(path.is_dir() for path in directories):
        raise SemanticError("V100 preflight required directory is missing")
    if not os.access(config.mpi_launcher, os.X_OK) or not os.access(
        config.clock_sync_binary,
        os.X_OK,
    ):
        raise SemanticError("V100 preflight executable is not executable")
    for rank_count, path in (
        (8, config.hostfile_8),
        (16, config.hostfile_16),
    ):
        actual = path.read_text(encoding="ascii")
        expected = _expected_hostfile(rank_count)
        if actual != expected:
            if not actual.startswith("10.0.0.104 "):
                raise SemanticError("node4 must be first in each hostfile")
            raise SemanticError("V100 hostfile content is invalid")
    environment = config.environment
    if environment["NCCL_IB_DISABLE"] != "1":
        raise SemanticError("NCCL_IB_DISABLE must be one")
    if environment["NCCL_ALGO"] != "MSCCL,RING":
        raise SemanticError("NCCL_ALGO must be MSCCL,RING")
    if environment["NCCL_PROTO"] != "Simple":
        raise SemanticError("NCCL_PROTO must be Simple")
    if environment["VERICCL_ONLINE_INTER_NODE"] != "1":
        raise SemanticError("VERICCL_ONLINE_INTER_NODE must be one")
    if config.max_clock_uncertainty_us != 50.0:
        raise SemanticError("clock uncertainty threshold must be 50 us")
    if executor is None:
        return
    request = ProcessRequest(
        command=(
            "ssh",
            "-o",
            "BatchMode=yes",
            config.remote_host,
            "true",
        ),
        environment=os.environ,
        label="V100 remote preflight",
        timeout_s=30.0,
    )
    result = executor.run(request)
    if result.returncode != 0:
        raise SemanticError("V100 remote SSH preflight failed")


def _runtime_dependencies(config: V100ExperimentConfig, rank_count: int):
    delegate = SubprocessCommandExecutor()
    policy = ExperimentPathPolicy(config.experiment_root)
    stager = SshFileStager(
        delegate=delegate,
        remote_host=config.remote_host,
        path_policy=policy,
    )
    for count in (8, 16):
        path = config.hostfile(count)
        stager.upload(path, path)
    if rank_count == 2:
        path = config.hostfile(2)
        write_hostfile(path, 2)
        stager.upload(path, path)
    remote_executor = SshStagingCommandExecutor(
        delegate=delegate,
        stager=stager,
        remote_host=config.remote_host,
        path_policy=policy,
    )
    collector = RemoteTraceCollector(stager=stager)
    return remote_executor, collector


def _factory_environment(
    config: V100ExperimentConfig,
    rank_count: int,
) -> Mapping[str, str]:
    environment = dict(config.environment)
    environment.update(
        {
            "VERICCL_CALIBRATION_CACHE_PATH": str(
                config.calibration_cache_path
            ),
            "VERICCL_CLOCK_SYNC_BINARY": str(config.clock_sync_binary),
            "VERICCL_MAX_CLOCK_UNCERTAINTY_US": str(
                config.max_clock_uncertainty_us
            ),
            "VERICCL_MPI_HOSTFILE": str(config.hostfile(rank_count)),
            "VERICCL_MPI_LAUNCHER": str(config.mpi_launcher),
            "VERICCL_MSCCL_BUILD_DIR": str(config.msccl_library_path),
            "VERICCL_NCCL_TESTS_BUILD_DIR": str(
                config.nccl_tests_binary_directory
            ),
        }
    )
    return MappingProxyType(environment)


def _online_factory(config: V100ExperimentConfig, rank_count: int):
    remote_executor, collector = _runtime_dependencies(config, rank_count)
    return build_online_context_factory(
        _factory_environment(config, rank_count),
        executor=remote_executor,
        trace_collector=collector,
    )


def _online_report_is_valid(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload["validation"]["online"]["status"] == "valid"
    except (OSError, UnicodeError, ValueError, KeyError, TypeError):
        return False


def solve_case(
    case: ExperimentCase,
    config: V100ExperimentConfig,
    *,
    atom_path: Optional[Path] = None,
    execute: Callable[[RunContext], object] = execute_solve,
    online_factory=None,
    state_store: Optional[ExperimentStateStore] = None,
) -> TaskRecord:
    if not isinstance(case, ExperimentCase):
        raise SemanticError("solve_case requires an ExperimentCase")
    if not isinstance(config, V100ExperimentConfig):
        raise SemanticError("solve_case requires V100ExperimentConfig")
    atom = (
        config.repo_root / "vericcl/examples/atom/default.json"
        if atom_path is None
        else _absolute_path(atom_path, "atom_path")
    )
    input_sha256 = _case_input_sha256(case, atom)
    command = (
        "python",
        "-m",
        "vericcl.experiments.v100",
        "solve",
        "--case",
        case.task_id,
    )
    started = _utc_now()
    running = TaskRecord(
        task_id=case.task_id,
        status=TaskStatus.RUNNING,
        input_sha256=input_sha256,
        output_sha256=None,
        command=command,
        returncode=None,
        started_at_utc=started,
    )
    if state_store is not None:
        state_store.put(running)
    try:
        if online_factory is None:
            runtime_factory = None

            def selected_factory(*args, **kwargs):
                nonlocal runtime_factory
                if runtime_factory is None:
                    runtime_factory = _online_factory(
                        config,
                        case.rank_count,
                    )
                return runtime_factory(*args, **kwargs)
        else:
            selected_factory = online_factory
        context = RunContext(
            topology_path=case.topology_path,
            sketch_path=case.sketch_path,
            atom_path=atom,
            output_base=config.experiment_root / "runs" / case.task_id,
            run_id=case.task_id,
            online=True,
            tune=True,
            timeout_s=10800.0,
            environment_signature="v100-k16",
            online_context_factory=selected_factory,
        )
        artifacts = execute(context)
        final_xml = artifacts.final_xml
        final_report = artifacts.final_report
        valid = (
            artifacts.final_candidate_id is not None
            and final_xml is not None
            and Path(final_xml).is_file()
            and Path(final_xml).stat().st_size > 0
            and final_report is not None
            and Path(final_report).is_file()
            and Path(final_report).stat().st_size > 0
            and _online_report_is_valid(Path(final_report))
        )
        if not valid:
            raise SemanticError(
                "solve result has no online-valid final artifact"
            )
        record = TaskRecord(
            task_id=case.task_id,
            status=TaskStatus.PASSED,
            input_sha256=input_sha256,
            output_sha256=_file_sha256(Path(final_xml)),
            command=command,
            returncode=0,
            log_path=str(final_xml),
            started_at_utc=started,
            finished_at_utc=_utc_now(),
        )
    except Exception as error:
        record = TaskRecord(
            task_id=case.task_id,
            status=TaskStatus.FAILED,
            input_sha256=input_sha256,
            output_sha256=None,
            command=command,
            returncode=1,
            failure_code=type(error).__name__,
            log_path=None,
            started_at_utc=started,
            finished_at_utc=_utc_now(),
        )
    if state_store is not None:
        state_store.put(record)
    return record


def smoke(
    config: V100ExperimentConfig,
    *,
    factory_builder: Callable = _online_factory,
    run: Callable = run_online_validation,
    state_store: Optional[ExperimentStateStore] = None,
) -> TaskRecord:
    manifest = load_experiment_manifest(
        config.manifest_path,
        repo_root=config.repo_root,
    )
    case = manifest.cases[0]
    input_sha256 = _case_input_sha256(case, manifest.atom_path)
    task_id = "smoke-2x1-128m"
    command = (
        "python",
        "-m",
        "vericcl.experiments.v100",
        "smoke",
    )
    started = _utc_now()
    if state_store is not None:
        state_store.put(
            TaskRecord(
                task_id=task_id,
                status=TaskStatus.RUNNING,
                input_sha256=input_sha256,
                output_sha256=None,
                command=command,
                returncode=None,
                started_at_utc=started,
            )
        )
    try:
        inputs = resolve_inputs(
            case.topology_path,
            case.sketch_path,
            manifest.atom_path,
        )
        topology = representative_calibration_topology(
            load_topology(inputs),
            "inter_node",
        )
        request = CalibrationRequest(
            link_class="inter_node",
            slice_size_bytes=inputs.hyperparameters.slice_size_bytes,
            max_calibration_channels=1,
            datatype="float",
        )
        benchmark = build_calibration_benchmark(
            request,
            topology,
            concurrency=1,
        )
        directory = config.experiment_root / "smoke"
        xml_path = directory / "inter-node-k01-128m.xml"
        atomic_write_text(xml_path, benchmark.artifact.xml_text)
        factory = factory_builder(config, 2)
        context = factory(
            benchmark.artifact,
            benchmark.schedule,
            benchmark.inputs,
            xml_path,
            directory / "traces",
            False,
            1800.0,
            False,
        )
        result = run(context)
        valid = (
            result.release_status is OnlineStageStatus.PASSED
            and result.online_operator_validation is OnlineStageStatus.PASSED
            and result.failure_code is None
        )
        if not valid:
            raise SemanticError(
                result.failure_message or "V100 smoke validation failed"
            )
        report_path = directory / "smoke.validation.json"
        atomic_write_text(
            report_path,
            json.dumps(
                {
                    "failure_code": result.failure_code,
                    "online_operator_validation": (
                        result.online_operator_validation.value
                    ),
                    "release_status": result.release_status.value,
                    "trace_clock_uncertainty_us": (
                        result.trace_clock_uncertainty_us
                    ),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
        )
        record = TaskRecord(
            task_id=task_id,
            status=TaskStatus.PASSED,
            input_sha256=input_sha256,
            output_sha256=_file_sha256(xml_path),
            command=command,
            returncode=0,
            log_path=str(report_path),
            started_at_utc=started,
            finished_at_utc=_utc_now(),
        )
    except Exception as error:
        record = TaskRecord(
            task_id=task_id,
            status=TaskStatus.FAILED,
            input_sha256=input_sha256,
            output_sha256=None,
            command=command,
            returncode=1,
            failure_code=type(error).__name__,
            started_at_utc=started,
            finished_at_utc=_utc_now(),
        )
    if state_store is not None:
        state_store.put(record)
    return record


def _select_cases(
    manifest: ExperimentManifest,
    requested: Sequence[str],
) -> Tuple[ExperimentCase, ...]:
    by_id = {case.task_id: case for case in manifest.cases}
    unknown = tuple(value for value in requested if value not in by_id)
    if unknown:
        raise SemanticError(
            "unknown experiment case: {}".format(", ".join(unknown))
        )
    if requested:
        return tuple(by_id[value] for value in requested)
    return manifest.cases


def _record_is_reusable(
    record: Optional[TaskRecord],
    case: ExperimentCase,
    atom_path: Path,
) -> bool:
    if (
        record is None
        or record.status is not TaskStatus.PASSED
        or record.log_path is None
        or record.output_sha256 is None
    ):
        return False
    output = Path(record.log_path)
    return (
        output.is_file()
        and record.input_sha256 == _case_input_sha256(case, atom_path)
        and record.output_sha256 == _file_sha256(output)
    )


def solve_matrix(
    config: V100ExperimentConfig,
    *,
    case_ids: Sequence[str] = (),
    resume: bool = False,
) -> Tuple[TaskRecord, ...]:
    manifest = load_experiment_manifest(
        config.manifest_path,
        repo_root=config.repo_root,
    )
    store = ExperimentStateStore(config.state_path)
    previous = store.load()
    records = []
    for case in _select_cases(manifest, case_ids):
        if resume and _record_is_reusable(
            previous.get(case.task_id),
            case,
            manifest.atom_path,
        ):
            records.append(previous[case.task_id])
            continue
        records.append(
            solve_case(
                case,
                config,
                atom_path=manifest.atom_path,
                state_store=store,
            )
        )
    return tuple(records)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vericcl-v100")
    parser.add_argument(
        "stage",
        choices=(
            "preflight",
            "smoke",
            "solve",
            "benchmark",
            "summarize",
            "all",
        ),
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _build_parser().parse_args(argv)
    try:
        config = load_v100_config(arguments.config)
        if arguments.stage in {"preflight", "all"}:
            preflight(config, executor=SubprocessCommandExecutor())
        if arguments.stage in {"smoke", "all"}:
            record = smoke(
                config,
                state_store=ExperimentStateStore(config.state_path),
            )
            if record.status is not TaskStatus.PASSED:
                return 1
        if arguments.stage in {"benchmark", "summarize"}:
            raise SemanticError(
                "{} stage is not available in this build".format(
                    arguments.stage
                )
            )
        if arguments.stage in {"solve", "all"}:
            records = solve_matrix(
                config,
                case_ids=tuple(arguments.case),
                resume=arguments.resume,
            )
            return 0 if all(
                record.status is TaskStatus.PASSED for record in records
            ) else 1
    except VeriCCLError as error:
        print("VeriCCL V100 experiment failed: {}".format(error))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
