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
from vericcl.artifacts.writer import atomic_write_bytes, atomic_write_text
from vericcl.errors import SemanticError, VeriCCLError
from vericcl.experiments.model import (
    ExperimentCase,
    ExperimentManifest,
    load_experiment_manifest,
)
from vericcl.experiments.performance import (
    PerformanceResult,
    XmlSource,
    build_performance_command,
    evaluate_msccl_activation,
    select_baselines,
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
    ProcessResult,
    SubprocessCommandExecutor,
)
from vericcl.verification.online.nccl_tests import parse_nccl_tests_table
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


def _xml_source(value: object) -> XmlSource:
    try:
        return value if isinstance(value, XmlSource) else XmlSource(value)
    except (TypeError, ValueError) as error:
        raise SemanticError("performance XML source is invalid") from error


def build_performance_environment(
    config: V100ExperimentConfig,
    case: ExperimentCase,
    xml_path: Path,
    *,
    source: object,
) -> Mapping[str, str]:
    selected_source = _xml_source(source)
    xml = _absolute_path(xml_path, "performance XML path")
    if not xml.is_relative_to(config.experiment_root):
        raise SemanticError("performance XML is outside the experiment root")
    visible_count = case.rank_count // 2
    if visible_count not in {4, 8}:
        raise SemanticError("performance GPU count is unsupported")
    buffsize = (
        2 * case.slice_size_bytes
        if selected_source is XmlSource.VERICCL
        else 2 * 1024 * 1024
    )
    environment = {
        "PATH": (
            "/opt/openmpi-4.1.8/bin:/usr/local/cuda-12.8/bin:"
            "/usr/bin:/bin"
        ),
        "LD_LIBRARY_PATH": (
            "/opt/openmpi-4.1.8/lib:{}:/usr/local/cuda-12.8/lib64:"
            "/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu"
        ).format(config.msccl_library_path),
        "CUDA_VISIBLE_DEVICES": ",".join(
            str(rank) for rank in range(visible_count)
        ),
        "MSCCL_XML_FILES": str(xml),
        "NCCL_ALGO": "MSCCL,RING",
        "NCCL_BUFFSIZE": str(buffsize),
        "NCCL_DEBUG": "INFO",
        "NCCL_IB_DISABLE": "1",
        "NCCL_IGNORE_DISABLED_P2P": "1",
        "NCCL_NET_GDR_LEVEL": "0",
        "NCCL_NET_GDR_READ": "0",
        "NCCL_P2P_LEVEL": "NVL",
        "NCCL_PROTO": "Simple",
        "NCCL_SOCKET_IFNAME": config.environment.get(
            "NCCL_SOCKET_IFNAME",
            "eno0,enp4s0",
        ),
    }
    return MappingProxyType(environment)


def build_mpi_prefix(
    config: V100ExperimentConfig,
    rank_count: int,
    exported_names: Sequence[str],
) -> Tuple[str, ...]:
    hostfile = config.hostfile(rank_count)
    prefix = [
        str(config.mpi_launcher),
        "--allow-run-as-root",
        "--prefix",
        str(config.mpi_launcher.parent.parent),
        "-np",
        str(rank_count),
        "--hostfile",
        str(hostfile),
        "-mca",
        "pml",
        "ob1",
        "-mca",
        "btl",
        "tcp,self,vader",
        "-mca",
        "btl_vader_single_copy_mechanism",
        "none",
        "-mca",
        "btl_tcp_if_include",
        "10.0.0.0/24",
    ]
    names = tuple(exported_names)
    if not all(
        isinstance(name, str) and name and "\x00" not in name
        for name in names
    ):
        raise SemanticError("MPI export name is invalid")
    for name in sorted(set(names)):
        prefix.extend(("-x", name))
    return tuple(prefix)


def _measurement_payload(run) -> dict:
    def placement(value):
        if value is None:
            return None
        return {
            "algorithm_bandwidth_gbps": value.algorithm_bandwidth_gbps,
            "bus_bandwidth_gbps": value.bus_bandwidth_gbps,
            "time_us": value.time_us,
            "wrong_count": value.wrong_count,
        }

    return {
        "datatype": run.datatype,
        "element_count": run.element_count,
        "in_place": placement(run.in_place),
        "message_size_bytes": run.message_size_bytes,
        "metadata_fields": list(run.metadata_fields),
        "out_of_place": placement(run.out_of_place),
    }


def _activation_payload(value) -> dict:
    return {
        "confirmed": value.confirmed,
        "info_loaded": value.info_loaded,
        "relative_busbw_difference": value.relative_busbw_difference,
        "threshold": value.threshold,
    }


def benchmark_xml(
    case: ExperimentCase,
    config: V100ExperimentConfig,
    *,
    source: object,
    xml_path: Path,
    begin: str,
    end: str,
    output_directory: Path,
    executor=None,
    timeout_s: float = 1800.0,
    xml_name: Optional[str] = None,
    task_id: Optional[str] = None,
) -> PerformanceResult:
    selected_source = _xml_source(source)
    xml = _absolute_path(xml_path, "performance XML path")
    directory = _absolute_path(
        output_directory,
        "performance output directory",
    )
    if not directory.is_relative_to(config.experiment_root):
        raise SemanticError(
            "performance output is outside the experiment root"
        )
    selected_task_id = (
        "perf-{}-{}".format(case.task_id, selected_source.value)
        if task_id is None
        else task_id
    )
    selected_xml_name = xml.name if xml_name is None else xml_name
    environment = build_performance_environment(
        config,
        case,
        xml,
        source=selected_source,
    )
    binary_name = {
        "ag": "all_gather_perf",
        "ar": "all_reduce_perf",
    }[case.collective_label]
    command = build_mpi_prefix(
        config,
        case.rank_count,
        tuple(environment),
    ) + build_performance_command(
        binary=str(config.nccl_tests_binary_directory / binary_name),
        begin=begin,
        end=end,
        factor=2,
        iterations=15,
    )
    directory.mkdir(parents=True, exist_ok=True)
    command_path = directory / "command.txt"
    environment_path = directory / "environment.json"
    stdout_path = directory / "stdout.log"
    stderr_path = directory / "stderr.log"
    atomic_write_text(command_path, " ".join(command) + "\n")
    atomic_write_text(
        environment_path,
        json.dumps(
            dict(environment),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
    )
    selected_executor = executor
    if selected_executor is None:
        selected_executor, _ = _runtime_dependencies(
            config,
            case.rank_count,
        )
    process = selected_executor.run(
        ProcessRequest(
            command=command,
            environment=environment,
            label="{} performance".format(selected_source.value),
            timeout_s=timeout_s,
        )
    )
    if not isinstance(process, ProcessResult):
        raise SemanticError("performance executor returned an invalid result")
    atomic_write_text(stdout_path, process.stdout)
    atomic_write_text(stderr_path, process.stderr)
    if process.returncode != 0:
        raise SemanticError(
            "performance process failed with status {}".format(
                process.returncode
            )
        )
    runs = parse_nccl_tests_table(process.stdout)
    combined = process.stdout + "\n" + process.stderr
    activation = tuple(
        evaluate_msccl_activation(combined, run) for run in runs
    )
    measurements_path = directory / "measurements.json"
    activation_path = directory / "activation.json"
    atomic_write_text(
        measurements_path,
        json.dumps(
            {
                "collective_label": case.collective_label,
                "runs": [_measurement_payload(run) for run in runs],
                "schema_version": 1,
                "source": selected_source.value,
                "task_id": selected_task_id,
                "topology_name": case.topology_name,
                "xml_name": selected_xml_name,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
    )
    atomic_write_text(
        activation_path,
        json.dumps(
            {
                "activation": [
                    _activation_payload(value) for value in activation
                ],
                "schema_version": 1,
                "task_id": selected_task_id,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
    )
    return PerformanceResult(
        task_id=selected_task_id,
        topology_name=case.topology_name,
        collective_label=case.collective_label,
        source=selected_source,
        xml_name=selected_xml_name,
        runs=runs,
        activation=activation,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )


def _nccl_size(size_bytes: int) -> str:
    gibibyte = 1024 * 1024 * 1024
    mebibyte = 1024 * 1024
    if size_bytes % gibibyte == 0:
        return "{}G".format(size_bytes // gibibyte)
    if size_bytes % mebibyte == 0:
        return "{}M".format(size_bytes // mebibyte)
    raise SemanticError("performance size is not MiB aligned")


def _benchmark_input_sha256(
    case: ExperimentCase,
    config: V100ExperimentConfig,
    xml_path: Path,
    source: XmlSource,
    begin: str,
    end: str,
) -> str:
    payload = {
        "begin": begin,
        "case": case.task_id,
        "end": end,
        "source": source.value,
        "xml_sha256": _file_sha256(xml_path),
        "environment": dict(
            build_performance_environment(
                config,
                case,
                xml_path,
                source=source,
            )
        ),
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()


def _copy_baselines(
    config: V100ExperimentConfig,
) -> Mapping[Path, str]:
    destination = config.experiment_root / "baselines"
    mapping = {}
    index = []
    for source in sorted(config.baseline_source.rglob("*.xml")):
        if not source.is_file() or source.stat().st_size <= 0:
            continue
        digest = _file_sha256(source)
        target = destination / "{}.xml".format(digest)
        if not target.exists() or _file_sha256(target) != digest:
            atomic_write_bytes(target, source.read_bytes())
        mapping[target] = source.name
        index.append(
            {
                "copied_path": str(target),
                "original_path": str(source),
                "sha256": digest,
            }
        )
    if not mapping:
        raise SemanticError("baseline source contains no XML files")
    atomic_write_text(
        destination / "index.json",
        json.dumps(
            {"baselines": index},
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
    )
    return MappingProxyType(mapping)


def _performance_task_id(
    case: ExperimentCase,
    source: XmlSource,
    xml_path: Path,
) -> str:
    if source is XmlSource.VERICCL:
        return "perf-{}-vericcl".format(case.task_id)
    return "perf-{}-{}-baseline-{}".format(
        case.topology_name,
        case.collective_label,
        _file_sha256(xml_path)[:12],
    )


def _performance_reusable(
    record: Optional[TaskRecord],
    input_sha256: str,
) -> bool:
    if (
        record is None
        or record.status is not TaskStatus.PASSED
        or record.input_sha256 != input_sha256
        or record.output_sha256 is None
        or record.log_path is None
    ):
        return False
    stdout_path = Path(record.log_path)
    measurements = stdout_path.parent / "measurements.json"
    return (
        stdout_path.is_file()
        and measurements.is_file()
        and _file_sha256(measurements) == record.output_sha256
    )


def _run_benchmark_task(
    case: ExperimentCase,
    config: V100ExperimentConfig,
    *,
    source: XmlSource,
    xml_path: Path,
    xml_name: str,
    begin: str,
    end: str,
    output_directory: Path,
    store: ExperimentStateStore,
    resume: bool,
    executor=None,
) -> TaskRecord:
    task_id = _performance_task_id(case, source, xml_path)
    input_sha256 = _benchmark_input_sha256(
        case,
        config,
        xml_path,
        source,
        begin,
        end,
    )
    previous = store.load().get(task_id)
    if resume and _performance_reusable(previous, input_sha256):
        assert previous is not None
        return previous
    command = (
        "python",
        "-m",
        "vericcl.experiments.v100",
        "benchmark",
        "--case",
        case.task_id,
    )
    started = _utc_now()
    store.put(
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
        result = benchmark_xml(
            case,
            config,
            source=source,
            xml_path=xml_path,
            begin=begin,
            end=end,
            output_directory=output_directory,
            executor=executor,
            xml_name=xml_name,
            task_id=task_id,
        )
        measurements = result.stdout_path.parent / "measurements.json"
        record = TaskRecord(
            task_id=task_id,
            status=TaskStatus.PASSED,
            input_sha256=input_sha256,
            output_sha256=_file_sha256(measurements),
            command=command,
            returncode=0,
            log_path=str(result.stdout_path),
            started_at_utc=started,
            finished_at_utc=_utc_now(),
        )
    except Exception as error:
        output_directory.mkdir(parents=True, exist_ok=True)
        error_path = output_directory / "runner-error.log"
        atomic_write_text(error_path, "{}\n".format(error))
        record = TaskRecord(
            task_id=task_id,
            status=TaskStatus.FAILED,
            input_sha256=input_sha256,
            output_sha256=None,
            command=command,
            returncode=1,
            failure_code=type(error).__name__,
            log_path=str(error_path),
            started_at_utc=started,
            finished_at_utc=_utc_now(),
        )
    store.put(record)
    return record


def benchmark_matrix(
    config: V100ExperimentConfig,
    *,
    case_ids: Sequence[str] = (),
    resume: bool = False,
    executor=None,
) -> Tuple[TaskRecord, ...]:
    manifest = load_experiment_manifest(
        config.manifest_path,
        repo_root=config.repo_root,
    )
    cases = _select_cases(manifest, case_ids)
    state = ExperimentStateStore(config.state_path)
    solve_records = state.load()
    records = []
    selected_executor = executor
    if selected_executor is None:
        selected_executor, _ = _runtime_dependencies(
            config,
            cases[0].rank_count,
        )
    for case in cases:
        solve_record = solve_records.get(case.task_id)
        if (
            solve_record is None
            or solve_record.status is not TaskStatus.PASSED
            or solve_record.log_path is None
        ):
            continue
        xml_path = Path(solve_record.log_path)
        size = _nccl_size(case.message_size_bytes)
        records.append(
            _run_benchmark_task(
                case,
                config,
                source=XmlSource.VERICCL,
                xml_path=xml_path,
                xml_name=xml_path.name,
                begin=size,
                end=size,
                output_directory=(
                    config.experiment_root
                    / "performance"
                    / "vericcl"
                    / case.task_id
                ),
                store=state,
                resume=resume,
                executor=selected_executor,
            )
        )

    copied = _copy_baselines(config)
    pairs = {}
    for case in cases:
        pairs.setdefault(
            (case.topology_name, case.collective_label),
            case,
        )
    collective_names = {"ag": "allgather", "ar": "allreduce"}
    for (topology_name, collective_label), case in sorted(pairs.items()):
        selected = select_baselines(
            copied,
            collective=collective_names[collective_label],
            rank_count=case.rank_count,
        )
        if not selected:
            raise SemanticError(
                "no baseline XML matches {} {}".format(
                    topology_name,
                    collective_label,
                )
            )
        for xml_path in selected:
            digest = _file_sha256(xml_path)
            records.append(
                _run_benchmark_task(
                    case,
                    config,
                    source=XmlSource.BASELINE,
                    xml_path=xml_path,
                    xml_name=copied[xml_path],
                    begin="4M",
                    end="2G",
                    output_directory=(
                        config.experiment_root
                        / "performance"
                        / "baseline"
                        / "{}-{}".format(
                            topology_name,
                            collective_label,
                        )
                        / digest[:12]
                    ),
                    store=state,
                    resume=resume,
                    executor=selected_executor,
                )
            )
    return tuple(records)


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
        if arguments.stage in {"solve", "all"}:
            records = solve_matrix(
                config,
                case_ids=tuple(arguments.case),
                resume=arguments.resume,
            )
            if not all(
                record.status is TaskStatus.PASSED for record in records
            ):
                return 1
        if arguments.stage in {"benchmark", "all"}:
            records = benchmark_matrix(
                config,
                case_ids=tuple(arguments.case),
                resume=arguments.resume,
            )
            if not all(
                record.status is TaskStatus.PASSED for record in records
            ):
                return 1
        if arguments.stage == "summarize":
            raise SemanticError("summarize stage is not available in this build")
    except VeriCCLError as error:
        print("VeriCCL V100 experiment failed: {}".format(error))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
