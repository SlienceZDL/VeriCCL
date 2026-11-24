"""Staged V100 experiment harness on the v5 pipeline.

Stages: ``preflight -> smoke -> solve -> benchmark -> summarize``
(``python -m vericcl.experiments.v100 <stage> --config <config.json>``).

Inputs. Cases come from a schema_version 2 manifest
(:mod:`vericcl.experiments.model`): v5 topology, sketch and a *logical* v5
runtime per case plus one shared atom. The logical runtime holds the
experiment's runtime meaning (rank-to-node/device placement on logical hosts,
explicit ``NCCL_BUFFSIZE``, scratch and memory budgets). :func:`bind_runtime`
binds it to this deployment from :class:`V100ExperimentConfig`: logical hosts
are replaced, in order of first rank, by the hosts of the configured
hostfiles (``remote_host`` first); ``runtime.paths`` gets ``mpi_bin`` from the
MPI launcher plus the config ``runtime_paths`` (``vericcl_driver`` and
``vericcl_native`` are required; see ``examples/v100-experiments``);
allowlisted config environment entries, the MPI transport and the NCCL
transport of the benchmark stage (:data:`BENCHMARK_TRANSPORT`) fill
environment keys the case does not set, so calibration and trace runs measure
the transport the benchmark uses. Case values (e.g. ``NCCL_BUFFSIZE``) always
win.

Runs. ``solve`` runs :func:`vericcl.workflow_v5.run_primary` (purpose
``solve``) with project root ``<experiment_root>/runs/<task_id>``; the v5 run
directory is ``<project root>/examples/<task_id>/output/<run_id>`` and the
bound runtime is kept at ``<project root>/inputs/runtime.json``. ``smoke``
runs purpose ``calibrate`` on a 2-rank inter-node case derived from the first
manifest case under ``<experiment_root>/smoke``.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import re
import traceback
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping, Optional, Sequence, Tuple

from vericcl.artifacts.writer import atomic_write_bytes, atomic_write_text
from vericcl.errors import SemanticError, VeriCCLError
from vericcl.execution.context import RunContext, StorageFailure, finalize_run
from vericcl.execution.events import emit_event
from vericcl.execution.security import ENVIRONMENT_ALLOWLIST, is_foreign_home_path
from vericcl.execution.status import RunResult
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
from vericcl.experiments.report import write_report
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
from vericcl.input.loader import load_request
from vericcl.workflow_v5 import PrimaryRequest, run_primary


HOSTS = {
    2: (("worker-a", 1), ("worker-b", 1)),
    8: (("worker-a", 4), ("worker-b", 4)),
    16: (("worker-a", 8), ("worker-b", 8)),
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
        "runtime_paths",
    }
)
_REQUIRED_RUNTIME_PATHS = ("vericcl_driver", "vericcl_native")
_RUNTIME_PATH_FIELDS = frozenset(
    {
        "cuda_home",
        "msccl_root",
        "nccl_tests_root",
        "vericcl_driver",
        "vericcl_native",
    }
)
_HOSTFILE_FIELDS = frozenset({"8", "16"})
SMOKE_TASK_ID = "smoke-2x1-128m"
SMOKE_TOTAL_SIZE_BYTES = 128 * 1024 * 1024
SMOKE_TIMEOUT_S = 10800.0
# NCCL transport of the benchmark stage: InfiniBand off (socket transport),
# no GPUDirect RDMA, P2P over NVLink only. bind_runtime puts the same values
# into every case runtime so calibration and trace runs measure this transport.
BENCHMARK_TRANSPORT = MappingProxyType(
    {
        "NCCL_IB_DISABLE": "1",
        "NCCL_IGNORE_DISABLED_P2P": "1",
        "NCCL_NET_GDR_LEVEL": "0",
        "NCCL_NET_GDR_READ": "0",
        "NCCL_P2P_LEVEL": "NVL",
    }
)
# Measurement settings v5 requires explicitly. The values are those of the
# two-server tuning example inputs, where every calibrating sketch used
# a 0.1 fit-error limit, 1200 s per measurement job and a 2 s cleanup reserve.
CALIBRATION_MAX_RELATIVE_FIT_ERROR = 0.1
MEASUREMENT_JOB_TIMEOUT_S = 1200
CLEANUP_RESERVE_S = 2
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
    if is_foreign_home_path(resolved):
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


def _document_text(document: Mapping) -> str:
    return json.dumps(document, indent=1, sort_keys=True) + "\n"


def _read_document(path: Path, label: str) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise SemanticError("{} is unreadable".format(label)) from error
    if not isinstance(value, dict):
        raise SemanticError("{} must be a JSON object".format(label))
    return value


def _request_sha256(parts: Sequence[Tuple[str, str]]) -> str:
    """Digest of labelled request inputs given as (label, sha256 hex)."""
    digest = hashlib.sha256()
    for label, value in parts:
        digest.update(label.encode("ascii"))
        digest.update(b"\x00")
        digest.update(bytes.fromhex(value))
    return digest.hexdigest()


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _case_input_sha256(
    case: ExperimentCase,
    atom_path: Path,
    runtime_text: str,
) -> str:
    """Identity of a case request: its files plus the bound runtime."""
    return _request_sha256(
        (
            ("topology", _file_sha256(case.topology_path)),
            ("sketch", _file_sha256(case.sketch_path)),
            ("atom", _file_sha256(atom_path)),
            ("runtime", _text_sha256(runtime_text)),
        )
    )


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
    # Optional v5 ``runtime.paths`` entries bound into every case runtime
    # (``vericcl_driver``/``vericcl_native`` are required by the online
    # stages of the solve and smoke runs).
    runtime_paths: Mapping[str, str] = MappingProxyType({})

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
        paths = _string_mapping(self.runtime_paths, "runtime_paths")
        unknown = sorted(set(paths).difference(_RUNTIME_PATH_FIELDS))
        if unknown:
            raise SemanticError(
                "runtime_paths has unknown entries: {}".format(
                    ", ".join(unknown)
                )
            )
        object.__setattr__(
            self,
            "runtime_paths",
            MappingProxyType(
                {
                    key: str(_absolute_path(value, "runtime_paths." + key))
                    for key, value in sorted(paths.items())
                }
            ),
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
    runtime_paths = payload["runtime_paths"]
    if not isinstance(runtime_paths, dict):
        raise SemanticError("V100 config runtime_paths must be an object")
    missing = [name for name in _REQUIRED_RUNTIME_PATHS if name not in runtime_paths]
    if missing:
        raise SemanticError(
            "V100 config runtime_paths requires {}".format(", ".join(missing))
        )
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
        runtime_paths=runtime_paths,
    )


def write_hostfile(path: Path, rank_count: int, *, hostnames=None) -> None:
    if rank_count not in HOSTS:
        raise SemanticError("unsupported V100 rank count")
    names = tuple(hostnames) if hostnames is not None else tuple(host for host, _ in HOSTS[rank_count])
    if len(names) != 2 or len(set(names)) != 2 or not all(
            isinstance(name, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name) for name in names):
        raise SemanticError("hostfile requires two distinct explicit hostnames")
    atomic_replace_text(Path(path), ''.join('{} slots={}\n'.format(name, rank_count // 2) for name in names))


def _configured_hosts(config):
    previous = None
    for count, path in ((8, config.hostfile_8), (16, config.hostfile_16)):
        lines = path.read_text(encoding='ascii').splitlines()
        hosts = []
        for line in lines:
            match = re.fullmatch(r'([A-Za-z0-9][A-Za-z0-9_.-]*) slots=([0-9]+)', line)
            if not match or int(match.group(2)) != count // 2:
                raise SemanticError('V100 hostfile content is invalid')
            hosts.append(match.group(1))
        if len(hosts) != 2 or len(set(hosts)) != 2:
            raise SemanticError('V100 hostfile requires two distinct hosts')
        if hosts[0] != config.remote_host:
            raise SemanticError('remote_host must be first in each hostfile')
        if previous is not None and tuple(hosts) != previous:
            raise SemanticError('V100 hostfiles must preserve the same rank host order')
        previous = tuple(hosts)
    return previous


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
    _configured_hosts(config)
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
    for name in _REQUIRED_RUNTIME_PATHS:
        binary = config.runtime_paths.get(name)
        if binary is None:
            raise SemanticError(
                "V100 preflight requires runtime_paths.{}".format(name)
            )
        if not Path(binary).is_file() or not os.access(binary, os.X_OK):
            raise SemanticError(
                "V100 preflight runtime_paths.{} is not executable".format(
                    name
                )
            )
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
        write_hostfile(path, 2, hostnames=_configured_hosts(config))
        stager.upload(path, path)
    remote_executor = SshStagingCommandExecutor(
        delegate=delegate,
        stager=stager,
        remote_host=config.remote_host,
        path_policy=policy,
    )
    collector = RemoteTraceCollector(stager=stager)
    return remote_executor, collector


def _deployment_environment(config: V100ExperimentConfig) -> dict:
    """Allowlisted v5 runtime environment contributed by the deployment.

    Mirrors the MPI/NCCL transport of the benchmark stage
    (:func:`build_mpi_prefix`, :func:`build_performance_environment`) within
    the keys a v5 runtime may carry.
    """
    environment = {
        key: value
        for key, value in config.environment.items()
        if key in ENVIRONMENT_ALLOWLIST
    }
    for key, value in BENCHMARK_TRANSPORT.items():
        environment.setdefault(key, value)
    environment["PATH"] = "{}:{}".format(
        config.mpi_launcher.parent,
        config.environment.get("PATH", "/usr/bin:/bin"),
    )
    environment["LD_LIBRARY_PATH"] = ":".join(
        value
        for value in (
            str(config.msccl_library_path),
            str(config.mpi_launcher.parent.parent / "lib"),
            config.environment.get("LD_LIBRARY_PATH", ""),
        )
        if value
    )
    environment.setdefault("OMPI_MCA_pml", "ob1")
    environment.setdefault("OMPI_MCA_btl", "tcp,self,vader")
    interface = config.environment.get("VERICCL_MPI_TCP_IF_INCLUDE")
    if interface:
        environment.setdefault("OMPI_MCA_btl_tcp_if_include", interface)
    return environment


def bind_runtime(
    config: V100ExperimentConfig,
    runtime: Mapping,
) -> dict:
    """Bind a logical v5 runtime document to the configured deployment.

    Logical hosts are mapped, in order of their first (lowest) rank, onto the
    hosts of the configured hostfiles (``remote_host`` first). Environment
    keys set by the case win over deployment defaults, so an experiment's
    explicit ``NCCL_BUFFSIZE`` is preserved. ``runtime.paths`` gets
    ``mpi_bin`` and the configured ``runtime_paths``.
    """
    if not isinstance(config, V100ExperimentConfig):
        raise SemanticError("bind_runtime requires V100ExperimentConfig")
    document = copy.deepcopy(dict(runtime))
    rank_map = document.get("rank_map")
    if not isinstance(rank_map, list) or not rank_map or not all(
        isinstance(entry, dict)
        and isinstance(entry.get("rank"), int)
        and isinstance(entry.get("host"), str)
        for entry in rank_map
    ):
        raise SemanticError("case runtime rank_map is invalid")
    logical = []
    for entry in sorted(rank_map, key=lambda item: item["rank"]):
        if entry["host"] not in logical:
            logical.append(entry["host"])
    hosts = _configured_hosts(config)
    if len(logical) > len(hosts):
        raise SemanticError(
            "case runtime uses more hosts than the V100 hostfiles provide"
        )
    mapping = dict(zip(logical, hosts))
    document["rank_map"] = [
        dict(entry, host=mapping[entry["host"]]) for entry in rank_map
    ]
    case_environment = document.get("environment")
    if not isinstance(case_environment, dict):
        raise SemanticError("case runtime environment is invalid")
    environment = _deployment_environment(config)
    environment.update(case_environment)
    document["environment"] = dict(sorted(environment.items()))
    paths = dict(document.get("paths") or {})
    paths["mpi_bin"] = str(config.mpi_launcher.parent)
    paths.update(config.runtime_paths)
    document["paths"] = dict(sorted(paths.items()))
    return document


def _bound_runtime_text(
    config: V100ExperimentConfig,
    runtime_path: Path,
) -> str:
    return _document_text(
        bind_runtime(config, _read_document(runtime_path, "case runtime"))
    )


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def _execute_run(
    execute: Callable,
    request: PrimaryRequest,
    context: RunContext,
):
    """Run one v5 request and finalize it (cf. vericcl.cli.lifecycle)."""
    try:
        result = execute(request, context)
        if not isinstance(result, RunResult):
            raise SemanticError("v5 executor returned an invalid result")
    except StorageFailure:
        raise
    except Exception as error:
        emit_event(
            context,
            "execution_exception",
            {"message": str(error), "traceback": traceback.format_exc()},
        )
        result = RunResult(
            False,
            "failed",
            stop_reason="exception",
            failure_category="internal",
            optional_failures=(str(error),),
        )
    return result, finalize_run(context, result)


def _start_run(
    project_root: Path,
    case_id: str,
    files: Mapping[str, Path],
    *,
    purpose: str,
    run_id: Optional[str],
):
    inputs = load_request(dict(files), purpose=purpose)
    context = RunContext.create(
        project_root,
        case_id,
        run_id or _run_id(),
        input_files={
            name + ".json": path for name, path in files.items()
        },
        environment=dict(inputs.request_config["runtime"]["environment"]),
    )
    return PrimaryRequest(inputs, purpose=purpose), context


def _require_completed(result: RunResult, finalized, context: RunContext, label: str) -> None:
    if result.execution != "completed" or not finalized.complete:
        details = tuple(result.optional_failures) + tuple(finalized.errors)
        raise SemanticError(
            "{} run did not complete (execution={}, stop_reason={}, "
            "exit_code={}): {}{}".format(
                label,
                result.execution,
                result.stop_reason,
                finalized.exit_code,
                context.root,
                "".join("\n" + str(item) for item in details),
            )
        )


def accepted_xml(run_root: Path) -> Path:
    """The non-empty accepted schedule XML recorded by a v5 run."""
    root = Path(run_root).resolve()
    payload = _read_document(root / "best" / "accepted.json", "accepted schedule")
    candidate = payload.get("candidate")
    relative = candidate.get("xml") if isinstance(candidate, dict) else None
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise SemanticError("accepted schedule names no XML")
    xml = (root / relative).resolve()
    if not xml.is_relative_to(root):
        raise SemanticError("accepted schedule XML escapes the run directory")
    if not xml.is_file() or xml.stat().st_size <= 0:
        raise SemanticError("accepted schedule XML is missing or empty")
    return xml


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
    # A VeriCCL XML runs with the explicit NCCL_BUFFSIZE of the case's v5
    # runtime it was synthesized for; baselines keep the fixed 2 MiB.
    buffsize = (
        case.nccl_buffsize_bytes
        if selected_source is XmlSource.VERICCL
        else 2 * 1024 * 1024
    )
    environment = {
        "PATH": str(config.mpi_launcher.parent) + ':' + config.environment.get('PATH', os.environ.get('PATH', '/usr/bin:/bin')),
        "LD_LIBRARY_PATH": ':'.join((str(config.msccl_library_path),
            str(config.mpi_launcher.parent.parent / 'lib'),
            config.environment.get('LD_LIBRARY_PATH', os.environ.get('LD_LIBRARY_PATH', '')))),
        "CUDA_VISIBLE_DEVICES": ",".join(
            str(rank) for rank in range(visible_count)
        ),
        "MSCCL_XML_FILES": str(xml),
        "NCCL_ALGO": "MSCCL,RING",
        "NCCL_BUFFSIZE": str(buffsize),
        "NCCL_DEBUG": "INFO",
        "NCCL_PROTO": "Simple",
        **BENCHMARK_TRANSPORT,
    }
    if config.environment.get('NCCL_SOCKET_IFNAME'):
        environment['NCCL_SOCKET_IFNAME'] = config.environment['NCCL_SOCKET_IFNAME']
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
    ]
    interface = config.environment.get("VERICCL_MPI_TCP_IF_INCLUDE")
    if interface:
        prefix.extend(("-mca", "btl_tcp_if_include", interface))
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
    destination.mkdir(parents=True, exist_ok=True)
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
        atomic_replace_text(error_path, "{}\n".format(error))
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


def _failure_record(
    task_id: str,
    input_sha256: str,
    command: Tuple[str, ...],
    started: str,
    error: Exception,
    directory: Path,
) -> TaskRecord:
    directory.mkdir(parents=True, exist_ok=True)
    error_path = directory / "runner-error.log"
    atomic_replace_text(error_path, "{}\n".format(error))
    return TaskRecord(
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


def solve_case(
    case: ExperimentCase,
    config: V100ExperimentConfig,
    *,
    atom_path: Optional[Path] = None,
    execute: Callable[[PrimaryRequest, RunContext], RunResult] = run_primary,
    state_store: Optional[ExperimentStateStore] = None,
    run_id: Optional[str] = None,
) -> TaskRecord:
    """Solve one case with the v5 pipeline (purpose ``solve``).

    The run's project root is ``<experiment_root>/runs/<task_id>``. The task
    passes only when the run completed and ``best/accepted.json`` names an
    existing non-empty XML; that XML is recorded as ``log_path`` with its
    digest as ``output_sha256`` for the benchmark stage. ``execute`` takes
    ``(PrimaryRequest, RunContext)`` and returns a ``RunResult``.
    """
    if not isinstance(case, ExperimentCase):
        raise SemanticError("solve_case requires an ExperimentCase")
    if not isinstance(config, V100ExperimentConfig):
        raise SemanticError("solve_case requires V100ExperimentConfig")
    atom = (
        load_experiment_manifest(
            config.manifest_path,
            repo_root=config.repo_root,
        ).atom_path
        if atom_path is None
        else _absolute_path(atom_path, "atom_path")
    )
    runtime_text = _bound_runtime_text(config, case.runtime_path)
    input_sha256 = _case_input_sha256(case, atom, runtime_text)
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
    project_root = config.experiment_root / "runs" / case.task_id
    try:
        runtime_path = project_root / "inputs" / "runtime.json"
        atomic_replace_text(runtime_path, runtime_text)
        request, context = _start_run(
            project_root,
            case.task_id,
            {
                "topology": case.topology_path,
                "sketch": case.sketch_path,
                "atom": atom,
                "runtime": runtime_path,
            },
            purpose="solve",
            run_id=run_id,
        )
        result, finalized = _execute_run(execute, request, context)
        _require_completed(result, finalized, context, "V100 solve")
        final_xml = accepted_xml(context.root)
        record = TaskRecord(
            task_id=case.task_id,
            status=TaskStatus.PASSED,
            input_sha256=input_sha256,
            output_sha256=_file_sha256(final_xml),
            command=command,
            returncode=0,
            log_path=str(final_xml),
            started_at_utc=started,
            finished_at_utc=_utc_now(),
        )
    except Exception as error:
        record = _failure_record(
            case.task_id,
            input_sha256,
            command,
            started,
            error,
            project_root,
        )
    if state_store is not None:
        state_store.put(record)
    return record


def build_smoke_documents(
    config: V100ExperimentConfig,
    manifest: ExperimentManifest,
) -> dict:
    """Topology, sketch and bound runtime of the 2-rank inter-node smoke.

    The first manifest case supplies the documents. The topology keeps the
    first rank of the first two nodes (gateways preferred) with the directed
    links between them, their shared resources and local costs. The sketch
    calibrates one channel on a 128 MiB message with the case slice size;
    the runtime keeps the case environment (explicit ``NCCL_BUFFSIZE``) and
    the two ranks' logical placement before :func:`bind_runtime`.
    """
    case = manifest.cases[0]
    topology = _read_document(case.topology_path, "smoke topology")
    sketch = _read_document(case.sketch_path, "smoke sketch")
    runtime = _read_document(case.runtime_path, "smoke runtime")
    nodes = sorted(topology.get("nodes", ()), key=lambda node: node["id"])
    if len(nodes) < 2:
        raise SemanticError("V100 smoke requires an inter-node topology")
    selected = tuple(
        min(node.get("gateways") or node["ranks"]) for node in nodes[:2]
    )
    remap = {rank: index for index, rank in enumerate(selected)}
    pairs = {(selected[0], selected[1]), (selected[1], selected[0])}
    links = [
        edge
        for edge in topology["directed_links"]
        if (edge["src"], edge["dst"]) in pairs
    ]
    if len(links) != 2:
        raise SemanticError("V100 smoke ranks lack bidirectional links")
    used = {
        resource for edge in links for resource in edge.get("resources", ())
    }
    smoke_topology = {
        "name": "{} smoke 2x1 inter-node".format(
            topology.get("name", case.topology_name)
        ),
        "ranks": 2,
        "nodes": [
            {"id": 0, "ranks": [0], "gateways": [0]},
            {"id": 1, "ranks": [1], "gateways": [1]},
        ],
        "directed_links": [
            dict(edge, src=remap[edge["src"]], dst=remap[edge["dst"]])
            for edge in links
        ],
        "shared_resources": [
            dict(
                resource,
                links=[
                    [remap[src], remap[dst]]
                    for src, dst in resource.get("links", ())
                    if (src, dst) in pairs
                ],
            )
            for resource in topology.get("shared_resources", ())
            if resource["id"] in used
        ],
        "local_costs": [
            dict(entry, rank=remap[entry["rank"]])
            for entry in topology.get("local_costs", ())
            if entry["rank"] in remap
        ],
    }
    smoke_sketch = copy.deepcopy(sketch)
    hyperparameters = smoke_sketch["hyperparameters"]
    slice_size = hyperparameters.get("slice_size_bytes")
    if not slice_size or SMOKE_TOTAL_SIZE_BYTES % slice_size:
        raise SemanticError("V100 smoke slice size must divide 128 MiB")
    hyperparameters["total_size_bytes"] = SMOKE_TOTAL_SIZE_BYTES
    solver = smoke_sketch.setdefault("solver", {})
    solver["max_channels"] = 1
    solver["channel_candidates"] = [1]
    budgets = smoke_sketch.setdefault("budgets", {})
    budgets["max_online_validation_per_round_s"] = SMOKE_TIMEOUT_S
    budgets["max_online_validation_total_s"] = SMOKE_TIMEOUT_S
    budgets["max_tuning_rounds"] = 0
    calibration = smoke_sketch.setdefault("calibration", {})
    calibration["enabled"] = True
    calibration["candidate_limit"] = 1
    calibration.setdefault("max_relative_fit_error", CALIBRATION_MAX_RELATIVE_FIT_ERROR)
    calibration.setdefault("job_timeout_s", MEASUREMENT_JOB_TIMEOUT_S)
    smoke_sketch["trace_and_measure"] = {"enabled": False}
    placement = {entry["rank"]: entry for entry in runtime["rank_map"]}
    smoke_runtime = copy.deepcopy(runtime)
    smoke_runtime["rank_map"] = [
        dict(placement[rank], rank=remap[rank]) for rank in selected
    ]
    return {
        "topology": smoke_topology,
        "sketch": smoke_sketch,
        "runtime": bind_runtime(config, smoke_runtime),
    }


def smoke(
    config: V100ExperimentConfig,
    *,
    execute: Callable[[PrimaryRequest, RunContext], RunResult] = run_primary,
    state_store: Optional[ExperimentStateStore] = None,
    run_id: Optional[str] = None,
) -> TaskRecord:
    """Calibrate the 2-rank inter-node link with a v5 ``calibrate`` run.

    Passes iff the run completed; the finalized ``run-summary.json`` is the
    task's log and output. Failures keep their message in
    ``<experiment_root>/smoke/runner-error.log``.
    """
    manifest = load_experiment_manifest(
        config.manifest_path,
        repo_root=config.repo_root,
    )
    documents = build_smoke_documents(config, manifest)
    texts = {name: _document_text(value) for name, value in documents.items()}
    input_sha256 = _request_sha256(
        (
            ("topology", _text_sha256(texts["topology"])),
            ("sketch", _text_sha256(texts["sketch"])),
            ("atom", _file_sha256(manifest.atom_path)),
            ("runtime", _text_sha256(texts["runtime"])),
        )
    )
    task_id = SMOKE_TASK_ID
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
    directory = config.experiment_root / "smoke"
    try:
        files = {}
        for name in ("topology", "sketch", "runtime"):
            files[name] = directory / "inputs" / (name + ".json")
            atomic_replace_text(files[name], texts[name])
        files["atom"] = manifest.atom_path
        request, context = _start_run(
            directory,
            task_id,
            files,
            purpose="calibrate",
            run_id=run_id,
        )
        result, finalized = _execute_run(execute, request, context)
        _require_completed(result, finalized, context, "V100 smoke calibration")
        summary = context.root / "run-summary.json"
        record = TaskRecord(
            task_id=task_id,
            status=TaskStatus.PASSED,
            input_sha256=input_sha256,
            output_sha256=_file_sha256(summary),
            command=command,
            returncode=0,
            log_path=str(summary),
            started_at_utc=started,
            finished_at_utc=_utc_now(),
        )
    except Exception as error:
        record = _failure_record(
            task_id,
            input_sha256,
            command,
            started,
            error,
            directory,
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
    config: V100ExperimentConfig,
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
        and record.input_sha256
        == _case_input_sha256(
            case,
            atom_path,
            _bound_runtime_text(config, case.runtime_path),
        )
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
            config,
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
        if arguments.stage in {"summarize", "all"}:
            write_report(config.experiment_root)
    except VeriCCLError as error:
        print("VeriCCL V100 experiment failed: {}".format(error))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
