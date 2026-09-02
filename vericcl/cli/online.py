from __future__ import annotations

import math
import os
import hashlib
from pathlib import Path
from time import monotonic as _monotonic
from typing import Mapping

from vericcl.errors import InputValidationError, SemanticError
from vericcl.semantics.collective import CollectiveKind
from vericcl.topology.loader import load_topology
from vericcl.topology.model import (
    DirectedLink,
    LinkKey,
    Topology,
)
from vericcl.artifacts.writer import atomic_write_text
from vericcl.verification.online.cache import (
    CalibrationCache,
    EnvironmentSignature,
)
from vericcl.verification.online.calibration import (
    CalibrationRequest,
    calibration_point_from_trace,
)
from vericcl.verification.online.calibration_xml import (
    build_calibration_benchmark,
)
from vericcl.verification.online.model import NcclTestRequest
from vericcl.verification.online.pipeline import (
    CalibrationPlan,
    OnlineContext,
    OnlineStageStatus,
    run_online_validation,
)
from vericcl.verification.online.runner import (
    SubprocessCommandExecutor,
    process_environment,
)


_DATATYPES = {
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

_CALIBRATION_LINK_CLASSES = frozenset({"intra_node", "inter_node"})
_CALIBRATION_PATH_VARIABLES = (
    "CUDA_VISIBLE_DEVICES",
    "LD_LIBRARY_PATH",
    "NCCL_IB_DISABLE",
    "NCCL_IB_HCA",
    "NCCL_NET_GDR_LEVEL",
    "NCCL_P2P_LEVEL",
    "NCCL_P2P_DISABLE",
    "NCCL_SHM_DISABLE",
    "NCCL_SOCKET_IFNAME",
    "NCCL_TOPO_FILE",
    "VERICCL_MPI_HOSTFILE",
    "VERICCL_MPI_LAUNCHER",
    "VERICCL_MSCCL_BUILD_DIR",
    "VERICCL_NCCL_TESTS_BUILD_DIR",
)
_CALIBRATION_FILE_VARIABLES = frozenset(
    {"NCCL_TOPO_FILE", "VERICCL_MPI_HOSTFILE"}
)


def _required(environment: Mapping[str, str], name: str) -> str:
    value = environment.get(name)
    if not isinstance(value, str) or not value:
        raise InputValidationError(
            "online validation requires {}".format(name)
        )
    return value


def _boolean(environment: Mapping[str, str], name: str) -> bool:
    value = environment.get(name, "0")
    if value not in {"0", "1"}:
        raise InputValidationError("{} must be zero or one".format(name))
    return value == "1"


def _nonnegative_float(
    environment: Mapping[str, str],
    name: str,
    default: str,
) -> float:
    try:
        value = float(environment.get(name, default))
    except ValueError as error:
        raise InputValidationError("{} must be numeric".format(name)) from error
    if not math.isfinite(value) or value < 0.0:
        raise InputValidationError("{} must be non-negative".format(name))
    return value


def _positive_integer(
    environment: Mapping[str, str],
    name: str,
    default: str,
) -> int:
    try:
        value = int(environment.get(name, default), 10)
    except (TypeError, ValueError) as error:
        raise InputValidationError(
            "{} must be a positive integer".format(name)
        ) from error
    if value < 1:
        raise InputValidationError(
            "{} must be a positive integer".format(name)
        )
    return value


def _calibration_link_class(environment: Mapping[str, str]) -> str:
    value = _required(environment, "VERICCL_CALIBRATION_LINK_CLASS")
    if value not in _CALIBRATION_LINK_CLASSES:
        raise InputValidationError(
            "VERICCL_CALIBRATION_LINK_CLASS has an invalid link class"
        )
    return value


def _representative_topology(topology: Topology, link_class: str) -> Topology:
    matching = tuple(
        (key, edge)
        for key, edge in topology.links.items()
        if (
            topology.node_membership[key.src_rank]
            == topology.node_membership[key.dst_rank]
        )
        == (link_class == "intra_node")
    )
    if not matching:
        raise InputValidationError(
            "topology has no link for the requested calibration link class"
        )
    _, representative = matching[0]
    key = LinkKey(0, 1)
    edge = DirectedLink(
        key=key,
        max_channels=representative.max_channels,
        performance=representative.performance,
        resource_ids=(),
    )
    inter_node = link_class == "inter_node"
    return Topology(
        rank_count=2,
        links={key: edge},
        shared_resources={},
        node_membership={0: 0, 1: 1 if inter_node else 0},
        gateways=frozenset({0, 1}) if inter_node else frozenset(),
        warnings=(),
    )


def _calibration_path_variables(
    environment: Mapping[str, str],
) -> tuple:
    result = []
    names = set(_CALIBRATION_PATH_VARIABLES)
    names.update(
        name
        for name in environment
        if name.startswith(("NCCL_", "UCX_"))
    )
    for name in sorted(names):
        value = environment.get(name)
        if not isinstance(value, str) or not value:
            continue
        if name in _CALIBRATION_FILE_VARIABLES:
            path = Path(value).expanduser().resolve()
            try:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError as error:
                raise InputValidationError(
                    "calibration signature file is unreadable: {}".format(
                        name
                    )
                ) from error
            value = "{}#sha256={}".format(path, digest)
        result.append((name, value))
    return tuple(result)


def build_online_context_factory(
    environment: Mapping[str, str] = os.environ,
):
    values = dict(environment)
    msccl_directory = Path(_required(values, "VERICCL_MSCCL_BUILD_DIR"))
    nccl_tests_directory = Path(
        _required(values, "VERICCL_NCCL_TESTS_BUILD_DIR")
    )
    clock_sync_binary = Path(
        _required(values, "VERICCL_CLOCK_SYNC_BINARY")
    )
    inter_node = _boolean(values, "VERICCL_ONLINE_INTER_NODE")
    operator_mpi_launcher = Path(
        _required(values, "VERICCL_MPI_LAUNCHER")
    )
    operator_mpi_hostfile = (
        Path(_required(values, "VERICCL_MPI_HOSTFILE"))
        if inter_node
        else None
    )
    uncertainty = _nonnegative_float(
        values,
        "VERICCL_MAX_CLOCK_UNCERTAINTY_US",
        "10.0",
    )
    calibration_link_class = _calibration_link_class(values)
    force_recalibrate = _boolean(values, "VERICCL_FORCE_RECALIBRATE")
    gpu_model = _required(values, "VERICCL_GPU_MODEL")
    nic_model = _required(values, "VERICCL_NIC_MODEL")
    cuda_version = _required(values, "VERICCL_CUDA_VERSION")
    nccl_version = _required(values, "VERICCL_NCCL_VERSION")
    msccl_version = _required(values, "VERICCL_MSCCL_VERSION")
    trace_record_capacity = _positive_integer(
        values,
        "VERICCL_TRACE_RECORDS",
        "1048576",
    )
    calibration_inter_node = calibration_link_class == "inter_node"
    calibration_mpi_launcher = operator_mpi_launcher
    calibration_mpi_hostfile = operator_mpi_hostfile
    if calibration_inter_node and calibration_mpi_launcher is None:
        calibration_mpi_launcher = Path(
            _required(values, "VERICCL_MPI_LAUNCHER")
        )
    if calibration_inter_node and calibration_mpi_hostfile is None:
        calibration_mpi_hostfile = Path(
            _required(values, "VERICCL_MPI_HOSTFILE")
        )
    executor = SubprocessCommandExecutor()
    try:
        calibration_cache = CalibrationCache(
            Path(_required(values, "VERICCL_CALIBRATION_CACHE_PATH"))
        )
    except SemanticError as error:
        raise InputValidationError(str(error)) from error

    def factory(
        artifact,
        schedule,
        inputs,
        xml_path,
        traces_dir,
        tuning_requested,
        timeout_s,
        calibration_enabled=True,
    ) -> OnlineContext:
        spec = inputs.collective
        request = NcclTestRequest(
            kind=spec.kind,
            message_size_bytes=(
                inputs.rank_count * inputs.hyperparameters.total_size_bytes
                if spec.kind is CollectiveKind.ALL_GATHER
                else inputs.hyperparameters.total_size_bytes
            ),
            datatype=_DATATYPES.get(spec.datatype, spec.datatype),
            reduction_op=spec.reduction_op,
            root=spec.root,
            inplace=spec.inplace,
            binary_directory=str(nccl_tests_directory),
            gpus_per_process=1,
        )
        calibration_plan = None
        if calibration_enabled:
            topology = load_topology(inputs)
            calibration_topology = _representative_topology(
                topology,
                calibration_link_class,
            )
            edge = calibration_topology.links[LinkKey(0, 1)]
            maximum = min(
                inputs.hyperparameters.max_calibration_channels,
                edge.max_channels,
            )
            calibration_request = CalibrationRequest(
                link_class=calibration_link_class,
                slice_size_bytes=inputs.hyperparameters.slice_size_bytes,
                max_calibration_channels=maximum,
                datatype=_DATATYPES.get(spec.datatype, spec.datatype),
            )
            count = calibration_request.benchmark_slice_count
            effective = (
                0
                if count is None
                else min(maximum, 32, count)
            )
            signatures = tuple(
                EnvironmentSignature(
                    link_class=calibration_link_class,
                    topology_signature=topology.isomorphism_signature,
                    gpu_model=gpu_model,
                    nic_model=nic_model,
                    cuda_version=cuda_version,
                    nccl_version=nccl_version,
                    msccl_version=msccl_version,
                    protocol="Simple",
                    slice_size_bytes=calibration_request.slice_size_bytes,
                    benchmark_size_bytes=(
                        calibration_request.benchmark_size_bytes
                    ),
                    concurrency=concurrency,
                    nccl_buffsize_bytes=(
                        2 * calibration_request.slice_size_bytes
                    ),
                    chunk_steps=4,
                    slice_steps=4,
                    path_variables=_calibration_path_variables(values),
                )
                for concurrency in range(1, effective + 1)
            )
            calibration_started = None

            def measure_point(signature):
                nonlocal calibration_started
                now = _monotonic()
                if calibration_started is None:
                    calibration_started = now
                remaining_budget = float(timeout_s) - (
                    now - calibration_started
                )
                if remaining_budget <= 0.0:
                    raise SemanticError(
                        "calibration wall-clock budget expired"
                    )
                remaining_points = max(
                    1,
                    effective - signature.concurrency + 1,
                )
                benchmark = build_calibration_benchmark(
                    calibration_request,
                    calibration_topology,
                    concurrency=signature.concurrency,
                )
                directory = Path(traces_dir) / "calibration-{}-k{:02d}".format(
                    calibration_link_class,
                    signature.concurrency,
                )
                directory.mkdir(parents=True, exist_ok=True)
                calibration_xml = directory / "benchmark.xml"
                atomic_write_text(
                    calibration_xml,
                    benchmark.artifact.xml_text,
                )
                calibration_context = OnlineContext(
                    artifact=benchmark.artifact,
                    schedule=benchmark.schedule,
                    inputs=benchmark.inputs,
                    request=NcclTestRequest(
                        kind=CollectiveKind.BROADCAST,
                        message_size_bytes=(
                            calibration_request.benchmark_size_bytes
                        ),
                        datatype=calibration_request.datatype,
                        reduction_op=None,
                        root=0,
                        inplace=False,
                        binary_directory=str(nccl_tests_directory),
                        gpus_per_process=1,
                    ),
                    xml_paths=(calibration_xml,),
                    msccl_library_path=msccl_directory,
                    executor=executor,
                    environment=process_environment(values),
                    inter_node=calibration_inter_node,
                    mpi_launcher=calibration_mpi_launcher,
                    mpi_hostfile=(
                        calibration_mpi_hostfile
                        if calibration_inter_node
                        else None
                    ),
                    trace_file_prefix=directory / "msccl-step",
                    clock_sync_binary=clock_sync_binary,
                    max_clock_uncertainty_us=uncertainty,
                    online_tuning_requested=False,
                    timeout_s=remaining_budget / remaining_points,
                    trace_record_capacity=trace_record_capacity,
                )
                measured = run_online_validation(calibration_context)
                if (
                    measured.release_status is not OnlineStageStatus.PASSED
                    or measured.online_operator_validation
                    is not OnlineStageStatus.PASSED
                    or measured.trace_analysis is None
                ):
                    raise SemanticError(
                        measured.failure_message
                        or "calibration online validation failed"
                    )
                return calibration_point_from_trace(
                    calibration_request,
                    signature.concurrency,
                    measured.trace_analysis,
                )

            calibration_plan = CalibrationPlan(
                request=calibration_request,
                alpha_us=edge.performance.alpha_us,
                signatures=signatures,
                cache=calibration_cache,
                measure_point=measure_point,
                force_recalibrate=(
                    force_recalibrate
                    or inputs.hyperparameters.force_recalibrate
                ),
            )
        return OnlineContext(
            artifact=artifact,
            schedule=schedule,
            inputs=inputs,
            request=request,
            xml_paths=(Path(xml_path),),
            msccl_library_path=msccl_directory,
            executor=executor,
            environment=process_environment(values),
            inter_node=inter_node,
            mpi_launcher=operator_mpi_launcher,
            mpi_hostfile=operator_mpi_hostfile,
            trace_file_prefix=Path(traces_dir) / "msccl-step",
            clock_sync_binary=clock_sync_binary,
            max_clock_uncertainty_us=uncertainty,
            calibration_plan=calibration_plan,
            online_tuning_requested=tuning_requested,
            timeout_s=timeout_s,
            trace_record_capacity=trace_record_capacity,
        )

    return factory
