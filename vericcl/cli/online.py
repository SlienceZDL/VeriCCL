from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Mapping

from vericcl.errors import InputValidationError
from vericcl.verification.online.model import NcclTestRequest
from vericcl.verification.online.pipeline import OnlineContext
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
    mpi_launcher = (
        Path(_required(values, "VERICCL_MPI_LAUNCHER"))
        if inter_node
        else None
    )
    mpi_hostfile = (
        Path(_required(values, "VERICCL_MPI_HOSTFILE"))
        if inter_node
        else None
    )
    uncertainty = _nonnegative_float(
        values,
        "VERICCL_MAX_CLOCK_UNCERTAINTY_US",
        "10.0",
    )

    def factory(
        artifact,
        schedule,
        inputs,
        xml_path,
        traces_dir,
        tuning_requested,
        timeout_s,
    ) -> OnlineContext:
        spec = inputs.collective
        request = NcclTestRequest(
            kind=spec.kind,
            message_size_bytes=inputs.hyperparameters.total_size_bytes,
            datatype=_DATATYPES.get(spec.datatype, spec.datatype),
            reduction_op=spec.reduction_op,
            root=spec.root,
            inplace=spec.inplace,
            binary_directory=str(nccl_tests_directory),
        )
        return OnlineContext(
            artifact=artifact,
            schedule=schedule,
            inputs=inputs,
            request=request,
            xml_paths=(Path(xml_path),),
            msccl_library_path=msccl_directory,
            executor=SubprocessCommandExecutor(),
            environment=process_environment(values),
            inter_node=inter_node,
            mpi_launcher=mpi_launcher,
            mpi_hostfile=mpi_hostfile,
            trace_file_prefix=Path(traces_dir) / "msccl-step",
            clock_sync_binary=clock_sync_binary,
            max_clock_uncertainty_us=uncertainty,
            online_tuning_requested=tuning_requested,
            timeout_s=timeout_s,
        )

    return factory
