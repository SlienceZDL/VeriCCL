from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Mapping, Optional, Tuple

from lxml import etree
import pytest

from vericcl.semantics.collective import CollectiveKind
from vericcl.verification.online.model import NcclTestRequest
from vericcl.verification.online.runner import (
    NcclTestsRunner,
    SubprocessCommandExecutor,
    process_environment,
)


MIB = 1024 * 1024
CALIBRATION_BYTES = 128 * MIB


@dataclass(frozen=True)
class HardwareConfig:
    msccl_build_directory: Path
    nccl_tests_build_directory: Path
    gpu_count: int
    mpi_launcher: Path
    slice_size_bytes: int
    hostfile: Optional[Path]
    timeout_s: float


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        pytest.skip("{} is required for VeriCCL hardware tests".format(name))
    return value


def _existing_directory(name: str) -> Path:
    path = Path(_required(name))
    if not path.is_dir():
        pytest.skip("{} does not name an existing directory".format(name))
    return path


def _executable(name: str) -> Path:
    path = Path(_required(name))
    if not path.is_file() or not os.access(path, os.X_OK):
        pytest.skip("{} does not name an executable file".format(name))
    return path


def hardware_config(
    *,
    minimum_gpu_count: int,
    require_hostfile: bool,
) -> HardwareConfig:
    if os.environ.get("VERICCL_HARDWARE_ENABLE") != "1":
        pytest.skip(
            "VERICCL_HARDWARE_ENABLE=1 is required for hardware tests"
        )
    try:
        gpu_count = int(_required("VERICCL_GPU_COUNT"))
        slice_size = int(_required("VERICCL_SLICE_SIZE_BYTES"))
        timeout_s = float(os.environ.get("VERICCL_HARDWARE_TIMEOUT_S", "3600"))
    except ValueError:
        pytest.skip("VeriCCL hardware numeric environment is invalid")
    if gpu_count < minimum_gpu_count:
        pytest.skip(
            "VERICCL_GPU_COUNT must be at least {}".format(
                minimum_gpu_count
            )
        )
    if slice_size < 1 or timeout_s <= 0:
        pytest.skip("VeriCCL hardware size or timeout is invalid")
    hostfile = None
    if require_hostfile:
        hostfile = Path(_required("VERICCL_MPI_HOSTFILE"))
        if not hostfile.is_file():
            pytest.skip("VERICCL_MPI_HOSTFILE does not exist")
    return HardwareConfig(
        msccl_build_directory=_existing_directory(
            "VERICCL_MSCCL_BUILD_DIR"
        ),
        nccl_tests_build_directory=_existing_directory(
            "VERICCL_NCCL_TESTS_BUILD_DIR"
        ),
        gpu_count=gpu_count,
        mpi_launcher=_executable("VERICCL_MPI_LAUNCHER"),
        slice_size_bytes=slice_size,
        hostfile=hostfile,
        timeout_s=timeout_s,
    )


def xml_directory(name: str) -> Path:
    return _existing_directory(name)


def parse_xml_contract(path: Path) -> Mapping[str, str]:
    if not path.is_file():
        pytest.skip("required XML is missing: {}".format(path))
    try:
        root = etree.fromstring(path.read_bytes())
    except (OSError, etree.XMLSyntaxError) as error:
        pytest.fail("hardware XML cannot be parsed: {}".format(error))
    return dict(root.attrib)


def runtime_environment(
    config: HardwareConfig,
    xml_path: Path,
    *,
    trace_enabled: bool = False,
) -> Mapping[str, str]:
    values = {
        "NCCL_ALGO": "MSCCL",
        "NCCL_PROTO": "Simple",
        "NCCL_BUFFSIZE": str(2 * config.slice_size_bytes),
        "MSCCL_XML_FILES": str(xml_path),
        "VERICCL_EXPECTED_MSCCL_CHUNKSTEPS": "4",
        "VERICCL_EXPECTED_MSCCL_SLICESTEPS": "4",
        "VERICCL_TRACE_ENABLE": "1" if trace_enabled else "0",
    }
    inherited = os.environ.get("LD_LIBRARY_PATH", "")
    values["LD_LIBRARY_PATH"] = (
        str(config.msccl_build_directory)
        if not inherited
        else str(config.msccl_build_directory) + os.pathsep + inherited
    )
    return process_environment(values)


def launcher_prefix(
    config: HardwareConfig,
    environment: Mapping[str, str],
    *,
    process_count: int,
) -> Tuple[str, ...]:
    command = [
        str(config.mpi_launcher),
        "-np",
        str(process_count),
    ]
    if config.hostfile is not None:
        command.extend(("--hostfile", str(config.hostfile)))
    exported = (
        "LD_LIBRARY_PATH",
        "MSCCL_XML_FILES",
        "NCCL_ALGO",
        "NCCL_BUFFSIZE",
        "NCCL_PROTO",
        "VERICCL_EXPECTED_MSCCL_CHUNKSTEPS",
        "VERICCL_EXPECTED_MSCCL_SLICESTEPS",
        "VERICCL_TRACE_ENABLE",
    )
    for key in exported:
        assert key in environment
        command.extend(("-x", key))
    return tuple(command)


def nccl_request(
    config: HardwareConfig,
    kind: CollectiveKind,
    message_size_bytes: int,
    *,
    inplace: bool,
) -> NcclTestRequest:
    return NcclTestRequest(
        kind=kind,
        message_size_bytes=message_size_bytes,
        datatype=os.environ.get("VERICCL_NCCL_TESTS_DATATYPE", "float"),
        reduction_op=(
            "sum"
            if kind
            in {
                CollectiveKind.REDUCE,
                CollectiveKind.ALL_REDUCE,
                CollectiveKind.REDUCE_SCATTER,
            }
            else None
        ),
        root=(
            0
            if kind in {CollectiveKind.BROADCAST, CollectiveKind.REDUCE}
            else None
        ),
        inplace=inplace,
        binary_directory=str(config.nccl_tests_build_directory),
    )


def runner(
    config: HardwareConfig,
    environment: Mapping[str, str],
    *,
    process_count: int,
) -> NcclTestsRunner:
    return NcclTestsRunner(
        SubprocessCommandExecutor(),
        environment=environment,
        launcher_prefix=launcher_prefix(
            config,
            environment,
            process_count=process_count,
        ),
        timeout_s=config.timeout_s,
    )
