from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Callable, Dict, Tuple

from vericcl.errors import SemanticError
from vericcl.semantics.collective import CollectiveKind
from vericcl.verification.online.model import (
    NcclTestMeasurement,
    NcclTestRequest,
    NcclTestRun,
)


_BINARY_NAMES = {
    CollectiveKind.BROADCAST: "broadcast_perf",
    CollectiveKind.REDUCE: "reduce_perf",
    CollectiveKind.ALL_GATHER: "all_gather_perf",
    CollectiveKind.ALL_REDUCE: "all_reduce_perf",
    CollectiveKind.ALL_TO_ALL: "alltoall_perf",
    CollectiveKind.REDUCE_SCATTER: "reduce_scatter_perf",
}

_OPTION_PATTERN = re.compile(r"(?<!\w)-[A-Za-z](?!\w)")


def _binary(request: NcclTestRequest) -> str:
    name = _BINARY_NAMES[request.kind]
    if request.binary_directory is None:
        return name
    return str(Path(request.binary_directory) / name)


def build_nccl_tests_command(
    request: NcclTestRequest,
) -> Tuple[str, ...]:
    return _build_nccl_tests_command(request, warmup=5, checks=1)


def build_nccl_tests_trace_command(
    request: NcclTestRequest,
) -> Tuple[str, ...]:
    return _build_nccl_tests_command(request, warmup=0, checks=0)


def _build_nccl_tests_command(
    request: NcclTestRequest,
    *,
    warmup: int,
    checks: int,
) -> Tuple[str, ...]:
    if not isinstance(request, NcclTestRequest):
        raise SemanticError("request must be an NcclTestRequest")
    size = str(request.message_size_bytes)
    command = [
        _binary(request),
        "-g",
        str(request.gpus_per_process),
        "-b",
        size,
        "-e",
        size,
        "-w",
        str(warmup),
        "-n",
        "20",
        "-c",
        str(checks),
        "-d",
        request.datatype,
    ]
    if request.reduction_op is not None:
        command.extend(("-o", request.reduction_op))
    if request.root is not None:
        command.extend(("-r", str(request.root)))
    return tuple(command)


def _integer(value: str, field: str, *, positive: bool) -> int:
    try:
        result = int(value, 0)
    except ValueError as error:
        raise SemanticError("{} is not an integer".format(field)) from error
    if result < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        raise SemanticError("{} must be {}".format(field, qualifier))
    return result


def _number(value: str, field: str, *, positive: bool) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise SemanticError("{} is not numeric".format(field)) from error
    minimum_valid = result > 0.0 if positive else result >= 0.0
    if not math.isfinite(result) or not minimum_valid:
        qualifier = "positive" if positive else "non-negative"
        raise SemanticError(
            "{} must be finite and {}".format(field, qualifier)
        )
    return result


def _measurement(values: Tuple[str, ...]) -> NcclTestMeasurement:
    if len(values) != 4:
        raise SemanticError("nccl-tests measurement must contain four fields")
    measurement = NcclTestMeasurement(
        time_us=_number(values[0], "nccl-tests time", positive=True),
        algorithm_bandwidth_gbps=_number(
            values[1],
            "nccl-tests algorithm bandwidth",
            positive=False,
        ),
        bus_bandwidth_gbps=_number(
            values[2],
            "nccl-tests bus bandwidth",
            positive=False,
        ),
        wrong_count=_integer(
            values[3],
            "nccl-tests wrong count",
            positive=False,
        ),
    )
    if measurement.wrong_count:
        raise SemanticError("nccl-tests correctness check failed")
    return measurement


def _performance_row(
    fields: Tuple[str, ...],
    expected_bytes: int,
) -> NcclTestRun:
    if len(fields) < 7:
        raise SemanticError("nccl-tests performance row is incomplete")
    message_size = _integer(
        fields[0],
        "nccl-tests message size",
        positive=True,
    )
    if message_size != expected_bytes:
        raise SemanticError(
            "nccl-tests message size does not match the request"
        )
    element_count = _integer(
        fields[1],
        "nccl-tests element count",
        positive=True,
    )
    datatype = fields[2]
    if not datatype:
        raise SemanticError("nccl-tests datatype is missing")
    if len(fields) >= 11:
        prefix = fields[:-8]
        out_of_place = _measurement(fields[-8:-4])
        in_place = _measurement(fields[-4:])
    else:
        prefix = fields[:-4]
        out_of_place = _measurement(fields[-4:])
        in_place = None
    if len(prefix) < 3:
        raise SemanticError("nccl-tests performance row prefix is incomplete")
    return NcclTestRun(
        message_size_bytes=message_size,
        element_count=element_count,
        datatype=datatype,
        metadata_fields=tuple(prefix[3:]),
        out_of_place=out_of_place,
        in_place=in_place,
    )


def parse_nccl_tests_output(
    text: str,
    expected_bytes: int,
) -> Tuple[NcclTestRun, ...]:
    if not isinstance(text, str):
        raise SemanticError("nccl-tests output must be a string")
    if (
        isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes < 1
    ):
        raise SemanticError("expected_bytes must be a positive integer")
    runs = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = tuple(stripped.split())
        if len(fields) < 7:
            continue
        try:
            int(fields[0], 0)
        except ValueError:
            continue
        runs.append(_performance_row(fields, expected_bytes))
    if not runs:
        raise SemanticError("nccl-tests output contains no performance row")
    return tuple(runs)


class NcclTestsHelpValidator:
    def __init__(self) -> None:
        self._supported_by_binary: Dict[str, frozenset[str]] = {}

    def validate(
        self,
        request: NcclTestRequest,
        load_help: Callable[[Tuple[str, ...]], str],
    ) -> None:
        if not isinstance(request, NcclTestRequest):
            raise SemanticError("request must be an NcclTestRequest")
        if not callable(load_help):
            raise SemanticError("load_help must be callable")
        binary = _binary(request)
        supported = self._supported_by_binary.get(binary)
        if supported is None:
            help_text = load_help((binary, "--help"))
            if not isinstance(help_text, str) or not help_text:
                raise SemanticError("nccl-tests help output is empty")
            supported = frozenset(_OPTION_PATTERN.findall(help_text))
            self._supported_by_binary[binary] = supported
        required = frozenset(
            value
            for value in build_nccl_tests_command(request)[1:]
            if value.startswith("-")
        )
        missing = tuple(sorted(required - supported))
        if missing:
            raise SemanticError(
                "nccl-tests binary does not support options: {}".format(
                    ", ".join(missing)
                )
            )
