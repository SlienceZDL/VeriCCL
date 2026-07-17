from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from vericcl.errors import SemanticError
from vericcl.input.json_codec import sha256_json
from vericcl.verification.online.calibration import CalibrationPoint


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise SemanticError("{} must be a non-empty string".format(field))
    return value


def _positive_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SemanticError("{} must be a positive integer".format(field))
    return value


def _digest(value: object, field: str) -> str:
    result = _identifier(value, field)
    if len(result) != 64:
        raise SemanticError("{} must be a SHA-256 digest".format(field))
    try:
        int(result, 16)
    except ValueError as error:
        raise SemanticError(
            "{} must be a SHA-256 digest".format(field)
        ) from error
    return result.lower()


@dataclass(frozen=True)
class EnvironmentSignature:
    link_class: str
    topology_signature: str
    gpu_model: str
    nic_model: str
    cuda_version: str
    nccl_version: str
    msccl_version: str
    protocol: str
    slice_size_bytes: int
    benchmark_size_bytes: int
    concurrency: int
    nccl_buffsize_bytes: int
    chunk_steps: int
    slice_steps: int
    path_variables: Tuple[Tuple[str, str], ...]

    def __post_init__(self) -> None:
        for field in (
            "link_class",
            "gpu_model",
            "nic_model",
            "cuda_version",
            "nccl_version",
            "msccl_version",
            "protocol",
        ):
            _identifier(getattr(self, field), "environment.{}".format(field))
        object.__setattr__(
            self,
            "topology_signature",
            _digest(
                self.topology_signature,
                "environment.topology_signature",
            ),
        )
        for field in (
            "slice_size_bytes",
            "benchmark_size_bytes",
            "concurrency",
            "nccl_buffsize_bytes",
            "chunk_steps",
            "slice_steps",
        ):
            _positive_integer(
                getattr(self, field),
                "environment.{}".format(field),
            )
        try:
            paths = tuple(self.path_variables)
        except TypeError as error:
            raise SemanticError(
                "environment.path_variables must be iterable"
            ) from error
        normalized = []
        for item in paths:
            if not isinstance(item, tuple) or len(item) != 2:
                raise SemanticError(
                    "environment path variable entries must be pairs"
                )
            key, value = item
            normalized.append(
                (
                    _identifier(key, "environment path variable name"),
                    _identifier(value, "environment path variable value"),
                )
            )
        names = tuple(key for key, _ in normalized)
        if len(names) != len(set(names)):
            raise SemanticError(
                "environment path variable names must be unique"
            )
        object.__setattr__(
            self,
            "path_variables",
            tuple(sorted(normalized)),
        )


def environment_signature_sha256(signature: EnvironmentSignature) -> str:
    if not isinstance(signature, EnvironmentSignature):
        raise SemanticError("signature must be an EnvironmentSignature")
    return sha256_json(signature)


class CalibrationCache:
    def __init__(self) -> None:
        self._points: Dict[str, CalibrationPoint] = {}

    def put(
        self,
        signature: EnvironmentSignature,
        point: CalibrationPoint,
    ) -> None:
        if not isinstance(signature, EnvironmentSignature):
            raise SemanticError("cache signature is invalid")
        if not isinstance(point, CalibrationPoint):
            raise SemanticError("cache point is invalid")
        if signature.concurrency != point.concurrency:
            raise SemanticError(
                "cache signature and point concurrency differ"
            )
        self._points[environment_signature_sha256(signature)] = point

    def get(
        self,
        signature: EnvironmentSignature,
        *,
        force_recalibrate: bool = False,
    ) -> Optional[CalibrationPoint]:
        if not isinstance(signature, EnvironmentSignature):
            raise SemanticError("cache signature is invalid")
        if not isinstance(force_recalibrate, bool):
            raise SemanticError("force_recalibrate must be a boolean")
        if force_recalibrate:
            return None
        return self._points.get(environment_signature_sha256(signature))
