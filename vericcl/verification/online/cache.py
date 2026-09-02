from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

from vericcl.errors import SemanticError
from vericcl.input.json_codec import sha256_json
from vericcl.verification.online.calibration import CalibrationPoint
from vericcl.verification.online.statistics import summarize_runs


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
    benchmark_inplace: bool
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
        if not isinstance(self.benchmark_inplace, bool):
            raise SemanticError("environment.benchmark_inplace must be boolean")
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
    def __init__(self, path: Optional[Path] = None) -> None:
        self._points: Dict[str, CalibrationPoint] = {}
        self._path = None if path is None else Path(path)
        if self._path is not None and self._path.exists():
            with self._locked():
                self._load()

    @contextmanager
    def _locked(self):
        if self._path is None:
            yield
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self._path.with_name(self._path.name + ".lock")
        try:
            with lock_path.open("a+", encoding="utf-8") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError as error:
            raise SemanticError(
                "calibration cache lock could not be acquired"
            ) from error

    def _load(self) -> None:
        assert self._path is not None
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
            if (
                not isinstance(payload, dict)
                or payload.get("schema_version") != 1
                or not isinstance(payload.get("points"), dict)
            ):
                raise ValueError("invalid cache schema")
            points = {}
            for raw_digest, raw_point in payload["points"].items():
                digest = _digest(raw_digest, "cache point digest")
                if not isinstance(raw_point, dict):
                    raise ValueError("invalid cache point")
                point = CalibrationPoint(
                    concurrency=raw_point["concurrency"],
                    duration_statistics=summarize_runs(
                        raw_point["samples_us"]
                    ),
                    full_wave_count=raw_point["full_wave_count"],
                    tail_transfer_count=raw_point["tail_transfer_count"],
                )
                points[digest] = point
        except (
            KeyError,
            OSError,
            TypeError,
            ValueError,
            SemanticError,
        ) as error:
            raise SemanticError("calibration cache file is invalid") from error
        self._points = points

    def _persist(self) -> None:
        if self._path is None:
            return
        payload = {
            "schema_version": 1,
            "points": {
                digest: {
                    "concurrency": point.concurrency,
                    "samples_us": point.duration_statistics.samples_us,
                    "full_wave_count": point.full_wave_count,
                    "tail_transfer_count": point.tail_transfer_count,
                }
                for digest, point in sorted(self._points.items())
            },
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_name(
            "{}.tmp.{}".format(self._path.name, os.getpid())
        )
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(payload, sort_keys=True, indent=2) + "\n"
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self._path)
            directory_fd = os.open(str(self._path.parent), os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError as error:
            try:
                temporary.unlink()
            except OSError:
                pass
            raise SemanticError(
                "calibration cache file could not be written"
            ) from error

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
        digest = environment_signature_sha256(signature)
        if self._path is None:
            self._points[digest] = point
            return
        with self._locked():
            if self._path.exists():
                self._load()
            self._points[digest] = point
            self._persist()

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
        if self._path is not None:
            with self._locked():
                if self._path.exists():
                    self._load()
        point = self._points.get(environment_signature_sha256(signature))
        if point is not None and not point.stable:
            return None
        return point
