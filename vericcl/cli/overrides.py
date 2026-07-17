from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional

from vericcl.errors import InputValidationError, SemanticError
from vericcl.input.json_codec import canonical_json
from vericcl.semantics.collective import CollectiveKind


@dataclass(frozen=True)
class SemanticOverrides:
    operator: Optional[str] = None
    total_size_bytes: Optional[int] = None
    slice_size_bytes: Optional[int] = None
    root: Optional[int] = None
    inplace: Optional[bool] = None

    def __post_init__(self) -> None:
        if self.operator is not None:
            try:
                CollectiveKind(self.operator)
            except (TypeError, ValueError) as error:
                raise SemanticError("semantic override operator is invalid") from error
        for field in ("total_size_bytes", "slice_size_bytes"):
            value = getattr(self, field)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise SemanticError(
                    "semantic override {} must be positive".format(field)
                )
        if self.root is not None and (
            isinstance(self.root, bool)
            or not isinstance(self.root, int)
            or self.root < 0
        ):
            raise SemanticError("semantic override root must be non-negative")
        if self.inplace is not None and not isinstance(self.inplace, bool):
            raise SemanticError("semantic override inplace must be boolean")

    @property
    def empty(self) -> bool:
        return all(
            value is None
            for value in (
                self.operator,
                self.total_size_bytes,
                self.slice_size_bytes,
                self.root,
                self.inplace,
            )
        )


def require_input_files(*paths: Path) -> None:
    for raw_path in paths:
        try:
            path = Path(raw_path)
        except TypeError as error:
            raise InputValidationError("input path is invalid") from error
        if not path.is_file():
            raise InputValidationError(
                "required input file is missing: {}".format(path)
            )


def _read_sketch(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise InputValidationError("sketch JSON could not be read") from error
    if not isinstance(payload, dict):
        raise InputValidationError("sketch JSON root must be an object")
    collective = payload.get("collective")
    hyperparameters = payload.get("hyperparameters")
    if not isinstance(collective, dict) or not isinstance(
        hyperparameters,
        dict,
    ):
        raise InputValidationError(
            "sketch collective and hyperparameters must be objects"
        )
    return payload


def _changes(payload: dict, overrides: SemanticOverrides) -> tuple:
    collective = payload["collective"]
    hyperparameters = payload["hyperparameters"]
    values = (
        (collective, "operator", overrides.operator),
        (hyperparameters, "total_size_bytes", overrides.total_size_bytes),
        (hyperparameters, "slice_size_bytes", overrides.slice_size_bytes),
        (collective, "root", overrides.root),
        (collective, "inplace", overrides.inplace),
    )
    return tuple(
        (mapping, field, value)
        for mapping, field, value in values
        if value is not None and mapping.get(field) != value
    )


def resolve_semantic_overrides(
    sketch_path: Path,
    overrides: SemanticOverrides,
    *,
    allow_override: bool,
    output_dir: Path,
) -> Path:
    if not isinstance(overrides, SemanticOverrides):
        raise SemanticError("overrides must be SemanticOverrides")
    if not isinstance(allow_override, bool):
        raise SemanticError("allow_override must be boolean")
    path = Path(sketch_path)
    if overrides.empty:
        return path
    payload = _read_sketch(path)
    changes = _changes(payload, overrides)
    if not changes:
        return path
    if not allow_override:
        fields = ", ".join(sorted(field for _, field, _ in changes))
        raise InputValidationError(
            "CLI semantic values have conflicts with sketch: {}; use "
            "--override-input to apply them".format(fields)
        )
    for mapping, field, value in changes:
        mapping[field] = value
    hyperparameters = payload["hyperparameters"]
    if "input_chunkup" in hyperparameters:
        total = hyperparameters.get("total_size_bytes")
        slice_size = hyperparameters.get("slice_size_bytes")
        if (
            isinstance(total, int)
            and not isinstance(total, bool)
            and isinstance(slice_size, int)
            and not isinstance(slice_size, bool)
            and slice_size > 0
            and total % slice_size == 0
        ):
            hyperparameters["input_chunkup"] = total // slice_size
    destination_dir = Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / "effective-sketch.json"
    if destination.exists():
        raise FileExistsError(
            "effective sketch already exists: {}".format(destination)
        )
    destination.write_text(
        canonical_json(payload) + "\n",
        encoding="utf-8",
    )
    return destination
