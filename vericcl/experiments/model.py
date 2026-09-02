from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Tuple

from vericcl.errors import SemanticError
from vericcl.input.loader import resolve_inputs
from vericcl.semantics.collective import CollectiveKind


_MANIFEST_FIELDS = frozenset(
    {"schema_version", "atom", "topologies", "collectives", "sizes"}
)
_LABEL_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_COLLECTIVE_KINDS = {
    "ag": CollectiveKind.ALL_GATHER,
    "ar": CollectiveKind.ALL_REDUCE,
}


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not _LABEL_PATTERN.fullmatch(value):
        raise SemanticError("{} is invalid".format(field))
    return value


def _positive_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SemanticError("{} must be a positive integer".format(field))
    return value


def _path(value: object, field: str) -> Path:
    try:
        path = Path(value)
    except TypeError as error:
        raise SemanticError("{} is invalid".format(field)) from error
    if not path.is_absolute():
        raise SemanticError("{} must be absolute".format(field))
    return path.resolve()


def _unique_labels(value: object, field: str) -> Tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise SemanticError("{} must be a non-empty list".format(field))
    labels = tuple(_identifier(item, field) for item in value)
    if len(labels) != len(set(labels)):
        raise SemanticError("{} contains duplicates".format(field))
    return labels


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SemanticError("experiment manifest contains duplicate fields")
        result[key] = value
    return result


def _repository_file(root: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise SemanticError("{} must be a non-empty path".format(field))
    relative = Path(value)
    if relative.is_absolute():
        raise SemanticError("{} must be repository-relative".format(field))
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise SemanticError("{} is outside the repository".format(field))
    if not path.is_file():
        raise SemanticError("{} does not exist".format(field))
    return path


@dataclass(frozen=True)
class ExperimentCase:
    task_id: str
    topology_name: str
    collective_label: str
    size_label: str
    topology_path: Path
    sketch_path: Path
    rank_count: int
    message_size_bytes: int
    slice_size_bytes: int

    def __post_init__(self) -> None:
        _identifier(self.task_id, "experiment task_id")
        _identifier(self.topology_name, "experiment topology_name")
        label = _identifier(
            self.collective_label,
            "experiment collective_label",
        )
        if label not in _COLLECTIVE_KINDS:
            raise SemanticError("experiment collective label is unsupported")
        _identifier(self.size_label, "experiment size_label")
        object.__setattr__(
            self,
            "topology_path",
            _path(self.topology_path, "experiment topology_path"),
        )
        object.__setattr__(
            self,
            "sketch_path",
            _path(self.sketch_path, "experiment sketch_path"),
        )
        _positive_integer(self.rank_count, "experiment rank_count")
        _positive_integer(
            self.message_size_bytes,
            "experiment message_size_bytes",
        )
        _positive_integer(
            self.slice_size_bytes,
            "experiment slice_size_bytes",
        )


@dataclass(frozen=True)
class ExperimentManifest:
    atom_path: Path
    cases: Tuple[ExperimentCase, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "atom_path",
            _path(self.atom_path, "experiment atom_path"),
        )
        try:
            cases = tuple(self.cases)
        except TypeError as error:
            raise SemanticError("experiment cases must be iterable") from error
        if not cases or not all(isinstance(case, ExperimentCase) for case in cases):
            raise SemanticError("experiment cases are invalid")
        task_ids = tuple(case.task_id for case in cases)
        if len(task_ids) != len(set(task_ids)):
            raise SemanticError("experiment task IDs must be unique")
        object.__setattr__(self, "cases", cases)


def load_experiment_manifest(
    path: Path,
    *,
    repo_root: Path,
) -> ExperimentManifest:
    root = _path(repo_root, "repository root")
    manifest_path = _path(path, "experiment manifest path")
    if not manifest_path.is_relative_to(root):
        raise SemanticError("experiment manifest is outside the repository")
    try:
        payload = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except SemanticError:
        raise
    except (OSError, UnicodeError, ValueError) as error:
        raise SemanticError("experiment manifest is unreadable") from error
    if not isinstance(payload, dict):
        raise SemanticError("experiment manifest must be an object")
    if set(payload) != _MANIFEST_FIELDS:
        raise SemanticError("experiment manifest fields are invalid")
    if payload["schema_version"] != 1:
        raise SemanticError("experiment manifest schema is unsupported")
    atom_path = _repository_file(root, payload["atom"], "manifest atom")
    topologies = _unique_labels(payload["topologies"], "manifest topologies")
    collectives = _unique_labels(
        payload["collectives"],
        "manifest collectives",
    )
    if any(value not in _COLLECTIVE_KINDS for value in collectives):
        raise SemanticError("manifest collective is unsupported")
    sizes = _unique_labels(payload["sizes"], "manifest sizes")

    cases = []
    for topology_name in topologies:
        topology_path = _repository_file(
            root,
            "exp/topo/{}.json".format(topology_name),
            "manifest topology",
        )
        for collective_label in collectives:
            for size_label in sizes:
                sketch_path = _repository_file(
                    root,
                    "exp/sketch/{0}/{1}/{1}-{2}.json".format(
                        topology_name,
                        collective_label,
                        size_label,
                    ),
                    "manifest sketch",
                )
                resolved = resolve_inputs(
                    topology_path,
                    sketch_path,
                    atom_path,
                )
                if resolved.collective.kind is not _COLLECTIVE_KINDS[
                    collective_label
                ]:
                    raise SemanticError(
                        "manifest collective differs from its sketch"
                    )
                message_size = resolved.hyperparameters.total_size_bytes
                if resolved.collective.kind is CollectiveKind.ALL_GATHER:
                    message_size *= resolved.rank_count
                cases.append(
                    ExperimentCase(
                        task_id="{}-{}-{}".format(
                            topology_name,
                            collective_label,
                            size_label,
                        ),
                        topology_name=topology_name,
                        collective_label=collective_label,
                        size_label=size_label,
                        topology_path=topology_path,
                        sketch_path=sketch_path,
                        rank_count=resolved.rank_count,
                        message_size_bytes=message_size,
                        slice_size_bytes=(
                            resolved.hyperparameters.slice_size_bytes
                        ),
                    )
                )
    return ExperimentManifest(atom_path=atom_path, cases=tuple(cases))
