"""Experiment manifests made of v5 request documents.

A manifest (``schema_version`` 2) names one shared v5 atom and an ordered list
of cases. Each case names a v5 topology, a v5 sketch and a v5 runtime
document, all as repository-relative paths::

    {
      "schema_version": 2,
      "atom": "tests/fixtures/experiments-v5/atom.json",
      "cases": [
        {"topology_name": "v100-n2g4", "collective_label": "ag", "size_label": "4m",
         "topology": ".../topology/v100-n2g4.json",
         "sketch": ".../sketch/v100-n2g4/ag-4m.json",
         "runtime": ".../runtime/v100-n2g4/ag-4m.json"}
      ]
    }

The case runtime is the experiment's part of the v5 runtime: a complete,
loadable runtime whose ``rank_map`` uses *logical* host names (one per
topology node, ordered by first rank) and whose environment carries the
explicit ``NCCL_BUFFSIZE`` of the experiment. Deployment details (real host
names, tool paths, MPI/NCCL interface settings) are bound later by the
harness, e.g. :func:`vericcl.experiments.v100.bind_runtime`.

Every case is resolved with :func:`vericcl.input.loader.load_request` while
loading, and its metadata (rank count, nccl-tests message size, slice size,
NCCL buffer size) comes from the resolved v5 inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Tuple

from vericcl.errors import SemanticError, VeriCCLError
from vericcl.input.loader import load_request
from vericcl.semantics.collective import CollectiveKind


MANIFEST_SCHEMA_VERSION = 2
_MANIFEST_FIELDS = frozenset({"schema_version", "atom", "cases"})
_CASE_FIELDS = frozenset(
    {
        "topology_name",
        "collective_label",
        "size_label",
        "topology",
        "sketch",
        "runtime",
    }
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
    """One experiment case: v5 topology, sketch and runtime documents.

    ``message_size_bytes`` is the nccl-tests size (all-gather: per-rank input
    times ranks), ``slice_size_bytes`` the explicit v5 slice size and
    ``nccl_buffsize_bytes`` the explicit ``NCCL_BUFFSIZE`` of the case runtime.
    """

    task_id: str
    topology_name: str
    collective_label: str
    size_label: str
    topology_path: Path
    sketch_path: Path
    runtime_path: Path
    rank_count: int
    message_size_bytes: int
    slice_size_bytes: int
    nccl_buffsize_bytes: int

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
        for field in ("topology_path", "sketch_path", "runtime_path"):
            object.__setattr__(
                self,
                field,
                _path(getattr(self, field), "experiment " + field),
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
        _positive_integer(
            self.nccl_buffsize_bytes,
            "experiment nccl_buffsize_bytes",
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


def case_request_files(case: ExperimentCase, atom_path: Path) -> dict:
    """The v5 request files of a case with its logical (unbound) runtime."""
    return {
        "topology": case.topology_path,
        "sketch": case.sketch_path,
        "atom": Path(atom_path),
        "runtime": case.runtime_path,
    }


def _load_case(root: Path, atom_path: Path, value: object, index: int) -> ExperimentCase:
    field = "manifest case {}".format(index)
    if not isinstance(value, dict) or set(value) != _CASE_FIELDS:
        raise SemanticError("{} fields are invalid".format(field))
    topology_name = _identifier(value["topology_name"], field + " topology_name")
    collective_label = _identifier(value["collective_label"], field + " collective_label")
    if collective_label not in _COLLECTIVE_KINDS:
        raise SemanticError("manifest collective is unsupported")
    size_label = _identifier(value["size_label"], field + " size_label")
    files = {
        "topology": _repository_file(root, value["topology"], field + " topology"),
        "sketch": _repository_file(root, value["sketch"], field + " sketch"),
        "atom": atom_path,
        "runtime": _repository_file(root, value["runtime"], field + " runtime"),
    }
    try:
        resolved = load_request(files, purpose="solve")
    except (VeriCCLError, ValueError) as error:
        raise SemanticError(
            "{} is not a valid v5 request: {}".format(field, error)
        ) from error
    if resolved.collective.kind is not _COLLECTIVE_KINDS[collective_label]:
        raise SemanticError("manifest collective differs from its sketch")
    slice_size = resolved.hyperparameters.slice_size_bytes
    if slice_size is None:
        raise SemanticError("{} sketch must set an explicit slice size".format(field))
    environment = resolved.request_config["runtime"]["environment"]
    return ExperimentCase(
        task_id="{}-{}-{}".format(topology_name, collective_label, size_label),
        topology_name=topology_name,
        collective_label=collective_label,
        size_label=size_label,
        topology_path=files["topology"],
        sketch_path=files["sketch"],
        runtime_path=files["runtime"],
        rank_count=resolved.rank_count,
        message_size_bytes=resolved.size_accounting["nccl_tests_size_bytes"],
        slice_size_bytes=slice_size,
        nccl_buffsize_bytes=int(environment["NCCL_BUFFSIZE"]),
    )


def load_experiment_manifest(
    path: Path,
    *,
    repo_root: Path,
) -> ExperimentManifest:
    """Load a schema_version 2 manifest and resolve every case as v5 inputs."""
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
    if payload["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise SemanticError("experiment manifest schema is unsupported")
    atom_path = _repository_file(root, payload["atom"], "manifest atom")
    raw_cases = payload["cases"]
    if not isinstance(raw_cases, list) or not raw_cases:
        raise SemanticError("manifest cases must be a non-empty list")
    cases = tuple(
        _load_case(root, atom_path, value, index)
        for index, value in enumerate(raw_cases)
    )
    return ExperimentManifest(atom_path=atom_path, cases=cases)
