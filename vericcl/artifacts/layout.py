from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from vericcl.errors import SemanticError
from vericcl.input.models import ResolvedInput


_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def format_scale(size_bytes: int) -> str:
    if isinstance(size_bytes, bool) or not isinstance(size_bytes, int):
        raise SemanticError("size_bytes must be an integer")
    if size_bytes < 1:
        raise SemanticError("size_bytes must be positive")
    for suffix, divisor in (
        ("TiB", 1024**4),
        ("GiB", 1024**3),
        ("MiB", 1024**2),
        ("KiB", 1024),
    ):
        if size_bytes >= divisor and size_bytes % divisor == 0:
            return "{}{}".format(size_bytes // divisor, suffix)
    return "{}B".format(size_bytes)


@dataclass(frozen=True)
class RunLayout:
    root: Path
    resolved_input: Path
    summary: Path
    schedules: Path
    reports: Path
    traces: Path
    artifact_prefix: str

    def __post_init__(self) -> None:
        paths = (
            self.root,
            self.resolved_input,
            self.summary,
            self.schedules,
            self.reports,
            self.traces,
        )
        if not all(isinstance(path, Path) for path in paths):
            raise SemanticError("run layout paths must be Path values")
        if (
            not isinstance(self.artifact_prefix, str)
            or not self.artifact_prefix
            or "optimal" in self.artifact_prefix.lower()
        ):
            raise SemanticError("run layout artifact_prefix is invalid")


def _validate_run_id(run_id: object) -> str:
    if (
        not isinstance(run_id, str)
        or not _RUN_ID.fullmatch(run_id)
        or "optimal" in run_id.lower()
    ):
        raise SemanticError("run_id is invalid")
    return run_id


def create_run_layout(
    base: Path,
    inputs: ResolvedInput,
    run_id: str,
) -> RunLayout:
    if not isinstance(inputs, ResolvedInput):
        raise SemanticError("inputs must be a ResolvedInput")
    try:
        output_base = Path(base)
    except TypeError as error:
        raise SemanticError("base must be path-like") from error
    validated_run_id = _validate_run_id(run_id)
    operator = inputs.collective.kind.value
    scale = format_scale(inputs.hyperparameters.total_size_bytes)
    prefix = "vericcl_{}_{}".format(operator, scale)
    root = output_base / "{}_{}".format(prefix, validated_run_id)

    output_base.mkdir(parents=True, exist_ok=True)
    if root.exists():
        if not root.is_dir():
            raise FileExistsError("run path already exists and is not a directory")
        if any(root.iterdir()):
            raise FileExistsError(
                "refusing to overwrite non-empty run directory: {}".format(root)
            )
    else:
        root.mkdir()
    schedules = root / "schedules"
    reports = root / "reports"
    traces = root / "traces"
    for path in (schedules, reports, traces):
        path.mkdir()
    return RunLayout(
        root=root,
        resolved_input=root / "resolved-input.json",
        summary=root / "run-summary.json",
        schedules=schedules,
        reports=reports,
        traces=traces,
        artifact_prefix=prefix,
    )
