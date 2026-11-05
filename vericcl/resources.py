"""Locate project and installed case inputs without escaping their case."""

from __future__ import annotations

from pathlib import Path
import sysconfig


def example_input_path(
    case_id: str,
    relative_path: str,
    *,
    project_root: Path | None = None,
) -> Path:
    if (not isinstance(case_id, str) or not case_id or case_id in {".", ".."}
            or "/" in case_id or "\\" in case_id):
        raise ValueError("case_id must be one directory name")
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts or "\\" in str(relative_path):
        raise ValueError("example input path must stay inside its case input")
    if project_root is None:
        checkout = Path(__file__).resolve().parents[1]
        project_root = (checkout if (checkout / "examples").is_dir() else
                        Path(sysconfig.get_path("data")) / "share/vericcl")
    root = Path(project_root).resolve()
    case_root = (root / "examples" / case_id / "input").resolve()
    if not case_root.is_relative_to(root / "examples"):
        raise ValueError("case input path resolves outside project examples")
    result = (case_root / relative).resolve()
    if not result.is_relative_to(case_root):
        raise ValueError("example input path resolves outside its case input")
    if not result.is_file():
        raise FileNotFoundError(result)
    return result
