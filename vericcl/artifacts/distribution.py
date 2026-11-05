"""Explicit source-package integrity checks; never an execution admission gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


def _source_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("source manifest path must be a nonempty string")
    item = Path(relative)
    if item.is_absolute() or ".." in item.parts or "\\" in relative:
        raise ValueError("source manifest path must remain inside the project")
    result = (root / item).resolve()
    if not result.is_relative_to(root):
        raise ValueError("source manifest path resolves outside the project")
    return result


def verify_source_manifest(project_root: Path) -> tuple[dict[str, str], ...]:
    """Return missing/modified files without changing source or runtime state."""
    root = Path(project_root).resolve()
    manifest_path = _source_path(root, "docs/source-provenance.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("unsupported source manifest schema")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("source manifest files must be an object")
    findings = []
    for relative, digest in sorted(files.items()):
        path = _source_path(root, relative)
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("source manifest checksum must be a SHA-256 hex digest")
        if not path.is_file():
            findings.append({"path": relative, "status": "missing"})
            continue
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != digest:
            findings.append({"path": relative, "status": "modified"})
    return tuple(findings)
