#!/usr/bin/env python3
"""Verify the VeriCCL MSCCL trace patch against an untouched source tree."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


RUNTIME_ROOT = Path(__file__).resolve().parents[1]
PATCH_FILE = (
    RUNTIME_ROOT / "patches" / "0001-vericcl-fixed-step-trace.patch"
)
FORMAT_HEADER = RUNTIME_ROOT / "include" / "vericcl_trace_format.h"
METADATA_FILE = RUNTIME_ROOT / "upstream.json"
REQUIRED_FILES = (
    "src/include/msccl.h",
    "src/collectives/device/primitives.h",
    "src/collectives/device/prims_simple.h",
    "src/collectives/device/msccl_interpreter.h",
    "src/init.cc",
)
SCAN_FILES = REQUIRED_FILES[1:]


class RawStepTraceRecord(ctypes.Structure):
    _fields_ = (
        ("rank", ctypes.c_uint32),
        ("tb_id", ctypes.c_uint16),
        ("step_index", ctypes.c_uint16),
        ("endpoint_type", ctypes.c_uint16),
        ("peer", ctypes.c_int16),
        ("channel", ctypes.c_uint16),
        ("iteration", ctypes.c_uint32),
        ("tb_reach", ctypes.c_uint64),
        ("dependency_done", ctypes.c_uint64),
        ("transfer_start", ctypes.c_uint64),
        ("transfer_end", ctypes.c_uint64),
        ("flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    )


def _load_metadata(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise ValueError("unsupported MSCCL metadata schema")
    return value


def _git_head(source_root: Path) -> str:
    completed = subprocess.run(
        ("git", "-C", str(source_root), "rev-parse", "HEAD"),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ValueError("source root must be a Git checkout")
    return completed.stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_reference(source_root: Path, destination: Path) -> None:
    for relative in REQUIRED_FILES:
        source = source_root / relative
        if not source.is_file():
            raise ValueError(f"missing reference file: {source}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    target_header = destination / "src/include/vericcl_trace_format.h"
    shutil.copy2(FORMAT_HEADER, target_header)


def _run_patch(root: Path, *, dry_run: bool) -> None:
    command = ["patch", "--batch", "--forward", "-p1"]
    if dry_run:
        command.append("--dry-run")
    command.extend(("-i", str(PATCH_FILE)))
    completed = subprocess.run(
        command,
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        output = completed.stdout + completed.stderr
        raise RuntimeError(f"patch {'dry run' if dry_run else 'apply'} failed:\n{output}")


def _verify_layout() -> None:
    header = FORMAT_HEADER.read_text(encoding="utf-8")
    if ctypes.sizeof(RawStepTraceRecord) != 64:
        raise ValueError("Python raw trace layout is not 64 bytes")
    if RawStepTraceRecord.tb_reach.offset != 24:
        raise ValueError("Python raw trace timestamp alignment changed")
    if "sizeof(VericclRawStepTraceRecord) == 64" not in header:
        raise ValueError("C raw trace layout assertion is missing")


def _verify_patched_sources(root: Path) -> None:
    combined = "\n".join(
        (root / relative).read_text(encoding="utf-8")
        for relative in SCAN_FILES
    )
    if "MSCCLTRACE|" in combined:
        raise ValueError("patched source still contains per-step trace printf")
    if re.search(r"printf\([^;]*Rank:%d,Bid:%d,Count:%d", combined, re.DOTALL):
        raise ValueError("patched source still contains aggregate timing printf")


def _verify_hashes(root: Path, metadata: dict) -> None:
    for relative, expected in metadata.get("patched_files", {}).items():
        path = root / relative
        if not path.is_file():
            raise ValueError(f"missing patched file: {path}")
        if _sha256(path) != expected:
            raise ValueError(f"patched file hash mismatch: {relative}")


def _verify_pinned_revision(source_root: Path, metadata: dict, key: str) -> None:
    expected = metadata.get(key)
    if expected and _git_head(source_root) != expected:
        raise ValueError(f"source root is not at the pinned MSCCL revision: {expected}")


def verify(source_root: Path, *, patched_tree: bool = False) -> None:
    if not PATCH_FILE.is_file():
        raise ValueError(f"missing patch: {PATCH_FILE}")
    metadata = _load_metadata(METADATA_FILE)
    _verify_layout()
    if patched_tree:
        _verify_pinned_revision(source_root, metadata, "patched_commit")
        _verify_patched_sources(source_root)
        _verify_hashes(source_root, metadata)
        return

    _verify_pinned_revision(source_root, metadata, "upstream_commit")
    with tempfile.TemporaryDirectory(prefix="vericcl-msccl-patch-") as temp:
        root = Path(temp)
        _copy_reference(source_root, root)
        _run_patch(root, dry_run=True)
        _run_patch(root, dry_run=False)
        _verify_patched_sources(root)
        _verify_hashes(root, metadata)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--base-tree", action="store_true")
    modes.add_argument("--patched-tree", action="store_true")
    args = parser.parse_args(argv)
    try:
        verify(args.source_root.resolve(), patched_tree=args.patched_tree)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"verification failed: {error}", file=sys.stderr)
        return 1
    print("verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
