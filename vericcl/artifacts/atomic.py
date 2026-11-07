"""Durable atomic no-replace file publication.

Kept free of the report/sidecar stack so that lightweight helpers (run
contexts, remote measurement helpers, owned solver processes) can publish
evidence without importing the whole artifact writer. ``vericcl.artifacts.writer``
re-exports these functions unchanged.
"""
from __future__ import annotations

import os
from pathlib import Path
import uuid

from vericcl.errors import SemanticError


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    if not isinstance(path, Path):
        raise SemanticError("atomic write path must be a Path")
    if not isinstance(data, bytes):
        raise SemanticError("atomic write data must be bytes")
    if os.path.lexists(path):
        raise FileExistsError("artifact already exists: {}".format(path))
    temporary = path.with_name(
        ".{}.{}.tmp".format(path.name, uuid.uuid4().hex)
    )
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        # Atomic no-replace publication also rejects dangling symlinks and a
        # competing writer that creates the destination after the first check.
        os.link(temporary, path)
        temporary.unlink()
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_text(path: Path, text: str) -> None:
    if not isinstance(text, str):
        raise SemanticError("atomic write text must be a string")
    atomic_write_bytes(path, text.encode("utf-8"))
