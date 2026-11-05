"""Append-only UTF-8 event protocol shared by Python and native producers."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time


@dataclass(frozen=True)
class EventScan:
    events: tuple[dict, ...]
    truncated_tail: bool


def read_events(path: Path) -> EventScan:
    raw = Path(path).read_bytes()
    lines = raw.splitlines(keepends=True)
    events = []
    truncated = False
    for index, line in enumerate(lines):
        if index == len(lines) - 1 and not line.endswith(b"\n"):
            truncated = True
            break
        value = json.loads(line)
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            raise ValueError("invalid event record at line " + str(index + 1))
        events.append(value)
    return EventScan(tuple(events), truncated)


def emit_event(context, kind: str, data=None, *, rank=None) -> dict:
    if not isinstance(kind, str) or not kind or len(kind) > 128:
        raise ValueError("invalid event kind")
    if rank is not None and (isinstance(rank, bool) or not isinstance(rank, int) or rank < 0):
        raise ValueError("rank must be a nonnegative integer")
    with context.lock:
        context.ensure_writable()
        context.sequence += 1
        event = {
            "schema_version": 1, "task_id": context.task_id, "run_id": context.run_id,
            "sequence": context.sequence, "time_utc": datetime.now(timezone.utc).isoformat(),
            "monotonic_ns": time.monotonic_ns(), "node": context.node,
            "rank": rank, "kind": kind, "data": context.redactor.value(data or {}),
        }
        encoded = (json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        prefix = "" if rank is None else "rank-" + str(rank) + "."
        text = "{time_utc} {kind} {data}\n".format(**event).encode("utf-8")
        try:
            for name, content in ((prefix + "events.jsonl", encoded), (prefix + "run.log", text)):
                path = context.artifact(name)
                with path.open("ab") as stream:
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
        except OSError as error:
            context.fail_storage(error)
        return event


def emit_native_event(context, record: bytes, *, rank: int) -> dict:
    if len(record) > 1024 * 1024:
        raise ValueError("native event exceeds 1 MiB")
    value = json.loads(record)
    if not isinstance(value, dict) or set(value) - {"schema_version", "kind", "data"}:
        raise ValueError("invalid native event fields")
    if value.get("schema_version") != 1 or not isinstance(value.get("data", {}), dict):
        raise ValueError("invalid native event schema")
    return emit_event(context, value["kind"], value.get("data"), rank=rank)
