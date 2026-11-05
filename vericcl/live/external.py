"""Bounded append-only JSONL adapter for explicitly configured telemetry producers."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat

from vericcl.input.loader import _reject_duplicate_keys, _reject_nonfinite_json
from vericcl.live.telemetry import parse_external


class ExternalTelemetryFeed:
    def __init__(self, path, *, group_id, legal_links, max_age_s=60):
        self.path = Path(path)
        if not self.path.is_absolute():
            raise ValueError('external telemetry needs an explicit absolute path')
        self.group, self.links, self.max_age = group_id, set(legal_links), max_age_s
        self.offset = 0; self.identity = None; self.pending = b''; self.seen = {}

    def poll(self, *, version, request_id):
        try:
            descriptor = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except FileNotFoundError:
            if self.identity is not None:
                raise ValueError('external telemetry file disappeared after observation')
            return ()
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
                raise ValueError('external telemetry must be an owned regular file')
            identity = (info.st_dev, info.st_ino)
            if self.identity is not None and (self.identity != identity or info.st_size < self.offset):
                raise ValueError('external telemetry file was replaced or truncated')
            self.identity = identity
            os.lseek(descriptor, self.offset, os.SEEK_SET)
            # Read at most 64 KiB per poll, and retain an incomplete final line.
            data = os.read(descriptor, 65537)
            self.offset += len(data); self.pending += data
        finally:
            os.close(descriptor)
        now = datetime.now(timezone.utc).timestamp()
        self.seen = {key: value for key, value in self.seen.items() if now - value[1] <= self.max_age}
        result = []
        while b'\n' in self.pending:
            line, self.pending = self.pending.split(b'\n', 1)
            if not line or len(line) > 65536:
                raise ValueError('external telemetry records must contain 1 byte to 64 KiB')
            value = json.loads(line.decode('utf-8'), object_pairs_hook=_reject_duplicate_keys,
                               parse_constant=_reject_nonfinite_json)
            event = parse_external(value, group_id=self.group, legal_links=self.links,
                                   current_version=version, request_id=request_id)
            stamp = datetime.fromisoformat(value['observed_at'].replace('Z', '+00:00')).timestamp()
            if not -5 <= now - stamp <= self.max_age:
                raise ValueError('external telemetry timestamp is expired or in the future')
            digest = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            key = (value['source'], value['event_id'])
            # Identity is group-wide; changing source must not reuse an event id.
            prior = next((v for k, v in self.seen.items() if k[1] == value['event_id']), None)
            if prior:
                if prior[0] != digest:
                    raise ValueError('external telemetry event identity reused with different contents')
                continue
            if len(self.seen) >= 4096:
                raise ValueError('external telemetry exceeds 4096 recent event identities')
            self.seen[key] = (digest, stamp); result.append(event)
        if len(self.pending) > 65536:
            raise ValueError('external telemetry record exceeds 64 KiB')
        return tuple(result)
