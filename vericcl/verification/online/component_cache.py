"""Exact-identity calibration reuse; schedule similarity is a separate policy."""
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import fcntl
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import uuid

from vericcl.artifacts.writer import atomic_write_bytes
from vericcl.errors import SemanticError
from vericcl.performance.model import ServiceCost
from vericcl.performance.profiles import RateProfile
from vericcl.verification.search_coverage import digest


@dataclass(frozen=True)
class CalibrationQuery:
    path: Path
    component: str
    signature: str
    now: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    enabled: bool = False
    max_age_s: float | None = None

    def __post_init__(self):
        if not isinstance(self.component, str) or not self.component or not isinstance(self.signature, str) or not self.signature:
            raise ValueError('calibration lookup needs exact component and applicability identities')
        if not isinstance(self.enabled, bool) or not isinstance(self.now, datetime) or self.now.tzinfo is None:
            raise ValueError('calibration lookup needs an explicit policy and aware current time')
        if self.enabled and (isinstance(self.max_age_s, bool) or not isinstance(self.max_age_s, (int, float)) or
                             not math.isfinite(self.max_age_s) or self.max_age_s <= 0):
            raise ValueError('cross-task calibration reuse requires a finite positive original-measurement age limit')


@dataclass(frozen=True)
class ReuseResult:
    status: str
    values: tuple = ()
    reason: str | None = None


def _cache_key(component, signature):
    return digest({'component': component, 'signature': signature})


@lru_cache(maxsize=2)
def _read_index(path, device, inode, modified, changed, size):
    value = json.loads(Path(path).read_text())
    if value.get('schema_version') != 2 or not isinstance(value.get('records'), dict):
        raise ValueError('unknown component cache format')
    return value


def _load(path):
    try:
        stat = path.stat()
    except FileNotFoundError:
        return {'schema_version': 2, 'records': {}}
    return _read_index(str(path.resolve()), stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)


def save_calibration(path, component, signature, values):
    path = Path(path)
    values = tuple(values)
    is_profile = component.startswith('profile:')
    if not values or any(not isinstance(v, RateProfile if is_profile else ServiceCost) or v.source not in {'measured', 'history'} for v in values):
        raise ValueError('only measured calibration components can be cached')
    for value in values:
        if value.measured_at is None:
            raise ValueError('cached calibration requires its original measurement time')
    payload = {'component': component, 'signature': signature, 'kind': 'profile' if is_profile else 'service',
               'values': [v.as_json() if is_profile else asdict(v) for v in values]}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        cached = _load(path)
        cache = dict(cached) | {'records': dict(cached['records'])}
        cache['records'][_cache_key(component, signature)] = {'payload': payload, 'sha256': digest(payload)}
        temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.pending')
        try:
            atomic_write_bytes(temporary, (json.dumps(cache, indent=2, allow_nan=False) + '\n').encode())
            os.replace(temporary, path)
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temporary.unlink(missing_ok=True)


def lookup_calibration(query):
    if not isinstance(query, CalibrationQuery):
        raise ValueError('calibration lookup requires CalibrationQuery')
    if not query.enabled:
        return ReuseResult('disabled')
    try:
        cache = _load(Path(query.path))
        entry = cache['records'].get(_cache_key(query.component, query.signature))
        if entry is None:
            return ReuseResult('miss')
        payload = entry['payload']
        if digest(payload) != entry['sha256'] or payload['component'] != query.component or payload['signature'] != query.signature:
            return ReuseResult('invalid_cache', reason='component identity or checksum mismatch')
        constructor = RateProfile if payload['kind'] == 'profile' else ServiceCost if payload['kind'] == 'service' else None
        if constructor is None:
            raise ValueError('unknown cached value type')
        values = []
        for raw in payload['values']:
            value = constructor(**raw)
            if value.source not in {'measured', 'history'} or value.measured_at is None:
                raise ValueError('cache contains an unmeasured value')
            stamp = datetime.fromisoformat(value.measured_at.replace('Z', '+00:00'))
            age = (query.now - stamp).total_seconds()
            if 0 <= age <= query.max_age_s:
                values.append(replace(value, source='history'))
        return ReuseResult('hit', tuple(values)) if values else ReuseResult('miss', reason='original measurement age is outside the allowed range')
    except (OSError, ValueError, KeyError, TypeError, SemanticError) as error:
        return ReuseResult('invalid_cache', reason=str(error))
