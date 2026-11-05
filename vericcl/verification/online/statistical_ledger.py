"""Task-wide, append-only allocation of measured-acceptance attempts."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random

from vericcl.errors import SemanticError
from vericcl.execution.context import StorageFailure


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


@dataclass(frozen=True)
class AttemptPlan:
    attempt_index: int
    alpha: float
    pair_count: int
    seed: int
    orders: tuple[tuple[str, ...], ...]
    incumbent: str
    candidate: str
    minimum_gain: float
    warmup: int = 5
    timed: int = 20
    max_cv: float = .05

    @classmethod
    def from_record(cls, record):
        values = dict(record['plan'])
        values['orders'] = tuple(tuple(row) for row in values['orders'])
        return cls(**values)


class StatisticalLedger:
    def __init__(self, path, *, task_id, task_alpha=.05, single_attempt_cap=.05,
                 run_context=None):
        for name, value in (('task_alpha', task_alpha), ('single_attempt_cap', single_attempt_cap)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= .05:
                raise SemanticError(name + ' must be finite and in (0, .05]')
        if not isinstance(task_id, str) or not task_id:
            raise SemanticError('statistical ledger needs a task identity')
        self.path = Path(path)
        self.run_context = run_context
        if run_context is not None:
            if task_id != run_context.task_id:
                raise SemanticError('statistical ledger task differs from run')
            self.path = run_context.artifact(self.path.relative_to(run_context.root))
        self.header = dict(event='ledger_created', schema_version=1, task_id=task_id,
                           task_alpha=float(task_alpha), single_attempt_cap=float(single_attempt_cap))
        with self._locked() as (stream, events):
            if not events:
                self._append(stream, events, self.header)
            elif any(events[0].get(key) != value for key, value in self.header.items()):
                raise SemanticError('statistical ledger identity or allocation policy changed')

    @contextmanager
    def _locked(self):
        if self.run_context is not None:
            self.run_context.ensure_writable()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open('a+b') as stream:
                fcntl.flock(stream, fcntl.LOCK_EX)
                stream.seek(0)
                events, previous = [], ''
                for line in stream:
                    if not line.endswith(b'\n'):
                        raise ValueError('incomplete statistical ledger record')
                    record = json.loads(line)
                    digest = record.pop('sha256')
                    if record['previous'] != previous or record['sequence'] != len(events) or hashlib.sha256(_canonical(record)).hexdigest() != digest:
                        raise ValueError('statistical ledger chain is corrupt')
                    record['sha256'] = digest
                    events.append(record)
                    previous = digest
                yield stream, events
        except (OSError, ValueError, KeyError, TypeError) as error:
            if self.run_context is not None:
                self.run_context.fail_storage(error)
            raise StorageFailure('statistical ledger cannot be safely read or written: ' + str(error)) from error

    @staticmethod
    def _append(stream, events, event):
        record = dict(event, sequence=len(events), previous=events[-1]['sha256'] if events else '',
                      recorded_at=datetime.now(timezone.utc).isoformat())
        record['sha256'] = hashlib.sha256(_canonical(record)).hexdigest()
        data = _canonical(record) + b'\n'
        stream.seek(0, os.SEEK_END)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
        descriptor = os.open(Path(stream.name).parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        events.append(record)

    def events(self):
        with self._locked() as (_, events):
            return tuple(events)

    def attempts(self):
        return tuple(AttemptPlan.from_record(event) for event in self.events() if event['event'] == 'attempt_reserved')

    def reserve(self, incumbent, candidate, *, seed, pair_count=20, minimum_gain=0.0,
                warmup=5, timed=20, max_rounds=3, max_cv=.05):
        from dataclasses import asdict
        if not all(isinstance(value, str) and value for value in (incumbent, candidate)) or incumbent == candidate:
            raise SemanticError('measured comparison needs two distinct candidate identities')
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise SemanticError('measurement seed must be a nonnegative integer')
        if isinstance(pair_count, bool) or not isinstance(pair_count, int) or pair_count < 2 or pair_count % 2:
            raise SemanticError('pair count must be even and at least two')
        if isinstance(minimum_gain, bool) or not isinstance(minimum_gain, (int, float)) or not math.isfinite(minimum_gain) or not 0 <= minimum_gain < 1:
            raise SemanticError('minimum gain must be finite and in [0, 1)')
        if type(warmup) is not int or warmup < 0 or type(timed) is not int or timed < 1 or type(max_rounds) is not int or not 1 <= max_rounds <= 3:
            raise SemanticError('invalid warmup, timed iterations or stability-round limit')
        if isinstance(max_cv, bool) or not isinstance(max_cv, (int, float)) or not math.isfinite(max_cv) or not 0 < max_cv <= .05:
            raise SemanticError('stability CV threshold must be in (0, .05]')
        with self._locked() as (stream, events):
            index = 1 + sum(event['event'] == 'attempt_reserved' for event in events)
            alpha = min(self.header['single_attempt_cap'], self.header['task_alpha'] / (index * (index + 1)))
            if alpha <= 0:
                raise SemanticError('attempt significance level is below representable range')
            # Increase N only for the pre-data all-win feasibility check. Even N preserves balance.
            required = math.ceil(-math.log2(alpha))
            pair_count = max(pair_count, required + required % 2)
            while math.ldexp(1.0, -pair_count) > alpha:
                pair_count += 2
            actual_seed = int.from_bytes(hashlib.sha256(_canonical([seed, index, self.header['task_id']])).digest()[:8], 'big')
            generator = random.Random(actual_seed)
            orders = []
            for _ in range(max_rounds):
                order = ['AB'] * (pair_count // 2) + ['BA'] * (pair_count // 2)
                generator.shuffle(order)
                orders.append(tuple(order))
            plan = AttemptPlan(index, alpha, pair_count, actual_seed, tuple(orders), incumbent, candidate, float(minimum_gain), warmup, timed, float(max_cv))
            self._append(stream, events, dict(event='attempt_reserved', plan=asdict(plan)))
            return plan

    def append(self, plan, event, payload):
        from dataclasses import asdict
        if event not in {'sample', 'round_finished', 'attempt_finished'}:
            raise SemanticError('unknown statistical ledger event')
        with self._locked() as (stream, events):
            plans = [item for item in events if item['event'] == 'attempt_reserved' and item['plan']['attempt_index'] == plan.attempt_index]
            if len(plans) != 1 or AttemptPlan.from_record(plans[0]) != plan:
                raise SemanticError('attempt plan differs from the reserved plan')
            if any(item['event'] == 'attempt_finished' and item['attempt_index'] == plan.attempt_index for item in events):
                raise SemanticError('statistical attempt is already closed')
            self._append(stream, events, dict(event=event, attempt_index=plan.attempt_index, payload=payload))
