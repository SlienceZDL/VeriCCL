"""Task wall budgets and CPU ownership shared by synthesis and repair."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import threading
import time

from vericcl.errors import SemanticError


class BudgetExhausted(SemanticError):
    pass


class ResourceBusy(SemanticError):
    pass


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise SemanticError(name + ' must be finite and positive')
    return float(value)


def _count(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SemanticError(name + ' must be a positive integer')
    return value


def available_cpu_slots():
    """Respect affinity and Linux cgroup quotas; never expand the caller's quota."""
    slots = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    paths = [Path('/sys/fs/cgroup/cpu.max')]
    try:
        for entry in Path('/proc/self/cgroup').read_text().splitlines():
            hierarchy, controllers, relative = entry.split(':', 2)
            if not controllers:
                current = Path('/sys/fs/cgroup') / relative.lstrip('/')
                while current.is_relative_to('/sys/fs/cgroup'):
                    paths.append(current / 'cpu.max')
                    if current == Path('/sys/fs/cgroup'):
                        break
                    current = current.parent
    except (OSError, ValueError):
        pass
    for path in set(paths):
        try:
            quota, period = path.read_text().split()
            if quota != 'max':
                slots = min(slots, int(quota) // int(period))
        except (OSError, ValueError, ZeroDivisionError):
            pass
    try:
        quota = int(Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us').read_text())
        period = int(Path('/sys/fs/cgroup/cpu/cpu.cfs_period_us').read_text())
        if quota > 0:
            slots = min(slots, quota // period)
    except (OSError, ValueError, ZeroDivisionError):
        pass
    if slots < 1:
        raise ResourceBusy('the effective CPU quota does not provide one whole managed CPU slot')
    return slots


@dataclass
class StandaloneLease:
    deadline: float
    cleanup_reserve_s: float = 0
    clock: object = time.monotonic
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    @classmethod
    def after(cls, seconds, *, cleanup_reserve_s=0, clock=time.monotonic):
        seconds = _positive(seconds, 'lease seconds')
        if not 0 <= cleanup_reserve_s < seconds:
            raise SemanticError('cleanup reserve must be nonnegative and below the lease duration')
        return cls(clock() + seconds, cleanup_reserve_s, clock)

    @property
    def work_deadline(self):
        return self.deadline - self.cleanup_reserve_s

    @property
    def cancelled(self):
        return self._cancel.is_set()

    def cancel(self):
        self._cancel.set()

    def remaining_seconds(self):
        return max(0., self.work_deadline - self.clock())

    def expired(self):
        return self.cancelled or self.remaining_seconds() <= 0

    def check(self):
        if self.expired():
            raise BudgetExhausted('budget cancelled or exhausted')


@dataclass
class BudgetLease:
    manager: object
    stage: str
    started_at: float
    deadline: float
    work_deadline: float
    model: bool = False
    threads: int = 0
    parent: object = None
    closed: bool = False
    cleanup_confirmed: bool = False
    unresolved: bool = False

    @property
    def cancelled(self):
        return self.manager._cancel.is_set() or (self.parent is not None and self.parent.cancelled)

    def remaining_seconds(self):
        return max(0., self.work_deadline - self.manager.clock())

    def expired(self):
        return self.cancelled or self.closed or self.remaining_seconds() <= 0

    def check(self):
        if self.expired():
            raise BudgetExhausted('budget cancelled, closed or exhausted')

    def close(self, *, cleanup_confirmed):
        self.manager._close(self, cleanup_confirmed=cleanup_confirmed)


class BudgetManager:
    def __init__(self, *, stage_limits, cpu_budget, per_model_timeout_s,
                 max_threads_per_model=12, max_parallel_models=1, cleanup_reserve_s,
                 max_tuning_rounds, max_run_wall_s=None, available_cpus=None, clock=time.monotonic):
        self.clock = clock
        self.started_at = clock()
        self.stage_limits = {key: (_positive(pair[0], 'per-round budget'), _positive(pair[1], 'total budget'))
                             for key, pair in stage_limits.items()}
        self.cpu_budget = min(_count(cpu_budget, 'cpu budget'), available_cpu_slots() if available_cpus is None
                              else _count(available_cpus, 'available CPUs'))
        self.max_threads_per_model = _count(max_threads_per_model, 'per-model CPU limit')
        self.max_parallel_models = _count(max_parallel_models, 'parallel model limit')
        self.per_model_timeout_s = _positive(per_model_timeout_s, 'per-model timeout')
        self.cleanup_reserve_s = _positive(cleanup_reserve_s, 'cleanup reserve')
        if isinstance(max_tuning_rounds, bool) or not isinstance(max_tuning_rounds, int) or max_tuning_rounds < 0:
            raise SemanticError('maximum rounds must be a nonnegative integer')
        self.max_tuning_rounds = max_tuning_rounds
        self.max_run_wall_s = None if max_run_wall_s is None else _positive(max_run_wall_s, 'wall limit')
        self.consumed = {key: 0. for key in self.stage_limits}
        self.model_work_s = {key: 0. for key in self.stage_limits}
        self.round_consumed = {key: 0. for key in self.stage_limits}
        self.started_rounds = 0
        self.elapsed_before_resume_s = 0.
        self.cpu_in_use = 0
        self._leases = []
        self._condition = threading.Condition()
        self._cancel = threading.Event()

    def _global_deadline(self):
        return math.inf if self.max_run_wall_s is None else self.started_at + self.max_run_wall_s - self.elapsed_before_resume_s

    def cancel(self):
        self._cancel.set()
        with self._condition:
            self._condition.notify_all()

    def acquire(self, stage, *, model=False, parent=None, threads=None, wait=True):
        with self._condition:
            if stage not in self.stage_limits:
                raise SemanticError('stage has no configured budget: ' + stage)
            if self._cancel.is_set():
                raise BudgetExhausted('task cancelled')
            if not model:
                if parent is not None or any(not item.closed and not item.model and item.stage == stage for item in self._leases):
                    raise ResourceBusy('a stage already has an active root lease')
                now = self.clock()
                per_round, total = self.stage_limits[stage]
                deadline = min(now + min(per_round - self.round_consumed[stage], total - self.consumed[stage]), self._global_deadline())
                work_deadline = deadline - self.cleanup_reserve_s
                if work_deadline <= now:
                    raise BudgetExhausted(stage + ' budget exhausted')
                result = BudgetLease(self, stage, now, deadline, work_deadline)
            else:
                if not any(parent is item for item in self._leases) or parent.model or parent.stage != stage:
                    raise SemanticError('model work needs the matching active parent stage lease')
                requested = min(self.max_threads_per_model, self.cpu_budget) if threads is None else _count(threads, 'requested threads')
                if requested > min(self.max_threads_per_model, self.cpu_budget):
                    raise SemanticError('model thread request exceeds its effective CPU quota')
                while True:
                    parent.check()
                    active = sum(item.model and not item.closed for item in self._leases)
                    if active < self.max_parallel_models and self.cpu_in_use + requested <= self.cpu_budget:
                        break
                    if not wait:
                        raise ResourceBusy('managed CPU/model slots are occupied')
                    self._condition.wait(min(.05, parent.remaining_seconds()))
                now = self.clock()
                deadline = min(now + self.per_model_timeout_s, parent.deadline)
                work_deadline = min(deadline - self.cleanup_reserve_s, parent.work_deadline)
                if work_deadline <= now:
                    raise BudgetExhausted('insufficient model budget after cleanup reservation')
                self.cpu_in_use += requested
                result = BudgetLease(self, stage, now, deadline, work_deadline, True, requested, parent)
            self._leases.append(result)
            return result

    def _close(self, lease, *, cleanup_confirmed):
        with self._condition:
            if not any(lease is item for item in self._leases):
                raise SemanticError('unknown budget lease')
            if lease.closed:
                return
            if not cleanup_confirmed:
                lease.unresolved = True
                return
            if any(not child.closed and child.parent is lease for child in self._leases):
                raise ResourceBusy('child work has not acknowledged exit')
            elapsed = max(0., self.clock() - lease.started_at)
            if lease.model:
                self.cpu_in_use -= lease.threads
                self.model_work_s[lease.stage] += elapsed
            else:
                self.consumed[lease.stage] += elapsed
                self.round_consumed[lease.stage] += elapsed
            lease.closed, lease.cleanup_confirmed, lease.unresolved = True, True, False
            self._condition.notify_all()

    def start_tuning_round(self):
        with self._condition:
            if self._cancel.is_set() or self.started_rounds >= self.max_tuning_rounds:
                raise BudgetExhausted('task cancelled or tuning round limit reached')
            if self.clock() >= self._global_deadline() - self.cleanup_reserve_s:
                raise BudgetExhausted('task wall budget exhausted')
            for stage, (_, total) in self.stage_limits.items():
                if stage != 'initial' and total - self.consumed[stage] <= self.cleanup_reserve_s:
                    raise BudgetExhausted(stage + ' cumulative budget exhausted')
            self.round_consumed = {key: 0. for key in self.stage_limits}
            self.started_rounds += 1
            return self.started_rounds

    def snapshot(self):
        with self._condition:
            return {'consumed': dict(self.consumed), 'model_work_s': dict(self.model_work_s),
                    'round_consumed': dict(self.round_consumed), 'started_rounds': self.started_rounds, 'cpu_in_use': self.cpu_in_use,
                    'elapsed_wall_s': self.elapsed_before_resume_s + max(0., self.clock() - self.started_at),
                    'active_leases': [{'stage': item.stage, 'model': item.model, 'threads': item.threads,
                                       'unresolved': item.unresolved, 'started_at': item.started_at, 'deadline': item.deadline, 'elapsed_s': max(0., self.clock() - item.started_at)}
                                      for item in self._leases if not item.closed]}

    def restore_usage(self, consumed, *, started_rounds, elapsed_wall_s, model_work_s=None, round_consumed=None):
        with self._condition:
            if any(not item.closed for item in self._leases):
                raise SemanticError('cannot restore usage while owned work is active')
            if set(consumed) - self.consumed.keys():
                raise SemanticError('unknown restored budget stage')
            if (isinstance(started_rounds, bool) or not isinstance(started_rounds, int)
                    or started_rounds < self.started_rounds or not math.isfinite(elapsed_wall_s)
                    or elapsed_wall_s < self.elapsed_before_resume_s):
                raise SemanticError('restored usage cannot refund rounds or wall time')
            for key, value in consumed.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < self.consumed[key]:
                    raise SemanticError('restored usage cannot refund consumed stage time')
            model_work_s = model_work_s or {}
            if set(model_work_s) - self.model_work_s.keys():
                raise SemanticError('unknown restored model-work stage')
            for key, value in model_work_s.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < self.model_work_s[key]:
                    raise SemanticError('restored usage cannot refund consumed model time')
            round_consumed = round_consumed or {}
            if set(round_consumed) - self.round_consumed.keys() or any(
                    isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                    or value < self.round_consumed[key] for key, value in round_consumed.items()):
                raise SemanticError('restored usage cannot refund the current round')
            self.round_consumed.update(round_consumed)
            self.model_work_s.update(model_work_s)
            self.consumed.update(consumed)
            self.started_rounds = started_rounds
            self.elapsed_before_resume_s = elapsed_wall_s


def effective_cpu_budget(inputs):
    declared = getattr(inputs, 'request_config', {}).get('sketch', {}).get('solver', {}).get('cpu_budget')
    available = available_cpu_slots()
    return available if declared is None else min(available, _count(declared, 'CPU budget'))
