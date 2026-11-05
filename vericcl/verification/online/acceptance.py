"""Preplanned paired performance acceptance, separate from model correctness."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from fractions import Fraction
import math
import secrets
import statistics
from typing import Callable

from vericcl.errors import SemanticError
from vericcl.execution.context import StorageFailure


def _binomial_fraction(wins: int, pairs: int) -> Fraction:
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (wins, pairs)) or pairs < 1 or not 0 <= wins <= pairs:
        raise SemanticError('invalid win count or pair count')
    return Fraction(sum(math.comb(pairs, k) for k in range(wins, pairs + 1)), 2 ** pairs)


def one_sided_binomial(wins: int, pairs: int) -> float:
    return float(_binomial_fraction(wins, pairs))


@dataclass(frozen=True)
class LaunchRequest:
    launch_id: str
    attempt_index: int
    round_index: int
    pair_index: int
    side: str
    warmup: int = 5
    timed: int = 20
    trace_enabled: bool = False
    lease: object = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class LaunchSample:
    time_us: float
    correct: bool
    gpu_released: bool
    trace_enabled: bool
    launch_id: str
    driver_kind: str
    runtime_identity: str = 'unspecified'
    evidence_paths: tuple[str, ...] = ()
    gpu_work_quiescent: bool = False
    resources_retained: bool = False


@dataclass
class AcceptanceContext:
    enabled: bool = False
    driver: Callable | None = None
    seed: int | None = None
    minimum_gain: float = 0.0
    pair_count: int = 20
    lease: object = None
    allow_test_samples: bool = False
    warmup: int = 5
    timed: int = 20
    max_rounds: int = 3
    max_cv: float = .05


@dataclass(frozen=True)
class MeasuredDecision:
    status: str
    accepted: bool = False
    attempt_index: int | None = None
    alpha: float | None = None
    pair_count: int = 0
    selected_round: int | None = None
    wins: int | None = None
    p_value: float | None = None
    median_gain: float | None = None
    issues: tuple[str, ...] = ()
    evidence_kind: str = 'none'


def _evidence_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, dict):
        return {key: _evidence_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_evidence_value(item) for item in value]
    return value


def _cv(values):
    return statistics.pstdev(values) / statistics.fmean(values)


def _validate_sample(sample, request, context):
    if not isinstance(sample, LaunchSample) or isinstance(sample.time_us, bool) or not isinstance(sample.time_us, (int, float)) or not math.isfinite(sample.time_us) or sample.time_us <= 0:
        raise SemanticError('performance launch did not return a finite positive sample')
    work_complete = sample.gpu_released is True or (sample.driver_kind == 'native_msccl_live' and
        sample.gpu_work_quiescent is True and sample.resources_retained is True)
    if sample.correct is not True or not work_complete or sample.trace_enabled is not False or sample.launch_id != request.launch_id:
        raise SemanticError('performance sample has incorrect data, unresolved GPU work, trace overhead or wrong identity')
    if sample.driver_kind not in {'native_msccl', 'native_msccl_live', 'nccl_tests'}:
        if not context.allow_test_samples or sample.driver_kind != 'unit_fixture':
            raise SemanticError('formal measurement requires a native driver')
    if not isinstance(sample.runtime_identity, str) or not sample.runtime_identity or (sample.runtime_identity == 'unspecified' and sample.driver_kind != 'unit_fixture'):
        raise SemanticError('performance sample has no runtime identity')


def compare_measured(incumbent, candidate, ledger, context: AcceptanceContext) -> MeasuredDecision:
    if not isinstance(context.enabled, bool):
        raise SemanticError('trace_and_measure.enabled must be boolean')
    if not context.enabled:
        return MeasuredDecision('disabled')
    seed = secrets.randbits(64) if context.seed is None else context.seed
    plan = ledger.reserve(incumbent, candidate, seed=seed, pair_count=context.pair_count, minimum_gain=context.minimum_gain,
                          warmup=context.warmup, timed=context.timed, max_rounds=context.max_rounds, max_cv=context.max_cv)
    base = dict(attempt_index=plan.attempt_index, alpha=plan.alpha, pair_count=plan.pair_count)
    runtime_identity, source = None, None
    try:
        if not callable(context.driver):
            raise SemanticError('enabled measured acceptance requires a launch driver')
        for round_index, orders in enumerate(plan.orders, 1):
            groups = {'A': [], 'B': []}
            for pair_index, order in enumerate(orders, 1):
                for side in order:
                    if context.lease is not None:
                        context.lease.check()
                    request = LaunchRequest(f'attempt-{plan.attempt_index}-round-{round_index}-pair-{pair_index}-{side}',
                        plan.attempt_index, round_index, pair_index, side, plan.warmup, plan.timed, lease=context.lease)
                    value = context.driver(incumbent if side == 'A' else candidate, request)
                    # Preserve rejected native samples as well as qualified samples.
                    ledger.append(plan, 'sample', dict(request={item.name: getattr(request, item.name) for item in fields(request) if item.name != 'lease'},
                        sample=_evidence_value(asdict(value)) if isinstance(value, LaunchSample) else {'invalid_result_type': type(value).__name__}))
                    _validate_sample(value, request, context)
                    identity = (value.runtime_identity, value.driver_kind)
                    if runtime_identity is not None and identity != runtime_identity:
                        raise SemanticError('runtime identity changed during paired comparison')
                    runtime_identity, source = identity, value.driver_kind
                    groups[side].append(value.time_us)
            cvs = {side: _cv(values) for side, values in groups.items()}
            stable = all(value <= plan.max_cv for value in cvs.values())
            ledger.append(plan, 'round_finished', dict(round_index=round_index, coefficient_of_variation=cvs,
                                                      both_stable=stable, samples_us=groups))
            if not stable:
                continue
            gains = tuple((old - new) / old for old, new in zip(groups['A'], groups['B']))
            wins = sum(value > plan.minimum_gain for value in gains)
            p_value, median_gain = one_sided_binomial(wins, plan.pair_count), statistics.median(gains)
            accepted = median_gain > plan.minimum_gain and _binomial_fraction(wins, plan.pair_count) <= Fraction.from_float(plan.alpha)
            decision = MeasuredDecision('accepted' if accepted else 'no_measured_gain', accepted,
                **base, selected_round=round_index, wins=wins, p_value=p_value, median_gain=median_gain,
                evidence_kind=source)
            ledger.append(plan, 'attempt_finished', asdict(decision))
            return decision
        decision = MeasuredDecision('unstable', **base, evidence_kind=source or 'none')
    except StorageFailure:
        # A partially persisted attempt is still allocated. Never launch a new job after this.
        raise
    except (Exception, KeyboardInterrupt) as error:
        decision = MeasuredDecision('measurement_failed', **base, issues=(type(error).__name__ + ': ' + str(error),),
                                    evidence_kind=source or 'none')
        ledger.append(plan, 'attempt_finished', asdict(decision))
        if isinstance(error, KeyboardInterrupt):
            raise
        return decision
    ledger.append(plan, 'attempt_finished', asdict(decision))
    return decision
