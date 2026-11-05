"""Disjoint qualified windows, residence limits and explicit emergency events."""
from dataclasses import dataclass, field
import math
import statistics


@dataclass(frozen=True)
class TriggerEvent:
    kind: str
    request_id: int
    version: str
    changes: dict = field(default_factory=dict)
    invalid_links: tuple = ()
    evidence: dict = field(default_factory=dict)


class LiveTrigger:
    def __init__(self, *, window_requests=8, consecutive_windows=2, relative_change_threshold=.1,
                 min_residence_requests=32, baseline_max_cv=.05):
        if any(type(v) is not int or v < 1 for v in (window_requests, consecutive_windows)):
            raise ValueError('live windows must contain positive integer request counts')
        if type(min_residence_requests) is not int or min_residence_requests < 0:
            raise ValueError('live residence must be a nonnegative request count')
        if not 0 < relative_change_threshold < 1 or not 0 <= baseline_max_cv <= .05:
            raise ValueError('invalid live threshold or baseline stability bound')
        self.window_size, self.required = window_requests, consecutive_windows
        self.threshold, self.residence, self.max_cv = relative_change_threshold, min_residence_requests, baseline_max_cv
        self.version = None; self.last_request = 0; self.successes = 0; self.last_switch = None
        self.window = []; self.baseline = {}; self.streaks = {}; self.latest_changes = {}

    def switched(self, version, *, successful_request):
        self.version = version; self.last_switch = self.successes
        self.window.clear(); self.baseline.clear(); self.streaks.clear(); self.latest_changes.clear()
        self.last_request = max(self.last_request, successful_request)

    def rebase_model(self):
        # A drift ratio applies once to the model tied to its baseline. Start
        # fresh windows after applying it, avoiding repeated multiplication.
        self.window.clear(); self.baseline.clear(); self.streaks.clear(); self.latest_changes.clear()

    def observe_request(self, summary):
        if summary.request_id <= self.last_request:
            return None
        if self.version is not None and summary.version != self.version:
            raise ValueError('request telemetry belongs to an uncommitted version')
        self.version = summary.version; self.last_request = summary.request_id; self.successes += 1
        if not summary.qualified:
            return None
        self.window.append(summary)
        if len(self.window) < self.window_size:
            return None
        window, self.window = self.window, []
        keys = set.intersection(*(set(item.metrics) for item in window))
        changes = {}
        for key in keys:
            values = [item.metrics[key] for item in window]
            median = statistics.median(values)
            if key not in self.baseline:
                mean = statistics.fmean(values)
                if mean > 0 and statistics.pstdev(values) / mean <= self.max_cv:
                    self.baseline[key] = median
                continue
            delta = median / self.baseline[key] - 1
            changed = abs(delta) >= self.threshold or math.isclose(abs(delta), self.threshold, rel_tol=1e-12, abs_tol=1e-15)
            direction = (1 if delta > 0 else -1) if changed else 0
            previous, count = self.streaks.get(key, (0, 0))
            count = count + 1 if direction and direction == previous else 1 if direction else 0
            self.streaks[key] = (direction, count)
            if count >= self.required:
                changes[key] = delta
        for key in set(self.streaks) - keys:
            self.streaks[key] = (0, 0)
        self.latest_changes = changes
        if changes and (self.last_switch is None or self.successes - self.last_switch >= self.residence):
            return TriggerEvent('performance', summary.request_id, summary.version, changes,
                evidence={'window_request_ids': [s.request_id for s in window], 'baseline': dict(self.baseline),
                          'source': 'software_device_counters', 'disjoint_windows': True})
        return None

    def emergency(self, kind, *, request_id, version, evidence, invalid_links=()):
        if kind not in {'timeout', 'fault'}:
            raise ValueError('invalid emergency kind')
        # A timeout carries no inferred device or link identity.
        return TriggerEvent(kind, request_id, version, invalid_links=tuple(invalid_links) if kind == 'fault' else (), evidence=dict(evidence))
