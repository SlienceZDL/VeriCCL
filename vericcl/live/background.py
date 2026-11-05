"""One cancellable CPU analysis task per group, with coalesced pending events."""
from dataclasses import dataclass
import threading

from vericcl.execution.budget import StandaloneLease


@dataclass(frozen=True)
class BackgroundResult:
    base_version: str
    events: tuple
    value: object
    error: str | None = None


class BackgroundWorker:
    def __init__(self, solve, *, timeout_s, parent_lease=None):
        self.solve, self.timeout_s, self.parent_lease = solve, timeout_s, parent_lease
        self.lock = threading.Lock(); self.thread = None; self.lease = None
        self.pending = {}; self.completed = None; self.closed = False

    @property
    def pending_count(self):
        with self.lock:
            return len(self.pending)

    @property
    def idle(self):
        with self.lock:
            return self.thread is None and self.completed is None and not self.pending

    def offer(self, snapshot, event):
        with self.lock:
            if self.closed:
                raise RuntimeError('background controller closed')
            key = (event.kind, event.version, tuple(sorted(event.invalid_links)))
            self.pending[key] = event
            if self.thread is None and self.completed is None:
                self._start(snapshot)

    def _start(self, snapshot):
        events = tuple(sorted(self.pending.values(), key=lambda e: (e.kind != 'fault', e.request_id)))
        self.pending.clear()
        seconds = min(self.timeout_s, self.parent_lease.remaining_seconds()) if self.parent_lease else self.timeout_s
        lease = StandaloneLease.after(seconds, cleanup_reserve_s=min(.5, seconds / 4)); self.lease = lease
        def run():
            value, error = None, None
            try:
                value = self.solve(dict(snapshot), events, lease)
            except Exception as failure:
                error = str(failure)
            with self.lock:
                self.completed = BackgroundResult(snapshot['version'], events, value, error)
        self.thread = threading.Thread(target=run, name='vericcl-background-repair', daemon=False)
        self.thread.start()

    def take_completed(self):
        with self.lock:
            if self.completed is None:
                return None
            result = self.completed; self.completed = None; self.thread = None; self.lease = None
            return result

    @staticmethod
    def valid_for(result, version):
        return result.error is None and result.base_version == version

    def start_pending(self, snapshot):
        with self.lock:
            if self.pending and self.thread is None and self.completed is None and not self.closed:
                self._start(snapshot)

    def join(self, seconds):
        thread = self.thread
        if thread is not None:
            thread.join(seconds)

    def close(self, seconds):
        with self.lock:
            self.closed = True; self.pending.clear()
            if self.lease is not None:
                self.lease.cancel()
        self.join(seconds)
        return self.thread is None or not self.thread.is_alive()
