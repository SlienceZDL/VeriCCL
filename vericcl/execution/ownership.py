"""Durable per-GPU ownership shared by calibration and schedule execution."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import time


class GPUOwnershipRegistry:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _locked(self):
        with (self.root / 'ownership.lock').open('a+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = self.root / 'ownership.json'
            values = json.loads(path.read_text()) if path.exists() else {}
            if not isinstance(values, dict):
                raise RuntimeError('invalid GPU ownership registry')
            yield values

    def _write(self, values, event):
        path = self.root / 'ownership.json'
        temporary = self.root / ('ownership.%d.partial' % os.getpid())
        with temporary.open('x') as stream:
            json.dump(values, stream, sort_keys=True, indent=2)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        with (self.root / 'transitions.jsonl').open('a') as stream:
            stream.write(json.dumps(event | {'time_ns': time.time_ns()}, sort_keys=True) + '\n')
            stream.flush(); os.fsync(stream.fileno())
        descriptor = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _identities(devices, owner):
        devices = tuple(sorted(set(devices)))
        if not devices or not all(isinstance(d, str) and d for d in devices) or not isinstance(owner, str) or not owner:
            raise ValueError('GPU ownership requires concrete device UUIDs and owner identity')
        return devices

    def acquire(self, devices, owner, evidence):
        devices = self._identities(devices, owner)
        with self._locked() as values:
            for device in devices:
                previous = values.get(device)
                if previous and previous['state'] != 'released' and previous['owner'] != owner:
                    raise RuntimeError('GPU ownership retained by another run: ' + device)
                if previous and previous['owner'] == owner and previous['state'] == 'quarantined':
                    raise RuntimeError('GPU ownership retained after unresolved execution: ' + device)
            for device in devices:
                values[device] = dict(owner=owner, state='active', evidence=evidence)
            self._write(values, dict(action='acquire', devices=devices, owner=owner, evidence=evidence))

    def finish(self, devices, owner, *, process_clean, gpu_released):
        devices = self._identities(devices, owner)
        with self._locked() as values:
            if any(values.get(device, {}).get('owner') != owner for device in devices):
                raise RuntimeError('cannot release another GPU owner')
            state = 'released' if process_clean is True and gpu_released is True else 'quarantined'
            for device in devices:
                values[device] = values[device] | dict(state=state, process_clean=bool(process_clean), gpu_released=bool(gpu_released))
            self._write(values, dict(action='finish', devices=devices, owner=owner, state=state,
                                     process_clean=bool(process_clean), gpu_released=bool(gpu_released)))

    def snapshot(self):
        with self._locked() as values:
            return values
