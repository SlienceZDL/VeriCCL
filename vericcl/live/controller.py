"""Owned persistent native sessions and a bounded local command client."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import socket
import struct
import threading
import time
from types import SimpleNamespace

from vericcl.execution.budget import StandaloneLease
from vericcl.input.loader import _reject_duplicate_keys, _reject_nonfinite_json
from vericcl.live.manifest import export_manifest
from vericcl.live.native import NativeScheduleDriver


class LiveClient:
    def __init__(self, path, session_id):
        self.path, self.session_id = str(path), session_id
        self.next_id = 1
        self.pending = None
        self.pending_deadline = None
        self.lock = threading.Lock()

    def _exchange(self, value, deadline):
        def remaining():
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError('live control deadline expired')
            return left
        encoded = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
        if len(encoded) > 4 * 1024 * 1024:
            raise ValueError('live command exceeds 4 MiB')
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(remaining()); connection.connect(self.path)
            connection.settimeout(remaining()); connection.sendall(struct.pack('!I', len(encoded)) + encoded)
            def read(size):
                chunks = []
                while size:
                    connection.settimeout(remaining()); part = connection.recv(size)
                    if not part:
                        raise ConnectionError('native control connection closed without a result')
                    chunks.append(part); size -= len(part)
                return b''.join(chunks)
            length = struct.unpack('!I', read(4))[0]
            if not 0 < length <= 4 * 1024 * 1024:
                raise ValueError('native control frame length invalid')
            response = json.loads(read(length).decode(), object_pairs_hook=_reject_duplicate_keys, parse_constant=_reject_nonfinite_json)
        if not isinstance(response, dict):
            raise ValueError('native control response must be an object')
        return response

    def status(self, timeout_s):
        return self._exchange(dict(session=self.session_id, command_id=0, action='status'), time.monotonic() + timeout_s)

    def command(self, action, timeout_s, **parameters):
        with self.lock:
            request = dict(session=self.session_id, command_id=self.next_id, action=action, **parameters)
            if self.pending is None:
                self.pending, self.pending_deadline = request, time.monotonic() + timeout_s
            elif request != self.pending:
                raise RuntimeError('a previous live command has unresolved outcome')
            response = self._exchange(request, self.pending_deadline)
            return self._accept(response)

    def _accept(self, response):
        if 'error' in response:
            # A rejection is not permission to reuse an uncertain sequence.
            raise RuntimeError(response['error'])
        if response.get('command_id') != self.next_id or 'result' not in response:
            raise RuntimeError('live result has a different command identity')
        self.next_id += 1; self.pending = self.pending_deadline = None
        return response['result']

    def reconcile(self, timeout_s):
        with self.lock:
            if self.pending is None:
                return None
            status = self.status(timeout_s)
            if status.get('last_command_id') == self.next_id:
                return self._accept(status['last_command_reply'])
            raise RuntimeError('native command remains unresolved; no later command may start')


class NativeLiveSession:
    def __init__(self, context, inputs, bundle, *, timeout_s, project_root=None):
        settings = inputs.request_config['sketch']['live']
        if not settings['enabled']:
            raise ValueError('auxiliary live mode must be explicitly enabled')
        self.context, self.inputs, self.bundle = context, inputs, bundle
        self.driver = NativeScheduleDriver(context, inputs, project_root=project_root)
        self.lease = StandaloneLease.after(timeout_s)
        self.settings = settings
        self.timeout_s = timeout_s
        self.cancel = threading.Event(); self.prepared = threading.Event(); self.finished = threading.Event()
        self.payload = self.client = self.result = self.error = None
        self.thread = None

    def _prepared(self, payload):
        self.payload = payload
        self.prepared.set()

    def _run(self):
        try:
            from vericcl.input.schema import LIVE_TIMEOUTS
            self.result = self.driver._launch(self.bundle, self.inputs, launch_id='live-session',
                warmup=0, timed=1, lease=self.lease,
                live=dict(timeout_s=self.timeout_s, timeouts={key: self.settings[key] for key in LIVE_TIMEOUTS}),
                on_prepared=self._prepared, cancel_check=self.cancel.is_set)
        except BaseException as error:
            self.error = error
        finally:
            self.finished.set()

    def start(self):
        if self.thread is not None:
            raise RuntimeError('live session already started')
        self.thread = threading.Thread(target=self._run, name='vericcl-owned-live-session', daemon=False)
        self.thread.start()
        deadline = time.monotonic() + min(self.lease.remaining_seconds(), self.settings['version_prepare_timeout_s'] + self.settings['control_phase_timeout_s'] + 5)
        while time.monotonic() < deadline:
            if self.finished.is_set():
                raise RuntimeError('native live startup failed: ' + str(self.error or self.driver.last_evidence))
            if self.prepared.wait(.02):
                self.client = self.client or LiveClient(self.payload['native_config']['unix_socket'], self.payload['native_config']['session_id'])
                try:
                    status = self.client.status(min(1., max(.01, deadline - time.monotonic())))
                    if status['status'].get('uncertain'):
                        raise RuntimeError('native startup has unresolved ownership')
                    self.context.write_json('live/session.json', dict(root=self.payload['root'], group_id=self.context.task_id,
                        unix_socket=self.client.path, session_id=self.client.session_id, status=status))
                    return self
                except (FileNotFoundError, ConnectionRefusedError, TimeoutError, ConnectionError):
                    pass
        self.cancel.set()
        raise TimeoutError('native live startup exceeded its parent deadline')

    def execute(self):
        timeout = min(self.lease.remaining_seconds(), self.settings['request_prepare_timeout_s'] +
                      self.settings['execution_attempt_timeout_s'] + self.settings['recovery_timeout_s'] + self.settings['control_phase_timeout_s'])
        result = self.client.command('execute', timeout, remaining_s=timeout)
        self.context.write_json('live/request-%06d.json' % (self.client.next_id - 1), result)
        return result

    def prepare_version(self, bundle, inputs, *, lease=None):
        if self.client is None or self.finished.is_set():
            raise RuntimeError('native live session is not active')
        root = Path(self.payload['root']); version = bundle.xml_sha256
        settings = inputs.request_config['sketch']['trace_and_measure']
        trace = settings['enabled']; capacities = None
        if trace:
            from vericcl.verification.online.trace_format import RAW_HEADER_STRUCT, RAW_RECORD_STRUCT
            capacities = {rank: (settings['trace_capacity_bytes'] - RAW_HEADER_STRUCT.size) // RAW_RECORD_STRUCT.size for rank in range(inputs.rank_count)}
        manifest = export_manifest(bundle, inputs, root / 'versions' / version / 'schedule.xml', light_telemetry=True,
            trace_records=capacities, trace_prefix=root / 'versions' / version / 'trace-tmp' if trace else None, live_trace=trace)
        relative = 'incoming-version-' + version + '.json'
        payload = dict(xml=bundle.xml_text, manifest=manifest)
        if (root / relative).exists():
            if json.loads((root / relative).read_text()) != payload:
                raise ValueError('retained version manifest changed')
        else:
            self.context.write_json(str(root.relative_to(self.context.root)) + '/' + relative, payload)
        helper = SimpleNamespace(inputs=self.inputs, run_context=self.context, lease=self.lease)
        deadline = time.monotonic() + min(self.lease.remaining_seconds(), self.settings['version_prepare_timeout_s'])
        if lease is not None:
            deadline = min(deadline, lease.work_deadline)
        hashes = []
        for host in sorted({item['host'] for item in self.payload['job']['rank_map']}):
            if not self.driver.transport._is_local(host):
                process = self.driver.transport._call(helper,
                    ['scp', '-q', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', str(root / relative), host + ':' + str(root / relative)],
                    deadline - time.monotonic(), 'live-version-copy')
                if process.status != 'completed' or not process.cleanup_complete:
                    raise RuntimeError('immutable version distribution failed')
            record = self.driver.transport._helper(helper, host,
                ['stage-version', '--root', str(root), '--payload-file', relative], deadline - time.monotonic(), 'live-version-stage')
            hashes.append(record['manifest_sha256'])
        if len(set(hashes)) != 1:
            raise RuntimeError('prepared version manifest differs across hosts')
        return self.client.command('prepare', max(.01, deadline - time.monotonic()),
            manifest_path=str(root / 'versions' / version / 'manifest.json'), manifest_sha256=hashes[0],
            remaining_s=max(.001, deadline - time.monotonic()))

    def sample(self, bundle, inputs, *, warmup, timed, capture_trace, lease):
        from vericcl.verification.online.remote import checked_job_path
        self.prepare_version(bundle, inputs, lease=lease)
        lease.check(); seconds = min(lease.remaining_seconds(), self.lease.remaining_seconds())
        command = self.client.next_id
        result = self.client.command('sample', seconds, warmup=warmup, timed=timed,
                                     capture_trace=capture_trace, remaining_s=seconds)
        if result.get('sample_complete') is not True or result.get('version_id') != bundle.xml_sha256:
            raise RuntimeError('native boundary validation did not complete: ' + str(result))
        root = Path(self.payload['root']); capture_id = 'command-' + str(command)
        relative = 'live/validation/' + capture_id
        self.context.write_json(relative + '/result.json', result)
        helper = SimpleNamespace(inputs=self.inputs, run_context=self.context, lease=lease)
        collected, identities = {}, {}
        for host in sorted({item['host'] for item in self.payload['job']['rank_map']}):
            lease.check()
            inventory = self.driver.transport._helper(helper, host,
                ['live-capture-inventory', '--root', str(root), '--host', host, '--capture-id', capture_id],
                min(lease.remaining_seconds(), self.lease.remaining_seconds()), 'live-capture-inventory')
            identities[host] = inventory['runtime_identity']
            for name, metadata in inventory['files'].items():
                source = checked_job_path(root, name)
                destination = self.context.artifact(relative + '/collected/' + host + '/' + name)
                destination.parent.mkdir(parents=True, exist_ok=True)
                if self.driver.transport._is_local(host):
                    self.context.write_bytes(str(destination.relative_to(self.context.root)), source.read_bytes())
                else:
                    process = self.driver.transport._call(helper,
                        ['scp', '-q', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', host + ':' + str(source), str(destination)],
                        min(lease.remaining_seconds(), self.lease.remaining_seconds()), 'live-capture-copy')
                    if process.status != 'completed' or not process.cleanup_complete:
                        raise RuntimeError('live capture collection did not complete')
                data = destination.read_bytes()
                if len(data) != metadata['bytes'] or hashlib.sha256(data).hexdigest() != metadata['sha256']:
                    raise RuntimeError('live capture changed during collection')
                collected[host, Path(name).name] = destination
        fingerprint = hashlib.sha256(json.dumps({'hosts': identities,
            'environment': dict(inputs.request_config['runtime']['environment']), 'protocol': 'Simple'}, sort_keys=True).encode()).hexdigest()
        self.context.write_json(relative + '/inventory.json', {'runtime_identity': identities, 'fingerprint': fingerprint})
        return result, collected, fingerprint, self.context.artifact(relative)

    def close(self, *, timeout_s=None):
        seconds = self.settings['close_cleanup_timeout_s'] if timeout_s is None else min(timeout_s, self.settings['close_cleanup_timeout_s'])
        deadline = time.monotonic() + seconds
        if self.thread is None:
            return dict(closed=True, gpu_released=True, process_started=False)
        if not self.finished.is_set() and self.client is not None:
            try:
                if self.client.pending is not None:
                    self.client.reconcile(min(self.settings['control_phase_timeout_s'], max(.01, deadline - time.monotonic())))
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('shared close cleanup deadline expired')
                self.client.command('close', remaining, remaining_s=remaining)
            except (RuntimeError, OSError, TimeoutError):
                self.cancel.set()
        elif not self.finished.is_set():
            self.cancel.set()
        self.thread.join(max(0., deadline - time.monotonic()))
        if not self.finished.is_set():
            self.cancel.set()
            result = dict(closed=False, gpu_released=False, status='unresolved', message='owned launcher has not completed cleanup')
        else:
            evidence = self.driver.last_evidence or {}
            failure = str(self.error) if self.error is not None else evidence.get('error')
            released = evidence.get('cleanup_confirmed') is True and evidence.get('gpu_released') is True
            result = dict(closed=evidence.get('cleanup_confirmed') is True, gpu_released=evidence.get('gpu_released') is True,
                          status=('failed' if failure else 'completed') if released else 'unresolved',
                          error=failure)
        self.context.write_json('live/close.json', result, replace=True)
        return result
