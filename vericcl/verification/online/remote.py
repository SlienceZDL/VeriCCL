"""Owned SSH/MPI probe jobs with separate process and GPU acknowledgements.

All mutable remote paths are confined to the executing user home after symlink resolution.
The coordinator and every rank retain separate process birth identities.
"""
import argparse
import base64
from dataclasses import asdict, replace
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import struct
import subprocess
import statistics
import sys
import time
import uuid

from vericcl.execution.budget import BudgetExhausted
from vericcl.execution.context import DIRECTORIES, RunContext, StorageFailure
from vericcl.execution.events import emit_event
from vericcl.execution.processes import ProcessIdentity, ProcessLease, run_owned_process, signal_owned_process_group, _active_group_members, reconcile_owned_process
from vericcl.execution.security import Redactor, is_foreign_home_path, executing_user_home


def checked_job_path(root, relative):
    root = Path(root).resolve()
    item = Path(relative)
    if item.is_absolute() or '..' in item.parts or '\\' in str(relative):
        raise ValueError('remote job path must be relative to its owned root')
    result = (root / item).resolve()
    if not result.is_relative_to(root):
        raise ValueError('remote job path resolves outside its owned root')
    return result


def owned_home_path(path):
    path = Path(path)
    boundary = executing_user_home()
    if not path.is_absolute() or not path.resolve().is_relative_to(boundary):
        raise ValueError('native job writes must remain within the executing user home after resolving symlinks')
    return path.resolve()


def _host(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value):
        raise ValueError('host must be an explicit SSH alias, hostname or IPv4 address')
    return value


def _immutable(path, data):
    path = owned_home_path(path)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('refusing to overwrite another remote job artifact')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def combine_rank_samples(job, artifact, results):
    expected_ranks = {item['rank'] for item in job.rank_map}
    if len(results) != len(expected_ranks) or {r.get('rank') for r in results} != expected_ranks:
        raise ValueError('native sample is missing a rank or contains duplicate ranks')
    endpoints = {}
    for result in results:
        if result.get('job_id') != job.job_id or result.get('status') != 'completed':
            raise ValueError('native rank did not complete the requested job')
        if result.get('correct') is not True or result.get('gpu_released') is not True:
            raise ValueError('native correctness or GPU release acknowledgement is missing')
        for key in ('datatype', 'reduction_op', 'payload_bytes', 'timed_iterations'):
            if result.get(key) != getattr(job, key):
                raise ValueError('native sample configuration mismatch: ' + key)
        for endpoint in result.get('endpoints', []):
            key = (result['rank'], endpoint.get('tb'), endpoint.get('step'))
            if key in endpoints:
                raise ValueError('duplicate native endpoint sample')
            for name in ('service_ns', 'samples', 'invalid_samples', 'dependency_wait_ns'):
                value = endpoint.get(name)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError('invalid native sample counter')
            if endpoint['samples'] != job.timed_iterations or endpoint['invalid_samples'] or endpoint['service_ns'] <= 0:
                raise ValueError('incomplete or invalid native sample count')
            endpoints[key] = endpoint
    expected = {(e['rank'], e['tb'], e['step']) for e in artifact.endpoints}
    if set(endpoints) != expected:
        raise ValueError('native endpoint coverage differs from the prepared XML')
    by_flow = {}
    native_kinds = {'send': 0, 'recv': 1, 'rrc': 4, 'copy': 6, 'reduce': 7}
    for definition in artifact.endpoints:
        endpoint = endpoints[(definition['rank'], definition['tb'], definition['step'])]
        if endpoint.get('kind') != native_kinds[definition['role']] or endpoint.get('channel') != definition['channel']:
            raise ValueError('native primitive or channel mismatch')
        by_flow.setdefault(definition['flow_id'], []).append(endpoint['service_ns'] / endpoint['samples'] / 1000)
    return {'flow_us': {flow: max(values) for flow, values in by_flow.items()},
            'pairing_rule': 'max_endpoint_mean_service_once_per_physical_transfer',
            'correct': True, 'gpu_released': True, 'rank_results': results}


def _create_rank_context(root, task_id, rank):
    root = owned_home_path(root)
    root.mkdir(parents=True)
    context = RunContext(root, task_id=task_id, run_id='rank-' + str(rank), redactor=Redactor())
    for name in DIRECTORIES:
        (root / name).mkdir()
    context.write_json('run.json', {'task_id': task_id, 'rank': rank, 'kind': 'native_probe_rank'})
    emit_event(context, 'run_started')
    return context


def _machine_inventory(binary, environment):
    """Read identity facts only; absence confines later reuse to this task."""
    result = {'gpus': {}, 'network': {}, 'runtime_library': None, 'complete': False}
    try:
        command = ['nvidia-smi', '--query-gpu=index,uuid,pci.bus_id,name', '--format=csv,noheader']
        probe = subprocess.run(command, capture_output=True, text=True, timeout=3, check=True)
        for row in csv.reader(probe.stdout.splitlines(), skipinitialspace=True):
            result['gpus'][str(int(row[0]))] = {'uuid': row[1], 'pci_bus_id': row[2], 'name': row[3]}
        linkage = subprocess.run(['ldd', binary], env=dict(os.environ) | environment,
                                 capture_output=True, text=True, timeout=3, check=True)
        for line in linkage.stdout.splitlines():
            if line.strip().startswith('libnccl.so') and '=>' in line:
                path = Path(line.split('=>', 1)[1].strip().split(' (', 1)[0])
                if is_foreign_home_path(path):
                    raise ValueError('forbidden runtime library dependency')
                result['runtime_library'] = {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        for interface in sorted(Path('/sys/class/net').glob('*')):
            if not (interface / 'device').exists():
                continue
            facts = {}
            for attribute in ('address', 'device/vendor', 'device/device'):
                try:
                    facts[attribute] = (interface / attribute).read_text().strip()
                except OSError:
                    pass
            result['network'][interface.name] = facts
        for interface in sorted(Path('/sys/class/infiniband').glob('*')):
            facts = {}
            for attribute in ('node_guid', 'fw_ver', 'board_id'):
                try:
                    facts[attribute] = (interface / attribute).read_text().strip()
                except OSError:
                    pass
            result['network']['rdma:' + interface.name] = facts
        result['complete'] = bool(result['gpus'] and result['runtime_library'])
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        result['error'] = str(error)
    return result


def _prepare(root, encoded, host):
    root = owned_home_path(root)
    payload = encoded if isinstance(encoded, dict) else json.loads(base64.b64decode(encoded, validate=True))
    if payload['root'] != str(root) or host not in {m['host'] for m in payload['job']['rank_map']}:
        raise ValueError('remote preparation identity mismatch')
    owned_home_path(payload['project_root'])
    # Execution dependencies are read, never configured or installed here.
    for path in (payload['binary'], payload['python']):
        if is_foreign_home_path(path):
            raise ValueError('forbidden runtime dependency path')
        if not Path(path).is_absolute() or not Path(path).is_file():
            raise ValueError('remote executable is unavailable: ' + path)
    root.mkdir(parents=True, exist_ok=True)
    _immutable(root / 'launch.json', json.dumps(payload, sort_keys=True).encode())
    schedule = payload.get('kind') == 'schedule'
    _immutable(root / ('schedule.xml' if schedule else 'probe.xml'), payload['xml'].encode())
    if schedule:
        _immutable(root / 'manifest.json', json.dumps(payload['manifest'], sort_keys=True).encode())
        _immutable(root / 'native-config.json', json.dumps(payload['native_config'], sort_keys=True).encode())
    (root / 'native').mkdir(exist_ok=True)
    (root / 'tmp').mkdir(exist_ok=True)
    # Per-host monotonic expiry is established before MPI launch. Delayed rank
    # startup cannot acquire a fresh timeout after the coordinator has stopped.
    inventory = _machine_inventory(payload['binary'], payload['environment'])
    local_mapping = [m for m in payload['job']['rank_map'] if m['host'] == host]
    devices = [inventory.get('gpus', {}).get(str(m['device']), {}).get('uuid') for m in local_mapping]
    if not devices or any(not device for device in devices):
        raise ValueError('native launch requires actual GPU UUIDs for persistent ownership')
    from vericcl.execution.ownership import GPUOwnershipRegistry
    registry_root = owned_home_path(Path(payload['project_root']) / 'examples/_shared/output/native-ownership')
    GPUOwnershipRegistry(registry_root).acquire(devices, str(root), {'task_id': payload['task_id'],
        'host': host, 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(), 'root': str(root)})
    _immutable(root / ('prepared-' + host + '.json'), json.dumps({
        'host': host, 'expires_monotonic': time.monotonic() + payload['timeout_s'],
        'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'binary_sha256': hashlib.sha256(Path(payload['binary']).read_bytes()).hexdigest(),
        'inventory': inventory, 'ownership_registry': str(registry_root), 'device_uuids': devices,
        'expected_ranks': [m['rank'] for m in payload['job']['rank_map'] if m['host'] == host],
    }).encode())
    return {'status': 'prepared', 'host': host, 'root': str(root)}


def _rank(root):
    root = owned_home_path(root)
    payload = json.loads((root / 'launch.json').read_text())
    rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    entry = next(item for item in payload['job']['rank_map'] if item['rank'] == rank)
    prepared = json.loads((root / ('prepared-' + entry['host'] + '.json')).read_text())
    if prepared['boot_id'] != Path('/proc/sys/kernel/random/boot_id').read_text().strip():
        raise ValueError('remote host restarted after preparation')
    context = _create_rank_context(root / 'ranks' / str(rank), payload['task_id'], rank)
    # A stop marker closes future launches. Missing rank acknowledgements remain
    # uncertain; the coordinator never infers release from an absent PID file.
    remaining = prepared['expires_monotonic'] - time.monotonic()
    if (root / 'STOP').exists() or remaining <= 1:
        context.write_json('rank-result.json', {'rank': rank, 'status': 'not_started',
                           'cleanup_complete': True, 'gpu_released': True})
        return 4
    job = payload['job']
    if payload.get('kind') == 'schedule':
        command = [payload['binary'], 'schedule', '--config', str(root / 'native-config.json')]
    else:
        command = _probe_command(payload, root, rank)
    result = run_owned_process(context, command, ProcessLease.after(remaining, cleanup_reserve_s=min(3, remaining / 4)),
                               cwd=payload['project_root'], environment=payload['environment'], job_id='native', rank=rank)
    native = root / 'native' / ('rank-%d.json' % rank)
    evidence = json.loads(native.read_text()) if native.is_file() else {}
    context.write_json('rank-result.json', result.as_json() | {'rank': rank, 'gpu_released': evidence.get('gpu_released') is True})
    return 0 if result.status == 'completed' and evidence.get('gpu_released') is True else 4


def _probe_command(payload, root, rank):
    job = payload['job']
    mapping = {item['original_rank']: item['rank'] for item in job['rank_map']}
    command = [payload['binary'], 'probe', '--job-id', job['job_id'], '--xml', str(root / 'probe.xml'),
               '--output-dir', str(root / 'native'), '--payload-bytes', str(job['payload_bytes']),
               '--datatype', job['datatype'], '--reduction-op', job['reduction_op'],
               '--devices', ','.join(str(item['device']) for item in job['rank_map']),
               '--warmup', str(job['warmup_iterations']), '--timed', str(job['timed_iterations']),
               '--atol', str(payload['tolerances']['absolute']), '--rtol', str(payload['tolerances']['relative'])]
    if payload.get('trace_plan'):
        command += ['--capture-trace', '1', '--trace-records', str(payload['trace_plan']['capacity_by_rank'][str(rank)])]
    for flow in job['flows']:
        command += ['--flow', '%s:%d:%d:%d' % (flow['kind'], mapping[flow['src_rank']], mapping[flow['dst_rank']], flow['channel'])]
    return command


def _reconcile_rank_cleanup(root, rank):
    """Join a native GPU receipt with cleanup of its exact owned process group.

    MPI may terminate a rank wrapper after its native child has written the
    release receipt. Process absence supplies only the separate OS evidence.
    """
    root = Path(root)
    try:
        payload = json.loads((root / 'launch.json').read_text())
        entry = next(item for item in payload['job']['rank_map'] if item['rank'] == rank)
        prepared = json.loads((root / ('prepared-' + _host(entry['host']) + '.json')).read_text())
        prefix = 'ranks/%d/jobs/native/' % rank
        identity = ProcessIdentity(**json.loads(checked_job_path(root, prefix + 'identity.json').read_text()))
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        if (identity.pid <= 0 or identity.pid != identity.pgid or identity.start_ticks <= 0
                or identity.boot_id != boot or prepared.get('boot_id') != boot
                or rank not in prepared.get('expected_ranks', [])):
            return {}
        request = json.loads(checked_job_path(root, prefix + 'request.json').read_text())
        program = request['program']
        if (request.get('rank') != rank or Path(program['path']).resolve() != Path(payload['binary']).resolve()
                or program.get('sha256') != prepared.get('binary_sha256')
                or not program.get('sha256') or _active_group_members(identity)):
            return {}
        native_path = checked_job_path(root, 'native/rank-%d.json' % rank)
        data = native_path.read_bytes() if native_path.is_file() else None
        native = json.loads(data) if data is not None else {}
        return dict(rank=rank, status='reconciled_cleanup', cleanup_complete=True,
                    gpu_released=native.get('rank') == rank and native.get('gpu_released') is True,
                    identity=asdict(identity), program=program,
                    native_receipt='native/rank-%d.json' % rank,
                    native_receipt_sha256=hashlib.sha256(data).hexdigest() if data is not None else None,
                    process_evidence=prefix + 'identity.json', request_evidence=prefix + 'request.json')
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return {}


def _rank_cleanup_receipt(root, rank):
    path = checked_job_path(root, 'ranks/%d/rank-result.json' % rank)
    original = json.loads(path.read_text()) if path.is_file() else {}
    if original.get('rank') != rank:
        original = {}
    if original.get('cleanup_complete') is True and original.get('gpu_released') is True:
        return original
    reconciled = _reconcile_rank_cleanup(root, rank)
    if reconciled.get('gpu_released') is True:
        _immutable(root / 'ranks' / str(rank) / 'cleanup-reconciliation.json',
                   json.dumps(reconciled, sort_keys=True).encode())
    return reconciled or original


def collected_cleanup_receipts(collections):
    """One receipt per rank, including a separately recorded reconciliation."""
    receipts = {}
    for collection in collections:
        for relative, encoded in collection['files'].items():
            if not relative.endswith(('/rank-result.json', '/cleanup-reconciliation.json')):
                continue
            item = json.loads(base64.b64decode(encoded, validate=True))
            rank = item.get('rank')
            if not isinstance(rank, int) or isinstance(rank, bool):
                continue
            old = receipts.get(rank, {})
            if (rank not in receipts or
                    (item.get('cleanup_complete') is True and item.get('gpu_released') is True) or
                    old.get('cleanup_complete') is not True):
                receipts[rank] = item
    return receipts


def _cleanup(root, host, timeout):
    root = owned_home_path(root)
    prepared_path = root / ('prepared-' + _host(host) + '.json')
    if not prepared_path.exists():
        return {'host': host, 'cleanup_complete': False, 'gpu_released': False, 'reason': 'preparation acknowledgement missing'}
    prepared = json.loads(prepared_path.read_text())
    _immutable(root / 'STOP', b'stop\n')
    deadline = time.monotonic() + timeout
    observed, term = {}, set()
    acknowledgements = [{} for _ in prepared['expected_ranks']]
    while time.monotonic() < deadline:
        for rank in prepared['expected_ranks']:
            identity_path = root / 'ranks' / str(rank) / 'jobs/native/identity.json'
            if identity_path.exists():
                identity = ProcessIdentity(**json.loads(identity_path.read_text()))
                observed[rank] = identity
                if identity.matches() and _active_group_members(identity):
                    signal_owned_process_group(identity, signal.SIGKILL if rank in term else signal.SIGTERM)
                    term.add(rank)
        acknowledgements = [_rank_cleanup_receipt(root, rank) for rank in prepared['expected_ranks']]
        if all(a.get('cleanup_complete') for a in acknowledgements):
            break
        time.sleep(min(.05, max(0, deadline - time.monotonic())))
    process_complete = all(a.get('cleanup_complete') for a in acknowledgements)
    # If the wrapper died, exact process absence can establish OS cleanup only.
    if not process_complete and len(observed) == len(prepared['expected_ranks']):
        process_complete = all(not _active_group_members(identity) for identity in observed.values())
    return {'host': host, 'cleanup_complete': process_complete,
            'gpu_released': all(a.get('gpu_released') for a in acknowledgements),
            'rank_acknowledgements': acknowledgements}


_COLLECTION_PAGE_MIN_BYTES = 32 * 1024 * 1024


def _collection_inventory(root, host, prepared, payload, *, receipts_only=False):
    receipts, artifacts = [], []
    for rank in prepared['expected_ranks']:
        receipts.extend(['native/rank-%d.json' % rank, 'ranks/%d/rank-result.json' % rank,
                         'ranks/%d/cleanup-reconciliation.json' % rank,
                         'ranks/%d/jobs/native/identity.json' % rank, 'ranks/%d/jobs/native/request.json' % rank])
        if receipts_only:
            continue
        artifacts.extend(['ranks/%d/jobs/native/stdout.log' % rank, 'ranks/%d/jobs/native/stderr.log' % rank])
        if payload.get('trace_plan'):
            artifacts.append('native/trace.rank-%d.bin' % rank)
            if payload.get('kind') == 'schedule':
                artifacts.append('native/clock.rank-%d.log' % rank)
        if payload.get('kind') == 'schedule':
            artifacts.extend(str(path.relative_to(root)) for path in (root / 'native' / ('rank-%d' % rank)).glob('*.json'))
    groups, metadata = [], {}
    for names in (receipts, artifacts):
        present = []
        for name in sorted(set(names)):
            path = checked_job_path(root, name)
            if path.is_file():
                stat = path.stat()
                metadata[name] = (stat.st_size, stat.st_mtime_ns, stat.st_dev, stat.st_ino)
                present.append(name)
        groups.append(present)
    digest = hashlib.sha256(json.dumps([str(root), host, groups, metadata], sort_keys=True).encode()).hexdigest()
    return groups[0], groups[1], metadata, digest


def _collect(root, host, *, cursor=None, inventory_digest=None):
    root = owned_home_path(root)
    host = _host(host)
    prepared = json.loads((root / ('prepared-' + host + '.json')).read_text())
    payload = json.loads((root / 'launch.json').read_text())
    for rank in prepared['expected_ranks']:
        _rank_cleanup_receipt(root, rank)
    receipt_page = cursor == 0
    receipts, artifacts, metadata, digest = _collection_inventory(root, host, prepared, payload,
                                                                 receipts_only=receipt_page)
    legacy = cursor is None
    if not legacy and (type(cursor) is not int or cursor < 0 or cursor > max(1, len(artifacts))
            or (cursor > 1 and inventory_digest is None)
            or (inventory_digest is not None and inventory_digest != digest)):
        raise ValueError('invalid collection cursor or changed inventory snapshot')
    trace_plan = payload.get('trace_plan')
    maximum_bytes = max(_COLLECTION_PAGE_MIN_BYTES,
        sum(trace_plan['capacity_by_rank'].values()) * 64 + 1024 * 1024) if trace_plan else _COLLECTION_PAGE_MIN_BYTES
    files, total = {}, 0
    # A coordinator already running the old protocol omits cursor. Preserve
    # its complete bounded response while new coordinators request page zero.
    names = receipts + artifacts if legacy else (receipts if cursor == 0 else artifacts[cursor - 1:])
    for relative in names:
        size = metadata[relative][0]
        if total + size > maximum_bytes:
            if legacy or cursor == 0 or not files:
                raise ValueError('remote collection file or receipt page exceeds its preallocated evidence bound; original files are retained')
            break
        path = checked_job_path(root, relative)
        before = path.stat()
        data = path.read_bytes()
        after = path.stat()
        if (len(data) != size or any((stat.st_size, stat.st_mtime_ns, stat.st_dev, stat.st_ino)
                                    != metadata[relative] for stat in (before, after))):
            raise ValueError('collection inventory snapshot changed while reading a file')
        total += len(data)
        files[relative] = base64.b64encode(data).decode()
    if _collection_inventory(root, host, prepared, payload, receipts_only=receipt_page)[3] != digest:
        raise ValueError('collection inventory snapshot changed during page collection')
    if (legacy or cursor == 0) and prepared.get('ownership_registry'):
        from vericcl.execution.ownership import GPUOwnershipRegistry
        acknowledgements = collected_cleanup_receipts([{'files': files}])
        complete = set(acknowledgements) == set(prepared['expected_ranks'])
        never_started = (root / 'NOT_STARTED').is_file() and not (root / 'ranks').exists()
        GPUOwnershipRegistry(prepared['ownership_registry']).finish(prepared['device_uuids'], str(root),
            process_clean=never_started or (complete and all(r.get('cleanup_complete') is True for r in acknowledgements.values())),
            gpu_released=never_started or (complete and all(r.get('gpu_released') is True for r in acknowledgements.values())))
    end = 0 if legacy or cursor == 0 else cursor - 1 + len(files)
    # Page zero never even traverses bulk files. The first bulk request
    # establishes its own inventory, so slow storage cannot delay receipts.
    next_cursor = None if legacy else (1 if cursor == 0 else
                                       (end + 1 if end < len(artifacts) else None))
    return {'host': host, 'prepared': prepared, 'files': files, 'page_cursor': cursor,
            'next_cursor': next_cursor, 'snapshot': {'digest': digest,
                'scope': 'receipts' if receipt_page else 'complete',
                'total_bytes': sum(value[0] for value in metadata.values()), 'receipt_files': len(receipts),
                'artifact_files': len(artifacts), 'page_limit_bytes': maximum_bytes}}


class NativeProbeDriver:
    driver_kind = 'native_msccl'

    def __init__(self, *, project_root=None, python=None):
        self.project_root = Path(project_root or Path(__file__).resolve().parents[3]).resolve()
        self.python = str(python or sys.executable)
        self.quarantined = set()
        self.sequence = 0
        self.successful_launch_seconds = {}
        self.local_addresses = {'127.0.0.1'}
        if sys.platform == 'linux':
            import fcntl
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as descriptor:
                for _, name in socket.if_nameindex():
                    try:
                        result = fcntl.ioctl(descriptor.fileno(), 0x8915, struct.pack('256s', name.encode()[:15]))
                        self.local_addresses.add(socket.inet_ntoa(result[20:24]))
                    except OSError:
                        pass
        self.local_aliases = {}

    def _is_local(self, host):
        if host not in self.local_aliases:
            try:
                addresses = {entry[4][0] for entry in socket.getaddrinfo(_host(host), None, socket.AF_INET)}
                self.local_aliases[host] = bool(addresses & self.local_addresses)
            except socket.gaierror:
                self.local_aliases[host] = False
        return self.local_aliases[host]

    def _call(self, context, command, timeout, label, *, environment=None, cancel_check=None):
        self.sequence += 1
        reserve = min(1., timeout / 4)
        if timeout <= .1:
            raise BudgetExhausted('no remaining remote transport budget')
        # Driver instances are recreated on resume and between components.
        # Their local sequence cannot identify retained jobs across instances.
        job_id = 'probe-%06d-%s-%s' % (self.sequence, label, uuid.uuid4().hex)
        return run_owned_process(context.run_context, command, ProcessLease.after(timeout, cleanup_reserve_s=reserve),
                                 cwd=self.project_root, environment=environment,
                                 job_id=job_id, cancel_check=cancel_check)

    def _helper(self, context, host, command, timeout, label):
        argv = [self.python, '-m', 'vericcl.verification.online.remote', *command]
        if self._is_local(host):
            command = argv
        else:
            remote = shlex.join(['env', 'PYTHONPATH=' + str(self.project_root), *argv])
            command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', _host(host), remote]
        result = self._call(context, command, timeout, label, environment={'PYTHONPATH': str(self.project_root)})
        if result.status != 'completed' or not result.cleanup_complete:
            raise RuntimeError('remote %s failed on %s; inspect %s' % (label, host, result.stderr_path))
        try:
            return json.loads(result.stdout_path.read_text().strip().splitlines()[-1])
        except (ValueError, IndexError) as error:
            raise ValueError('invalid remote helper result on ' + host) from error

    def _collect_evidence(self, context, hosts, root, deadline, label, destination):
        """Persist every host's receipt page before spending time on bulk logs."""
        collected, errors, pending = [], [], []

        def receive(host, args, allowance):
            page = self._helper(context, host, args, allowance, label)
            if page.get('host') != host:
                raise ValueError('collection host differs from requested host')
            collected.append(page)
            archived = {}
            for name, encoded in page['files'].items():
                checked_job_path(root, name)
                data = base64.b64decode(encoded, validate=True)
                context.run_context.write_bytes(destination + '/' + host + '/' + name, data)
                archived[name] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
            if '--cursor' in args and args[args.index('--cursor') + 1] != '0':
                # Bulk evidence is already durable. Keep its index, not the
                # accumulated base64 payload of an arbitrarily long session.
                page['archived_files'], page['files'] = archived, {}
            return page

        for index, host in enumerate(hosts):
            try:
                allowance = (deadline - time.monotonic()) / (len(hosts) - index + 1)
                if allowance <= .1:
                    raise BudgetExhausted('receipt collection deadline exhausted')
                page = receive(host, ['collect', '--root', str(root), '--host', host, '--cursor', '0'], allowance)
                if page.get('artifact_error'):
                    errors.append(dict(host=host, phase='artifacts', error=page['artifact_error']))
                elif page.get('next_cursor') is not None:
                    pending.append((host, page))
            except StorageFailure:
                raise
            except Exception as error:
                errors.append(dict(host=host, phase='receipts', error=str(error)))
        for host, page in pending:
            try:
                previous = 0
                digest = None
                while page.get('next_cursor') is not None:
                    cursor = page['next_cursor']
                    if type(cursor) is not int or cursor <= previous:
                        raise ValueError('collection cursor did not advance')
                    allowance = deadline - time.monotonic()
                    if allowance <= .1:
                        raise BudgetExhausted('artifact collection deadline exhausted')
                    args = ['collect', '--root', str(root), '--host', host, '--cursor', str(cursor)]
                    if digest is not None:
                        args.extend(['--inventory-digest', digest])
                    page = receive(host, args, allowance)
                    returned_digest = page.get('snapshot', {}).get('digest')
                    if not isinstance(returned_digest, str) or not returned_digest or (digest is not None and returned_digest != digest):
                        raise ValueError('collection snapshot identity changed')
                    digest = returned_digest
                    previous = cursor
            except StorageFailure:
                raise
            except Exception as error:
                errors.append(dict(host=host, phase='artifacts', error=str(error)))
        return collected, errors

    def _launch(self, job, artifact, context, round_index, launch_index, timeout, *, trace_plan=None):
        from .probe_jobs import probe_artifact_key
        run = context.run_context
        root = owned_home_path(run.artifact('models/probes/%s/round-%02d/launch-%03d' % (probe_artifact_key(job), round_index, launch_index)))
        paths = context.inputs.request_config['runtime']['paths']
        binary = paths.get('vericcl_driver')
        mpi_bin = paths.get('mpi_bin')
        if not binary or not mpi_bin:
            raise ValueError('native probes require runtime.paths.vericcl_driver and mpi_bin')
        hosts = sorted({_host(m['host']) for m in job.rank_map})
        environment = dict(context.inputs.request_config['runtime']['environment'])
        environment.update(PYTHONPATH=str(self.project_root), VERICCL_TRACE_ENABLE='0',
                           TMPDIR=str(root / 'tmp'), NCCL_PROTO='Simple')
        payload = {'root': str(root), 'project_root': str(self.project_root), 'binary': str(binary),
                   'python': self.python, 'job': asdict(job), 'xml': artifact.xml, 'environment': environment,
                   'task_id': run.task_id, 'timeout_s': timeout,
                   'tolerances': dict(context.inputs.request_config['sketch']['tolerances']),
                   'trace_plan': None if trace_plan is None else {'capture_id': trace_plan.capture_id, 'capacity_by_rank': dict(trace_plan.capacity_by_rank)}}
        encoded = base64.b64encode(json.dumps(payload).encode()).decode()
        deadline = time.monotonic() + timeout
        reserve = min(10., timeout / 3)
        prepared, collected, cleanup, error, fatal = [], [], [], None, None
        started, execution = False, None
        try:
            for host in hosts:
                remaining = deadline - time.monotonic() - reserve
                self._helper(context, host, ['prepare', '--root', str(root), '--host', host, '--payload', encoded], min(10., remaining), 'prepare')
                prepared.append(host)
            # rankfile preserves original-to-MPI mapping even when ranks from a
            # host are interleaved. Every native rank uses its explicit GPU id.
            root.mkdir(parents=True, exist_ok=True)
            (root / 'tmp').mkdir(exist_ok=True)
            rankfile = root / 'rankfile'
            _immutable(rankfile, ''.join('rank %d=%s slot=0\n' % (m['rank'], 'localhost' if self._is_local(m['host']) else m['host']) for m in job.rank_map).encode())
            command = [str(Path(mpi_bin) / 'mpirun'), '--rankfile', str(rankfile), '--bind-to', 'none',
                       '-np', str(len(job.rank_map)), self.python, '-m', 'vericcl.verification.online.remote',
                       'rank', '--root', str(root)]
            started = True
            execution = self._call(context, command, deadline - time.monotonic() - reserve, 'mpi', environment=environment)
            run.write_json(str((root / 'launcher-result.json').relative_to(run.root)), execution.as_json())
            if execution.status != 'completed' or not execution.cleanup_complete:
                error = 'MPI launch did not complete cleanly'
        except (BudgetExhausted, StorageFailure) as failure:
            fatal, error = failure, str(failure)
        except Exception as failure:
            error = str(failure)
        finally:
            # Cleanup and collection have independent attempts for each host.
            # A failed inter-host channel does not discard other host evidence.
            for host in prepared:
                try:
                    allowance = max(.2, (deadline - time.monotonic()) / max(1, 2 * len(prepared)))
                    if started and error:
                        ack = self._helper(context, host, ['cleanup', '--root', str(root), '--host', host,
                                           '--timeout', str(max(.05, allowance / 2))], allowance, 'cleanup')
                        cleanup.append(ack)
                except StorageFailure:
                    raise
                except Exception as failure:
                    error = (error + '; ' if error else '') + str(failure)
            collected, collection_errors = self._collect_evidence(context, prepared, root, deadline, 'collect',
                str(root.relative_to(run.root)) + '/collected')
            if collection_errors:
                error = '; '.join(([error] if error else []) + [str(item) for item in collection_errors])
        rank_results = []
        for collection in collected:
            for relative, data in collection['files'].items():
                if relative.startswith('native/rank-') and relative.endswith('.json'):
                    rank_results.append(json.loads(base64.b64decode(data)))
        wrappers = collected_cleanup_receipts(collected)
        complete = set(wrappers) == {m['rank'] for m in job.rank_map}
        released = complete and all(w.get('gpu_released') is True for w in wrappers.values())
        process_clean = complete and all(w.get('cleanup_complete') is True for w in wrappers.values())
        launcher_clean = (not started or (execution is not None and
            (execution.cleanup_complete is True or reconcile_owned_process(run, execution))))
        if started and not released:
            self.quarantined.update((m['host'], m['device']) for m in job.rank_map)
            run.write_json('live/native-resource-ownership.json', {
                'quarantined': [list(item) for item in sorted(self.quarantined)],
                'last_job_id': job.job_id, 'reason': 'GPU release acknowledgement missing'}, replace=True)
        evidence = {'error': error, 'gpu_released': released, 'cleanup_confirmed': process_clean and launcher_clean,
                    'launcher_cleanup_confirmed': launcher_clean,
                    'cleanup': cleanup, 'rank_results': rank_results, 'prepared_hosts': prepared}
        run.write_json(str((root / 'launch-result.json').relative_to(run.root)), evidence)
        if fatal is not None:
            raise fatal
        if trace_plan is not None:
            from .trace_once import RawTraceCapture
            from .clock_sync import parse_clock_sync_output
            trace_paths, trace_metadata, clocks = {}, {}, []
            for mapping in job.rank_map:
                rank = mapping['rank']
                path = root / 'collected' / mapping['host'] / 'native' / ('trace.rank-%d.bin' % rank)
                if path.is_file():
                    trace_paths[rank] = path
                facts = next((item for item in rank_results if item['rank'] == rank), {})
                trace_metadata[rank] = facts
                try:
                    clocks.extend(parse_clock_sync_output(facts.get('clock_sync_output', '')))
                except Exception as failure:
                    trace_metadata[rank] = facts | {'clock_error': str(failure)}
            trace_issues = () if not error else ({'rank': None, 'reason': error},)
            return RawTraceCapture(trace_paths, tuple(clocks), trace_metadata, 'native_msccl', trace_issues), evidence
        if error:
            return None, evidence
        try:
            sample = combine_rank_samples(job, artifact, rank_results)
            sample.update(round=round_index, launch=launch_index)
            inventories = {item['host']: item['prepared'] for item in collected}
            runtime_facts, complete = [], True
            for mapping in job.rank_map:
                prepared = inventories[mapping['host']]
                inventory = prepared.get('inventory', {})
                device = inventory.get('gpus', {}).get(str(mapping['device']))
                facts = next(r for r in rank_results if r['rank'] == mapping['rank'])
                complete = complete and bool(inventory.get('complete') and device)
                if job.host_scope == 'inter' and not inventory.get('network'):
                    complete = False
                runtime_facts.append({'host': mapping['host'], 'device': mapping['device'], 'gpu': device,
                    'runtime_library': inventory.get('runtime_library'), 'binary_sha256': prepared['binary_sha256'],
                    'cuda_runtime_version': facts['cuda_runtime_version'], 'cuda_driver_version': facts['cuda_driver_version'],
                    'nccl_version': facts['nccl_version'], 'compute_capability': facts['compute_capability'],
                    'network': inventory.get('network') if job.host_scope == 'inter' else None})
            sample['runtime_identity'] = {'ranks': runtime_facts,
                'environment': dict(context.inputs.request_config['runtime']['environment']), 'protocol': 'Simple',
                'task_scope': None if complete else run.task_id}
            sample['measured_at'] = datetime.now(timezone.utc).isoformat()
            return sample, evidence
        except ValueError as failure:
            evidence['error'] = str(failure)
            run.write_json(str((root / 'launch-result.json').relative_to(run.root)), evidence, replace=True)
            return None, evidence

    def execute(self, job, context):
        from .probe_jobs import ProbeResult, probe_artifact_key
        from .probe_xml import build_probe_xml
        job = replace(job, execution_id=job.execution_id or uuid.uuid4().hex)
        if context.run_context is None:
            raise ValueError('native measurement requires an owned RunContext')
        owned_home_path(self.project_root)
        if job.reduction_op == 'avg':
            return ProbeResult(job.job_id, 'unsupported', self.driver_kind,
                message='the pinned CustomCollective runtime lacks average-reduction dispatch',
                cleanup_confirmed=True, gpu_released=True,
                provenance={'capability_check': 'source_checked_before_GPU_launch'})
        ownership = context.run_context.artifact('live/native-resource-ownership.json')
        if ownership.is_file():
            retained = json.loads(ownership.read_text())
            self.quarantined.update(tuple(item) for item in retained['quarantined'])
        if any((m['host'], m['device']) in self.quarantined for m in job.rank_map):
            return ProbeResult(job.job_id, 'blocked_owned_resources', self.driver_kind,
                               message='an earlier job has no GPU release acknowledgement')
        artifact = build_probe_xml(job)
        context.run_context.write_bytes('models/probes/' + probe_artifact_key(job) + '/probe.xml', artifact.xml.encode())
        samples, rounds = [], []
        # job_timeout_s bounds this complete mode/payload job, including its
        # stability retries, transport, initialization and cleanup.
        deadline = time.monotonic() + min(job.timeout_s, context.lease.remaining_seconds())
        clean = released = True
        placement = tuple(sorted((item['host'], item['device']) for item in job.rank_map))
        longest_launch_s = self.successful_launch_seconds.get(placement, 0.0)
        for round_index in range(1, job.max_rounds + 1):
            current = []
            for launch_index in range(1, job.launches + 1):
                context.lease.check()
                remaining = min(deadline - time.monotonic(), context.lease.remaining_seconds())
                # A launch includes preparation, MPI startup and collection.
                # Do not knowingly start it with less than a recent complete
                # launch plus variability and the ten-second cleanup allowance.
                # This is admission control, not an extension of either deadline.
                # A new mode/payload on the same devices still pays the same
                # preparation/launcher costs. Keep successful timings across
                # jobs; a failed long timeout must not block independent probes.
                admission_s = max(30.0, 10.0 + 1.5 * longest_launch_s)
                if remaining <= admission_s:
                    return ProbeResult(job.job_id, 'timeout', self.driver_kind, tuple(samples),
                                       'probe job time limit reached before starting another launch',
                                       cleanup_confirmed=clean, gpu_released=released,
                                       provenance={'rounds': rounds, 'remaining_s': remaining,
                                                   'next_launch_admission_s': admission_s})
                launched_at = time.monotonic()
                sample, evidence = self._launch(job, artifact, context, round_index, launch_index, remaining)
                longest_launch_s = max(longest_launch_s, time.monotonic() - launched_at)
                if sample is not None and evidence['cleanup_confirmed'] is True and evidence['gpu_released'] is True:
                    self.successful_launch_seconds[placement] = longest_launch_s
                clean = clean and evidence['cleanup_confirmed']
                released = released and evidence['gpu_released']
                if not clean or not released or context.run_context.active_jobs:
                    return ProbeResult(job.job_id, 'failed', self.driver_kind, tuple(samples),
                        evidence.get('error') or 'probe cleanup or GPU release is unconfirmed',
                        cleanup_confirmed=clean and not context.run_context.active_jobs,
                        gpu_released=released, provenance={'rounds': rounds})
                if sample is None:
                    return ProbeResult(job.job_id, 'failed', self.driver_kind, tuple(samples), evidence['error'],
                                       cleanup_confirmed=clean, gpu_released=released, provenance={'rounds': rounds})
                current.append(sample); samples.append(sample)
                context.run_context.write_json('models/probes/%s/sample-%02d-%03d.json' % (probe_artifact_key(job), round_index, launch_index), sample)
            cvs = {}
            for flow in job.flows:
                values = [sample['flow_us'][flow.flow_id] for sample in current]
                cvs[flow.flow_id] = statistics.pstdev(values) / statistics.mean(values)
            rounds.append({'round': round_index, 'cv_by_flow': cvs, 'stable': all(v <= job.max_cv for v in cvs.values())})
            if rounds[-1]['stable']:
                identities = [sample['runtime_identity'] for sample in current]
                if any(identity != identities[0] for identity in identities[1:]):
                    return ProbeResult(job.job_id, 'environment_changed', self.driver_kind, tuple(samples),
                        'runtime identity changed within the measurement round', cleanup_confirmed=clean,
                        gpu_released=released, provenance={'rounds': rounds})
                return ProbeResult(job.job_id, 'completed', self.driver_kind, tuple(samples),
                                   cleanup_confirmed=clean, gpu_released=released,
                                   provenance={'selected_round': round_index, 'rounds': rounds,
                                               'measurement_id': job.execution_id,
                                               'measured_at': min(sample['measured_at'] for sample in current),
                                               'runtime_identity': identities[0],
                                               'measurement': 'actual_msccl_primitive_service'})
        return ProbeResult(job.job_id, 'unstable', self.driver_kind, tuple(samples),
                           'no stable round within configured retry count', cleanup_confirmed=clean,
                           gpu_released=released, provenance={'rounds': rounds})


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('cancel-before-launch', 'stage', 'prepare-file', 'prepare', 'rank', 'cleanup', 'collect'))
    parser.add_argument('--root', required=True)
    parser.add_argument('--host')
    parser.add_argument('--payload')
    parser.add_argument('--cursor', type=int)
    parser.add_argument('--inventory-digest')
    parser.add_argument('--timeout', type=float, default=2)
    args = parser.parse_args(argv)
    if args.action == 'rank':
        return _rank(args.root)
    if args.action == 'cancel-before-launch':
        root = owned_home_path(args.root)
        _immutable(root / 'STOP', b'stop\n')
        if (root / 'ranks').exists():
            raise ValueError('rank launch already observed; cannot certify pre-launch cancellation')
        _immutable(root / 'NOT_STARTED', b'coordinator cancelled before MPI launch\n')
        result = {'status': 'cancelled_before_launch', 'root': str(root)}
    elif args.action == 'stage':
        root = owned_home_path(args.root)
        root.mkdir(parents=True, exist_ok=True)
        result = {'root': str(root), 'status': 'staged'}
    elif args.action == 'prepare-file':
        root = owned_home_path(args.root)
        result = _prepare(root, json.loads((root / 'incoming-payload.json').read_text()), _host(args.host))
    elif args.action == 'prepare':
        result = _prepare(args.root, args.payload, _host(args.host))
    elif args.action == 'cleanup':
        if not math.isfinite(args.timeout) or not 0 < args.timeout <= 30:
            raise ValueError('cleanup timeout must be finite and at most 30 seconds')
        result = _cleanup(args.root, args.host, args.timeout)
    else:
        result = _collect(args.root, args.host, cursor=args.cursor, inventory_digest=args.inventory_digest)
    print(json.dumps(result))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


class NativeProbeTraceDriver:
    """Real once-only capture of the independent probe XML (development/calibration).

    Whole-collective capture uses the schedule executor; this adapter deliberately
    keeps the independent-slot probe semantics explicit.
    """
    def __init__(self, job, context, *, driver=None):
        self.job, self.context = job, context
        self.driver = driver or NativeProbeDriver()
        self.last_evidence = None

    def capture_once(self, plan):
        from .probe_xml import build_probe_xml
        job = replace(self.job, execution_id=uuid.uuid4().hex, warmup_iterations=plan.warmup_iterations,
                      timed_iterations=1)
        if tuple(plan.rank_ids) != tuple(range(len(job.rank_map))):
            raise ValueError('probe trace sidecar uses dense MPI rank identities')
        if any((item['host'], item['device']) in self.driver.quarantined for item in job.rank_map):
            raise ValueError('probe device has unresolved previous ownership')
        artifact = build_probe_xml(job)
        if hashlib.sha256(artifact.xml.encode()).hexdigest() != plan.xml_sha256:
            raise ValueError('capture plan XML differs from probe XML')
        timeout = min(job.timeout_s, self.context.lease.remaining_seconds())
        result, evidence = self.driver._launch(job, artifact, self.context, 1, 1, timeout, trace_plan=plan)
        self.last_evidence = evidence
        if result is None:
            raise ValueError('trace launch or GPU cleanup failed: ' + str(evidence.get('error')))
        return result
