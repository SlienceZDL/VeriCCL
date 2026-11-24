"""Full checked schedules executed by the shared native C API core."""
from __future__ import annotations

import base64
import hashlib
import json
import math
from pathlib import Path
import socket
import statistics
import time
from types import SimpleNamespace
import uuid

from vericcl.execution.budget import BudgetExhausted
from vericcl.execution.context import StorageFailure
from vericcl.execution.processes import reconcile_owned_process
from vericcl.input.schema import LIVE_TIMEOUTS
from vericcl.live.manifest import export_manifest
from vericcl.verification.online.acceptance import LaunchSample
from vericcl.verification.online.clock_sync import parse_clock_sync_output
from vericcl.verification.online.remote import (NativeProbeDriver, _host, _immutable, owned_home_path,
                                                 collected_cleanup_receipts, unbound_rankfile)
from vericcl.verification.online.trace_once import RawTraceCapture


def combine_schedule_results(records, *, rank_count, version, contract, warmup, timed):
    if len(records) != rank_count or {r.get('rank') for r in records} != set(range(rank_count)):
        raise ValueError('schedule measurement is missing a rank or has duplicate ranks')
    for record in records:
        required = dict(correct=True, gpu_completed=True, gpu_released=True, trace_enabled=False,
                        version_id=version, contract_digest=contract, warmup_iterations=warmup, timed_iterations=timed)
        if any(record.get(key) != value for key, value in required.items()):
            raise ValueError('schedule receipt differs from the requested complete execution')
        times = record.get('device_times_us', [])
        if len(times) != timed or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in times):
            raise ValueError('invalid native device durations')
    completed_times = [max(record['device_times_us'][i] for record in records) for i in range(timed)]
    return {'time_us': statistics.fmean(completed_times), 'invocation_times_us': completed_times,
            'aggregation': 'mean_of_slowest_rank_per_invocation'}


class NativeScheduleDriver:
    driver_kind = 'native_msccl'

    def __init__(self, context, inputs, *, project_root=None):
        self.context, self.inputs = context, inputs
        self.project_root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
        self.transport = NativeProbeDriver(project_root=self.project_root)
        self.last_evidence = None

    def _control(self, inputs):
        runtime = inputs.request_config['runtime']
        first_host = runtime['rank_map'][0]['host']
        configured = runtime['control']
        # Ephemeral local ports are permitted for one-shot measurement only.
        # Persistent live groups always use the explicitly configured endpoint.
        addresses = socket.getaddrinfo(_host(first_host), None, socket.AF_INET, socket.SOCK_STREAM)
        address = addresses[0][4][0]
        if not self.transport._is_local(first_host) and not configured.get('port'):
            raise ValueError('a remote rank 0 requires an explicit runtime.control.port')
        bind = configured.get('bind_address', address)
        if bind != '0.0.0.0':
            address = bind
        port = configured.get('port')
        if port is None:
            with socket.socket() as descriptor:
                descriptor.bind((bind, 0))
                port = descriptor.getsockname()[1]
        return address, bind, port

    def _launch(self, bundle, inputs, *, launch_id, warmup, timed, lease, trace_plan=None, self_test_api=False,
                live=None, on_prepared=None, cancel_check=None, sequence=None):
        if lease is None:
            raise ValueError('native schedule execution requires a parent time lease')
        lease.check()
        runtime = inputs.request_config['runtime']
        if live is not None:
            if not self.transport._is_local(runtime['rank_map'][0]['host']):
                raise ValueError('the persistent controller must run on the explicitly mapped rank-0 host')
            owned_home_path(Path(runtime['control']['unix_socket']))
        paths = runtime['paths']
        binary, mpi_bin = paths.get('vericcl_native'), paths.get('mpi_bin')
        if not binary or not mpi_bin:
            raise ValueError('full schedule execution requires runtime.paths.vericcl_native and mpi_bin')
        mapping = [dict(item) for item in runtime['rank_map']]
        if any((m['host'], m['device']) in self.transport.quarantined for m in mapping):
            raise ValueError('an earlier native job retains an overlapping GPU')
        settings = inputs.request_config['sketch']['trace_and_measure']
        timeout = min(live['timeout_s'] if live else settings.get('job_timeout_s', lease.remaining_seconds()), lease.remaining_seconds())
        if timeout <= 3:
            raise BudgetExhausted('insufficient time for native startup, execution and cleanup')
        deadline = time.monotonic() + timeout
        reserve = min(10., timeout / 3)
        unique = uuid.uuid4().hex
        root = owned_home_path(self.context.artifact('jobs/schedule-' + unique))
        root.mkdir(parents=True)
        live_trace = live is not None and settings['enabled']
        if live_trace:
            from vericcl.verification.online.trace_format import RAW_HEADER_STRUCT, RAW_RECORD_STRUCT
            capacities = {rank: (settings['trace_capacity_bytes'] - RAW_HEADER_STRUCT.size) // RAW_RECORD_STRUCT.size for rank in range(inputs.rank_count)}
            prefix = root / 'versions' / bundle.xml_sha256 / 'trace-tmp'
        else:
            capacities = None if trace_plan is None else trace_plan.capacity_by_rank
            prefix = None if trace_plan is None else root / 'native/trace'
        manifest = export_manifest(bundle, inputs, root / 'schedule.xml', trace_records=capacities,
            trace_prefix=prefix, light_telemetry=live is not None, live_trace=live_trace)
        address, bind, port = self._control(inputs)
        phase_seconds = max(.1, min(10., (timeout - reserve) / 2))
        native = dict(schema_version=1, group_id=self.context.task_id, session_id=unique, rank_map=mapping,
            coordinator_address=address, bind_address=bind, port=port, output_directory=str(root / 'native'),
            manifest_path=str(root / 'manifest.json'), timeouts={key: phase_seconds for key in LIVE_TIMEOUTS},
            device_memory_budget_bytes=runtime.get('device_memory_budget_bytes', 0),
            recovery=dict(max_retries=0, fault_inject_request=0, fault_inject_attempts=0),
            warmup=warmup, timed=timed, capture_trace=trace_plan is not None, self_test_api=self_test_api)
        # Communicator setup includes all peer connections and XML admission.
        # Its allowance scales within the existing complete job deadline;
        # short control/cleanup waits retain their separate bounded defaults.
        native['timeouts']['version_prepare_timeout_s'] = max(phase_seconds, (timeout - reserve) / 2)
        versions = {}
        if sequence:
            if live is not None or trace_plan is not None:
                raise ValueError('a measurement sequence runs in finite mode without a detailed trace')
            entries = []
            for label, item_bundle, item_inputs, item_warmup, item_timed in sequence:
                digest = item_bundle.xml_sha256
                if digest not in versions:
                    item_manifest = export_manifest(item_bundle, item_inputs, root / 'versions' / digest / 'schedule.xml',
                        trace_records=None, trace_prefix=None, light_telemetry=False, live_trace=False)
                    if item_manifest['contract_digest'] != manifest['contract_digest']:
                        raise ValueError('a measurement sequence must keep one external collective contract')
                    versions[digest] = dict(xml=item_bundle.xml_text, manifest=item_manifest)
                encoded = json.dumps(versions[digest]['manifest'], sort_keys=True).encode()
                entries.append(dict(label=label, manifest_path=str(root / 'versions' / digest / 'manifest.json'),
                                    manifest_sha256=hashlib.sha256(encoded).hexdigest(), warmup=item_warmup, timed=item_timed))
            native['sequence'] = entries
        if live is not None:
            native.update(mode='live', unix_socket=runtime['control']['unix_socket'], timeouts=dict(live['timeouts']))
            configuration = inputs.request_config['sketch']['live']
            native['recovery'] = dict(max_retries=configuration['max_retries'],
                fault_inject_request=configuration['fault_injection']['request_id'],
                fault_inject_attempts=configuration['fault_injection']['attempts'])
        environment = dict(runtime['environment']) | dict(PYTHONPATH=str(self.project_root),
            TMPDIR=str(root / 'tmp'), VERICCL_TRACE_ENABLE='0', NCCL_PROTO='Simple')
        payload = dict(kind='schedule', root=str(root), project_root=str(self.project_root), binary=binary,
            python=self.transport.python, task_id=self.context.task_id, timeout_s=timeout,
            job=dict(job_id=launch_id, rank_map=mapping), xml=bundle.xml_text, manifest=manifest,
            native_config=native, environment=environment, versions=versions,
            trace_plan=None if trace_plan is None else dict(capacity_by_rank=dict(trace_plan.capacity_by_rank)))
        relative = str(root.relative_to(self.context.root))
        self.context.write_json(relative + '/incoming-payload.json', payload)
        helper_context = SimpleNamespace(run_context=self.context, inputs=inputs, lease=lease)
        hosts = sorted({_host(m['host']) for m in mapping})
        prepared, collections, cleanups, cores = [], [], [], {}
        launched, failure, fatal, process = False, None, None, None
        try:
            for host in hosts:
                left = min(10., deadline - time.monotonic() - reserve)
                self.transport._helper(helper_context, host, ['stage', '--root', str(root)], left, 'schedule-stage')
                if not self.transport._is_local(host):
                    copied = self.transport._call(helper_context,
                        ['scp', '-q', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', str(root / 'incoming-payload.json'),
                         host + ':' + str(root / 'incoming-payload.json')], left, 'schedule-copy')
                    if copied.status != 'completed' or not copied.cleanup_complete:
                        raise RuntimeError('schedule payload copy did not complete')
                record = self.transport._helper(helper_context, host, ['prepare-file', '--root', str(root), '--host', host],
                                                min(15., deadline - time.monotonic() - reserve), 'schedule-prepare')
                cores[host] = record.get('cpu_cores') if isinstance(record, dict) else None
                prepared.append(host)
            (root / 'tmp').mkdir(exist_ok=True)
            rankfile = root / 'rankfile'
            _immutable(rankfile, unbound_rankfile(mapping, self.transport._is_local, cores))
            # A failed rank still returns nonzero and leaves failure evidence.
            # Let its peers finish their bounded native cleanup before Open MPI
            # exits. Some Open MPI builds return zero in this mode, so durable
            # native and wrapper receipts decide success below.
            command = [str(Path(mpi_bin) / 'mpirun'), '--mca', 'orte_abort_on_non_zero_status', '0',
                       '--rankfile', str(rankfile), '--bind-to', 'none', '-np', str(len(mapping)),
                       self.transport.python, '-m', 'vericcl.verification.online.remote', 'rank', '--root', str(root)]
            if on_prepared is not None:
                on_prepared(payload)
            launched = True
            process = self.transport._call(helper_context, command, deadline - time.monotonic() - reserve,
                                           'schedule-mpi', environment=environment, cancel_check=cancel_check)
            self.context.write_json(relative + '/launcher-result.json', process.as_json())
            if process.status != 'completed' or not process.cleanup_complete:
                failure = 'native schedule MPI launch failed or has unresolved cleanup'
        except (BudgetExhausted, StorageFailure) as error:
            fatal, failure = error, str(error)
        except Exception as error:
            failure = str(error)
        finally:
            for host in prepared:
                try:
                    allowance = max(.2, (deadline - time.monotonic()) / max(1, 2 * len(prepared)))
                    if not launched:
                        self.transport._helper(helper_context, host, ['cancel-before-launch', '--root', str(root)], allowance, 'schedule-cancel')
                    elif failure:
                        cleanups.append(self.transport._helper(helper_context, host,
                            ['cleanup', '--root', str(root), '--host', host, '--timeout', str(min(30., max(.05, allowance / 2)))], allowance, 'schedule-cleanup'))
                except StorageFailure:
                    raise
                except Exception as error:
                    failure = (failure + '; ' if failure else '') + str(error)
            collections, collection_errors = self.transport._collect_evidence(helper_context, prepared, root,
                deadline, 'schedule-collect', relative + '/collected')
            if collection_errors:
                failure = '; '.join(([failure] if failure else []) + [str(item) for item in collection_errors])
        records, execution_receipts = [], []
        for collection in collections:
            for name, data in collection['files'].items():
                if name.startswith('native/rank-') and name.count('/') == 1 and name.endswith('.json'):
                    records.append(json.loads(base64.b64decode(data)))
                elif name.startswith('ranks/') and name.count('/') == 2 and name.endswith('/rank-result.json'):
                    execution_receipts.append(json.loads(base64.b64decode(data)))
        expected_ranks = {m['rank'] for m in mapping}

        def complete_rank_set(items):
            return (len(items) == len(expected_ranks)
                    and all(isinstance(item, dict) and type(item.get('rank')) is int for item in items)
                    and {item['rank'] for item in items} == expected_ranks)

        if launched:
            errors = []
            if not (complete_rank_set(records) and all(r.get('correct') is True
                    and r.get('gpu_completed') is True and not r.get('error') for r in records)):
                errors.append('native rank execution receipts are incomplete or report failure')
            # Reconciliation proves cleanup only. A success JSON may precede a
            # later native failure (for example MPI_Finalize), which the original
            # owning wrapper records independently.
            if not (complete_rank_set(execution_receipts) and all(r.get('status') == 'completed'
                    and type(r.get('returncode')) is int and r['returncode'] == 0 for r in execution_receipts)):
                errors.append('native process execution receipts are incomplete or report failure')
            if errors:
                failure = '; '.join(([failure] if failure else []) + errors)
        wrappers = collected_cleanup_receipts(collections)
        complete = set(wrappers) == expected_ranks
        released = complete and all(r.get('gpu_released') is True for r in wrappers.values())
        clean = complete and all(r.get('cleanup_complete') is True for r in wrappers.values())
        launcher_clean = (not launched or (process is not None and
            (process.cleanup_complete is True or reconcile_owned_process(self.context, process))))
        clean = clean and launcher_clean
        if launched and not released:
            self.transport.quarantined.update((m['host'], m['device']) for m in mapping)
            self.context.write_json('live/native-resource-ownership.json',
                dict(quarantined=[list(item) for item in sorted(self.transport.quarantined)],
                     last_job_id=launch_id, reason='native GPU release receipt missing'), replace=True)
        evidence = dict(launch_id=launch_id, error=failure, gpu_released=released, cleanup_confirmed=clean,
                        collection_errors=collection_errors,
                        launcher_cleanup_confirmed=launcher_clean,
                        rank_results=records, execution_receipts=execution_receipts,
                        cleanup=cleanups, prepared_hosts=prepared, root=str(root))
        self.context.write_json(relative + '/launch-result.json', evidence)
        self.last_evidence = evidence
        if fatal is not None:
            raise fatal
        return root, manifest, records, collections, evidence

    def measure(self, bundle, inputs, launch):
        if launch.trace_enabled:
            raise ValueError('formal performance launches must have trace disabled')
        root, manifest, records, collections, evidence = self._launch(bundle, inputs,
            launch_id=launch.launch_id, warmup=launch.warmup, timed=launch.timed, lease=launch.lease)
        if evidence['error'] or not evidence['gpu_released'] or not evidence['cleanup_confirmed']:
            raise ValueError('native schedule measurement failed: ' + str(evidence))
        sample = combine_schedule_results(records, rank_count=inputs.rank_count,
            version=manifest['version_id'], contract=manifest['contract_digest'], warmup=launch.warmup, timed=launch.timed)
        identity = {'ranks': sorted([dict(rank=r['rank'], facts=r['runtime_identity']) for r in records], key=lambda r: r['rank']),
            'hosts': {c['host']: {key: c['prepared'][key] for key in ('binary_sha256', 'inventory')} for c in collections},
            'environment': dict(inputs.request_config['runtime']['environment']), 'protocol': 'Simple',
            'mpi_launcher_parameters': {'orte_abort_on_non_zero_status': '0'}}
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        self.context.write_json(str(root.relative_to(self.context.root)) + '/measurement.json', sample | dict(runtime_identity=identity, fingerprint=fingerprint))
        return LaunchSample(sample['time_us'], True, True, False, launch.launch_id, 'native_msccl', fingerprint,
                            (str(root / 'measurement.json'), str(root / 'launch-result.json')))

    def _identity(self, inputs, records, collections):
        identity = {'ranks': sorted([dict(rank=r['rank'], facts=r['runtime_identity']) for r in records], key=lambda r: r['rank']),
            'hosts': {c['host']: {key: c['prepared'][key] for key in ('binary_sha256', 'inventory')} for c in collections},
            'environment': dict(inputs.request_config['runtime']['environment']), 'protocol': 'Simple',
            'mpi_launcher_parameters': {'orte_abort_on_non_zero_status': '0'}}
        return identity, hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    def measure_sequence(self, items, *, lease):
        """Measure a planned order of launches in one native process group.

        ``items`` is ``[(bundle, inputs, LaunchRequest)]`` in execution order.
        Every item is an independent sampling episode (its own warmup, timed
        invocations, numerical checks and device-timed receipts) of a version
        prepared once in the shared group, so process startup, communicator
        setup and collection are paid once per sequence instead of per sample.
        Returns one LaunchSample per item, in order.
        """
        if not items:
            return []
        if any(request.trace_enabled for _, _, request in items):
            raise ValueError('formal performance launches must have trace disabled')
        first_bundle, first_inputs, _ = items[0]
        sequence = [(request.launch_id, bundle, item_inputs, request.warmup, request.timed)
                    for bundle, item_inputs, request in items]
        root, manifest, records, collections, evidence = self._launch(first_bundle, first_inputs,
            launch_id='sequence-' + items[0][2].launch_id, warmup=0, timed=1, lease=lease, sequence=sequence)
        if evidence['error'] or not evidence['gpu_released'] or not evidence['cleanup_confirmed']:
            raise ValueError('native sequence measurement failed: ' + str(evidence))
        combine_schedule_results(records, rank_count=first_inputs.rank_count, version=manifest['version_id'],
                                 contract=manifest['contract_digest'], warmup=0, timed=1)
        entries = {}
        # Per-entry receipts are collected artifacts (written below collected/<host>/).
        for host in sorted({item['host'] for item in first_inputs.request_config['runtime']['rank_map']}):
            for path in sorted((root / 'collected' / host / 'native').glob('rank-*/sequence-*.json')):
                record = json.loads(path.read_text())
                entries.setdefault(int(record['index']), []).append(record)
        identity, fingerprint = self._identity(first_inputs, records, collections)
        relative = str(root.relative_to(self.context.root))
        samples = []
        for index, (bundle, item_inputs, request) in enumerate(items):
            # Release is proven once for the whole group (wrapper receipts above).
            ranks = [dict(record, gpu_released=evidence['gpu_released']) for record in entries.get(index, [])]
            if any(record.get('label') != request.launch_id for record in ranks):
                raise ValueError('sequence receipt belongs to another launch')
            sample = combine_schedule_results(ranks, rank_count=item_inputs.rank_count, version=bundle.xml_sha256,
                contract=manifest['contract_digest'], warmup=request.warmup, timed=request.timed)
            path = self.context.write_json('%s/sequence/%04d-measurement.json' % (relative, index),
                sample | dict(launch_id=request.launch_id, xml_sha256=bundle.xml_sha256, runtime_identity=identity,
                              fingerprint=fingerprint, launch_scope='shared_native_sequence'))
            samples.append(LaunchSample(sample['time_us'], True, True, False, request.launch_id, 'native_msccl', fingerprint,
                                        (str(path), str(root / 'launch-result.json'))))
        return samples

    def trace_driver(self, bundle, inputs, lease):
        parent = self
        class TraceDriver:
            def capture_once(self, plan):
                root, _, records, collections, evidence = parent._launch(bundle, inputs,
                    launch_id=plan.capture_id, warmup=plan.warmup_iterations, timed=1, lease=lease, trace_plan=plan)
                paths, metadata, clocks = {}, {}, []
                for mapping in inputs.request_config['runtime']['rank_map']:
                    rank, host = mapping['rank'], mapping['host']
                    base = root / 'collected' / host / 'native'
                    path = base / ('trace.rank-%d.bin' % rank)
                    if path.is_file():
                        paths[rank] = path
                    metadata[rank] = next((r for r in records if r.get('rank') == rank), {})
                    clock = base / ('clock.rank-%d.log' % rank)
                    if clock.is_file():
                        clocks.extend(parse_clock_sync_output(clock.read_text()))
                failure = evidence['error'] or (None if evidence['gpu_released'] and evidence['cleanup_confirmed'] else 'native ownership release is incomplete')
                issues = () if not failure else (dict(rank=None, reason=failure, blocks_acceptance=True),)
                return RawTraceCapture(paths, tuple(clocks), metadata, 'native_msccl', issues)
        return TraceDriver()
