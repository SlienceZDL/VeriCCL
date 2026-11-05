"""Actual persistent requests and immutable version resources; no speedup claim."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os
from vericcl.verification.online.remote import owned_home_path
from vericcl.artifacts.layout import resolve_case_paths

from tests.integration.test_lowering_v5 import candidate_fixture, model_for
from vericcl.cli.live import run_requests
from vericcl.execution.budget import StandaloneLease
from vericcl.execution.context import RunContext, finalize_run
from vericcl.execution.status import RunResult
from vericcl.input.loader import load_request
from vericcl.input.schema import LIVE_TIMEOUTS
from vericcl.live.controller import NativeLiveSession
from vericcl.topology import load_topology
from vericcl.workflow_state import plain
from vericcl.xml.lower import lower_and_check
from vericcl.xml.v5_lower import evaluate_program
from vericcl.xml.v5_program import emit_native_xml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--driver', required=True)
    parser.add_argument('--port', required=True, type=int)
    parser.add_argument('--project-root', type=Path)
    parser.add_argument('--devices', required=True, help='two local GPU indices, for example 0,1')
    parser.add_argument('--mpi-bin', required=True)
    parser.add_argument('--inplace', action='store_true')
    parser.add_argument('--inject-attempts', type=int, default=0)
    parser.add_argument('--validation', action='store_true')
    parser.add_argument('--paired-validation', action='store_true')
    parser.add_argument('--expect-exhausted', action='store_true')
    args = parser.parse_args()
    if args.expect_exhausted and args.inject_attempts <= 2:
        parser.error('expected exhaustion requires more than two injected attempts')
    if args.paired_validation:
        args.validation = True
    try:
        devices = [int(value) for value in args.devices.split(',')]
        if len(devices) != 2 or len(set(devices)) != 2 or min(devices) < 0:
            raise ValueError('two distinct nonnegative GPU indices are required')
    except ValueError as error:
        parser.error(str(error))
    root = owned_home_path(args.project_root or Path(__file__).resolve().parents[1])
    paths = resolve_case_paths(root, 'native-live', args.run_id)
    source = paths.input / args.run_id
    source.mkdir(parents=True)
    candidate, lowering = candidate_fixture(source, 'allreduce', inplace=args.inplace)
    config = plain(lowering.inputs.request_config)
    config['sketch']['live'].update(enabled=True, **{key: 15 for key in LIVE_TIMEOUTS})
    if args.inject_attempts:
        config['sketch']['live']['fault_injection'] = dict(request_id=1, attempts=args.inject_attempts)
    config['sketch']['solver']['max_channels'] = 3
    if args.validation:
        config['sketch']['trace_and_measure'].update(enabled=True, job_timeout_s=30, trace_capacity_bytes=1048576)
        config['sketch']['budgets'].update(max_online_validation_per_round_s=60, max_online_validation_total_s=120)
    if args.paired_validation:
        config['sketch']['trace_and_measure'].update(pairs=6, warmup_iterations=1, timed_iterations=2, max_rounds=1)
    config['runtime']['rank_map'] = [dict(rank=r, host='localhost', device=devices[r]) for r in range(2)]
    config['runtime']['paths'] = dict(vericcl_native=args.driver, mpi_bin=args.mpi_bin)
    config['runtime']['control'] = dict(bind_address='127.0.0.1', port=args.port,
        unix_socket=str(root / 'examples/native-live' / ('control-' + args.run_id + '.sock')))
    config['runtime']['environment'].update(NCCL_DEBUG='WARN')
    if os.environ.get('LD_LIBRARY_PATH'):
        config['runtime']['environment']['LD_LIBRARY_PATH'] = os.environ['LD_LIBRARY_PATH']
    files = {}
    for name, value in config.items():
        path = source / (name + '.json'); path.write_text(json.dumps(value, indent=2) + '\n'); files[name] = path
    inputs = load_request(files, purpose='live'); topology = load_topology(inputs)
    lowering = replace(lowering, inputs=inputs, topology=topology, model=model_for(topology), lease=StandaloneLease.after(20))
    baseline = lower_and_check(candidate, lowering)
    assert baseline.can_execute
    versions = [baseline]
    for channel in (1, 2):
        storage = replace(baseline.storage_plan, instructions=tuple(
            replace(op, channel=channel) if op.kind != 'copy' else op for op in baseline.storage_plan.instructions))
        program = replace(baseline.program, channels=channel + 1, blocks=tuple(
            replace(block, channel=channel) if block.send_peer >= 0 or block.recv_peer >= 0 else block for block in baseline.program.blocks))
        bundle = evaluate_program(candidate, storage, program, emit_native_xml(program, inputs), lowering)
        assert bundle.can_execute, bundle.checks
        versions.append(bundle)
    run = RunContext.create(root, 'native-live', args.run_id,
                            input_files={name + '.json': path for name, path in files.items()})
    run.write_json('invocation.json', dict(argv=__import__('sys').argv, scope=__doc__))
    for i, bundle in enumerate(versions):
        run.write_bytes('candidates/version-%d.xml' % i, bundle.xml_text.encode())
        run.write_json('candidates/version-%d.sidecar.json' % i, bundle.sidecar)
    session = NativeLiveSession(run, inputs, baseline, timeout_s=120, project_root=root)
    success = False; receipts = []; states = []; validation = []; expected_error = None; paired = None
    try:
        session.start()
        for index, version_index in enumerate((0, 1, 0, 2)):
            if index:
                prepared = session.prepare_version(versions[version_index], inputs)
                assert prepared.get('prepared') and not prepared.get('activated')
            receipt = run_requests(session, 1)[0]
            assert receipt['request_id'] > (receipts[-1]['request_id'] if receipts else 0) and receipt['version_id'] == versions[version_index].xml_sha256
            assert receipt['attempt'] == (args.inject_attempts if index == 0 else 0)
            assert all(r['telemetry'] and all(step['samples'] == 1 and not step['invalid_samples'] for step in r['telemetry']) for r in receipt['ranks'])
            receipts.append(receipt)
            states.append(session.client.status(2))
            if args.validation and index == 0:
                from vericcl.live.validation import LiveBoundaryValidator
                from vericcl.verification.online.trace_once import capture_once, TraceContext
                from vericcl.verification.online.trace_format import RAW_HEADER_STRUCT, RAW_RECORD_STRUCT
                from vericcl.verification.online.acceptance import LaunchRequest
                from vericcl.xml.trace_sidecar import build_v5_trace_sidecar
                validator = LiveBoundaryValidator(session)
                for cycle in range(2):
                    phase = StandaloneLease.after(30)
                    capture = capture_once(build_v5_trace_sidecar(baseline), TraceContext(True, (0, 1),
                        validator.trace_driver(baseline, inputs),
                        {rank: (1048576 - RAW_HEADER_STRUCT.size) // RAW_RECORD_STRUCT.size for rank in range(2)}, phase, run))
                    validation.append(dict(cycle=cycle, trace_status=capture.status, complete=capture.global_complete,
                                           capture_id=capture.capture_id, issues=plain(capture.issues)))
                    assert capture.global_complete, capture.issues
                    sample = validator.measure(baseline, inputs, LaunchRequest('cycle-%d' % cycle, 1, 1, 1, 'A',
                        warmup=1, timed=3, lease=phase))
                    assert sample.correct and sample.gpu_work_quiescent and sample.resources_retained and not sample.gpu_released
                    validation[-1]['trace_off_sample'] = plain(sample)
                if args.paired_validation:
                    paired = validator(baseline, versions[1], inputs, StandaloneLease.after(60))
                    assert paired['trace_complete'] and paired['measured']['status'] in {'accepted', 'no_measured_gain', 'unstable'}, paired
        assert states[-1]['status']['retained_versions'] == 2
        success = True
    except RuntimeError as error:
        if not args.expect_exhausted:
            raise
        expected_error = str(error)
    finally:
        closed = session.close()
        exhaustion = {}
        if args.expect_exhausted and expected_error and not receipts:
            base = Path(session.payload['root'])
            for rank in range(2):
                records = [json.loads(path.read_text()) for path in sorted(base.glob('collected/*/native/rank-%d/recovery-request-1-attempt-*.json' % rank))]
                exhaustion[str(rank)] = {'attempts': [r['attempt'] for r in records],
                    'deadline_count': len({r['recovery_deadline_ns'] for r in records}),
                    'sources': sorted({r['source'] for r in records})}
            success = all(value == {'attempts': [0, 1, 2], 'deadline_count': 1, 'sources': ['fault_injection']} for value in exhaustion.values())
        success = success and closed['status'] == 'completed' and closed['gpu_released']
        report = dict(success=success, request_ids=[r['request_id'] for r in receipts],
            version_ids=[r['version_id'] for r in receipts], states=states, close=closed,
            actual_gpu_execution=True, recovery_tested=bool(args.inject_attempts),
            fault_source='fault_injection' if args.inject_attempts else None, inplace=args.inplace, speedup_claim=False)
        report['boundary_validation'] = validation
        report.update(expected_exhaustion=args.expect_exhausted, expected_error=expected_error,
                      exhaustion=exhaustion, paired_validation=paired)
        run.write_json('reports/native-live.json', report)
        print(json.dumps(report))
        if session.finished.is_set():
            finalize_run(run, RunResult(success, 'completed' if success else 'failed',
                stop_reason='live_version_smoke', failure_category=None if success else 'online'))
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
