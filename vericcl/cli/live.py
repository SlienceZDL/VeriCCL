"""A finite, reproducible driver for the persistent C API session."""
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import json
import math
import time
import traceback

from vericcl.cli.lifecycle import _overrides
from vericcl.errors import InputValidationError, SolverUnavailableError
from vericcl.execution.context import RunContext, StorageFailure, finalize_run
from vericcl.execution.events import emit_event
from vericcl.execution.status import RunResult
from vericcl.input.loader import load_request, _reject_duplicate_keys, _reject_nonfinite_json
from vericcl.live.controller import NativeLiveSession
from vericcl.performance.serialization import load_model
from vericcl.workflow_v5 import PrimaryRequest, PrimaryWorkflow


def run_requests(session, count, *, controller=None):
    receipts, previous = [], 0
    for _ in range(count):
        if controller is not None:
            controller.before_request()
        receipt = session.execute()
        if receipt.get('post_completion_error'):
            raise RuntimeError('completed live request has an unresolved post-completion action: ' + receipt['post_completion_error'])
        ranks = receipt.get('ranks', ())
        if (receipt.get('logical_success') is not True or len(ranks) != session.inputs.rank_count
                or {r.get('rank') for r in ranks} != set(range(session.inputs.rank_count))
                or any(r.get('gpu_completed') is not True or r.get('correct') is not True for r in ranks)):
            raise RuntimeError('live request lacks a numerically checked complete group receipt')
        identity, version = receipt.get('request_id'), receipt.get('version_id')
        if (not isinstance(identity, int) or isinstance(identity, bool) or identity <= previous or not version
                or any(r.get('request_id') != identity or r.get('version_id') != version for r in ranks)):
            raise RuntimeError('live logical request identity is duplicated or inconsistent')
        previous = identity; receipts.append(receipt)
        if controller is not None:
            controller.observe(receipt)
    return receipts


def live_run(args):
    if args.requests < 1 or not math.isfinite(args.session_timeout_s) or args.session_timeout_s <= 0:
        raise InputValidationError('live requires positive --requests and a finite positive --session-timeout-s')
    if (args.xml is None) != (args.sidecar is None):
        raise InputValidationError('live --xml and --sidecar must be supplied together')
    files = {name: getattr(args, name) for name in ('topology', 'sketch', 'atom', 'runtime')}
    inputs = load_request(files, _overrides(args.overrides), override_input=args.override_input, purpose='live')
    if not inputs.request_config['sketch']['live']['enabled']:
        raise InputValidationError('live requires sketch.live.enabled=true')
    xml = args.xml.read_text() if args.xml else None
    sidecar = json.loads(args.sidecar.read_text(), object_pairs_hook=_reject_duplicate_keys,
                         parse_constant=_reject_nonfinite_json) if args.sidecar else None
    model = load_model(args.model) if args.model else None
    snapshots = {name + '.json': path for name, path in files.items()}
    if args.xml:
        snapshots.update({'schedule.xml': args.xml, 'sidecar.json': args.sidecar})
    if args.model:
        snapshots['model.json'] = args.model
    run_id = args.run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    context = RunContext.create(args.project_root, args.case, run_id, input_files=snapshots,
                                environment=dict(inputs.request_config['runtime']['environment']))
    context.write_json('invocation.json', {'argv': getattr(args, 'invocation_argv', []),
        'cwd': str(Path.cwd().resolve()), 'command': 'live'})
    session = controller = None
    result = RunResult(False, 'not_started', stop_reason='live_not_started', failure_category='online')
    try:
        workflow = PrimaryWorkflow(PrimaryRequest(inputs, purpose='verify' if xml else 'solve',
            xml_text=xml, sidecar=sidecar, cost_model=model), context)
        result = workflow.run()
        if result.contract_met:
            # The task wall clock remains the parent of the separate live clock.
            seconds = min(args.session_timeout_s, workflow.budgets._global_deadline() - time.monotonic())
            session = NativeLiveSession(context, workflow.best_inputs, workflow.best, timeout_s=seconds,
                                        project_root=args.project_root)
            session.start()
            from vericcl.live.adaptive import AdaptiveLiveController
            controller = AdaptiveLiveController(session, workflow)
            receipts = run_requests(session, args.requests, controller=controller)
            context.write_json('live/requests.json', dict(requested=args.requests, completed=len(receipts),
                request_ids=[r['request_id'] for r in receipts], version_ids=[r['version_id'] for r in receipts]))
            result = replace(result, stop_reason='live_request_count')
    except StorageFailure:
        raise
    except BaseException as error:
        category = 'cancelled' if isinstance(error, KeyboardInterrupt) else 'backend' if isinstance(error, SolverUnavailableError) else 'online'
        emit_event(context, 'live_exception', {'message': str(error), 'traceback': traceback.format_exc()})
        result = RunResult(False, 'cancelled' if category == 'cancelled' else 'failed', stop_reason='live_exception',
                           failure_category=category, optional_failures=(str(error),))
    finally:
        if session is not None:
            deadline = time.monotonic() + session.settings['close_cleanup_timeout_s']
            if controller is not None and not controller.close(max(0., deadline - time.monotonic())):
                result = RunResult(False, 'incomplete', stop_reason='live_background_cleanup_unresolved', failure_category='online')
            closed = session.close(timeout_s=max(0., deadline - time.monotonic()))
            if closed.get('status') == 'failed':
                result = RunResult(False, 'failed', stop_reason='live_execution_failed', failure_category='online',
                                   optional_failures=(str(closed),))
            elif closed.get('status') != 'completed':
                result = RunResult(False, 'incomplete', stop_reason='live_ownership_unresolved', failure_category='online',
                                   optional_failures=(str(closed),))
    if session is not None and not session.finished.is_set():
        # The owned supervisor still writes evidence. Its parent deadline and
        # cancellation remain active; do not seal this run underneath it.
        print('VeriCCL：常驻清理尚未完成，资源保留；目录=' + str(context.root))
        return result.exit_code
    finalized = finalize_run(context, result)
    print('VeriCCL：状态=%s，结束原因=%s，退出码=%d，目录=%s' %
          (result.execution, result.stop_reason, finalized.exit_code, context.root))
    return finalized.exit_code
