"""CLI orchestration; application work starts after snapshots and input validation."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback
import uuid

from vericcl.errors import InputValidationError, SolverUnavailableError
from vericcl.execution.context import RunContext, StorageFailure, finalize_run
from vericcl.execution.events import emit_event
from vericcl.execution.status import RunResult
from vericcl.input.loader import load_request, _reject_duplicate_keys, _reject_nonfinite_json
from vericcl.performance.serialization import load_model
from vericcl.workflow_state import restore_checkpoint, checkpoint_inputs
from vericcl.workflow_v5 import PrimaryRequest, run_primary


def _overrides(values):
    result = {}
    for item in values:
        if '=' not in item:
            raise InputValidationError('--set requires FIELD=JSON')
        path, text = item.split('=', 1)
        if path in result:
            raise InputValidationError('duplicate --set field: ' + path)
        result[path] = json.loads(text, object_pairs_hook=_reject_duplicate_keys, parse_constant=_reject_nonfinite_json)
    return result


def _execute(request, context, **kwargs):
    try:
        result = run_primary(request, context, **kwargs)
    except StorageFailure:
        raise
    except BaseException as error:
        if isinstance(error, KeyboardInterrupt):
            category, execution = 'cancelled', 'cancelled'
        elif isinstance(error, SolverUnavailableError):
            category, execution = 'backend', 'failed'
        else:
            category, execution = 'internal', 'failed'
        emit_event(context, 'execution_exception', {'message': str(error), 'traceback': traceback.format_exc()})
        result = RunResult(False, execution, stop_reason='exception', failure_category=category,
                           optional_failures=(str(error),))
    finalized = finalize_run(context, result)
    print('VeriCCL: status=%s, stop_reason=%s, model_coverage=%s, exit_code=%d, directory=%s' %
          (result.execution, result.stop_reason, result.model_coverage, finalized.exit_code, context.root))
    return finalized.exit_code


def new_run(args):
    files = {name: getattr(args, name) for name in ('topology', 'sketch', 'atom', 'runtime')}
    inputs = load_request(files, _overrides(args.overrides), override_input=args.override_input, purpose=args.command)
    if args.command == 'verify' and args.tune:
        budgets = inputs.request_config['sketch']['budgets']
        required = ('max_tuning_rounds', 'max_repair_per_round_s', 'max_repair_total_s')
        for key in required:
            if key not in budgets:
                raise InputValidationError('verify --tune requires sketch.budgets.' + key)
    xml = args.xml.read_text() if args.command == 'verify' else None
    sidecar = json.loads(args.sidecar.read_text(), object_pairs_hook=_reject_duplicate_keys,
                         parse_constant=_reject_nonfinite_json) if args.command == 'verify' else None
    model = load_model(args.model) if getattr(args, 'model', None) is not None else None
    run_id = args.run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    snapshots = {name + '.json': path for name, path in files.items()}
    if args.command == 'verify':
        snapshots.update({'schedule.xml': args.xml, 'sidecar.json': args.sidecar})
        if args.model is not None:
            snapshots['model.json'] = args.model
    if args.command == 'solve' and inputs.request_config['sketch'].get('size_bands'):
        # One standard solve per size band, then one MSCCL_XML_FILES bundle.
        from vericcl.bands import run_bands, size_bands
        size_bands(inputs)
        args.run_id_resolved = run_id
        args.initial_model = model
        overrides = _overrides(args.overrides)

        def create(run):
            created = RunContext.create(args.project_root, args.case, run, input_files=snapshots,
                                        environment=dict(inputs.request_config['runtime']['environment']))
            created.write_json('invocation.json', {'argv': getattr(args, 'invocation_argv', []),
                'cwd': str(Path.cwd().resolve()), 'command': args.command})
            return created

        def load(band_files, band_overrides):
            return load_request(band_files, band_overrides, override_input=True, purpose='solve')
        return run_bands(args, files, overrides, inputs, execute=_execute, create_context=create, load=load)
    context = RunContext.create(args.project_root, args.case, run_id, input_files=snapshots,
                                environment=dict(inputs.request_config['runtime']['environment']))
    context.write_json('invocation.json', {'argv': getattr(args, 'invocation_argv', []),
        'cwd': str(Path.cwd().resolve()), 'command': args.command})
    return _execute(PrimaryRequest(inputs, purpose=args.command, xml_text=xml, sidecar=sidecar,
                                   tune=getattr(args, 'tune', False), cost_model=model), context)


def inspect_run(args):
    root = args.run.resolve()
    metadata = json.loads((root / 'run.json').read_text())
    result = {'run': str(root), 'task_id': metadata['task_id']}
    for name, relative in (('summary', 'run-summary.json'), ('best', 'best/accepted.json'), ('checkpoint', 'checkpoints/latest.json')):
        path = root / relative
        if path.exists():
            if not path.resolve().is_relative_to(root):
                raise InputValidationError('inspect artifact escapes run boundary')
            result[name] = json.loads(path.read_text())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def resume_run(args):
    context, state, budgets = restore_checkpoint(args.run)
    if state['phase'] == 'finished':
        summary = context.artifact('run-summary.json')
        if summary.is_file():
            result = json.loads(summary.read_text())
            print('VeriCCL: task already finished; preserving the original result. stop_reason=%s, directory=%s' % (result['stop_reason'], context.root))
            return int(result['exit_code'])
    inputs = checkpoint_inputs(context, state)
    original_config = state.get('extra', {}).get('original_config')
    original_files = state.get('extra', {}).get('original_files')
    purpose = state.get('extra', {}).get('purpose', 'solve')
    if original_files:
        for relative in original_files.values():
            if hashlib.sha256(context.artifact(relative).read_bytes()).hexdigest() != state['checksums'].get(relative):
                raise ValueError('checkpoint original request document changed: ' + relative)
        inputs = load_request({name: context.artifact(relative) for name, relative in original_files.items()}, purpose=purpose)
    elif original_config:
        prefix = 'tmp/resume-original-' + uuid.uuid4().hex
        files = {name: context.write_json(prefix + '/' + name + '.json', value) for name, value in original_config.items()}
        inputs = load_request(files, purpose=purpose)
    context.resumed = True
    return _execute(PrimaryRequest(inputs, purpose=purpose, tune=state.get('extra', {}).get('tune', False)),
                    context, restored=state, budgets=budgets)
