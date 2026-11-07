"""Durable lifecycle snapshots; identities, evidence and consumed budgets survive resume."""
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from enum import Enum
import hashlib
import json
from pathlib import Path
import time
import uuid

from vericcl.execution.budget import BudgetManager, effective_cpu_budget
from vericcl.execution.context import RunContext, StorageFailure
from vericcl.execution.events import emit_event, read_events
from vericcl.execution.security import Redactor
from vericcl.input.loader import load_request
from vericcl.performance.ratios import ratio_to_json
from vericcl.performance.simulation import Prediction
from vericcl.verification.online.model_publication import model_json


def plain(value):
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {field.name: (ratio_to_json(value.exact_completion_ratio)
                            if isinstance(value, Prediction) and field.name == 'exact_completion_ratio'
                            else plain(getattr(value, field.name))) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def build_budgets(inputs, *, purpose='solve'):
    config = inputs.request_config['sketch']['budgets']
    stages = {}
    if 'max_initial_synthesis_s' in config:
        stages['initial'] = (config['max_initial_synthesis_s'],) * 2
    for stage, name in (('online', 'online_validation'), ('offline', 'offline_validation'), ('repair', 'repair')):
        if stage == 'online' and purpose != 'calibrate' and not (inputs.request_config['sketch']['calibration']['enabled'] or inputs.request_config['sketch']['trace_and_measure']['enabled']):
            continue
        per, total = 'max_' + name + '_per_round_s', 'max_' + name + '_total_s'
        if per in config and total in config:
            stages[stage] = (config[per], config[total])
    return BudgetManager(stage_limits=stages, cpu_budget=effective_cpu_budget(inputs),
        per_model_timeout_s=inputs.solver.per_model_timeout_s, max_threads_per_model=inputs.solver.max_threads_per_model,
        max_parallel_models=inputs.solver.max_parallel_models, cleanup_reserve_s=config['cleanup_reserve_s'],
        max_tuning_rounds=config.get('max_tuning_rounds', 0), max_run_wall_s=config.get('max_run_wall_s'))


def persist_candidate(context, bundle, inputs, *, origin, ordinal=None):
    name = context.relative_artifact('candidates/' + (ordinal or uuid.uuid4().hex))
    record = {'record': name + '/candidate.json', 'origin': origin, 'xml_sha256': bundle.xml_sha256, 'can_execute': bundle.can_execute,
        'checks': {key: plain(value) for key, value in bundle.checks.items()},
        'prediction': None if bundle.prediction is None else plain(bundle.prediction),
        'input_sha256': inputs.input_sha256, 'effective_config': plain(inputs.request_config)}
    if bundle.xml_text is not None:
        context.write_bytes(name + '/schedule.xml', bundle.xml_text.encode())
        context.write_json(name + '/sidecar.json', bundle.sidecar)
        record.update(xml=name + '/schedule.xml', sidecar=name + '/sidecar.json')
    context.write_json(name + '/candidate.json', record)
    return record


def _checkpoint_document(context, relative, value):
    """Write an immutable checkpoint document once per distinct content.

    Checkpoint files are never replaced or deleted, and every state records
    and resume verifies the SHA-256 of each referenced file. A later
    checkpoint whose input or model document is byte-identical therefore
    references the earlier immutable copy instead of writing (and fsyncing)
    another one; its own state.json is still new.
    """
    data = context.encode_json(value)
    digest = hashlib.sha256(data).hexdigest()
    shared = getattr(context, 'checkpoint_documents', None)
    if shared is None:
        shared = context.checkpoint_documents = {}
    existing = shared.get(digest)
    if existing is not None and context.artifact(existing).is_file():
        return existing, digest
    path = context.write_bytes(relative, data)
    stored = str(path.relative_to(context.root))
    shared[digest] = stored
    return stored, digest


def save_checkpoint(context, inputs, budgets, model, bundle, *, phase, search=None, bundle_record=None, extra=None):
    prefix = context.relative_artifact('checkpoints/' + uuid.uuid4().hex)
    files, checksums = {}, {}
    for name, document in inputs.request_config.items():
        if name in {'topology', 'sketch', 'atom', 'runtime'}:
            files[name], checksums_value = _checkpoint_document(context, prefix + '/' + name + '.json', plain(document))
            checksums[files[name]] = checksums_value
    model_relative, model_checksum = _checkpoint_document(context, prefix + '/model.json', model_json(model))
    checksums[model_relative] = model_checksum
    if bundle is not None:
        bundle_record = bundle_record or persist_candidate(context, bundle, inputs, origin='checkpoint')
        for name in ('xml', 'sidecar'):
            if name in bundle_record:
                checksums[bundle_record[name]] = hashlib.sha256(context.artifact(bundle_record[name]).read_bytes()).hexdigest()
    initial_version = (extra or {}).get('initial_model_version')
    if initial_version:
        initial_relative = context.relative_artifact('models/' + initial_version + '.json')
        checksums[initial_relative] = hashlib.sha256(context.artifact(initial_relative).read_bytes()).hexdigest()
    trace_reference = (extra or {}).get('best_trace')
    if trace_reference is not None:
        relative = trace_reference['path']
        checksum = hashlib.sha256(context.artifact(relative).read_bytes()).hexdigest()
        if checksum != trace_reference['sha256']:
            raise ValueError('trace causal report checksum differs before checkpoint')
        checksums[relative] = checksum
    state = {'schema_version': 5, 'task_id': context.task_id, 'run_id': context.run_id,
        'phase': phase, 'files': files, 'input_sha256': inputs.input_sha256,
        'value_sources': plain(inputs.value_sources), 'model': model_relative, 'bundle': bundle_record,
        'budgets': budgets.snapshot(), 'search': search or {}, 'extra': extra or {}, 'checksums': checksums,
        'saved_monotonic': time.monotonic(), 'saved_wall': time.time(),
        'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    context.write_json(prefix + '/state.json', state)
    context.write_json('checkpoints/latest.json', {'state': prefix + '/state.json',
        'sha256': hashlib.sha256(context.artifact(prefix + '/state.json').read_bytes()).hexdigest()}, replace=True)
    emit_event(context, 'checkpoint_saved', {'phase': phase, 'state': prefix + '/state.json'})
    return state


def checkpoint_inputs(context, state):
    inputs = load_request({key: context.artifact(value) for key, value in state['files'].items()},
                          purpose=state.get('extra', {}).get('purpose', 'solve'))
    from vericcl.input.schema import _freeze
    return replace(inputs, input_sha256=state['input_sha256'], value_sources=_freeze(state['value_sources']))


def _resume_ownership(context):
    from vericcl.execution.processes import ProcessIdentity, _active_group_members
    for path in context.artifact('jobs').glob('*/identity.json'):
        identity = ProcessIdentity(**json.loads(path.read_text()))
        receipt = path.with_name('result.json')
        if receipt.is_file() and json.loads(receipt.read_text()).get('cleanup_complete') is True:
            continue
        if identity.matches() or _active_group_members(identity):
            raise ValueError('resume requires confirmed termination of owned job: ' + path.parent.name)
        # Absence can discharge a local CPU process, never remote/GPU ownership.
        if not path.parent.name.startswith('scip-'):
            raise ValueError('resume lacks a cleanup receipt for non-solver job: ' + path.parent.name)
    ownership = context.artifact('live/native-resource-ownership.json')
    if ownership.is_file():
        data = json.loads(ownership.read_text())
        if data.get('quarantined') or data.get('unresolved'):
            raise ValueError('resume has unresolved native GPU ownership')


def restore_checkpoint(root):
    root = Path(root).resolve()
    metadata = json.loads((root / 'run.json').read_text())
    # Reuse the original case boundary checks, including existing symlinks.
    from vericcl.artifacts.layout import resolve_case_paths
    project = root.parents[3]
    expected = resolve_case_paths(project, metadata['case_id'], metadata['run_id']).run
    if expected != root or metadata['root'] != str(root):
        raise ValueError('resume path differs from the original case run boundary')
    context = RunContext(root, task_id=metadata['task_id'], run_id=metadata['run_id'], redactor=Redactor())
    latest = json.loads(context.artifact('checkpoints/latest.json').read_text())
    raw = context.artifact(latest['state']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != latest['sha256']:
        raise ValueError('checkpoint state checksum differs')
    state = json.loads(raw)
    if state['task_id'] != context.task_id or state['run_id'] != context.run_id:
        raise ValueError('checkpoint task identity differs')
    for relative, checksum in state['checksums'].items():
        if hashlib.sha256(context.artifact(relative).read_bytes()).hexdigest() != checksum:
            raise ValueError('checkpoint artifact checksum differs: ' + relative)
    _resume_ownership(context)
    scan = read_events(context.artifact('events.jsonl'))
    if any(event['task_id'] != context.task_id or event['run_id'] != context.run_id for event in scan.events):
        raise ValueError('event log task identity differs')
    sequences = [event['sequence'] for event in scan.events]
    if sequences != sorted(set(sequences)):
        raise ValueError('event log sequence is not strictly increasing')
    context.sequence = max(sequences, default=0)
    if scan.truncated_tail:
        original = context.artifact('events.jsonl').read_bytes()
        context.write_bytes('reports/truncated-events-' + uuid.uuid4().hex + '.jsonl', original)
        context.write_bytes('events.jsonl', original[:original.rfind(b'\n') + 1], replace=True)
    budgets = build_budgets(checkpoint_inputs(context, state), purpose=state.get('extra', {}).get('purpose', 'solve'))
    same_boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip() == state['boot_id']
    downtime = time.monotonic() - state['saved_monotonic'] if same_boot else time.time() - state['saved_wall']
    if downtime < 0:
        raise ValueError('resume clock regressed; elapsed budget cannot be established')
    usage = state['budgets']
    consumed, model_work = dict(usage['consumed']), dict(usage['model_work_s'])
    round_consumed = dict(usage.get('round_consumed', {}))
    for item in usage['active_leases']:
        elapsed = item['elapsed_s'] + min(downtime, max(0., item['deadline'] - state['saved_monotonic']))
        target = model_work if item['model'] else consumed
        target[item['stage']] += elapsed
        if not item['model']:
            round_consumed[item['stage']] = round_consumed.get(item['stage'], 0) + elapsed
    budgets.restore_usage(consumed, started_rounds=usage['started_rounds'],
        elapsed_wall_s=usage['elapsed_wall_s'] + downtime, model_work_s=model_work, round_consumed=round_consumed)
    emit_event(context, 'run_resumed', {'phase': state['phase'], 'downtime_s': downtime, 'budgets': budgets.snapshot()})
    return context, state, budgets
