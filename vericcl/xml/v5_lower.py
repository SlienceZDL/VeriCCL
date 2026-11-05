"""The common v5 candidate admission boundary and reloadable XML sidecar."""
from dataclasses import asdict, replace
from collections.abc import Mapping
from types import MappingProxyType
import hashlib
import json

from lxml import etree

from vericcl.errors import SemanticError
from vericcl.execution.budget import BudgetExhausted
from vericcl.input.schema import DATATYPE_BYTES
from vericcl.performance.simulation import predict
from vericcl.semantics.collective import OutputSlot, CollectiveKind, required_outputs
from vericcl.topology.model import LinkKey
from vericcl.semantics.ranges import DataRange
from vericcl.semantics.versions import StateVersion
from vericcl.semantics.version_schedule import VersionSchedule, VersionOperation, OutputBinding, replay_version_schedule
from vericcl.verification.model import CheckResult, ValidationStatus
from vericcl.xml.capacity import ExecutionShape, BlockCapacity, StepCapacity, check_capacity, simple_payload_capacity
from .v5_model import StorageCandidate, CandidateBundle, Value, Instruction, StoragePlan, NativeStep, NativeBlock, NativeProgram, Address
from .v5_buffers import build_storage_plan, canonical_region
from .v5_program import lower_program, build_executable, emit_native_xml, execution_dependencies, _endpoints, _dependency_ancestors


DIMENSIONS = ('semantic', 'storage', 'endpoint', 'deadlock', 'xml', 'launch', 'compatibility', 'prediction')


def _check(dimension, status='valid', message='checked', evidence=None):
    return CheckResult(dimension, ValidationStatus(status), 'v5_' + dimension + '_' + status, message, evidence or {})


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (frozenset, set)):
        return [_plain(item) for item in sorted(value)]
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _serialize_storage(storage):
    return {'values': [_plain(asdict(value)) for value in storage.values.values()],
        'initial_ids': list(storage.initial_ids), 'instructions': [_plain(asdict(op)) for op in storage.instructions],
        'addresses': {key: asdict(value) for key, value in storage.addresses.items()},
        'outputs': [{'rank': slot.rank, 'offset': slot.offset, 'state_id': key} for slot, key in storage.outputs.items()],
        'input_slots': dict(storage.input_slots), 'output_slots': dict(storage.output_slots),
        'scratch_slots': dict(storage.scratch_slots), 'reuse': list(storage.reuse), 'stage_count': storage.stage_count,
        'reference_orders_omitted': list(storage.reference_orders_omitted)}


def _deserialize_storage(raw):
    values = {value['state_id']: Value(**{**value, 'contributors': frozenset(value['contributors'])}) for value in raw['values']}
    if len(values) != len(raw['values']):
        raise SemanticError('duplicate state identities in sidecar')
    instructions = tuple(Instruction(**{**value, 'inputs': tuple(value['inputs']),
        'participants': tuple(value['participants']), 'dependencies': frozenset(value['dependencies'])}) for value in raw['instructions'])
    addresses = {key: Address(**value) for key, value in raw['addresses'].items()}
    outputs = {OutputSlot(value['rank'], value['offset']): value['state_id'] for value in raw['outputs']}
    if len(outputs) != len(raw['outputs']):
        raise SemanticError('duplicate output bindings in sidecar')
    return StoragePlan(values, tuple(raw['initial_ids']), instructions, addresses, outputs,
        {int(key): value for key, value in raw['input_slots'].items()},
        {int(key): value for key, value in raw['output_slots'].items()},
        {int(key): value for key, value in raw['scratch_slots'].items()}, tuple(raw['reuse']), raw['stage_count'],
        tuple(tuple(pair) for pair in raw['reference_orders_omitted']))


def _serialize_program(program):
    return {'rank_count': program.rank_count, 'channels': program.channels, 'chunks_per_loop': program.chunks_per_loop,
        'blocks': [_plain(asdict(block)) for block in program.blocks], 'input_slots': dict(program.input_slots),
        'output_slots': dict(program.output_slots), 'scratch_slots': dict(program.scratch_slots)}


def _deserialize_program(raw):
    blocks = []
    for block in raw['blocks']:
        steps = tuple(NativeStep(**{**step, 'src': None if step['src'] is None else Address(**step['src']),
                                   'dst': None if step['dst'] is None else Address(**step['dst'])}) for step in block['steps'])
        blocks.append(NativeBlock(**{**block, 'steps': steps}))
    return NativeProgram(raw['rank_count'], raw['channels'], raw['chunks_per_loop'], tuple(blocks),
        {int(key): value for key, value in raw['input_slots'].items()},
        {int(key): value for key, value in raw['output_slots'].items()},
        {int(key): value for key, value in raw['scratch_slots'].items()})


def _version_schedule(storage, executable, prediction, inputs):
    n, size = inputs.hyperparameters.slice_count, inputs.hyperparameters.slice_size_bytes
    initial = []
    for key in storage.initial_ids:
        value = storage.values[key]
        expected_address = canonical_region(Address(value.rank, 'i', value.logical_position), inputs)
        actual_address = canonical_region(storage.addresses[key], inputs)
        if expected_address != actual_address:
            raise SemanticError('initial state does not use its actual runtime input address')
        initial.append(StateVersion(key, value.rank, value.logical_position * size, (value.logical_position + 1) * size,
            tuple(DataRange(s // n, (s % n) * size, (s % n + 1) * size) for s in sorted(value.contributors)), actual_address, 0.))
    work = {op.operation_id: op for op in executable.operations}
    if prediction.t_pred_us is None:
        # Untimed data/alias validation is still useful when a cost component
        # is absent. No execution admission or T_pred comes from these ticks.
        starts, ends, pending = {}, {}, set(work)
        while pending:
            ready = sorted(key for key in pending if set(work[key].dependencies) <= ends.keys())
            if not ready:
                raise SemanticError('semantic replay has a dependency cycle')
            for key in ready:
                starts[key] = max((ends[p] for p in work[key].dependencies), default=0.)
                ends[key] = starts[key] + 1.; pending.remove(key)
    else:
        starts, ends = prediction.start_times, prediction.end_times
    operations = tuple(VersionOperation(op.operation_id, 'reduce' if op.kind in {'rrc', 'reduce'} else 'copy',
        op.inputs, op.output, canonical_region(storage.addresses[op.output], inputs), starts[op.operation_id], ends[op.operation_id],
        op.stage_id, work[op.operation_id].dependencies, op.participants) for op in storage.instructions)
    outputs = tuple(OutputBinding(slot.rank, slot.offset * size, key) for slot, key in storage.outputs.items())
    return VersionSchedule(inputs.rank_count, inputs.hyperparameters.total_size_bytes, size, storage.stage_count,
                           tuple(initial), operations, outputs)


def _validate_bindings(program, storage, context):
    dependencies, actual = execution_dependencies(program, lease=context.lease)
    if len({op.operation_id for op in storage.instructions}) != len(storage.instructions):
        raise SemanticError('semantic operation identities are duplicated')
    if actual.keys() != {op.operation_id for op in storage.instructions}:
        raise SemanticError('native and semantic operation coverage differs')
    # Validate local memory order independently of the collapsed transfer DAG.
    # Matching endpoints do not begin or complete atomically on two GPUs.
    local_edges, local_locations = {}, {}
    for block in program.blocks:
        context.lease.check()
        for index, step in enumerate(block.steps):
            key = (block.rank, block.tb_id, index)
            edges = set()
            if index:
                edges.add((block.rank, block.tb_id, index - 1))
            if step.dependency_tb >= 0:
                edges.add((block.rank, step.dependency_tb, step.dependency_step))
            local_edges.setdefault(block.rank, {})[key] = edges
            if step.kind != 'nop':
                local_locations[step.operation_id, block.rank] = key
    ancestors, bits = _dependency_ancestors(dependencies, context.lease)
    local_ancestors = {rank: _dependency_ancestors(edges, context.lease)
                       for rank, edges in local_edges.items()}
    for op in storage.instructions:
        context.lease.check()
        if any(predecessor not in bits or not (ancestors[op.operation_id] & bits[predecessor])
               for predecessor in op.dependencies):
            raise SemanticError('native program omits a required data or storage dependency')
        for block, step in actual[op.operation_id]:
            required = {local_locations[predecessor, block.rank] for predecessor in op.dependencies
                        if (predecessor, block.rank) in local_locations}
            local_masks, local_bits = local_ancestors[block.rank]
            mask = local_masks[local_locations[op.operation_id, block.rank]]
            if any(not (mask & local_bits[key]) for key in required):
                raise SemanticError('native local endpoint omits a required data or storage dependency')
        expected = _endpoints(op, storage)
        seen = [(block.rank, step.kind, step.src, step.dst) for block, step in actual[op.operation_id]]
        if sorted(seen, key=lambda value: (value[0], value[1])) != sorted(
                [(rank, kind, src, dst) for rank, direction, peer, channel, kind, src, dst in expected], key=lambda value: (value[0], value[1])):
            raise SemanticError('native buffer references or endpoint type disagree with state bindings')
        for block, step in actual[op.operation_id]:
            if op.kind != 'copy' and block.channel != op.channel:
                raise SemanticError('native channel differs from the selected channel')
            used = ([step.src] if step.kind == 's' else [step.dst] if step.kind == 'r'
                    else [step.src, step.dst] if step.kind in {'rrc', 'cpy', 're'} else [])
            for address in used:
                slots = {'i': program.input_slots, 'o': program.output_slots, 's': program.scratch_slots}
                if address is None or address.rank != block.rank or address.buffer not in slots or not 0 <= address.offset < slots[address.buffer][block.rank]:
                    raise SemanticError('native buffer address is outside its declared storage')


def _capacity(program, context):
    shape = ExecutionShape(program.rank_count, program.channels,
        tuple(BlockCapacity(block.rank, block.tb_id, block.channel, block.send_peer, block.recv_peer,
            tuple(StepCapacity(step.kind, -1 if step.src is None else step.src.offset, -1 if step.dst is None else step.dst.offset,
                step.dependency_tb, step.dependency_step, step.count) for step in block.steps)) for block in program.blocks),
        program.scratch_slots, program.chunks_per_loop)
    return check_capacity(shape, context.limits, slice_bytes=context.inputs.hyperparameters.slice_size_bytes,
        scratch_bytes_per_rank=context.inputs.request_config['runtime']['scratch_bytes_per_rank'])


def _request_contract(inputs):
    return {'ranks': inputs.rank_count, 'input_bytes': inputs.hyperparameters.total_size_bytes,
            'slice_bytes': inputs.hyperparameters.slice_size_bytes, 'operator': inputs.collective.kind.value,
            'datatype': inputs.collective.datatype, 'reduction_op': inputs.collective.reduction_op,
            'root': inputs.collective.root, 'inplace': inputs.collective.inplace}


def _launch_geometry(program, storage, context):
    inputs = context.inputs
    ranks, n = inputs.rank_count, inputs.hyperparameters.slice_count
    expected = required_outputs(inputs.collective, ranks, n)
    output_slots = {rank: max((slot.offset + 1 for slot in expected if slot.rank == rank), default=0) for rank in range(ranks)}
    chunks = n * ranks if inputs.collective.kind is CollectiveKind.ALL_GATHER else n
    if (program.rank_count != ranks or program.chunks_per_loop != chunks
            or program.input_slots != {rank: n for rank in range(ranks)} or program.output_slots != output_slots
            or program.input_slots != storage.input_slots or program.output_slots != storage.output_slots
            or program.scratch_slots != storage.scratch_slots):
        raise SemanticError('native runtime geometry differs from the resolved collective and partition')
    if not 1 <= program.channels <= inputs.solver.max_channels:
        raise SemanticError('native channel count exceeds the selected channel limit')
    for block in program.blocks:
        edges = ([] if block.send_peer < 0 else [LinkKey(block.rank, block.send_peer)])
        edges += [] if block.recv_peer < 0 else [LinkKey(block.recv_peer, block.rank)]
        for key in edges:
            edge = context.topology.link(key)
            maximum = min([inputs.solver.max_channels, edge.max_channels]
                          + [context.topology.shared_resources[name].max_channels for name in edge.resource_ids])
            if not 0 <= block.channel < maximum:
                raise SemanticError('native channel exceeds the directed link or shared-resource limit')


def _validate_xml_text(xml_text, program, inputs):
    parser = etree.XMLParser(resolve_entities=False, no_network=True, remove_blank_text=True)
    try:
        received = etree.fromstring(xml_text.encode(), parser)
        expected = etree.fromstring(emit_native_xml(program, inputs).encode(), parser)
    except (etree.XMLSyntaxError, ValueError) as error:
        raise SemanticError('XML is not well formed') from error
    if received.getroottree().docinfo.doctype:
        raise SemanticError('native XML cannot declare a DTD')
    if etree.tostring(received, method='c14n') != etree.tostring(expected, method='c14n'):
        raise SemanticError('XML structure or fields disagree with the physical step sidecar')


def _check_storage_values(storage, version, context):
    ledger, evidence = replay_version_schedule(version, context.inputs, lease=context.lease)
    n, size = context.inputs.hyperparameters.slice_count, context.inputs.hyperparameters.slice_size_bytes
    for state in ledger.states:
        context.lease.check()
        value = storage.values.get(state.state_id)
        if value is None or (state.rank, state.logical_start, state.logical_end) != (value.rank, value.logical_position * size, (value.logical_position + 1) * size):
            raise SemanticError('stored version and abstract state identity differ')
        expected = tuple(DataRange(s // n, (s % n) * size, (s % n + 1) * size) for s in sorted(value.contributors))
        if state.contributors != expected:
            raise SemanticError('stored version has different source contributions')
    for slot, key in storage.outputs.items():
        context.lease.check()
        if ledger.state(key).storage != canonical_region(Address(slot.rank, 'o', slot.offset), context.inputs):
            raise SemanticError('final result is not in the actual runtime output address')
    return evidence


def evaluate_program(source, storage, program, xml_text, context, *, supplied_sidecar=None):
    context.lease.check()
    checks = {key: _check(key, 'not_run', 'prerequisite pending') for key in DIMENSIONS}
    prediction = version = executable = capacity = None
    xml_hash = hashlib.sha256(xml_text.encode()).hexdigest()
    sidecar = supplied_sidecar or {}
    try:
        if supplied_sidecar is not None:
            if sidecar['schema_version'] != 5 or sidecar['xml_sha256'] != xml_hash:
                raise SemanticError('XML/sidecar hash or schema version differs')
            if sidecar['input_sha256'] != context.inputs.input_sha256:
                raise SemanticError('sidecar input identity differs from this request')
            if (not isinstance(sidecar['request_contract'], Mapping)
                    or type(sidecar['request_contract'].get('slice_bytes')) is not int):
                raise SemanticError('sidecar slice size must be an integer')
            if sidecar['request_contract'] != _request_contract(context.inputs):
                raise SemanticError('sidecar effective runtime request differs')
        _validate_xml_text(xml_text, program, context.inputs)
        checks['xml'] = _check('xml')
    except (SemanticError, KeyError) as error:
        checks['xml'] = _check('xml', 'invalid', str(error))
    try:
        execution_dependencies(program, lease=context.lease)
        checks['deadlock'] = _check('deadlock')
        _validate_bindings(program, storage, context)
        checks['endpoint'] = _check('endpoint')
        executable = build_executable(program, storage, context)
    except BudgetExhausted:
        raise
    except SemanticError as error:
        key = 'deadlock' if checks['deadlock'].status is ValidationStatus.NOT_RUN else 'endpoint'
        checks[key] = _check(key, 'invalid', str(error))
    try:
        capacity = _capacity(program, context)
        inputs = context.inputs
        _launch_geometry(program, storage, context)
        if simple_payload_capacity(int(inputs.request_config['runtime']['environment']['NCCL_BUFFSIZE']), DATATYPE_BYTES[inputs.collective.datatype]) < inputs.hyperparameters.slice_size_bytes:
            raise SemanticError('NCCL_BUFFSIZE cannot hold one whole Simple slice')
        if not capacity.compatible:
            raise SemanticError('runtime capacity: ' + ', '.join(sorted({issue.code for issue in capacity.violations})))
        checks['launch'] = _check('launch', evidence={'scope': 'static launch safety; node preflight is also required'})
    except SemanticError as error:
        checks['launch'] = _check('launch', 'invalid', str(error))
    if context.inputs.request_config['sketch']['runtime_compatibility']['enabled']:
        checks['compatibility'] = _check('compatibility', checks['launch'].status.value, checks['launch'].message)
    if executable is not None:
        try:
            prediction = predict(executable, context.model, exact=True, lease=context.lease)
            checks['prediction'] = _check('prediction', 'valid' if prediction.t_pred_us is not None else 'invalid',
                'complete dynamic replay' if prediction.t_pred_us is not None else 'cost coverage is incomplete',
                {'missing_components': sorted(prediction.missing_components), 'model_version': prediction.model_version,
                 'estimated_operations': [op.operation_id for op in executable.operations if op.source_override],
                 'native_postprocess': 'avg: final output division uses a copy-cost estimate' if
                     context.inputs.collective.reduction_op == 'avg' else None})
            version = _version_schedule(storage, executable, prediction, context.inputs)
            evidence = _check_storage_values(storage, version, context)
            if supplied_sidecar is not None:
                recorded = json.loads(json.dumps(sidecar['version_schedule']))
                rebuilt = _plain(asdict(version))
                for document in (recorded, rebuilt):
                    for operation in document['operations']:
                        operation.pop('start_us', None); operation.pop('end_us', None)
                if recorded != rebuilt:
                    raise SemanticError('semantic sidecar differs from replayed native state mapping')
            checks['semantic'] = _check('semantic', evidence=evidence)
            checks['storage'] = _check('storage')
        except BudgetExhausted:
            raise
        except (SemanticError, KeyError) as error:
            key = 'prediction' if prediction is None else 'semantic'
            checks[key] = _check(key, 'invalid', str(error))
    if supplied_sidecar is None:
        sidecar = {'schema_version': 5, 'xml_sha256': xml_hash, 'input_sha256': context.inputs.input_sha256,
            'request_contract': _request_contract(context.inputs),
            'model_version': context.model.version, 'program': _serialize_program(program),
            'storage': _serialize_storage(storage), 'storage_reuse': list(storage.reuse),
            'version_schedule': None if version is None else _plain(asdict(version)),
            'reference_orders_omitted': list(storage.reference_orders_omitted)}
    return CandidateBundle(source, xml_text, xml_hash, sidecar, program, version, executable, prediction,
                           MappingProxyType(checks), capacity, storage)


def lower_and_check(candidate, context):
    context.lease.check()
    try:
        storage = candidate.storage_plan if isinstance(candidate, StorageCandidate) else build_storage_plan(candidate, context)
        program = lower_program(storage, context)
        xml = emit_native_xml(program, context.inputs)
    except BudgetExhausted:
        raise
    except SemanticError as error:
        checks = {key: _check(key, 'not_run', 'lowering prerequisite pending') for key in DIMENSIONS}
        checks['storage'] = _check('storage', 'invalid', str(error))
        return CandidateBundle(candidate, None, None, {}, None, None, None, None, MappingProxyType(checks))
    return evaluate_program(candidate, storage, program, xml, context)


def rejected_xml_bundle(xml_text, sidecar, reason):
    checks = {key: _check(key, 'not_run', 'invalid artifact prerequisite') for key in DIMENSIONS}
    checks['xml'] = _check('xml', 'invalid', str(reason))
    return CandidateBundle(('xml',), xml_text, None, sidecar, None, None, None, None, MappingProxyType(checks))


def check_xml_bundle(xml_text, sidecar, context):
    try:
        storage = _deserialize_storage(sidecar['storage'])
        program = _deserialize_program(sidecar['program'])
        return evaluate_program(('xml', sidecar.get('xml_sha256')), storage, program, xml_text, context, supplied_sidecar=sidecar)
    except BudgetExhausted:
        raise
    except (SemanticError, KeyError, ValueError, TypeError) as error:
        return rejected_xml_bundle(xml_text, sidecar, error)
