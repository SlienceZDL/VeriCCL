"""Lower state instructions to the pinned MSCCL Simple/count-one format."""
from collections import defaultdict, deque
from dataclasses import replace
from types import MappingProxyType

from lxml import etree

from vericcl.errors import SemanticError
from vericcl.performance.events import WorkOperation, Executable
from vericcl.semantics.collective import CollectiveKind
from vericcl.topology.model import LinkKey
from .v5_model import NativeStep, NativeBlock, NativeProgram, Address


def _endpoints(op, storage):
    source = storage.addresses[op.inputs[0]]
    destination = storage.addresses[op.output]
    if op.kind == 'copy':
        return ((op.rank, 'copy', -1, -1, 'cpy', source, destination),)
    receiver_source = storage.addresses[op.inputs[1]] if op.kind == 'rrc' else source
    return ((source.rank, 'send', destination.rank, op.channel, 's', source, destination),
            (destination.rank, 'recv', source.rank, op.channel, 'rrc' if op.kind == 'rrc' else 'r', receiver_source, destination))


def lower_program(storage, context):
    layout = context.inputs.request_config['runtime']['tb_layout']
    blocks, locations, step_owner, dependencies_by_step = defaultdict(list), {}, {}, {}
    endpoint_bits, endpoint_ancestors, endpoint_order = {}, {}, {}
    block_ancestors, rank_sequence = defaultdict(int), defaultdict(int)
    for op in storage.instructions:
        context.lease.check()
        endpoints = _endpoints(op, storage)
        ranks = {endpoint[0] for endpoint in endpoints}
        assigned = defaultdict(list)
        for predecessor in sorted(op.dependencies):
            candidates = [rank for rank in ranks if (predecessor, rank) in locations]
            if not candidates:
                raise SemanticError('a required dependency has no same-rank endpoint: ' + predecessor + ' -> ' + op.operation_id)
            # Each local endpoint protects its own reads and writes. A receive
            # can wait while its matching send already reads a stale buffer.
            # Operation-level transitivity is insufficient for removing these
            # waits; same-TB order below is a valid local implication.
            for rank in candidates:
                assigned[rank].append(locations[predecessor, rank])
        for rank, direction, peer, channel, kind, source, destination in endpoints:
            key = (rank, 'fused' if layout == 'fused' and direction != 'copy' else direction, peer, channel)
            # Reduce only the GPU-local endpoint graph. Prior work on this TB
            # and explicit waits imply their local ancestors, independently of
            # when the matching endpoint on another GPU starts or completes.
            # Newest endpoints are visited first: each ancestor was emitted
            # earlier, so a covered target can be removed without losing order.
            covered = block_ancestors[key]
            waits = []
            for target in sorted(set(assigned[rank]), key=endpoint_order.__getitem__, reverse=True):
                if not covered & endpoint_bits[target]:
                    waits.append(target)
                    covered |= endpoint_ancestors[target]
            for index, target in enumerate(waits[:-1]):
                identifier = 'nop:%s:%d:%d' % (op.operation_id, rank, index)
                step = NativeStep(identifier, op.operation_id, 'nop', None, None)
                blocks[key].append(step); step_owner[identifier] = key; dependencies_by_step[identifier] = target
            identifier = '%s:%s:%d' % (op.operation_id, direction, rank)
            step = NativeStep(identifier, op.operation_id, kind, source, destination)
            blocks[key].append(step); step_owner[identifier] = key
            if waits:
                dependencies_by_step[identifier] = waits[-1]
            locations[op.operation_id, rank] = identifier
            endpoint_order[identifier] = rank_sequence[rank]
            endpoint_bits[identifier] = 1 << rank_sequence[rank]
            rank_sequence[rank] += 1
            endpoint_ancestors[identifier] = covered | endpoint_bits[identifier]
            block_ancestors[key] = endpoint_ancestors[identifier]
    target_blocks = {step_owner[target] for target in dependencies_by_step.values()}
    block_ids = {}
    for rank in range(context.inputs.rank_count):
        keys = sorted((key for key in blocks if key[0] == rank), key=lambda key: (key not in target_blocks, key))
        block_ids.update({key: index for index, key in enumerate(keys)})
    step_location = {step.step_id: (block_ids[key], index) for key, steps in blocks.items() for index, step in enumerate(steps)}
    depended = set(dependencies_by_step.values())
    result = []
    for key in sorted(blocks, key=lambda key: (key[0], block_ids[key])):
        rank, direction, peer, channel = key
        steps = []
        for step in blocks[key]:
            dep = step_location[dependencies_by_step[step.step_id]] if step.step_id in dependencies_by_step else (-1, -1)
            steps.append(replace(step, dependency_tb=dep[0], dependency_step=dep[1], has_dependence=step.step_id in depended))
        send_peer = peer if direction in {'send', 'fused'} and any(step.kind == 's' for step in steps) else -1
        recv_peer = peer if direction in {'recv', 'fused'} and any(step.kind in {'r', 'rrc'} for step in steps) else -1
        result.append(NativeBlock(rank, block_ids[key], max(channel, 0), send_peer, recv_peer, tuple(steps)))
    n = context.inputs.hyperparameters.slice_count
    chunks = n * context.inputs.rank_count if context.inputs.collective.kind is CollectiveKind.ALL_GATHER else n
    channels = max((block.channel for block in result), default=0) + 1
    return NativeProgram(context.inputs.rank_count, channels, chunks, tuple(result), storage.input_slots,
                         storage.output_slots, storage.scratch_slots)


def _dependency_order(dependencies, lease=None):
    if lease is not None:
        lease.check()
    degrees = {key: len(predecessors) for key, predecessors in dependencies.items()}
    successors = {key: [] for key in degrees}
    for index, (key, predecessors) in enumerate(dependencies.items()):
        if lease is not None and index % 128 == 0:
            lease.check()
        for predecessor in predecessors:
            if predecessor not in successors:
                raise SemanticError('native dependency names an unknown operation')
            successors[predecessor].append(key)
    ready = deque(key for key, degree in degrees.items() if degree == 0)
    order = []
    while ready:
        if lease is not None and len(order) % 128 == 0:
            lease.check()
        key = ready.popleft()
        order.append(key)
        for follower in successors[key]:
            degrees[follower] -= 1
            if degrees[follower] == 0:
                ready.append(follower)
    if len(order) != len(degrees):
        raise SemanticError('native endpoint/TB dependency graph is deadlocked')
    return order


def _dependency_ancestors(dependencies, lease=None):
    """Exact reachability using one bit per vertex, independent of vertex names."""
    order = _dependency_order(dependencies, lease)
    bits = {key: 1 << index for index, key in enumerate(order)}
    ancestors = {}
    for index, key in enumerate(order):
        if lease is not None and index % 128 == 0:
            lease.check()
        mask = 0
        for predecessor in dependencies[key]:
            mask |= bits[predecessor] | ancestors[predecessor]
        ancestors[key] = mask
    return ancestors, bits


def execution_dependencies(program, lease=None):
    """Collapse native dependency-only NOP chains onto the following work."""
    if lease is not None:
        lease.check()
    steps = {(block.rank, block.tb_id, index): step for block in program.blocks for index, step in enumerate(block.steps)}
    dependencies, endpoint_counts, by_op = defaultdict(set), defaultdict(list), defaultdict(list)
    for block in program.blocks:
        if lease is not None:
            lease.check()
        previous = None
        waiting = set()
        for index, step in enumerate(block.steps):
            if lease is not None and index % 128 == 0:
                lease.check()
            if (step.dependency_tb == -1) != (step.dependency_step == -1):
                raise SemanticError('native dependency must name both TB and step')
            if step.dependency_tb >= 0:
                target = steps.get((block.rank, step.dependency_tb, step.dependency_step))
                if target is None or target.kind == 'nop' or not target.has_dependence:
                    raise SemanticError('native dependency target is absent or does not publish completion')
                if step.dependency_tb == block.tb_id and step.dependency_step >= index:
                    raise SemanticError('native dependency points to the same or a later step')
                waiting.add(target.operation_id)
            if step.kind == 'nop':
                if step.dependency_tb < 0:
                    raise SemanticError('dependency-only NOP must carry a dependency')
                continue
            if waiting and step.dependency_tb < 0:
                raise SemanticError('the final instruction of a NOP dependency chain must also carry a dependency')
            if previous is not None and previous != step.operation_id:
                dependencies[step.operation_id].add(previous)
            dependencies[step.operation_id].update(waiting)
            if step.operation_id in dependencies[step.operation_id]:
                raise SemanticError('native operation depends on its own completion')
            waiting.clear(); previous = step.operation_id
            endpoint_counts[step.operation_id].append((block, step))
            by_op[step.operation_id].append((block, step))
        if waiting:
            raise SemanticError('a trailing NOP dependency chain has no native operation')
    for identifier, members in endpoint_counts.items():
        if len(members) == 1 and members[0][1].kind in {'cpy', 're'}:
            continue
        send = [(block, step) for block, step in members if step.kind == 's']
        recv = [(block, step) for block, step in members if step.kind in {'r', 'rrc'}]
        if len(members) != 2 or len(send) != len(recv) or len(send) != 1:
            raise SemanticError('physical transfer must have exactly one send and one matching receive')
        sb, ss = send[0]; rb, rs = recv[0]
        if sb.send_peer != rb.rank or rb.recv_peer != sb.rank or sb.channel != rb.channel:
            raise SemanticError('physical endpoint peers or channels do not match')
    _dependency_order({key: dependencies[key] for key in by_op}, lease)
    return {identifier: frozenset(dependencies[identifier]) for identifier in by_op}, by_op


def build_executable(program, storage, context):
    dependencies, endpoints = execution_dependencies(program, lease=context.lease)
    instructions = {op.operation_id: op for op in storage.instructions}
    if endpoints.keys() != instructions.keys():
        raise SemanticError('native program and semantic operations do not have the same coverage')
    operations = []
    for identifier, instruction in instructions.items():
        source = storage.addresses[instruction.inputs[0]]
        destination = storage.addresses[instruction.output]
        if instruction.kind in {'send', 'rrc'}:
            suffix = '%d:%d' % (source.rank, destination.rank)
            cost_key = ('rrc:' if instruction.kind == 'rrc' else 'link:') + suffix
            edge_resources = context.topology.link(LinkKey(source.rank, destination.rank)).resource_ids
            resources = tuple(sorted(set(('link:' + suffix,) + edge_resources)))
            flow = '%s:%d:%d:c%d' % (instruction.kind, source.rank, destination.rank, instruction.channel)
        else:
            cost_key = '%s:%d' % (instruction.kind, instruction.rank)
            resources, flow = (), None
        operations.append(WorkOperation(identifier, instruction.kind, cost_key, context.inputs.hyperparameters.slice_size_bytes,
            tuple(sorted(dependencies[identifier])), resources, flow))
    outputs = tuple(sorted({storage.values[state_id].producer for state_id in storage.outputs.values()
                            if storage.values[state_id].producer is not None}))
    if context.inputs.collective.reduction_op == 'avg':
        # The native API sums first and divides each valid output after the
        # rank's whole communication kernel has completed. Its memory pass is
        # estimated from the current copy component until separately measured.
        outputs = []
        for rank, count in sorted(program.output_slots.items()):
            if not count:
                continue
            identifier = 'native-postscale:%d' % rank
            local = {step.operation_id for block in program.blocks if block.rank == rank
                     for step in block.steps if step.kind != 'nop'}
            operations.append(WorkOperation(identifier, 'copy', 'copy:%d' % rank,
                count * context.inputs.hyperparameters.slice_size_bytes,
                tuple(sorted(local)), source_override='estimated'))
            outputs.append(identifier)
    return Executable(tuple(operations), outputs, dict(context.model.context))


def emit_native_xml(program, inputs):
    runtime_bytes = inputs.hyperparameters.total_size_bytes * (inputs.rank_count if inputs.collective.kind is CollectiveKind.ALL_GATHER else 1)
    root = etree.Element('algo', name='vericcl', nchannels=str(program.channels), nchunksperloop=str(program.chunks_per_loop),
        proto='Simple', coll=inputs.collective.kind.value, inplace=str(int(inputs.collective.inplace)), redop='nop',
        ngpus=str(program.rank_count), minBytes=str(runtime_bytes), maxBytes=str(runtime_bytes + 1))
    for rank in range(program.rank_count):
        gpu = etree.SubElement(root, 'gpu', id=str(rank), i_chunks=str(program.input_slots[rank]),
            o_chunks=str(program.output_slots[rank]), s_chunks=str(program.scratch_slots[rank]))
        for block in (b for b in program.blocks if b.rank == rank):
            tb = etree.SubElement(gpu, 'tb', id=str(block.tb_id), send=str(block.send_peer), recv=str(block.recv_peer), chan=str(block.channel))
            for index, step in enumerate(block.steps):
                etree.SubElement(tb, 'step', s=str(index), type=step.kind,
                    srcbuf=step.src.buffer if step.src else 'i', srcoff=str(step.src.offset if step.src else -1),
                    dstbuf=step.dst.buffer if step.dst else 'o', dstoff=str(step.dst.offset if step.dst else -1),
                    cnt=str(step.count), depid=str(step.dependency_tb), deps=str(step.dependency_step), hasdep=str(int(step.has_dependence)))
    etree.indent(root, space='  ')
    return etree.tostring(root, encoding='unicode') + '\n'
