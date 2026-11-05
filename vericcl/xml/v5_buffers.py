"""Build explicit state versions and allocate storage with read-completion edges."""
from collections import defaultdict
from dataclasses import replace
import heapq
from types import MappingProxyType

from vericcl.errors import SemanticError
from vericcl.semantics.collective import CollectiveKind, required_outputs
from vericcl.semantics.ranges import DataRange, bind_constraints
from vericcl.semantics.versions import BufferRegion
from vericcl.topology.model import LinkKey
from .v5_model import Address, NetworkOperation, NetworkCandidate, Value, Instruction, StoragePlan


def canonical_region(address, inputs):
    """Resolve actual in-place pointer aliases in byte coordinates."""
    rank, buffer, offset = address.rank, address.buffer, address.offset
    n, size = inputs.hyperparameters.slice_count, inputs.hyperparameters.slice_size_bytes
    if buffer == 's':
        return BufferRegion(rank, 'scratch', offset * size, size, 'scratch')
    if not inputs.collective.inplace:
        return BufferRegion(rank, 'input' if buffer == 'i' else 'output', offset * size, size,
                            'input' if buffer == 'i' else 'output')
    kind = inputs.collective.kind
    if kind is CollectiveKind.ALL_GATHER:
        base = rank * n if buffer == 'i' else 0
    elif kind is CollectiveKind.REDUCE_SCATTER:
        base = rank * (n // inputs.rank_count) if buffer == 'o' else 0
    else:
        base = 0
    return BufferRegion(rank, 'inplace', (base + offset) * size, size, 'data')


def _all_contributors(inputs, members):
    n = inputs.hyperparameters.slice_count
    logical = next(iter(members)) % n
    return frozenset(rank * n + logical for rank in range(inputs.rank_count))


def normalize_network(candidate, context):
    from vericcl.solver.trees.timing import RoutedCandidate
    from vericcl.planner.group_search import HierarchyCandidate
    from vericcl.semantics.atom import Schedule
    inputs = context.inputs
    if isinstance(candidate, NetworkCandidate):
        if (candidate.rank_count, candidate.slice_count, candidate.slice_size_bytes) != (inputs.rank_count, inputs.hyperparameters.slice_count, inputs.hyperparameters.slice_size_bytes):
            raise SemanticError("network repair dimensions differ from the resolved request")
        return candidate.operations
    if isinstance(candidate, RoutedCandidate):
        from vericcl.solver.trees.operators import check_routed_semantics
        check_routed_semantics(candidate.bundle, context.lease)
        targets = {d.demand_id: d.contributors for d in candidate.bundle.demands}
        return tuple(NetworkOperation(op.operation_id, op.kind, *op.edge, candidate.schedule.channels[op.operation_id],
            op.stage_id, op.contributors, targets[op.demand_id], tuple(range(inputs.rank_count)), op.dependencies,
            frozenset(candidate.schedule.dependencies[op.operation_id]), candidate.schedule.start_us[op.operation_id])
            for op in candidate.bundle.operations)
    schedule = candidate.schedule if isinstance(candidate, HierarchyCandidate) else candidate
    if not isinstance(schedule, Schedule):
        raise SemanticError('v5 lowering needs a hierarchy or fixed-route schedule')
    if (schedule.rank_count, schedule.slice_count, schedule.slice_size_bytes) != (
            inputs.rank_count, inputs.hyperparameters.slice_count, inputs.hyperparameters.slice_size_bytes):
        raise SemanticError('candidate dimensions differ from the resolved request')
    semantics = schedule.metadata.get('semantic_predecessors', {})
    roles = schedule.metadata.get('transfer_participants', {})
    targets = schedule.metadata.get('transfer_target_contributors', {})
    return tuple(NetworkOperation(t.transfer_id, 'rrc' if t.kind == 'REDUCE' else 'send', t.src_rank, t.dst_rank,
        t.channel, t.stage_id, t.member_slice_ids,
        frozenset(targets.get(t.transfer_id, _all_contributors(inputs, t.member_slice_ids))),
        tuple(roles.get(t.transfer_id, range(inputs.rank_count))), frozenset(semantics.get(t.transfer_id, t.predecessor_ids)),
        t.predecessor_ids, t.st_time) for t in schedule.transfers)


def _fixed_priority_order(by_id, precedence, lease, priority, cycle_message):
    remaining = {key: len(precedence[key]) for key in by_id}
    successors = defaultdict(list)
    for key, dependencies in precedence.items():
        lease.check()
        for dependency in dependencies:
            if dependency not in by_id:
                raise SemanticError(cycle_message)
            successors[dependency].append(key)
    ready = [(priority(by_id[key]), key) for key, count in remaining.items() if not count]
    heapq.heapify(ready)
    ordered = []
    while ready:
        lease.check()
        _, key = heapq.heappop(ready)
        ordered.append(by_id[key])
        for follower in successors[key]:
            remaining[follower] -= 1
            if not remaining[follower]:
                heapq.heappush(ready, (priority(by_id[follower]), follower))
    if len(ordered) != len(by_id):
        raise SemanticError(cycle_message)
    return tuple(ordered)


def _network_order(operations, lease):
    by_id = {op.operation_id: op for op in operations}
    if len(by_id) != len(operations) or any(key not in by_id for op in operations for key in op.dependencies):
        raise SemanticError('network dependencies or operation identities are invalid')
    precedence = {}
    for op in operations:
        references = set()
        for key in op.reference_dependencies:
            if key not in by_id:
                raise SemanticError('unknown network reference dependency')
            other = by_id[key]
            if {op.src_rank, op.dst_rank} & {other.src_rank, other.dst_rank}:
                references.add(key)
        precedence[op.operation_id] = op.dependencies | references
    return _fixed_priority_order(by_id, precedence, lease,
        lambda item: (item.priority_time, item.stage_id, item.operation_id),
        'network semantic dependencies contain a cycle')


def build_storage_plan(candidate, context):
    inputs, lease = context.inputs, context.lease
    n, size, ranks = inputs.hyperparameters.slice_count, inputs.hyperparameters.slice_size_bytes, inputs.rank_count
    operations = _network_order(normalize_network(candidate, context), lease)
    expected = required_outputs(inputs.collective, ranks, n)
    source_ranks = (inputs.collective.root,) if inputs.collective.kind is CollectiveKind.BROADCAST else range(ranks)
    states, by_position, initial, addresses, instructions = {}, defaultdict(list), [], {}, []
    for rank in source_ranks:
        for logical in range(n):
            identifier = 'input-r%d-s%d' % (rank, logical)
            value = Value(identifier, rank, logical, frozenset({rank * n + logical}), None)
            states[identifier] = value; by_position[rank, logical].append(value)
            initial.append(identifier); addresses[identifier] = Address(rank, 'i', logical)
    constraints = bind_constraints(inputs)
    operation_by_id = {op.operation_id: op for op in operations}
    omitted = []
    for op in operations:
        lease.check()
        if not op.contributors or len({s % n for s in op.contributors}) != 1:
            raise SemanticError('a physical transfer must carry one logical slice position')
        logical = min(op.contributors) % n
        edge = context.topology.link(LinkKey(op.src_rank, op.dst_rank))
        if not 0 <= op.channel < min(inputs.solver.max_channels, edge.max_channels):
            raise SemanticError('candidate channel is outside the selected topology limit')
        if op.src_rank not in op.participants or op.dst_rank not in op.participants:
            raise SemanticError('candidate uses a GPU outside its declared role group')
        ranges = tuple(DataRange(s // n, logical * size, (logical + 1) * size) for s in op.contributors)
        if not constraints.allows(ranges, op.src_rank, op.dst_rank, op.stage_id):
            raise SemanticError('candidate violates a source-byte directed-edge constraint')
        matches = [value for value in by_position[op.src_rank, logical] if value.contributors == op.contributors]
        if not matches:
            raise SemanticError('network source state is unavailable: ' + op.operation_id)
        source = matches[-1]
        read_ids = [source.state_id]
        combined = op.contributors
        if op.kind == 'rrc':
            possible = [value for value in by_position[op.dst_rank, logical]
                        if not value.contributors & op.contributors and value.contributors <= op.target_contributors]
            if not possible:
                raise SemanticError('network reduction has no disjoint destination state')
            accumulator = max(enumerate(possible), key=lambda pair: (len(pair[1].contributors), pair[0]))[1]
            read_ids.append(accumulator.state_id)
            combined |= accumulator.contributors
        elif op.kind != 'send':
            raise SemanticError('network operation must be send or rrc')
        dependencies = set(op.dependencies)
        dependencies.update(states[key].producer for key in read_ids if states[key].producer is not None)
        for predecessor in op.reference_dependencies - dependencies:
            prior = operation_by_id.get(predecessor)
            if prior is None:
                raise SemanticError('candidate has an unknown reference dependency')
            if {prior.src_rank, prior.dst_rank} & {op.src_rank, op.dst_rank}:
                dependencies.add(predecessor)
            else:
                omitted.append((predecessor, op.operation_id))
        identifier = 'state-' + op.operation_id
        value = Value(identifier, op.dst_rank, logical, combined, op.operation_id)
        states[identifier] = value; by_position[op.dst_rank, logical].append(value)
        instructions.append(Instruction(op.operation_id, op.kind, tuple(read_ids), identifier, op.dst_rank,
            op.src_rank, op.channel, op.stage_id, op.participants, frozenset(dependencies), op.priority_time))
    outputs, direct_outputs = {}, {}
    final_stage = max((op.stage_id for op in operations), default=0)
    final_priority = max((op.priority_time for op in operations), default=0.) + 1.
    for slot, contributors in sorted(expected.items()):
        matches = [value for value in by_position[slot.rank, min(contributors) % n] if value.contributors == contributors]
        if not matches:
            raise SemanticError('candidate leaves a required collective output unavailable')
        value = matches[-1]
        target = Address(slot.rank, 'o', slot.offset)
        initial_address = addresses.get(value.state_id)
        if initial_address is not None and canonical_region(initial_address, inputs) == canonical_region(target, inputs):
            outputs[slot] = value.state_id
            continue
        if value.producer is not None and not inputs.collective.inplace:
            direct_outputs[value.state_id] = target
            outputs[slot] = value.state_id
            continue
        identifier = 'output-copy-r%d-o%d' % (slot.rank, slot.offset)
        final_id = 'state-' + identifier
        states[final_id] = Value(final_id, value.rank, value.logical_position, value.contributors, identifier)
        addresses[final_id] = target
        deps = frozenset({value.producer}) if value.producer else frozenset()
        instructions.append(Instruction(identifier, 'copy', (value.state_id,), final_id, slot.rank, -1, -1,
                                         final_stage, (slot.rank,), deps, final_priority))
        outputs[slot] = final_id
    # Every final in-place write waits for readers of overlapping original
    # storage. Receiver scratch avoids cyclic send/receive overwrite hazards.
    readers = defaultdict(set)
    for op in instructions:
        for state_id in op.inputs:
            readers[state_id].add(op.operation_id)
    # Every canonical region is one aligned slice, including in-place aliases.
    # Only the identical physical coordinate can overlap an initial region.
    initial_regions = defaultdict(list)
    for state_id in initial:
        region = canonical_region(addresses[state_id], inputs)
        initial_regions[region.rank, region.buffer_id, region.offset].append((state_id, region))
    protected = []
    for op in instructions:
        lease.check()
        dependencies = set(op.dependencies)
        if op.kind == 'copy':
            target = canonical_region(addresses[op.output], inputs)
            for state_id, region in initial_regions[target.rank, target.buffer_id, target.offset]:
                if target.overlaps(region):
                    dependencies.update(readers[state_id] - {op.operation_id})
        protected.append(replace(op, dependencies=frozenset(dependencies)))
    instructions = protected
    ordered, done, pending = [], set(), {op.operation_id: op for op in instructions}
    remaining_reads = {key: len(value) for key, value in readers.items()}
    scratch_states = {key for key, value in states.items() if value.producer is not None and key not in direct_outputs and key not in addresses}
    scratch_capacity = inputs.request_config['runtime']['scratch_bytes_per_rank'] // size
    # With a tight budget, consume existing intermediates before opening more.
    tight = scratch_capacity < max(1, len(scratch_states))
    if not tight:
        ordered = _fixed_priority_order(pending, {key: op.dependencies for key, op in pending.items()},
            lease, lambda op: (op.priority_time, op.operation_id),
            'storage and reference orders contain a dependency cycle')
        pending = {}
    while pending:
        lease.check()
        ready = [op for op in pending.values() if op.dependencies <= done]
        if not ready:
            raise SemanticError('storage and reference orders contain a dependency cycle')
        def priority(op):
            releases = sum(key in scratch_states and remaining_reads.get(key, 0) == 1 for key in op.inputs)
            return (-releases if tight else 0, op.priority_time, op.operation_id)
        op = min(ready, key=priority)
        ordered.append(op); done.add(op.operation_id); pending.pop(op.operation_id)
        for key in op.inputs:
            remaining_reads[key] -= 1
    position = {op.operation_id: i for i, op in enumerate(ordered)}
    ancestors, slots, reuse, final_instructions = {}, defaultdict(dict), [], []
    for op in ordered:
        lease.check()
        dependencies = set(op.dependencies)
        upstream = set(dependencies)
        for dep in dependencies:
            upstream.update(ancestors[dep])
        if op.output in direct_outputs:
            addresses[op.output] = direct_outputs[op.output]
        elif op.output not in addresses:
            pool = slots[op.rank]
            available = []
            for index, prior in pool.items():
                consumers = readers[prior]
                overwrite_input = prior in op.inputs and op.kind == 'rrc'
                previous_readers = consumers - ({op.operation_id} if overwrite_input else set())
                if all(position[reader] < position[op.operation_id] for reader in previous_readers):
                    available.append((not previous_readers <= upstream, index, prior, previous_readers))
            causal = [choice for choice in available if not choice[0]]
            if causal or (len(pool) >= scratch_capacity and available):
                _, index, prior, consumers = min(causal or available, key=lambda choice: (choice[0], choice[1]))
                dependencies.update(consumers)
                reuse.append({'writer': op.operation_id, 'old_state': prior, 'new_state': op.output,
                              'previous_readers': sorted(consumers), 'rank': op.rank, 'offset': index})
            elif len(pool) < scratch_capacity:
                index = len(pool)
            else:
                raise SemanticError('scratch storage budget cannot fit this execution order')
            addresses[op.output] = Address(op.rank, 's', index)
            pool[index] = op.output
        all_ancestors = set(dependencies)
        for dep in dependencies:
            all_ancestors.update(ancestors[dep])
        ancestors[op.operation_id] = all_ancestors
        final_instructions.append(replace(op, dependencies=frozenset(dependencies)))
    output_slots = {rank: max((slot.offset + 1 for slot in expected if slot.rank == rank), default=0) for rank in range(ranks)}
    stage_count = inputs.atom_constraints.stage_num or max((op.stage_id for op in operations), default=0) + 1
    return StoragePlan(MappingProxyType(states), tuple(initial), tuple(final_instructions), MappingProxyType(addresses),
        MappingProxyType(outputs), MappingProxyType({rank: n for rank in range(ranks)}), MappingProxyType(output_slots),
        MappingProxyType({rank: len(slots[rank]) for rank in range(ranks)}), tuple(reuse), stage_count, tuple(omitted))
