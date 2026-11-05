"""Cause-scoped, integer-slice repair with required native event optimization."""
from dataclasses import dataclass, field, replace
from fractions import Fraction
from itertools import product

from vericcl.errors import SemanticError, SolverUnavailableError
from vericcl.execution.budget import BudgetExhausted
from vericcl.xml.v5_model import NetworkOperation, NetworkCandidate, StorageCandidate
from vericcl.xml.v5_buffers import build_storage_plan
from vericcl.xml.v5_program import _endpoints
from vericcl.xml.lower import lower_and_check
from vericcl.xml.parser import check_xml_bundle
from vericcl.verification.bottlenecks import StrategyRecord, AnalysisContext, legal_strategy, _paths, mandatory_dependencies
from vericcl.topology.model import LinkKey
from .event_columns import solve_event_columns


class DomainError(SemanticError):
    pass


@dataclass(frozen=True)
class RepairDomain:
    operation_ids: frozenset[str]
    family: str
    cause_ids: tuple = ()
    fixed_decisions: tuple = ('data_allocation', 'path', 'communication_resources', 'outside_relative_order')


def build_domain(cause, action):
    if not isinstance(action, StrategyRecord) or not action.operation_ids:
        raise SemanticError('repair domain requires an explicit strategy and operation range')
    return RepairDomain(frozenset(action.operation_ids), action.family,
                        action.cause_ids + ((cause.cause_id,) if cause is not None else ()))


@dataclass
class RepairContext:
    lowering: object
    mode: str = 'strategy_first'
    candidate_batch_size: int = 4
    request_inputs: object = None
    records: list = field(default_factory=list)
    model_calls: int = 0

    def __post_init__(self):
        if self.mode not in {'strategy_first', 'joint'} or isinstance(self.candidate_batch_size, bool) or not isinstance(self.candidate_batch_size, int) or self.candidate_batch_size < 1:
            raise SemanticError('invalid repair mode or model batch size')

    @property
    def lease(self):
        return self.lowering.lease


@dataclass(frozen=True)
class OutsideCheck:
    valid: bool
    differences: tuple


@dataclass(frozen=True)
class RepairResult:
    status: str
    incumbent: object
    candidate: object
    domain: RepairDomain
    model_calls: int
    attempts: tuple
    solver_evidence: tuple
    search_complete: bool = False
    acceptance_basis: str = 'same_model_complete_prediction'
    full_search_proven: bool = False
    context: object = None


def _inside_ids(bundle, domain):
    aliases = dict(getattr(bundle.source, 'replaced_operations', ()))
    return set(domain.operation_ids) | {key for key, original in aliases.items() if original in domain.operation_ids}


def _signature(op, storage):
    return (op.kind, op.peer, op.rank, op.channel, op.stage_id, op.participants,
            storage.values[op.inputs[0]].contributors, storage.values[op.output].contributors)


def outside_invariants(before, after, domain):
    if after.storage_plan is None or after.program is None:
        return OutsideCheck(False, ('missing native candidate',))
    outside = {op.operation_id: op for op in before.storage_plan.instructions if op.operation_id not in domain.operation_ids}
    current = {op.operation_id: op for op in after.storage_plan.instructions}
    differences = []
    if before.sidecar.get('request_contract') != after.sidecar.get('request_contract'):
        differences.append('local repair changed the external collective/partition contract')
    before_work = {op.operation_id: op for op in before.executable.operations}
    after_work = {op.operation_id: op for op in after.executable.operations} if after.executable else {}
    for key in outside:
        if key in before_work and key in after_work:
            left, right = before_work[key], after_work[key]
            if (left.resources, left.flow_id, left.payload_bytes, left.not_before_us) != (right.resources, right.flow_id, right.payload_bytes, right.not_before_us):
                differences.append('outside communication resources changed: ' + key)
    for key, op in outside.items():
        if key not in current or _signature(op, before.storage_plan) != _signature(current[key], after.storage_plan):
            differences.append('outside operation changed: ' + key)
    def queues(bundle):
        result = {}
        for block in bundle.program.blocks:
            keys = tuple(step.operation_id for step in block.steps if step.kind != 'nop' and step.operation_id in outside)
            if keys:
                result[block.rank, block.send_peer, block.recv_peer, block.channel] = keys
        return result
    if queues(before) != queues(after):
        differences.append('outside TB relative order changed')
    return OutsideCheck(not differences, tuple(differences))


def _network(bundle):
    storage = bundle.storage_plan
    ops = {op.operation_id: op for op in storage.instructions}
    network_ids = {key for key, op in ops.items() if op.kind in {'send', 'rrc'}}
    def upstream(state_id):
        producer = storage.values[state_id].producer
        if producer is None:
            return set()
        if producer in network_ids:
            return {producer}
        return {key for state in ops[producer].inputs for key in upstream(state)}
    result = []
    for key, op in ops.items():
        if key not in network_ids:
            continue
        source, target = storage.values[op.inputs[0]], storage.values[op.output]
        deps = frozenset(dep for state in op.inputs for dep in upstream(state))
        result.append(NetworkOperation(key, op.kind, source.rank, op.rank, op.channel, op.stage_id,
            source.contributors, target.contributors, op.participants, deps, frozenset(),
            bundle.prediction.start_times.get(key, op.priority_time)))
    return tuple(result)


def _topological(operations, constraints, lease, *, priority=None):
    by_id = {op.operation_id: op for op in operations}
    pending, ordered, done = set(by_id), [], set()
    while pending:
        lease.check()
        ready = [key for key in pending if constraints[key] <= done]
        if not ready:
            raise SemanticError('proposed execution order creates a dependency cycle')
        key = min(ready, key=lambda key: ((priority or {}).get(key, by_id[key].priority_time), key))
        ordered.append(by_id[key]); done.add(key); pending.remove(key)
    return tuple(ordered)


def _substitute(bundle, strategy, context):
    paths, channels = dict(strategy.paths), dict(strategy.channels)
    result, aliases = [], []
    for op in _network(bundle):
        path = paths.get(op.operation_id, (op.src_rank, op.dst_rank))
        channel = channels.get(op.operation_id, op.channel)
        previous = None
        for index, (src, dst) in enumerate(zip(path, path[1:])):
            last = index == len(path) - 2
            key = op.operation_id if last else 'repair:%s:hop%d:%d-%d' % (op.operation_id, index, src, dst)
            deps = frozenset({previous}) if previous else op.dependencies
            result.append(replace(op, operation_id=key, kind=op.kind if last else 'send', src_rank=src, dst_rank=dst,
                                  channel=channel, dependencies=deps, reference_dependencies=frozenset()))
            if key != op.operation_id:
                aliases.append((key, op.operation_id))
            previous = key
    return NetworkCandidate(context.lowering.inputs.rank_count, context.lowering.inputs.hyperparameters.slice_count,
        context.lowering.inputs.hyperparameters.slice_size_bytes, tuple(result), 'local_repair', tuple(aliases))


def _path_strategies(bundle, domain, strategy, context):
    if context.mode != 'joint':
        yield strategy
        return
    storage = bundle.storage_plan
    selected = [op for op in storage.instructions if op.operation_id in domain.operation_ids and op.kind in {'send', 'rrc'}]
    def visit(index, choices):
        context.lease.check()
        if index == len(selected):
            yield replace(strategy, paths=tuple(choices))
            return
        op = selected[index]
        src = storage.values[op.inputs[0]].rank
        fixed = dict(strategy.paths).get(op.operation_id)
        options = (fixed,) if fixed is not None else ((src, op.rank),)
        for path in options:
            yield from visit(index + 1, choices + [(op.operation_id, path)])
        if fixed is None:
            for path in _paths(src, op.rank, op.participants, context.lowering.topology, context.lease):
                yield from visit(index + 1, choices + [(op.operation_id, path)])
    yield from visit(0, [])


def _queue_orders(keys, outside, mandatory, lease):
    """All per-TB orders; frozen outside members retain their subsequence."""
    keys = tuple(keys)
    frozen = tuple(key for key in keys if key in outside)
    def visit(prefix, remaining):
        lease.check()
        if not remaining:
            yield prefix
            return
        first_outside = next((key for key in frozen if key in remaining), None)
        for key in keys:
            if key not in remaining or (key in outside and key != first_outside) or mandatory[key] & remaining:
                continue
            yield from visit(prefix + (key,), remaining - {key})
    yield from visit((), frozenset(keys))


def _storage_orders(storage, bundle, domain, strategy, aliases, context):
    inside = set(domain.operation_ids) | {key for key, old in aliases if old in domain.operation_ids}
    outside = {op.operation_id for op in bundle.storage_plan.instructions} - set(domain.operation_ids)
    mandatory = mandatory_dependencies(storage, context.lowering.inputs)
    # Keep actual pre-existing outside TB order, but do not freeze old times.
    for block in bundle.program.blocks:
        sequence = tuple(dict.fromkeys(step.operation_id for step in block.steps if step.kind != 'nop' and step.operation_id in outside))
        for first, second in zip(sequence, sequence[1:]):
            mandatory[second] = mandatory[second] | {first}
    for first, second in strategy.orders:
        mandatory[second] = mandatory[second] | {first}
    ops = tuple(replace(op, dependencies=mandatory[op.operation_id]) for op in storage.instructions)
    try:
        ordered = _topological(ops, mandatory, context.lease)
    except SemanticError:
        return
    # Initial feasible order is a warm-start column; it still enters SCIP.
    yield replace(storage, instructions=ordered)
    groups = {}
    for op in ordered:
        for rank, direction, peer, channel, *_ in _endpoints(op, storage):
            group = (rank, 'fused' if context.lowering.inputs.request_config['runtime']['tb_layout'] == 'fused'
                     and direction != 'copy' else direction, peer, channel)
            groups.setdefault(group, []).append(op.operation_id)
    varying, seen_groups = [], set()
    for keys in groups.values():
        key = tuple(keys)
        if len(key) > 1 and set(key) & inside and key not in seen_groups:
            varying.append(key); seen_groups.add(key)
    def visit(index, dependencies):
        context.lease.check()
        if index == len(varying):
            try:
                chosen = _topological(ops, dependencies, context.lease)
            except SemanticError:
                return
            yield replace(storage, instructions=tuple(replace(op, dependencies=frozenset(dependencies[op.operation_id])) for op in chosen))
            return
        keys = varying[index]
        for order in _queue_orders(keys, outside, mandatory, context.lease):
            updated = {key: set(value) for key, value in dependencies.items()}
            for first, second in zip(order, order[1:]):
                updated[second].add(first)
            yield from visit(index + 1, updated)
    yield from visit(0, mandatory)


def _root_networks(bundle, domain, strategy, context):
    from vericcl.solver.trees.operators import routing_demands, _columns, _physical_operation
    inputs, topology = context.lowering.inputs, context.lowering.topology
    if inputs.collective.kind.value != 'allreduce':
        raise DomainError('internal roots only apply to AllReduce')
    ops = bundle.storage_plan.instructions
    positions = {bundle.storage_plan.values[op.output].logical_position for op in ops if op.operation_id in domain.operation_ids}
    required = {op.operation_id for op in ops if bundle.storage_plan.values[op.output].logical_position in positions}
    if required - domain.operation_ids:
        raise DomainError('internal root change needs every operation of its affected logical slices: ' + ','.join(sorted(required)))
    old_network = _network(bundle)
    affected = [op for op in old_network if op.operation_id in required]
    if (any(set(op.participants) != set(range(inputs.rank_count)) for op in affected)
            or inputs.atom_constraints.stage_num not in {None, 1, 2}):
        raise DomainError('grouped root changes require the hierarchy structural branch to preserve role/stage interfaces')
    parameters = dict(strategy.parameters)
    if set(parameters) - {'root'}:
        raise DomainError('unknown internal-root parameter')
    requested = parameters.get('root')
    if requested is not None and (isinstance(requested, bool) or not isinstance(requested, int) or not 0 <= requested < inputs.rank_count):
        raise DomainError('invalid internal root')
    demands = [demand for demand in routing_demands(inputs) if demand.logical_position in positions]
    outside = tuple(op for op in old_network if op.operation_id not in required)
    def visit(index, added, aliases):
        context.lease.check()
        if index == len(demands):
            yield NetworkCandidate(inputs.rank_count, inputs.hyperparameters.slice_count, inputs.hyperparameters.slice_size_bytes,
                                   outside + tuple(added), 'internal_root_repair', tuple(aliases))
            return
        demand = demands[index]
        original = next(op.operation_id for op in ops if op.operation_id in required
                        and bundle.storage_plan.values[op.output].logical_position == demand.logical_position)
        for route in _columns(inputs, topology, demand, context.lease):
            if requested is not None and route.root != requested:
                continue
            physical = _physical_operation(inputs, topology, demand, route, context.lease)
            mapping = {op.operation_id: 'root-repair:%d:%s' % (route.root, op.operation_id) for op in physical}
            converted = [NetworkOperation(mapping[op.operation_id], op.kind, *op.edge, 0, op.stage_id,
                op.contributors, demand.contributors, tuple(range(inputs.rank_count)),
                frozenset(mapping[key] for key in op.dependencies), frozenset(), float(index)) for op in physical]
            yield from visit(index + 1, added + converted, aliases + [(op.operation_id, original) for op in converted])
    yield from visit(0, [], [])


def _network_variants(bundle, domain, strategy, context):
    if strategy.family == 'internal_root':
        for network in _root_networks(bundle, domain, strategy, context):
            yield strategy, network
    else:
        for paths in _path_strategies(bundle, domain, strategy, context):
            context.lease.check()
            if legal_strategy(paths, bundle, AnalysisContext(context.lowering)):
                yield paths, _substitute(bundle, paths, context)


def implementations(bundle, domain, strategy, context):
    for paths, network in _network_variants(bundle, domain, strategy, context):
        fixed = dict(paths.channels)
        changing = [op for op in network.operations if op.operation_id in domain.operation_ids
                    or dict(network.replaced_operations).get(op.operation_id) in domain.operation_ids]
        domains = []
        for op in changing:
            original = dict(network.replaced_operations).get(op.operation_id, op.operation_id)
            edge = context.lowering.topology.links[LinkKey(op.src_rank, op.dst_rank)]
            maximum = min(context.lowering.inputs.solver.max_channels, edge.max_channels,
                          *(context.lowering.topology.shared_resources[key].max_channels for key in edge.resource_ids))
            domains.append((fixed[original],) if original in fixed else tuple(dict.fromkeys((op.channel,) + tuple(range(maximum)))))
        for values in product(*domains):
            context.lease.check()
            assigned = dict(zip((op.operation_id for op in changing), values))
            candidate = replace(network, operations=tuple(replace(op, channel=assigned.get(op.operation_id, op.channel)) for op in network.operations))
            try:
                storage = build_storage_plan(candidate, context.lowering)
                for ordered in _storage_orders(storage, bundle, domain, paths, candidate.replaced_operations, context):
                    yield StorageCandidate(ordered, candidate, candidate.replaced_operations)
            except BudgetExhausted:
                raise
            except SemanticError as error:
                context.records.append({'phase': 'repair_lowering', 'status': 'invalid_candidate', 'message': str(error)})


def solve_repair(bundle, domain, strategy_or_domain, context):
    initial_calls, attempts, evidence, seen = context.model_calls, [], [], set()
    unresolved = False
    incumbent, pending, exhausted = bundle, [], False
    def outcome(status, candidate=None):
        return RepairResult(status, bundle, candidate, domain, context.model_calls - initial_calls,
                            tuple(attempts), tuple(evidence), exhausted and not unresolved, context=context.lowering)
    def solve_batch():
        nonlocal pending, unresolved
        # Include a current-model incumbent so a bad proposal cannot replace it.
        columns = (incumbent,) + tuple(pending)
        try:
            selected, result = solve_event_columns(columns, context)
        except BudgetExhausted:
            if context.lease.expired():
                raise
            unresolved = True
            evidence.append({'status': 'model_budget_exhausted', 'columns': len(columns), 'eager_start_enforced': True})
            pending = []
            return None
        evidence.append({'status': result.status, 'solver_status': result.solver_status, 'primal': result.primal,
                         'dual_bound': result.dual_bound, 'scope': 'current_finite_event_columns',
                         'columns': len(columns), 'eager_start_enforced': True})
        pending = []
        if not result.numerical_optimal:
            unresolved = True
        if selected is not None and selected > 0:
            candidate = columns[selected]
            if Fraction(*candidate.prediction.exact_completion_ratio) < Fraction(*incumbent.prediction.exact_completion_ratio):
                return candidate
        return None
    try:
        context.lease.check()
        if context.request_inputs is not None:
            from vericcl.planner.parameters import initial_parameter_domain, resolve_parameters
            parameters = initial_parameter_domain(context.request_inputs, fixed_slice_bytes=context.lowering.inputs.hyperparameters.slice_size_bytes,
                                                  lease=context.lease)
            if not parameters.candidates:
                raise DomainError('no current-S channel count satisfies necessary capacity constraints')
            upper = max(parameters.candidates, key=lambda candidate: candidate.channels)
            context.lowering = replace(context.lowering, inputs=resolve_parameters(context.request_inputs, upper))
        if not domain.operation_ids or bundle.storage_plan is None:
            return outcome('invalid_domain')
        if domain.operation_ids - {op.operation_id for op in bundle.storage_plan.instructions}:
            return outcome('invalid_domain')
        incumbent = check_xml_bundle(bundle.xml_text, bundle.sidecar, context.lowering)
        if not incumbent.can_execute:
            return outcome('incumbent_unavailable')
        source = (strategy_or_domain,) if isinstance(strategy_or_domain, StrategyRecord) else iter(strategy_or_domain)
        valid_strategies = 0
        for strategy in source:
            context.lease.check()
            if (not isinstance(strategy, StrategyRecord) or set(strategy.operation_ids) - domain.operation_ids
                    or not legal_strategy(strategy, incumbent, AnalysisContext(context.lowering))):
                attempts.append({'status': 'invalid_strategy', 'strategy_id': getattr(strategy, 'strategy_id', None)})
                continue
            valid_strategies += 1
            attempts.append({'status': 'started', 'strategy_id': strategy.strategy_id})
            for candidate_source in implementations(incumbent, domain, strategy, context):
                candidate = lower_and_check(candidate_source, context.lowering)
                if not candidate.can_execute:
                    attempts.append({'status': 'invalid_candidate', 'checks': {key: result.message for key, result in candidate.checks.items() if result.status.value == 'invalid'}})
                    continue
                external = outside_invariants(incumbent, candidate, domain)
                if not external.valid:
                    attempts.append({'status': 'outside_changed', 'differences': external.differences})
                    continue
                if candidate.xml_sha256 in seen:
                    continue
                seen.add(candidate.xml_sha256)
                pending.append(candidate)
                if len(pending) >= context.candidate_batch_size:
                    improved = solve_batch()
                    if improved is not None:
                        return outcome('improved', improved)
        exhausted = True
        if pending:
            improved = solve_batch()
            if improved is not None:
                return outcome('improved', improved)
        return outcome(('search_incomplete' if unresolved else 'no_improvement') if valid_strategies else 'invalid_strategy')
    except BudgetExhausted as error:
        attempts.append({'status': 'budget_exhausted', 'message': str(error)})
        return outcome('budget_exhausted')
    except DomainError as error:
        attempts.append({'status': 'invalid_domain', 'message': str(error)})
        return outcome('invalid_domain')
    except SolverUnavailableError:
        raise
    except SemanticError as error:
        attempts.append({'status': 'solver_or_model_error', 'message': str(error)})
        return outcome('analysis_error')
