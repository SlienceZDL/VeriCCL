"""Causal observations and lazy, legal repair opportunities for v5 artifacts."""
from dataclasses import dataclass, field, replace
from itertools import combinations

from vericcl.errors import SemanticError
from vericcl.execution.budget import BudgetExhausted
from vericcl.performance.simulation import predict
from vericcl.semantics.ranges import DataRange, bind_constraints
from vericcl.topology.model import LinkKey
from .bdd_backend import CompactBDD, BDDBackendError, StrategyDecisionBDD
from .flow_index import build_causal_index
from .search_coverage import REPAIR_FAMILIES, SearchCoverage, digest, model_digest
from vericcl.xml.v5_lower import _serialize_storage, _request_contract


@dataclass(frozen=True)
class SearchScope:
    scope_id: str
    families: tuple = REPAIR_FAMILIES
    batch_size: int = 16

    def __post_init__(self):
        if (not self.scope_id or not self.families or len(set(self.families)) != len(self.families)
                or set(self.families) - set(REPAIR_FAMILIES) or isinstance(self.batch_size, bool)
                or not isinstance(self.batch_size, int) or self.batch_size < 1):
            raise SemanticError('invalid repair search scope')


@dataclass(frozen=True)
class StrategyRecord:
    family: str
    operation_ids: tuple[str, ...]
    paths: tuple = ()
    orders: tuple = ()
    channels: tuple = ()
    parameters: tuple = ()
    cause_ids: tuple = ()

    @property
    def strategy_id(self):
        # Causal descriptions may overlap; identical decisions remain one item.
        return digest({'family': self.family, 'domain': sorted(self.operation_ids),
                       'paths': sorted(self.paths), 'orders': sorted(self.orders),
                       'channels': sorted(self.channels), 'parameters': sorted(self.parameters)})


@dataclass(frozen=True)
class BottleneckCause:
    cause_id: str
    kind: str
    operation_ids: tuple
    related_operations: tuple
    logical_positions: tuple
    resources: tuple
    evidence: dict


@dataclass(frozen=True)
class AnalysisBatch:
    status: str
    causes: tuple
    strategies: tuple
    coverage: dict
    evidence: dict


@dataclass
class AnalysisContext:
    lowering: object
    backend: str = 'cudd'
    strategy_sources: dict = field(default_factory=dict)
    trace_analysis: dict | None = None
    _binding: str | None = None
    _stream: object = None
    _coverage: object = None
    _seen: set = field(default_factory=set)
    _causes: tuple = ()
    _prediction: object = None
    _index: object = None
    _exhausted: bool = False
    _trace_priorities: dict = field(default_factory=dict)


def mandatory_dependencies(storage, inputs=None):
    result = {op.operation_id: {storage.values[key].producer for key in op.inputs
                               if storage.values[key].producer is not None} for op in storage.instructions}
    for reuse in storage.reuse:
        result[reuse['writer']].update(reuse['previous_readers'])
    if inputs is not None:
        from vericcl.xml.v5_buffers import canonical_region
        readers = {}
        position = {op.operation_id: index for index, op in enumerate(storage.instructions)}
        for op in storage.instructions:
            for key in op.inputs:
                readers.setdefault(key, set()).add(op.operation_id)
        regions = {key: canonical_region(address, inputs) for key, address in storage.addresses.items()}
        for op in storage.instructions:
            target = regions[op.output]
            for key, value in storage.values.items():
                if key == op.output or (value.producer is not None and position[value.producer] >= position[op.operation_id]):
                    continue
                if target.overlaps(regions[key]):
                    result[op.operation_id].update(readers.get(key, set()) - {op.operation_id})
    return {key: frozenset(values) for key, values in result.items()}


def _ancestors(dependencies):
    result, pending = {}, set(dependencies)
    while pending:
        ready = sorted(key for key in pending if dependencies[key] <= result.keys())
        if not ready:
            raise SemanticError('repair dependency graph contains a cycle')
        for key in ready:
            result[key] = set(dependencies[key])
            for dep in dependencies[key]:
                result[key].update(result[dep])
            pending.remove(key)
    return result


# Mandatory dependencies and their transitive closure for recently checked
# (storage plan, inputs) objects. Both are immutable for a lowered bundle, and
# the entry keeps strong references, so identity checks cannot alias a reused
# id. Before this cache every generated strategy rebuilt the region-overlap
# dependency map and its closure, although only order strategies change it.
_LEGALITY_CACHE = {}
_LEGALITY_LIMIT = 8


def _legality_base(storage, inputs):
    key = (id(storage), id(inputs))
    entry = _LEGALITY_CACHE.get(key)
    if entry is not None and entry[0] is storage and entry[1] is inputs:
        return entry[2], entry[3]
    mandatory = mandatory_dependencies(storage, inputs)
    try:
        ancestors = _ancestors(mandatory)
    except SemanticError:
        ancestors = None
    if len(_LEGALITY_CACHE) >= _LEGALITY_LIMIT:
        _LEGALITY_CACHE.pop(next(iter(_LEGALITY_CACHE)))
    _LEGALITY_CACHE[key] = (storage, inputs, mandatory, ancestors)
    return mandatory, ancestors


def _paths(src, dst, participants, topology, lease):
    adjacency = {rank: [] for rank in participants}
    for edge in sorted(topology.links):
        if edge.src_rank in adjacency and edge.dst_rank in adjacency:
            adjacency[edge.src_rank].append(edge.dst_rank)
    stack = [(src, (src,))]
    while stack:
        lease.check()
        rank, path = stack.pop()
        if rank == dst:
            if len(path) > 2:
                yield path
            continue
        for nxt in reversed(adjacency[rank]):
            if nxt not in path:
                stack.append((nxt, path + (nxt,)))


def legal_strategy(strategy, bundle, context):
    """Filter declarative choices without treating busy links as forbidden."""
    storage, inputs, topology = bundle.storage_plan, context.lowering.inputs, context.lowering.topology
    ops = {op.operation_id: op for op in storage.instructions}
    if (strategy.family not in REPAIR_FAMILIES or not strategy.operation_ids
            or len(set(strategy.operation_ids)) != len(strategy.operation_ids)
            or set(strategy.operation_ids) - ops.keys()):
        return False
    if any(key not in strategy.operation_ids for key, _ in strategy.paths + strategy.channels):
        return False
    if len({key for key, _ in strategy.paths}) != len(strategy.paths) or len({key for key, _ in strategy.channels}) != len(strategy.channels):
        return False
    n, size = inputs.hyperparameters.slice_count, inputs.hyperparameters.slice_size_bytes
    constraints = bind_constraints(inputs)
    for key, path in strategy.paths:
        op = ops[key]
        source = storage.values[op.inputs[0]]
        if (op.kind not in {'send', 'rrc'} or len(path) < 2 or path[0] != source.rank or path[-1] != op.rank
                or len(set(path)) != len(path) or set(path) - set(op.participants)):
            return False
        ranges = tuple(DataRange(s // n, (s % n) * size, (s % n + 1) * size) for s in source.contributors)
        for src, dst in zip(path, path[1:]):
            edge = topology.links.get(LinkKey(src, dst))
            if edge is None or not constraints.allows(ranges, src, dst, op.stage_id):
                return False
    for key, channel in strategy.channels:
        op = ops[key]
        if op.kind not in {'send', 'rrc'} or isinstance(channel, bool) or not isinstance(channel, int) or channel < 0:
            return False
        edge = topology.links[LinkKey(storage.values[op.inputs[0]].rank, op.rank)]
        limit = min(inputs.solver.max_channels, edge.max_channels,
                    *(topology.shared_resources[r].max_channels for r in edge.resource_ids))
        if channel >= limit:
            return False
    mandatory, ancestors = _legality_base(storage, inputs)
    for first, second in strategy.orders:
        if first == second or {first, second} - set(strategy.operation_ids):
            return False
    if ancestors is None:
        # The mandatory graph itself is cyclic, so no strategy is legal.
        return False
    if not strategy.orders:
        return True
    if len(strategy.orders) == 1:
        # Adding "first before second" closes a cycle exactly when first
        # already depends (transitively) on second.
        first, second = strategy.orders[0]
        return second not in ancestors[first]
    dependencies = dict(mandatory)
    for first, second in strategy.orders:
        dependencies[second] = dependencies[second] | {first}
    try:
        _ancestors(dependencies)
    except SemanticError:
        return False
    return True


def _causes(bundle, index, prediction):
    ops = {op.operation_id: op for op in bundle.storage_plan.instructions}
    work = {op.operation_id: op for op in bundle.executable.operations}
    causes = []
    for key in sorted(index.critical_operations, key=lambda k: (-prediction.end_times[k], k)):
        if key not in ops:
            causes.append(BottleneckCause('cause:' + key, 'fixed_native_postprocess',
                (key,), (), (), (), {'service_us': index.service_us[key],
                'start_us': prediction.start_times[key], 'finish_us': prediction.end_times[key],
                'evidence_level': 'estimated_memory_pass',
                'repair_scope': 'fixed native postprocessing; optimize its upstream schedule'}))
            continue
        op = ops[key]
        logical = bundle.storage_plan.values[op.output].logical_position
        competitors = index.related[key]
        kind = 'resource_overlap' if competitors else 'tb_wait' if index.wait_us[key] > 0 else 'critical_service'
        causes.append(BottleneckCause('cause:' + key, kind, (key,), competitors, (logical,), work[key].resources,
            {'service_us': index.service_us[key], 'wait_us': index.wait_us[key], 'start_us': prediction.start_times[key],
             'finish_us': prediction.end_times[key], 'resource_competitors': list(competitors),
             'completion_tails': list(index.tails), 'evidence_level': 'current_model_causal_observation',
             'causal_scope': 'dependency and overlapping-resource superset',
             'strict_global_gain': 'requires_complete_candidate_replay'}))
    return tuple(causes)


def _trace_causes(bundle, report):
    storage = bundle.storage_plan
    ops = {op.operation_id: op for op in storage.instructions}
    related = {key: set() for key in report['operations']}
    for group in report['resource_overlap_groups'].values():
        for witness in group['witnesses']:
            related[witness['left']].add(witness['right'])
            related[witness['right']].add(witness['left'])
    causes = []
    for key in report['priority_operations']:
        observation = report['operations'][key]
        if not observation['local_endpoints']:
            continue
        evidence = dict(observation, capture_id=report['capture_id'],
            completion_tails=report['completion_tails'], completion_scope=report['completion_scope'],
            evidence_level='trace_observation_with_inferred_causes',
            causal_scope='executable dependencies and connected possible resource overlaps',
            strict_global_gain='requires_complete_current_model_replay_and_configured_measurement')
        causes.append(BottleneckCause('trace:' + str(report['capture_id']) + ':' + key,
            'trace_completion_tail' if key in report['completion_tails'] else 'trace_upstream_region',
            (key,), tuple(sorted(related[key])),
            (storage.values[ops[key].output].logical_position,) if key in ops else (),
            tuple(observation['resources']), evidence))
    return tuple(causes)


def _priority(key, index, context):
    return (context._trace_priorities.get(key, len(context._trace_priorities)),
            key not in index.critical_operations, key)


def _default_source(family, bundle, index, context):
    storage, topology, inputs = bundle.storage_plan, context.lowering.topology, context.lowering.inputs
    network = [op for op in storage.instructions if op.kind in {'send', 'rrc'}]
    # Critical closure first, then every remaining operation: priority is not a search cutoff.
    network.sort(key=lambda op: _priority(op.operation_id, index, context))
    if family in {'path', 'allocation'}:
        for op in network:
            source = storage.values[op.inputs[0]]
            for path in _paths(source.rank, op.rank, op.participants, topology, context.lowering.lease):
                if family == 'path':
                    yield StrategyRecord(family, (op.operation_id,), paths=((op.operation_id, path),))
                else:
                    # Complete slices are the integer allocation units. The domain
                    # includes compatible sibling slices, so the repair model can
                    # change several shares while retaining each source identity.
                    peers = tuple(other.operation_id for other in network if other.stage_id == op.stage_id
                        and other.kind == op.kind and other.rank == op.rank
                        and storage.values[other.inputs[0]].rank == source.rank and other.participants == op.participants)
                    if len(peers) > 1:
                        yield StrategyRecord(family, peers, paths=((op.operation_id, path),))
    elif family == 'order':
        seen = set()
        blocks = bundle.program.blocks
        if context.trace_analysis is not None:
            blocks = sorted(blocks, key=lambda block: min(
                (_priority(step.operation_id, index, context) for step in block.steps if step.kind != 'nop'),
                default=(len(context._trace_priorities), True, '')))
        for block in blocks:
            keys = tuple(dict.fromkeys(step.operation_id for step in block.steps if step.kind != 'nop'))
            passes = (True, False) if context.trace_analysis is not None else (None,)
            for prioritized in passes:
                for first, second in combinations(keys, 2):
                    context.lowering.lease.check()
                    if prioritized is not None and (first in context._trace_priorities or second in context._trace_priorities) != prioritized:
                        continue
                    if (second, first) not in seen:
                        seen.add((second, first))
                        yield StrategyRecord('order', (first, second), orders=((second, first),))
    elif family == 'channel':
        for op in network:
            edge = topology.links[LinkKey(storage.values[op.inputs[0]].rank, op.rank)]
            limit = min(inputs.solver.max_channels, edge.max_channels,
                        *(topology.shared_resources[r].max_channels for r in edge.resource_ids))
            for channel in range(limit):
                if channel != op.channel:
                    yield StrategyRecord('channel', (op.operation_id,), channels=((op.operation_id, channel),))
    elif family == 'joint':
        # Related-resource/data domains expand monotonically. These records leave
        # choices open for the joint solver instead of materializing their product.
        storage_keys = {op.operation_id for op in storage.instructions}
        seen = set()
        def domains(seeds, *, observed):
            pending, domain = list(seeds), set()
            while pending:
                context.lowering.lease.check()
                domain.update(pending)
                yield tuple(sorted(domain & storage_keys))
                expanded = {p for key in domain for p in index.predecessors[key]} | {p for key in domain for p in index.related[key]}
                if observed:
                    groups = {group for key in domain for group in context.trace_analysis['operations'][key]['overlap_groups']}
                    expanded.update(key for group in groups
                        for key in context.trace_analysis['resource_overlap_groups'][group]['operation_ids'])
                pending = sorted(expanded - domain)
        streams = []
        if context.trace_analysis is not None:
            seeds = list(index.tails) + context.trace_analysis['completion_tails']
            seeds += [key for keys in context.trace_analysis['local_completion_tails'].values() for key in keys
                      if context.trace_analysis['operations'][key]['frontier_us_bounds'] is None]
            streams.append(domains(seeds, observed=True))
        # Trace may add or prioritize domains, never replace the smaller domains
        # generated by the original model-only expansion sequence.
        streams.append(domains(index.tails, observed=False))
        for stream in streams:
            for selected in stream:
                if selected not in seen and len(selected) > 1:
                    seen.add(selected)
                    yield StrategyRecord('joint', selected)
        all_keys = tuple(sorted(op.operation_id for op in storage.instructions))
        if all_keys and all_keys not in seen:
            yield StrategyRecord('joint', all_keys)
    elif family == 'internal_root' and inputs.collective.kind.value == 'allreduce':
        yield StrategyRecord(family, tuple(op.operation_id for op in storage.instructions))
    elif family == 'structure':
        yield StrategyRecord(family, tuple(op.operation_id for op in storage.instructions))
    # A partition change needs separate measured/modelled benefit evidence. Its
    # branch stays unresolved until T12 supplies that evidence and a new S.


def _source(bundle, scope, context):
    iterators = {family: iter(context.strategy_sources.get(family, _default_source)(family, bundle, context._index, context))
                 for family in scope.families}
    # Round robin avoids starving order/channel choices behind a large path grammar.
    while iterators:
        for family in tuple(iterators):
            context.lowering.lease.check()
            try:
                yield family, next(iterators[family])
            except StopIteration:
                context._coverage.exhausted(family)
                del iterators[family]


def analyze(bundle, model, scope, context):
    selected = []
    try:
        context.lowering.lease.check()
        CompactBDD({'availability': 1}, backend=context.backend)
        if bundle.storage_plan is None or bundle.executable is None:
            raise SemanticError('causal analysis requires a lowered executable and state mapping')
        if context.trace_analysis is not None:
            from .online.trace_causality import validate_trace_causality
            validate_trace_causality(context.trace_analysis, bundle)
        binding = digest({'xml': bundle.xml_sha256, 'model': model_digest(model), 'scope': scope.scope_id,
                          'families': scope.families, 'input': context.lowering.inputs.input_sha256,
                          'effective': _request_contract(context.lowering.inputs),
                          'topology': context.lowering.topology.isomorphism_signature,
                          'storage': _serialize_storage(bundle.storage_plan), 'trace': context.trace_analysis})
        if context._binding != binding:
            prediction = predict(bundle.executable, model, exact=True, lease=context.lowering.lease)
            if prediction.t_pred_us is None:
                return AnalysisBatch('model_unavailable', (), (), {'proven_no_improvement': False},
                                     {'missing_components': sorted(prediction.missing_components)})
            index = build_causal_index(bundle.executable, prediction,
                                       semantic_predecessors=_legality_base(bundle.storage_plan, context.lowering.inputs)[0])
            context._binding, context._prediction, context._index = binding, prediction, index
            context._causes = _causes(bundle, index, prediction)
            context._trace_priorities = {}
            if context.trace_analysis is not None:
                context._trace_priorities = {key: ordinal for ordinal, key in enumerate(context.trace_analysis['priority_operations'])}
                context._causes = _trace_causes(bundle, context.trace_analysis) + context._causes
            context._coverage = SearchCoverage(scope.scope_id, model, scope.families)
            context._seen, context._exhausted = set(), False
            context._stream = _source(bundle, scope, context)
        while len(selected) < scope.batch_size and not context._exhausted:
            context.lowering.lease.check()
            try:
                family, strategy = next(context._stream)
            except StopIteration:
                context._exhausted = True
                break
            if not isinstance(strategy, StrategyRecord) or strategy.family != family or not legal_strategy(strategy, bundle, context):
                context._coverage.filtered += 1
                continue
            key = strategy.strategy_id
            if key in context._seen:
                context._coverage.duplicates += 1
                continue
            context._seen.add(key)
            context._coverage.generated(family)
            if context.trace_analysis is not None:
                strategy = replace(strategy, cause_ids=tuple('trace:' + str(context.trace_analysis['capture_id']) + ':' + key
                    for key in strategy.operation_ids if key in context._trace_priorities
                    and context.trace_analysis['operations'][key]['local_endpoints']))
            selected.append(strategy)
        if selected:
            bdd = StrategyDecisionBDD(len(selected), backend=context.backend)
            choices = bdd.select((index,) for index in range(len(selected)))
            strategies = tuple(selected[row[0]] for row in choices)
            status = 'opportunities'
        else:
            strategies = ()
            status = 'candidates_exhausted' if context._exhausted else 'empty_batch'
        return AnalysisBatch(status, context._causes, strategies, context._coverage.snapshot(),
            {'prediction_model': model.version, 't_pred_us': context._prediction.t_pred_us,
             'completion_tails': list(context._index.tails), 'backend': context.backend,
             'trace_analysis': None if context.trace_analysis is None else {
                 key: context.trace_analysis[key] for key in ('capture_id', 'xml_sha256', 'completion_tails',
                     'definite_completion_tails', 'completion_scope', 'global_no_improvement_proven')},
             'remaining_scope': 'generation exhaustion does not discharge unevaluated legal decisions'})
    except BudgetExhausted as error:
        status, message = 'budget_exhausted', str(error)
    except (SemanticError, BDDBackendError, ValueError, KeyError, TypeError) as error:
        status, message = 'analysis_error', str(error)
    return AnalysisBatch(status, context._causes, tuple(selected),
                         context._coverage.snapshot() if context._coverage else {'proven_no_improvement': False},
                         {'message': message, 'prediction_model': model.version})
