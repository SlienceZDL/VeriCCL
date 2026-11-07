"""Six-operator routing, integer demand columns and scoped route evidence.

Route operations have source contributions and true directed edges. Buffer
addresses, concrete TB ownership and complete executable T_pred are supplied
by the common lowering gate, not by the bandwidth abstraction here.
"""
from collections import Counter, defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from fractions import Fraction
from types import MappingProxyType
import time

from vericcl.errors import ConstructionInfeasibleError, SemanticError
from vericcl.execution.budget import BudgetExhausted, StandaloneLease, effective_cpu_budget
from vericcl.semantics.collective import CollectiveKind, required_outputs
from vericcl.semantics.ranges import DataRange, bind_constraints
from vericcl.solver.scip_api import ScipAdapter
from vericcl.topology.model import LinkKey
from .batch import _check
from .packing import pack_bandwidth
from .switch_paths import restore_physical_paths


@dataclass
class RouteContext:
    lease: object
    column_batch_size: int = 8
    records: list = field(default_factory=list)
    model_calls: int = 0

    def __post_init__(self):
        if isinstance(self.column_batch_size, bool) or not isinstance(self.column_batch_size, int) or self.column_batch_size < 1:
            raise SemanticError('column batch size must be a positive integer')


@contextmanager
def model_lease(inputs, context):
    parent = context.lease
    parent.check()
    child = parent.manager.acquire(parent.stage, model=True, parent=parent) if hasattr(parent, 'manager') else None
    granted = child or parent
    threads = child.threads if child is not None else min(inputs.solver.max_threads_per_model, effective_cpu_budget(inputs))
    class LimitedLease(StandaloneLease):
        @property
        def cancelled(self):
            return self._cancel.is_set() or granted.cancelled
    limited = LimitedLease(time.monotonic() + min(inputs.solver.per_model_timeout_s, granted.remaining_seconds()))
    limited.hard_deadline = granted.deadline
    try:
        yield limited, threads
    finally:
        if child is not None:
            child.close(cleanup_confirmed=not getattr(limited, 'ownership_unresolved', False))


@dataclass(frozen=True)
class RoutingDemand:
    demand_id: str
    mode: str
    logical_position: int
    contributors: frozenset[int]
    source_rank: int | None
    destinations: tuple[int, ...]
    output_slots: tuple


@dataclass(frozen=True)
class Route:
    root: int
    reduce_edges: tuple = ()
    send_edges: tuple = ()

    @property
    def edges(self):
        return self.reduce_edges + self.send_edges


@dataclass(frozen=True)
class RouteBatch:
    batch_id: str
    demand_ids: tuple[str, ...]
    route: Route
    physical_paths: tuple

    @property
    def slice_count(self):
        return len(self.demand_ids)


@dataclass(frozen=True)
class RouteOperation:
    operation_id: str
    demand_id: str
    kind: str
    edge: tuple[int, int]
    logical_position: int
    contributors: frozenset[int]
    payload_bytes: int
    stage_id: int
    dependencies: frozenset[str]
    resources: tuple[str, ...]


@dataclass(frozen=True)
class RouteBundle:
    inputs: object
    topology: object
    model_version: str
    demands: tuple[RoutingDemand, ...]
    batches: tuple[RouteBatch, ...]
    operations: tuple[RouteOperation, ...]
    link_load_bytes: object
    resource_load_bytes: object
    bandwidth_time_us: float
    bandwidth_time_ratio: tuple[int, int]
    routing_scope: str
    routing_exact_optimal: bool
    column_space_complete: bool
    generated_columns: int
    routing_solver_result: object = None
    _search: object = field(default=None, repr=False, compare=False)


def routing_demands(inputs):
    n = inputs.hyperparameters.slice_count
    if not n:
        raise SemanticError('routing needs the resolved positive slice partition')
    outputs = required_outputs(inputs.collective, inputs.rank_count, n)
    grouped = defaultdict(list)
    for slot, contributors in outputs.items():
        grouped[contributors].append(slot)
    kind = inputs.collective.kind
    mode = ('allreduce' if kind is CollectiveKind.ALL_REDUCE else
            'reduce' if kind in {CollectiveKind.REDUCE, CollectiveKind.REDUCE_SCATTER} else
            'path' if kind is CollectiveKind.ALL_TO_ALL else 'broadcast')
    demands = []
    for i, (contributors, slots) in enumerate(sorted(grouped.items(), key=lambda item: tuple(sorted(item[0])))):
        source = min(contributors) // n if len(contributors) == 1 else None
        demands.append(RoutingDemand('d-%08d' % i, mode, min(contributors) % n, contributors, source,
                                      tuple(sorted({slot.rank for slot in slots})), tuple(sorted(slots))))
    return tuple(demands)


def _physical_operation(inputs, topology, demand, route, lease=None):
    """Build data dependencies; reduction siblings remain reorderable."""
    size, n = inputs.hyperparameters.slice_size_bytes, inputs.hyperparameters.slice_count
    operations, incoming = [], defaultdict(list)
    contributors = {rank: frozenset({rank * n + demand.logical_position})
                    for rank in range(inputs.rank_count)} if route.reduce_edges else {}
    pending = set(route.reduce_edges)
    while pending:
        _check(lease)
        ready = sorted((src, dst) for src, dst in pending if not any(target == src for _, target in pending))
        if not ready:
            raise SemanticError('reduction route contains a directed cycle')
        for src, dst in ready:
            identifier = demand.demand_id + '-r-%d-%d' % (src, dst)
            operations.append(RouteOperation(identifier, demand.demand_id, 'rrc', (src, dst), demand.logical_position,
                contributors[src], size, 0, frozenset(incoming[src]), topology.link(LinkKey(src, dst)).resource_ids))
            incoming[dst].append(identifier)
            contributors[dst] = contributors[dst] | contributors[src]
            pending.remove((src, dst))
    ready_rank = {route.root}
    preceding = {route.root: frozenset(incoming[route.root])}
    pending = set(route.send_edges)
    stage = 1 if demand.mode == 'allreduce' and inputs.atom_constraints.stage_num != 1 else 0
    while pending:
        _check(lease)
        ready = sorted((src, dst) for src, dst in pending if src in ready_rank)
        if not ready:
            raise SemanticError('distribution route is unreachable or cyclic')
        for src, dst in ready:
            if dst in ready_rank:
                raise SemanticError('distribution route has multiple parents or a cycle')
            identifier = demand.demand_id + '-s-%d-%d' % (src, dst)
            operations.append(RouteOperation(identifier, demand.demand_id, 'send', (src, dst), demand.logical_position,
                demand.contributors, size, stage, preceding[src], topology.link(LinkKey(src, dst)).resource_ids))
            ready_rank.add(dst)
            preceding[dst] = frozenset({identifier})
            pending.remove((src, dst))
    return tuple(operations)


def _legal_route(inputs, topology, demand, route, lease=None):
    constraints = bind_constraints(inputs)
    n, size = inputs.hyperparameters.slice_count, inputs.hyperparameters.slice_size_bytes
    for op in _physical_operation(inputs, topology, demand, route, lease):
        ranges = tuple(DataRange(s // n, (s % n) * size, (s % n + 1) * size) for s in op.contributors)
        if not constraints.allows(ranges, *op.edge, op.stage_id):
            return False
    return True


def _reachable(nodes, root, edges):
    reached = {root}
    while True:
        new = reached | {dst for src, dst in edges if src in reached}
        if new == reached:
            return new == set(nodes)
        reached = new


def tree_columns(nodes, root, edges, lease=None):
    """Lazy complete directed arborescence enumeration with prefix pruning.

    Each nonroot chooses one parent. Prefixes containing a cycle or whose
    remaining graph cannot reach every node are discarded. This enumeration
    is a finite column space, not a bound on how many columns may be searched.
    """
    nodes, edges = tuple(nodes), tuple(sorted(set(edges)))
    if not _reachable(nodes, root, edges):
        return
    # A breadth-first tree supplies an inexpensive first feasible column.
    reached, queue, seed = {root}, deque([root]), []
    while queue:
        src = queue.popleft()
        for a, dst in edges:
            if a == src and dst not in reached:
                reached.add(dst); queue.append(dst); seed.append((src, dst))
    seed = tuple(sorted(seed))
    yield seed
    targets = tuple(rank for rank in nodes if rank != root)
    choices = {target: tuple(src for src, dst in edges if dst == target) for target in targets}
    def visit(index, parents):
        _check(lease)
        if index == len(targets):
            result = tuple(sorted((src, dst) for dst, src in parents.items()))
            if result != seed:
                yield result
            return
        target = targets[index]
        for src in choices[target]:
            cursor, seen = src, {target}
            while cursor in parents and cursor not in seen:
                seen.add(cursor); cursor = parents[cursor]
            if cursor in seen:
                continue
            updated = {**parents, target: src}
            remaining = tuple((a, b) for a, b in edges if b not in updated or updated[b] == a)
            if _reachable(nodes, root, remaining):
                yield from visit(index + 1, updated)
    yield from visit(0, {})


def _paths(source, target, edges, lease):
    def visit(path):
        _check(lease)
        if path[-1] == target:
            yield tuple(zip(path, path[1:])); return
        for src, dst in edges:
            if src == path[-1] and dst not in path:
                yield from visit(path + (dst,))
    yield from visit((source,))


def _columns(inputs, topology, demand, lease):
    nodes = tuple(range(inputs.rank_count))
    edges = tuple((key.src_rank, key.dst_rank) for key in topology.links)
    if demand.mode == 'path':
        for path in _paths(demand.source_rank, demand.destinations[0], edges, lease):
            route = Route(demand.source_rank, send_edges=path)
            if _legal_route(inputs, topology, demand, route, lease):
                yield route
        return
    roots = nodes if demand.mode == 'allreduce' else ((demand.destinations[0],) if demand.mode == 'reduce' else (demand.source_rank,))
    def at_root(root):
        if demand.mode == 'broadcast':
            for tree in tree_columns(nodes, root, edges, lease):
                yield Route(root, send_edges=tree)
        elif demand.mode == 'reduce':
            for tree in tree_columns(nodes, root, tuple((b, a) for a, b in edges), lease):
                yield Route(root, reduce_edges=tuple(sorted((b, a) for a, b in tree)))
        else:
            for reduce_tree in tree_columns(nodes, root, tuple((b, a) for a, b in edges), lease):
                for send_tree in tree_columns(nodes, root, edges, lease):
                    yield Route(root, tuple(sorted((b, a) for a, b in reduce_tree)), send_tree)
    pending = deque(iter(at_root(root)) for root in roots)
    while pending:
        generator = pending.popleft()
        try:
            route = next(generator)
        except StopIteration:
            continue
        pending.append(generator)
        if _legal_route(inputs, topology, demand, route, lease):
            yield route


def _group_key(inputs, demand):
    size = inputs.hyperparameters.slice_size_bytes
    start, end = demand.logical_position * size, (demand.logical_position + 1) * size
    n = inputs.hyperparameters.slice_count
    sources = {item // n for item in demand.contributors}
    restrictions = tuple((ban.data.source_rank, ban.src_rank, ban.dst_rank, ban.stage_id)
        for ban in bind_constraints(inputs).bans
        if ban.data.source_rank in sources and max(start, ban.data.start) < min(end, ban.data.end))
    return demand.mode, demand.source_rank, demand.destinations, restrictions


def _capacities(topology, size):
    times = {}
    for key, edge in topology.links.items():
        beta = edge.performance.beta_us_per_byte
        if beta is None:
            raise SemanticError('constructive routing requires a declared per-byte link cost')
        times[key.src_rank, key.dst_rank] = Fraction(str(beta)) * size
    resources = {}
    for name, resource in topology.shared_resources.items():
        if resource.capacity_bytes_per_us is None:
            raise SemanticError('constructive routing requires explicit shared-resource capacity')
        resources[name] = Fraction(size) / Fraction(str(resource.capacity_bytes_per_us))
    return times, resources


def _loads(topology, route):
    links = Counter(route.edges)
    resources = Counter(resource for edge in route.edges for resource in topology.link(LinkKey(*edge)).resource_ids)
    return links, resources


def _bundle(inputs, topology, model_version, demands, allocations, *, exact, complete, generated, scope, result=None, search=None, lease=None):
    by_id = {d.demand_id: d for d in demands}
    batches, operations, link_counts, resource_counts = [], [], Counter(), Counter()
    for i, (ids, route) in enumerate(allocations):
        _check(lease)
        batches.append(RouteBatch('batch-%08d' % i, tuple(ids), route, restore_physical_paths(topology, route.edges)))
        links, resources = _loads(topology, route)
        link_counts.update({key: value * len(ids) for key, value in links.items()})
        resource_counts.update({key: value * len(ids) for key, value in resources.items()})
        for identifier in ids:
            operations.extend(_physical_operation(inputs, topology, by_id[identifier], route, lease))
    times, resources = _capacities(topology, inputs.hyperparameters.slice_size_bytes)
    bandwidth = max([Fraction(0)] + [times[key] * count for key, count in link_counts.items()]
                    + [resources[key] * count for key, count in resource_counts.items()])
    size = inputs.hyperparameters.slice_size_bytes
    bundle = RouteBundle(inputs, topology, model_version, demands, tuple(batches), tuple(operations),
        MappingProxyType({key: count * size for key, count in link_counts.items()}),
        MappingProxyType({key: count * size for key, count in resource_counts.items()}), float(bandwidth),
        (bandwidth.numerator, bandwidth.denominator), scope, exact, complete, generated, result, search)
    check_routed_semantics(bundle, lease)
    return bundle


def check_routed_semantics(bundle, lease=None):
    """Independent contribution replay before physical buffer lowering."""
    expected = {d.demand_id: d for d in bundle.demands}
    if len(expected) != len(bundle.demands):
        raise SemanticError('route demand identities must be unique')
    assigned = [identifier for batch in bundle.batches for identifier in batch.demand_ids]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(expected):
        raise SemanticError('route batches must assign each complete slice demand exactly once')
    n = bundle.inputs.hyperparameters.slice_count
    operations = {op.operation_id: op for op in bundle.operations}
    if len(operations) != len(bundle.operations):
        raise SemanticError('route operation identities must be unique')
    canonical = {}
    for batch in bundle.batches:
        for identifier in batch.demand_ids:
            canonical.update((op.operation_id, op) for op in _physical_operation(
                bundle.inputs, bundle.topology, expected[identifier], batch.route, lease))
    if operations != canonical:
        raise SemanticError('route operations or data dependencies disagree with the fixed routes')
    by_demand = defaultdict(list)
    for op in bundle.operations:
        by_demand[op.demand_id].append(op)
    constraints = bind_constraints(bundle.inputs)
    for demand in bundle.demands:
        _check(lease)
        states = {rank: frozenset({rank * n + demand.logical_position}) for rank in range(bundle.inputs.rank_count)}
        if demand.mode in {'broadcast', 'path'}:
            states = {demand.source_rank: demand.contributors}
        done, pending = set(), {op.operation_id: op for op in by_demand[demand.demand_id]}
        while pending:
            _check(lease)
            ready = sorted((op for op in pending.values() if op.dependencies <= done), key=lambda op: op.operation_id)
            if not ready:
                raise SemanticError('route dependencies are cyclic or missing')
            for op in ready:
                src, dst = op.edge
                if states.get(src) != op.contributors:
                    raise SemanticError('routed transfer reads unavailable source contributions')
                if op.payload_bytes != bundle.inputs.hyperparameters.slice_size_bytes:
                    raise SemanticError('route payload must be exactly one whole slice')
                if op.resources != bundle.topology.link(LinkKey(src, dst)).resource_ids:
                    raise SemanticError('route resources must match the actual directed link')
                size = op.payload_bytes
                ranges = tuple(DataRange(s // n, (s % n) * size, (s % n + 1) * size) for s in op.contributors)
                if not constraints.allows(ranges, src, dst, op.stage_id):
                    raise SemanticError('route violates a source-byte directed-edge constraint')
                if op.kind == 'rrc':
                    prior = states.get(dst, frozenset())
                    if not prior or prior & op.contributors:
                        raise SemanticError('routed reduction repeats or lacks a contribution')
                    states[dst] = prior | op.contributors
                elif op.kind == 'send':
                    states[dst] = op.contributors
                else:
                    raise SemanticError('unknown routed network operation kind')
                done.add(op.operation_id); del pending[op.operation_id]
        if any(states.get(rank) != demand.contributors for rank in demand.destinations):
            raise SemanticError('route does not produce every required output contribution')
    # Demand interfaces also have to agree with the external collective.
    actual_outputs = {slot: demand.contributors for demand in bundle.demands for slot in demand.output_slots}
    if len(actual_outputs) != sum(len(demand.output_slots) for demand in bundle.demands):
        raise SemanticError('routed demand output slots must be unique')
    if actual_outputs != dict(required_outputs(bundle.inputs.collective, bundle.inputs.rank_count, n)):
        raise SemanticError('route demand interfaces differ from collective output semantics')
    return True


class RouteSearch:
    def __init__(self, inputs, topology, version, demands, context):
        self.inputs, self.topology, self.version, self.demands, self.context = inputs, topology, version, demands, context
        groups = defaultdict(list)
        for demand in demands:
            groups[_group_key(inputs, demand)].append(demand)
        self.groups = tuple(tuple(items) for items in groups.values())
        self.generators = [iter(_columns(inputs, topology, items[0], context.lease)) for items in self.groups]
        self.columns = [[] for _ in self.groups]
        self.complete = [False] * len(self.groups)
        self.incumbent = None
        self._capacity_times = None
        self._column_data = {}

    def _times(self):
        if self._capacity_times is None:
            self._capacity_times = _capacities(self.topology, self.inputs.hyperparameters.slice_size_bytes)
        return self._capacity_times

    def _column(self, route):
        """Exact loads and float constraint coefficients of one route column.

        Columns are only appended between solves, so every later model reuses
        these values instead of recomputing them with rational arithmetic.
        """
        data = self._column_data.get(route)
        if data is None:
            times, resource_times = self._times()
            links, resources = _loads(self.topology, route)
            data = (links, resources,
                    {edge: float(duration * links[edge]) for edge, duration in times.items()},
                    {resource: float(duration * resources[resource]) for resource, duration in resource_times.items()})
            self._column_data[route] = data
        return data

    def expand(self):
        changed = False
        for index, generator in enumerate(self.generators):
            if self.complete[index]:
                continue
            for _ in range(self.context.column_batch_size):
                _check(self.context.lease)
                try:
                    route = next(generator)
                except StopIteration:
                    self.complete[index] = True; changed = True; break
                self.columns[index].append(route); changed = True
            if not self.columns[index] and self.complete[index]:
                raise ConstructionInfeasibleError('complete route-column space cannot satisfy a slice demand')
        return changed

    def solve(self):
        inputs, context = self.inputs, self.context
        times, resource_times = self._times()
        with model_lease(inputs, context) as (lease, threads):
            gp, model = ScipAdapter.create_model('vericcl-integer-route-columns')
            try:
                ScipAdapter.configure(model, seed=inputs.solver.solver_seed, threads=threads, gap=inputs.solver.mip_gap, lease=lease)
                variables, loads = {}, {}
                objective = model.addVar('bandwidth_time', lb=0)
                for group_index, columns in enumerate(self.columns):
                    if not columns:
                        raise ConstructionInfeasibleError('no feasible route column was generated')
                    for column_index, route in enumerate(columns):
                        key = group_index, column_index
                        variables[key] = model.addVar('copies-%d-%d' % key, vtype='I', lb=0, ub=len(self.groups[group_index]))
                        loads[key] = self._column(route)
                    model.addCons(gp.quicksum(variables[group_index, j] for j in range(len(columns))) == len(self.groups[group_index]))
                # Same terms, coefficients and order as before (the serialized
                # model is unchanged); only the rational products are cached.
                for edge in times:
                    model.addCons(gp.quicksum(variables[key] * load[2][edge] for key, load in loads.items()) <= objective)
                for resource in resource_times:
                    model.addCons(gp.quicksum(variables[key] * load[3][resource] for key, load in loads.items()) <= objective)
                model.setObjective(objective)
                initial = {key: len(self.groups[key[0]]) if key[1] == 0 else 0 for key in variables}
                if self.incumbent is not None:
                    previous = {identifier: batch.route for batch in self.incumbent.batches for identifier in batch.demand_ids}
                    initial = {key: sum(previous[d.demand_id] == self.columns[key[0]][key[1]] for d in self.groups[key[0]]) for key in variables}
                def cost(counts):
                    # Exact bandwidth time: per-edge integer load totals over
                    # the columns actually used, then one rational product.
                    # Equal to summing duration * load * count over every column.
                    link_totals = dict.fromkeys(times, 0)
                    resource_totals = dict.fromkeys(resource_times, 0)
                    for key, count in counts.items():
                        if count:
                            links, resources = loads[key][0], loads[key][1]
                            for edge, value in links.items():
                                if edge in link_totals:
                                    link_totals[edge] += value * count
                            for resource, value in resources.items():
                                if resource in resource_totals:
                                    resource_totals[resource] += value * count
                    return max([Fraction(0)] + [times[edge] * total for edge, total in link_totals.items()]
                               + [resource_times[resource] * total for resource, total in resource_totals.items()])
                ScipAdapter.warm_start(model, [(variables[key], value) for key, value in initial.items()] + [(objective, float(cost(initial)))], complete=True)
                context.model_calls += 1
                result = model.optimize()
                chosen = initial
                if result.primal is not None:
                    proposed = {key: int(round(result.solution[var.name])) for key, var in variables.items()}
                    valid = all(0 <= value <= len(self.groups[key[0]]) for key, value in proposed.items())
                    valid &= all(sum(proposed[i, j] for j in range(len(columns))) == len(self.groups[i]) for i, columns in enumerate(self.columns))
                    if valid and cost(proposed) <= cost(initial):
                        chosen = proposed
                allocations = []
                for i, group in enumerate(self.groups):
                    cursor = 0
                    for j, route in enumerate(self.columns[i]):
                        count = chosen[i, j]
                        if count:
                            allocations.append((tuple(d.demand_id for d in group[cursor:cursor + count]), route))
                            cursor += count
                self.incumbent = _bundle(inputs, self.topology, self.version, self.demands, allocations, exact=False,
                    complete=all(self.complete), generated=sum(map(len, self.columns)), scope='integer_route_columns', result=result, search=self,
                    lease=context.lease)
                context.records.append({'phase': 'integer_routing', 'columns': self.incumbent.generated_columns,
                    'column_space_complete': all(self.complete), 'status': result.status,
                    'bandwidth_time_ratio': self.incumbent.bandwidth_time_ratio, 'bound_scope': 'numerical_column_model'})
                return self.incumbent
            finally:
                model.freeProb()


def construct_routes(inputs, topology, model, context):
    ScipAdapter.require()
    context.lease.check()
    if model is not None:
        from vericcl.performance.topology import project_topology
        topology = project_topology(topology, model, inputs.hyperparameters.slice_size_bytes)
    version = 'initial' if model is None else model.version
    demands = routing_demands(inputs)
    stages = inputs.atom_constraints.stage_num
    if stages is not None and stages not in ({1, 2} if inputs.collective.kind is CollectiveKind.ALL_REDUCE else {1}):
        raise ConstructionInfeasibleError('this constructive route grammar does not match the fixed logical stage count')
    times, _ = _capacities(topology, inputs.hyperparameters.slice_size_bytes)
    symmetric = all((b, a) in times and times[b, a] == value for (a, b), value in times.items())
    fast = (not topology.shared_resources and not inputs.atom_constraints.forbidden_transfers and symmetric
            and all(d.mode in {'broadcast', 'reduce'} for d in demands))
    if fast:
        reduction = demands[0].mode == 'reduce'
        root_items = defaultdict(list)
        for demand in demands:
            root_items[demand.destinations[0] if reduction else demand.source_rank].append(demand.demand_id)
        packed = pack_bandwidth(tuple(range(inputs.rank_count)), {(b, a) if reduction else (a, b): value for (a, b), value in times.items()}, root_items, context.lease)
        allocations = [(batch.items, Route(batch.root, tuple(sorted((b, a) for a, b in batch.edges)), ())) if reduction
                       else (batch.items, Route(batch.root, (), tuple(sorted(batch.edges)))) for batch in packed.batches]
        return _bundle(inputs, topology, version, demands, allocations, exact=True, complete=True,
                       generated=len(allocations), scope=packed.scope, lease=context.lease)
    search = RouteSearch(inputs, topology, version, demands, context)
    search.expand()
    return search.solve()
