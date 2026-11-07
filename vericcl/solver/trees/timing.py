"""Required native SCIP channel/order solve after integer bandwidth routing.

The search uses conservative fixed-concurrency durations as an initial
scheduling objective. It exports only realizable precedence and channel
decisions. The common executable lowering replays actual dynamic bandwidth,
local operations, buffer and TB dependencies to obtain complete T_pred.
"""
from collections import deque
from dataclasses import dataclass, field, replace
import heapq
from itertools import combinations, permutations
import random
from types import MappingProxyType

from vericcl.errors import SemanticError
from vericcl.execution.budget import BudgetExhausted
from vericcl.solver.scip_api import ScipAdapter
from vericcl.solver.scheduling import fixed_topology_transfer_duration_us
from vericcl.topology.model import LinkKey
from .operators import model_lease, check_routed_semantics, Route, _bundle, _legal_route


@dataclass(frozen=True)
class RoutedSchedule:
    operations: tuple
    channels: object
    dependencies: object
    start_us: object
    finish_us: object


@dataclass(frozen=True)
class RoutedCandidate:
    bundle: object
    schedule: RoutedSchedule
    communication_time_us: float
    initial_time_us: float
    solver_result: object
    model_calls: int
    incumbent_preserved: bool = True
    runtime_checked: bool = False
    full_search_proven: bool = False


def _same_accumulator(left, right):
    return (left.kind == right.kind == 'rrc' and left.demand_id == right.demand_id
            and left.edge[1] == right.edge[1])


def _network_conflict(left, right):
    return left.edge == right.edge or bool(set(left.resources) & set(right.resources))


def _domain(bundle):
    topology = bundle.topology
    channels, durations = {}, {}
    for op in bundle.operations:
        edge = topology.link(LinkKey(*op.edge))
        maximum = min([bundle.inputs.solver.max_channels, edge.max_channels]
                      + [topology.shared_resources[r].max_channels for r in op.resources])
        channels[op.operation_id] = tuple(range(maximum))
        durations[op.operation_id] = fixed_topology_transfer_duration_us(topology, edge.key, op.payload_bytes,
                                                                        bundle.inputs.solver.max_channels)
    return channels, durations


def _dependency_graph(operations, dependencies):
    successors = {op.operation_id: [] for op in operations}
    degrees = {}
    for identifier in successors:
        degrees[identifier] = len(dependencies[identifier])
        for predecessor in dependencies[identifier]:
            if predecessor not in successors:
                raise SemanticError('fixed-route decisions contain a missing dependency')
            successors[predecessor].append(identifier)
    return degrees, successors


def _schedule(operations, channels, dependencies, start, finish):
    return RoutedSchedule(tuple(operations), MappingProxyType(dict(channels)),
        MappingProxyType({key: frozenset(value) for key, value in dependencies.items()}),
        MappingProxyType(start), MappingProxyType(finish))


def _earliest(operations, channels, dependencies, durations, lease=None):
    if lease is not None:
        lease.check()
    degrees, successors = _dependency_graph(operations, dependencies)
    ready = deque(identifier for identifier, degree in degrees.items() if degree == 0)
    start, finish = {}, {}
    while ready:
        if lease is not None:
            lease.check()
        identifier = ready.popleft()
        start[identifier] = max((finish[p] for p in dependencies[identifier]), default=0.)
        finish[identifier] = start[identifier] + durations[identifier]
        for follower in successors[identifier]:
            degrees[follower] -= 1
            if degrees[follower] == 0:
                ready.append(follower)
    if len(finish) != len(degrees):
        raise SemanticError('fixed-route execution decisions contain a dependency cycle')
    return _schedule(operations, channels, dependencies, start, finish)


@dataclass
class _ReadyGroup:
    keys: tuple
    channel: int
    duration: float
    releases: list = field(default_factory=list)
    eligible: list = field(default_factory=list)
    version: int = 0

    def candidate(self, tails, finishes):
        frontier = max((finishes[tails[key]] for key in self.keys if key in tails), default=0.)
        while self.releases and self.releases[0][0] <= frontier:
            _, identifier = heapq.heappop(self.releases)
            if identifier not in finishes:
                heapq.heappush(self.eligible, identifier)
        while self.eligible and self.eligible[0] in finishes:
            heapq.heappop(self.eligible)
        if self.eligible:
            start, identifier = frontier, self.eligible[0]
        else:
            while self.releases and self.releases[0][1] in finishes:
                heapq.heappop(self.releases)
            if not self.releases:
                return None
            start, identifier = self.releases[0]
        return (start + self.duration, start, identifier, self.channel)


def _resource_keys(op, channel):
    keys = [('edge', op.edge, channel)]
    keys.extend(('resource', resource, channel) for resource in sorted(set(op.resources)))
    if op.kind == 'rrc':
        keys.append(('accumulator', op.demand_id, op.edge[1]))
    return tuple(keys)


def _initial(bundle, domains, durations, lease=None):
    operations = {op.operation_id: op for op in bundle.operations}
    degrees, successors = _dependency_graph(bundle.operations,
        {identifier: op.dependencies for identifier, op in operations.items()})
    selected, dependencies, starts, finishes, tails = {}, {}, {}, {}, {}
    groups, group_ids, queue = [], {}, []

    def add_ready(identifier):
        op = operations[identifier]
        release = max((finishes[p] for p in op.dependencies), default=0.)
        changed = set()
        for channel in domains[identifier]:
            keys = _resource_keys(op, channel)
            signature = (keys, channel, durations[identifier])
            if signature not in group_ids:
                group_ids[signature] = len(groups)
                groups.append(_ReadyGroup(keys, channel, durations[identifier]))
            index = group_ids[signature]
            heapq.heappush(groups[index].releases, (release, identifier))
            changed.add(index)
        return changed

    def enqueue(index):
        group = groups[index]
        group.version += 1
        candidate = group.candidate(tails, finishes)
        if candidate is not None:
            heapq.heappush(queue, candidate + (index, group.version))

    changed = set()
    for identifier, degree in degrees.items():
        if lease is not None:
            lease.check()
        if degree == 0:
            changed.update(add_ready(identifier))
    for index in changed:
        enqueue(index)
    while queue:
        if lease is not None:
            lease.check()
        previous = heapq.heappop(queue)
        index, version = previous[4:]
        group = groups[index]
        if version != group.version:
            continue
        candidate = group.candidate(tails, finishes)
        if candidate is None:
            continue
        # Resource frontiers and removals only increase existing keys. New
        # ready operations are enqueued immediately because they may lower it.
        if candidate != previous[:4]:
            enqueue(index)
            continue
        end, start, identifier, channel = candidate
        predecessors = set(operations[identifier].dependencies)
        predecessors.update(tails[key] for key in group.keys if key in tails)
        selected[identifier], dependencies[identifier] = channel, frozenset(predecessors)
        starts[identifier], finishes[identifier] = start, end
        for key in group.keys:
            tails[key] = identifier
        changed = {index}
        for follower in successors[identifier]:
            degrees[follower] -= 1
            if degrees[follower] == 0:
                changed.update(add_ready(follower))
        for changed_index in changed:
            enqueue(changed_index)
    if len(finishes) != len(operations):
        raise SemanticError('fixed-route operation dependencies cannot be scheduled')
    return _schedule(bundle.operations, selected, dependencies, starts, finishes)


def _makespan(schedule):
    return max(schedule.finish_us.values(), default=0.)


def solve_fixed_seed(bundle, context):
    """Check and optimize a compact fixed-order seed before the full MILP.

    This model retains only one feasible set of channel/order decisions. Its
    numerical optimum is not an optimum over other orders or channels.
    """
    check_routed_semantics(bundle, context.lease)
    domains, durations = _domain(bundle)
    initial = _initial(bundle, domains, durations, context.lease)
    initial_time = _makespan(initial)
    with model_lease(bundle.inputs, context) as (lease, threads):
        _, model = ScipAdapter.create_model('vericcl-fixed-route-seed')
        try:
            ScipAdapter.configure(model, seed=bundle.inputs.solver.solver_seed, threads=threads,
                                  gap=bundle.inputs.solver.mip_gap, lease=lease)
            starts, seed = {}, []
            makespan = model.addVar('communication_finish', lb=0, ub=initial_time)
            seed.append((makespan, initial_time))
            for op in bundle.operations:
                identifier = op.operation_id
                starts[identifier] = model.addVar('start-' + identifier, lb=0, ub=initial_time)
                seed.append((starts[identifier], initial.start_us[identifier]))
                model.addCons(makespan >= starts[identifier] + durations[identifier])
            for identifier, predecessors in initial.dependencies.items():
                for predecessor in predecessors:
                    model.addCons(starts[identifier] >= starts[predecessor] + durations[predecessor])
            model.setObjective(makespan)
            accepted = ScipAdapter.warm_start(model, seed, complete=True)
            if not accepted:
                raise SemanticError('the fixed-route seed failed native SCIP feasibility checks')
            context.model_calls += 1
            result = model.optimize()
            context.records.append({'phase': 'fixed_route_timing', 'status': result.status,
                'initial_communication_time_us': initial_time, 'communication_time_us': initial_time,
                'native_warm_start_accepted': accepted, 'scope': 'fixed_route_seed_timing'})
            return RoutedCandidate(bundle, initial, initial_time, initial_time, result, context.model_calls)
        finally:
            model.freeProb()


def solve_fixed_routes(bundle, context):
    check_routed_semantics(bundle, context.lease)
    domains, durations = _domain(bundle)
    initial = _initial(bundle, domains, durations, context.lease)
    initial_time = _makespan(initial)
    with model_lease(bundle.inputs, context) as (lease, threads):
        gp, model = ScipAdapter.create_model('vericcl-fixed-routes-order-channels')
        try:
            ScipAdapter.configure(model, seed=bundle.inputs.solver.solver_seed, threads=threads,
                                  gap=bundle.inputs.solver.mip_gap, lease=lease)
            seed, choice, starts, ends, pairs = [], {}, {}, {}, []
            makespan = model.addVar('communication_finish', lb=0, ub=initial_time)
            seed.append((makespan, initial_time))
            for op in bundle.operations:
                identifier = op.operation_id
                starts[identifier] = model.addVar('start-' + identifier, lb=0, ub=initial_time)
                ends[identifier] = model.addVar('end-' + identifier, lb=0, ub=initial_time)
                seed.extend(((starts[identifier], initial.start_us[identifier]), (ends[identifier], initial.finish_us[identifier])))
                for channel in domains[identifier]:
                    choice[identifier, channel] = model.addVar('channel-%s-%d' % (identifier, channel), vtype='B')
                    seed.append((choice[identifier, channel], int(initial.channels[identifier] == channel)))
                model.addCons(gp.quicksum(choice[identifier, c] for c in domains[identifier]) == 1)
                model.addCons(ends[identifier] == starts[identifier] + durations[identifier])
                model.addCons(makespan >= ends[identifier])
            for op in bundle.operations:
                for predecessor in op.dependencies:
                    model.addCons(starts[op.operation_id] >= ends[predecessor])
            for index, (left, right) in enumerate(combinations(bundle.operations, 2)):
                if index % 256 == 0:
                    lease.check()
                accumulator = _same_accumulator(left, right)
                network = _network_conflict(left, right)
                if not accumulator and not network:
                    continue
                a, b = left.operation_id, right.operation_id
                order = model.addVar('order-%d' % index, vtype='B')
                # The seed's resource conflicts never overlap. An arbitrary
                # direction for inactive channel pairs is still a valid seed.
                before = int((initial.start_us[a], a) < (initial.start_us[b], b))
                seed.append((order, before))
                pairs.append((left, right, order))
                if accumulator:
                    ScipAdapter.indicator(model, order, True, ends[a] <= starts[b])
                    ScipAdapter.indicator(model, order, False, ends[b] <= starts[a])
                    continue
                for channel in sorted(set(domains[a]) & set(domains[b])):
                    same = model.addVar('same-%d-%d' % (index, channel), vtype='B')
                    first = model.addVar('first-%d-%d' % (index, channel), vtype='B')
                    second = model.addVar('second-%d-%d' % (index, channel), vtype='B')
                    model.addConsAnd([choice[a, channel], choice[b, channel]], same)
                    model.addConsAnd([same, order], first)
                    model.addCons(second == same - first)
                    ScipAdapter.indicator(model, first, True, ends[a] <= starts[b])
                    ScipAdapter.indicator(model, second, True, ends[b] <= starts[a])
                    active = int(initial.channels[a] == initial.channels[b] == channel)
                    seed.extend(((same, active), (first, active * before), (second, active * (1 - before))))
            model.setObjective(makespan)
            warm_start_accepted = ScipAdapter.warm_start(model, seed, complete=True)
            if not warm_start_accepted:
                raise SemanticError('the fixed-route initial schedule failed native SCIP feasibility checks')
            context.model_calls += 1
            result = model.optimize()
            chosen = initial
            if result.primal is not None:
                selected = {op.operation_id: max(domains[op.operation_id],
                    key=lambda c: result.solution[choice[op.operation_id, c].name]) for op in bundle.operations}
                dependencies = {op.operation_id: set(op.dependencies) for op in bundle.operations}
                for left, right, order in pairs:
                    a, b = left.operation_id, right.operation_id
                    if _same_accumulator(left, right) or selected[a] == selected[b]:
                        predecessor, follower = (a, b) if result.solution[order.name] > .5 else (b, a)
                        dependencies[follower].add(predecessor)
                try:
                    proposed = _earliest(bundle.operations, selected, dependencies, durations, context.lease)
                    if _makespan(proposed) <= initial_time + 1e-9:
                        chosen = proposed
                except BudgetExhausted:
                    raise
                except SemanticError:
                    # A numerical/discrete extraction failure cannot displace
                    # the independently checked initial fixed-route schedule.
                    context.records.append({'phase': 'fixed_route_timing', 'status': 'invalid_solver_order',
                                            'incumbent_retained': True})
            context.records.append({'phase': 'fixed_route_timing', 'status': result.status,
                'initial_communication_time_us': initial_time, 'communication_time_us': _makespan(chosen),
                'native_warm_start_accepted': warm_start_accepted, 'scope': 'fixed_route_conservative_timing'})
            return RoutedCandidate(bundle, chosen, _makespan(chosen), initial_time, result, context.model_calls)
        finally:
            model.freeProb()


def _route_signature(bundle):
    """Everything the fixed-route timing models read from a route bundle."""
    return (bundle.operations, tuple((batch.demand_ids, batch.route) for batch in bundle.batches))


def _binomial_reduce_edges(root, order, available):
    """Binomial in-tree to ``root``: position i sends its partial result to the
    position with i's lowest set bit cleared, so the root's accumulator
    receives only ceil(log2 P) partials instead of P-1 serialized ones."""
    sequence = (root,) + tuple(order)
    edges = []
    for index in range(1, len(sequence)):
        edge = (sequence[index], sequence[index & (index - 1)])
        if edge not in available:
            return None
        edges.append(edge)
    return tuple(sorted(edges))


def _reduction_orders(root, nodes, rng, limit):
    others = [rank for rank in nodes if rank != root]
    orders, seen = [], set()

    def add(sequence):
        sequence = tuple(sequence)
        if sequence not in seen:
            seen.add(sequence)
            orders.append(sequence)

    add(sorted(others, key=lambda rank: (rank ^ root, rank)))
    ring = sorted(others, key=lambda rank: (rank - root) % len(nodes))
    for shift in range(len(ring)):
        add(ring[shift:] + ring[:shift])
    if len(others) <= 5:
        for permutation in permutations(others):
            add(permutation)
    else:
        attempts = 0
        while len(orders) < limit and attempts < 4 * limit:
            attempts += 1
            sample = list(others)
            rng.shuffle(sample)
            add(sample)
    return orders


def reshape_reductions(bundle, context, *, passes=2, order_limit=24):
    """Latency-aware reduction trees for a fixed integer allocation.

    The integer routing objective counts link and resource loads only, so all
    reduction trees tie and the breadth-first seed (a star whose root
    accumulator serializes P-1 receive-reduce steps) is kept. This bounded
    coordinate search keeps every batch's root, slices and distribution tree,
    tries binomial reduction trees over several rank orders, and scores each
    complete allocation with the same fixed-duration list scheduler used for
    the seed (it models edge, shared-resource and accumulator conflicts). It
    returns a variant only when that estimate strictly improves; the variant is
    then timed and lowered like any other candidate, so acceptance still uses
    the exact complete T_pred.
    """
    inputs, topology = bundle.inputs, bundle.topology
    nodes = tuple(range(inputs.rank_count))
    if len(nodes) < 3 or not any(batch.route.reduce_edges for batch in bundle.batches):
        return None
    available = {(key.src_rank, key.dst_rank) for key in topology.links}
    by_id = {demand.demand_id: demand for demand in bundle.demands}
    rng = random.Random(inputs.solver.solver_seed)

    def build(routes):
        return _bundle(inputs, topology, bundle.model_version, bundle.demands,
                       [(batch.demand_ids, route) for batch, route in zip(bundle.batches, routes)],
                       exact=False, complete=bundle.column_space_complete, generated=bundle.generated_columns,
                       scope='reduction_tree_reshaping', lease=context.lease)

    def estimate(candidate):
        # Lexicographic (makespan, total finish time): the makespan is a global
        # maximum, so improving one root's tree while the others still bound it
        # would otherwise look like no progress and stall the search.
        domains, durations = _domain(candidate)
        schedule = _initial(candidate, domains, durations, context.lease)
        return _makespan(schedule), sum(schedule.finish_us.values())

    routes = [batch.route for batch in bundle.batches]
    before = best = estimate(bundle)
    orders = {}
    evaluated = 0
    for _ in range(passes):
        improved = False
        for index, batch in enumerate(bundle.batches):
            route = routes[index]
            if not route.reduce_edges:
                continue
            for order in orders.setdefault(route.root, _reduction_orders(route.root, nodes, rng, order_limit)):
                context.lease.check()
                edges = _binomial_reduce_edges(route.root, order, available)
                if edges is None or edges == routes[index].reduce_edges:
                    continue
                trial = Route(route.root, edges, route.send_edges)
                if not all(_legal_route(inputs, topology, by_id[identifier], trial, context.lease)
                           for identifier in batch.demand_ids):
                    continue
                proposal = routes[:index] + [trial] + routes[index + 1:]
                evaluated += 1
                value = estimate(build(proposal))
                if value < best:
                    best, routes, improved = value, proposal, True
        if not improved:
            break
    context.records.append({'phase': 'reduction_tree_reshaping', 'estimated_before_us': before[0],
                            'estimated_after_us': best[0], 'evaluated': evaluated,
                            'scope': 'fixed allocation; binomial reduction trees; fixed-duration estimate'})
    # Only a strictly shorter estimated makespan justifies timing another candidate.
    return build(routes) if best[0] < before[0] else None


class CandidateStream:
    def __init__(self, bundle, context):
        self.bundle, self.context, self.stop_reason, self._started = bundle, context, None, False

    def __iter__(self):
        if self._started:
            raise SemanticError('fixed-route candidate streams can be consumed once')
        self._started = True
        bundle = self.bundle
        timed = {}
        reshaped = False
        try:
            while True:
                # Yield a native-checked fixed-order seed first. Later full
                # channel/order search uses only original data dependencies;
                # a model timeout cannot erase this already yielded candidate.
                timing_returned = False
                signature = _route_signature(bundle)
                if signature in timed:
                    # Column generation can return an integer allocation that
                    # was already timed. Its timing models and schedules are
                    # identical, so intermediate repeats only continue the
                    # route search. The bundle that closes the column space is
                    # still reported, bound to the already checked schedules,
                    # so the stream's last candidates carry that evidence.
                    timing_returned = True
                    self.context.records.append({'phase': 'fixed_route_timing', 'status': 'duplicate_routes_skipped',
                                                 'column_space_complete': bundle.column_space_complete})
                    if bundle.column_space_complete:
                        for previous in timed[signature]:
                            yield replace(previous, bundle=bundle)
                else:
                    try:
                        produced = []
                        candidate = solve_fixed_seed(bundle, self.context)
                        timing_returned = True
                        produced.append(candidate)
                        yield candidate
                        if bundle.operations:
                            candidate = solve_fixed_routes(bundle, self.context)
                            produced.append(candidate)
                            yield candidate
                        # Only completely timed bundles are skipped later; a
                        # bundle whose timing hit a model budget is retried.
                        timed[signature] = tuple(produced)
                    except BudgetExhausted:
                        self.context.lease.check()
                        self.context.records.append({'phase': 'fixed_route_timing', 'status': 'timing_model_budget'})
                if not reshaped and timing_returned:
                    # One latency-aware alternative for the first allocation:
                    # same slices, roots and distribution trees, reshaped
                    # reduction trees. It is an extra candidate, never a
                    # replacement, and the column search continues unchanged.
                    reshaped = True
                    variant = reshape_reductions(bundle, self.context)
                    if variant is not None and _route_signature(variant) not in timed:
                        try:
                            produced = []
                            candidate = solve_fixed_seed(variant, self.context)
                            produced.append(candidate)
                            yield candidate
                            if variant.operations:
                                candidate = solve_fixed_routes(variant, self.context)
                                produced.append(candidate)
                                yield candidate
                            timed[_route_signature(variant)] = tuple(produced)
                        except BudgetExhausted:
                            self.context.lease.check()
                            self.context.records.append({'phase': 'fixed_route_timing', 'status': 'timing_model_budget',
                                                         'scope': 'reduction_tree_reshaping'})
                search = bundle._search
                if search is None:
                    self.stop_reason = 'initial_route_seed_complete' if timing_returned else 'timing_model_budget'
                    return
                if all(search.complete):
                    self.stop_reason = 'initial_column_space_exhausted' if timing_returned else 'timing_model_budget'
                    return
                while search.expand():
                    try:
                        bundle = search.solve()
                        break
                    except BudgetExhausted:
                        self.context.lease.check()
                        self.context.records.append({'phase': 'integer_routing', 'status': 'routing_model_budget'})
                        if all(search.complete):
                            self.stop_reason = 'routing_model_budget'
                            return
        except BudgetExhausted:
            self.stop_reason = 'initial_synthesis_budget'
            self.context.records.append({'phase': 'constructive_trees', 'status': self.stop_reason})
