"""Scheduling-bottleneck repair (repair type 3).

Used when a scheduling hotspot has no idle sibling flow in its own chunk, so
re-dividing slices (type 2) cannot balance it. Pipeline:

1. Related schedules (symbolic execution). Starting from the hotspot, collect
   the transfers that occupy the congested resource during the hot buckets,
   the threadblock/order predecessors that decided their start (critical
   branches of the symbolic start times) and their data consumers up to
   ``dependency_depth``. The scope is capped at ``max_scope_operations``,
   ranked hotspot first, then competitors by overlap, then the dependency cone,
   so it never grows to the whole topology.
2. Repair plans (BDD). A transfer on the congested resource may be replaced by
   a path that starts at *any* rank that already owns its value (AllGather and
   AllReduce: once a rank holds a slice it can forward it) and runs over links
   that are idle in the hot buckets. Plan legality is decided with BDD
   relations: an owner whose copy is produced after the transfer (Before
   relation of the symbolic execution) would create a cycle; edges must be
   allowed for the slice's source bytes and idle; paths come from bounded
   backward reachability over the idle-edge relation.
3. Coarse screening (symbolic estimate). Each plan's arrival is estimated from
   the owner's ready time plus the hop costs, slowed by each link's measured
   utilization in the hot window. Plans that cannot arrive earlier than the
   current transfer are dropped; the best ``plans_per_slice`` per transfer
   survive.
4. Atom constraints. Every transfer in the scope becomes a repair atom
   {s: slice vector, pd: candidate paths (the current one first), t: start/end
   variables}; consumers in the scope keep their path but get time variables.
5. Local MILP (SCIP, in-process). Binary path choice per atom, hop timing,
   data precedence (owner ready time, hop order, consumers after arrival),
   disjunctive sharing of every link/resource inside the scope, no later
   arrival for transfers whose consumers lie outside the scope; minimize the
   latest finish in the scope, then the number of changed paths.

The MILP choice and the best single plans are proposed as complete networks,
lowered, repaired for correctness if needed (type 1), checked for unchanged
outside work and compared with the incumbent on the complete predicted time.
"""
from collections import defaultdict
from dataclasses import dataclass, field

from vericcl.errors import SemanticError
from vericcl.semantics.atom import SliceVec
from vericcl.semantics.ranges import DataRange, bind_constraints
from vericcl.topology.model import LinkKey
from vericcl.xml.v5_model import NetworkCandidate
from .correctness import _network_from_storage, route_ops, symbolic_ownership
from .strategies import RepairDomain


@dataclass(frozen=True)
class BottleneckSettings:
    max_scope_operations: int = 48
    dependency_depth: int = 2
    plans_per_slice: int = 3
    max_plans: int = 24

    @classmethod
    def from_inputs(cls, inputs):
        config = (inputs.request_config or {}).get('sketch', {}).get('repair', {}) or {}
        values = dict(config.get('bottleneck') or {})
        return cls(**{key: values[key] for key in cls.__dataclass_fields__ if values.get(key) is not None})


@dataclass(frozen=True)
class Plan:
    operation_id: str
    path: tuple
    owner: int
    ready_us: float
    estimate_us: float


@dataclass(frozen=True)
class RepairAtom:
    """Atom-shaped decision: slice vector, candidate paths, time variables."""
    s: SliceVec
    operation_id: str
    pd: tuple                      # candidate paths; index 0 is the current path
    owners: tuple                  # owner rank of each candidate path
    ready_us: tuple                # owner ready time of each candidate path
    deadline_us: float | None      # outside consumers must not see a later arrival


@dataclass
class BottleneckScope:
    hotspot: str
    resource: str
    buckets: tuple
    reroutable: list               # transfers on the congested resource
    consumers: list                # data consumers kept on their path
    evidence: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 1. related schedules

def _resources(op, topology):
    edge = topology.links[LinkKey(op.src_rank, op.dst_rank)]
    return ('link:%d:%d' % (op.src_rank, op.dst_rank),) + tuple(edge.resource_ids)


def related_scope(hotspot, execution, ops, prediction, topology, settings):
    by_id = {op.operation_id: op for op in ops}
    consumers_of = defaultdict(set)
    for op in ops:
        for dep in op.dependencies:
            consumers_of[dep].add(op.operation_id)
    # Competitors on the congested resource, by time overlap with the hotspot.
    start, end = prediction.start_times, prediction.end_times
    h0, h1 = start[hotspot.operation_id], end[hotspot.operation_id]
    competitors = [key for key in set(hotspot.related_operations) | {hotspot.operation_id}
                   if key in by_id and hotspot.congested_resource in _resources(by_id[key], topology)]
    overlap = lambda key: max(0.0, min(h1, end.get(key, 0.0)) - max(h0, start.get(key, 0.0)))
    reroutable = sorted(competitors, key=lambda key: (key != hotspot.operation_id, -overlap(key), key))
    # Order predecessors that decided the hotspot's start (symbolic critical branches).
    chain, current = [], hotspot.operation_id
    while True:
        label, predecessor = execution.critical_predecessor(current)
        if label not in {'tb', 'order'} or predecessor is None or predecessor not in by_id or predecessor in chain:
            break
        chain.append(predecessor)
        current = predecessor
    for key in chain:
        if key not in reroutable and hotspot.congested_resource in _resources(by_id[key], topology):
            reroutable.append(key)
    reroutable = reroutable[:max(1, settings.max_scope_operations // 2)]
    consumers, frontier = [], list(reroutable)
    for _ in range(settings.dependency_depth):
        nxt = []
        for key in frontier:
            for consumer in sorted(consumers_of[key]):
                if consumer not in reroutable and consumer not in consumers:
                    consumers.append(consumer)
                    nxt.append(consumer)
        frontier = nxt
    consumers = consumers[:max(0, settings.max_scope_operations - len(reroutable))]
    return BottleneckScope(hotspot.operation_id, hotspot.congested_resource, tuple(hotspot.buckets), reroutable,
                           consumers, {'tb_order_chain': chain})


# ---------------------------------------------------------------------------
# 2./3. plans: BDD legality and reachability, symbolic screening

def _hop_service(lowering, src, dst, kind):
    size = lowering.inputs.hyperparameters.slice_size_bytes
    model = lowering.model
    cost = model.service_for(('rrc:' if kind == 'rrc' else 'link:') + '%d:%d' % (src, dst), size) \
        or model.service_for('link:%d:%d' % (src, dst), size)
    if cost is None:
        raise SemanticError('no cost for link %d->%d' % (src, dst))
    return cost.alpha_us + cost.beta_us_per_byte * size


def _window_utilization(report, resource, buckets):
    return max((report.utilization.get((resource, b), 0.0) for b in buckets), default=0.0)


def repair_plans(scope, ops, execution, report, lowering, settings, hotspot_settings):
    from vericcl.symbolic.bdd import BddSpace
    from vericcl.verification.symbolic_check import ScheduleRelations
    inputs, topology = lowering.inputs, lowering.topology
    n, size = inputs.hyperparameters.slice_count, inputs.hyperparameters.slice_size_bytes
    ranks = inputs.rank_count
    by_id = {op.operation_id: op for op in ops}
    times = execution.times()                      # op -> (start, end) under the bound model
    ownership = symbolic_ownership(ops, inputs)
    constraints = bind_constraints(inputs)
    relations = ScheduleRelations(execution, inputs)
    space, before = relations.space, relations.before
    threshold = hotspot_settings.idle_utilization_threshold
    hops = hotspot_settings.max_alt_hops
    plans = defaultdict(list)
    for key in scope.reroutable:
        op = by_id[key]
        contributors = frozenset(op.contributors)
        ranges = tuple(DataRange(s // n, (s % n) * size, (s % n + 1) * size) for s in contributors)
        # Idle, legal edges for this slice (never the congested resource).
        rows = set()
        for link in topology.links:
            if link.src_rank not in op.participants or link.dst_rank not in op.participants:
                continue
            edge_resources = ('link:%d:%d' % (link.src_rank, link.dst_rank),) + tuple(topology.links[link].resource_ids)
            if scope.resource in edge_resources:
                continue
            if any(_window_utilization(report, r, scope.buckets) >= threshold for r in edge_resources if r.startswith('link:')):
                continue
            if constraints.allows(ranges, link.src_rank, link.dst_rank, op.stage_id):
                rows.add((link.src_rank, link.dst_rank))
        edges = space.relation(('rank', 'rank2'), rows)
        # Bounded backward reachability to the destination over idle edges.
        reach = [space.eq('rank', op.dst_rank)]
        for _ in range(hops):
            step = space.exist(('rank2',), edges & space.rename(reach[-1], {'rank': 'rank2'}))
            reach.append(reach[-1] | step)
        # Owners whose copy is ready before this transfer: producer not after it
        # (Before relation of the symbolic execution, as a BDD).
        owners = {}
        for rank, producer in ownership.owners(contributors, n).items():
            if rank == op.dst_rank:
                continue
            if producer is not None:
                if producer == key or producer not in relations.ops or key not in relations.ops:
                    continue
                after = before & space.eq('op', relations.ops[key]) & space.eq('op2', relations.ops[producer])
                if not space.is_empty(after):
                    continue
            if space.is_empty(reach[-1] & space.eq('rank', rank)):
                continue
            owners[rank] = 0.0 if producer is None else times[producer][1]
        adjacency = defaultdict(list)
        for src, dst in space.tuples(edges, ('rank', 'rank2')):
            adjacency[src].append(dst)
        # A rank that owns (or will receive) the value is never a relay: the
        # value would arrive there twice, and forwarding directly from that
        # rank is the same plan without the duplicate.
        eventual = set(ownership.owners(contributors, n)) | {
            other.dst_rank for other in ops if frozenset(other.contributors) == contributors}
        current_arrival = times[key][1]
        for owner, ready in owners.items():
            # Enumerate simple paths owner -> dst inside the reachable set.
            stack = [(owner, (owner,))]
            while stack:
                lowering.lease.check()
                rank, path = stack.pop()
                if rank == op.dst_rank:
                    if path != (op.src_rank, op.dst_rank):
                        estimate = ready
                        for index, (a, b) in enumerate(zip(path, path[1:])):
                            kind = op.kind if index == len(path) - 2 else 'send'
                            u = min(.9, _window_utilization(report, 'link:%d:%d' % (a, b), scope.buckets))
                            estimate += _hop_service(lowering, a, b, kind) / (1.0 - u)
                        if estimate < current_arrival:
                            plans[key].append(Plan(key, path, owner, ready, estimate))
                    continue
                remaining = hops - (len(path) - 1)
                if remaining <= 0:
                    continue
                for nxt in adjacency[rank]:
                    if nxt in path or (nxt != op.dst_rank and nxt in eventual):
                        continue
                    if nxt != op.dst_rank and space.is_empty(reach[remaining - 1] & space.eq('rank', nxt)):
                        continue
                    stack.append((nxt, path + (nxt,)))
        plans[key].sort(key=lambda plan: (plan.estimate_us, len(plan.path), plan.path))
        del plans[key][settings.plans_per_slice:]
    flat = sorted((plan for items in plans.values() for plan in items), key=lambda plan: plan.estimate_us)
    keep = {(plan.operation_id, plan.path) for plan in flat[:settings.max_plans]}
    return {key: [plan for plan in items if (key, plan.path) in keep] for key, items in plans.items() if items}


# ---------------------------------------------------------------------------
# 4. atom constraints

def repair_atoms(scope, plans, ops, prediction, lowering):
    by_id = {op.operation_id: op for op in ops}
    size = lowering.inputs.hyperparameters.slice_size_bytes
    in_scope = set(scope.reroutable) | set(scope.consumers)
    consumers_of = defaultdict(set)
    for op in ops:
        for dep in op.dependencies:
            consumers_of[dep].add(op.operation_id)
    atoms = []
    for key in scope.reroutable:
        op = by_id[key]
        options = [(op.src_rank, op.dst_rank)] + [plan.path for plan in plans.get(key, ())]
        owners = [op.src_rank] + [plan.owner for plan in plans.get(key, ())]
        current_ready = max((prediction.end_times.get(dep, 0.0) for dep in op.dependencies), default=0.0)
        ready = [current_ready] + [plan.ready_us for plan in plans.get(key, ())]
        outside = consumers_of[key] - in_scope
        deadline = prediction.end_times[key] if outside else None
        atoms.append(RepairAtom(SliceVec(min(op.contributors), size, op.stage_id), key, tuple(options),
                                tuple(owners), tuple(ready), deadline))
    return atoms


# ---------------------------------------------------------------------------
# 5. local MILP

def solve_local(scope, atoms, ops, prediction, report, lowering, context):
    from vericcl.solver.scip_api import ScipAdapter
    from vericcl.solver.trees.operators import model_lease
    by_id = {op.operation_id: op for op in ops}
    horizon = prediction.t_pred_us * 2 + 1.0
    M = horizon
    with model_lease(lowering.inputs, context) as (lease, threads):
        module, model = ScipAdapter.create_model('vericcl-bottleneck-local')
        try:
            ScipAdapter.configure(model, seed=lowering.inputs.solver.solver_seed, threads=threads, gap=0, lease=lease)
            latest = model.addVar('latest_finish', lb=0, ub=horizon)
            choice, arrival, intervals, changed = {}, {}, [], []
            for atom in atoms:
                op = by_id[atom.operation_id]
                ys = [model.addVar('y-%s-%d' % (atom.operation_id, k), vtype='B') for k in range(len(atom.pd))]
                model.addCons(module.quicksum(ys) == 1)
                choice[atom.operation_id] = ys
                changed.extend(ys[1:])
                arrive = model.addVar('a-%s' % atom.operation_id, lb=0, ub=horizon)
                arrival[atom.operation_id] = arrive
                for k, path in enumerate(atom.pd):
                    previous = None
                    for index, (a, b) in enumerate(zip(path, path[1:])):
                        kind = op.kind if index == len(path) - 2 else 'send'
                        u = min(.9, _window_utilization(report, 'link:%d:%d' % (a, b), scope.buckets)) if k else 0.0
                        duration = _hop_service(lowering, a, b, kind) / (1.0 - u)
                        if k == 0:
                            # The current transfer keeps its predicted (contended) service time.
                            duration = max(duration, prediction.end_times[atom.operation_id] - prediction.start_times[atom.operation_id])
                        s = model.addVar('s-%s-%d-%d' % (atom.operation_id, k, index), lb=0, ub=horizon)
                        model.addCons(s >= atom.ready_us[k] if previous is None else s >= previous + 0)
                        e = s + duration
                        previous = e
                        resources = ('link:%d:%d' % (a, b),) + tuple(lowering.topology.links[LinkKey(a, b)].resource_ids)
                        intervals.append((s, duration, ys[k], resources))
                    model.addCons(arrive >= previous - M * (1 - ys[k]))
                if atom.deadline_us is not None:
                    model.addCons(arrive <= atom.deadline_us + 1e-6)
                model.addCons(latest >= arrive)
            for key in scope.consumers:
                op = by_id[key]
                s = model.addVar('s-%s' % key, lb=0, ub=horizon)
                duration = prediction.end_times[key] - prediction.start_times[key]
                for dep in op.dependencies:
                    model.addCons(s >= arrival[dep] if dep in arrival else s >= prediction.end_times.get(dep, 0.0))
                arrival[key] = s + duration
                model.addCons(latest >= s + duration)
                resources = ('link:%d:%d' % (op.src_rank, op.dst_rank),) + tuple(lowering.topology.links[LinkKey(op.src_rank, op.dst_rank)].resource_ids)
                intervals.append((s, duration, None, resources))
            # Disjunctive sharing inside the scope: two active intervals on one
            # resource do not overlap (bandwidth sharing ~ serialization).
            by_resource = defaultdict(list)
            for index, (_, _, _, resources) in enumerate(intervals):
                for resource in resources:
                    by_resource[resource].append(index)
            pairs = 0
            for resource, members in by_resource.items():
                for i in range(len(members)):
                    for j in range(i + 1, len(members)):
                        (si, di, yi, _), (sj, dj, yj, _) = intervals[members[i]], intervals[members[j]]
                        if yi is not None and yi is yj:
                            continue
                        z = model.addVar('z-%d-%d-%d' % (members[i], members[j], pairs), vtype='B')
                        pairs += 1
                        slack = (0 if yi is None else (1 - yi)) + (0 if yj is None else (1 - yj))
                        model.addCons(sj >= si + di - M * (1 - z) - M * slack)
                        model.addCons(si >= sj + dj - M * z - M * slack)
            step = 1e-3 * max(1.0, min((d for _, d, _, _ in intervals), default=1.0))
            model.setObjective(latest + step * module.quicksum(changed))
            result = model.optimize()
            if result.primal is None:
                return None, {'status': result.status, 'pairs': pairs}
            selected = {key: max(range(len(ys)), key=lambda k: result.solution[ys[k].name]) for key, ys in choice.items()}
            return selected, {'status': result.status, 'primal': result.primal, 'pairs': pairs,
                              'atoms': len(atoms), 'consumers': len(scope.consumers)}
        finally:
            model.freeProb()


# ---------------------------------------------------------------------------
# proposals

def _network(ops, choices, plans_by_path, lowering, aliases, owners_of):
    inputs = lowering.inputs
    result, extra = [], []
    for op in ops:
        path = choices.get(op.operation_id)
        if path is None or path == (op.src_rank, op.dst_rank):
            result.append(op)
            continue
        hops, new_aliases = route_ops(op, path, owners_of[op.operation_id], 'hot')
        result.extend(hops)
        extra.extend(new_aliases)
    return NetworkCandidate(inputs.rank_count, inputs.hyperparameters.slice_count,
                            inputs.hyperparameters.slice_size_bytes, tuple(result), 'bottleneck_repair',
                            tuple(aliases) + tuple(extra))


def bottleneck_proposals(bundle, strategy, report, context):
    from vericcl.symbolic.executor import execute_bundle
    from vericcl.verification.hotspots import HotspotSettings
    lowering = context.lowering
    parameters = dict(strategy.parameters)
    hotspot = next((item for item in report.hotspots if item.operation_id == parameters['hotspot']), None)
    if hotspot is None:
        return [], {'status': 'hotspot_not_in_report'}
    settings = BottleneckSettings.from_inputs(lowering.inputs)
    hotspot_settings = HotspotSettings.from_inputs(lowering.inputs)
    ops = _network_from_storage(bundle)
    from vericcl.symbolic_full.config import enabled as full_symbolic_enabled
    full_evidence = None
    if full_symbolic_enabled(lowering.inputs):
        # sketch.symbolic.enabled: related schedules from the full symbolic
        # execution, plans screened as BDD path conditions (symbolic_full/repair.py).
        from vericcl.symbolic_full.repair import scope_and_plans
        backend = lowering.inputs.request_config['sketch']['solver'].get('bdd_backend', 'cudd')
        scope, plans, full_evidence = scope_and_plans(hotspot, bundle, ops, report, lowering, settings,
                                                      hotspot_settings, backend=backend)
    else:
        execution = execute_bundle(bundle, lowering.inputs, lowering.model)
        scope = related_scope(hotspot, execution, ops, bundle.prediction, lowering.topology, settings)
        plans = repair_plans(scope, ops, execution, report, lowering, settings, hotspot_settings)
    evidence = {'scope': {'reroutable': list(scope.reroutable), 'consumers': list(scope.consumers),
                          **scope.evidence},
                'plans': {key: [list(plan.path) for plan in items] for key, items in plans.items()}}
    if full_evidence is not None:
        evidence['full_symbolic'] = full_evidence
    if not plans:
        return [], dict(evidence, status='no_plan')
    atoms = repair_atoms(scope, plans, ops, bundle.prediction, lowering)
    ownership = symbolic_ownership(ops, lowering.inputs)
    n = lowering.inputs.hyperparameters.slice_count
    owners_of = {key: ownership.owners(frozenset(op.contributors), n) for key, op in
                 ((k, next(o for o in ops if o.operation_id == k)) for k in scope.reroutable)}
    aliases = tuple(getattr(bundle.source, 'replaced_operations', ()) or ())
    proposals, seen = [], set()
    selected, solver = solve_local(scope, atoms, ops, bundle.prediction, report, lowering, context)
    evidence['local_milp'] = solver
    if selected:
        choices = {atom.operation_id: atom.pd[selected[atom.operation_id]] for atom in atoms
                   if selected[atom.operation_id] > 0}
        if choices:
            seen.add(tuple(sorted(choices.items())))
            proposals.append(({'kind': 'local_milp', 'paths': {k: list(v) for k, v in choices.items()}},
                              _network(ops, choices, plans, lowering, aliases, owners_of)))
    for plan in sorted((p for items in plans.values() for p in items), key=lambda p: p.estimate_us):
        key = ((plan.operation_id, plan.path),)
        if key in seen:
            continue
        seen.add(key)
        proposals.append(({'kind': 'single_plan', 'operation': plan.operation_id, 'path': list(plan.path),
                           'estimate_us': plan.estimate_us},
                          _network(ops, {plan.operation_id: plan.path}, plans, lowering, aliases, owners_of)))
        if len(proposals) >= max(2, context.candidate_batch_size):
            break
    return proposals, evidence


def bottleneck_strategies(report, bundle, lowering, skip=()):
    from vericcl.verification.bottlenecks import StrategyRecord
    storage_ids = {op.operation_id for op in bundle.storage_plan.instructions}
    for hotspot in report.hotspots:
        if hotspot.operation_id in skip or hotspot.operation_id not in storage_ids:
            continue
        domain = tuple(sorted(({hotspot.operation_id} | set(hotspot.related_operations)) & storage_ids))
        yield 'bottleneck', StrategyRecord('bottleneck', domain,
                                           parameters=(('hotspot', hotspot.operation_id),
                                                       ('resource', hotspot.congested_resource)),
                                           cause_ids=('hotspot:' + hotspot.operation_id,))


def solve_bottleneck(bundle, strategy, report, context):
    from .proposals import evaluate_proposals, prepare_lowering
    prepare_lowering(context)
    proposals, evidence = bottleneck_proposals(bundle, strategy, report, context)
    # Only rerouted transfers change; consumers keep their path and identity.
    rerouted = set(evidence.get('scope', {}).get('reroutable', ())) or set(strategy.operation_ids)
    domain = RepairDomain(frozenset(rerouted), 'bottleneck', strategy.cause_ids)
    result = evaluate_proposals(bundle, domain, proposals, context, label='bottleneck')
    from dataclasses import replace
    return replace(result, solver_evidence=result.solver_evidence + ({'bottleneck': evidence},))
