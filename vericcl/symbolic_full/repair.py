"""Related-schedule scope and repair-plan screening for scheduling-bottleneck repair.

1. Related schedules (symbolic execution). The transfers on the congested
   resource that overlap the hotspot in time, the chain of order predecessors
   that decided the hotspot's start (critical branches of the symbolic start
   terms: threadblock order, dependency flags, FIFO slots, buffer slots), and
   their data consumers up to ``dependency_depth`` (images over the op-level
   wait relation). The scope is capped, so it never grows to the topology.
2. Candidate reroutes. For each transfer in the scope, paths start at a rank
   that already holds the transfer's exact contribution set before it starts
   (symbolic ownership), run over links that are allowed for its source bytes,
   legal for relays and idle in the hot buckets, and reach the destination
   within ``max_alt_hops`` (backward BDD images). A holder whose copy is
   produced downstream of the transfer is excluded (it would create a cycle).
3. Decisions as BDD path conditions. Each transfer in the scope gets a
   decision variable d_op (0 keeps its path, k picks candidate k). The guarded
   transfer relation T_g(slice, rank, rank2, stage, d...) is the union of each
   candidate's hops under d_op == k. Bad decisions are the projection of
   T_g & (Forbidden | ~Allowed) onto the decision variables; conflicting
   combinations are pairs of candidates of different transfers that share a
   link. Legal = ~Bad & ~Conflict. A candidate survives when some legal
   assignment selects it and its estimated arrival (holder ready time plus hop
   costs, slowed by the link utilization in the hot window) beats the current
   one. The surviving plans go to the unchanged atom constraints and local MILP.
"""
from __future__ import annotations

from collections import defaultdict

from .cache import execution_for
from .checks import allowed_relation, forbidden_relation
from .config import settings as read_settings
from .hotspots import holders
from .space import FullSpace
from .view import critical_chain, operation_edges, operation_view


def scope_and_plans(hotspot, bundle, ops, report, lowering, settings, hotspot_settings, *, backend='cudd'):
    from vericcl.tuning.bottleneck import BottleneckScope, Plan, _hop_service, _window_utilization
    inputs, topology = lowering.inputs, lowering.topology
    full = read_settings(inputs)
    execution = execution_for(bundle, inputs, backend=backend, lease=lowering.lease)
    execution.bind(lowering.model, bundle.prediction)
    view = operation_view(execution, topology)
    by_id = {op.operation_id: op for op in ops}
    if hotspot.operation_id not in view or hotspot.operation_id not in by_id:
        return BottleneckScope(hotspot.operation_id, hotspot.congested_resource, tuple(hotspot.buckets), [], [],
                               {'mode': 'full', 'status': 'hotspot_not_executed'}), {}, {'mode': 'full'}
    hot = view[hotspot.operation_id]

    # -- 1. related schedules ---------------------------------------------------------
    resource = hotspot.congested_resource
    overlap = lambda key: max(0.0, min(hot.end, view[key].end) - max(hot.start, view[key].start))
    competitors = [key for key in set(hotspot.related_operations) | {hotspot.operation_id}
                   if key in view and key in by_id and resource in view[key].resources]
    reroutable = sorted(competitors, key=lambda key: (key != hotspot.operation_id, -overlap(key), key))
    chain = []
    for index in critical_chain(execution, hot.node, limit=64)[1:]:
        label, _ = execution.critical_predecessor(index)
        key = execution.nodes[index].operation_id
        if key and key not in chain and key != hotspot.operation_id:
            chain.append(key)
            if key in by_id and key in view and resource in view[key].resources and key not in reroutable:
                reroutable.append(key)
        if label not in {'tb', 'dep', 'fifo', 'slot'}:
            break
    reroutable = reroutable[:max(1, settings.max_scope_operations // 2)]
    edges = operation_edges(execution)
    consumers_of = defaultdict(set)
    for source, target in edges:
        consumers_of[source].add(target)
    consumers, frontier = [], list(reroutable)
    for _ in range(settings.dependency_depth):
        nxt = []
        for key in frontier:
            for consumer in sorted(consumers_of[key]):
                if consumer in by_id and consumer not in reroutable and consumer not in consumers:
                    consumers.append(consumer)
                    nxt.append(consumer)
        frontier = nxt
    consumers = consumers[:max(0, settings.max_scope_operations - len(reroutable))]
    scope = BottleneckScope(hotspot.operation_id, resource, tuple(hotspot.buckets), reroutable, consumers,
                            {'mode': 'full', 'order_chain': chain})

    # -- 2. candidate reroutes ----------------------------------------------------------
    keys = sorted(view)
    op_index = {key: index for index, key in enumerate(keys)}
    n, size = execution.n, execution.slice_bytes
    s = FullSpace(((('op', len(keys)), ('op2', len(keys))),
                   (('slice', max(1, execution.ranks * n)),),
                   (('rank', execution.ranks), ('rank2', execution.ranks)),
                   (('stage', execution.space.sizes['stage']),)), backend=backend)
    wait = s.relation(('op', 'op2'), {(op_index[a], op_index[b]) for a, b in edges if a in op_index and b in op_index})
    allowed = allowed_relation(s, topology)
    forbidden = forbidden_relation(s, inputs, n, size)
    threshold = hotspot_settings.idle_utilization_threshold
    hops = hotspot_settings.max_alt_hops
    eventual_by_set = defaultdict(set)
    for contributors, owners in execution.ownership().items():
        eventual_by_set[contributors].update(rank for rank, _, _ in owners)
    candidates = {}
    for key in reroutable:
        op, network_op = view[key], by_id[key]
        downstream = s.eq('op', op_index[key])
        while True:
            grown = downstream | s.image(downstream, wait, present=('op',), nxt=('op2',))
            if grown == downstream:
                break
            downstream = grown
        later = {keys[v] for v in s.values(downstream, 'op')}
        owned = {rank: value for rank, value in holders(execution, view, op).items()
                 if value[1] is None or execution.nodes[value[1]].operation_id not in later}
        banned = s.exist(('slice', 'stage'), forbidden & s.set_of('slice', sorted(op.contributors)) & s.eq('stage', op.stage))
        rows = set()
        for link in topology.links:
            if link.src_rank not in network_op.participants or link.dst_rank not in network_op.participants:
                continue
            names = ('link:%d:%d' % (link.src_rank, link.dst_rank),) + tuple(topology.links[link].resource_ids)
            if resource in names or any(_window_utilization(report, r, scope.buckets) >= threshold
                                        for r in names if r.startswith('link:')):
                continue
            rows.add((link.src_rank, link.dst_rank))
        idle_edges = s.relation(('rank', 'rank2'), rows) & allowed & ~banned
        reach = [s.eq('rank', op.dst_rank)]
        for _ in range(hops):
            reach.append(reach[-1] | s.preimage(reach[-1], idle_edges, present=('rank',), nxt=('rank2',)))
        adjacency = defaultdict(list)
        for a, b in s.tuples(idle_edges, ('rank', 'rank2')):
            adjacency[a].append(b)
        eventual = eventual_by_set.get(op.contributors, set()) | {op.dst_rank}
        found = []
        for owner, (ready, _) in sorted(owned.items()):
            if s.is_empty(reach[-1] & s.eq('rank', owner)):
                continue
            stack = [(owner, (owner,))]
            while stack:
                lowering.lease.check()
                rank, path = stack.pop()
                if rank == op.dst_rank:
                    if path != (op.src_rank, op.dst_rank):
                        estimate = ready
                        for position, (a, b) in enumerate(zip(path, path[1:])):
                            kind = network_op.kind if position == len(path) - 2 else 'send'
                            u = min(.9, _window_utilization(report, 'link:%d:%d' % (a, b), scope.buckets))
                            estimate += _hop_service(lowering, a, b, kind) / (1.0 - u)
                        if estimate < op.end:
                            found.append(Plan(key, path, owner, ready, estimate))
                    continue
                remaining = hops - (len(path) - 1)
                if remaining <= 0:
                    continue
                for nxt in adjacency[rank]:
                    if nxt in path or (nxt != op.dst_rank and nxt in eventual):
                        continue
                    if nxt != op.dst_rank and s.is_empty(reach[remaining - 1] & s.eq('rank', nxt)):
                        continue
                    stack.append((nxt, path + (nxt,)))
        found.sort(key=lambda plan: (plan.estimate_us, len(plan.path), plan.path))
        if found:
            candidates[key] = found[:full.max_plan_choices]

    # -- 3. decisions as BDD path conditions ------------------------------------------------
    decision = {}
    for key, found in candidates.items():
        name = 'd%d' % len(decision)
        s.add_field(name, len(found) + 1)
        decision[key] = name
    guarded = s.false
    uses = defaultdict(list)
    for key, found in candidates.items():
        op = view[key]
        contrib = s.set_of('slice', sorted(op.contributors))
        for k, plan in enumerate(found, start=1):
            hops_rel = s.relation(('rank', 'rank2'), zip(plan.path, plan.path[1:]))
            guarded |= s.eq(decision[key], k) & contrib & hops_rel & s.eq('stage', op.stage)
            for link in zip(plan.path, plan.path[1:]):
                uses[link].append((key, k))
    decisions = list(decision.values())
    domain = s.true
    for name in decisions:
        domain &= s.domain(name)
    bad = s.exist(('slice', 'rank', 'rank2', 'stage'), guarded & (forbidden | ~allowed))
    conflict = s.false
    for link, users in uses.items():
        for a in range(len(users)):
            for b in range(a + 1, len(users)):
                (ka, ia), (kb, ib) = users[a], users[b]
                if ka != kb:
                    conflict |= s.eq(decision[ka], ia) & s.eq(decision[kb], ib)
    legal = domain & ~bad & ~conflict
    plans, rejected = {}, 0
    for key, found in candidates.items():
        others = [name for name in decisions if name != decision[key]]
        selectable = s.values(s.exist(others, legal), decision[key])
        kept = [plan for k, plan in enumerate(found, start=1) if k in selectable]
        rejected += len(found) - len(kept)
        if kept:
            plans[key] = kept[:settings.plans_per_slice]
    flat = sorted((plan for items in plans.values() for plan in items), key=lambda plan: plan.estimate_us)
    keep = {(plan.operation_id, plan.path) for plan in flat[:settings.max_plans]}
    plans = {key: [plan for plan in items if (key, plan.path) in keep] for key, items in plans.items()}
    plans = {key: items for key, items in plans.items() if items}
    evidence = {'mode': 'full', 'candidates': {key: len(found) for key, found in candidates.items()},
                'decision_variables': len(decisions),
                'legal_assignments': s.count(legal, decisions) if decisions else 0,
                'rejected_by_bdd': rejected, 'order_chain': chain}
    return scope, plans, evidence
