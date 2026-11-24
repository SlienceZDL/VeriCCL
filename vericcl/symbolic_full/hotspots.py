"""Scheduling hotspots by counters and BDD reasoning over the full symbolic execution.

A hotspot is a transfer that is held up - slowed while sharing a link or
resource, or kept waiting by threadblock order, dependency flags, a full
connection FIFO or a busy buffer slot - while another link that could carry
its data toward the destination is idle or relatively idle at the same time.

1. Counters, per (resource, time bucket): busy time, bytes, active transfers,
   idle time, utilization. Per transfer: resource wait (service time beyond
   its uncontended cost), order wait (start minus data readiness when the
   deciding wait is tb/dep/fifo/slot, read from the symbolic max terms).
2. BDD sets.
   Congested(op, res, bucket)   op is held up while on res in bucket
   Idle(res2, bucket)           utilization below the idle threshold
   Holder(op, rank)             ranks that hold op's exact contribution set
                                before op starts (symbolic ownership: once a
                                rank holds data it can forward it)
   Allowed_op(rank, rank2)      topology links, gateway rule, and no ban for
                                op's source bytes and stage
   Reach_op(rank)               ranks that reach op's destination over
                                Allowed_op within max_alt_hops - 1 hops
                                (backward images)
   Alt(op, res2)                res2 is the first link h->x of such a path,
                                h a holder, (h, x) not op's own link
   Hotspot = exists bucket: Congested & Alt & Idle
3. Related schedules: the transfers active on the congested resource in the
   hot buckets, the chain of order predecessors that decided the start, and
   the dependency cone between the hotspot and the completion tail (forward
   image fixpoint from the hotspot intersected with the backward fixpoint
   from the tail), capped at max_related_operations.
4. Ranking: (resource wait + order wait) x 2 on the critical path, then bytes.
"""
from __future__ import annotations

from collections import defaultdict

from vericcl.verification.hotspots import Hotspot, HotspotReport, HotspotSettings

from .checks import allowed_relation, forbidden_relation
from .config import settings as read_settings
from .space import FullSpace
from .view import critical_chain, operation_edges, operation_view


def _capacity(resource, model, slice_bytes):
    if resource in model.resources:
        return model.resources[resource]
    if resource.startswith('link:'):
        cost = model.service_for(resource, slice_bytes)
        if cost is not None and cost.beta_us_per_byte > 0:
            return 1.0 / cost.beta_us_per_byte
    return None


def holders(execution, view, op):
    """Ranks holding op's exact contribution set no later than op's start (and when)."""
    result = {}
    for rank, end, writer in execution.ownership().get(op.contributors, ()):
        ready = execution.value(end)
        if writer is not None and execution.nodes[writer].operation_id == op.operation_id:
            continue
        if rank != op.dst_rank and ready <= op.start + 1e-9:
            if rank not in result or ready < result[rank][0]:
                result[rank] = (ready, writer)
    return result


def locate(execution, bundle, inputs, topology, model, *, settings=None, backend='cudd'):
    settings = settings or HotspotSettings.from_inputs(inputs)
    full = read_settings(inputs)
    execution.bind(model, bundle.prediction)
    view = operation_view(execution, topology)
    network = {key: op for key, op in view.items() if op.kind != 'copy'}
    t_pred = max((op.end for op in view.values()), default=0.0)
    if t_pred <= 0 or not network:
        return HotspotReport((), {'t_pred_us': t_pred, 'mode': 'full'}, settings)
    width = t_pred / settings.bucket_count

    # -- counters ---------------------------------------------------------------
    resources = sorted({r for op in network.values() for r in op.resources}
                       | {'link:%d:%d' % (k.src_rank, k.dst_rank) for k in topology.links})
    res_index = {name: index for index, name in enumerate(resources)}
    busy, carried, active = defaultdict(float), defaultdict(float), defaultdict(set)
    buckets_of = defaultdict(set)
    for key, op in network.items():
        duration = max(op.end - op.start, 1e-12)
        rate = op.bytes / duration
        first, last = int(op.start // width), min(int(min(op.end, t_pred - 1e-9) // width), settings.bucket_count - 1)
        for bucket in range(first, last + 1):
            low, high = max(op.start, bucket * width), min(op.end, (bucket + 1) * width)
            if high <= low:
                continue
            buckets_of[key].add(bucket)
            for resource in op.resources:
                busy[resource, bucket] = min(width, busy[resource, bucket] + (high - low))
                carried[resource, bucket] += rate * (high - low)
                active[resource, bucket].add(key)
    utilization = {}
    for resource in resources:
        cap = _capacity(resource, model, execution.slice_bytes)
        for bucket in range(settings.bucket_count):
            utilization[resource, bucket] = 0.0 if not cap else carried[resource, bucket] / (cap * width)

    # -- waits from the symbolic max terms -------------------------------------
    resource_wait, order_wait, labels = {}, {}, {}
    for key, op in network.items():
        node = execution.nodes[op.node]
        cost = model.service_for(node.cost_key, op.bytes) if node.cost_key else None
        ideal = (cost.alpha_us + cost.beta_us_per_byte * op.bytes) if cost is not None else (op.end - op.start)
        resource_wait[key] = max(0.0, (op.end - op.start) - ideal)
        label, _ = execution.critical_predecessor(op.node)
        labels[key] = label
        order_wait[key] = max(0.0, op.start - op.data_ready) if label in {'tb', 'dep', 'fifo', 'slot'} else 0.0

    # -- BDD relations -------------------------------------------------------------
    keys = sorted(network)
    op_index = {key: index for index, key in enumerate(keys)}
    n = execution.n
    stages = execution.space.sizes['stage']
    s = FullSpace(((('op', len(keys)), ('op2', len(keys)), ('op3', len(keys))),
                   (('res', len(resources)), ('res2', len(resources))),
                   (('bucket', settings.bucket_count),),
                   (('slice', max(1, execution.ranks * n)),),
                   (('rank', execution.ranks), ('rank2', execution.ranks), ('rank3', execution.ranks)),
                   (('stage', stages),)), backend=backend)
    congested_rows = set()
    for key, op in network.items():
        ideal = max((op.end - op.start) - resource_wait[key], 1e-9)
        slowed = resource_wait[key] > settings.wait_threshold_ratio * ideal
        delayed = order_wait[key] > settings.wait_threshold_ratio * ideal
        for resource in op.resources:
            for bucket in buckets_of[key]:
                shared = len(active[resource, bucket]) > 1
                if (slowed and shared) or delayed:
                    congested_rows.add((op_index[key], res_index[resource], bucket))
    congested = s.relation(('op', 'res', 'bucket'), congested_rows)
    idle = s.relation(('res2', 'bucket'), {(res_index[r], b) for (r, b), value in utilization.items()
                                           if r.startswith('link:') and value < settings.idle_utilization_threshold})
    allowed = allowed_relation(s, topology)
    forbidden = forbidden_relation(s, inputs, n, execution.slice_bytes)
    link_res = s.relation(('rank', 'rank2', 'res2'), {(k.src_rank, k.dst_rank, res_index['link:%d:%d' % (k.src_rank, k.dst_rank)])
                                                      for k in topology.links})
    alt = s.false
    holder_table = {}
    for key in keys:
        op = network[key]
        if not op.contributors:
            continue
        owned = holders(execution, view, op)
        holder_table[key] = owned
        if not owned:
            continue
        banned = s.exist(('slice', 'stage'), forbidden & s.set_of('slice', sorted(op.contributors)) & s.eq('stage', op.stage))
        edges = allowed & ~banned
        reach = s.eq('rank', op.dst_rank)
        for _ in range(max(0, settings.max_alt_hops - 1)):
            reach = reach | s.preimage(reach, edges, present=('rank',), nxt=('rank2',))
        firsts = (s.set_of('rank', sorted(owned)) & edges & s.rename(reach, {'rank': 'rank2'})
                  & ~(s.eq('rank', op.src_rank) & s.eq('rank2', op.dst_rank)))
        alt |= s.eq('op', op_index[key]) & s.exist(('rank', 'rank2'), firsts & link_res)
    hot = congested & alt & idle                                    # (op, res, res2, bucket)
    hot_ops = s.values(s.exist(('res', 'res2', 'bucket'), hot), 'op')

    # -- related schedules and ranking ----------------------------------------------
    tail_key = max(network, key=lambda key: (network[key].end, key))
    critical_nodes = set(critical_chain(execution, network[tail_key].node))
    critical = {execution.nodes[i].operation_id for i in critical_nodes if execution.nodes[i].operation_id}
    edges = s.relation(('op', 'op2'), {(op_index[a], op_index[b]) for a, b in operation_edges(execution)
                                       if a in op_index and b in op_index})
    upstream_of_tail = s.eq('op', op_index[tail_key])
    while True:
        grown = upstream_of_tail | s.preimage(upstream_of_tail, edges, present=('op',), nxt=('op2',))
        if grown == upstream_of_tail:
            break
        upstream_of_tail = grown
    hotspots = []
    for value in hot_ops:
        key = keys[value]
        op = network[key]
        rows = s.tuples(hot & s.eq('op', value), ('res', 'res2', 'bucket'))
        resource = max({resources[r] for r, _, _ in rows},
                       key=lambda r: (max(len(active[r, b]) for _, _, b in rows), r))
        hot_buckets = sorted({b for _, _, b in rows})
        alternatives = sorted({tuple(int(v) for v in resources[r2].split(':')[1:]) for _, r2, _ in rows})
        competitors = {other for b in hot_buckets for other in active[resource, b]} - {key}
        for index in critical_chain(execution, op.node, limit=64)[1:]:
            label, _ = execution.critical_predecessor(index)
            other = execution.nodes[index].operation_id
            if other and other != key:
                competitors.add(other)
            if label not in {'tb', 'dep', 'fifo', 'slot'}:
                break
        downstream = s.eq('op', value)
        while True:
            grown = downstream | s.image(downstream, edges, present=('op',), nxt=('op2',))
            if grown == downstream:
                break
            downstream = grown
        cone = downstream & upstream_of_tail
        cone_ops = [keys[v] for v in s.values(cone, 'op') if keys[v] != key]
        related = sorted(competitors)[:full.max_related_operations]
        room = full.max_related_operations - len(related)
        related += [other for other in cone_ops if other not in related][:max(0, room)]
        on_critical = key in critical
        score = (resource_wait[key] + order_wait[key]) * (2.0 if on_critical else 1.0)
        affects_tail = on_critical or any(other in critical for other in competitors)
        hotspots.append(Hotspot(key, resource, op.src_rank, op.dst_rank, tuple(alternatives), tuple(hot_buckets),
                                resource_wait[key], order_wait[key], on_critical, tuple(related),
                                len(s.values(downstream, 'op')) - 1, score,
                                {'evidence_level': 'full_symbolic_execution_bdd_counters',
                                 'deciding_wait': labels[key], 'data_ready_us': op.data_ready,
                                 'holders': sorted(holder_table.get(key, {})),
                                 'peak_utilization': max(utilization[resource, b] for b in hot_buckets),
                                 'alternative_utilization': {'link:%d:%d' % link: min(utilization['link:%d:%d' % link, b]
                                                                                      for b in hot_buckets)
                                                             for link in alternatives},
                                 'tail_cone_operations': len(cone_ops)},
                                affects_tail))
    located = len(hotspots)
    if settings.tail_only:
        hotspots = [item for item in hotspots if item.affects_tail]
    hotspots.sort(key=lambda h: (-h.score, -network[h.operation_id].bytes, h.operation_id))
    busy_links = [value for (r, _), value in utilization.items() if r.startswith('link:')]
    counters = {'mode': 'full', 't_pred_us': t_pred, 'bucket_width_us': width,
                'congested_operations': len({row[0] for row in congested_rows}),
                'hotspot_operations': located, 'tail_hotspot_operations': sum(h.affects_tail for h in hotspots),
                'max_link_utilization': max(busy_links, default=0.0),
                'idle_link_buckets': sum(value < settings.idle_utilization_threshold for value in busy_links),
                'resource_wait_us': sum(resource_wait.values()), 'order_wait_us': sum(order_wait.values()),
                'critical_path_length': len(critical), 'bdd_nodes': s.node_count()}
    return HotspotReport(tuple(hotspots[:settings.max_hotspots]), counters, settings, dict(utilization))
