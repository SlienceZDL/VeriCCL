"""Scheduling-hotspot localization with symbolic execution, BDDs and counters.

A scheduling hotspot is an operation that is congested waiting for other
schedules - it shares a link or resource with concurrent transfers (contention
slowdown) or waits behind other work on its threadblock - while some other
link that could carry it towards its destination is idle or relatively idle
at the same time.

Counters (from the symbolic execution bound to the current model and the
simulator's rate segments):
  per (resource, time bucket): busy time, bytes, concurrent operations,
                               utilization = bytes / (capacity * width)
  per operation:               contention_us (slowdown above ideal service),
                               tb_wait_us / order_wait_us (from the deciding
                               branch of its symbolic start time)

BDD relations over (op, link, link2, bucket, rank, rank2):
  Occ(op, link, bucket)           op transfers on link during bucket
  Congested(op, link, bucket)     Occ & link shared & op slowed or delayed
  Idle(link2, bucket)             utilization(link2, bucket) < idle threshold
  Alt(op, link2)                  link2 is the first hop of a legal path from
                                  op's source to its destination within
                                  max_alt_hops (bounded backward reachability
                                  over the topology relation), avoiding link
  Hotspot(op, link, link2) = exists bucket. Congested & Alt & Idle
Related schedules of a hotspot are the operations that occupy the congested
link in the same buckets plus the operations downstream of it
(Before(op, x) from the dependency closure).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Tuple

from vericcl.symbolic.bdd import Index
from vericcl.symbolic.executor import SymbolicExecution
from vericcl.verification.symbolic_check import ScheduleRelations
from vericcl.topology.model import LinkKey


@dataclass(frozen=True)
class HotspotSettings:
    bucket_count: int = 64
    wait_threshold_ratio: float = 0.05
    idle_utilization_threshold: float = 0.5
    max_alt_hops: int = 3
    max_hotspots: int = 16
    # Only hotspots that can move the completion tail produce repairs: the
    # operation is on the critical path, or it competes for the congested
    # resource with an operation that is.
    tail_only: bool = True

    @classmethod
    def from_inputs(cls, inputs):
        config = (inputs.request_config or {}).get('sketch', {}).get('repair', {}) or {}
        values = dict(config.get('hotspot') or {})
        return cls(**{key: values[key] for key in cls.__dataclass_fields__ if values.get(key) is not None})


@dataclass(frozen=True)
class Hotspot:
    operation_id: str
    congested_resource: str
    src_rank: int
    dst_rank: int
    alternative_links: Tuple[Tuple[int, int], ...]
    buckets: Tuple[int, ...]
    contention_us: float
    wait_us: float
    on_critical_path: bool
    related_operations: Tuple[str, ...]      # competitors on the congested resource
    downstream_operations: int               # size of the dependency cone
    score: float
    evidence: Dict[str, object] = field(default_factory=dict)
    affects_tail: bool = True


@dataclass(frozen=True)
class HotspotReport:
    hotspots: Tuple[Hotspot, ...]
    counters: Dict[str, object]
    settings: HotspotSettings
    # (resource, bucket) -> utilization, for repair triggers that look for idle paths.
    utilization: Dict[Tuple[str, int], float] = field(default_factory=dict)


def _resources_of(execution: SymbolicExecution, key):
    record = execution.operations[key]
    if record.kind == 'copy':
        return ()
    link = 'link:%d:%d' % (record.src_rank, record.dst_rank)
    return tuple(dict.fromkeys((link,) + tuple(record.resources)))


def locate_hotspots(execution: SymbolicExecution, bundle, inputs, topology, model, *,
                    settings: HotspotSettings | None = None, relations: ScheduleRelations | None = None) -> HotspotReport:
    settings = settings or HotspotSettings.from_inputs(inputs)
    prediction = bundle.prediction
    rel = relations or ScheduleRelations(execution, inputs)
    s = rel.space
    times = execution.times()
    t_pred = max((end for _, end in times.values()), default=0.0)
    if t_pred <= 0:
        return HotspotReport((), {'t_pred_us': t_pred}, settings)
    width = t_pred / settings.bucket_count

    # ---- counters --------------------------------------------------------------
    resources = Index(sorted({r for key in execution.order for r in _resources_of(execution, key)}
                             | {'link:%d:%d' % (k.src_rank, k.dst_rank) for k in topology.links}))
    bytes_rb = defaultdict(float)              # (resource, bucket) -> bytes
    active_rb = defaultdict(set)               # (resource, bucket) -> ops
    op_buckets = defaultdict(set)
    for segment in prediction.segments:
        if segment.operation_id not in execution.operations:
            continue
        first, last = int(segment.start_us // width), int(min(segment.end_us, t_pred - 1e-9) // width)
        for bucket in range(first, min(last, settings.bucket_count - 1) + 1):
            low, high = max(segment.start_us, bucket * width), min(segment.end_us, (bucket + 1) * width)
            if high <= low:
                continue
            op_buckets[segment.operation_id].add(bucket)
            for resource in _resources_of(execution, segment.operation_id):
                bytes_rb[resource, bucket] += segment.rate_bytes_per_us * (high - low)
                active_rb[resource, bucket].add(segment.operation_id)

    def capacity(resource):
        if resource in model.resources:
            return model.resources[resource]
        if resource.startswith('link:'):
            _, src, dst = resource.split(':')
            cost = model.service_for(resource, inputs.hyperparameters.slice_size_bytes)
            return 1.0 / cost.beta_us_per_byte if cost else None
        return None

    utilization = {}
    for resource in resources.items:
        cap = capacity(resource)
        for bucket in range(settings.bucket_count):
            utilization[resource, bucket] = 0.0 if not cap else bytes_rb[resource, bucket] / (cap * width)

    contention, tb_wait = {}, {}
    for key in execution.order:
        record = execution.operations[key]
        start, end = times[key]
        if record.cost_key is not None:
            ideal = model.service_for(record.cost_key, record.payload_bytes)
            ideal_us = ideal.alpha_us + ideal.beta_us_per_byte * record.payload_bytes
            contention[key] = max(0.0, (end - start) - ideal_us)
        else:
            contention[key] = 0.0
        label, predecessor = execution.critical_predecessor(key)
        if label in {'tb', 'order'} and predecessor is not None:
            data_ready = max((times[p][1] for p, kind in record.dependencies.items() if kind == 'data'), default=0.0)
            tb_wait[key] = max(0.0, start - data_ready)
        else:
            tb_wait[key] = 0.0

    # ---- BDD relations ---------------------------------------------------------------
    s.add_field('res', len(resources))
    s.add_field('res2', len(resources))
    s.add_field('bucket', settings.bucket_count)
    occ_rows, congested_rows = set(), set()
    for key in execution.order:
        if key not in op_buckets:
            continue
        record = execution.operations[key]
        ideal_us = (times[key][1] - times[key][0]) - contention[key]
        slowed = contention[key] > settings.wait_threshold_ratio * max(ideal_us, 1e-9)
        delayed = tb_wait[key] > settings.wait_threshold_ratio * max(ideal_us, 1e-9)
        for resource in _resources_of(execution, key):
            for bucket in op_buckets[key]:
                row = (rel.ops[key], resources[resource], bucket)
                occ_rows.add(row)
                shared = len(active_rb[resource, bucket]) > 1
                if (slowed and shared) or delayed:
                    congested_rows.add(row)
    fields = ('op', 'res', 'bucket')
    occ = s.relation(fields, occ_rows)
    congested = s.relation(fields, congested_rows)
    idle = s.relation(('res2', 'bucket'), {
        (resources[r], b) for (r, b), value in utilization.items()
        if r.startswith('link:') and value < settings.idle_utilization_threshold})

    # Topology relation E(rank, rank2) and bounded backward reachability to each destination.
    edges = s.relation(('rank', 'rank2'), {(k.src_rank, k.dst_rank) for k in topology.links})
    reach_cache = {}

    def reaches(dst, hops):
        key = (dst, hops)
        if key not in reach_cache:
            node = s.eq('rank', dst)
            for _ in range(hops):
                # predecessors: {rank | exists rank2: E(rank, rank2) & node(rank2)}
                node = node | s.exist(('rank2',), edges & s.rename(node, {'rank': 'rank2'}))
            reach_cache[key] = node
        return reach_cache[key]

    alt_rows = set()
    for key in execution.order:
        record = execution.operations[key]
        if record.kind == 'copy' or key not in op_buckets:
            continue
        # first hops (src -> x), x != dst, from which dst is reachable in <= max_alt_hops - 1 hops
        targets = reaches(record.dst_rank, settings.max_alt_hops - 1) & ~s.eq('rank', record.dst_rank)
        first = s.exist(('rank',), (edges & s.eq('rank', record.src_rank)) & s.rename(targets, {'rank': 'rank2'}))
        for (x,) in s.tuples(first, ('rank2',)):
            link = 'link:%d:%d' % (record.src_rank, x)
            if link in resources:
                alt_rows.add((rel.ops[key], resources[link]))
    alt = s.relation(('op', 'res2'), alt_rows)
    hotspot = s.exist(('bucket',), congested & alt & idle)            # (op, res, res2)
    hot_buckets = congested & alt & idle                              # with bucket

    # ---- ranking and related schedules ------------------------------------------------
    critical = set(execution.critical_path())
    grouped = defaultdict(lambda: {'alternatives': set(), 'buckets': set(), 'resources': set()})
    for op_value, res_value, res2_value, bucket in s.tuples(hot_buckets, ('op', 'res', 'res2', 'bucket')):
        entry = grouped[op_value]
        entry['alternatives'].add(resources.items[res2_value])
        entry['buckets'].add(bucket)
        entry['resources'].add(resources.items[res_value])
    hotspots = []
    for op_value, entry in grouped.items():
        key = rel.op_name(op_value)
        record = execution.operations[key]
        resource = max(entry['resources'], key=lambda r: (max(len(active_rb[r, b]) for b in entry['buckets']), r))
        competitors = {other for b in entry['buckets'] for other in active_rb[resource, b]} - {key}
        # Schedules it is waiting for: the chain of threadblock/order
        # predecessors that decided its start (from the symbolic max terms).
        current = key
        while True:
            label, predecessor = execution.critical_predecessor(current)
            if label not in {'tb', 'order'} or predecessor is None:
                break
            competitors.add(predecessor)
            current = predecessor
        competitors = sorted(competitors)
        downstream = s.exist(('op',), rel.before & s.eq('op', op_value))
        cone = len(s.tuples(downstream, ('op2',)))
        score = (contention[key] + tb_wait[key]) * (2.0 if key in critical else 1.0)
        alternatives = tuple(sorted(tuple(int(v) for v in link.split(':')[1:]) for link in entry['alternatives']))
        hotspots.append(Hotspot(key, resource, record.src_rank, record.dst_rank, alternatives,
                                tuple(sorted(entry['buckets'])), contention[key], tb_wait[key], key in critical,
                                tuple(competitors), cone, score,
                                {'peak_utilization': max(utilization[resource, b] for b in entry['buckets']),
                                 'alternative_utilization': {link: min(utilization[link, b] for b in entry['buckets'])
                                                              for link in sorted(entry['alternatives'])}},
                                key in critical or any(other in critical for other in competitors)))
    located = len(hotspots)
    if settings.tail_only:
        hotspots = [hotspot for hotspot in hotspots if hotspot.affects_tail]
    hotspots.sort(key=lambda h: (-h.score, -h.downstream_operations, h.operation_id))
    counters = {
        't_pred_us': t_pred, 'bucket_width_us': width,
        'congested_operations': len({row[0] for row in congested_rows}),
        'hotspot_operations': located,
        'tail_hotspot_operations': sum(hotspot.affects_tail for hotspot in hotspots),
        'max_link_utilization': max((v for (r, _), v in utilization.items() if r.startswith('link:')), default=0.0),
        'critical_path_length': len(critical),
    }
    return HotspotReport(tuple(hotspots[:settings.max_hotspots]), counters, settings, dict(utilization))


def _shortest(topology, start, goal, allowed, banned):
    """BFS path start -> goal inside `allowed` ranks, never visiting `banned`."""
    previous, frontier = {start: None}, [start]
    while frontier:
        nxt = []
        for rank in frontier:
            for key in topology.links:
                if key.src_rank == rank and key.dst_rank in allowed and key.dst_rank not in previous and key.dst_rank not in banned:
                    previous[key.dst_rank] = rank
                    nxt.append(key.dst_rank)
        frontier = nxt
        if goal in previous:
            path = [goal]
            while previous[path[-1]] is not None:
                path.append(previous[path[-1]])
            return tuple(reversed(path))
    return None


def hotspot_strategies(report: HotspotReport, bundle, lowering):
    """Repair strategies derived from hotspots, most promising first.

    A hotspot whose chunk also has a flow over idle links is balanced by
    re-dividing the chunk's slices among its flows (repair type 2,
    ``reallocation``). Every other hotspot gets a scheduling-bottleneck repair
    (repair type 3, ``bottleneck``): related schedules, owner reroute plans over
    idle links, BDD/symbolic screening and a local MILP.
    """
    from vericcl.tuning.allocation import reallocation_strategies
    from vericcl.tuning.bottleneck import bottleneck_strategies
    covered = set()
    for family, record in reallocation_strategies(report, bundle, lowering):
        covered.add(dict(record.parameters)['hotspot'])
        yield family, record
    yield from bottleneck_strategies(report, bundle, lowering, skip=covered)
