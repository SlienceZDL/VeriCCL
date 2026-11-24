"""Cross-path data-volume allocation (repair type 2).

A chunk is one source's data for one routing demand group (the group the
initial synthesis also allocated over: same mode, source, destinations and
byte bans). It is split into slices (logical positions), and every slice of
the chunk travels from the source to the destinations along one route. Slices
of a chunk that use the same route form one flow, so a chunk can reach the
same destination through several flows.

The completion time is the latest finish over all flows, so it is shortest
when the flows (of this chunk and of the other chunks sharing their links)
finish at about the same time. When a scheduling hotspot sits on one flow of a
chunk while another flow of the same chunk runs over idle links, the chunk's
slices are re-divided among its existing flows, e.g. 8/1 -> 6/3.

Only existing slices move, and only onto routes the chunk already uses; the
slice geometry and every route stay as synthesized. The new counts come from a
small SCIP model that minimizes the largest resource load (fixed load of all
other work plus the scope's flows under the current cost model); a few
incremental moves from the hot flow to the idle flow are proposed as well.
All proposals are lowered, checked and compared with the incumbent on the
complete predicted time.
"""
from collections import defaultdict
from dataclasses import dataclass, replace

from vericcl.errors import SemanticError
from vericcl.semantics.ranges import DataRange, bind_constraints
from vericcl.topology.model import LinkKey
from vericcl.xml.v5_model import NetworkCandidate, NetworkOperation
from .correctness import _network_from_storage, _without
from .strategies import RepairDomain


@dataclass(frozen=True)
class Unit:
    demand_id: str
    chunk: str
    position: int
    operations: tuple
    shape: frozenset


@dataclass(frozen=True)
class ReallocationSettings:
    max_chunks: int = 8
    max_incremental_moves: int = 4

    @classmethod
    def from_inputs(cls, inputs):
        config = (inputs.request_config or {}).get('sketch', {}).get('repair', {}) or {}
        values = dict(config.get('reallocation') or {})
        return cls(**{key: values[key] for key in cls.__dataclass_fields__ if values.get(key) is not None})


def _shape(ops):
    return frozenset((op.src_rank, op.dst_rank, op.kind, op.stage_id) for op in ops)


def units_of(ops, inputs):
    """Group transfers by routing demand (one logical slice of one chunk)."""
    from vericcl.solver.trees.operators import routing_demands, _group_key
    n = inputs.hyperparameters.slice_count
    demands = routing_demands(inputs)
    by_position = defaultdict(list)
    for demand in demands:
        by_position[demand.logical_position].append(demand)
    chunk_names, chunk_of = {}, {}
    for demand in demands:
        key = repr(_group_key(inputs, demand))
        chunk_names.setdefault(key, 'chunk%d' % len(chunk_names))
        chunk_of[demand.demand_id] = chunk_names[key]
    grouped = defaultdict(list)
    op_unit = {}
    for op in ops:
        contributors = frozenset(op.contributors)
        position = min(contributors) % n
        demand = next((d for d in by_position[position] if contributors <= d.contributors), None)
        if demand is None:
            raise SemanticError('transfer belongs to no routing demand: ' + op.operation_id)
        grouped[demand.demand_id].append(op)
        op_unit[op.operation_id] = demand.demand_id
    positions = {d.demand_id: d.logical_position for d in demands}
    units = {key: Unit(key, chunk_of[key], positions[key], tuple(o.operation_id for o in sorted(items, key=lambda o: (o.priority_time, o.operation_id))),
                       _shape(items)) for key, items in grouped.items()}
    return units, op_unit


def chunk_flows(units):
    """chunk -> {route shape: [unit ids]}."""
    result = defaultdict(lambda: defaultdict(list))
    for unit in units.values():
        result[unit.chunk][unit.shape].append(unit.demand_id)
    return {chunk: dict(flows) for chunk, flows in result.items()}


def _resources(op, topology):
    edge = topology.links[LinkKey(op.src_rank, op.dst_rank)]
    return ('link:%d:%d' % (op.src_rank, op.dst_rank),) + tuple(edge.resource_ids)


def _load(op, lowering):
    """Per-resource busy time (us) of one transfer under the current model."""
    size = lowering.inputs.hyperparameters.slice_size_bytes
    model, topology = lowering.model, lowering.topology
    result = {}
    for resource in _resources(op, topology):
        if resource.startswith('link:'):
            prefix = 'rrc:' if op.kind == 'rrc' else 'link:'
            cost = model.service_for(prefix + resource[5:], size) or model.service_for(resource, size)
            if cost is not None:
                result[resource] = cost.alpha_us + cost.beta_us_per_byte * size
        elif resource in model.resources:
            result[resource] = size / model.resources[resource]
    return result


def _legal(ops, position, lowering):
    inputs = lowering.inputs
    n, size = inputs.hyperparameters.slice_count, inputs.hyperparameters.slice_size_bytes
    constraints = bind_constraints(inputs)
    for op in ops:
        moved = {(s // n) * n + position for s in op.contributors}
        ranges = tuple(DataRange(s // n, (s % n) * size, (s % n + 1) * size) for s in moved)
        if not constraints.allows(ranges, op.src_rank, op.dst_rank, op.stage_id):
            return False
    return True


def clone_unit(unit, template_ops, lowering, base_priority):
    """The transfers of ``unit`` along the route of ``template_ops``."""
    n = lowering.inputs.hyperparameters.slice_count
    remap = lambda values: frozenset((s // n) * n + unit.position for s in values)
    names = {op.operation_id: 'alloc:%s:%s' % (unit.demand_id, op.operation_id) for op in template_ops}
    first = min(op.priority_time for op in template_ops)
    return [replace(op, operation_id=names[op.operation_id], contributors=remap(op.contributors),
                    target_contributors=remap(op.target_contributors),
                    dependencies=frozenset(names[d] for d in op.dependencies if d in names),
                    reference_dependencies=frozenset(),
                    priority_time=base_priority + (op.priority_time - first)) for op in template_ops]


# ---------------------------------------------------------------------------
# trigger: hotspot on one flow, idle path on another flow of the same chunk

def reallocation_strategies(report, bundle, lowering):
    """StrategyRecords for hotspots whose chunk has an idle sibling flow."""
    from vericcl.verification.bottlenecks import StrategyRecord
    settings = ReallocationSettings.from_inputs(lowering.inputs)
    ops = _network_from_storage(bundle)
    by_id = {op.operation_id: op for op in ops}
    units, op_unit = units_of(ops, lowering.inputs)
    flows = chunk_flows(units)
    threshold = report.settings.idle_utilization_threshold
    utilization = getattr(report, 'utilization', {}) or {}
    storage_ids = {op.operation_id for op in bundle.storage_plan.instructions}
    for hotspot in report.hotspots:
        unit_id = op_unit.get(hotspot.operation_id)
        if unit_id is None:
            continue
        unit = units[unit_id]
        hot_flow = unit.shape
        idle = []
        for shape, members in flows[unit.chunk].items():
            if shape == hot_flow:
                continue
            resources = {r for key in units[members[0]].operations for r in _resources(by_id[key], lowering.topology)}
            if hotspot.congested_resource in resources:
                continue
            peak = max((utilization.get((r, b), 0.0) for r in resources for b in hotspot.buckets), default=0.0)
            if peak < threshold:
                idle.append((peak, sorted(members)[0]))
        if not idle:
            continue
        chunks = [unit.chunk]
        # Other chunks that also load the congested resource and can shift
        # their own slices: balance is over all flows sharing the hotspot.
        for chunk, shapes in sorted(flows.items()):
            if chunk in chunks or len(shapes) < 2 or len(chunks) >= settings.max_chunks:
                continue
            if any(hotspot.congested_resource in _resources(by_id[key], lowering.topology)
                   for members in shapes.values() for key in units[members[0]].operations):
                chunks.append(chunk)
        domain = tuple(sorted({key for u in units.values() if u.chunk in chunks for key in u.operations} & storage_ids))
        yield 'reallocation', StrategyRecord(
            'reallocation', domain,
            parameters=(('chunks', tuple(chunks)), ('hot_unit', unit_id), ('hotspot', hotspot.operation_id),
                        ('idle_unit', min(idle)[1]), ('resource', hotspot.congested_resource)),
            cause_ids=('hotspot:' + hotspot.operation_id,))


# ---------------------------------------------------------------------------
# counts model and proposals

def _flow_loads(units, flows, by_id, lowering):
    loads = {}
    for chunk, shapes in flows.items():
        for shape, members in shapes.items():
            total = defaultdict(float)
            for key in units[members[0]].operations:
                for resource, value in _load(by_id[key], lowering).items():
                    total[resource] += value
            loads[chunk, shape] = dict(total)
    return loads


def solve_counts(units, flows, chunks, by_id, lowering, context):
    """SCIP: per-flow slice counts minimizing the largest resource load."""
    from vericcl.solver.scip_api import ScipAdapter
    from vericcl.solver.trees.operators import model_lease
    loads = _flow_loads(units, {c: flows[c] for c in chunks}, by_id, lowering)
    fixed = defaultdict(float)
    for unit in units.values():
        if unit.chunk in chunks:
            continue
        for key in unit.operations:
            for resource, value in _load(by_id[key], lowering).items():
                fixed[resource] += value
    legal = {(chunk, shape): [u for members in flows[chunk].values() for u in members
                              if _legal([by_id[k] for k in units[flows[chunk][shape][0]].operations], units[u].position, lowering)]
             for chunk in chunks for shape in flows[chunk]}
    resources = sorted({r for value in loads.values() for r in value})
    step = min((v for value in loads.values() for v in value.values() if v > 0), default=1.0)
    with model_lease(lowering.inputs, context) as (lease, threads):
        module, model = ScipAdapter.create_model('vericcl-reallocation-counts')
        try:
            ScipAdapter.configure(model, seed=lowering.inputs.solver.solver_seed, threads=threads, gap=0, lease=lease)
            makespan = model.addVar('max_resource_load', lb=0)
            x, moved = {}, []
            for chunk in chunks:
                total = sum(len(members) for members in flows[chunk].values())
                for index, (shape, members) in enumerate(sorted(flows[chunk].items(), key=lambda item: sorted(item[1]))):
                    var = model.addVar('x-%s-%d' % (chunk, index), vtype='I', lb=0, ub=len(legal[chunk, shape]))
                    x[chunk, shape] = var
                    delta = model.addVar('d-%s-%d' % (chunk, index), lb=0)
                    model.addCons(delta >= var - len(members))
                    model.addCons(delta >= len(members) - var)
                    moved.append(delta)
                model.addCons(module.quicksum(x[chunk, shape] for shape in flows[chunk]) == total)
            for resource in resources:
                terms = [x[key] * loads[key][resource] for key in x if resource in loads[key]]
                model.addCons(makespan >= fixed[resource] + module.quicksum(terms))
            # Fewest moved slices among equally balanced allocations.
            model.setObjective(makespan + 1e-3 * step * module.quicksum(moved))
            result = model.optimize()
            if result.primal is None:
                return None
            return {key: int(round(result.solution[var.name])) for key, var in x.items()}
        finally:
            model.freeProb()


def _assignment(units, flows, chunk, counts, by_id, lowering, finish):
    """Units that change flow: latest-finishing units leave over-full flows first."""
    moves = []
    surplus, deficit = [], []
    for shape, members in flows[chunk].items():
        target = counts[chunk, shape]
        if len(members) > target:
            leaving = sorted(members, key=lambda u: (-finish.get(u, 0.0), u))[:len(members) - target]
            surplus.extend(leaving)
        elif target > len(members):
            deficit.extend([shape] * (target - len(members)))
    for unit_id in surplus:
        for index, shape in enumerate(deficit):
            template = [by_id[k] for k in units[flows[chunk][shape][0]].operations]
            if _legal(template, units[unit_id].position, lowering):
                moves.append((unit_id, shape))
                del deficit[index]
                break
    return moves


def build_network(ops, units, flows, moves, by_id, lowering, aliases=()):
    inputs = lowering.inputs
    removed, added, new_aliases = set(), [], []
    for unit_id, shape in moves:
        unit = units[unit_id]
        chunk_members = flows[unit.chunk][shape]
        template = [by_id[k] for k in units[chunk_members[0]].operations]
        base = min(by_id[k].priority_time for k in unit.operations)
        cloned = clone_unit(unit, template, lowering, base)
        removed.update(unit.operations)
        added.extend(cloned)
        new_aliases.extend((op.operation_id, unit.operations[0]) for op in cloned)
    network = _without(list(ops), removed) + added
    return NetworkCandidate(inputs.rank_count, inputs.hyperparameters.slice_count,
                            inputs.hyperparameters.slice_size_bytes, tuple(network), 'reallocation_repair',
                            tuple(aliases) + tuple(new_aliases))


def reallocation_proposals(bundle, strategy, context):
    lowering = context.lowering
    parameters = dict(strategy.parameters)
    ops = _network_from_storage(bundle)
    by_id = {op.operation_id: op for op in ops}
    units, _ = units_of(ops, lowering.inputs)
    flows = chunk_flows(units)
    chunks = [chunk for chunk in parameters['chunks'] if chunk in flows]
    end = bundle.prediction.end_times
    finish = {key: max(end.get(k, 0.0) for k in unit.operations) for key, unit in units.items()}
    aliases = tuple(getattr(bundle.source, 'replaced_operations', ()) or ())
    proposals, seen = [], set()
    counts = solve_counts(units, flows, chunks, by_id, lowering, context)
    if counts is not None:
        moves = [move for chunk in chunks for move in _assignment(units, flows, chunk, counts, by_id, lowering, finish)]
        if moves:
            key = tuple(sorted(moves, key=repr))
            seen.add(key)
            description = {'kind': 'balanced_counts', 'counts': {
                '%s/flow%d' % (chunk, index): counts[chunk, shape]
                for chunk in chunks for index, shape in enumerate(sorted(flows[chunk], key=lambda s: sorted(flows[chunk][s])))},
                'moved': len(moves)}
            proposals.append((description, build_network(ops, units, flows, moves, by_id, lowering, aliases)))
    # Incremental moves hot flow -> idle flow, latest-finishing slices first.
    hot, idle = units.get(parameters['hot_unit']), units.get(parameters['idle_unit'])
    if hot is not None and idle is not None:
        hot_members = sorted(flows[hot.chunk][hot.shape], key=lambda u: (-finish.get(u, 0.0), u))
        template = [by_id[k] for k in idle.operations]
        candidates = [u for u in hot_members if _legal(template, units[u].position, lowering)]
        settings = ReallocationSettings.from_inputs(lowering.inputs)
        for count in range(1, min(settings.max_incremental_moves, len(candidates)) + 1):
            if count >= len(hot_members):
                break
            moves = [(u, idle.shape) for u in candidates[:count]]
            key = tuple(sorted(moves, key=repr))
            if key in seen:
                continue
            seen.add(key)
            proposals.append(({'kind': 'incremental', 'moved': count,
                               'from': len(hot_members), 'to': len(flows[idle.chunk][idle.shape])},
                              build_network(ops, units, flows, moves, by_id, lowering, aliases)))
    return proposals


def solve_reallocation(bundle, strategy, context):
    from .proposals import evaluate_proposals, prepare_lowering
    prepare_lowering(context)
    domain = RepairDomain(frozenset(strategy.operation_ids), 'reallocation', strategy.cause_ids)
    proposals = reallocation_proposals(bundle, strategy, context)
    return evaluate_proposals(bundle, domain, proposals, context, label='reallocation')
