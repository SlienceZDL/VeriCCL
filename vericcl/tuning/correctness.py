"""Correctness repair (repair type 1).

A candidate that violates a hard correctness constraint is repaired locally
before it is given up. Violations are found in two layers:

* Network semantics: a symbolic execution of value ownership over the
  physical transfers. It records which rank holds which contribution set after
  every transfer and checks the result against the collective's required
  outputs and the source-byte edge constraints.
    - missing delivery     -> add the cheapest legal hops from a rank that
                              already owns the value
    - unavailable source   -> add the missing dependency on the transfer that
                              delivers the value, else reroute from an owner
    - forbidden edge       -> reroute the transfer over legal edges
    - duplicate reduction  -> rebuild that logical position with a fresh legal
                              route column
* Runtime order: the symbolic+BDD check of the lowered program
  (``symbolic_check``) and the native dependency check.
    - deadlock / causality -> reorder every threadblock queue by one global
                              topological order of the mandatory dependencies
    - buffer WAR/WAW       -> add reader-before-overwriter, then reorder
    - XML/program mismatch -> lower again from the network operations

Every proposal is lowered and re-checked. A candidate is rejected only when no
proposal passes within the step limit. Failures outside correctness (launch
capacity, missing cost coverage) are not handled here.
"""
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
import heapq

from vericcl.errors import SemanticError
from vericcl.execution.budget import BudgetExhausted
from vericcl.semantics.collective import CollectiveKind, required_outputs
from vericcl.semantics.ranges import DataRange, bind_constraints
from vericcl.xml.v5_model import NetworkCandidate, NetworkOperation, StorageCandidate

CORRECTNESS_CHECKS = ('semantic', 'storage', 'endpoint', 'deadlock', 'xml')
# Network violations are fixed first: order repairs on a semantically wrong
# network would only be undone by the next lowering.
NETWORK_PRIORITY = ('cycle', 'unavailable', 'forbidden', 'duplicate', 'missing')
ORDER_KINDS = ('deadlock', 'causality', 'buffer_safety')


@dataclass(frozen=True)
class Violation:
    kind: str
    witness: Mapping
    message: str = ''


@dataclass
class CorrectnessReport:
    status: str = 'valid'          # valid | repaired | rejected | not_correctness
    steps: list = field(default_factory=list)

    def as_json(self):
        return {'status': self.status, 'steps': list(self.steps)}


# ---------------------------------------------------------------------------
# network operations of a (possibly failed) candidate

def _position(contributors, n):
    return min(contributors) % n


def _network_from_storage(bundle):
    storage = bundle.storage_plan
    ops = {op.operation_id: op for op in storage.instructions}
    network_ids = {key for key, op in ops.items() if op.kind in {'send', 'rrc'}}
    starts = bundle.prediction.start_times if bundle.prediction is not None else {}

    def upstream(state_id, seen=()):
        producer = storage.values[state_id].producer
        if producer is None or producer in seen:
            return set()
        if producer in network_ids:
            return {producer}
        return {key for state in ops[producer].inputs for key in upstream(state, seen + (producer,))}
    result = []
    for key, op in ops.items():
        if key not in network_ids:
            continue
        source, target = storage.values[op.inputs[0]], storage.values[op.output]
        deps = frozenset(dep for state in op.inputs for dep in upstream(state))
        result.append(NetworkOperation(key, op.kind, source.rank, op.rank, op.channel, op.stage_id,
                                       source.contributors, target.contributors, op.participants, deps,
                                       frozenset(), starts.get(key, op.priority_time)))
    return tuple(result)


def network_operations(bundle, lowering):
    """Physical transfers of a candidate, whichever lowering stage it reached."""
    source = bundle.source
    if isinstance(source, NetworkCandidate):
        return tuple(source.operations), tuple(source.replaced_operations)
    aliases = tuple(getattr(source, 'replaced_operations', ()) or ())
    if bundle.storage_plan is not None:
        return _network_from_storage(bundle), aliases
    if isinstance(source, StorageCandidate):
        origin = source.origin
        if isinstance(origin, NetworkCandidate):
            return tuple(origin.operations), tuple(origin.replaced_operations)
        raise SemanticError('storage candidate without a network origin')
    from vericcl.xml.v5_buffers import normalize_network
    return tuple(normalize_network(source, lowering)), aliases


# ---------------------------------------------------------------------------
# symbolic execution of value ownership

def _dependency_order(ops, *, reference=True):
    by_id = {op.operation_id: op for op in ops}
    deps = {key: {d for d in (op.dependencies | (op.reference_dependencies if reference else frozenset())) if d in by_id}
            for key, op in by_id.items()}
    successors = defaultdict(list)
    for key, values in deps.items():
        for value in values:
            successors[value].append(key)
    remaining = {key: len(values) for key, values in deps.items()}
    ready = [(by_id[key].priority_time, key) for key, count in remaining.items() if count == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        _, key = heapq.heappop(ready)
        order.append(by_id[key])
        for nxt in successors[key]:
            remaining[nxt] -= 1
            if remaining[nxt] == 0:
                heapq.heappush(ready, (by_id[nxt].priority_time, nxt))
    if len(order) != len(by_id):
        return None
    return order


@dataclass
class Ownership:
    """Who holds which contribution set, and which transfer produced it."""
    holds: dict                      # (rank, position) -> {contributors: producer op id or None}
    violations: list
    order: tuple

    def owners(self, contributors, n):
        position = _position(contributors, n)
        key = frozenset(contributors)
        return {rank: values[key] for (rank, p), values in self.holds.items()
                if p == position and key in values}


def symbolic_ownership(ops, inputs):
    """Execute the network symbolically over ownership; collect violations."""
    n, ranks = inputs.hyperparameters.slice_count, inputs.rank_count
    order = _dependency_order(ops)
    violations = []
    if order is None:
        violations.append(Violation('cycle', {}, 'network dependencies contain a cycle'))
        order = _dependency_order(ops, reference=False)
        if order is None:
            return Ownership({}, violations, ())
    holds = defaultdict(dict)
    kind = inputs.collective.kind
    sources = (inputs.collective.root,) if kind is CollectiveKind.BROADCAST else range(ranks)
    for rank in sources:
        for logical in range(n):
            holds[rank, logical][frozenset({rank * n + logical})] = None
    constraints = bind_constraints(inputs)
    size = inputs.hyperparameters.slice_size_bytes
    for op in order:
        contributors = frozenset(op.contributors)
        if not contributors:
            violations.append(Violation('forbidden', {'operation': op.operation_id}, 'transfer carries no slice'))
            continue
        position = _position(contributors, n)
        ranges = tuple(DataRange(s // n, (s % n) * size, (s % n + 1) * size) for s in contributors)
        if not constraints.allows(ranges, op.src_rank, op.dst_rank, op.stage_id):
            violations.append(Violation('forbidden', {'operation': op.operation_id, 'src': op.src_rank,
                                                      'dst': op.dst_rank, 'stage': op.stage_id,
                                                      'contributors': sorted(contributors)},
                                        'transfer uses a forbidden source-byte edge'))
        if contributors not in holds[op.src_rank, position]:
            violations.append(Violation('unavailable', {'operation': op.operation_id, 'rank': op.src_rank,
                                                        'contributors': sorted(contributors)},
                                        'transfer source does not own its value'))
        if op.kind == 'send':
            holds[op.dst_rank, position][contributors] = op.operation_id
            continue
        target = frozenset(op.target_contributors) if op.target_contributors else None
        partners = [held for held in holds[op.dst_rank, position]
                    if not held & contributors and (target is None or held | contributors <= target)]
        if not partners:
            violations.append(Violation('duplicate', {'operation': op.operation_id, 'position': position,
                                                      'contributors': sorted(contributors)},
                                        'reduction has no disjoint destination value'))
            continue
        partner = max(partners, key=lambda held: (len(held), sorted(held)))
        holds[op.dst_rank, position][partner | contributors] = op.operation_id
    for slot, contributors in required_outputs(inputs.collective, ranks, n).items():
        key = frozenset(contributors)
        if key not in holds[slot.rank, _position(key, n)]:
            violations.append(Violation('missing', {'rank': slot.rank, 'contributors': sorted(key)},
                                        'required output is never delivered'))
    rank_of = {kind: index for index, kind in enumerate(NETWORK_PRIORITY)}
    violations.sort(key=lambda v: rank_of[v.kind])
    return Ownership(holds, violations, tuple(order))


# ---------------------------------------------------------------------------
# network fixes

def _hop_cost(lowering, src, dst):
    size = lowering.inputs.hyperparameters.slice_size_bytes
    try:
        cost = lowering.model.service_for('link:%d:%d' % (src, dst), size) if lowering.model is not None else None
    except (KeyError, SemanticError):
        cost = None
    return 1.0 if cost is None else cost.alpha_us + cost.beta_us_per_byte * size


def legal_path(lowering, contributors, starts, goal, stage, participants, *, avoid=frozenset()):
    """Cheapest legal path from any of ``starts`` (rank -> ready time) to ``goal``."""
    inputs, topology = lowering.inputs, lowering.topology
    n, size = inputs.hyperparameters.slice_count, inputs.hyperparameters.slice_size_bytes
    constraints = bind_constraints(inputs)
    ranges = tuple(DataRange(s // n, (s % n) * size, (s % n + 1) * size) for s in contributors)
    allowed = set(participants)
    adjacency = defaultdict(list)
    for key in topology.links:
        if key.src_rank in allowed and key.dst_rank in allowed and key.dst_rank not in avoid:
            if constraints.allows(ranges, key.src_rank, key.dst_rank, stage):
                adjacency[key.src_rank].append(key.dst_rank)
    best, previous = {}, {}
    queue = []
    for rank, ready in starts.items():
        if rank in allowed and (rank not in best or ready < best[rank]):
            best[rank] = ready
            previous[rank] = None
            heapq.heappush(queue, (ready, rank))
    while queue:
        lowering.lease.check()
        cost, rank = heapq.heappop(queue)
        if cost > best.get(rank, float('inf')):
            continue
        if rank == goal:
            path = [goal]
            while previous[path[-1]] is not None:
                path.append(previous[path[-1]])
            return tuple(reversed(path))
        for nxt in adjacency[rank]:
            value = cost + _hop_cost(lowering, rank, nxt)
            if value < best.get(nxt, float('inf')):
                best[nxt], previous[nxt] = value, rank
                heapq.heappush(queue, (value, nxt))
    return None


def _ready_times(owners, ops_by_id):
    return {rank: (0.0 if producer is None else ops_by_id[producer].priority_time + 1e-6)
            for rank, producer in owners.items()}


def route_ops(op, path, producer_of, prefix):
    """Replace ``op`` by hops along ``path``; the final hop keeps its identity."""
    result, aliases, previous = [], [], None
    for index, (src, dst) in enumerate(zip(path, path[1:])):
        last = index == len(path) - 2
        key = op.operation_id if last else '%s:%s:hop%d:%d-%d' % (prefix, op.operation_id, index, src, dst)
        if previous is not None:
            deps = frozenset({previous})
        else:
            producer = producer_of.get(src)
            deps = frozenset() if producer is None else frozenset({producer})
        result.append(replace(op, operation_id=key, kind=op.kind if last else 'send', src_rank=src, dst_rank=dst,
                              dependencies=deps, reference_dependencies=frozenset(),
                              priority_time=op.priority_time + index * 1e-6))
        if not last:
            aliases.append((key, op.operation_id))
        previous = key
    return result, aliases


def _without(ops, removed):
    removed = set(removed)
    return [replace(op, dependencies=op.dependencies - removed,
                    reference_dependencies=op.reference_dependencies - removed)
            for op in ops if op.operation_id not in removed]


def reroute_position(ops, position, lowering, *, tag='fix'):
    """Replace every transfer of one logical position by a fresh legal route column."""
    from vericcl.solver.trees.operators import routing_demands, _columns, _physical_operation
    inputs, topology, lease = lowering.inputs, lowering.topology, lowering.lease
    n = inputs.hyperparameters.slice_count
    demands = [d for d in routing_demands(inputs) if d.logical_position == position]
    old = [op for op in ops if _position(op.contributors, n) == position
           and any(frozenset(op.contributors) <= d.contributors for d in demands)]
    keep = _without(ops, {op.operation_id for op in old})
    usage = defaultdict(int)
    for op in keep:
        usage[op.src_rank, op.dst_rank] += 1
    base = min((op.priority_time for op in old), default=0.0)
    added, aliases = [], []
    for demand in demands:
        best = None
        for index, route in enumerate(_columns(inputs, topology, demand, lease)):
            if index >= 64:
                break
            physical = _physical_operation(inputs, topology, demand, route, lease)
            score = (max((usage[op.edge] for op in physical), default=0), sum(usage[op.edge] for op in physical), len(physical))
            if best is None or score < best[0]:
                best = (score, route, physical)
        if best is None:
            return None, ()
        _, route, physical = best
        mapping = {op.operation_id: '%s:pos%d:%s' % (tag, position, op.operation_id) for op in physical}
        original = next((op.operation_id for op in old if frozenset(op.contributors) <= demand.contributors), None)
        for op in physical:
            usage[op.edge] += 1
            added.append(NetworkOperation(mapping[op.operation_id], op.kind, *op.edge, 0, op.stage_id,
                                          op.contributors, demand.contributors, tuple(range(inputs.rank_count)),
                                          frozenset(mapping[key] for key in op.dependencies), frozenset(), base))
            if original is not None:
                aliases.append((mapping[op.operation_id], original))
    return keep + added, tuple(aliases)


def _fix_network(ops, aliases, ownership, violation, lowering):
    inputs = lowering.inputs
    n = inputs.hyperparameters.slice_count
    ops = list(ops)
    by_id = {op.operation_id: op for op in ops}
    witness = violation.witness

    def candidate(new_ops, new_aliases=()):
        return NetworkCandidate(inputs.rank_count, n, inputs.hyperparameters.slice_size_bytes, tuple(new_ops),
                                'correctness_repair', tuple(aliases) + tuple(new_aliases))
    if violation.kind == 'cycle':
        # Reference (TB-order) dependencies are hints; data dependencies stay.
        return candidate([replace(op, reference_dependencies=frozenset()) for op in ops]), 'drop_order_hints'
    if violation.kind == 'duplicate':
        rebuilt, extra = reroute_position(ops, witness['position'], lowering)
        return (candidate(rebuilt, extra), 'reroute_position') if rebuilt else (None, None)
    if violation.kind == 'unavailable':
        op = by_id[witness['operation']]
        contributors = frozenset(op.contributors)
        ancestors = _descendants(ops, op.operation_id)
        deliverers = [other for other in ops if other.operation_id != op.operation_id and other.dst_rank == op.src_rank
                      and other.operation_id not in ancestors
                      and (frozenset(other.contributors) == contributors if other.kind == 'send'
                           else frozenset(other.target_contributors or ()) >= contributors)]
        sends = [other for other in deliverers if other.kind == 'send']
        if sends:
            first = min(sends, key=lambda other: (other.priority_time, other.operation_id))
            fixed = replace(op, dependencies=op.dependencies | {first.operation_id})
            return candidate([fixed if other is op else other for other in ops]), 'add_dependency'
        return _reroute(ops, op, ownership, lowering, candidate, 'reroute_from_owner')
    if violation.kind == 'forbidden':
        op = by_id.get(witness.get('operation'))
        if op is None:
            return None, None
        return _reroute(ops, op, ownership, lowering, candidate, 'reroute_legal_edges')
    if violation.kind == 'missing':
        contributors = frozenset(witness['contributors'])
        position = _position(contributors, n)
        reductions = any(op.kind == 'rrc' and _position(op.contributors, n) == position for op in ops)
        if reductions or len(contributors) > 1:
            rebuilt, extra = reroute_position(ops, position, lowering)
            return (candidate(rebuilt, extra), 'reroute_position') if rebuilt else (None, None)
        owners = ownership.owners(contributors, n)
        if not owners:
            return None, None
        stage = max((op.stage_id for op in ops if frozenset(op.contributors) == contributors), default=0)
        path = legal_path(lowering, contributors, _ready_times(owners, by_id), witness['rank'], stage,
                          range(inputs.rank_count), avoid=frozenset(owners))
        if path is None or len(path) < 2:
            return None, None
        target = next((frozenset(op.target_contributors) for op in ops if frozenset(op.contributors) == contributors
                       and op.target_contributors), contributors)
        ready = _ready_times(owners, by_id)[path[0]]
        template = NetworkOperation('fix:deliver:%d:%d' % (min(contributors), witness['rank']), 'send', path[0], path[-1],
                                    0, stage, contributors, target, tuple(range(inputs.rank_count)), frozenset(),
                                    frozenset(), ready)
        added, _ = route_ops(template, path, owners, 'fix')
        return candidate(ops + added), 'add_delivery'
    return None, None


def _descendants(ops, root):
    successors = defaultdict(set)
    for op in ops:
        for dep in op.dependencies | op.reference_dependencies:
            successors[dep].add(op.operation_id)
    seen, stack = set(), [root]
    while stack:
        key = stack.pop()
        for nxt in successors[key]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def _reroute(ops, op, ownership, lowering, candidate, action):
    n = lowering.inputs.hyperparameters.slice_count
    contributors = frozenset(op.contributors)
    by_id = {other.operation_id: other for other in ops}
    # Owners whose copy comes from this transfer or anything after it would
    # create a cycle; the destination itself is the goal, not a start.
    downstream = _descendants(ops, op.operation_id) | {op.operation_id}
    owners = {rank: producer for rank, producer in ownership.owners(contributors, n).items()
              if rank != op.dst_rank and (producer is None or producer not in downstream)}
    if not owners:
        return None, None
    # Ranks that own or receive the value are starts, never relays (no duplicates).
    eventual = (set(ownership.owners(contributors, n)) | {other.dst_rank for other in ops
                if frozenset(other.contributors) == contributors}) - {op.dst_rank}
    path = legal_path(lowering, contributors, _ready_times(owners, by_id), op.dst_rank, op.stage_id, op.participants,
                      avoid=frozenset(eventual))
    if path is None or len(path) < 2:
        return None, None
    hops, aliases = route_ops(op, path, owners, 'fix')
    rebuilt = []
    for other in ops:
        rebuilt.extend(hops if other is op else [other])
    return candidate(rebuilt, aliases), action


# ---------------------------------------------------------------------------
# runtime-order fixes

def _reorder(bundle, lowering, extra=()):
    from vericcl.verification.bottlenecks import mandatory_dependencies
    storage = bundle.storage_plan
    deps = {key: set(values) for key, values in mandatory_dependencies(storage, lowering.inputs).items()}
    for first, second in extra:
        if first in deps and second in deps:
            deps[second].add(first)
    starts = bundle.prediction.start_times if bundle.prediction is not None else {}
    by_id = {op.operation_id: op for op in storage.instructions}
    successors = defaultdict(list)
    for key, values in deps.items():
        for value in values:
            successors[value].append(key)
    remaining = {key: len(values) for key, values in deps.items()}
    ready = [(starts.get(key, by_id[key].priority_time), key) for key, count in remaining.items() if count == 0]
    heapq.heapify(ready)
    ordered = []
    while ready:
        lowering.lease.check()
        _, key = heapq.heappop(ready)
        ordered.append(replace(by_id[key], dependencies=frozenset(deps[key])))
        for nxt in successors[key]:
            remaining[nxt] -= 1
            if remaining[nxt] == 0:
                heapq.heappush(ready, (starts.get(nxt, by_id[nxt].priority_time), nxt))
    if len(ordered) != len(by_id):
        return None
    source = bundle.source
    origin = source.origin if isinstance(source, StorageCandidate) else source
    aliases = tuple(getattr(source, 'replaced_operations', ()) or ())
    return StorageCandidate(replace(storage, instructions=tuple(ordered)), origin, aliases)


def _check_violations(bundle):
    """Violations reported by the lowered-program checks (symbolic+BDD, native)."""
    found = []
    for key in CORRECTNESS_CHECKS:
        check = bundle.checks.get(key)
        if check is None or check.status.value != 'invalid':
            continue
        evidence = check.evidence if isinstance(check.evidence, Mapping) else {}
        symbolic = evidence.get('symbolic_bdd') or {}
        witnesses = evidence.get('witnesses') or {}
        named = [name for name, status in symbolic.items() if status == 'invalid']
        if named:
            found.extend(Violation(name, witnesses.get(name) or {}, check.message) for name in named)
        elif key == 'deadlock':
            found.append(Violation('deadlock', {}, check.message))
        else:
            found.append(Violation('relower', {'check': key}, check.message))
    order = {kind: index for index, kind in enumerate(ORDER_KINDS)}
    found.sort(key=lambda v: order.get(v.kind, len(order)))
    return found


def _fix_order(bundle, violation, lowering):
    if bundle.storage_plan is None:
        return None, None
    if violation.kind in {'deadlock', 'causality'}:
        extra = ()
        if violation.kind == 'causality' and isinstance(violation.witness, Mapping):
            extra = ((violation.witness.get('producer'), violation.witness.get('reader')),)
        proposal = _reorder(bundle, lowering, extra)
        return (proposal, 'global_topological_order') if proposal is not None else (None, None)
    if violation.kind == 'buffer_safety':
        witness = violation.witness if isinstance(violation.witness, Mapping) else {}
        proposal = _reorder(bundle, lowering, ((witness.get('reader'), witness.get('overwriter')),))
        return (proposal, 'reader_before_overwriter') if proposal is not None else (None, None)
    return None, None


def repair_correctness(bundle, lowering, *, max_steps=8):
    """Return ``(bundle, report)``; the bundle is None when no repair passes."""
    from vericcl.xml.lower import lower_and_check
    report = CorrectnessReport()
    if bundle.can_execute:
        return bundle, report
    current, seen, relowered = bundle, set(), False
    for _ in range(max_steps):
        lowering.lease.check()
        failed = {key for key in CORRECTNESS_CHECKS
                  if key in current.checks and current.checks[key].status.value == 'invalid'}
        if not failed:
            report.status = 'not_correctness' if not report.steps else 'rejected'
            return None, report
        try:
            ops, aliases = network_operations(current, lowering)
        except SemanticError as error:
            report.status = 'rejected'
            report.steps.append({'action': 'extract_network', 'error': str(error)})
            return None, report
        ownership = symbolic_ownership(ops, lowering.inputs)
        proposal = action = violation = None
        for violation in ownership.violations:
            proposal, action = _fix_network(ops, aliases, ownership, violation, lowering)
            if proposal is not None:
                break
        if proposal is None and not ownership.violations:
            for violation in _check_violations(current):
                proposal, action = _fix_order(current, violation, lowering)
                if proposal is not None:
                    break
            if proposal is None and not relowered:
                # Lower again from the transfers alone: fresh storage, slots and order.
                relowered = True
                violation = violation or Violation('relower', {}, '')
                proposal, action = NetworkCandidate(
                    lowering.inputs.rank_count, lowering.inputs.hyperparameters.slice_count,
                    lowering.inputs.hyperparameters.slice_size_bytes,
                    tuple(replace(op, reference_dependencies=frozenset()) for op in ops),
                    'correctness_repair', aliases), 'relower_from_network'
        if proposal is None:
            report.status = 'rejected'
            report.steps.append({'violation': None if violation is None else violation.kind,
                                 'witness': None if violation is None else _plain(violation.witness),
                                 'action': None})
            return None, report
        try:
            current = lower_and_check(proposal, lowering)
        except BudgetExhausted:
            raise
        except SemanticError as error:
            report.status = 'rejected'
            report.steps.append({'violation': violation.kind, 'action': action, 'error': str(error)})
            return None, report
        identity = current.xml_sha256 or repr(sorted(current.checks.items()))
        report.steps.append({'violation': violation.kind, 'witness': _plain(violation.witness), 'action': action,
                             'checks': {key: value.status.value for key, value in current.checks.items()}})
        if identity in seen:
            report.status = 'rejected'
            return None, report
        seen.add(identity)
        if current.can_execute:
            report.status = 'repaired'
            return current, report
    report.status = 'rejected'
    return None, report


def _plain(value):
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(item) for item in value]
    return value
