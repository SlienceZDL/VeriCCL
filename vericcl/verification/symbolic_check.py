"""Correctness of a lowered schedule by symbolic execution + BDD reasoning.

The SymbolicExecutor produces Symbols (single-step transfers) and atoms
{s, pd, t}. From them this module builds BDD relations and checks:

  coverage        Required(slice, rank) is contained in the fixpoint
                  Reach = Init u Image(Reach, T)
  causality       every read happens after the producer of the read state:
                  Reads(op, st) & Produces(st, p) & ~Before(p, op) = 0
  exactly_once    reductions combine disjoint contributor sets, and every final
                  output carries exactly its required contributors
  constraints     T4(slice, src, dst, stage) & Forbidden = 0
  deadlock        no (x, x) in the transitive closure of the wait-for relation
  buffer_safety   an overwrite of a buffer region happens after every read of
                  the previous value: Overwrite(w, st) & Reads(r, st) & ~Before(r, w)
                  & (r != w) = 0
  xml_consistency the transfer relation of the emitted threadblock program
                  equals the schedule's transfer relation

Each violated check returns a concrete witness picked from the violating BDD.
"""
from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
import threading
from typing import Dict, Mapping, Optional, Tuple

from vericcl.errors import SemanticError
from vericcl.semantics.collective import required_outputs
from vericcl.semantics.ranges import DataRange, bind_constraints
from vericcl.symbolic.bdd import BddSpace, Index
from vericcl.symbolic.executor import SymbolicExecution, execute_bundle

CHECKS = ('coverage', 'causality', 'exactly_once', 'constraints', 'deadlock', 'buffer_safety', 'xml_consistency')


@dataclass(frozen=True)
class SymbolicCheckResult:
    valid: bool
    checks: Mapping[str, str]                    # name -> 'valid' | 'invalid'
    witnesses: Mapping[str, object] = field(default_factory=dict)
    statistics: Mapping[str, object] = field(default_factory=dict)

    @property
    def message(self):
        failed = [name for name in CHECKS if self.checks.get(name) == 'invalid']
        if not failed:
            return 'symbolic execution + BDD: all checks hold'
        return '; '.join('%s violated, witness %s' % (name, self.witnesses.get(name)) for name in failed)


class ScheduleRelations:
    """BDD encodings shared by correctness checking and hotspot localization."""

    def __init__(self, execution: SymbolicExecution, inputs, *, backend: str = 'cudd'):
        self.execution, self.inputs = execution, inputs
        n, ranks = execution.slice_count, execution.rank_count
        self.ops = Index(execution.order)
        self.states = Index(execution.atoms)
        stages = 1 + max((record.stage for record in execution.operations.values()), default=0)
        channels = 1 + max((record.channel for record in execution.operations.values()), default=0)
        self.space = BddSpace({
            'op': len(self.ops), 'op2': len(self.ops), 'op3': len(self.ops),
            'state': len(self.states), 'slice': max(1, n * ranks),
            'rank': ranks, 'rank2': ranks, 'stage': stages, 'channel': channels,
        }, backend=backend)
        s = self.space
        records = execution.operations
        network = [key for key in execution.order if records[key].kind in {'send', 'rrc'}]
        # T(slice, rank, rank2): contributions carried by each single-step transfer.
        self.transfer = s.relation(('slice', 'rank', 'rank2'), {
            (c, records[key].src_rank, records[key].dst_rank)
            for key in network for c in execution.atoms[records[key].inputs[0]].contributors})
        self.transfer_staged = s.relation(('slice', 'rank', 'rank2', 'stage'), {
            (c, records[key].src_rank, records[key].dst_rank, records[key].stage)
            for key in network for c in execution.atoms[records[key].inputs[0]].contributors})
        self.ops_transfer = s.relation(('op', 'rank', 'rank2', 'channel'), {
            (self.ops[key], records[key].src_rank, records[key].dst_rank, records[key].channel) for key in network})
        # D(op, op2): op2 waits for op (op precedes op2).
        self.depends = s.relation(('op', 'op2'), {
            (self.ops[p], self.ops[key]) for key in execution.order for p in records[key].dependencies})
        self._before = None
        self.reads = s.relation(('op', 'state'), {
            (self.ops[key], self.states[st]) for key in execution.order for st in records[key].inputs})
        self.produces = s.relation(('state', 'op2'), {
            (self.states[st], self.ops[p]) for st, p in execution.producer.items() if p is not None})
        self.contrib = s.relation(('state', 'slice'), {
            (self.states[st], c) for st, atom in execution.atoms.items() for c in atom.contributors})

    @property
    def before(self):
        """Transitive closure of the dependency relation: Before(op, op2)."""
        if self._before is None:
            self._before = self.space.closure(self.depends, left='op', right='op2', middle='op3')
        return self._before

    def op_name(self, value):
        return self.ops.items[value]

    def state_name(self, value):
        return self.states.items[value]


def _coverage(rel: ScheduleRelations, required):
    s = rel.space
    init = s.relation(('slice', 'rank'), {
        (c, atom.rank) for st, atom in rel.execution.atoms.items()
        if rel.execution.producer[st] is None for c in atom.contributors})
    reach = init
    while True:
        grown = reach | s.image(reach, rel.transfer, present=('rank',), nxt=('rank2',))
        if grown == reach:
            break
        reach = grown
    missing = required & ~reach
    return s.is_empty(missing), s.witness(missing, ('slice', 'rank'))


def check_execution(execution: SymbolicExecution, bundle, inputs, *, backend: str = 'cudd') -> SymbolicCheckResult:
    rel = ScheduleRelations(execution, inputs, backend=backend)
    s, records = rel.space, execution.operations
    results, witnesses = {}, {}
    n, ranks = execution.slice_count, execution.rank_count
    expected = required_outputs(inputs.collective, ranks, n)

    # coverage ---------------------------------------------------------------
    required = s.relation(('slice', 'rank'), {(c, slot.rank) for slot, contributors in expected.items() for c in contributors})
    ok, witness = _coverage(rel, required)
    results['coverage'] = 'valid' if ok else 'invalid'
    if not ok:
        witnesses['coverage'] = {'slice': witness[0], 'rank': witness[1]}

    # deadlock (cycles in the wait-for relation) -----------------------------
    cycle = rel.before & s.field_equal('op', 'op2')
    results['deadlock'] = 'valid' if s.is_empty(cycle) else 'invalid'
    if results['deadlock'] == 'invalid':
        witnesses['deadlock'] = rel.op_name(s.witness(cycle, ('op',))[0])

    # causality: a state is read only after its producer completed -----------
    # late(op, op2) = op reads a state produced by op2; require Before(op2, op).
    late = s.exist(('state',), rel.reads & rel.produces)
    produced_before = s.rename(rel.before, {'op': 'op3'})
    produced_before = s.rename(produced_before, {'op2': 'op'})
    produced_before = s.rename(produced_before, {'op3': 'op2'})     # Before(op2 -> op)
    violation = late & ~produced_before
    results['causality'] = 'valid' if s.is_empty(violation) else 'invalid'
    if results['causality'] == 'invalid':
        reader, producer = s.witness(violation, ('op', 'op2'))
        witnesses['causality'] = {'reader': rel.op_name(reader), 'producer': rel.op_name(producer)}

    # exactly once ---------------------------------------------------------------
    pairs = {(rel.states[records[key].inputs[0]], rel.states[records[key].inputs[1]])
             for key in execution.order if records[key].kind == 'rrc'}
    overlap_witness = None
    if pairs:
        s.add_field('state2', len(rel.states))
        pair_rel = s.relation(('state', 'state2'), pairs)
        contrib2 = s.rename(rel.contrib, {'state': 'state2'})
        overlap = s.exist(('state', 'state2'), pair_rel & rel.contrib & contrib2)
        if not s.is_empty(overlap):
            overlap_witness = {'duplicate_contribution': s.witness(overlap, ('slice',))[0]}
    absent = sorted(str(state_id) for state_id in execution.outputs.values() if state_id not in rel.states)
    if absent and overlap_witness is None:
        overlap_witness = {'missing_output_state': absent[0]}
    present = {slot: state_id for slot, state_id in execution.outputs.items() if state_id in rel.states}
    final_rows = {(rel.states[state_id], c) for slot, state_id in present.items() for c in expected[slot]}
    final_states = s.relation(('state',), {(rel.states[state_id],) for state_id in present.values()})
    wanted = s.relation(('state', 'slice'), final_rows)
    mismatch = s.xor(final_states & rel.contrib, wanted)
    if overlap_witness is None and not s.is_empty(mismatch):
        state, contribution = s.witness(mismatch, ('state', 'slice'))
        overlap_witness = {'output_state': rel.state_name(state), 'contribution': contribution}
    results['exactly_once'] = 'valid' if overlap_witness is None else 'invalid'
    if overlap_witness is not None:
        witnesses['exactly_once'] = overlap_witness

    # forbidden edges / hard constraints ----------------------------------------
    constraints = bind_constraints(inputs)
    size = inputs.hyperparameters.slice_size_bytes
    banned = set()
    for slice_id, src, dst, stage in s.tuples(rel.transfer_staged, ('slice', 'rank', 'rank2', 'stage')):
        ranges = (DataRange(slice_id // n, (slice_id % n) * size, (slice_id % n + 1) * size),)
        if not constraints.allows(ranges, src, dst, stage):
            banned.add((slice_id, src, dst, stage))
    forbidden = s.relation(('slice', 'rank', 'rank2', 'stage'), banned)
    hit = rel.transfer_staged & forbidden
    results['constraints'] = 'valid' if s.is_empty(hit) else 'invalid'
    if results['constraints'] == 'invalid':
        witnesses['constraints'] = dict(zip(('slice', 'src', 'dst', 'stage'), s.witness(hit, ('slice', 'rank', 'rank2', 'stage'))))

    # buffer safety (write-after-read on the same region) ------------------------
    from vericcl.xml.v5_buffers import canonical_region
    storage = bundle.storage_plan
    position = {key: index for index, key in enumerate(execution.order)}
    by_region = defaultdict(list)
    for state_id in execution.atoms:
        region = canonical_region(storage.addresses[state_id], inputs)
        producer = execution.producer[state_id]
        by_region[region.rank, region.buffer_id, region.offset].append(
            (-1 if producer is None else position[producer], state_id))
    overwrite_rows = set()
    for versions in by_region.values():
        versions.sort()
        for (_, old), (_, new) in zip(versions, versions[1:]):
            writer = execution.producer[new]
            if writer is not None:
                overwrite_rows.add((rel.ops[writer], rel.states[old]))
    overwrite = s.relation(('op2', 'state'), overwrite_rows)
    hazard = s.exist(('state',), overwrite & rel.reads) & ~rel.before & ~s.field_equal('op', 'op2')
    results['buffer_safety'] = 'valid' if s.is_empty(hazard) else 'invalid'
    if results['buffer_safety'] == 'invalid':
        reader, writer = s.witness(hazard, ('op', 'op2'))
        witnesses['buffer_safety'] = {'reader': rel.op_name(reader), 'overwriter': rel.op_name(writer)}

    # emitted threadblock program equals the schedule ----------------------------
    sends, recvs = set(), set()
    for block in bundle.program.blocks:
        for step in block.steps:
            if step.operation_id not in rel.ops:
                continue
            if step.kind == 's':
                sends.add((rel.ops[step.operation_id], block.rank, block.send_peer, block.channel))
            elif step.kind in {'r', 'rrc'}:
                recvs.add((rel.ops[step.operation_id], block.recv_peer, block.rank, block.channel))
    fields = ('op', 'rank', 'rank2', 'channel')
    send_rel, recv_rel = s.relation(fields, sends), s.relation(fields, recvs)
    differ = s.xor(send_rel, rel.ops_transfer) | s.xor(recv_rel, rel.ops_transfer)
    results['xml_consistency'] = 'valid' if s.is_empty(differ) else 'invalid'
    if results['xml_consistency'] == 'invalid':
        witness = s.witness(differ, fields)
        witnesses['xml_consistency'] = {'operation': rel.op_name(witness[0]), 'src': witness[1],
                                        'dst': witness[2], 'channel': witness[3]}

    valid = all(results.get(name) == 'valid' for name in CHECKS)
    statistics = {'operations': len(execution.order), 'symbols': len(execution.symbols),
                  'atoms': len(execution.atoms), 'bdd_backend': s.backend,
                  'bdd_nodes': len(s.bdd)}
    return SymbolicCheckResult(valid, results, witnesses, statistics)


def check_bundle(bundle, inputs, model, *, backend: str = 'cudd') -> Tuple[SymbolicCheckResult, SymbolicExecution]:
    execution = execute_bundle(bundle, inputs, model)
    return check_execution(execution, bundle, inputs, backend=backend), execution


# Correctness does not depend on the cost model. A schedule re-lowered after a
# model update (same XML, storage plan and request) reuses its verdict.
_CHECK_CACHE: 'OrderedDict[tuple, SymbolicCheckResult]' = OrderedDict()
_CHECK_CACHE_LIMIT = 512
_CHECK_LOCK = threading.Lock()


def cached_check(key, compute) -> Tuple[SymbolicCheckResult, bool]:
    """(result, hit). ``key`` must identify the exact lowered program and request."""
    with _CHECK_LOCK:
        hit = _CHECK_CACHE.get(key)
        if hit is not None:
            _CHECK_CACHE.move_to_end(key)
            return hit, True
    result = compute()
    with _CHECK_LOCK:
        _CHECK_CACHE[key] = result
        while len(_CHECK_CACHE) > _CHECK_CACHE_LIMIT:
            _CHECK_CACHE.popitem(last=False)
    return result, False


def clear_check_cache():
    with _CHECK_LOCK:
        _CHECK_CACHE.clear()
