"""Correctness of an emitted XML by full symbolic execution and BDD reasoning.

Every verdict is a BDD computation over relations produced by the symbolic
execution of the XML itself (``executor``):

  coverage        Reach = fixpoint of Init | Image(Reach, T); Required <= Reach,
                  with T(slice, rank, rank2) the executed transfers
  exactly_once    reductions combine disjoint contribution sets, and every output
                  buffer's final contribution set equals the required one
  causality       every read is ordered after the write of the version it reads
                  (RAW) and never reads an empty region: Need & ~Before = 0
  buffer_safety   a region is overwritten only after every read of the previous
                  version (WAR) and writes are ordered (WAW); loop passes use
                  disjoint sub-ranges of every buffer (MSCCL gridOffset)
  constraints     T(slice, rank, rank2, stage) & Forbidden = 0, Forbidden built
                  from the users' byte-range bans and the fixed stage count
  relay_legality  T & ~Allowed = 0: links of the topology, and between nodes
                  only from a gateway to a gateway
  deadlock        the K-pass progress model (threadblock order, dependency
                  flags, send/receive pairing, bounded connection FIFO) has no
                  wait cycle at the window's smallest, representative and
                  largest message size
  fifo_order      on every connection the k-th send and the k-th receive carry
                  the same chunk count and the same logical transfer
  xml_consistency the executed XML transfer relation (slice, rank, rank2,
                  channel) equals the logical schedule's

A failed check returns a witness picked from the violating BDD (or the cycle).
"""
from __future__ import annotations

import math
import time

from vericcl.input.schema import DATATYPE_BYTES
from vericcl.semantics.collective import CollectiveKind, required_outputs
from vericcl.semantics.ranges import DataRange, bind_constraints
from vericcl.verification.symbolic_check import SymbolicCheckResult
from vericcl.xml.capacity import simple_payload_capacity
from vericcl.xml.v5_model import Address
from vericcl.xml.v5_buffers import canonical_region

from .config import settings as read_settings
from .executor import RECV_KINDS, execute_xml

CHECKS = ('coverage', 'exactly_once', 'causality', 'buffer_safety', 'constraints', 'relay_legality',
          'deadlock', 'fifo_order', 'xml_consistency')


def check_sizes(execution):
    """Message sizes MSCCL can run this XML at: window minimum, representative, maximum."""
    inputs, program = execution.inputs, execution.program
    element = DATATYPE_BYTES[inputs.collective.datatype]
    unit = program.chunks_per_loop * element
    rep = inputs.hyperparameters.total_size_bytes
    if inputs.collective.kind is CollectiveKind.ALL_GATHER:
        rep *= inputs.rank_count
    low = max(unit, -(-program.min_bytes // unit) * unit)
    high = ((program.max_bytes - 1) // unit) * unit if program.max_bytes > 0 else rep
    sizes = sorted({size for size in (low, rep, high) if size >= unit and (program.max_bytes <= 0 or
                    program.min_bytes <= size < program.max_bytes)} or {rep})
    buffsize = int(inputs.request_config['runtime']['environment']['NCCL_BUFFSIZE'])
    chunk_bytes = simple_payload_capacity(buffsize, element)
    plans = []
    for size in sizes:
        per_chunk = size // program.chunks_per_loop
        loops = max(1, -(-per_chunk // chunk_bytes))
        passes = min(loops, execution.settings.loop_iterations)
        message = min(per_chunk, chunk_bytes)
        plans.append({'bytes': size, 'loops': loops, 'passes': passes,
                      'fifo_messages': execution.fifo_messages(message)})
    return plans


def _transfer_relations(execution):
    s = execution.space
    plain = s.false
    staged = s.false
    channelled = s.false
    for transfer in execution.transfers:
        pair = s.eq('rank', transfer.src_rank) & s.eq('rank2', transfer.dst_rank)
        plain |= transfer.contrib & pair
        staged |= transfer.contrib & pair & s.eq('stage', transfer.stage)
        channelled |= transfer.contrib & pair & s.eq('channel', transfer.channel)
    return plain, staged, channelled


def _schedule_relation(execution, storage):
    """T_sched(slice, rank, rank2, channel) of the logical schedule (storage plan)."""
    s = execution.space
    relation = s.false
    for op in storage.instructions:
        if op.kind not in {'send', 'rrc'}:
            continue
        source = storage.values[op.inputs[0]]
        contrib = s.set_of('slice', sorted(source.contributors))
        relation |= contrib & s.eq('rank', storage.addresses[op.inputs[0]].rank) & s.eq('rank2', op.rank) \
            & s.eq('channel', max(op.channel, 0))
    return relation


def forbidden_relation(s, inputs, n, size):
    """Forbidden(slice, rank, rank2, stage) in any space with those fields."""
    constraints = bind_constraints(inputs)
    relation = s.false
    for ban in constraints.bans:
        rows = []
        for position in range(n):
            slice_id = ban.data.source_rank * n + position
            piece = DataRange(ban.data.source_rank, position * size, (position + 1) * size)
            if ban.data.intersects(piece):
                rows.append(slice_id)
        if rows and ban.stage_id < s.sizes['stage']:
            relation |= s.set_of('slice', rows) & s.eq('rank', ban.src_rank) & s.eq('rank2', ban.dst_rank) \
                & s.eq('stage', ban.stage_id)
    if constraints.fixed_stage_count is not None:
        relation |= ~s.less_than('stage', constraints.fixed_stage_count) & s.domain('stage')
    return relation


def allowed_relation(s, topology):
    """Allowed(rank, rank2): topology links; between nodes only gateway to gateway."""
    rows = []
    for link in topology.links:
        src, dst = link.src_rank, link.dst_rank
        same = topology.node_membership.get(src) == topology.node_membership.get(dst)
        if same or not topology.gateways or (src in topology.gateways and dst in topology.gateways):
            rows.append((src, dst))
    return s.relation(('rank', 'rank2'), rows)


def full_check(xml_text, storage, program, executable, inputs, topology, *, backend='cudd', lease=None):
    """(SymbolicCheckResult, FullExecution) for one lowered candidate."""
    started = time.perf_counter()
    settings = read_settings(inputs)
    execution = execute_xml(xml_text, inputs, program=program, storage=storage, executable=executable,
                            settings=settings, backend=backend, lease=lease)
    s = execution.space
    results, witnesses = {name: 'valid' for name in CHECKS}, {}

    def fail(name, witness):
        results[name] = 'invalid'
        witnesses.setdefault(name, witness)

    for name, items in execution.violations.items():
        if items:
            fail(name if name in results else 'xml_consistency', items[0])
    executed = bool(execution.order)
    plain, staged, channelled = _transfer_relations(execution)

    # coverage: image fixpoint from the initial contents over the executed transfers
    n, ranks = execution.n, execution.ranks
    expected = required_outputs(inputs.collective, ranks, n)
    required = s.relation(('slice', 'rank'), {(c, slot.rank) for slot, cs in expected.items() for c in cs})
    reach = s.relation(('slice', 'rank'), execution.initial)
    rounds = 0
    while True:
        rounds += 1
        grown = reach | s.image(reach, plain, present=('rank',), nxt=('rank2',))
        if grown == reach:
            break
        reach = grown
    missing = required & ~reach
    if executed and not s.is_empty(missing):
        fail('coverage', dict(zip(('slice', 'rank'), s.witness(missing, ('slice', 'rank')))))

    # exactly once at the outputs: final contribution set == required set
    if executed:
        for slot, contributors in expected.items():
            region = canonical_region(Address(slot.rank, 'o', slot.offset), inputs)
            final = execution.final_contents(region)
            want = s.set_of('slice', sorted(contributors))
            have = s.false if final is None else final.contrib
            difference = s.xor(have, want)
            if not s.is_empty(difference):
                fail('exactly_once', {'output_rank': slot.rank, 'offset': slot.offset,
                                      'contribution': s.witness(difference, ('slice',))[0],
                                      'present': s.values(have, 'slice')[:8], 'required': sorted(contributors)[:8]})
                break

    # ordering of reads and writes on every region (single pass)
    if executed:
        raw, war = set(), set()
        for versions in execution.memory.values():
            for position, version in enumerate(versions):
                for reader in version.readers:
                    if version.writer is not None and reader != version.writer:
                        raw.add((version.writer, reader))
                    if position + 1 < len(versions):
                        nxt = versions[position + 1].writer
                        if nxt is not None and nxt != reader:
                            war.add((reader, nxt))
                if position + 1 < len(versions) and version.writer is not None \
                        and versions[position + 1].writer not in {None, version.writer}:
                    war.add((version.writer, versions[position + 1].writer))
        for name, rows in (('causality', raw), ('buffer_safety', war)):
            need = s.relation(('node', 'node2'), rows)
            violation = execution.unordered(need)
            if not s.is_empty(violation):
                first, second = s.witness(violation, ('node', 'node2'))
                fail(name, {'first': execution.describe(first), 'then': execution.describe(second),
                            'reason': 'not ordered by the progress model'})

    # hard constraints and relay legality
    forbidden = forbidden_relation(s, inputs, execution.n, execution.slice_bytes)
    hit = staged & forbidden
    if not s.is_empty(hit):
        fail('constraints', dict(zip(('slice', 'src', 'dst', 'stage'), s.witness(hit, ('slice', 'rank', 'rank2', 'stage')))))
    allowed = allowed_relation(s, topology)
    illegal = s.exist(('slice',), plain) & ~allowed
    if not s.is_empty(illegal):
        fail('relay_legality', dict(zip(('src', 'dst'), s.witness(illegal, ('rank', 'rank2')))))

    # runtime progress at the window's sizes
    sizes = check_sizes(execution)
    progress = []
    for plan in sizes:
        verdict = execution.check_progress(plan['passes'], plan['fifo_messages'])
        progress.append(dict(plan, rounds=verdict['rounds'], edges=verdict['edges'], deadlock=verdict['cycle'] is not None))
        if verdict['cycle'] is not None:
            fail('deadlock', {'bytes': plan['bytes'], 'passes': plan['passes'],
                              'fifo_messages': plan['fifo_messages'], 'cycle': verdict['cycle']})

    # per-connection sequences: count and logical transfer of the k-th send/receive
    rows_send, rows_recv = set(), set()
    connection_index = {key: index for index, key in enumerate(execution.connections)}
    for key, members in execution.sends.items():
        for k, index in enumerate(members):
            node = execution.nodes[index]
            rows_send.add((connection_index[key], k, node.count, node.operation_id or ''))
    for key, members in execution.recvs.items():
        for k, index in enumerate(members):
            node = execution.nodes[index]
            rows_recv.add((connection_index[key], k, node.count, node.operation_id or ''))
    if execution.connections:
        s_seq = _sequence_space(execution, rows_send | rows_recv)
        sent = s_seq.relation(('conn', 'seq', 'cnt', 'op'), _encode(rows_send, s_seq.op_index))
        received = s_seq.relation(('conn', 'seq', 'cnt', 'op'), _encode(rows_recv, s_seq.op_index))
        differ = s_seq.xor(sent, received)
        if not s_seq.is_empty(differ):
            conn, seq, count, op = s_seq.witness(differ, ('conn', 'seq', 'cnt', 'op'))
            fail('fifo_order', {'connection': list(execution.connections[conn]), 'position': seq, 'count': count,
                                'operation': s_seq.op_names[op]})

    # emitted XML == logical schedule
    if storage is not None and executed:
        scheduled = _schedule_relation(execution, storage)
        differ = s.xor(channelled, scheduled)
        if not s.is_empty(differ):
            fail('xml_consistency', dict(zip(('slice', 'src', 'dst', 'channel'),
                                             s.witness(differ, ('slice', 'rank', 'rank2', 'channel')))))

    valid = all(results[name] == 'valid' for name in CHECKS)
    statistics = {'mode': 'full', 'xml_steps': len(execution.nodes), 'executed_steps': len(execution.order),
                  'transfers': len(execution.transfers), 'symbols': len(execution.symbols), 'atoms': len(execution.atoms),
                  'regions': len(execution.memory), 'coverage_rounds': rounds, 'progress': progress,
                  'bdd_backend': s.backend, 'bdd_nodes': s.node_count(),
                  'seconds': round(time.perf_counter() - started, 3)}
    return SymbolicCheckResult(valid, results, witnesses, statistics), execution


def _encode(rows, op_index):
    return {(conn, seq, count, op_index[op]) for conn, seq, count, op in rows}


def _sequence_space(execution, rows):
    from .space import FullSpace
    names = sorted({op for _, _, _, op in rows})
    space = FullSpace(((('conn', max(1, len(execution.connections))),),
                       (('seq', 1 + max((seq for _, seq, _, _ in rows), default=0)),),
                       (('cnt', 1 + max((count for _, _, count, _ in rows), default=1)),),
                       (('op', max(1, len(names))),)), backend=execution.space.backend)
    space.op_index = {name: index for index, name in enumerate(names)}
    space.op_names = names
    return space
