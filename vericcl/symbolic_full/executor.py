"""Symbolic execution of an MSCCL XML program over per-rank buffer state.

Input: the XML text (read independently by ``xmlprog``) and, optionally, the
lowering's own program, storage plan and executable, used only to name steps
(operation ids, logical stage, cost component) and never as the source of
data contents.

1. Progress model. Every XML step is a node. A node waits for
   * ``tb``    the previous step of its threadblock (and, across loop passes,
               the last step of the previous pass);
   * ``dep``   the step named by depid/deps (MSCCL dependency flag);
   * ``data``  the k-th send of its connection (it is the k-th receive);
   * ``fifo``  the receive that frees the buffer slot a send needs: a
               connection holds C messages, so send k+C waits for receive k.
   Sends and receives pair by order on each (src rank, dst rank, channel)
   connection; the sequence numbers continue from one loop pass to the next.
   The graph of K passes is a BDD over (node, iter); peeling it layer by layer
   yields an execution order, and a non-empty remainder is a wait cycle, i.e.
   a deadlock.
2. State. Memory is keyed by the canonical buffer region of each chunk
   (in-place aliases resolved). A cell holds a version: its contribution set as
   a BDD over the ``slice`` field, its path pd (the Symbols that carried the
   contributions from their source ranks), its start/end time terms and the
   node that wrote it. Loop passes address disjoint sub-ranges of every buffer
   (MSCCL adds gridOffset to input, output and scratch offsets), so one pass
   determines the data semantics.
3. Steps. ``s`` reads its source into a message; ``r`` writes the message;
   ``rrc``/``rrs``/``rrcs`` reduce the message with the local source (the two
   contribution sets must be disjoint) and write and/or forward the result;
   ``rcs`` writes and forwards; ``cpy`` copies; ``re`` reduces locally. Each
   transfer creates one Symbol (src rank, dst rank, ready_time term) - the only
   way Symbols are made here - and every written version is an atom
   {s, pd, t}.
4. Times. start(node) = max over its waits (labelled tb/dep/data/fifo, the
   end of the version it reads, and the end of every read of the version it
   overwrites); end(node) = start + alpha[cost] + bytes * beta_eff[node] for
   receives and copies. Terms are symbolic; ``bind`` substitutes a cost model
   (and the simulator's contention as beta_eff) without re-executing.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Tuple

from vericcl.errors import SemanticError
from vericcl.semantics.atom import Atom, PathDAG, SliceVec, TimeVec, _new_symbol
from vericcl.semantics.collective import CollectiveKind
from vericcl.symbolic.terms import TermFactory, evaluate
from vericcl.xml.v5_buffers import canonical_region
from vericcl.xml.v5_model import Address

from .config import FullSymbolicSettings
from .space import FullSpace
from .xmlprog import XProgram, parse_xml

SEND_KINDS = frozenset({'s', 'rcs', 'rrs', 'rrcs'})
RECV_KINDS = frozenset({'r', 'rrc', 'rcs', 'rrs', 'rrcs'})
REDUCING = frozenset({'rrc', 'rrs', 'rrcs', 're'})
WRITES = frozenset({'r', 'rrc', 'rcs', 'rrcs', 'cpy', 're'})
LOCAL_SOURCE = frozenset({'s', 'rrc', 'rrs', 'rrcs', 'cpy', 're'})
KNOWN = SEND_KINDS | RECV_KINDS | frozenset({'cpy', 're', 'nop'})


@dataclass(frozen=True)
class Node:
    index: int
    rank: int
    tb: int
    step: int
    kind: str
    channel: int
    send_peer: int
    recv_peer: int
    count: int
    operation_id: Optional[str]
    stage: int
    cost_key: Optional[str]


@dataclass
class Version:
    region: object
    number: int
    writer: Optional[int]
    contrib: object                 # BDD over 'slice'
    pd: PathDAG
    start: object                   # Term
    end: object                     # Term
    readers: List[int] = field(default_factory=list)


@dataclass(frozen=True)
class Transfer:
    send: int
    recv: int
    src_rank: int
    dst_rank: int
    channel: int
    stage: int
    contrib: object                 # BDD over 'slice'
    symbol: object
    operation_id: Optional[str]


@dataclass(frozen=True)
class LoweringLink:
    """Names for XML steps from the lowering (identity, stage, cost) - never data."""
    steps: Dict[Tuple[int, int, int], str]
    stages: Dict[str, int]
    costs: Dict[str, Tuple[str, int]]
    resources: Dict[str, Tuple[str, ...]]

    @classmethod
    def from_lowering(cls, program, storage, executable):
        steps = {}
        if program is not None:
            for block in program.blocks:
                for index, step in enumerate(block.steps):
                    if step.kind != 'nop':
                        steps[block.rank, block.tb_id, index] = step.operation_id
        stages = {} if storage is None else {op.operation_id: op.stage_id for op in storage.instructions}
        costs, resources = {}, {}
        for op in (() if executable is None else executable.operations):
            if op.kind != 'nop':
                costs[op.operation_id] = (op.cost_key, op.payload_bytes)
                resources[op.operation_id] = tuple(op.resources)
        return cls(steps, stages, costs, resources)


def _ceil_div(a, b):
    return -(-a // b)


class FullExecution:
    """Result of one symbolic execution (unbound terms; ``bind`` evaluates them)."""

    def __init__(self, program: XProgram, inputs, link: LoweringLink, settings: FullSymbolicSettings,
                 backend='cudd', lease=None):
        self.program, self.inputs, self.link, self.settings = program, inputs, link, settings
        self.lease = lease
        self.n = inputs.hyperparameters.slice_count
        self.ranks = inputs.rank_count
        self.slice_bytes = inputs.hyperparameters.slice_size_bytes
        self.factory = TermFactory()
        self.violations: Dict[str, list] = defaultdict(list)
        self.nodes: List[Node] = []
        self.by_position: Dict[Tuple[int, int, int], int] = {}
        self._build_nodes()
        self.preds: Dict[int, Dict[int, str]] = {node.index: {} for node in self.nodes}
        self.sends: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
        self.recvs: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
        self._build_connections()
        stages = 1 + max([node.stage for node in self.nodes] + [inputs.atom_constraints.stage_num or 1])
        channels = 1 + max([node.channel for node in self.nodes] + [0])
        self.max_iterations = max(1, settings.loop_iterations)
        self.space = FullSpace((
            (('slice', max(1, self.ranks * self.n)),),
            (('rank', self.ranks), ('rank2', self.ranks), ('rank3', self.ranks)),
            (('channel', channels),),
            (('stage', stages),),
            (('node', max(1, len(self.nodes))), ('node2', max(1, len(self.nodes))), ('node3', max(1, len(self.nodes)))),
            (('iter', self.max_iterations), ('iter2', self.max_iterations), ('iter3', self.max_iterations)),
        ), backend=backend)
        self.edge_rows: Dict[Tuple[int, int], List[tuple]] = {}
        self.progress: Dict[Tuple[int, int], dict] = {}
        self.order: Tuple[int, ...] = ()
        self.start: Dict[int, object] = {}
        self.end: Dict[int, object] = {}
        self.data_ready: Dict[int, object] = {}
        self.branches: Dict[int, tuple] = {}            # node -> ((label, term), ...) of its start max
        self.memory: Dict[object, List[Version]] = defaultdict(list)
        self.reads: Dict[int, List[Tuple[object, int]]] = defaultdict(list)
        self.writes: Dict[int, List[Tuple[object, int]]] = defaultdict(list)
        self.transfers: List[Transfer] = []
        self.symbols: Dict[int, object] = {}
        self.atoms: Dict[str, Atom] = {}
        self.initial: List[Tuple[int, int]] = []        # (slice, rank) present before execution
        self._binding = None
        self._memo = {}
        self._before = None

    # ------------------------------------------------------------------ nodes
    def _build_nodes(self):
        for block in sorted(self.program.blocks, key=lambda b: (b.rank, b.tb)):
            for step in block.steps:
                if step.kind not in KNOWN:
                    self.violations['xml_consistency'].append({'rank': block.rank, 'tb': block.tb, 'step': step.index,
                                                               'unknown_kind': step.kind})
                operation = self.link.steps.get((block.rank, block.tb, step.index))
                stage = self.link.stages.get(operation, 0) if operation else 0
                cost = self.link.costs.get(operation, (None, None))[0] if operation else None
                if cost is None and step.kind in RECV_KINDS and block.recv >= 0:
                    cost = ('rrc:%d:%d' if step.kind in REDUCING else 'link:%d:%d') % (block.recv, block.rank)
                node = Node(len(self.nodes), block.rank, block.tb, step.index, step.kind, block.channel,
                            block.send, block.recv, step.count, operation, stage, cost)
                self.nodes.append(node)
                self.by_position[block.rank, block.tb, step.index] = node.index
        self._steps = {(s.rank, s.tb, s.index): s for b in self.program.blocks for s in b.steps}

    def step_of(self, node: Node):
        return self._steps[node.rank, node.tb, node.step]

    # ------------------------------------------------------------ connections
    def _build_connections(self):
        blocks = defaultdict(list)
        for node in self.nodes:
            blocks[node.rank, node.tb].append(node.index)
        self.blocks = {key: sorted(members, key=lambda i: self.nodes[i].step) for key, members in blocks.items()}
        owners = defaultdict(set)
        for (rank, tb), members in sorted(self.blocks.items()):
            previous = None
            for index in members:
                node = self.nodes[index]
                step = self.step_of(node)
                if previous is not None:
                    self.preds[index][previous] = 'tb'
                previous = index
                if step.depid >= 0:
                    target = self.by_position.get((rank, step.depid, step.deps))
                    if target is None or not self.step_of(self.nodes[target]).hasdep:
                        # MSCCL would wait for a flag nobody sets.
                        self.violations['deadlock'].append({'node': index, 'rank': rank, 'tb': tb, 'step': node.step,
                                                            'reason': 'dependency target does not publish completion',
                                                            'depid': step.depid, 'deps': step.deps})
                    elif target != index:
                        self.preds[index][target] = 'dep'
                if node.kind in SEND_KINDS:
                    if node.send_peer < 0:
                        self.violations['fifo_order'].append({'node': index, 'reason': 'send step on a threadblock without a send peer'})
                    else:
                        key = (rank, node.send_peer, node.channel)
                        self.sends[key].append(index)
                        owners['send', key].add(tb)
                if node.kind in RECV_KINDS:
                    if node.recv_peer < 0:
                        self.violations['fifo_order'].append({'node': index, 'reason': 'receive step on a threadblock without a receive peer'})
                    else:
                        key = (node.recv_peer, rank, node.channel)
                        self.recvs[key].append(index)
                        owners['recv', key].add(tb)
        for (direction, key), tbs in owners.items():
            if len(tbs) > 1:
                self.violations['fifo_order'].append({'connection': list(key), 'direction': direction,
                                                      'reason': 'several threadblocks share one connection direction',
                                                      'tbs': sorted(tbs)})
        self.connections = sorted(set(self.sends) | set(self.recvs))
        for key in self.connections:
            sends, recvs = self.sends.get(key, []), self.recvs.get(key, [])
            if len(sends) != len(recvs):
                self.violations['fifo_order'].append({'connection': list(key), 'sends': len(sends), 'receives': len(recvs),
                                                      'reason': 'send and receive counts differ'})
            for send, recv in zip(sends, recvs):
                self.preds[recv][send] = 'data'

    # --------------------------------------------------------------- progress
    def fifo_messages(self, message_bytes: int) -> int:
        buffsize = int(self.inputs.request_config['runtime']['environment']['NCCL_BUFFSIZE'])
        step_bytes = max(1, buffsize // self.settings.fifo_slots)
        slots = max(1, min(self.settings.chunk_steps, _ceil_div(max(1, message_bytes), step_bytes)))
        return max(1, self.settings.fifo_slots // slots)

    def progress_rows(self, passes: int, capacity: int):
        """Edges (pred node, pred pass, node, pass) of the K-pass progress model."""
        key = (passes, capacity)
        if key in self.edge_rows:
            return self.edge_rows[key]
        rows = []
        for it in range(passes):
            for index, preds in self.preds.items():
                for pred, label in preds.items():
                    if label in {'tb', 'dep', 'data'}:
                        rows.append((pred, it, index, it))
            if it + 1 < passes:
                for members in self.blocks.values():
                    rows.append((members[-1], it, members[0], it + 1))
        for key_ in self.connections:
            sends, recvs = self.sends.get(key_, []), self.recvs.get(key_, [])
            length = min(len(sends), len(recvs))
            if not length:
                continue
            for it in range(passes):
                for k in range(length):
                    target = it * length + k + capacity
                    t_it, t_k = divmod(target, length)
                    if t_it < passes:
                        rows.append((recvs[k], it, sends[t_k], t_it))
        self.edge_rows[key] = rows
        return rows

    def check_progress(self, passes: int, capacity: int):
        """Peel the K-pass progress graph; returns (layers, cycle witness or None)."""
        key = (passes, capacity)
        if key in self.progress:
            return self.progress[key]
        s = self.space
        rows = self.progress_rows(passes, capacity)
        relation = s.relation(('node', 'iter', 'node2', 'iter2'), rows)
        nodes = s.less_than('node', len(self.nodes)) & s.less_than('iter', passes)
        layers, remaining = s.peel(nodes, relation, present=('node', 'iter'), nxt=('node2', 'iter2'), lease=self.lease)
        cycle = None
        if not s.is_empty(remaining):
            cycle = self._cycle(relation, remaining)
        result = {'layers': layers, 'cycle': cycle, 'passes': passes, 'capacity': capacity,
                  'edges': len(rows), 'rounds': len(layers)}
        self.progress[key] = result
        return result

    def _cycle(self, relation, remaining):
        s = self.space
        current = s.witness(remaining, ('node', 'iter'))
        seen, path = {}, []
        while current is not None and current not in seen:
            seen[current] = len(path)
            path.append(current)
            preds = s.exist(('node2', 'iter2'), relation & s.eq('node2', current[0]) & s.eq('iter2', current[1])) & remaining
            current = s.witness(preds, ('node', 'iter'))
            if len(path) > 4 * len(self.nodes) + 8:
                break
        if current is None:
            return [self.describe(node, it) for node, it in path[:16]]
        loop = path[seen[current]:]
        return [self.describe(node, it) for node, it in reversed(loop)][:64]

    def describe(self, index, it=0):
        node = self.nodes[index]
        return {'rank': node.rank, 'tb': node.tb, 'step': node.step, 'pass': it, 'kind': node.kind,
                'operation': node.operation_id}

    # --------------------------------------------------------------- execution
    def run(self):
        single = self.check_progress(1, self.fifo_messages(self.slice_bytes))
        if single['cycle'] is not None:
            self.violations['deadlock'].append({'cycle': single['cycle'], 'passes': 1})
            return self
        s = self.space
        order = []
        for layer in single['layers']:
            order.extend(sorted(node for node, it in s.tuples(layer, ('node', 'iter'))))
        self.order = tuple(order)
        self._initial_state()
        queues = defaultdict(deque)
        for index in self.order:
            if self.lease is not None and len(self.start) % 1024 == 0:
                self.lease.check()
            self._step(self.nodes[index], queues)
        return self

    def _region(self, rank, buffer, offset):
        return canonical_region(Address(rank, buffer, offset), self.inputs)

    def _initial_state(self):
        kind = self.inputs.collective.kind
        root = self.inputs.collective.root
        sources = [root] if kind in {CollectiveKind.BROADCAST, CollectiveKind.SCATTER} else range(self.ranks)
        zero = self.factory.zero
        for rank in sources:
            for position in range(self.n):
                slice_id = rank * self.n + position
                region = self._region(rank, 'i', position)
                self.memory[region].append(Version(region, 0, None, self.space.eq('slice', slice_id),
                                                   PathDAG((), frozenset({rank}), rank), zero, zero))
                self.initial.append((slice_id, rank))

    def _read(self, index, rank, buffer, offset):
        region = self._region(rank, buffer, offset)
        versions = self.memory.get(region)
        if not versions:
            self.violations['causality'].append({'reader': self.describe(index), 'region': [region.rank, region.buffer_id, region.offset],
                                                 'reason': 'reads a buffer region that holds no data'})
            return None
        version = versions[-1]
        version.readers.append(index)
        self.reads[index].append((region, version.number))
        return version

    def _write(self, index, rank, buffer, offset, contrib, pd, start, end):
        region = self._region(rank, buffer, offset)
        versions = self.memory[region]
        version = Version(region, len(versions), index, contrib, pd, start, end)
        versions.append(version)
        self.writes[index].append((region, version.number))
        return version

    def _step(self, node: Node, queues):
        f, s = self.factory, self.space
        step = self.step_of(node)
        branches = [('%s:%d' % (label, pred), self.end[pred]) for pred, label in sorted(self.preds[node.index].items())]
        chunks = range(max(1, step.count))
        local = []
        if node.kind in LOCAL_SOURCE:
            for c in chunks:
                version = self._read(node.index, node.rank, step.srcbuf, step.srcoff + c)
                local.append(version)
                if version is not None and version.writer is not None:
                    branches.append(('data_ready:%d' % version.writer, version.end))
        accumulators = []
        if node.kind == 're':
            for c in chunks:
                accumulators.append(self._read(node.index, node.rank, step.dstbuf, step.dstoff + c))
        if node.kind in WRITES:
            for c in chunks:
                versions = self.memory.get(self._region(node.rank, step.dstbuf, step.dstoff + c))
                if versions:
                    for reader in versions[-1].readers:
                        if reader != node.index and reader in self.end:
                            branches.append(('slot:%d' % reader, self.end[reader]))
        start = f.max(tuple(branches)) if branches else f.zero
        self.branches[node.index] = tuple(branches)
        # Data readiness alone (no threadblock, dependency, FIFO or slot waits):
        # the time the step could start if only its data mattered.
        data = tuple(branch for branch in branches if branch[0].startswith('data'))
        self.data_ready[node.index] = f.max(data) if data else f.zero
        message = None
        if node.kind in RECV_KINDS:
            key = (node.recv_peer, node.rank, node.channel)
            message = queues[key].popleft() if queues[key] else None
            if message is None:
                self.violations['fifo_order'].append({'receiver': self.describe(node.index), 'reason': 'receive without a matching send'})
        payload_bytes = self.slice_bytes * max(1, step.count)
        if node.kind in RECV_KINDS or node.kind in {'cpy', 're'}:
            cost = node.cost_key or ('copy:%d' % node.rank)
            end = f.add(start, f.param('alpha', (cost, payload_bytes)), f.mul(float(payload_bytes), f.param('beta_eff', node.index)))
        else:
            end = start
        self.start[node.index], self.end[node.index] = start, end
        # data semantics, chunk by chunk
        outgoing = []
        for c in chunks:
            value = None
            if message is not None:
                received = message['payload'][c] if c < len(message['payload']) else None
                if received is not None:
                    symbol = _new_symbol(message['src'], node.rank, received['ready'], node.stage,
                                         'REDUCE' if node.kind in REDUCING else 'SEND',
                                         node.operation_id or 'n%d' % node.index,
                                         'rrc' if node.kind in REDUCING else 'send', node.channel, s.true)
                    self.symbols.setdefault(node.index, symbol)
                    self.transfers.append(Transfer(message['send'], node.index, message['src'], node.rank, node.channel,
                                                   node.stage, received['contrib'], symbol, node.operation_id))
                    if node.kind in REDUCING and node.kind != 're':
                        mine = local[c] if c < len(local) else None
                        if mine is not None:
                            overlap = mine.contrib & received['contrib']
                            if not s.is_empty(overlap):
                                self.violations['exactly_once'].append({'node': self.describe(node.index),
                                    'duplicate_contribution': s.values(overlap, 'slice')[:4]})
                            value = (mine.contrib | received['contrib'], mine.pd.union(received['pd'], node.rank, symbol))
                    else:
                        value = (received['contrib'], received['pd'].extend(symbol))
            elif node.kind == 'cpy' and c < len(local) and local[c] is not None:
                symbol = _new_symbol(node.rank, node.rank, local[c].end, node.stage, 'SEND',
                                     node.operation_id or 'n%d' % node.index, 'copy', None, s.true)
                self.symbols.setdefault(node.index, symbol)
                value = (local[c].contrib, local[c].pd.extend(symbol))
            elif node.kind == 're' and c < len(local) and local[c] is not None:
                acc = accumulators[c] if c < len(accumulators) else None
                if acc is not None:
                    overlap = acc.contrib & local[c].contrib
                    if not s.is_empty(overlap):
                        self.violations['exactly_once'].append({'node': self.describe(node.index),
                            'duplicate_contribution': s.values(overlap, 'slice')[:4]})
                    symbol = _new_symbol(node.rank, node.rank, local[c].end, node.stage, 'REDUCE',
                                         node.operation_id or 'n%d' % node.index, 'copy', None, s.true)
                    value = (acc.contrib | local[c].contrib, acc.pd.union(local[c].pd, node.rank, symbol))
            elif node.kind in SEND_KINDS and c < len(local) and local[c] is not None:
                outgoing.append({'contrib': local[c].contrib, 'pd': local[c].pd, 'ready': local[c].end})
                continue
            if value is not None:
                if node.kind in {'r', 'rrc', 'rcs', 'rrcs', 'cpy', 're'}:
                    version = self._write(node.index, node.rank, step.dstbuf, step.dstoff + c, value[0], value[1], start, end)
                    self._atom(version, node)
                if node.kind in {'rcs', 'rrs', 'rrcs'}:
                    outgoing.append({'contrib': value[0], 'pd': value[1], 'ready': end})
            elif node.kind in SEND_KINDS and node.kind != 's':
                outgoing.append(None)
        if node.kind in SEND_KINDS and node.send_peer >= 0:
            queues[node.rank, node.send_peer, node.channel].append(
                {'payload': outgoing, 'src': node.rank, 'send': node.index})

    def _atom(self, version: Version, node: Node):
        contributors = frozenset(self.space.values(version.contrib, 'slice'))
        if not contributors:
            return
        state = 'r%d:%s:%d:v%d' % (version.region.rank, version.region.buffer_id, version.region.offset, version.number)
        self.atoms[state] = Atom(SliceVec(min(contributors), self.slice_bytes, node.stage), version.pd,
                                 TimeVec(version.start, version.end), contributors, state)

    # ------------------------------------------------------------ relations
    def wait_relation(self, *, with_tb=True):
        """Single-pass wait edges W(node, node2): node2 waits for node."""
        key = 'with_tb' if with_tb else 'without_tb'
        cache = getattr(self, '_waits', None)
        if cache is None:
            cache = self._waits = {}
        if key not in cache:
            rows = [(pred, index) for index, preds in self.preds.items() for pred, label in preds.items()
                    if with_tb or label != 'tb']
            if with_tb:
                capacity = self.fifo_messages(self.slice_bytes)
                for conn in self.connections:
                    sends, recvs = self.sends.get(conn, []), self.recvs.get(conn, [])
                    length = min(len(sends), len(recvs))
                    for k in range(max(0, length - capacity)):
                        rows.append((recvs[k], sends[k + capacity]))
            cache[key] = self.space.relation(('node', 'node2'), rows)
        return cache[key]

    def unordered(self, need, *, with_tb=True):
        """Pairs (a, b) of ``need`` such that b is not ordered after a by the waits (BDD reachability)."""
        return self.space.unreached(need, self.wait_relation(with_tb=with_tb), left='node', right='node2',
                                    middle='node3', lease=self.lease)

    def before(self):
        """Full transitive closure Before(node, node2) (small programs and tests)."""
        if self._before is None:
            self._before = self.space.closure(self.wait_relation(), left=('node',), right=('node2',), middle=('node3',))
        return self._before

    def ownership(self):
        """contribution set -> [(rank, ready end term, writer node)] for every version ever held."""
        if getattr(self, '_ownership', None) is None:
            owners = defaultdict(list)
            for region, versions in self.memory.items():
                for version in versions:
                    contributors = frozenset(self.space.values(version.contrib, 'slice'))
                    if contributors:
                        owners[contributors].append((region.rank, version.end, version.writer))
            self._ownership = owners
        return self._ownership

    def final_contents(self, region):
        versions = self.memory.get(region)
        return versions[-1] if versions else None

    # ----------------------------------------------------------------- times
    def bind(self, model, prediction=None):
        values = {}
        for node in self.nodes:
            if node.index not in self.end or (node.kind not in RECV_KINDS and node.kind not in {'cpy', 're'}):
                continue
            payload = self.slice_bytes * max(1, node.count)
            cost_key = node.cost_key or ('copy:%d' % node.rank)
            cost = model.service_for(cost_key, payload) if model is not None else None
            alpha = cost.alpha_us if cost is not None else 0.0
            beta = cost.beta_us_per_byte if cost is not None else 0.0
            beta_eff = beta
            op = node.operation_id
            if prediction is not None and op is not None and op in prediction.end_times:
                transfer = prediction.end_times[op] - prediction.start_times[op] - alpha
                if transfer > 0:
                    beta_eff = max(beta, transfer / payload)
            values['alpha', (cost_key, payload)] = alpha
            values['beta_eff', node.index] = beta_eff
        self._binding, self._memo = values, {}
        return self

    def value(self, term):
        if self._binding is None:
            raise SemanticError('bind a cost model before evaluating symbolic times')
        return evaluate(term, lambda key: self._binding.get(key, 0.0), self._memo)

    def times(self):
        return {index: (self.value(self.start[index]), self.value(self.end[index])) for index in self.order}

    def critical_predecessor(self, index):
        """(kind, node) of the wait that decided the start (its max branch); data wins ties."""
        branches = self.branches.get(index, ())
        if not branches:
            return None, None
        best = max(range(len(branches)),
                   key=lambda i: (self.value(branches[i][1]), branches[i][0].startswith('data'), -i))
        kind, _, pred = branches[best][0].partition(':')
        return kind, int(pred)


def execute_xml(xml_text, inputs, *, program=None, storage=None, executable=None, settings=None,
                backend='cudd', lease=None) -> FullExecution:
    from .config import settings as read_settings
    settings = settings or read_settings(inputs)
    link = LoweringLink.from_lowering(program, storage, executable)
    return FullExecution(parse_xml(xml_text), inputs, link, settings, backend=backend, lease=lease).run()
