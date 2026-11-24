"""Symbolic execution of a lowered schedule.

The executor walks the lowered program (storage instructions plus the native
threadblock order) in dependency order. Every physical operation becomes one
Symbol (single-step transfer); every produced state becomes an Atom
{s, pd, t} whose path pd is the union of the Symbols that carried its
contributions from their source ranks, and whose times are symbolic terms:

    st(op) = max over predecessors p of ed(p)        (labelled data / tb / order)
    ed(op) = st(op) + alpha[cost] + bytes * beta_eff[op]

beta_eff[op] = beta[cost] * slowdown[op], where slowdown >= 1 is the contention
the event simulator observed for that operation. Evaluating ed under the same
model reproduces the simulator's completion times; critical branches of the
max terms attribute waiting to data, threadblock order or explicit ordering.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Mapping, Optional, Tuple

from vericcl.errors import SemanticError
from vericcl.symbolic.atom import Atom, PathDAG, SliceVec, Symbol, TimeVec, _make_symbol
from vericcl.symbolic.terms import Term, TermFactory, critical_branch, evaluate


@dataclass
class OperationRecord:
    operation_id: str
    kind: str                   # send | rrc | copy
    src_rank: int
    dst_rank: int
    channel: int
    stage: int
    cost_key: Optional[str]
    payload_bytes: int
    resources: Tuple[str, ...]
    inputs: Tuple[str, ...]
    output: str
    dependencies: Dict[str, str] = field(default_factory=dict)   # predecessor -> label


@dataclass
class SymbolicExecution:
    factory: TermFactory
    operations: Dict[str, OperationRecord]
    order: Tuple[str, ...]
    symbols: Dict[str, Symbol]
    atoms: Dict[str, Atom]              # state_id -> atom
    start: Dict[str, Term]
    end: Dict[str, Term]
    producer: Dict[str, Optional[str]]  # state_id -> operation
    outputs: Mapping[object, str]       # output slot -> final state id
    slice_count: int
    rank_count: int
    _memo: Dict[int, float] = field(default_factory=dict)
    _binding: Optional[Mapping] = None

    # -- evaluation -------------------------------------------------------
    def bind(self, model, prediction=None):
        """Bind alpha/beta from `model` and contention slowdown from `prediction`."""
        values = {}
        for op in self.operations.values():
            if op.cost_key is None:
                continue
            cost = model.service_for(op.cost_key, op.payload_bytes)
            if cost is None:
                raise SemanticError('no cost for ' + op.cost_key + ' at %d bytes' % op.payload_bytes)
            values['alpha', op.cost_key] = cost.alpha_us
            values['beta', op.cost_key] = cost.beta_us_per_byte
            beta_eff = cost.beta_us_per_byte
            if prediction is not None and op.operation_id in prediction.end_times:
                service = prediction.end_times[op.operation_id] - prediction.start_times[op.operation_id]
                transfer = service - cost.alpha_us
                if transfer > 0:
                    beta_eff = max(cost.beta_us_per_byte, transfer / op.payload_bytes)
            values['beta_eff', op.operation_id] = beta_eff
        self._binding, self._memo = values, {}
        return self

    def value(self, term: Term) -> float:
        if self._binding is None:
            raise SemanticError('bind a cost model before evaluating symbolic times')
        return evaluate(term, self._binding.__getitem__, self._memo)

    def times(self) -> Dict[str, Tuple[float, float]]:
        return {key: (self.value(self.start[key]), self.value(self.end[key])) for key in self.order}

    def slowdown(self, operation_id: str) -> float:
        op = self.operations[operation_id]
        if op.cost_key is None:
            return 1.0
        return self._binding['beta_eff', operation_id] / self._binding['beta', op.cost_key]

    def critical_predecessor(self, operation_id: str) -> Tuple[Optional[str], Optional[str]]:
        """(label, predecessor) whose completion decided this operation's start."""
        term = self.start[operation_id]
        self.value(term)
        branch = critical_branch(term, self._memo)
        if branch is None:
            deps = self.operations[operation_id].dependencies
            if len(deps) == 1:
                predecessor, label = next(iter(deps.items()))
                return label, predecessor
            return None, None
        label, _ = branch
        kind, _, predecessor = label.partition(':')
        return kind, predecessor

    def critical_path(self) -> Tuple[str, ...]:
        """Operations on the path that decides the completion time, tail first."""
        if not self.order:
            return ()
        tail = max(self.order, key=lambda key: (self.value(self.end[key]), key))
        path, current = [], tail
        while current is not None:
            path.append(current)
            _, current = self.critical_predecessor(current)
        return tuple(path)


class SymbolicExecutor:
    def __init__(self, storage, program, executable, inputs):
        from vericcl.xml.v5_program import execution_dependencies
        self.storage, self.program, self.inputs = storage, program, inputs
        self.factory = TermFactory()
        self.dependencies, _ = execution_dependencies(program)
        self.work = {op.operation_id: op for op in executable.operations}
        self.slice_count = inputs.hyperparameters.slice_count
        self.rank_count = inputs.rank_count
        self.slice_size = inputs.hyperparameters.slice_size_bytes
        self._tb_previous = {}
        for block in program.blocks:
            previous = None
            for step in block.steps:
                if step.kind == 'nop':
                    continue
                if previous is not None and previous != step.operation_id:
                    self._tb_previous.setdefault(step.operation_id, set()).add(previous)
                previous = step.operation_id

    def _record(self, instruction) -> OperationRecord:
        addresses = self.storage.addresses
        source = addresses[instruction.inputs[0]]
        destination = addresses[instruction.output]
        work = self.work.get(instruction.operation_id)
        producers = {self.storage.values[key].producer for key in instruction.inputs}
        labels = {}
        for predecessor in self.dependencies.get(instruction.operation_id, ()):
            if predecessor in producers:
                labels[predecessor] = 'data'
            elif predecessor in self._tb_previous.get(instruction.operation_id, ()):
                labels[predecessor] = 'tb'
            else:
                labels[predecessor] = 'order'
        return OperationRecord(
            instruction.operation_id, instruction.kind, source.rank, destination.rank,
            max(instruction.channel, 0), instruction.stage_id,
            None if work is None else work.cost_key,
            self.slice_size if work is None else work.payload_bytes,
            () if work is None else tuple(work.resources),
            tuple(instruction.inputs), instruction.output, labels)

    def step(self, execution: SymbolicExecution, record: OperationRecord) -> Symbol:
        """Execute one operation symbolically; returns its Symbol."""
        f = execution.factory
        for state_id in record.inputs:
            if state_id not in execution.atoms:
                raise SemanticError('symbolic execution reads a state that does not exist yet: ' + state_id)
        branches = tuple(('%s:%s' % (label, predecessor), execution.end[predecessor])
                         for predecessor, label in sorted(record.dependencies.items()))
        start = f.max(branches)
        if record.cost_key is None:
            end = start
        else:
            end = f.add(start, f.param('alpha', record.cost_key),
                        f.mul(float(record.payload_bytes), f.param('beta_eff', record.operation_id)))
        source_atom = execution.atoms[record.inputs[0]]
        symbol = _make_symbol(record.operation_id, record.src_rank, record.dst_rank,
                              source_atom.t.ed_time, 'rrc' if record.kind == 'rrc' else
                              ('copy' if record.kind == 'copy' else 'send'),
                              record.channel, record.stage)
        value = self.storage.values[record.output]
        path = source_atom.pd
        if record.kind == 'rrc':
            accumulator = execution.atoms[record.inputs[1]]
            path = path.union(accumulator.pd, record.dst_rank, symbol)
        else:
            path = path.extend(symbol)
        contributors = frozenset(value.contributors)
        execution.atoms[record.output] = Atom(
            SliceVec(min(contributors), self.slice_size, record.stage), path,
            TimeVec(start, end), contributors, record.output)
        execution.start[record.operation_id], execution.end[record.operation_id] = start, end
        execution.symbols[record.operation_id] = symbol
        execution.producer[record.output] = record.operation_id
        return symbol

    def run(self) -> SymbolicExecution:
        storage = self.storage
        records = {op.operation_id: self._record(op) for op in storage.instructions}
        execution = SymbolicExecution(self.factory, records, (), {}, {}, {}, {}, {},
                                      dict(storage.outputs), self.slice_count, self.rank_count)
        zero = self.factory.zero
        for state_id in storage.initial_ids:
            value = storage.values[state_id]
            rank = storage.addresses[state_id].rank
            contributors = frozenset(value.contributors)
            execution.atoms[state_id] = Atom(
                SliceVec(min(contributors), self.slice_size, 0),
                PathDAG(frozenset(), frozenset(s // self.slice_count for s in contributors) | {rank}, rank),
                TimeVec(zero, zero), contributors, state_id)
            execution.producer[state_id] = None
        # Kahn order over the full native dependency relation.
        indegree = {key: len(record.dependencies) for key, record in records.items()}
        followers = defaultdict(list)
        for key, record in records.items():
            for predecessor in record.dependencies:
                if predecessor not in records:
                    raise SemanticError('dependency on an unknown operation: ' + predecessor)
                followers[predecessor].append(key)
        ready = deque(sorted(key for key, degree in indegree.items() if degree == 0))
        order = []
        while ready:
            key = ready.popleft()
            self.step(execution, records[key])
            order.append(key)
            for follower in sorted(followers[key]):
                indegree[follower] -= 1
                if indegree[follower] == 0:
                    ready.append(follower)
        if len(order) != len(records):
            raise SemanticError('symbolic execution found a dependency cycle (deadlock)')
        execution.order = tuple(order)
        return execution


def execute_bundle(bundle, inputs, model=None):
    """Symbolically execute a lowered candidate and bind its model/prediction."""
    if bundle.storage_plan is None or bundle.program is None or bundle.executable is None:
        raise SemanticError('symbolic execution needs a fully lowered candidate')
    execution = SymbolicExecutor(bundle.storage_plan, bundle.program, bundle.executable, inputs).run()
    if model is not None:
        execution.bind(model, bundle.prediction)
    return execution
