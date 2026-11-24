"""Symbolic time expressions for symbolic execution of collective schedules.

A term is an immutable, hash-consed DAG node. Times are never stored as plain
floats inside symbolic state: every ready, start and end time is a term over
cost-model parameters (alpha/beta per cost component), payload sizes and the
observed contention slowdown of each operation. Evaluating a term under a cost
model gives the concrete time; `critical_args` tells which branch of each
`max` decided it, which is how waiting is attributed to data, threadblock
order or shared-resource contention without re-running the simulation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Tuple

from vericcl.errors import SemanticError


@dataclass(frozen=True, eq=False)
class Term:
    op: str                     # const | param | add | mul | max
    args: Tuple['Term', ...] = ()
    value: float = 0.0          # const value or mul coefficient
    key: Tuple = ()             # param identity
    label: str = ''             # max: role of each branch, comma separated

    def __hash__(self):
        return id(self)

    def __repr__(self):
        if self.op == 'const':
            return repr(self.value)
        if self.op == 'param':
            return 'P%s' % (self.key,)
        if self.op == 'mul':
            return '%r*%r' % (self.value, self.args[0])
        return '%s(%s)' % (self.op, ', '.join(repr(a) for a in self.args))


class TermFactory:
    """Interns terms so structurally equal expressions are the same object.

    `intern=False` builds terms without a table; used for long-lived schedule
    atoms so a process does not accumulate every constant it ever saw.
    """

    def __init__(self, intern: bool = True):
        self._table: Dict[tuple, Term] | None = {} if intern else None
        self.zero = self.const(0.0)

    def _intern(self, op, args=(), value=0.0, key=(), label=''):
        if self._table is None:
            return Term(op, tuple(args), float(value), tuple(key), label)
        identity = (op, tuple(id(a) for a in args), float(value), tuple(key), label)
        term = self._table.get(identity)
        if term is None:
            term = Term(op, tuple(args), float(value), tuple(key), label)
            self._table[identity] = term
        return term

    def const(self, value: float) -> Term:
        if value != value or value < 0:
            raise SemanticError('symbolic constants must be nonnegative numbers')
        return self._intern('const', value=value)

    def param(self, *key) -> Term:
        if not key:
            raise SemanticError('a symbolic parameter needs an identity')
        return self._intern('param', key=key)

    def add(self, *terms: Term) -> Term:
        flat = []
        constant = 0.0
        for term in terms:
            if term.op == 'add':
                flat.extend(term.args)
            elif term.op == 'const':
                constant += term.value
            else:
                flat.append(term)
        if constant:
            flat.append(self.const(constant))
        if not flat:
            return self.zero
        if len(flat) == 1:
            return flat[0]
        return self._intern('add', tuple(flat))

    def mul(self, coefficient: float, term: Term) -> Term:
        if coefficient == 0 or term is self.zero:
            return self.zero
        if coefficient == 1:
            return term
        if term.op == 'const':
            return self.const(coefficient * term.value)
        return self._intern('mul', (term,), value=coefficient)

    def max(self, branches: Mapping[str, Term] | Tuple[Tuple[str, Term], ...]) -> Term:
        """Max over labeled branches; labels name the waiting cause."""
        items = tuple(branches.items()) if isinstance(branches, Mapping) else tuple(branches)
        unique = {}
        for label, term in items:
            if term is not self.zero and term not in unique.values():
                unique[label] = term
        if not unique:
            return self.zero
        if len(unique) == 1:
            return next(iter(unique.values()))
        labels = tuple(unique)
        return self._intern('max', tuple(unique[label] for label in labels), label=','.join(labels))


SCHEDULE_TERMS = TermFactory(intern=False)


class SymValue(float):
    """A concrete time that also carries the symbolic term it was derived from.

    It behaves as a float everywhere (comparison, arithmetic, JSON, exact
    rational conversion via str), so schedule code keeps working, while the
    symbolic dependency structure stays attached to atoms and Symbols.
    """
    __slots__ = ('term',)

    def __new__(cls, value, term: 'Term | None' = None):
        self = super().__new__(cls, value)
        self.term = SCHEDULE_TERMS.const(float(value)) if term is None else term
        return self

    def __reduce__(self):
        return (SymValue, (float(self), self.term))


def evaluate(term: Term, parameter: Callable[[Tuple], float], memo: Dict[int, float] | None = None) -> float:
    """Evaluate a term; `parameter` binds each param key to a float."""
    memo = {} if memo is None else memo
    stack = [(term, False)]
    while stack:
        node, expanded = stack.pop()
        if id(node) in memo:
            continue
        if node.op == 'const':
            memo[id(node)] = node.value
            continue
        if node.op == 'param':
            memo[id(node)] = float(parameter(node.key))
            continue
        if not expanded:
            stack.append((node, True))
            stack.extend((arg, False) for arg in node.args if id(arg) not in memo)
            continue
        values = [memo[id(arg)] for arg in node.args]
        if node.op == 'add':
            memo[id(node)] = sum(values)
        elif node.op == 'mul':
            memo[id(node)] = node.value * values[0]
        elif node.op == 'max':
            memo[id(node)] = max(values)
        else:
            raise SemanticError('unknown symbolic term operator ' + node.op)
    return memo[id(term)]


def critical_branch(term: Term, memo: Dict[int, float]) -> Tuple[str, Term] | None:
    """For a max term already evaluated in `memo`, the deciding (label, branch)."""
    if term.op != 'max':
        return None
    labels = term.label.split(',')
    index = max(range(len(term.args)), key=lambda i: (memo[id(term.args[i])], -i))
    return labels[index], term.args[index]
