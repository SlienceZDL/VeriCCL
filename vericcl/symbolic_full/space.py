"""BDD space with interleaved variable groups (CUDD through ``dd``).

A field is an unsigned integer stored in binary. Fields declared in one group
(for example ``node``, ``node2``, ``node3``) have their bits interleaved, most
significant bit first, so relations between them (an edge ``node -> node2``,
its composition through ``node3``) stay small. All reasoning is done with
conjunction, disjunction, existential quantification and variable renaming:
images, backward reachability, transitive closure and layered peeling.
"""
from __future__ import annotations

import importlib
from typing import Dict, Iterable, Mapping, Sequence, Tuple

from vericcl.errors import SemanticError


class FullSpace:
    def __init__(self, groups: Sequence[Sequence[Tuple[str, int]]] = (), *, backend: str = 'cudd'):
        if backend not in {'cudd', 'autoref'}:
            raise SemanticError('unknown explicit BDD backend: ' + str(backend))
        try:
            factory = importlib.import_module('dd.' + backend).BDD
        except (ImportError, AttributeError) as error:
            raise SemanticError('requested BDD backend is unavailable: ' + backend) from error
        self.backend = backend
        self.bdd = factory()
        # The interleaved order is chosen on purpose; automatic reordering
        # spends most of its time on short-lived row BDDs.
        try:
            self.bdd.configure(reordering=False)
        except (AttributeError, TypeError, ValueError):
            pass
        self.sizes: Dict[str, int] = {}
        self.vars: Dict[str, Tuple[str, ...]] = {}       # index = bit position
        self._cubes = {}
        for group in groups:
            self.add_group(group)

    # -- declaration -------------------------------------------------------
    def add_group(self, group: Sequence[Tuple[str, int]]) -> None:
        group = [(name, size) for name, size in group]
        for name, size in group:
            if name in self.sizes:
                raise SemanticError('BDD field declared twice: ' + name)
            if isinstance(size, bool) or not isinstance(size, int) or size < 1:
                raise SemanticError('BDD field size must be a positive integer')
        width = max(max(1, (size - 1).bit_length()) for _, size in group)
        declared = {name: [None] * width for name, _ in group}
        for bit in reversed(range(width)):
            for name, _ in group:
                var = '%s__%d' % (name, bit)
                self.bdd.declare(var)
                declared[name][bit] = var
        for name, size in group:
            self.sizes[name] = size
            self.vars[name] = tuple(declared[name])

    def add_field(self, name: str, size: int) -> None:
        self.add_group(((name, size),))

    # -- construction --------------------------------------------------------
    @property
    def true(self):
        return self.bdd.true

    @property
    def false(self):
        return self.bdd.false

    def _assignment(self, name, value):
        if not 0 <= value < self.sizes[name]:
            raise SemanticError('value %r outside BDD field %s' % (value, name))
        return {var: bool(value >> bit & 1) for bit, var in enumerate(self.vars[name])}

    def eq(self, name: str, value: int):
        key = (name, value)
        cube = self._cubes.get(key)
        if cube is None:
            cube = self.bdd.cube(self._assignment(name, value))
            self._cubes[key] = cube
        return cube

    def row(self, fields: Sequence[str], values: Sequence[int]):
        node = self.bdd.true
        for name, value in zip(fields, values):
            node &= self.eq(name, value)
        return node

    def relation(self, fields: Sequence[str], rows: Iterable[Sequence[int]]):
        # Rows are merged in chunks: OR-ing every row into one growing BDD costs
        # rows x size, while each chunk stays small before joining the result.
        result, chunk, count = self.bdd.false, self.bdd.false, 0
        for values in rows:
            chunk |= self.row(fields, values)
            count += 1
            if count % 512 == 0:
                result |= chunk
                chunk = self.bdd.false
        return result | chunk

    def set_of(self, name: str, values: Iterable[int]):
        return self.relation((name,), ((value,) for value in values))

    def less_than(self, name: str, bound: int):
        """{x | x < bound} built from the bits (no enumeration)."""
        variables = self.vars[name]
        if bound >= 1 << len(variables):
            return self.bdd.true
        if bound <= 0:
            return self.bdd.false
        node, prefix = self.bdd.false, self.bdd.true
        for bit in reversed(range(len(variables))):
            x = self.bdd.var(variables[bit])
            if bound >> bit & 1:
                node |= prefix & ~x
                prefix &= x
            else:
                prefix &= ~x
        return node

    def domain(self, name: str):
        return self.less_than(name, self.sizes[name])

    def field_equal(self, left: str, right: str):
        if len(self.vars[left]) != len(self.vars[right]):
            raise SemanticError('compared BDD fields have different widths')
        node = self.bdd.true
        for a, b in zip(self.vars[left], self.vars[right]):
            x, y = self.bdd.var(a), self.bdd.var(b)
            node &= (x & y) | (~x & ~y)
        return node

    @staticmethod
    def xor(left, right):
        return (left & ~right) | (~left & right)

    # -- reasoning -----------------------------------------------------------
    def exist(self, fields: Sequence[str], node):
        names = [var for name in fields for var in self.vars[name]]
        return self.bdd.exist(names, node) if names else node

    def rename(self, node, mapping: Mapping[str, str]):
        substitution = {}
        for source, target in mapping.items():
            if len(self.vars[source]) != len(self.vars[target]):
                raise SemanticError('renamed BDD fields have different widths')
            substitution.update(zip(self.vars[source], self.vars[target]))
        return self.bdd.let(substitution, node) if substitution else node

    def image(self, states, relation, *, present: Sequence[str], nxt: Sequence[str]):
        """Successors: {y | exists x: S(x) & R(x, y)}, returned over the present fields."""
        step = self.exist(present, states & relation)
        return self.rename(step, dict(zip(nxt, present)))

    def preimage(self, states, relation, *, present: Sequence[str], nxt: Sequence[str]):
        """Predecessors: {x | exists y: R(x, y) & S(y)}."""
        shifted = self.rename(states, dict(zip(present, nxt)))
        return self.exist(nxt, relation & shifted)

    def closure(self, relation, *, left: Sequence[str], right: Sequence[str], middle: Sequence[str]):
        """Transitive closure by iterative squaring (about log2(depth) rounds)."""
        closure = relation
        while True:
            first = self.rename(closure, dict(zip(right, middle)))
            second = self.rename(closure, dict(zip(left, middle)))
            updated = closure | self.exist(middle, first & second)
            if updated == closure:
                return closure
            closure = updated

    def unreached(self, need, relation, *, left: str, right: str, middle: str, lease=None):
        """Pairs of ``need`` (left, right) with no path left -> right in ``relation``.

        Equivalent to need & ~closure(relation) without building the closure:
        pairs that are edges or two-step paths are discharged first, then
        multi-source forward images from the remaining sources run only until
        every remaining pair is covered (or nothing new is reached).
        """
        residual = need & ~relation
        if self.is_empty(residual):
            return residual
        shifted = self.rename(self.rename(relation, {right: middle}), {left: right})   # E(right, middle)
        two = self.rename(self.exist((right,), relation & shifted), {middle: right})
        residual &= ~two
        if self.is_empty(residual):
            return residual
        sources = self.exist((right,), residual)
        reached = (relation | two) & sources
        frontier = reached
        while not self.is_empty(residual & ~reached):
            if lease is not None:
                lease.check()
            step = self.rename(self.exist((right,), frontier & shifted), {middle: right})
            new = step & ~reached
            if self.is_empty(new):
                break
            reached |= new
            frontier = new
        return residual & ~reached

    def peel(self, nodes, relation, *, present: Sequence[str], nxt: Sequence[str], lease=None):
        """Topological layers of a graph given as BDDs.

        Each round removes the nodes whose predecessors are all removed. The
        layers are a valid execution order; nodes left at the end all have a
        remaining predecessor, so they contain a cycle (or wait on one).
        """
        layers, live = [], nodes
        while not self.is_empty(live):
            if lease is not None:
                lease.check()
            waiting = self.image(live, relation, present=present, nxt=nxt)   # nodes with a live predecessor
            ready = live & ~waiting
            if self.is_empty(ready):
                break
            layers.append(ready)
            live = live & ~ready
        return layers, live

    # -- inspection ----------------------------------------------------------
    def is_empty(self, node) -> bool:
        return node == self.bdd.false

    def tuples(self, node, fields: Sequence[str], limit: int | None = None):
        care = {var for name in fields for var in self.vars[name]}
        support = set(self.bdd.support(node))
        projected = self.exist([name for name in self.vars if name not in fields
                                and support.intersection(self.vars[name])], node)
        result = set()
        for assignment in self.bdd.pick_iter(projected, care_vars=care):
            values = tuple(sum(1 << bit for bit, var in enumerate(self.vars[name]) if assignment.get(var))
                           for name in fields)
            if all(v < self.sizes[name] for v, name in zip(values, fields)):
                result.add(values)
                if limit is not None and len(result) >= limit:
                    break
        return tuple(sorted(result))

    def witness(self, node, fields: Sequence[str]):
        rows = self.tuples(node, fields, limit=1)
        return rows[0] if rows else None

    def values(self, node, name: str):
        return tuple(row[0] for row in self.tuples(node, (name,)))

    def count(self, node, fields: Sequence[str]) -> int:
        names = [name for name in self.vars if name not in fields]
        projected = self.exist(names, node)
        variables = len([var for name in fields for var in self.vars[name]])
        return int(self.bdd.count(projected, nvars=variables)) if variables else int(not self.is_empty(node))

    def node_count(self) -> int:
        return len(self.bdd)
