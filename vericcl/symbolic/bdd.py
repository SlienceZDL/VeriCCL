"""Multi-domain BDD space (CUDD by default) for symbolic schedule reasoning.

Each named field is an unsigned integer domain encoded in binary. Relations are
BDDs over several fields; reasoning uses conjunction, existential
quantification, variable renaming and relational image/closure, so sets such
as "reachable (slice, rank)", "op A must precede op B" or "congested op with an
idle alternative link" are computed symbolically rather than enumerated.
"""
from __future__ import annotations

import importlib
from typing import Dict, Iterable, Mapping, Sequence, Tuple

from vericcl.errors import SemanticError


class BddSpace:
    def __init__(self, domains: Mapping[str, int], *, backend: str = 'cudd'):
        if backend not in {'cudd', 'autoref'}:
            raise SemanticError('unknown explicit BDD backend: ' + str(backend))
        try:
            factory = importlib.import_module('dd.' + backend).BDD
        except (ImportError, AttributeError) as error:
            # Selected backend missing is an explicit failure (no fallback).
            raise SemanticError('requested BDD backend is unavailable: ' + backend) from error
        self.backend = backend
        self.bdd = factory()
        self.sizes: Dict[str, int] = {}
        self.vars: Dict[str, Tuple[str, ...]] = {}
        self._cubes = {}
        for name, size in domains.items():
            self.add_field(name, size)

    def add_field(self, name: str, size: int) -> None:
        if name in self.sizes:
            if self.sizes[name] != size:
                raise SemanticError('BDD field %s redeclared with another size' % name)
            return
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise SemanticError('BDD field size must be a positive integer')
        width = max(1, (size - 1).bit_length())
        names = tuple('%s__%d' % (name, bit) for bit in range(width))
        self.bdd.declare(*names)
        self.sizes[name], self.vars[name] = size, names

    # -- construction ---------------------------------------------------
    @property
    def true(self):
        return self.bdd.true

    @property
    def false(self):
        return self.bdd.false

    def eq(self, name: str, value: int):
        key = (name, value)
        cube = self._cubes.get(key)
        if cube is None:
            if not 0 <= value < self.sizes[name]:
                raise SemanticError('value %r outside BDD field %s' % (value, name))
            cube = self.bdd.true
            for bit, var in enumerate(self.vars[name]):
                literal = self.bdd.var(var)
                cube &= literal if value >> bit & 1 else ~literal
            self._cubes[key] = cube
        return cube

    def row(self, fields: Sequence[str], values: Sequence[int]):
        node = self.bdd.true
        for name, value in zip(fields, values):
            node &= self.eq(name, value)
        return node

    def relation(self, fields: Sequence[str], rows: Iterable[Sequence[int]]):
        node = self.bdd.false
        for values in rows:
            node |= self.row(fields, values)
        return node

    def domain(self, name: str):
        """Valid values of a field (value < size)."""
        return self.relation((name,), ((v,) for v in range(self.sizes[name])))

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
        """Symmetric difference (dd.cudd functions have no ^ operator)."""
        return (left & ~right) | (~left & right)

    # -- reasoning ------------------------------------------------------
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

    def image(self, states, transition, *, present: Sequence[str], nxt: Sequence[str]):
        """{next | exists present: states(present) & transition(present, next)}, renamed to present."""
        step = self.exist(present, states & transition)
        return self.rename(step, dict(zip(nxt, present)))

    def closure(self, relation, *, left: str, right: str, middle: str):
        """Transitive closure of a binary relation over one field type."""
        # Iterative squaring: R := R | R.R doubles the covered path length per
        # round, so a dependency chain of depth d needs about log2(d) rounds.
        closure = relation
        while True:
            composed = self.exist(
                (middle,),
                self.rename(closure, {right: middle}) & self.rename(closure, {left: middle}))
            updated = closure | composed
            if updated == closure:
                return closure
            closure = updated

    # -- inspection -----------------------------------------------------
    def is_empty(self, node) -> bool:
        return node == self.bdd.false

    def tuples(self, node, fields: Sequence[str], limit: int | None = None):
        care = {var for name in fields for var in self.vars[name]}
        result = []
        for assignment in self.bdd.pick_iter(node, care_vars=care):
            values = tuple(sum(1 << bit for bit, var in enumerate(self.vars[name]) if assignment.get(var))
                           for name in fields)
            if all(v < self.sizes[name] for v, name in zip(values, fields)):
                result.append(values)
                if limit is not None and len(result) >= limit:
                    break
        return tuple(sorted(set(result)))

    def witness(self, node, fields: Sequence[str]):
        rows = self.tuples(node, fields, limit=1)
        return rows[0] if rows else None

    def count(self, node, fields: Sequence[str]) -> int:
        return len(self.tuples(node, fields))


class Index:
    """Stable integer encoding of hashable identities for one BDD field."""

    def __init__(self, items: Iterable):
        self.items = tuple(dict.fromkeys(items))
        self.position = {item: index for index, item in enumerate(self.items)}

    def __len__(self):
        return max(1, len(self.items))

    def __getitem__(self, item):
        return self.position[item]

    def __contains__(self, item):
        return item in self.position
