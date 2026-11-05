from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping, Tuple

import importlib

from vericcl.errors import SemanticError
from vericcl.verification.model import ValidationStatus


class BDDBackendError(RuntimeError):
    pass


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise SemanticError("{} must be a non-empty string".format(field))
    return value


@dataclass(frozen=True)
class BDDAnalysisResult:
    status: ValidationStatus
    code: str
    message: str
    hints: Tuple[object, ...]
    evidence: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.status not in {
            ValidationStatus.VALID,
            ValidationStatus.ANALYSIS_ERROR,
        }:
            raise SemanticError("BDD analysis status must be valid or analysis_error")
        _identifier(self.code, "bdd_analysis.code")
        _identifier(self.message, "bdd_analysis.message")
        object.__setattr__(self, "hints", tuple(self.hints))
        if not isinstance(self.evidence, Mapping):
            raise SemanticError("bdd_analysis.evidence must be a mapping")
        object.__setattr__(
            self,
            "evidence",
            MappingProxyType(dict(self.evidence)),
        )


class BDDRelation:
    def __init__(self, backend: "CompactBDD", node: object) -> None:
        self._backend = backend
        self._node = node

    def _binary(self, other: "BDDRelation", operation: str) -> "BDDRelation":
        if not isinstance(other, BDDRelation) or other._backend is not self._backend:
            raise BDDBackendError("BDD relations must use the same backend")
        try:
            if operation == "union":
                node = self._node | other._node
            elif operation == "intersection":
                node = self._node & other._node
            elif operation == "difference":
                node = self._node & ~other._node
            else:
                raise BDDBackendError("unknown BDD relation operation")
        except BDDBackendError:
            raise
        except Exception as error:
            raise BDDBackendError(str(error)) from error
        return BDDRelation(self._backend, node)

    def union(self, other: "BDDRelation") -> "BDDRelation":
        return self._binary(other, "union")

    def intersection(self, other: "BDDRelation") -> "BDDRelation":
        return self._binary(other, "intersection")

    def difference(self, other: "BDDRelation") -> "BDDRelation":
        return self._binary(other, "difference")

    def complement(self) -> "BDDRelation":
        try:
            node = self._backend._universe & ~self._node
        except Exception as error:
            raise BDDBackendError(str(error)) from error
        return BDDRelation(self._backend, node)

    def tuples(self) -> Tuple[Tuple[int, ...], ...]:
        return self._backend._enumerate(self._node)


class CompactBDD:
    def __init__(self, domains: Mapping[str, int], *, backend: str = "cudd") -> None:
        if not isinstance(domains, Mapping) or not domains:
            raise BDDBackendError("BDD domains must be a non-empty mapping")
        normalized = []
        seen = set()
        for field, size in domains.items():
            try:
                name = _identifier(field, "bdd.domain")
            except SemanticError as error:
                raise BDDBackendError(str(error)) from error
            if name in seen:
                raise BDDBackendError("BDD domain names must be unique")
            if isinstance(size, bool) or not isinstance(size, int) or size < 1:
                raise BDDBackendError("BDD domain size must be a positive integer")
            seen.add(name)
            normalized.append((name, size))
        self._domains = tuple(normalized)
        self._widths = {
            field: max(1, (size - 1).bit_length())
            for field, size in self._domains
        }
        self._variables = {
            field: tuple(
                "{}__b{:02d}".format(field, bit)
                for bit in range(self._widths[field])
            )
            for field, _ in self._domains
        }
        try:
            if backend not in {"cudd", "autoref"}:
                raise BDDBackendError("unknown explicit BDD backend: " + str(backend))
            self.backend_name = backend
            try:
                factory = importlib.import_module("dd." + backend).BDD
            except (ImportError, AttributeError) as error:
                raise BDDBackendError("requested BDD backend is unavailable: " + backend) from error
            self._bdd = factory()
            self._bdd.declare(
                *(
                    variable
                    for field, _ in self._domains
                    for variable in self._variables[field]
                )
            )
            self._cube_cache = {}
            universe = self._bdd.true
            for field, size in self._domains:
                universe &= self._domain_bound(field, size)
            self._universe = universe
        except Exception as error:
            raise BDDBackendError(str(error)) from error

    @property
    def fields(self) -> Tuple[str, ...]:
        return tuple(field for field, _ in self._domains)

    @property
    def variable_count(self) -> int:
        return sum(self._widths.values())

    def _domain_bound(self, field: str, size: int):
        """Encode unsigned value < size in O(log(size)) Boolean operations."""
        if size == 1 << self._widths[field]:
            return self._bdd.true
        less = self._bdd.false
        equal = self._bdd.true
        for bit in reversed(range(self._widths[field])):
            literal = self._bdd.var(self._variables[field][bit])
            if size & (1 << bit):
                less |= equal & ~literal
                equal &= literal
            else:
                equal &= ~literal
        return less

    def _value_cube(self, field: str, value: int):
        key = (field, value)
        cached = self._cube_cache.get(key)
        if cached is not None:
            return cached
        node = self._bdd.true
        for bit, variable in enumerate(self._variables[field]):
            literal = self._bdd.var(variable)
            node &= literal if value & (1 << bit) else ~literal
        self._cube_cache[key] = node
        return node

    def relation(self, rows: Iterable[Tuple[int, ...]]) -> BDDRelation:
        try:
            normalized = tuple(tuple(row) for row in rows)
        except TypeError as error:
            raise BDDBackendError("BDD relation rows must be iterable") from error
        sizes = tuple(size for _, size in self._domains)
        for row in normalized:
            if len(row) != len(sizes):
                raise BDDBackendError("BDD relation row has the wrong arity")
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                or value >= size
                for value, size in zip(row, sizes)
            ):
                raise BDDBackendError("BDD relation row is outside its domain")
        try:
            node = self._bdd.false
            for row in normalized:
                cube = self._bdd.true
                for (field, _), value in zip(self._domains, row):
                    cube &= self._value_cube(field, value)
                node |= cube
        except Exception as error:
            raise BDDBackendError(str(error)) from error
        return BDDRelation(self, node)

    def _enumerate(self, node: object) -> Tuple[Tuple[int, ...], ...]:
        variables = {
            variable
            for field, _ in self._domains
            for variable in self._variables[field]
        }
        try:
            rows = []
            for assignment in self._bdd.pick_iter(node, care_vars=variables):
                values = []
                valid = True
                for field, size in self._domains:
                    value = sum(
                        (1 << bit) if assignment[variable] else 0
                        for bit, variable in enumerate(self._variables[field])
                    )
                    if value >= size:
                        valid = False
                        break
                    values.append(value)
                if valid:
                    rows.append(tuple(values))
        except Exception as error:
            raise BDDBackendError(str(error)) from error
        return tuple(sorted(set(rows)))


class StrategyDecisionBDD:
    """A compact strategy key plus only genuinely independent choice axes."""
    def __init__(self, strategy_count, independent_domains=None, *, backend="cudd"):
        domains = {"strategy": strategy_count}
        if "strategy" in (independent_domains or {}):
            raise BDDBackendError("strategy is a reserved decision field")
        domains.update(independent_domains or {})
        self.relation = CompactBDD(domains, backend=backend)

    @property
    def variable_count(self):
        return self.relation.variable_count

    def select(self, compatible, *, forbidden=()):
        return self.relation.relation(compatible).difference(self.relation.relation(forbidden)).tuples()
