"""VeriCCL atoms: {s, pd, t}, with single-step transfers defined by symbolic execution.

An atom is modelled as three vectors:

- s  = SliceVec(slice_id, slice_size, stage)
- pd = PathDAG: the union of Symbols carrying the slice (or, after a
       reduction, each contribution) from its source rank to the current rank
- t  = TimeVec(st_time, ed_time): start of the transfer and full arrival

A Symbol(src_rank, dst_rank, ready_time) is one single-step transfer. Symbols
exist only as results of symbolic execution: `symbolic_step` (and the lowered
program executor in vericcl.symbolic.executor) are the only constructors.
Their ready_time is a SymValue - a float that also carries a symbolic term -
so code that compares or adds times keeps working while the dependency
structure (ready of hop k = max(ready of hop k-1, decided arrival)) is kept.

`Atom.build(slice_id, slice_size_bytes, path, st_time, ed_time)` symbolically
executes a chain path given as PathStage groups and returns the {s, pd, t}
atom. The legacy read-only views (slice_id, path, st_time, ...) are derived
from s, pd and t.
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Collection, FrozenSet, Mapping, Optional, Tuple

from vericcl.errors import SemanticError
from vericcl.semantics.slice import source_rank
from vericcl.symbolic.terms import SCHEDULE_TERMS, SymValue, Term

if TYPE_CHECKING:
    from vericcl.input.models import ForbiddenTransfer


_TRANSFER_KINDS = frozenset({"REDUCE", "SEND"})
_SYMBOL_KINDS = frozenset({"send", "rrc", "copy"})


class _SymbolicConstruction(threading.local):
    depth = 0


_CONSTRUCTING = _SymbolicConstruction()


def _new_symbol(*args, **kwargs) -> "Symbol":
    """The single construction path for Symbols, used by symbolic execution."""
    _CONSTRUCTING.depth += 1
    try:
        return Symbol(*args, **kwargs)
    finally:
        _CONSTRUCTING.depth -= 1


def _integer(value: object, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SemanticError("{} must be an integer".format(field))
    if value < minimum:
        raise SemanticError("{} must be at least {}".format(field, minimum))
    return value


def _time(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SemanticError("{} must be a number".format(field))
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise SemanticError("{} must be finite and non-negative".format(field))
    return normalized


def _symtime(value: object, field: str) -> SymValue:
    if isinstance(value, SymValue):
        _time(float(value), field)
        return value
    return SymValue(_time(value, field))


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise SemanticError("{} must be a non-empty string".format(field))
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


# --------------------------------------------------------------------------
# Symbol: single-step transfer, produced only by symbolic execution
# --------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class Symbol:
    src_rank: int
    dst_rank: int
    ready_time: object                      # SymValue (schedule) or Term (lowered program)
    stage_id: Optional[int] = None
    operator: Optional[str] = None          # SEND | REDUCE (schedule level)
    symbol_id: Optional[str] = None         # physical operation identity, when known
    kind: Optional[str] = None              # send | rrc | copy (lowered program level)
    channel: Optional[int] = None
    guard: object = None                    # BDD path condition

    def __post_init__(self) -> None:
        if _CONSTRUCTING.depth <= 0:
            raise SemanticError("Symbols are created only by symbolic execution (symbolic_step)")
        _integer(self.src_rank, "symbol.src_rank")
        _integer(self.dst_rank, "symbol.dst_rank")
        if self.kind is not None and self.kind not in _SYMBOL_KINDS:
            raise SemanticError("unknown symbolic transfer kind " + str(self.kind))
        if self.kind != "copy" and self.src_rank == self.dst_rank:
            raise SemanticError("symbol ranks must be distinct")
        if self.operator is not None and self.operator not in _TRANSFER_KINDS:
            raise SemanticError("symbol operator must be SEND or REDUCE")
        if not isinstance(self.ready_time, Term):
            object.__setattr__(self, "ready_time", _symtime(self.ready_time, "symbol.ready_time"))

    def _key(self):
        ready = id(self.ready_time) if isinstance(self.ready_time, Term) else float(self.ready_time)
        return (self.symbol_id, self.src_rank, self.dst_rank, ready, self.stage_id, self.operator)

    def __eq__(self, other):
        return isinstance(other, Symbol) and self._key() == other._key()

    def __hash__(self):
        return hash(self._key())

    @property
    def ready_term(self) -> Term:
        return self.ready_time if isinstance(self.ready_time, Term) else self.ready_time.term

    def at_stage(self, stage_id: int, operator: str) -> "Symbol":
        if self.stage_id == stage_id and self.operator == operator:
            return self
        return _new_symbol(self.src_rank, self.dst_rank, self.ready_time, stage_id, operator,
                           self.symbol_id, self.kind, self.channel, self.guard)


def symbolic_step(src_rank: int, dst_rank: int, ready_time, *, previous: Optional[Symbol] = None,
                  stage_id: Optional[int] = None, operator: Optional[str] = None,
                  symbol_id: Optional[str] = None) -> Symbol:
    """Symbolically execute one transfer step of a schedule path.

    ready(step) = max(ready(previous step), decided ready time of this step);
    the concrete value is the decided ready time (validated to be monotone).
    """
    value = _time(float(ready_time) if isinstance(ready_time, SymValue) else ready_time, "symbol.ready_time")
    decided = ready_time.term if isinstance(ready_time, SymValue) else SCHEDULE_TERMS.const(value)
    if previous is not None:
        term = SCHEDULE_TERMS.max((("previous", previous.ready_term), ("decided", decided)))
    else:
        term = decided
    return _new_symbol(src_rank, dst_rank, SymValue(value, term), stage_id, operator, symbol_id)


def _executor_symbol(symbol_id, src_rank, dst_rank, ready_time, kind, channel, stage) -> Symbol:
    """Constructor used by the lowered-program symbolic executor."""
    return _new_symbol(src_rank, dst_rank, ready_time, stage, "REDUCE" if kind == "rrc" else "SEND",
                       symbol_id, kind, channel)


# --------------------------------------------------------------------------
# s, pd, t
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SliceVec:
    slice_id: int
    slice_size: int
    stage: int

    def __post_init__(self):
        _integer(self.slice_id, "atom.slice_id")
        _integer(self.slice_size, "atom.slice_size_bytes", minimum=1)
        _integer(self.stage, "atom.stage")


@dataclass(frozen=True)
class PathDAG:
    """Union of Symbols from the source rank(s) to the current rank (ordered)."""
    symbols: Tuple[Symbol, ...]
    roots: FrozenSet[int]
    current_rank: int

    def __post_init__(self):
        ordered = tuple(dict.fromkeys(self.symbols))
        if not all(isinstance(symbol, Symbol) for symbol in ordered):
            raise SemanticError("a path contains Symbol values only")
        object.__setattr__(self, "symbols", ordered)
        object.__setattr__(self, "roots", frozenset(self.roots))
        if not ordered:
            if self.current_rank not in self.roots:
                raise SemanticError("an empty path must sit at its source rank")
            return
        if self.current_rank not in {symbol.dst_rank for symbol in ordered}:
            raise SemanticError("atom path does not end at the current rank")
        reached, frontier, outgoing = set(self.roots), list(self.roots), {}
        for symbol in ordered:
            outgoing.setdefault(symbol.src_rank, []).append(symbol.dst_rank)
        while frontier:
            for nxt in outgoing.get(frontier.pop(), ()):
                if nxt not in reached:
                    reached.add(nxt)
                    frontier.append(nxt)
        if any(symbol.src_rank not in reached for symbol in ordered):
            raise SemanticError("path contains a transfer not connected to a source rank")

    def union(self, other: "PathDAG", current_rank: int, step: Optional[Symbol] = None) -> "PathDAG":
        return PathDAG(self.symbols + other.symbols + ((step,) if step is not None else ()),
                       self.roots | other.roots, current_rank)

    def extend(self, step: Symbol) -> "PathDAG":
        return PathDAG(self.symbols + (step,), self.roots, step.dst_rank)

    @property
    def hop_count(self) -> int:
        return len(self.symbols)

    @property
    def is_chain(self) -> bool:
        return all(a.dst_rank == b.src_rank for a, b in zip(self.symbols, self.symbols[1:]))


@dataclass(frozen=True)
class TimeVec:
    st_time: object       # SymValue (schedule) or Term (lowered program)
    ed_time: object


@dataclass(frozen=True)
class PathStage:
    """Read-only view of the consecutive Symbols of one logical stage."""
    stage_id: int
    operator: str
    symbols: Tuple[Symbol, ...]

    def __post_init__(self) -> None:
        _integer(self.stage_id, "path_stage.stage_id")
        if not isinstance(self.operator, str) or self.operator not in _TRANSFER_KINDS:
            raise SemanticError("path_stage.operator must be SEND or REDUCE")
        symbols = tuple(self.symbols)
        if not symbols:
            raise SemanticError("path_stage.symbols must not be empty")
        if not all(isinstance(symbol, Symbol) for symbol in symbols):
            raise SemanticError("path_stage.symbols must contain Symbol values")
        object.__setattr__(self, "symbols", tuple(s.at_stage(self.stage_id, self.operator) for s in symbols))

    @property
    def operation_count(self) -> int:
        return len(self.symbols)


@dataclass(frozen=True, eq=False)
class Atom:
    s: SliceVec
    pd: PathDAG
    t: TimeVec
    contributors: FrozenSet[int] = frozenset()
    state_id: str = ""

    def __post_init__(self):
        if not isinstance(self.s, SliceVec) or not isinstance(self.pd, PathDAG) or not isinstance(self.t, TimeVec):
            raise SemanticError("an atom is {s: SliceVec, pd: PathDAG, t: TimeVec}")
        object.__setattr__(self, "contributors", frozenset(self.contributors) or frozenset({self.s.slice_id}))

    # -- construction by symbolic execution of a chain path -----------------
    @classmethod
    def build(cls, slice_id: int, slice_size_bytes: int, path, st_time, ed_time) -> "Atom":
        stages = tuple(path)
        if not stages:
            raise SemanticError("atom.path must not be empty")
        if not all(isinstance(stage, PathStage) for stage in stages):
            raise SemanticError("atom.path must contain PathStage values")
        previous_stage = None
        symbols, previous = [], None
        for stage in stages:
            if previous_stage is not None and stage.stage_id <= previous_stage:
                raise SemanticError("atom path stage IDs must be strictly increasing")
            previous_stage = stage.stage_id
            for symbol in stage.symbols:
                if previous is not None:
                    if previous.dst_rank != symbol.src_rank:
                        raise SemanticError("atom path must be a contiguous rank chain")
                    if float(previous.ready_time) > float(symbol.ready_time):
                        raise SemanticError("atom path ready_time values must be non-decreasing")
                # Re-execute the step symbolically so ready(k) = max(ready(k-1), decided(k)).
                step = symbolic_step(symbol.src_rank, symbol.dst_rank, symbol.ready_time, previous=previous,
                                     stage_id=stage.stage_id, operator=stage.operator, symbol_id=symbol.symbol_id)
                symbols.append(step)
                previous = step
        st = _symtime(st_time, "atom.st_time")
        ed = _symtime(ed_time, "atom.ed_time")
        if float(st) > float(ed):
            raise SemanticError("atom.st_time must not exceed atom.ed_time")
        if float(st) < float(previous.ready_time):
            raise SemanticError("atom.st_time must not precede the current ready_time")
        start = SymValue(float(st), SCHEDULE_TERMS.max((("ready", previous.ready_term), ("decided", st.term))))
        end = SymValue(float(ed), SCHEDULE_TERMS.add(start.term, SCHEDULE_TERMS.const(float(ed) - float(st))))
        pd = PathDAG(tuple(symbols), frozenset({symbols[0].src_rank}), symbols[-1].dst_rank)
        return cls(SliceVec(_integer(slice_id, "atom.slice_id"), _integer(slice_size_bytes, "atom.slice_size_bytes", 1),
                            stages[-1].stage_id), pd, TimeVec(start, end), frozenset({slice_id}))

    def rebuild(self, **changes) -> "Atom":
        """Copy with changed slice_id / slice_size_bytes / path / st_time / ed_time."""
        allowed = {"slice_id", "slice_size_bytes", "path", "st_time", "ed_time"}
        if set(changes) - allowed:
            raise SemanticError("an atom can only be rebuilt from " + ", ".join(sorted(allowed)))
        values = {name: changes.get(name, getattr(self, name)) for name in allowed}
        return Atom.build(**values)

    def with_times(self, st_time, ed_time) -> "Atom":
        return Atom.build(self.slice_id, self.slice_size_bytes, self.path, st_time, ed_time)

    def with_path(self, path) -> "Atom":
        return Atom.build(self.slice_id, self.slice_size_bytes, path, self.st_time, self.ed_time)

    # -- value semantics ----------------------------------------------------
    def _key(self):
        times = tuple(id(v) if isinstance(v, Term) else float(v) for v in (self.t.st_time, self.t.ed_time))
        return (self.s, self.pd.symbols, times, self.contributors)

    def __eq__(self, other):
        return isinstance(other, Atom) and self._key() == other._key()

    def __hash__(self):
        return hash(self._key())

    # -- derived (legacy) views ---------------------------------------------
    @property
    def slice_id(self) -> int:
        return self.s.slice_id

    @property
    def slice_size_bytes(self) -> int:
        return self.s.slice_size

    @property
    def rank(self) -> int:
        return self.pd.current_rank

    @property
    def st_time(self):
        return self.t.st_time

    @property
    def ed_time(self):
        return self.t.ed_time

    @property
    def path(self) -> Tuple[PathStage, ...]:
        groups = []
        for symbol in self.pd.symbols:
            key = (symbol.stage_id if symbol.stage_id is not None else self.s.stage,
                   symbol.operator or "SEND")
            if groups and groups[-1][0] == key:
                groups[-1][1].append(symbol)
            else:
                groups.append((key, [symbol]))
        return tuple(PathStage(stage, operator, tuple(symbols)) for (stage, operator), symbols in groups)

    @property
    def stage_num(self) -> int:
        return len(self.path)

    @property
    def current_symbol(self) -> Symbol:
        return self.pd.symbols[-1]

    def validate_path_prefix(self, current_rank: int, slice_count: Optional[int] = None) -> None:
        _integer(current_rank, "current_rank")
        first_symbol = self.pd.symbols[0]
        if slice_count is not None:
            try:
                expected_source = source_rank(self.slice_id, slice_count)
            except ValueError as error:
                raise SemanticError(str(error)) from error
            if first_symbol.src_rank != expected_source:
                raise SemanticError("atom path does not start at the slice source rank")
        if self.current_symbol.dst_rank != current_rank:
            raise SemanticError("atom path does not end at the current rank")


# --------------------------------------------------------------------------
# Physical transfers and schedules (unchanged contracts)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Transfer:
    transfer_id: str
    kind: str
    src_rank: int
    dst_rank: int
    channel: int
    stage_id: int
    member_slice_ids: frozenset
    atoms: Tuple[Atom, ...]
    st_time: float
    ed_time: float
    predecessor_ids: frozenset

    def __post_init__(self) -> None:
        _identifier(self.transfer_id, "transfer.transfer_id")
        if not isinstance(self.kind, str) or self.kind not in _TRANSFER_KINDS:
            raise SemanticError("transfer.kind must be SEND or REDUCE")
        _integer(self.src_rank, "transfer.src_rank")
        _integer(self.dst_rank, "transfer.dst_rank")
        if self.src_rank == self.dst_rank:
            raise SemanticError("transfer ranks must be distinct")
        _integer(self.channel, "transfer.channel")
        _integer(self.stage_id, "transfer.stage_id")
        member_slice_ids = frozenset(self.member_slice_ids)
        if not member_slice_ids:
            raise SemanticError("transfer.member_slice_ids must not be empty")
        for slice_id in member_slice_ids:
            _integer(slice_id, "transfer.member_slice_ids")
        object.__setattr__(self, "member_slice_ids", member_slice_ids)
        atoms = tuple(self.atoms)
        if not atoms or not all(isinstance(atom, Atom) for atom in atoms):
            raise SemanticError("transfer.atoms must contain Atom values")
        atoms = tuple(sorted(atoms, key=lambda atom: atom.slice_id))
        object.__setattr__(self, "atoms", atoms)
        atom_slice_ids = frozenset(atom.slice_id for atom in atoms)
        if len(atoms) != len(atom_slice_ids) or atom_slice_ids != member_slice_ids:
            raise SemanticError("transfer atoms must match member_slice_ids exactly")
        object.__setattr__(self, "predecessor_ids", frozenset(self.predecessor_ids))
        for predecessor_id in self.predecessor_ids:
            _identifier(predecessor_id, "transfer.predecessor_ids")
        if self.transfer_id in self.predecessor_ids:
            raise SemanticError("transfer must not depend on itself")
        object.__setattr__(self, "st_time", _time(self.st_time, "transfer.st_time"))
        object.__setattr__(self, "ed_time", _time(self.ed_time, "transfer.ed_time"))
        if self.st_time > self.ed_time:
            raise SemanticError("transfer.st_time must not exceed transfer.ed_time")
        self._validate_atoms()

    def _validate_atoms(self) -> None:
        size_bytes = self.atoms[0].slice_size_bytes
        for atom in self.atoms:
            if atom.slice_size_bytes != size_bytes:
                raise SemanticError("transfer atoms must have equal slice sizes")
            # Atoms carry their own t = (st, ed); it may not start before the
            # physical transfer that delivers it.
            if float(atom.st_time) + 1e-9 < self.st_time:
                raise SemanticError("an atom cannot start before its physical transfer")
            current_stage = atom.path[-1]
            current_symbol = atom.current_symbol
            if (
                current_stage.stage_id != self.stage_id
                or current_stage.operator != self.kind
                or current_symbol.src_rank != self.src_rank
                or current_symbol.dst_rank != self.dst_rank
            ):
                raise SemanticError("transfer atom must end with the physical operation")
            atom.validate_path_prefix(current_rank=self.dst_rank)

    @property
    def physical_bytes(self) -> int:
        return self.atoms[0].slice_size_bytes

    def is_forbidden(self, forbidden: Collection[ForbiddenTransfer]) -> bool:
        return any(
            item.slice_id in self.member_slice_ids
            and item.src_rank == self.src_rank
            and item.dst_rank == self.dst_rank
            and item.stage_id == self.stage_id
            for item in forbidden
        )


@dataclass(frozen=True)
class Schedule:
    schedule_id: str
    transfers: Tuple[Transfer, ...]
    final_state_ids: Tuple[str, ...]
    rank_count: int
    slice_count: int
    slice_size_bytes: int
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        _identifier(self.schedule_id, "schedule.schedule_id")
        _integer(self.rank_count, "schedule.rank_count", minimum=1)
        _integer(self.slice_count, "schedule.slice_count", minimum=1)
        _integer(self.slice_size_bytes, "schedule.slice_size_bytes", minimum=1)
        transfers = tuple(self.transfers)
        if not all(isinstance(transfer, Transfer) for transfer in transfers):
            raise SemanticError("schedule.transfers must contain Transfer values")
        object.__setattr__(self, "transfers", transfers)
        transfer_ids = tuple(transfer.transfer_id for transfer in transfers)
        if len(transfer_ids) != len(set(transfer_ids)):
            raise SemanticError("schedule transfer IDs must be unique")
        final_state_ids = tuple(self.final_state_ids)
        for state_id in final_state_ids:
            _identifier(state_id, "schedule.final_state_ids")
        if len(final_state_ids) != len(set(final_state_ids)):
            raise SemanticError("schedule final state IDs must be unique")
        object.__setattr__(self, "final_state_ids", final_state_ids)
        if not isinstance(self.metadata, Mapping):
            raise SemanticError("schedule.metadata must be a mapping")
        object.__setattr__(self, "metadata", _freeze(self.metadata))
        self._validate_transfers()

    def _validate_transfers(self) -> None:
        path_scope = self.metadata.get("path_scope", "global")
        if path_scope not in {"global", "stage_suffix"}:
            raise SemanticError("schedule.metadata.path_scope must be global or stage_suffix")
        transfer_ids = frozenset(transfer.transfer_id for transfer in self.transfers)
        transfers_by_id = {transfer.transfer_id: transfer for transfer in self.transfers}
        path_roots = {}
        if path_scope == "stage_suffix":
            raw_path_roots = self.metadata.get("path_roots")
            if not isinstance(raw_path_roots, Mapping):
                raise SemanticError("stage_suffix schedule requires path_roots metadata")
            path_roots = dict(raw_path_roots)
            if set(path_roots) != transfer_ids:
                raise SemanticError("stage_suffix path_roots must cover every transfer exactly")
            for transfer_id, root in path_roots.items():
                _identifier(transfer_id, "schedule.metadata.path_roots")
                roots = root if isinstance(root, Mapping) else None
                if roots is not None:
                    roots = dict(roots)
                    if set(roots) != set(transfers_by_id[transfer_id].member_slice_ids):
                        raise SemanticError("per-atom path roots must cover transfer members")
                    values = roots.values()
                else:
                    values = (root,)
                for value in values:
                    _integer(value, "schedule.metadata.path_roots")
                    if value >= self.rank_count:
                        raise SemanticError("stage_suffix path root is outside the rank range")
        global_slice_count = self.rank_count * self.slice_count
        for transfer in self.transfers:
            if transfer.src_rank >= self.rank_count or transfer.dst_rank >= self.rank_count:
                raise SemanticError("transfer rank is outside the schedule rank range")
            if not transfer.predecessor_ids <= transfer_ids:
                raise SemanticError("transfer predecessor is missing from the schedule")
            for atom in transfer.atoms:
                if atom.slice_id >= global_slice_count:
                    raise SemanticError("atom slice_id is outside the global slice range")
                if atom.slice_size_bytes != self.slice_size_bytes:
                    raise SemanticError("atom slice size does not match the schedule slice size")
                if path_scope == "stage_suffix":
                    root = path_roots[transfer.transfer_id]
                    expected = root[atom.slice_id] if isinstance(root, Mapping) else root
                    if atom.path[0].symbols[0].src_rank != expected:
                        raise SemanticError("stage_suffix atom does not start at its declared root")
                atom.validate_path_prefix(
                    current_rank=transfer.dst_rank,
                    slice_count=(self.slice_count if path_scope == "global" else None),
                )
