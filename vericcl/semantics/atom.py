from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Collection, Mapping, Optional, Tuple

from vericcl.errors import SemanticError
from vericcl.semantics.slice import source_rank

if TYPE_CHECKING:
    from vericcl.input.models import ForbiddenTransfer


_TRANSFER_KINDS = frozenset({"REDUCE", "SEND"})


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


@dataclass(frozen=True)
class Symbol:
    src_rank: int
    dst_rank: int
    ready_time: float

    def __post_init__(self) -> None:
        _integer(self.src_rank, "symbol.src_rank")
        _integer(self.dst_rank, "symbol.dst_rank")
        if self.src_rank == self.dst_rank:
            raise SemanticError("symbol ranks must be distinct")
        object.__setattr__(
            self,
            "ready_time",
            _time(self.ready_time, "symbol.ready_time"),
        )


@dataclass(frozen=True)
class PathStage:
    stage_id: int
    operator: str
    symbols: Tuple[Symbol, ...]

    def __post_init__(self) -> None:
        _integer(self.stage_id, "path_stage.stage_id")
        if (
            not isinstance(self.operator, str)
            or self.operator not in _TRANSFER_KINDS
        ):
            raise SemanticError("path_stage.operator must be SEND or REDUCE")
        symbols = tuple(self.symbols)
        if not symbols:
            raise SemanticError("path_stage.symbols must not be empty")
        if not all(isinstance(symbol, Symbol) for symbol in symbols):
            raise SemanticError("path_stage.symbols must contain Symbol values")
        object.__setattr__(self, "symbols", symbols)

    @property
    def operation_count(self) -> int:
        return len(self.symbols)


@dataclass(frozen=True)
class Atom:
    slice_id: int
    slice_size_bytes: int
    path: Tuple[PathStage, ...]
    st_time: float
    ed_time: float

    def __post_init__(self) -> None:
        _integer(self.slice_id, "atom.slice_id")
        _integer(self.slice_size_bytes, "atom.slice_size_bytes", minimum=1)
        path = tuple(self.path)
        if not path:
            raise SemanticError("atom.path must not be empty")
        if not all(isinstance(stage, PathStage) for stage in path):
            raise SemanticError("atom.path must contain PathStage values")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "st_time", _time(self.st_time, "atom.st_time"))
        object.__setattr__(self, "ed_time", _time(self.ed_time, "atom.ed_time"))
        if self.st_time > self.ed_time:
            raise SemanticError("atom.st_time must not exceed atom.ed_time")
        self._validate_chain()
        if self.st_time < self.path[-1].symbols[-1].ready_time:
            raise SemanticError("atom.st_time must not precede the current ready_time")

    @property
    def stage_num(self) -> int:
        return len(self.path)

    @property
    def current_symbol(self) -> Symbol:
        return self.path[-1].symbols[-1]

    def _validate_chain(self) -> None:
        previous_stage_id: Optional[int] = None
        previous_symbol: Optional[Symbol] = None
        for stage in self.path:
            if previous_stage_id is not None and stage.stage_id <= previous_stage_id:
                raise SemanticError("atom path stage IDs must be strictly increasing")
            previous_stage_id = stage.stage_id
            for symbol in stage.symbols:
                if previous_symbol is not None:
                    if previous_symbol.dst_rank != symbol.src_rank:
                        raise SemanticError("atom path must be a contiguous rank chain")
                    if previous_symbol.ready_time > symbol.ready_time:
                        raise SemanticError(
                            "atom path ready_time values must be non-decreasing"
                        )
                previous_symbol = symbol

    def validate_path_prefix(
        self,
        current_rank: int,
        slice_count: Optional[int] = None,
    ) -> None:
        _integer(current_rank, "current_rank")
        first_symbol = self.path[0].symbols[0]
        if slice_count is not None:
            try:
                expected_source = source_rank(self.slice_id, slice_count)
            except ValueError as error:
                raise SemanticError(str(error)) from error
            if first_symbol.src_rank != expected_source:
                raise SemanticError(
                    "atom path does not start at the slice source rank"
                )
        if self.current_symbol.dst_rank != current_rank:
            raise SemanticError("atom path does not end at the current rank")


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
            raise SemanticError(
                "transfer atoms must match member_slice_ids exactly"
            )
        object.__setattr__(
            self,
            "predecessor_ids",
            frozenset(self.predecessor_ids),
        )
        for predecessor_id in self.predecessor_ids:
            _identifier(predecessor_id, "transfer.predecessor_ids")
        if self.transfer_id in self.predecessor_ids:
            raise SemanticError("transfer must not depend on itself")
        object.__setattr__(
            self,
            "st_time",
            _time(self.st_time, "transfer.st_time"),
        )
        object.__setattr__(
            self,
            "ed_time",
            _time(self.ed_time, "transfer.ed_time"),
        )
        if self.st_time > self.ed_time:
            raise SemanticError("transfer.st_time must not exceed transfer.ed_time")
        self._validate_atoms()

    def _validate_atoms(self) -> None:
        size_bytes = self.atoms[0].slice_size_bytes
        for atom in self.atoms:
            if atom.slice_size_bytes != size_bytes:
                raise SemanticError("transfer atoms must have equal slice sizes")
            if atom.st_time != self.st_time or atom.ed_time != self.ed_time:
                raise SemanticError("transfer and atom time intervals must match")
            current_stage = atom.path[-1]
            current_symbol = atom.current_symbol
            if (
                current_stage.stage_id != self.stage_id
                or current_stage.operator != self.kind
                or current_symbol.src_rank != self.src_rank
                or current_symbol.dst_rank != self.dst_rank
            ):
                raise SemanticError(
                    "transfer atom must end with the physical operation"
                )
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
        _integer(
            self.slice_size_bytes,
            "schedule.slice_size_bytes",
            minimum=1,
        )
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
            raise SemanticError(
                "schedule.metadata.path_scope must be global or stage_suffix"
            )
        transfer_ids = frozenset(
            transfer.transfer_id for transfer in self.transfers
        )
        path_roots = {}
        if path_scope == "stage_suffix":
            raw_path_roots = self.metadata.get("path_roots")
            if not isinstance(raw_path_roots, Mapping):
                raise SemanticError(
                    "stage_suffix schedule requires path_roots metadata"
                )
            path_roots = dict(raw_path_roots)
            if set(path_roots) != transfer_ids:
                raise SemanticError(
                    "stage_suffix path_roots must cover every transfer exactly"
                )
            for transfer_id, root in path_roots.items():
                _identifier(transfer_id, "schedule.metadata.path_roots")
                _integer(root, "schedule.metadata.path_roots")
                if root >= self.rank_count:
                    raise SemanticError(
                        "stage_suffix path root is outside the rank range"
                    )
        global_slice_count = self.rank_count * self.slice_count
        for transfer in self.transfers:
            if transfer.src_rank >= self.rank_count or transfer.dst_rank >= self.rank_count:
                raise SemanticError("transfer rank is outside the schedule rank range")
            if not transfer.predecessor_ids <= transfer_ids:
                raise SemanticError("transfer predecessor is missing from the schedule")
            for atom in transfer.atoms:
                if atom.slice_id >= global_slice_count:
                    raise SemanticError(
                        "atom slice_id is outside the global slice range"
                    )
                if atom.slice_size_bytes != self.slice_size_bytes:
                    raise SemanticError(
                        "atom slice size does not match the schedule slice size"
                    )
                if (
                    path_scope == "stage_suffix"
                    and atom.path[0].symbols[0].src_rank
                    != path_roots[transfer.transfer_id]
                ):
                    raise SemanticError(
                        "stage_suffix atom does not start at its declared root"
                    )
                atom.validate_path_prefix(
                    current_rank=transfer.dst_rank,
                    slice_count=(
                        self.slice_count if path_scope == "global" else None
                    ),
                )
