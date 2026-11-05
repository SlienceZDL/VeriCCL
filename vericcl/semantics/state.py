import math
from dataclasses import dataclass, replace
from typing import Dict, FrozenSet, Iterable, Mapping, Optional, Tuple

from vericcl.errors import SemanticError
from vericcl.semantics.atom import PathStage
from vericcl.semantics.slice import logical_slice_index


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


def _contributor_set(value: object, field: str) -> FrozenSet[int]:
    try:
        contributors = frozenset(value)
    except TypeError as error:
        raise SemanticError("{} must be an iterable of slice IDs".format(field)) from error
    if not contributors:
        raise SemanticError("{} must not be empty".format(field))
    for slice_id in contributors:
        _integer(slice_id, field)
    return contributors


@dataclass(frozen=True)
class PayloadState:
    state_id: str
    version: int
    rank: int
    logical_address: int
    contributors: FrozenSet[int]
    ready_time: float
    active: bool
    member_paths: Tuple[Tuple[int, Tuple[PathStage, ...]], ...]

    def __post_init__(self) -> None:
        _identifier(self.state_id, "payload_state.state_id")
        _integer(self.version, "payload_state.version")
        _integer(self.rank, "payload_state.rank")
        _integer(self.logical_address, "payload_state.logical_address")
        contributors = _contributor_set(
            self.contributors,
            "payload_state.contributors",
        )
        object.__setattr__(self, "contributors", contributors)
        object.__setattr__(
            self,
            "ready_time",
            _time(self.ready_time, "payload_state.ready_time"),
        )
        if not isinstance(self.active, bool):
            raise SemanticError("payload_state.active must be a boolean")
        member_paths = self._normalized_member_paths()
        object.__setattr__(self, "member_paths", member_paths)

    def _normalized_member_paths(
        self,
    ) -> Tuple[Tuple[int, Tuple[PathStage, ...]], ...]:
        try:
            raw_paths = tuple(self.member_paths)
        except TypeError as error:
            raise SemanticError(
                "payload_state.member_paths must be iterable"
            ) from error
        normalized = []
        seen = set()
        for item in raw_paths:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise SemanticError(
                    "payload_state.member_paths entries must contain slice_id and path"
                )
            slice_id = _integer(item[0], "payload_state.member_paths.slice_id")
            if slice_id in seen:
                raise SemanticError("payload_state.member_paths slice IDs must be unique")
            seen.add(slice_id)
            try:
                path = tuple(item[1])
            except TypeError as error:
                raise SemanticError(
                    "payload_state member path must be iterable"
                ) from error
            if not all(isinstance(stage, PathStage) for stage in path):
                raise SemanticError(
                    "payload_state member path must contain PathStage values"
                )
            normalized.append((slice_id, path))
        if seen != set(self.contributors):
            raise SemanticError(
                "payload_state.member_paths must match contributors exactly"
            )
        return tuple(sorted(normalized, key=lambda item: item[0]))


def initial_payload_states(
    rank_count: int,
    slice_count: int,
) -> Tuple[PayloadState, ...]:
    normalized_rank_count = _integer(rank_count, "rank_count", minimum=1)
    normalized_slice_count = _integer(slice_count, "slice_count", minimum=1)
    states = []
    for rank in range(normalized_rank_count):
        for address in range(normalized_slice_count):
            slice_id = rank * normalized_slice_count + address
            states.append(
                PayloadState(
                    state_id="state-r{}-a{}-v0".format(rank, address),
                    version=0,
                    rank=rank,
                    logical_address=logical_slice_index(
                        slice_id,
                        normalized_slice_count,
                    ),
                    contributors=frozenset({slice_id}),
                    ready_time=0.0,
                    active=True,
                    member_paths=((slice_id, ()),),
                )
            )
    return tuple(states)


class PayloadLedger:
    def __init__(self, states: Iterable[PayloadState] = ()) -> None:
        self._states: Dict[str, PayloadState] = {}
        self._next_versions: Dict[Tuple[int, int], int] = {}
        self._active_aggregates: Dict[Tuple[int, int], str] = {}
        self._inactive_ids = set()
        self._incomplete_outbound_counts: Dict[str, int] = {}
        for state in states:
            self._register(state)

    @property
    def inactive_ids(self) -> FrozenSet[str]:
        return frozenset(self._inactive_ids)

    @property
    def states(self) -> Tuple[PayloadState, ...]:
        return tuple(self._states[state_id] for state_id in sorted(self._states))

    def state(self, state_id: str) -> PayloadState:
        _identifier(state_id, "state_id")
        try:
            return self._states[state_id]
        except KeyError as error:
            raise SemanticError("unknown state version: {}".format(state_id)) from error

    def _active_state(self, state_id: str) -> PayloadState:
        state = self.state(state_id)
        if not state.active:
            raise SemanticError("state version is inactive: {}".format(state_id))
        return state

    def _register(self, state: PayloadState) -> None:
        if not isinstance(state, PayloadState):
            raise SemanticError("ledger states must be PayloadState values")
        if state.state_id in self._states:
            raise SemanticError("state IDs must be unique")
        key = (state.rank, state.logical_address)
        if state.active and len(state.contributors) > 1:
            if key in self._active_aggregates:
                raise SemanticError(
                    "active aggregate already exists at rank and logical address"
                )
            self._active_aggregates[key] = state.state_id
        self._states[state.state_id] = state
        self._next_versions[key] = max(
            self._next_versions.get(key, 0),
            state.version + 1,
        )
        if not state.active:
            self._inactive_ids.add(state.state_id)

    def _deactivate(self, state: PayloadState) -> None:
        inactive = replace(state, active=False)
        self._states[state.state_id] = inactive
        self._inactive_ids.add(state.state_id)
        key = (state.rank, state.logical_address)
        if self._active_aggregates.get(key) == state.state_id:
            del self._active_aggregates[key]

    def _target_is_available(
        self,
        rank: int,
        logical_address: int,
        consumed_ids: FrozenSet[str] = frozenset(),
    ) -> None:
        existing = self._active_aggregates.get((rank, logical_address))
        if existing is not None and existing not in consumed_ids:
            raise SemanticError(
                "active aggregate already exists at rank and logical address"
            )

    def _new_state(
        self,
        *,
        rank: int,
        logical_address: int,
        contributors: FrozenSet[int],
        ready_time: float,
        member_paths: Tuple[Tuple[int, Tuple[PathStage, ...]], ...],
    ) -> PayloadState:
        key = (rank, logical_address)
        version = self._next_versions.get(key, 0)
        while True:
            state_id = "state-r{}-a{}-v{}".format(rank, logical_address, version)
            if state_id not in self._states:
                break
            version += 1
        return PayloadState(
            state_id=state_id,
            version=version,
            rank=rank,
            logical_address=logical_address,
            contributors=contributors,
            ready_time=ready_time,
            active=True,
            member_paths=member_paths,
        )

    def reduce(
        self,
        left_id: str,
        right_id: str,
        dst_rank: int,
        ready_time: float,
    ) -> PayloadState:
        if left_id == right_id:
            raise SemanticError("REDUCE requires two distinct state versions")
        left = self._active_state(left_id)
        right = self._active_state(right_id)
        normalized_dst = _integer(dst_rank, "dst_rank")
        normalized_ready = _time(ready_time, "ready_time")
        if left.logical_address != right.logical_address:
            raise SemanticError("REDUCE inputs must have the same logical address")
        if left.contributors & right.contributors:
            raise SemanticError("REDUCE contributors must be disjoint")
        if normalized_dst not in {left.rank, right.rank}:
            raise SemanticError("REDUCE destination must own one input state")
        if normalized_ready < max(left.ready_time, right.ready_time):
            raise SemanticError("REDUCE ready_time precedes an input state")
        consumed_ids = frozenset({left.state_id, right.state_id})
        self._target_is_available(
            normalized_dst,
            left.logical_address,
            consumed_ids,
        )
        contributors = left.contributors | right.contributors
        paths: Mapping[int, Tuple[PathStage, ...]] = dict(left.member_paths)
        merged_paths = dict(paths)
        merged_paths.update(dict(right.member_paths))
        output = self._new_state(
            rank=normalized_dst,
            logical_address=left.logical_address,
            contributors=contributors,
            ready_time=normalized_ready,
            member_paths=tuple(sorted(merged_paths.items())),
        )
        self._deactivate(left)
        self._deactivate(right)
        self._register(output)
        return output

    def send(
        self,
        state_id: str,
        dst_rank: int,
        ready_time: float,
        required_contributors: FrozenSet[int],
    ) -> PayloadState:
        state = self.state(state_id)
        if self._incomplete_outbound_counts.get(state_id, 0) >= 1:
            raise SemanticError("incomplete state already sent")
        state = self._active_state(state_id)
        normalized_dst = _integer(dst_rank, "dst_rank")
        if normalized_dst == state.rank:
            raise SemanticError("SEND source and destination ranks must be distinct")
        normalized_ready = _time(ready_time, "ready_time")
        if normalized_ready < state.ready_time:
            raise SemanticError("SEND ready_time precedes the source state")
        required = _contributor_set(
            required_contributors,
            "required_contributors",
        )
        if not state.contributors <= required:
            raise SemanticError("state contains contributors outside required contributors")
        complete = state.contributors == required
        if len(state.contributors) > 1:
            self._target_is_available(normalized_dst, state.logical_address)
        output = self._new_state(
            rank=normalized_dst,
            logical_address=state.logical_address,
            contributors=state.contributors,
            ready_time=normalized_ready,
            member_paths=state.member_paths,
        )
        if not complete:
            self._deactivate(state)
            self._incomplete_outbound_counts[state_id] = 1
        self._register(output)
        return output

    def merge_local(
        self,
        state_id: str,
        local_state_id: str,
        ready_time: float,
    ) -> PayloadState:
        state = self._active_state(state_id)
        local_state = self._active_state(local_state_id)
        if state.rank != local_state.rank:
            raise SemanticError("local merge inputs must be at the same rank")
        if len(local_state.contributors) != 1:
            raise SemanticError("local contribution must contain one slice")
        return self.reduce(
            state_id,
            local_state_id,
            dst_rank=state.rank,
            ready_time=ready_time,
        )
