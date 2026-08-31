from dataclasses import dataclass, replace
from typing import Dict, FrozenSet, Mapping, Optional, Tuple

from vericcl.errors import SemanticError
from vericcl.planner.dual import DualTree, extract_dual_trees
from vericcl.planner.model import StageInterface
from vericcl.semantics.atom import Atom, PathStage, Schedule, Symbol, Transfer
from vericcl.semantics.collective import CollectiveKind, CollectiveSpec
from vericcl.topology.model import LaneKey


_REDUCTION_KINDS = frozenset(
    {CollectiveKind.REDUCE, CollectiveKind.REDUCE_SCATTER}
)


@dataclass(frozen=True)
class _Draft:
    transfer_id: str
    source_id: str
    tree: DualTree
    src_rank: int
    dst_rank: int
    channel: int
    stage_id: int
    member_slice_ids: FrozenSet[int]
    duration: float
    semantic_predecessors: FrozenSet[str]
    resource_slots: Mapping[str, int]


@dataclass(frozen=True)
class _TimedDraft:
    draft: _Draft
    st_time: float
    ed_time: float
    semantic_ready_time: float
    predecessor_ids: FrozenSet[str]


@dataclass(frozen=True)
class _AggregateState:
    state_id: str
    rank: int
    logical_position: int
    contributors: FrozenSet[int]
    version: int
    producer_id: Optional[str]


def _resource_slots(schedule: Schedule, transfer_id: str) -> Mapping[str, int]:
    values = schedule.metadata.get("resource_slots", {})
    if not isinstance(values, Mapping):
        raise SemanticError("resource_slots metadata must be a mapping")
    slots = values.get(transfer_id, {})
    if not isinstance(slots, Mapping):
        raise SemanticError("transfer resource slots must be a mapping")
    return dict(slots)


def _drafts(
    schedule: Schedule,
    trees: Tuple[DualTree, ...],
    routing_only: bool = False,
) -> Tuple[_Draft, ...]:
    drafts = []
    for tree in trees:
        identifiers = {
            transfer.transfer_id: "reduce-{}".format(transfer.transfer_id)
            for transfer in tree.transfers
        }
        for transfer in tree.transfers:
            child_transfers = [
                child
                for child in tree.transfers
                if child.src_rank == transfer.dst_rank
            ]
            drafts.append(
                _Draft(
                    transfer_id=identifiers[transfer.transfer_id],
                    source_id=transfer.transfer_id,
                    tree=tree,
                    src_rank=transfer.dst_rank,
                    dst_rank=transfer.src_rank,
                    channel=0 if routing_only else transfer.channel,
                    stage_id=transfer.stage_id,
                    member_slice_ids=transfer.member_slice_ids,
                    duration=transfer.ed_time - transfer.st_time,
                    semantic_predecessors=frozenset(
                        identifiers[child.transfer_id]
                        for child in child_transfers
                    ),
                    resource_slots=(
                        {}
                        if routing_only
                        else _resource_slots(
                            schedule,
                            transfer.transfer_id,
                        )
                    ),
                )
            )
    return tuple(sorted(drafts, key=lambda item: item.transfer_id))


def _schedule_drafts(
    drafts: Tuple[_Draft, ...],
    routing_only: bool = False,
) -> Tuple[_TimedDraft, ...]:
    pending = {draft.transfer_id: draft for draft in drafts}
    timed = {}
    lane_ready = {}
    lane_last = {}
    resource_ready = {}
    resource_last = {}
    while pending:
        ready = [
            draft
            for draft in pending.values()
            if draft.semantic_predecessors <= set(timed)
        ]
        if not ready:
            raise SemanticError("reversed REDUCE dependencies contain a cycle")
        choices = []
        for draft in ready:
            semantic_ready = max(
                (
                    timed[predecessor].ed_time
                    for predecessor in draft.semantic_predecessors
                ),
                default=0.0,
            )
            lane = LaneKey(draft.src_rank, draft.dst_rank, draft.channel)
            start = semantic_ready
            if not routing_only:
                start = max(
                    [semantic_ready, lane_ready.get(lane, 0.0)]
                    + [
                        resource_ready.get((resource_id, slot), 0.0)
                        for resource_id, slot in draft.resource_slots.items()
                    ]
                )
            choices.append((start, draft.transfer_id, semantic_ready, lane, draft))
        start, _, semantic_ready, lane, draft = min(choices)
        predecessors = set(draft.semantic_predecessors)
        if not routing_only:
            if lane in lane_last:
                predecessors.add(lane_last[lane])
            for resource_id, slot in draft.resource_slots.items():
                key = (resource_id, slot)
                if key in resource_last:
                    predecessors.add(resource_last[key])
        result = _TimedDraft(
            draft=draft,
            st_time=start,
            ed_time=start + draft.duration,
            semantic_ready_time=semantic_ready,
            predecessor_ids=frozenset(predecessors),
        )
        timed[draft.transfer_id] = result
        del pending[draft.transfer_id]
        if not routing_only:
            lane_ready[lane] = result.ed_time
            lane_last[lane] = draft.transfer_id
            for resource_id, slot in draft.resource_slots.items():
                key = (resource_id, slot)
                resource_ready[key] = result.ed_time
                resource_last[key] = draft.transfer_id
    return tuple(timed[key] for key in sorted(timed))


def _tree_key(tree: DualTree) -> tuple:
    return (
        tree.root_rank,
        tree.target_offset,
        tuple(sorted(tree.contributors)),
    )


def _chain_routing_accumulators(
    drafts: Tuple[_Draft, ...],
) -> Tuple[_Draft, ...]:
    groups: Dict[tuple, list] = {}
    for draft in drafts:
        groups.setdefault(
            (_tree_key(draft.tree), draft.dst_rank),
            [],
        ).append(draft)
    chained = []
    for key in sorted(groups):
        previous = None
        for draft in sorted(
            groups[key],
            key=lambda item: item.transfer_id,
        ):
            predecessors = set(draft.semantic_predecessors)
            if previous is not None:
                predecessors.add(previous)
            chained.append(
                replace(
                    draft,
                    semantic_predecessors=frozenset(predecessors),
                )
            )
            previous = draft.transfer_id
    return tuple(sorted(chained, key=lambda item: item.transfer_id))


def _logical_position(tree: DualTree, slice_count: int) -> int:
    positions = {member % slice_count for member in tree.contributors}
    if len(positions) != 1:
        raise SemanticError(
            "dual tree contributors must share one logical position"
        )
    return next(iter(positions))


def _contributors_identity(contributors: FrozenSet[int]) -> str:
    return ".".join(
        "{:08d}".format(value) for value in sorted(contributors)
    )


def _aggregate_state_id(
    tree: DualTree,
    logical_position: int,
    rank: int,
    version: int,
    contributors: FrozenSet[int],
) -> str:
    return (
        "aggregate-t{:08d}-q{}-r{:08d}-a{:08d}-v{:08d}-c{}"
    ).format(
        tree.root_rank,
        _contributors_identity(tree.contributors),
        rank,
        logical_position,
        version,
        _contributors_identity(contributors),
    )


def _routing_aggregate_metadata(
    timed: Tuple[_TimedDraft, ...],
    trees: Tuple[DualTree, ...],
    slice_count: int,
) -> Tuple[Mapping[str, object], Mapping[str, object], Mapping[tuple, Optional[str]]]:
    by_id = {item.draft.transfer_id: item for item in timed}
    pending = dict(by_id)
    completed = set()
    active = {}
    states: Dict[str, _AggregateState] = {}
    transitions = {}
    consumed_state_ids = set()

    def register(state: _AggregateState) -> _AggregateState:
        existing = states.get(state.state_id)
        if existing is not None and existing != state:
            raise SemanticError("aggregate state ID has conflicting content")
        states[state.state_id] = state
        return state

    def initial_state(tree: DualTree, rank: int) -> _AggregateState:
        logical_position = _logical_position(tree, slice_count)
        contributors = tree.members_by_rank.get(rank, frozenset())
        if not contributors:
            raise SemanticError(
                "routing-only REDUCE rank has no local accumulator state"
            )
        state = _AggregateState(
            state_id=_aggregate_state_id(
                tree,
                logical_position,
                rank,
                0,
                contributors,
            ),
            rank=rank,
            logical_position=logical_position,
            contributors=contributors,
            version=0,
            producer_id=None,
        )
        return register(state)

    while pending:
        ready = [
            item
            for item in pending.values()
            if item.draft.semantic_predecessors <= completed
        ]
        if not ready:
            raise SemanticError("aggregate state dependencies contain a cycle")
        item = min(
            ready,
            key=lambda value: (value.st_time, value.draft.transfer_id),
        )
        draft = item.draft
        tree_identity = _tree_key(draft.tree)
        source_key = (tree_identity, draft.src_rank)
        accumulator_key = (tree_identity, draft.dst_rank)
        source = active.get(source_key)
        if source is None:
            source = initial_state(draft.tree, draft.src_rank)
            active[source_key] = source
        accumulator = active.get(accumulator_key)
        if accumulator is None:
            accumulator = initial_state(draft.tree, draft.dst_rank)
            active[accumulator_key] = accumulator
        if source.contributors != draft.member_slice_ids:
            raise SemanticError(
                "REDUCE source state does not match transfer contributors"
            )
        if source.contributors & accumulator.contributors:
            raise SemanticError("REDUCE state contributors must be disjoint")
        consumed = (source.state_id, accumulator.state_id)
        if any(state_id in consumed_state_ids for state_id in consumed):
            raise SemanticError(
                "aggregate state version is consumed more than once"
            )
        consumed_state_ids.update(consumed)
        del active[source_key]
        del active[accumulator_key]
        produced_contributors = (
            source.contributors | accumulator.contributors
        )
        produced = register(
            _AggregateState(
                state_id=_aggregate_state_id(
                    draft.tree,
                    source.logical_position,
                    draft.dst_rank,
                    accumulator.version + 1,
                    produced_contributors,
                ),
                rank=draft.dst_rank,
                logical_position=source.logical_position,
                contributors=produced_contributors,
                version=accumulator.version + 1,
                producer_id=draft.transfer_id,
            )
        )
        active[accumulator_key] = produced
        transitions[draft.transfer_id] = {
            "consumed_state_ids": consumed,
            "produced_state_id": produced.state_id,
        }
        completed.add(draft.transfer_id)
        del pending[draft.transfer_id]

    terminals = {}
    for tree in trees:
        tree_identity = _tree_key(tree)
        key = (tree_identity, tree.root_rank)
        terminal = active.get(key)
        if terminal is None:
            terminal = initial_state(tree, tree.root_rank)
        if terminal.contributors != tree.contributors:
            raise SemanticError(
                "terminal aggregate state does not contain all contributors"
            )
        terminals[tree_identity] = terminal.producer_id
    state_metadata = {
        state_id: {
            "rank": state.rank,
            "logical_position": state.logical_position,
            "contributors": tuple(sorted(state.contributors)),
            "producer_id": state.producer_id,
        }
        for state_id, state in sorted(states.items())
    }
    return transitions, state_metadata, terminals


def _member_sources(tree: DualTree) -> Mapping[int, int]:
    return {
        member: rank
        for rank, members in tree.local_members
        for member in members
    }


def _member_symbols(
    item: _TimedDraft,
    timed_by_tree_edge: Mapping[Tuple[int, FrozenSet[int], int, int], _TimedDraft],
    member: int,
) -> Tuple[Symbol, ...]:
    parents = item.draft.tree.parents
    rank = _member_sources(item.draft.tree)[member]
    symbols = []
    while rank != item.draft.dst_rank:
        if rank not in parents:
            raise SemanticError("member path does not reach its REDUCE target")
        parent = parents[rank]
        timed = timed_by_tree_edge[
            (
                item.draft.tree.root_rank,
                item.draft.tree.contributors,
                rank,
                parent,
            )
        ]
        symbols.append(Symbol(rank, parent, timed.semantic_ready_time))
        rank = parent
    return tuple(symbols)


def _final_output_key(rank: int, offset: int) -> str:
    return "r{:08d}-o{:08d}".format(rank, offset)


def reverse_allgather_schedule(
    ag_schedule: Schedule,
    reduce_spec: CollectiveSpec,
    target_interface: StageInterface,
) -> Schedule:
    if not isinstance(ag_schedule, Schedule):
        raise SemanticError("ag_schedule must be a Schedule")
    if not isinstance(reduce_spec, CollectiveSpec):
        raise SemanticError("reduce_spec must be a CollectiveSpec")
    if reduce_spec.kind not in _REDUCTION_KINDS or not reduce_spec.reduction_op:
        raise SemanticError("dual conversion requires a reduction collective")
    if not isinstance(target_interface, StageInterface):
        raise SemanticError("target_interface must be a StageInterface")
    trees = extract_dual_trees(ag_schedule, target_interface)
    routing_only = ag_schedule.metadata.get("routing_only") is True
    drafts = _drafts(ag_schedule, trees, routing_only=routing_only)
    if routing_only:
        drafts = _chain_routing_accumulators(drafts)
    timed = _schedule_drafts(
        drafts,
        routing_only=routing_only,
    )
    aggregate_consumptions = {}
    aggregate_states = {}
    terminal_producers = {}
    if routing_only:
        (
            aggregate_consumptions,
            aggregate_states,
            terminal_producers,
        ) = _routing_aggregate_metadata(
            timed,
            trees,
            ag_schedule.slice_count,
        )
    timed_by_tree_edge = {
        (
            item.draft.tree.root_rank,
            item.draft.tree.contributors,
            item.draft.src_rank,
            item.draft.dst_rank,
        ): item
        for item in timed
    }
    transfers = []
    path_roots = {}
    semantic_predecessors = {}
    semantic_contributors = {}
    tree_contributors = {}
    resource_slots = {}
    path_prefixes = {}
    for item in sorted(
        timed,
        key=lambda value: (value.st_time, value.draft.transfer_id),
    ):
        draft = item.draft
        member_sources = _member_sources(draft.tree)
        atoms = tuple(
            Atom(
                slice_id=member,
                slice_size_bytes=ag_schedule.slice_size_bytes,
                path=(
                    PathStage(
                        stage_id=draft.stage_id,
                        operator="REDUCE",
                        symbols=_member_symbols(
                            item,
                            timed_by_tree_edge,
                            member,
                        ),
                    ),
                ),
                st_time=item.st_time,
                ed_time=item.ed_time,
            )
            for member in sorted(draft.member_slice_ids)
        )
        transfers.append(
            Transfer(
                transfer_id=draft.transfer_id,
                kind="REDUCE",
                src_rank=draft.src_rank,
                dst_rank=draft.dst_rank,
                channel=draft.channel,
                stage_id=draft.stage_id,
                member_slice_ids=draft.member_slice_ids,
                atoms=atoms,
                st_time=item.st_time,
                ed_time=item.ed_time,
                predecessor_ids=item.predecessor_ids,
            )
        )
        path_roots[draft.transfer_id] = {
            member: member_sources[member]
            for member in sorted(draft.member_slice_ids)
        }
        semantic_predecessors[draft.transfer_id] = tuple(
            sorted(draft.semantic_predecessors)
        )
        semantic_contributors[draft.transfer_id] = tuple(
            sorted(draft.member_slice_ids)
        )
        tree_contributors[draft.transfer_id] = tuple(
            sorted(draft.tree.contributors)
        )
        resource_slots[draft.transfer_id] = dict(draft.resource_slots)
        path_prefixes[draft.transfer_id] = {
            atom.slice_id: tuple(
                (symbol.src_rank, symbol.dst_rank)
                for stage in atom.path
                for symbol in stage.symbols
            )
            for atom in atoms
        }
    final_outputs = {
        _final_output_key(slot.rank, slot.offset): tuple(sorted(contributors))
        for slot, contributors in target_interface.values.items()
    }
    final_ready_times = {}
    final_dependencies = {}
    for tree in trees:
        key = _final_output_key(tree.root_rank, tree.target_offset)
        if routing_only:
            terminal = terminal_producers[_tree_key(tree)]
            dependencies = () if terminal is None else (terminal,)
            ready = (
                0.0
                if terminal is None
                else next(
                    item.ed_time
                    for item in timed
                    if item.draft.transfer_id == terminal
                )
            )
        else:
            dependencies = ()
            ready = max(
                (
                    item.ed_time
                    for item in timed
                    if item.draft.tree == tree
                    and item.draft.dst_rank == tree.root_rank
                ),
                default=0.0,
            )
        final_ready_times[key] = ready
        if routing_only:
            final_dependencies[key] = dependencies
    metadata = {
        "path_scope": "stage_suffix",
        "path_roots": path_roots,
        "reduction_dual": False,
        "dual_converted": True,
        "dual_source_schedule_id": ag_schedule.schedule_id,
        "reduction_op": reduce_spec.reduction_op,
        "semantic_predecessors": semantic_predecessors,
        "semantic_contributors": semantic_contributors,
        "tree_contributors": tree_contributors,
        "resource_slots": resource_slots,
        "final_outputs": final_outputs,
        "final_ready_times": final_ready_times,
    }
    if routing_only:
        metadata.update(
            {
                "routing_only": True,
                "path_prefixes": path_prefixes,
                "final_dependencies": final_dependencies,
                "aggregate_consumptions": aggregate_consumptions,
                "aggregate_states": aggregate_states,
            }
        )
    return Schedule(
        schedule_id="{}-reduce".format(ag_schedule.schedule_id),
        transfers=tuple(transfers),
        final_state_ids=tuple(
            "reduce-{}".format(key) for key in sorted(final_outputs)
        ),
        rank_count=ag_schedule.rank_count,
        slice_count=ag_schedule.slice_count,
        slice_size_bytes=ag_schedule.slice_size_bytes,
        metadata=metadata,
    )
