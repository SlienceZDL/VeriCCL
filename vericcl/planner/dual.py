from dataclasses import dataclass
from typing import FrozenSet, Mapping, Tuple

from vericcl.errors import SemanticError
from vericcl.planner.model import StageInterface
from vericcl.semantics.atom import Schedule, Transfer


@dataclass(frozen=True)
class DualTree:
    root_rank: int
    contributors: FrozenSet[int]
    transfers: Tuple[Transfer, ...]
    parent_by_rank: Tuple[Tuple[int, int], ...]
    local_members: Tuple[Tuple[int, FrozenSet[int]], ...]
    target_offset: int

    @property
    def parents(self) -> Mapping[int, int]:
        return dict(self.parent_by_rank)

    @property
    def members_by_rank(self) -> Mapping[int, FrozenSet[int]]:
        return dict(self.local_members)


def _transfer_root(schedule: Schedule, transfer: Transfer) -> int:
    roots = schedule.metadata["path_roots"][transfer.transfer_id]
    if isinstance(roots, Mapping):
        values = set(roots.values())
        if len(values) != 1:
            raise SemanticError("virtual dual transfer must have one tree root")
        return next(iter(values))
    return roots


def _tree_contributors(
    schedule: Schedule,
    transfer: Transfer,
) -> FrozenSet[int]:
    values = schedule.metadata.get("tree_contributors")
    if not isinstance(values, Mapping) or transfer.transfer_id not in values:
        raise SemanticError("dual schedule requires tree_contributors metadata")
    contributors = frozenset(values[transfer.transfer_id])
    if not contributors:
        raise SemanticError("dual tree contributors must not be empty")
    return contributors


def _validate_tree(
    root: int,
    contributors: FrozenSet[int],
    transfers: Tuple[Transfer, ...],
    target_interface: StageInterface,
) -> DualTree:
    by_edge = {}
    parent_by_rank = {}
    children = {}
    for transfer in transfers:
        edge = (transfer.src_rank, transfer.dst_rank)
        if edge in by_edge:
            raise SemanticError("dual tree contains a duplicate edge")
        if transfer.dst_rank == root:
            raise SemanticError("virtual dual tree enters its root")
        if transfer.dst_rank in parent_by_rank:
            raise SemanticError("virtual dual tree gives a rank multiple parents")
        by_edge[edge] = transfer
        parent_by_rank[transfer.dst_rank] = transfer.src_rank
        children.setdefault(transfer.src_rank, []).append(transfer.dst_rank)
    for rank in parent_by_rank:
        visited = {rank}
        current = rank
        while current != root:
            if current not in parent_by_rank:
                raise SemanticError("virtual dual tree is disconnected")
            current = parent_by_rank[current]
            if current in visited:
                raise SemanticError("virtual dual tree contains a cycle")
            visited.add(current)
    subtree_members = {root: contributors}
    for rank, parent in parent_by_rank.items():
        subtree_members[rank] = by_edge[(parent, rank)].member_slice_ids
    local_members = {}
    for rank, subtree in subtree_members.items():
        child_sets = [
            subtree_members[child] for child in children.get(rank, ())
        ]
        union = set()
        for child_set in child_sets:
            if union & child_set or not child_set <= subtree:
                raise SemanticError(
                    "dual subtree contributors must be disjoint and nested"
                )
            union.update(child_set)
        local_members[rank] = frozenset(subtree - union)
    owners = {}
    for rank, members in local_members.items():
        for member in members:
            if member in owners:
                raise SemanticError("dual contributor has multiple source ranks")
            owners[member] = rank
    if set(owners) != set(contributors):
        raise SemanticError("dual tree does not account for every contributor")
    targets = [
        slot
        for slot, values in target_interface.values.items()
        if slot.rank == root and values == contributors
    ]
    if len(targets) != 1:
        raise SemanticError(
            "dual target contributors do not match one tree root"
        )
    return DualTree(
        root_rank=root,
        contributors=contributors,
        transfers=tuple(sorted(transfers, key=lambda item: item.transfer_id)),
        parent_by_rank=tuple(sorted(parent_by_rank.items())),
        local_members=tuple(sorted(local_members.items())),
        target_offset=targets[0].offset,
    )


def extract_dual_trees(
    schedule: Schedule,
    target_interface: StageInterface,
) -> Tuple[DualTree, ...]:
    if not isinstance(schedule, Schedule):
        raise SemanticError("schedule must be a Schedule")
    if not isinstance(target_interface, StageInterface):
        raise SemanticError("target_interface must be a StageInterface")
    if schedule.metadata.get("path_scope") != "stage_suffix":
        raise SemanticError("dual schedule must use stage_suffix paths")
    if schedule.metadata.get("reduction_dual") is not True:
        raise SemanticError("schedule is not marked as a reduction dual")
    if not schedule.transfers:
        return tuple(
            DualTree(
                root_rank=slot.rank,
                contributors=contributors,
                transfers=(),
                parent_by_rank=(),
                local_members=((slot.rank, contributors),),
                target_offset=slot.offset,
            )
            for slot, contributors in target_interface.values.items()
        )
    grouped = {}
    for transfer in schedule.transfers:
        if transfer.kind != "SEND":
            raise SemanticError("virtual dual transfers must be SEND operations")
        root = _transfer_root(schedule, transfer)
        contributors = _tree_contributors(schedule, transfer)
        grouped.setdefault((root, contributors), []).append(transfer)
    trees = tuple(
        _validate_tree(
            root,
            contributors,
            tuple(transfers),
            target_interface,
        )
        for (root, contributors), transfers in sorted(
            grouped.items(),
            key=lambda item: (item[0][0], tuple(sorted(item[0][1]))),
        )
    )
    matched = {
        (tree.root_rank, tree.target_offset, tree.contributors)
        for tree in trees
    }
    expected = {
        (slot.rank, slot.offset, contributors)
        for slot, contributors in target_interface.values.items()
    }
    if matched != expected:
        raise SemanticError("dual trees do not cover the target interface")
    return trees
