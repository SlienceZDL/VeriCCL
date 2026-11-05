from vericcl.planner.model import StageInterface
from vericcl.semantics.atom import Atom, PathStage, Schedule, Symbol, Transfer
from vericcl.semantics.collective import (
    CollectiveKind,
    CollectiveSpec,
    OutputSlot,
)


def reduce_spec():
    return CollectiveSpec(
        kind=CollectiveKind.REDUCE,
        datatype="float32",
        reduction_op="sum",
        root=0,
    )


def reduce_target(rank_count):
    return StageInterface(
        {OutputSlot(0, 0): frozenset(range(rank_count))}
    )


def virtual_reduce_chain(rank_count):
    contributors = tuple(range(rank_count))
    transfers = []
    path_roots = {}
    semantic_contributors = {}
    tree_contributors = {}
    resource_slots = {}
    for edge_index in range(rank_count - 1):
        transfer_id = "virtual-t{:04d}".format(edge_index)
        members = frozenset(range(edge_index + 1, rank_count))
        symbols = tuple(
            Symbol(rank, rank + 1, float(2 * rank))
            for rank in range(edge_index + 1)
        )
        st_time = float(2 * edge_index)
        ed_time = st_time + 2.0
        atoms = tuple(
            Atom(
                slice_id=slice_id,
                slice_size_bytes=1024,
                path=(PathStage(0, "SEND", symbols),),
                st_time=st_time,
                ed_time=ed_time,
            )
            for slice_id in sorted(members)
        )
        transfers.append(
            Transfer(
                transfer_id=transfer_id,
                kind="SEND",
                src_rank=edge_index,
                dst_rank=edge_index + 1,
                channel=0,
                stage_id=0,
                member_slice_ids=members,
                atoms=atoms,
                st_time=st_time,
                ed_time=ed_time,
                predecessor_ids=(
                    frozenset({"virtual-t{:04d}".format(edge_index - 1)})
                    if edge_index
                    else frozenset()
                ),
            )
        )
        path_roots[transfer_id] = 0
        semantic_contributors[transfer_id] = tuple(sorted(members))
        tree_contributors[transfer_id] = contributors
        resource_slots[transfer_id] = {}
    return Schedule(
        schedule_id="virtual-chain",
        transfers=tuple(transfers),
        final_state_ids=(),
        rank_count=rank_count,
        slice_count=1,
        slice_size_bytes=1024,
        metadata={
            "path_scope": "stage_suffix",
            "path_roots": path_roots,
            "reduction_dual": True,
            "semantic_contributors": semantic_contributors,
            "tree_contributors": tree_contributors,
            "resource_slots": resource_slots,
        },
    )


def virtual_reduce_star(rank_count):
    contributors = tuple(range(rank_count))
    transfers = []
    path_roots = {}
    semantic_contributors = {}
    tree_contributors = {}
    resource_slots = {}
    for dst_rank in range(1, rank_count):
        transfer_id = "virtual-star-t{:04d}".format(dst_rank)
        members = frozenset({dst_rank})
        atom = Atom(
            slice_id=dst_rank,
            slice_size_bytes=1024,
            path=(
                PathStage(
                    0,
                    "SEND",
                    (Symbol(0, dst_rank, 0.0),),
                ),
            ),
            st_time=0.0,
            ed_time=2.0,
        )
        transfers.append(
            Transfer(
                transfer_id=transfer_id,
                kind="SEND",
                src_rank=0,
                dst_rank=dst_rank,
                channel=0,
                stage_id=0,
                member_slice_ids=members,
                atoms=(atom,),
                st_time=0.0,
                ed_time=2.0,
                predecessor_ids=frozenset(),
            )
        )
        path_roots[transfer_id] = 0
        semantic_contributors[transfer_id] = (dst_rank,)
        tree_contributors[transfer_id] = contributors
        resource_slots[transfer_id] = {}
    return Schedule(
        schedule_id="virtual-star",
        transfers=tuple(transfers),
        final_state_ids=(),
        rank_count=rank_count,
        slice_count=1,
        slice_size_bytes=1024,
        metadata={
            "path_scope": "stage_suffix",
            "path_roots": path_roots,
            "reduction_dual": True,
            "semantic_contributors": semantic_contributors,
            "tree_contributors": tree_contributors,
            "resource_slots": resource_slots,
        },
    )


def virtual_two_value_reduce():
    transfers = []
    path_roots = {}
    semantic_contributors = {}
    tree_contributors = {}
    resource_slots = {}
    for offset, member in enumerate((2, 3)):
        transfer_id = "virtual-value-t{:04d}".format(offset)
        st_time = float(2 * offset)
        ed_time = st_time + 2.0
        atom = Atom(
            slice_id=member,
            slice_size_bytes=1024,
            path=(PathStage(0, "SEND", (Symbol(0, 1, st_time),)),),
            st_time=st_time,
            ed_time=ed_time,
        )
        transfers.append(
            Transfer(
                transfer_id=transfer_id,
                kind="SEND",
                src_rank=0,
                dst_rank=1,
                channel=0,
                stage_id=0,
                member_slice_ids=frozenset({member}),
                atoms=(atom,),
                st_time=st_time,
                ed_time=ed_time,
                predecessor_ids=(
                    frozenset({"virtual-value-t0000"})
                    if offset
                    else frozenset()
                ),
            )
        )
        path_roots[transfer_id] = 0
        semantic_contributors[transfer_id] = (member,)
        tree_contributors[transfer_id] = (offset, member)
        resource_slots[transfer_id] = {"nic": 0}
    return Schedule(
        schedule_id="virtual-two-value",
        transfers=tuple(transfers),
        final_state_ids=(),
        rank_count=2,
        slice_count=2,
        slice_size_bytes=1024,
        metadata={
            "path_scope": "stage_suffix",
            "path_roots": path_roots,
            "reduction_dual": True,
            "semantic_contributors": semantic_contributors,
            "tree_contributors": tree_contributors,
            "resource_slots": resource_slots,
        },
    )
