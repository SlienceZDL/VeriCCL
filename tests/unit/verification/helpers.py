from dataclasses import replace

from vericcl.input.models import ForbiddenTransfer
from vericcl.semantics.atom import Atom, PathStage, Schedule, Symbol, Transfer
from vericcl.semantics.collective import (
    CollectiveKind,
    required_outputs,
)
from vericcl.topology.model import (
    DirectedLink,
    LinkKey,
    PerformanceCurve,
    SharedResource,
    Topology,
)

from tests.unit.xml.helpers import resolved, two_rank_allreduce_schedule


def inputs(kind=CollectiveKind.ALL_REDUCE, *, ranks=2, slices=1):
    return resolved(kind, ranks=ranks, slices=slices)


def topology(
    rank_count=2,
    *,
    links=None,
    shared=False,
    resource_channels=1,
):
    keys = tuple(
        LinkKey(src, dst)
        for src, dst in (
            links
            if links is not None
            else (
                (src, dst)
                for src in range(rank_count)
                for dst in range(rank_count)
                if src != dst
            )
        )
    )
    curve = PerformanceCurve(1.0, 2.0, {})
    resource_ids = ("nic",) if shared else ()
    edges = {
        key: DirectedLink(key, 2, curve, resource_ids) for key in keys
    }
    resources = (
        {
            "nic": SharedResource(
                "nic",
                keys,
                resource_channels,
                curve,
            )
        }
        if shared
        else {}
    )
    return Topology(
        rank_count=rank_count,
        links=edges,
        shared_resources=resources,
        node_membership={rank: 0 for rank in range(rank_count)},
        gateways=frozenset(),
        warnings=(),
    )


def forbidden_shared_transfer_inputs():
    value = inputs()
    forbidden = ForbiddenTransfer(
        slice_id=1,
        src_rank=0,
        dst_rank=1,
        stage_id=1,
    )
    return replace(
        value,
        atom_constraints=replace(
            value.atom_constraints,
            forbidden_transfers=(forbidden,),
        ),
    )


def duplicate_reduction_schedule():
    schedule = two_rank_allreduce_schedule()
    reduce_stage = PathStage(0, "REDUCE", (Symbol(1, 0, 0.0),))
    send_stage = PathStage(1, "SEND", (Symbol(0, 1, 1.0),))
    duplicate_stage = PathStage(2, "REDUCE", (Symbol(1, 0, 2.0),))
    duplicate = Transfer(
        transfer_id="duplicate-reduce",
        kind="REDUCE",
        src_rank=1,
        dst_rank=0,
        channel=0,
        stage_id=2,
        member_slice_ids=frozenset({0, 1}),
        atoms=(
            Atom(
                0,
                1024,
                (send_stage, duplicate_stage),
                2.0,
                3.0,
            ),
            Atom(
                1,
                1024,
                (reduce_stage, send_stage, duplicate_stage),
                2.0,
                3.0,
            ),
        ),
        st_time=2.0,
        ed_time=3.0,
        predecessor_ids=frozenset({"allreduce-send"}),
    )
    metadata = dict(schedule.metadata)
    semantic = dict(metadata["semantic_predecessors"])
    semantic[duplicate.transfer_id] = ("allreduce-send",)
    metadata["semantic_predecessors"] = semantic
    return replace(
        schedule,
        transfers=schedule.transfers + (duplicate,),
        metadata=metadata,
    )


def inactive_reuse_schedule():
    transfers = []
    for dst_rank in (1, 2):
        transfer_id = "incomplete-send-{}".format(dst_rank)
        atom = Atom(
            slice_id=0,
            slice_size_bytes=1024,
            path=(
                PathStage(
                    0,
                    "SEND",
                    (Symbol(0, dst_rank, 0.0),),
                ),
            ),
            st_time=0.0,
            ed_time=1.0,
        )
        transfers.append(
            Transfer(
                transfer_id=transfer_id,
                kind="SEND",
                src_rank=0,
                dst_rank=dst_rank,
                channel=0,
                stage_id=0,
                member_slice_ids=frozenset({0}),
                atoms=(atom,),
                st_time=0.0,
                ed_time=1.0,
                predecessor_ids=frozenset(),
            )
        )
    return Schedule(
        schedule_id="inactive-reuse",
        transfers=tuple(transfers),
        final_state_ids=("final-r0-o0",),
        rank_count=3,
        slice_count=1,
        slice_size_bytes=1024,
        metadata={
            "path_scope": "global",
            "semantic_predecessors": {
                transfer.transfer_id: () for transfer in transfers
            },
            "final_outputs": {
                "r00000000-o00000000": (0, 1, 2),
            },
            "final_dependencies": {
                "r00000000-o00000000": (),
            },
        },
    )


def reduce_scatter_schedule():
    transfers = []
    for logical_address, (src_rank, dst_rank, member) in enumerate(
        ((1, 0, 2), (0, 1, 1))
    ):
        transfer_id = "reduce-scatter-{}".format(logical_address)
        stage = PathStage(
            0,
            "REDUCE",
            (Symbol(src_rank, dst_rank, 0.0),),
        )
        transfers.append(
            Transfer(
                transfer_id=transfer_id,
                kind="REDUCE",
                src_rank=src_rank,
                dst_rank=dst_rank,
                channel=0,
                stage_id=0,
                member_slice_ids=frozenset({member}),
                atoms=(Atom(member, 1024, (stage,), 0.0, 1.0),),
                st_time=0.0,
                ed_time=1.0,
                predecessor_ids=frozenset(),
            )
        )
    outputs = required_outputs(
        inputs(CollectiveKind.REDUCE_SCATTER, slices=2).collective,
        2,
        2,
    )
    return Schedule(
        schedule_id="two-rank-reduce-scatter",
        transfers=tuple(transfers),
        final_state_ids=tuple(
            "final-r{}-o{}".format(slot.rank, slot.offset)
            for slot in sorted(outputs)
        ),
        rank_count=2,
        slice_count=2,
        slice_size_bytes=1024,
        metadata={
            "path_scope": "global",
            "semantic_predecessors": {
                transfer.transfer_id: () for transfer in transfers
            },
            "final_outputs": {
                "r{:08d}-o{:08d}".format(slot.rank, slot.offset): tuple(
                    sorted(contributors)
                )
                for slot, contributors in outputs.items()
            },
            "final_dependencies": {
                "r{:08d}-o{:08d}".format(slot.rank, slot.offset): (
                    "reduce-scatter-{}".format(slot.rank),
                )
                for slot in outputs
            },
        },
    )


def interleaved_concurrent_reduce_schedule():
    definitions = (
        ("reduce-a0", 1, 2, 0),
        ("reduce-b1", 1, 3, 1),
        ("reduce-c0", 2, 4, 0),
        ("reduce-d1", 2, 5, 1),
    )
    transfers = []
    for transfer_id, src_rank, member, channel in definitions:
        stage = PathStage(
            0,
            "REDUCE",
            (Symbol(src_rank, 0, 0.0),),
        )
        transfers.append(
            Transfer(
                transfer_id=transfer_id,
                kind="REDUCE",
                src_rank=src_rank,
                dst_rank=0,
                channel=channel,
                stage_id=0,
                member_slice_ids=frozenset({member}),
                atoms=(Atom(member, 1024, (stage,), 0.0, 1.0),),
                st_time=0.0,
                ed_time=1.0,
                predecessor_ids=frozenset(),
            )
        )
    outputs = required_outputs(
        inputs(CollectiveKind.REDUCE, ranks=3, slices=2).collective,
        3,
        2,
    )
    return Schedule(
        schedule_id="interleaved-concurrent-reduce",
        transfers=tuple(transfers),
        final_state_ids=("final-r0-o0", "final-r0-o1"),
        rank_count=3,
        slice_count=2,
        slice_size_bytes=1024,
        metadata={
            "path_scope": "global",
            "semantic_predecessors": {
                transfer.transfer_id: () for transfer in transfers
            },
            "final_outputs": {
                "r{:08d}-o{:08d}".format(slot.rank, slot.offset): tuple(
                    sorted(contributors)
                )
                for slot, contributors in outputs.items()
            },
            "final_dependencies": {},
        },
    )
