from dataclasses import replace
from pathlib import Path

from vericcl.input.loader import resolve_inputs
from vericcl.semantics.atom import Atom, PathStage, Schedule, Symbol, Transfer
from vericcl.semantics.collective import (
    CollectiveKind,
    CollectiveSpec,
    required_outputs,
)


PROJECT_ROOT = Path(__file__).parents[3]
EXAMPLES = PROJECT_ROOT / "vericcl" / "examples"


def collective_spec(kind, *, inplace=False, root=0):
    reduction = kind in {
        CollectiveKind.REDUCE,
        CollectiveKind.ALL_REDUCE,
        CollectiveKind.REDUCE_SCATTER,
    }
    rooted = kind in {CollectiveKind.BROADCAST, CollectiveKind.REDUCE}
    return CollectiveSpec(
        kind=kind,
        datatype="float32",
        reduction_op="sum" if reduction else None,
        root=root if rooted else None,
        inplace=inplace,
    )


def resolved(kind, *, ranks=2, slices=2, inplace=False, root=0):
    inputs = resolve_inputs(
        EXAMPLES / "topo" / "two_rank.json",
        EXAMPLES / "sketch" / "allreduce_8m_1m.json",
        EXAMPLES / "atom" / "default.json",
    )
    return replace(
        inputs,
        collective=collective_spec(kind, inplace=inplace, root=root),
        rank_count=ranks,
        hyperparameters=replace(
            inputs.hyperparameters,
            total_size_bytes=slices * 1024,
            slice_size_bytes=1024,
        ),
    )


def final_schedule(kind, *, ranks=2, slices=2, inplace=False, root=0):
    spec = collective_spec(kind, inplace=inplace, root=root)
    outputs = required_outputs(spec, ranks, slices)
    return Schedule(
        schedule_id="final-{}".format(kind.value),
        transfers=(),
        final_state_ids=tuple(
            "final-r{}-o{}".format(slot.rank, slot.offset)
            for slot in sorted(outputs)
        ),
        rank_count=ranks,
        slice_count=slices,
        slice_size_bytes=1024,
        metadata={
            "path_scope": "global",
            "semantic_predecessors": {},
            "final_outputs": {
                "r{:08d}-o{:08d}".format(slot.rank, slot.offset): tuple(
                    sorted(contributors)
                )
                for slot, contributors in outputs.items()
            },
            "final_dependencies": {
                "r{:08d}-o{:08d}".format(slot.rank, slot.offset): ()
                for slot in outputs
            },
        },
    )


def _atom(slice_id, size, path, st_time, ed_time):
    return Atom(
        slice_id=slice_id,
        slice_size_bytes=size,
        path=tuple(path),
        st_time=st_time,
        ed_time=ed_time,
    )


def _transfer(
    transfer_id,
    kind,
    src,
    dst,
    stage,
    members,
    paths,
    st_time,
    ed_time,
    predecessors=(),
):
    return Transfer(
        transfer_id=transfer_id,
        kind=kind,
        src_rank=src,
        dst_rank=dst,
        channel=0,
        stage_id=stage,
        member_slice_ids=frozenset(members),
        atoms=tuple(
            _atom(member, 1024, paths[member], st_time, ed_time)
            for member in sorted(members)
        ),
        st_time=st_time,
        ed_time=ed_time,
        predecessor_ids=frozenset(predecessors),
    )


def reduce_chain_schedule(*, slices=1, overlap=False):
    transfers = []
    semantic = {}
    for logical in range(slices):
        first_start = 0.0 if overlap else float(logical * 4)
        first_end = first_start + 2.0
        second_start = first_end
        second_end = second_start + 2.0
        member_rank_2 = 2 * slices + logical
        member_rank_1 = slices + logical
        first_id = "reduce-a{}-first".format(logical)
        second_id = "reduce-a{}-second".format(logical)
        first_stage = PathStage(
            0,
            "REDUCE",
            (Symbol(2, 1, first_start),),
        )
        second_stage = PathStage(
            1,
            "REDUCE",
            (Symbol(1, 0, second_start),),
        )
        transfers.append(
            _transfer(
                first_id,
                "REDUCE",
                2,
                1,
                0,
                (member_rank_2,),
                {member_rank_2: (first_stage,)},
                first_start,
                first_end,
            )
        )
        transfers.append(
            _transfer(
                second_id,
                "REDUCE",
                1,
                0,
                1,
                (member_rank_1, member_rank_2),
                {
                    member_rank_1: (
                        PathStage(
                            1,
                            "REDUCE",
                            (Symbol(1, 0, second_start),),
                        ),
                    ),
                    member_rank_2: (first_stage, second_stage),
                },
                second_start,
                second_end,
                (first_id,),
            )
        )
        semantic[first_id] = ()
        semantic[second_id] = (first_id,)
    spec = collective_spec(CollectiveKind.REDUCE, root=0)
    outputs = required_outputs(spec, 3, slices)
    return Schedule(
        schedule_id="reduce-chain",
        transfers=tuple(transfers),
        final_state_ids=tuple(
            "final-r{}-o{}".format(slot.rank, slot.offset)
            for slot in sorted(outputs)
        ),
        rank_count=3,
        slice_count=slices,
        slice_size_bytes=1024,
        metadata={
            "path_scope": "global",
            "semantic_predecessors": semantic,
            "final_outputs": {
                "r{:08d}-o{:08d}".format(slot.rank, slot.offset): tuple(
                    sorted(contributors)
                )
                for slot, contributors in outputs.items()
            },
            "final_dependencies": {
                "r{:08d}-o{:08d}".format(slot.rank, slot.offset): (
                    "reduce-a{}-second".format(slot.offset),
                )
                for slot in outputs
            },
        },
    )


def concurrent_reduce_star_schedule():
    transfers = []
    semantic = {}
    for source in range(1, 4):
        transfer_id = "reduce-star-{}".format(source)
        stage = PathStage(
            0,
            "REDUCE",
            (Symbol(source, 0, 0.0),),
        )
        transfers.append(
            _transfer(
                transfer_id,
                "REDUCE",
                source,
                0,
                0,
                (source,),
                {source: (stage,)},
                0.0,
                1.0,
            )
        )
        semantic[transfer_id] = ()
    outputs = required_outputs(
        collective_spec(CollectiveKind.REDUCE, root=0),
        4,
        1,
    )
    return Schedule(
        schedule_id="concurrent-reduce-star",
        transfers=tuple(transfers),
        final_state_ids=("final-r0-o0",),
        rank_count=4,
        slice_count=1,
        slice_size_bytes=1024,
        metadata={
            "path_scope": "global",
            "semantic_predecessors": semantic,
            "final_outputs": {
                "r{:08d}-o{:08d}".format(slot.rank, slot.offset): tuple(
                    sorted(contributors)
                )
                for slot, contributors in outputs.items()
            },
            "final_dependencies": {
                "r00000000-o00000000": tuple(sorted(semantic))
            },
        },
    )


def inplace_alltoall_overwrite_schedule():
    incoming = _transfer(
        "incoming",
        "SEND",
        0,
        1,
        0,
        (1,),
        {
            1: (
                PathStage(0, "SEND", (Symbol(0, 1, 0.0),)),
            )
        },
        0.0,
        1.0,
    )
    outgoing = _transfer(
        "outgoing",
        "SEND",
        1,
        0,
        0,
        (2,),
        {
            2: (
                PathStage(0, "SEND", (Symbol(1, 0, 0.0),)),
            )
        },
        0.0,
        1.0,
    )
    outputs = required_outputs(
        collective_spec(CollectiveKind.ALL_TO_ALL, inplace=True),
        2,
        2,
    )
    return Schedule(
        schedule_id="inplace-alltoall-overwrite",
        transfers=(incoming, outgoing),
        final_state_ids=tuple(
            "final-r{}-o{}".format(slot.rank, slot.offset)
            for slot in sorted(outputs)
        ),
        rank_count=2,
        slice_count=2,
        slice_size_bytes=1024,
        metadata={
            "path_scope": "global",
            "semantic_predecessors": {"incoming": (), "outgoing": ()},
            "final_outputs": {
                "r{:08d}-o{:08d}".format(slot.rank, slot.offset): tuple(
                    sorted(contributors)
                )
                for slot, contributors in outputs.items()
            },
            "final_dependencies": {},
        },
    )
