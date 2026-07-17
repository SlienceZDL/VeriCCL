from dataclasses import replace

from vericcl.input.models import ForbiddenTransfer
from vericcl.semantics.atom import Atom, PathStage, Schedule, Symbol, Transfer
from vericcl.semantics.collective import CollectiveKind
from vericcl.topology.model import (
    DirectedLink,
    LinkKey,
    PerformanceCurve,
    Topology,
)
from vericcl.xml.endpoints import EndpointType
from vericcl.xml.threadblocks import (
    Threadblock,
    ThreadblockKey,
    ThreadblockProgram,
    XmlStep,
)

from tests.unit.xml.helpers import resolved


def _transfer(
    transfer_id,
    src,
    dst,
    channel,
    member_paths,
    st_time,
    ed_time,
    predecessors=(),
    stage_id=0,
    kind="SEND",
):
    return Transfer(
        transfer_id=transfer_id,
        kind=kind,
        src_rank=src,
        dst_rank=dst,
        channel=channel,
        stage_id=stage_id,
        member_slice_ids=frozenset(member_paths),
        atoms=tuple(
            Atom(
                slice_id=member,
                slice_size_bytes=1024,
                path=path,
                st_time=st_time,
                ed_time=ed_time,
            )
            for member, path in sorted(member_paths.items())
        ),
        st_time=st_time,
        ed_time=ed_time,
        predecessor_ids=frozenset(predecessors),
    )


def two_members_with_shared_suffix():
    left = PathStage(0, "REDUCE", (Symbol(0, 2, 0.0),))
    right = PathStage(0, "REDUCE", (Symbol(1, 2, 0.0),))
    suffix = PathStage(1, "SEND", (Symbol(2, 3, 1.0),))
    transfers = (
        _transfer(
            "tx-left",
            0,
            2,
            0,
            {0: (left,)},
            0.0,
            1.0,
            kind="REDUCE",
        ),
        _transfer(
            "tx-right",
            1,
            2,
            0,
            {1: (right,)},
            0.0,
            1.0,
            kind="REDUCE",
        ),
        _transfer(
            "tx-shared",
            2,
            3,
            0,
            {0: (left, suffix), 1: (right, suffix)},
            1.0,
            2.0,
            ("tx-left", "tx-right"),
            stage_id=1,
        ),
    )
    return Schedule(
        schedule_id="two-members-shared-suffix",
        transfers=transfers,
        final_state_ids=(),
        rank_count=4,
        slice_count=1,
        slice_size_bytes=1024,
        metadata={
            "path_scope": "global",
            "semantic_predecessors": {
                "tx-left": (),
                "tx-right": (),
                "tx-shared": ("tx-left", "tx-right"),
            },
        },
    )


def crossing_flows_with_ready_wait():
    wait_first = PathStage(0, "SEND", (Symbol(0, 1, 0.0),))
    wait_full = PathStage(
        0,
        "SEND",
        (Symbol(0, 1, 0.0), Symbol(1, 3, 1.0)),
    )
    cross_first = PathStage(0, "SEND", (Symbol(4, 1, 0.0),))
    cross_middle = PathStage(
        0,
        "SEND",
        (Symbol(4, 1, 0.0), Symbol(1, 3, 1.0)),
    )
    cross_full = PathStage(
        0,
        "SEND",
        (
            Symbol(4, 1, 0.0),
            Symbol(1, 3, 1.0),
            Symbol(3, 5, 5.0),
        ),
    )
    transfers = (
        _transfer(
            "wait-first",
            0,
            1,
            0,
            {0: (wait_first,)},
            0.0,
            1.0,
        ),
        _transfer(
            "cross-first",
            4,
            1,
            0,
            {4: (cross_first,)},
            0.0,
            1.0,
        ),
        _transfer(
            "cross-middle",
            1,
            3,
            0,
            {4: (cross_middle,)},
            1.0,
            5.0,
            ("cross-first",),
        ),
        _transfer(
            "wait-middle",
            1,
            3,
            0,
            {0: (wait_full,)},
            5.0,
            6.0,
            ("wait-first",),
        ),
        _transfer(
            "cross-last",
            3,
            5,
            0,
            {4: (cross_full,)},
            5.0,
            6.0,
            ("cross-middle",),
        ),
    )
    return Schedule(
        schedule_id="crossing-flow-wait",
        transfers=transfers,
        final_state_ids=(),
        rank_count=6,
        slice_count=1,
        slice_size_bytes=1024,
        metadata={
            "path_scope": "global",
            "semantic_predecessors": {
                transfer.transfer_id: tuple(sorted(transfer.predecessor_ids))
                for transfer in transfers
            },
        },
    )


def crossing_topology():
    keys = tuple(
        LinkKey(src, dst)
        for src, dst in (
            (0, 1),
            (4, 1),
            (1, 3),
            (3, 5),
            (1, 2),
            (2, 3),
        )
    )
    curve = PerformanceCurve(1.0, 2.0, {})
    return Topology(
        rank_count=6,
        links={key: DirectedLink(key, 1, curve, ()) for key in keys},
        shared_resources={},
        node_membership={rank: 0 for rank in range(6)},
        gateways=frozenset(),
        warnings=(),
    )


def crossing_inputs(*, forbid_alternative=False):
    value = resolved(CollectiveKind.ALL_GATHER, ranks=6, slices=1)
    forbidden = (
        (
            ForbiddenTransfer(
                slice_id=0,
                src_rank=1,
                dst_rank=2,
                stage_id=0,
            ),
        )
        if forbid_alternative
        else ()
    )
    return replace(
        value,
        atom_constraints=replace(
            value.atom_constraints,
            forbidden_transfers=forbidden,
        ),
    )


def tb_order_case(*, necessary_order=False):
    prep_stage = PathStage(0, "SEND", (Symbol(2, 0, 0.0),))
    slow_stage = PathStage(
        0,
        "SEND",
        (Symbol(2, 0, 0.0), Symbol(0, 1, 5.0)),
    )
    fast_stage = PathStage(0, "SEND", (Symbol(0, 1, 0.0),))
    transfers = (
        _transfer(
            "prep",
            2,
            0,
            0,
            {2: (prep_stage,)},
            0.0,
            5.0,
        ),
        _transfer(
            "slow",
            0,
            1,
            0,
            {2: (slow_stage,)},
            5.0,
            6.0,
            ("prep",),
        ),
        _transfer(
            "fast",
            0,
            1,
            0,
            {0: (fast_stage,)},
            6.0,
            7.0,
        ),
    )
    schedule = Schedule(
        schedule_id="tb-order-case",
        transfers=transfers,
        final_state_ids=(),
        rank_count=3,
        slice_count=1,
        slice_size_bytes=1024,
        metadata={
            "path_scope": "global",
            "semantic_predecessors": {
                "prep": (),
                "slow": ("prep",),
                "fast": (),
            },
        },
    )

    slow = XmlStep(
        step_id="slow-send",
        node_id="slow",
        transfer_id="slow",
        endpoint_id="slow-send",
        xml_type=EndpointType.SEND,
        rank=0,
        peer=1,
        channel=0,
        src_ref=None,
        dst_ref=None,
        dependency_step_id=None,
        has_dependence=necessary_order,
        semantic_predecessor_node_ids=("prep",),
        member_slice_ids=frozenset({2}),
        solver_st_time=5.0,
        solver_ed_time=6.0,
        effective_st_time=5.0,
        effective_ed_time=6.0,
    )
    fast = XmlStep(
        step_id="fast-send",
        node_id="fast",
        transfer_id="fast",
        endpoint_id="fast-send",
        xml_type=EndpointType.SEND,
        rank=0,
        peer=1,
        channel=0,
        src_ref=None,
        dst_ref=None,
        dependency_step_id="slow-send" if necessary_order else None,
        has_dependence=False,
        semantic_predecessor_node_ids=(),
        member_slice_ids=frozenset({0}),
        solver_st_time=6.0,
        solver_ed_time=7.0,
        effective_st_time=6.0,
        effective_ed_time=7.0,
    )
    block = Threadblock(
        key=ThreadblockKey(0, "send", 1, 0),
        tb_id=0,
        steps=(slow, fast),
    )
    program = ThreadblockProgram(
        threadblocks=(block,),
        steps_by_id={slow.step_id: slow, fast.step_id: fast},
        transfer_steps={},
        node_steps={"slow": (slow.step_id,), "fast": (fast.step_id,)},
        referenced_step_ids=(
            frozenset({slow.step_id}) if necessary_order else frozenset()
        ),
        inversion_count=1,
    )
    return program, schedule
