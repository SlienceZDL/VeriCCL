from vericcl.semantics.atom import Atom, PathStage, Schedule, Symbol, Transfer
from vericcl.topology.model import (
    DirectedLink,
    LinkKey,
    PerformanceCurve,
    SharedResource,
    Topology,
)


def curve(alpha=1.0, invbw=2.0, bandwidth=None):
    return PerformanceCurve(
        alpha,
        invbw,
        {} if bandwidth is None else bandwidth,
    )


def simulation_topology(
    rank_count,
    link_curves,
    *,
    max_channels=4,
    shared_links=(),
    shared_curve=None,
    shared_channels=4,
):
    shared_keys = tuple(LinkKey(*link) for link in shared_links)
    links = {}
    for raw_key, performance in link_curves.items():
        key = LinkKey(*raw_key)
        links[key] = DirectedLink(
            key,
            max_channels,
            performance,
            ("nic",) if key in shared_keys else (),
        )
    resources = {}
    if shared_keys:
        resources["nic"] = SharedResource(
            "nic",
            shared_keys,
            shared_channels,
            curve() if shared_curve is None else shared_curve,
        )
    return Topology(
        rank_count=rank_count,
        links=links,
        shared_resources=resources,
        node_membership={rank: 0 for rank in range(rank_count)},
        gateways=frozenset(),
        warnings=(),
    )


def transfer(
    transfer_id,
    src_rank,
    dst_rank,
    channel,
    slice_id,
    slice_count,
    *,
    predecessors=(),
    kind="SEND",
    st_time=0.0,
    ed_time=1.0,
    prior_symbols=(),
):
    symbols = tuple(prior_symbols) + (
        Symbol(src_rank, dst_rank, st_time),
    )
    atom = Atom(
        slice_id=slice_id,
        slice_size_bytes=1024,
        path=(PathStage(0, kind, symbols),),
        st_time=st_time,
        ed_time=ed_time,
    )
    return Transfer(
        transfer_id=transfer_id,
        kind=kind,
        src_rank=src_rank,
        dst_rank=dst_rank,
        channel=channel,
        stage_id=0,
        member_slice_ids=frozenset({slice_id}),
        atoms=(atom,),
        st_time=st_time,
        ed_time=ed_time,
        predecessor_ids=frozenset(predecessors),
    )


def schedule(
    schedule_id,
    rank_count,
    slice_count,
    transfers,
    *,
    semantic_predecessors=None,
    resource_slots=None,
):
    values = tuple(transfers)
    semantic = (
        {
            item.transfer_id: tuple(sorted(item.predecessor_ids))
            for item in values
        }
        if semantic_predecessors is None
        else semantic_predecessors
    )
    return Schedule(
        schedule_id=schedule_id,
        transfers=values,
        final_state_ids=(),
        rank_count=rank_count,
        slice_count=slice_count,
        slice_size_bytes=1024,
        metadata={
            "path_scope": "global",
            "semantic_predecessors": semantic,
            "resource_slots": (
                {} if resource_slots is None else resource_slots
            ),
        },
    )


def opposite_direction_schedule():
    return schedule(
        "opposite-directions",
        2,
        1,
        (
            transfer("forward", 0, 1, 0, 0, 1),
            transfer("reverse", 1, 0, 0, 1, 1),
        ),
    )


def same_direction_schedule(channel_count):
    return schedule(
        "same-direction-{}".format(channel_count),
        2,
        channel_count,
        tuple(
            transfer(
                "send-{}".format(channel),
                0,
                1,
                channel,
                channel,
                channel_count,
            )
            for channel in range(channel_count)
        ),
    )


def relay_schedule():
    first = transfer("relay-first", 0, 1, 0, 0, 1)
    second = transfer(
        "relay-second",
        1,
        2,
        0,
        0,
        1,
        predecessors=("relay-first",),
        st_time=1.0,
        ed_time=2.0,
        prior_symbols=(Symbol(0, 1, 0.0),),
    )
    return schedule("relay", 3, 1, (first, second))
