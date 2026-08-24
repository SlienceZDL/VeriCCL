from dataclasses import replace

from vericcl.semantics.atom import PathStage, Schedule, Symbol
from vericcl.semantics.collective import CollectiveKind
from vericcl.topology.model import (
    DirectedLink,
    LaneKey,
    LinkKey,
    PerformanceCurve,
    SharedResource,
    Topology,
)
from vericcl.tuning.model import TuningOverlay
from vericcl.verification.bdd_flow import (
    FlowReplacementHint,
    analyze_flow_congestion,
)
from vericcl.verification.flow_index import build_flow_index

from tests.unit.verification.bdd_helpers import (
    _transfer,
    crossing_flows_with_ready_wait,
    crossing_inputs,
    crossing_topology,
    two_members_with_shared_suffix,
)
from tests.unit.xml.helpers import resolved


def overlay():
    return TuningOverlay(
        overlay_id="repair-overlay",
        parent_candidate_id="parent",
        channel_count=1,
    )


def waiting_case():
    schedule = crossing_flows_with_ready_wait()
    topology = crossing_topology()
    inputs = crossing_inputs()
    inputs = replace(
        inputs,
        strategies=replace(
            inputs.strategies,
            hierarchy=False,
            manual_hierarchy=False,
        ),
    )
    analysis = analyze_flow_congestion(schedule, topology, inputs)
    hint = next(
        item
        for item in analysis.hints
        if item.waiting_transfer_id == "wait-middle"
    )
    return schedule, topology, inputs, hint


def incomplete_leaf_hint(hint):
    candidate_id = hint.candidate_flow_ids[0]
    return replace(
        hint,
        candidate_paths={candidate_id: (1, 2)},
    )


def aggregate_topology():
    keys = tuple(
        LinkKey(src, dst)
        for src, dst in (
            (0, 2),
            (1, 2),
            (2, 3),
        )
    )
    curve = PerformanceCurve(1.0, 2.0, {})
    return Topology(
        rank_count=4,
        links={
            key: DirectedLink(
                key,
                2 if key == LinkKey(0, 2) else 1,
                curve,
                (),
            )
            for key in keys
        },
        shared_resources={},
        node_membership={rank: 0 for rank in range(4)},
        gateways=frozenset(),
        warnings=(),
    )


def aggregate_case():
    schedule = two_members_with_shared_suffix()
    topology = aggregate_topology()
    inputs = resolved(CollectiveKind.ALL_REDUCE, ranks=4, slices=1)
    flow = next(
        item
        for item in build_flow_index(schedule).flows
        if item.stage_id == 0 and item.member_slice_ids == frozenset({0})
    )
    candidate_id = "aggregate-alternative"
    hint = FlowReplacementHint(
        source_flow_id=flow.flow_id,
        demand_id=flow.demand_id,
        candidate_flow_ids=(candidate_id,),
        candidate_paths={candidate_id: (0, 2)},
        candidate_first_lanes={candidate_id: LaneKey(0, 2, 1)},
        divergence_rank=0,
        waiting_transfer_id="tx-left",
        bottleneck_lane=flow.lanes[0],
        wait_start_us=0.0,
        wait_end_us=1.0,
        earliest_candidate_start_us=0.0,
    )
    return schedule, topology, inputs, hint


def shared_prefix_case():
    schedule = crossing_flows_with_ready_wait()
    wait_stage = schedule.transfers[3].atoms[0].path[0]
    branches = []
    for leaf in (4, 5):
        stage = PathStage(
            0,
            "SEND",
            wait_stage.symbols + (Symbol(3, leaf, 6.0),),
        )
        branches.append(
            _transfer(
                "wait-branch-{}".format(leaf),
                3,
                leaf,
                0,
                {0: (stage,)},
                6.0,
                7.0,
                ("wait-middle",),
            )
        )
    metadata = dict(schedule.metadata)
    semantic = dict(metadata["semantic_predecessors"])
    semantic.update(
        {transfer.transfer_id: ("wait-middle",) for transfer in branches}
    )
    metadata["semantic_predecessors"] = semantic
    schedule = replace(
        schedule,
        transfers=schedule.transfers + tuple(branches),
        metadata=metadata,
    )

    topology = crossing_topology()
    curve = next(iter(topology.links.values())).performance
    links = dict(topology.links)
    for key in (LinkKey(3, 4), LinkKey(2, 4)):
        links[key] = DirectedLink(key, 1, curve, ())
    topology = replace(topology, links=links)
    inputs = crossing_inputs()
    flow = next(
        item
        for item in build_flow_index(schedule).flows
        if item.member_slice_ids == frozenset({0}) and item.leaf_rank == 4
    )
    candidate_id = "shared-prefix-alternative"
    hint = FlowReplacementHint(
        source_flow_id=flow.flow_id,
        demand_id=flow.demand_id,
        candidate_flow_ids=(candidate_id,),
        candidate_paths={candidate_id: (1, 2, 4)},
        candidate_first_lanes={candidate_id: LaneKey(1, 2, 0)},
        divergence_rank=1,
        waiting_transfer_id="wait-middle",
        bottleneck_lane=flow.lanes[1],
        wait_start_us=1.0,
        wait_end_us=5.0,
        earliest_candidate_start_us=1.0,
    )
    return schedule, topology, inputs, hint


def impact_case():
    definitions = (
        ("changed", 0, 1, 0, 0, 0.0, 1.0, ()),
        ("same-lane-later", 0, 1, 0, 0, 2.0, 3.0, ()),
        ("same-link", 0, 1, 1, 0, 0.0, 1.0, ()),
        ("shared-resource", 2, 3, 0, 2, 0.0, 1.0, ()),
        ("dependent", 4, 5, 0, 4, 3.0, 4.0, ("same-lane-later",)),
        ("recursive", 5, 4, 0, 5, 4.0, 5.0, ("dependent",)),
    )
    transfers = []
    for transfer_id, src, dst, channel, member, start, end, predecessors in definitions:
        stage = PathStage(0, "SEND", (Symbol(src, dst, start),))
        transfers.append(
            _transfer(
                transfer_id,
                src,
                dst,
                channel,
                {member: (stage,)},
                start,
                end,
                predecessors,
            )
        )
    schedule = Schedule(
        "impact-case",
        tuple(transfers),
        (),
        6,
        1,
        1024,
        {
            "path_scope": "global",
            "semantic_predecessors": {
                transfer.transfer_id: tuple(sorted(transfer.predecessor_ids))
                for transfer in transfers
            },
            "resource_slots": {
                "changed": {"nic": 0},
                "same-lane-later": {"nic": 0},
                "same-link": {"nic": 1},
                "shared-resource": {"nic": 1},
                "dependent": {},
                "recursive": {},
            },
        },
    )
    curve = PerformanceCurve(1.0, 2.0, {})
    nic_links = (LinkKey(0, 1), LinkKey(2, 3))
    links = (
        LinkKey(0, 1),
        LinkKey(2, 3),
        LinkKey(4, 5),
        LinkKey(5, 4),
    )
    topology = Topology(
        rank_count=6,
        links={
            key: DirectedLink(
                key,
                2,
                curve,
                ("nic",) if key in nic_links else (),
            )
            for key in links
        },
        shared_resources={
            "nic": SharedResource("nic", nic_links, 2, curve),
        },
        node_membership={rank: 0 for rank in range(6)},
        gateways=frozenset(),
        warnings=(),
    )
    return schedule, topology
