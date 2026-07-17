from dataclasses import replace

import pytest

from vericcl.composer import compose, recompute_earliest_times
from vericcl.errors import SemanticError
from vericcl.topology.model import (
    DirectedLink,
    LinkKey,
    PerformanceCurve,
    Topology,
)

from .test_compose import _pipeline_candidates, _pipeline_plan


pytestmark = pytest.mark.phase03


def _topology(*links):
    curve = PerformanceCurve(
        alpha_us=1.0,
        invbw_us=2.0,
        bandwidth_bytes_per_us={},
    )
    edges = {
        LinkKey(src, dst): DirectedLink(
            key=LinkKey(src, dst),
            max_channels=1,
            performance=curve,
            resource_ids=(),
        )
        for src, dst in links
    }
    return Topology(
        rank_count=3,
        links=edges,
        shared_resources={},
        node_membership={0: 0, 1: 0, 2: 0},
        gateways=frozenset(),
        warnings=(),
    )


def test_recompute_earliest_times_is_idempotent_for_composed_schedule():
    schedule = compose(_pipeline_plan(), _pipeline_candidates())

    recomputed = recompute_earliest_times(
        schedule,
        _topology((0, 1), (1, 2)),
    )

    assert recomputed == schedule


def test_recompute_earliest_times_rejects_missing_physical_link():
    schedule = compose(_pipeline_plan(), _pipeline_candidates())

    with pytest.raises(SemanticError, match="absent from the topology"):
        recompute_earliest_times(schedule, _topology((0, 1)))


@pytest.mark.parametrize(
    ("semantic", "message"),
    (
        ("invalid", "must be a mapping"),
        ({}, "cover every transfer"),
        (
            {
                "pipeline-first-t0": ("missing",),
                "pipeline-first-t1": (),
                "pipeline-second-t0": (),
                "pipeline-second-t1": (),
            },
            "missing from the schedule",
        ),
        (
            {
                "pipeline-first-t0": ("pipeline-first-t1",),
                "pipeline-first-t1": ("pipeline-first-t0",),
                "pipeline-second-t0": (),
                "pipeline-second-t1": (),
            },
            "contain a cycle",
        ),
    ),
)
def test_recompute_rejects_invalid_semantic_dependencies(
    semantic,
    message,
):
    schedule = compose(_pipeline_plan(), _pipeline_candidates())
    metadata = dict(schedule.metadata)
    metadata["semantic_predecessors"] = semantic
    invalid = replace(schedule, metadata=metadata)

    with pytest.raises(SemanticError, match=message):
        recompute_earliest_times(
            invalid,
            _topology((0, 1), (1, 2)),
        )


def test_recompute_rejects_channel_above_topology_limit():
    schedule = compose(_pipeline_plan(), _pipeline_candidates())
    transfers = list(schedule.transfers)
    transfers[0] = replace(transfers[0], channel=1)
    invalid = replace(schedule, transfers=tuple(transfers))

    with pytest.raises(SemanticError, match="channel exceeds"):
        recompute_earliest_times(
            invalid,
            _topology((0, 1), (1, 2)),
        )


def test_recompute_rejects_path_operation_missing_from_schedule():
    schedule = compose(_pipeline_plan(), _pipeline_candidates())
    transfer = next(
        item
        for item in schedule.transfers
        if item.transfer_id == "pipeline-second-t0"
    )
    transfer = replace(transfer, predecessor_ids=frozenset())
    metadata = dict(schedule.metadata)
    metadata["semantic_predecessors"] = {transfer.transfer_id: ()}
    metadata["resource_slots"] = {transfer.transfer_id: {}}
    incomplete = replace(
        schedule,
        transfers=(transfer,),
        metadata=metadata,
    )

    with pytest.raises(SemanticError, match="operation is missing"):
        recompute_earliest_times(
            incomplete,
            _topology((0, 1), (1, 2)),
        )
