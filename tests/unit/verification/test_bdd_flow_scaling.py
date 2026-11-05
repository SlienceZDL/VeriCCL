"""Exact flow-congestion semantics with work bounded by independent queries."""

from dataclasses import replace

import pytest

from vericcl.input.models import ForbiddenTransfer
from vericcl.semantics.atom import PathStage, Schedule, Symbol
from vericcl.semantics.collective import CollectiveKind
from vericcl.topology.model import (
    DirectedLink,
    LaneKey,
    LinkKey,
    PerformanceCurve,
    Topology,
)
from vericcl.verification import bdd_flow
from vericcl.verification.bdd_backend import CompactBDD
from vericcl.verification.bdd_flow import analyze_flow_congestion
from vericcl.verification.flow_index import LaneState
from vericcl.verification.model import ValidationStatus

from tests.unit.verification.bdd_helpers import _transfer
from tests.unit.xml.helpers import resolved


pytestmark = pytest.mark.phase05


def _parallel_waits(*, count=3, block_alternative=True, windows=None):
    """Each waiting flow has two other direct lanes and three detour lanes."""
    if windows is None:
        windows = ((0.0, 5.0, 1.0),) * count
    transfers = tuple(
        _transfer(
            "wait-{:03d}".format(member),
            0,
            2,
            0,
            {member: (PathStage(0, "SEND", (Symbol(0, 2, ready),)),)},
            deadline,
            deadline + duration,
        )
        for member, (ready, deadline, duration) in enumerate(windows)
    )
    if block_alternative:
        transfers += (
            _transfer(
                "blocker",
                0,
                1,
                1,
                {0: (PathStage(0, "SEND", (Symbol(0, 1, 0.0),)),)},
                0.0,
                5.0,
            ),
        )
    schedule = Schedule(
        "parallel-waits",
        transfers,
        (),
        3,
        count,
        1024,
        {
            "path_scope": "global",
            "semantic_predecessors": {
                transfer.transfer_id: () for transfer in transfers
            },
        },
    )
    curve = PerformanceCurve(1.0, 2.0, {})
    keys = (LinkKey(0, 2), LinkKey(0, 1), LinkKey(1, 2))
    topology = Topology(
        rank_count=3,
        links={key: DirectedLink(key, 3, curve, ()) for key in keys},
        shared_resources={},
        node_membership={rank: 0 for rank in range(3)},
        gateways=frozenset(),
        warnings=(),
    )
    inputs = resolved(CollectiveKind.ALL_GATHER, ranks=3, slices=count)
    inputs = replace(inputs, solver=replace(inputs.solver, max_channels=3))
    return schedule, topology, inputs


@pytest.mark.parametrize("count", (1, 3, 5))
def test_dependent_metadata_does_not_inflate_bdd_domain(count):
    # Reintroducing flow/demand/lane dimensions increases BDD work even though
    # each candidate ID already determines all of those attributes uniquely.
    result = analyze_flow_congestion(*_parallel_waits(count=count))

    assert result.status is ValidationStatus.VALID
    assert result.evidence == {
        "candidate_count": 5 * count,
        "relation_count": 4 * count,
        "bdd_variable_count": (5 * count - 1).bit_length(),
    }
    assert tuple(hint.waiting_transfer_id for hint in result.hints) == tuple(
        "wait-{:03d}".format(member) for member in range(count)
    )
    for hint in result.hints:
        assert tuple(
            (hint.candidate_paths[candidate], hint.candidate_first_lanes[candidate])
            for candidate in hint.candidate_flow_ids
        ) == (
            ((0, 2), LaneKey(0, 2, 1)),
            ((0, 2), LaneKey(0, 2, 2)),
            ((0, 1, 2), LaneKey(0, 1, 0)),
            ((0, 1, 2), LaneKey(0, 1, 2)),
        )
        assert hint.wait_interval_us == (0.0, 5.0)
        assert hint.earliest_candidate_start_us == 0.0


@pytest.mark.parametrize("block_alternative, row_budget", ((False, 0), (True, 3)))
def test_idle_relation_materializes_only_the_smaller_exact_partition(
    monkeypatch, block_alternative, row_budget
):
    # Encoding every compatible or every idle row regresses linear BDD cube
    # construction when nearly the whole candidate domain is idle.
    row_counts = []
    original_relation = CompactBDD.relation

    def measured_relation(self, rows):
        rows = tuple(rows)
        row_counts.append(len(rows))
        return original_relation(self, rows)

    monkeypatch.setattr(CompactBDD, "relation", measured_relation)
    result = analyze_flow_congestion(
        *_parallel_waits(block_alternative=block_alternative)
    )

    assert result.status is ValidationStatus.VALID
    assert result.evidence["candidate_count"] == 15
    assert result.evidence["relation_count"] == 15 - row_budget
    assert sum(len(hint.candidate_flow_ids) for hint in result.hints) == (
        15 - row_budget
    )
    assert sum(row_counts) <= row_budget


def test_repeated_queries_are_shared_without_collapsing_distinct_candidates(
    monkeypatch,
):
    # Dropping query-local reuse repeats the same expensive path/lane scans;
    # merging candidates instead would violate both counts and hint identities.
    path_queries = []
    idle_queries = []
    original_paths = bdd_flow.ranked_simple_paths
    original_idle = LaneState.earliest_start

    def measured_paths(*args, **kwargs):
        path_queries.append((args[0], args[1], args[2]))
        return original_paths(*args, **kwargs)

    def measured_idle(self, ready_time, deadline, duration):
        idle_queries.append((self.lane, ready_time, deadline, duration))
        return original_idle(self, ready_time, deadline, duration)

    monkeypatch.setattr(bdd_flow, "ranked_simple_paths", measured_paths)
    monkeypatch.setattr(LaneState, "earliest_start", measured_idle)
    result = analyze_flow_congestion(*_parallel_waits())

    assert result.status is ValidationStatus.VALID
    assert result.evidence["candidate_count"] == 15
    assert result.evidence["relation_count"] == 12
    assert len(result.hints) == 3
    assert len({
        candidate for hint in result.hints for candidate in hint.candidate_flow_ids
    }) == 12
    assert len(path_queries) == 1
    assert len(idle_queries) == 5


def test_path_cache_separates_member_specific_forbidden_links():
    schedule, topology, inputs = _parallel_waits(block_alternative=False)
    inputs = replace(
        inputs,
        atom_constraints=replace(
            inputs.atom_constraints,
            forbidden_transfers=(ForbiddenTransfer(1, 0, 1, 0),),
        ),
    )

    result = analyze_flow_congestion(schedule, topology, inputs)

    assert result.status is ValidationStatus.VALID
    assert result.evidence["candidate_count"] == 12
    assert result.evidence["relation_count"] == 12
    assert tuple(len(hint.candidate_flow_ids) for hint in result.hints) == (5, 2, 5)
    assert set(result.hints[1].candidate_paths.values()) == {(0, 2)}


@pytest.mark.parametrize(
    "windows, selected_counts, earliest",
    (
        (((0.0, 8.0, 1.0), (1.0, 8.0, 1.0)), (5, 5), (0.0, 1.0)),
        (((0.0, 3.0, 1.0), (0.0, 8.0, 1.0)), (4, 5), (0.0, 0.0)),
        (((0.0, 8.0, 1.0), (0.0, 8.0, 4.0)), (5, 4), (0.0, 0.0)),
    ),
)
def test_idle_cache_separates_readiness_deadline_and_duration(
    windows, selected_counts, earliest
):
    result = analyze_flow_congestion(*_parallel_waits(count=2, windows=windows))

    assert result.status is ValidationStatus.VALID
    assert result.evidence["candidate_count"] == 10
    assert tuple(len(hint.candidate_flow_ids) for hint in result.hints) == selected_counts
    assert tuple(hint.earliest_candidate_start_us for hint in result.hints) == earliest


def test_cached_ranked_paths_still_apply_each_flows_prefix_exclusion():
    transfers = []
    for source in (0, 2):
        first = PathStage(0, "SEND", (Symbol(source, 1, 0.0),))
        complete = PathStage(
            0, "SEND", (Symbol(source, 1, 0.0), Symbol(1, 3, 1.0))
        )
        transfers.extend((
            _transfer(
                "first-{}".format(source),  source, 1, 0,
                {source: (first,)}, 0.0, 1.0,
            ),
            _transfer(
                "wait-{}".format(source), 1, 3, 0,
                {source: (complete,)}, 5.0, 6.0,
            ),
        ))
    schedule = Schedule(
        "different-prefixes", tuple(transfers), (), 4, 1, 1024,
        {"path_scope": "global"},
    )
    curve = PerformanceCurve(1.0, 2.0, {})
    keys = tuple(
        LinkKey(src, dst)
        for src, dst in ((0, 1), (2, 1), (1, 3), (1, 0), (0, 3), (1, 2), (2, 3))
    )
    topology = Topology(
        4, {key: DirectedLink(key, 1, curve, ()) for key in keys}, {},
        {rank: 0 for rank in range(4)}, frozenset(), (),
    )
    inputs = resolved(CollectiveKind.ALL_GATHER, ranks=4, slices=1)

    result = analyze_flow_congestion(schedule, topology, inputs)

    assert result.status is ValidationStatus.VALID
    assert result.evidence["candidate_count"] == 2
    assert result.evidence["relation_count"] == 2
    assert tuple(
        (hint.waiting_transfer_id, tuple(hint.candidate_paths.values()))
        for hint in result.hints
    ) == (("wait-0", ((1, 2, 3),)), ("wait-2", ((1, 0, 3),)))
