import pytest

from vericcl.errors import SemanticError
from vericcl.semantics.collective import CollectiveKind
from vericcl.xml.buffers import build_buffer_plan
from vericcl.xml.model import AggregateValue, LocalCopy, PhysicalRef, RawValue

from tests.unit.xml.helpers import final_schedule, resolved


pytestmark = pytest.mark.phase04


@pytest.mark.parametrize(
    "kind,rank,source,logical,expected",
    [
        (CollectiveKind.BROADCAST, 1, 0, 1, ("o", 1)),
        (CollectiveKind.REDUCE, 0, 1, 1, ("o", 1)),
        (CollectiveKind.ALL_GATHER, 1, 1, 1, ("o", 3)),
        (CollectiveKind.ALL_REDUCE, 1, 1, 1, ("o", 1)),
        (CollectiveKind.ALL_TO_ALL, 1, 0, 1, ("o", 0)),
        (CollectiveKind.REDUCE_SCATTER, 1, 1, 1, ("o", 0)),
    ],
)
def test_final_offsets(kind, rank, source, logical, expected):
    plan = build_buffer_plan(
        final_schedule(kind, ranks=2, slices=2),
        resolved(kind, ranks=2, slices=2),
    )

    assert plan.final_ref(rank, source, logical).buffer_offset == expected


def test_declared_chunk_counts_follow_collective_contract():
    allgather = build_buffer_plan(
        final_schedule(CollectiveKind.ALL_GATHER),
        resolved(CollectiveKind.ALL_GATHER),
    )
    reduce_scatter = build_buffer_plan(
        final_schedule(CollectiveKind.REDUCE_SCATTER),
        resolved(CollectiveKind.REDUCE_SCATTER),
    )

    assert set(allgather.i_chunks.values()) == {2}
    assert set(allgather.o_chunks.values()) == {4}
    assert set(reduce_scatter.i_chunks.values()) == {2}
    assert set(reduce_scatter.o_chunks.values()) == {1}


@pytest.mark.parametrize(
    "factory,match",
    [
        (lambda: RawValue(True), "must be an integer"),
        (lambda: RawValue(-1), "must be at least"),
        (
            lambda: AggregateValue(0, frozenset({0}), 0),
            "at least two slices",
        ),
        (
            lambda: PhysicalRef(0, "x", 0, 0.0, 1.0),
            "must be i, o, or s",
        ),
        (
            lambda: PhysicalRef(0, "i", 0, 2.0, 1.0),
            "must not exceed",
        ),
        (
            lambda: LocalCopy(
                "copy",
                0,
                PhysicalRef(1, "i", 0, 0.0, 1.0),
                PhysicalRef(0, "o", 0, 0.0, 1.0),
                "state",
                0.0,
                1.0,
                "test",
            ),
            "copy rank",
        ),
    ],
)
def test_buffer_models_reject_invalid_values(factory, match):
    with pytest.raises(SemanticError, match=match):
        factory()


def test_final_ref_rejects_a_contributor_without_an_output():
    plan = build_buffer_plan(
        final_schedule(CollectiveKind.BROADCAST),
        resolved(CollectiveKind.BROADCAST),
    )

    with pytest.raises(SemanticError, match="exactly one output"):
        plan.final_ref(0, 1, 0)
