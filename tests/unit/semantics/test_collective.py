import pytest

from vericcl.errors import SemanticError
from vericcl.semantics.checker import check_final_states
from vericcl.semantics.collective import (
    CollectiveKind,
    CollectiveSpec,
    OutputSlot,
    required_outputs,
)
from vericcl.semantics.state import PayloadState


pytestmark = pytest.mark.phase01


def make_spec(kind):
    return CollectiveSpec(
        kind=kind,
        datatype="float32",
        reduction_op=(
            "sum"
            if kind
            in {
                CollectiveKind.REDUCE,
                CollectiveKind.ALL_REDUCE,
                CollectiveKind.REDUCE_SCATTER,
            }
            else None
        ),
        root=(
            0
            if kind
            in {
                CollectiveKind.BROADCAST,
                CollectiveKind.REDUCE,
                CollectiveKind.SCATTER,
                CollectiveKind.GATHER,
            }
            else None
        ),
    )


@pytest.mark.parametrize(
    "kind,slot,contributors",
    [
        (CollectiveKind.BROADCAST, OutputSlot(1, 0), frozenset({0})),
        (CollectiveKind.REDUCE, OutputSlot(0, 0), frozenset({0, 2})),
        (CollectiveKind.SCATTER, OutputSlot(1, 0), frozenset({1})),
        (CollectiveKind.GATHER, OutputSlot(0, 2), frozenset({2})),
        (CollectiveKind.ALL_GATHER, OutputSlot(1, 2), frozenset({2})),
        (CollectiveKind.ALL_REDUCE, OutputSlot(1, 0), frozenset({0, 2})),
        (CollectiveKind.ALL_TO_ALL, OutputSlot(1, 0), frozenset({1})),
        (CollectiveKind.REDUCE_SCATTER, OutputSlot(1, 0), frozenset({1, 3})),
    ],
)
def test_required_output_mapping(kind, slot, contributors):
    spec = make_spec(kind)

    assert required_outputs(spec, rank_count=2, slice_count=2)[slot] == contributors


@pytest.mark.parametrize(
    "kind,expected_count",
    [
        (CollectiveKind.BROADCAST, 4),
        (CollectiveKind.REDUCE, 2),
        (CollectiveKind.SCATTER, 2),
        (CollectiveKind.GATHER, 4),
        (CollectiveKind.ALL_GATHER, 8),
        (CollectiveKind.ALL_REDUCE, 4),
        (CollectiveKind.ALL_TO_ALL, 4),
        (CollectiveKind.REDUCE_SCATTER, 2),
    ],
)
def test_required_output_count(kind, expected_count):
    outputs = required_outputs(make_spec(kind), rank_count=2, slice_count=2)

    assert len(outputs) == expected_count


def make_final_states(spec, rank_count=2, slice_count=2):
    outputs = required_outputs(spec, rank_count, slice_count)
    states = []
    for index, (slot, contributors) in enumerate(outputs.items()):
        logical_addresses = {slice_id % slice_count for slice_id in contributors}
        assert len(logical_addresses) == 1
        states.append(
            PayloadState(
                state_id="final-{}".format(index),
                version=0,
                rank=slot.rank,
                logical_address=logical_addresses.pop(),
                contributors=contributors,
                ready_time=1.0,
                active=True,
                member_paths=tuple(
                    (slice_id, ()) for slice_id in sorted(contributors)
                ),
            )
        )
    return tuple(states)


@pytest.mark.parametrize("kind", list(CollectiveKind))
def test_exact_final_states_are_accepted(kind):
    spec = make_spec(kind)

    check_final_states(
        spec,
        rank_count=2,
        slice_count=2,
        states=make_final_states(spec),
    )


def test_missing_final_state_is_rejected():
    spec = make_spec(CollectiveKind.ALL_REDUCE)
    states = make_final_states(spec)

    with pytest.raises(SemanticError, match="missing final output"):
        check_final_states(spec, 2, 2, states[:-1])


def test_duplicate_final_state_is_rejected():
    spec = make_spec(CollectiveKind.ALL_REDUCE)
    states = make_final_states(spec)
    duplicate = PayloadState(
        state_id="duplicate",
        version=1,
        rank=states[0].rank,
        logical_address=states[0].logical_address,
        contributors=states[0].contributors,
        ready_time=2.0,
        active=True,
        member_paths=states[0].member_paths,
    )

    with pytest.raises(SemanticError, match="duplicate final output"):
        check_final_states(spec, 2, 2, states + (duplicate,))


def test_misaddressed_final_state_is_rejected():
    spec = make_spec(CollectiveKind.ALL_GATHER)
    states = list(make_final_states(spec))
    original = states[0]
    states[0] = PayloadState(
        state_id=original.state_id,
        version=original.version,
        rank=original.rank,
        logical_address=1,
        contributors=original.contributors,
        ready_time=original.ready_time,
        active=True,
        member_paths=original.member_paths,
    )

    with pytest.raises(SemanticError, match="logical address"):
        check_final_states(spec, 2, 2, states)


def test_inactive_final_state_is_rejected():
    spec = make_spec(CollectiveKind.REDUCE)
    states = list(make_final_states(spec))
    original = states[0]
    states[0] = PayloadState(
        state_id=original.state_id,
        version=original.version,
        rank=original.rank,
        logical_address=original.logical_address,
        contributors=original.contributors,
        ready_time=original.ready_time,
        active=False,
        member_paths=original.member_paths,
    )

    with pytest.raises(SemanticError, match="inactive"):
        check_final_states(spec, 2, 2, states)


def payload_state(state_id, rank, logical_address, contributors):
    contributor_set = frozenset(contributors)
    return PayloadState(
        state_id=state_id,
        version=0,
        rank=rank,
        logical_address=logical_address,
        contributors=contributor_set,
        ready_time=1.0,
        active=True,
        member_paths=tuple(
            (slice_id, ()) for slice_id in sorted(contributor_set)
        ),
    )


def test_final_state_contributor_range_and_logical_position_are_checked():
    spec = make_spec(CollectiveKind.ALL_REDUCE)

    with pytest.raises(SemanticError, match="global range"):
        check_final_states(spec, 2, 2, [payload_state("bad", 0, 0, {4})])
    with pytest.raises(SemanticError, match="different logical addresses"):
        check_final_states(spec, 2, 2, [payload_state("bad", 0, 0, {0, 1})])


def test_final_state_rank_is_checked():
    spec = make_spec(CollectiveKind.ALL_GATHER)

    with pytest.raises(SemanticError, match="rank range"):
        check_final_states(spec, 2, 2, [payload_state("bad", 2, 0, {0})])


@pytest.mark.parametrize(
    "kind",
    [CollectiveKind.ALL_GATHER, CollectiveKind.ALL_TO_ALL],
)
def test_non_reduction_outputs_require_single_contributor(kind):
    spec = make_spec(kind)

    with pytest.raises(SemanticError, match="one contributor"):
        check_final_states(spec, 2, 2, [payload_state("bad", 0, 0, {0, 2})])


def test_final_state_values_and_ids_are_checked():
    spec = make_spec(CollectiveKind.BROADCAST)
    states = list(make_final_states(spec))

    with pytest.raises(SemanticError, match="PayloadState"):
        check_final_states(spec, 2, 2, [object()])
    duplicate_id = payload_state(
        states[0].state_id,
        states[1].rank,
        states[1].logical_address,
        states[1].contributors,
    )
    with pytest.raises(SemanticError, match="state IDs must be unique"):
        check_final_states(spec, 2, 2, [states[0], duplicate_id])


def test_extra_final_output_is_rejected():
    spec = make_spec(CollectiveKind.REDUCE)
    states = make_final_states(spec)
    extra = payload_state("extra", 1, 0, {0, 2})

    with pytest.raises(SemanticError, match="extra final output"):
        check_final_states(spec, 2, 2, states + (extra,))


def test_incorrect_final_contributors_are_rejected():
    spec = make_spec(CollectiveKind.BROADCAST)
    states = list(make_final_states(spec))
    states[0] = payload_state(
        states[0].state_id,
        states[0].rank,
        states[0].logical_address,
        {2},
    )

    with pytest.raises(SemanticError, match="final contributors"):
        check_final_states(spec, 2, 2, states)
