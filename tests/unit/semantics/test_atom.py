from types import MappingProxyType

import pytest

from vericcl.errors import SemanticError
from vericcl.input.models import ForbiddenTransfer
from vericcl.semantics.atom import Atom, PathStage, Schedule, Symbol, Transfer


pytestmark = pytest.mark.phase01


def make_atom(
    *,
    slice_id=0,
    size_bytes=1024,
    symbols=(Symbol(0, 1, 0.0),),
    operator="SEND",
    stage_id=0,
    st_time=0.0,
    ed_time=2.0,
):
    return Atom(
        slice_id=slice_id,
        slice_size_bytes=size_bytes,
        path=(PathStage(stage_id, operator, symbols),),
        st_time=st_time,
        ed_time=ed_time,
    )


def make_transfer(
    *,
    transfer_id="transfer-0",
    member_slice_ids=frozenset({0}),
    atoms=None,
    kind="SEND",
    src_rank=0,
    dst_rank=1,
    channel=0,
    stage_id=0,
    st_time=0.0,
    ed_time=2.0,
    predecessor_ids=frozenset(),
):
    if atoms is None:
        atoms = tuple(
            make_atom(
                slice_id=slice_id,
                symbols=(Symbol(src_rank, dst_rank, st_time),),
                operator=kind,
                stage_id=stage_id,
                st_time=st_time,
                ed_time=ed_time,
            )
            for slice_id in sorted(member_slice_ids)
        )
    return Transfer(
        transfer_id=transfer_id,
        kind=kind,
        src_rank=src_rank,
        dst_rank=dst_rank,
        channel=channel,
        stage_id=stage_id,
        member_slice_ids=member_slice_ids,
        atoms=atoms,
        st_time=st_time,
        ed_time=ed_time,
        predecessor_ids=predecessor_ids,
    )


def test_atom_path_ends_at_current_rank():
    atom = make_atom(
        symbols=(Symbol(0, 1, 0.0), Symbol(1, 2, 4.0)),
        st_time=4.0,
        ed_time=6.0,
    )

    atom.validate_path_prefix(current_rank=2, slice_count=4)


def test_atom_path_must_start_at_slice_source():
    atom = make_atom(
        slice_id=4,
        symbols=(Symbol(0, 1, 0.0),),
    )

    with pytest.raises(SemanticError, match="source rank"):
        atom.validate_path_prefix(current_rank=1, slice_count=4)


def test_atom_path_must_be_a_contiguous_chain():
    with pytest.raises(SemanticError, match="contiguous"):
        make_atom(symbols=(Symbol(0, 1, 0.0), Symbol(2, 3, 1.0)))


def test_atom_start_must_follow_current_ready_time():
    with pytest.raises(SemanticError, match="ready_time"):
        make_atom(symbols=(Symbol(0, 1, 3.0),), st_time=2.0, ed_time=4.0)


def test_shared_transfer_counts_physical_bytes_once():
    transfer = make_transfer(member_slice_ids=frozenset({0, 4}))

    assert transfer.physical_bytes == 1024


def test_transfer_requires_one_atom_per_member_slice():
    atom = make_atom(slice_id=0)

    with pytest.raises(SemanticError, match="member_slice_ids"):
        make_transfer(member_slice_ids=frozenset({0, 4}), atoms=(atom,))


def test_transfer_atom_must_end_with_physical_operation():
    atom = make_atom(symbols=(Symbol(0, 2, 0.0),))

    with pytest.raises(SemanticError, match="physical operation"):
        make_transfer(atoms=(atom,), src_rank=0, dst_rank=1)


def test_forbidden_transfer_matches_any_shared_member():
    transfer = make_transfer(
        member_slice_ids=frozenset({0, 4}),
        kind="REDUCE",
    )
    forbidden = [ForbiddenTransfer(4, 0, 1, 0)]

    assert transfer.is_forbidden(forbidden) is True


def test_forbidden_transfer_requires_all_four_fields_to_match():
    transfer = make_transfer(member_slice_ids=frozenset({0, 4}))
    forbidden = [ForbiddenTransfer(4, 0, 1, 1)]

    assert transfer.is_forbidden(forbidden) is False


def test_schedule_validates_global_slice_identity_and_sizes():
    schedule = Schedule(
        schedule_id="schedule-0",
        transfers=(make_transfer(),),
        final_state_ids=("state-0",),
        rank_count=2,
        slice_count=4,
        slice_size_bytes=1024,
        metadata={"candidate": "test"},
    )

    assert schedule.transfers[0].transfer_id == "transfer-0"
    assert isinstance(schedule.metadata, MappingProxyType)


def test_schedule_rejects_duplicate_transfer_ids():
    first = make_transfer()
    second = make_transfer()

    with pytest.raises(SemanticError, match="unique"):
        Schedule(
            schedule_id="schedule-0",
            transfers=(first, second),
            final_state_ids=(),
            rank_count=2,
            slice_count=4,
            slice_size_bytes=1024,
            metadata={},
        )


def test_schedule_rejects_slice_id_outside_global_range():
    transfer = make_transfer(member_slice_ids=frozenset({8}))

    with pytest.raises(SemanticError, match="slice_id"):
        Schedule(
            schedule_id="schedule-0",
            transfers=(transfer,),
            final_state_ids=(),
            rank_count=2,
            slice_count=4,
            slice_size_bytes=1024,
            metadata={},
        )
