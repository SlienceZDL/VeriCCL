from types import MappingProxyType

import pytest

from vericcl.errors import SemanticError
from vericcl.input.models import ForbiddenTransfer
from vericcl.semantics.atom import Atom, PathStage, Schedule, Symbol, Transfer
from vericcl.semantics.atom import symbolic_step


pytestmark = pytest.mark.phase01


def make_atom(
    *,
    slice_id=0,
    size_bytes=1024,
    symbols=(symbolic_step(0, 1, 0.0),),
    operator="SEND",
    stage_id=0,
    st_time=0.0,
    ed_time=2.0,
):
    return Atom.build(
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
                symbols=(symbolic_step(src_rank, dst_rank, st_time),),
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
        symbols=(symbolic_step(0, 1, 0.0), symbolic_step(1, 2, 4.0)),
        st_time=4.0,
        ed_time=6.0,
    )

    atom.validate_path_prefix(current_rank=2, slice_count=4)


def test_atom_path_must_start_at_slice_source():
    atom = make_atom(
        slice_id=4,
        symbols=(symbolic_step(0, 1, 0.0),),
    )

    with pytest.raises(SemanticError, match="source rank"):
        atom.validate_path_prefix(current_rank=1, slice_count=4)


def test_atom_path_must_be_a_contiguous_chain():
    with pytest.raises(SemanticError, match="contiguous"):
        make_atom(symbols=(symbolic_step(0, 1, 0.0), symbolic_step(2, 3, 1.0)))


def test_atom_start_must_follow_current_ready_time():
    with pytest.raises(SemanticError, match="ready_time"):
        make_atom(symbols=(symbolic_step(0, 1, 3.0),), st_time=2.0, ed_time=4.0)


def test_shared_transfer_counts_physical_bytes_once():
    transfer = make_transfer(member_slice_ids=frozenset({0, 4}))

    assert transfer.physical_bytes == 1024


def test_transfer_requires_one_atom_per_member_slice():
    atom = make_atom(slice_id=0)

    with pytest.raises(SemanticError, match="member_slice_ids"):
        make_transfer(member_slice_ids=frozenset({0, 4}), atoms=(atom,))


def test_transfer_atom_must_end_with_physical_operation():
    atom = make_atom(symbols=(symbolic_step(0, 2, 0.0),))

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


@pytest.mark.parametrize(
    "args,match",
    [
        ((True, 1, 0.0), "must be an integer"),
        ((-1, 1, 0.0), "must be at least"),
        ((0, 0, 0.0), "distinct"),
        ((0, 1, "now"), "must be a number"),
        ((0, 1, float("inf")), "finite"),
    ],
)
def test_symbol_rejects_invalid_fields(args, match):
    with pytest.raises(SemanticError, match=match):
        symbolic_step(*args)


def test_path_stage_validates_operator_symbols_and_count():
    stage = PathStage(0, "SEND", (symbolic_step(0, 1, 0.0),))

    assert stage.operation_count == 1
    with pytest.raises(SemanticError, match="operator"):
        PathStage(0, "COPY", (symbolic_step(0, 1, 0.0),))
    with pytest.raises(SemanticError, match="must not be empty"):
        PathStage(0, "SEND", ())
    with pytest.raises(SemanticError, match="Symbol"):
        PathStage(0, "SEND", (object(),))


def test_atom_rejects_invalid_path_and_time_fields():
    with pytest.raises(SemanticError, match="must not be empty"):
        Atom.build(0, 1, (), 0.0, 1.0)
    with pytest.raises(SemanticError, match="PathStage"):
        Atom.build(0, 1, (object(),), 0.0, 1.0)
    with pytest.raises(SemanticError, match="must not exceed"):
        make_atom(st_time=2.0, ed_time=1.0)


def test_atom_rejects_non_increasing_stages_and_ready_times():
    first = PathStage(0, "SEND", (symbolic_step(0, 1, 2.0),))
    repeated = PathStage(0, "SEND", (symbolic_step(1, 2, 3.0),))
    earlier = PathStage(1, "SEND", (symbolic_step(1, 2, 1.0),))

    with pytest.raises(SemanticError, match="stage IDs"):
        Atom.build(0, 1, (first, repeated), 3.0, 4.0)
    with pytest.raises(SemanticError, match="non-decreasing"):
        Atom.build(0, 1, (first, earlier), 3.0, 4.0)


def test_atom_path_prefix_rejects_invalid_geometry_and_destination():
    atom = make_atom()

    with pytest.raises(SemanticError, match="slice_count"):
        atom.validate_path_prefix(current_rank=1, slice_count=0)
    with pytest.raises(SemanticError, match="current rank"):
        atom.validate_path_prefix(current_rank=2)


def test_transfer_rejects_invalid_identity_and_shape():
    atom = make_atom()

    with pytest.raises(SemanticError, match="non-empty string"):
        make_transfer(transfer_id="", atoms=(atom,))
    with pytest.raises(SemanticError, match="kind"):
        make_transfer(kind="COPY", atoms=(atom,))
    with pytest.raises(SemanticError, match="distinct"):
        make_transfer(src_rank=0, dst_rank=0, atoms=(atom,))
    with pytest.raises(SemanticError, match="member_slice_ids"):
        make_transfer(member_slice_ids=frozenset(), atoms=())
    with pytest.raises(SemanticError, match="Atom values"):
        make_transfer(atoms=(object(),))
    with pytest.raises(SemanticError, match="depend on itself"):
        make_transfer(predecessor_ids=frozenset({"transfer-0"}))


def test_transfer_rejects_mismatched_atom_sizes_and_times():
    first = make_atom(slice_id=0, size_bytes=1024)
    second = make_atom(slice_id=4, size_bytes=2048)

    with pytest.raises(SemanticError, match="equal slice sizes"):
        make_transfer(
            member_slice_ids=frozenset({0, 4}),
            atoms=(first, second),
        )
    # Atoms carry their own t = (st, ed); they may not start before the transfer.
    with pytest.raises(SemanticError, match="cannot start before its physical transfer"):
        make_transfer(atoms=(first,), st_time=1.0, ed_time=2.0)


def test_schedule_rejects_invalid_members_and_dependencies():
    transfer = make_transfer()
    missing_predecessor = make_transfer(
        predecessor_ids=frozenset({"missing"}),
    )

    with pytest.raises(SemanticError, match="Transfer values"):
        Schedule("schedule", (object(),), (), 2, 4, 1024, {})
    with pytest.raises(SemanticError, match="final state IDs must be unique"):
        Schedule("schedule", (transfer,), ("a", "a"), 2, 4, 1024, {})
    with pytest.raises(SemanticError, match="metadata"):
        Schedule("schedule", (transfer,), (), 2, 4, 1024, [])
    with pytest.raises(SemanticError, match="predecessor"):
        Schedule("schedule", (missing_predecessor,), (), 2, 4, 1024, {})
    with pytest.raises(SemanticError, match="slice size"):
        Schedule("schedule", (transfer,), (), 2, 4, 2048, {})


def test_schedule_freezes_nested_metadata():
    schedule = Schedule(
        "schedule",
        (make_transfer(),),
        (),
        2,
        4,
        1024,
        {"list": [1], "tuple": (2,), "set": {3}},
    )

    assert schedule.metadata["list"] == (1,)
    assert schedule.metadata["tuple"] == (2,)
    assert schedule.metadata["set"] == frozenset({3})


def test_stage_suffix_requires_an_exact_declared_path_root():
    transfer = make_transfer(member_slice_ids=frozenset({4}))

    with pytest.raises(SemanticError, match="path_roots"):
        Schedule(
            "schedule",
            (transfer,),
            (),
            2,
            4,
            1024,
            {"path_scope": "stage_suffix"},
        )
    with pytest.raises(SemanticError, match="declared root"):
        Schedule(
            "schedule",
            (transfer,),
            (),
            2,
            4,
            1024,
            {
                "path_scope": "stage_suffix",
                "path_roots": {"transfer-0": 1},
            },
        )

    schedule = Schedule(
        "schedule",
        (transfer,),
        (),
        2,
        4,
        1024,
        {
            "path_scope": "stage_suffix",
            "path_roots": {"transfer-0": 0},
        },
    )
    assert schedule.metadata["path_roots"]["transfer-0"] == 0


def test_stage_suffix_accepts_one_declared_root_per_member_atom():
    atoms = (
        make_atom(
            slice_id=1,
            symbols=(symbolic_step(1, 0, 2.0),),
            operator="REDUCE",
            st_time=2.0,
            ed_time=4.0,
        ),
        make_atom(
            slice_id=2,
            symbols=(symbolic_step(2, 1, 0.0), symbolic_step(1, 0, 2.0)),
            operator="REDUCE",
            st_time=2.0,
            ed_time=4.0,
        ),
    )
    transfer = make_transfer(
        member_slice_ids=frozenset({1, 2}),
        atoms=atoms,
        kind="REDUCE",
        src_rank=1,
        dst_rank=0,
        st_time=2.0,
        ed_time=4.0,
    )

    schedule = Schedule(
        "schedule",
        (transfer,),
        (),
        3,
        1,
        1024,
        {
            "path_scope": "stage_suffix",
            "path_roots": {"transfer-0": {1: 1, 2: 2}},
        },
    )

    assert schedule.metadata["path_roots"]["transfer-0"][2] == 2


def test_atom_is_slice_path_time_vectors_built_by_symbolic_execution():
    from vericcl.semantics.atom import PathDAG, SliceVec, TimeVec
    from vericcl.symbolic.terms import SymValue
    first, second = symbolic_step(0, 1, 1.0), symbolic_step(1, 2, 3.0)
    atom = Atom.build(slice_id=0, slice_size_bytes=64,
                      path=(PathStage(0, "SEND", (first,)), PathStage(1, "SEND", (second,))),
                      st_time=3.0, ed_time=5.0)
    assert isinstance(atom.s, SliceVec) and (atom.s.slice_id, atom.s.slice_size, atom.s.stage) == (0, 64, 1)
    assert isinstance(atom.pd, PathDAG) and atom.pd.roots == {0} and atom.pd.current_rank == 2
    assert [(s.src_rank, s.dst_rank, s.stage_id) for s in atom.pd.symbols] == [(0, 1, 0), (1, 2, 1)]
    assert isinstance(atom.t, TimeVec) and isinstance(atom.t.st_time, SymValue)
    # ready(hop 2) is symbolically max(ready(hop 1), decided arrival of hop 2)
    hop2 = atom.pd.symbols[1].ready_term
    assert hop2.op == "max" and hop2.label == "previous,decided"
    assert atom.pd.symbols[0].ready_term in hop2.args
    # legacy views are derived from s, pd and t
    assert atom.slice_id == 0 and atom.stage_num == 2 and float(atom.st_time) == 3.0
    assert atom.current_symbol.dst_rank == 2


def test_symbols_cannot_be_constructed_outside_symbolic_execution():
    with pytest.raises(SemanticError, match="symbolic execution"):
        Symbol(0, 1, 0.0)


def test_rebuild_and_with_times_reexecute_the_path():
    atom = make_atom()
    moved = atom.with_times(1.0, 3.0)
    assert (float(moved.st_time), float(moved.ed_time)) == (1.0, 3.0)
    assert moved.pd.symbols == atom.pd.symbols
    assert atom.rebuild(slice_id=4).slice_id == 4
    with pytest.raises(SemanticError):
        atom.rebuild(stage=3)
