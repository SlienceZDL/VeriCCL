from dataclasses import replace

import pytest

from vericcl.errors import SemanticError
from tests.unit.input.v5_support import request_documents, write_request


def constraint_set(tmp_path, reference=256, current=384, total=1536):
    from vericcl.input.loader import load_request
    from vericcl.semantics.ranges import bind_constraints
    docs = request_documents("allreduce", ranks=4)
    docs["sketch"]["hyperparameters"].update(total_size_bytes=total, slice_size_bytes=current,
                                              reference_slice_size_bytes=reference)
    docs["atom"]["forbidden_transfers"] = [[1, 1, 2, 1]]
    return bind_constraints(load_request(write_request(tmp_path, docs)))


def test_non_aligned_repartition_keeps_both_intersecting_slices_restricted(tmp_path):
    from vericcl.semantics.ranges import DataRange
    constraints = constraint_set(tmp_path)
    assert not constraints.allows((DataRange(0, 0, 384),), 1, 2, 1)
    assert not constraints.allows((DataRange(0, 384, 768),), 1, 2, 1)
    assert constraints.allows((DataRange(0, 768, 1152),), 1, 2, 1)
    assert constraints.allows((DataRange(1, 0, 384),), 1, 2, 1)


def test_smaller_partition_and_merged_payload_inherit_original_edge_restriction(tmp_path):
    from vericcl.semantics.ranges import DataRange
    constraints = constraint_set(tmp_path, current=128)
    assert not constraints.allows((DataRange(0, 256, 384), DataRange(1, 256, 384)), 1, 2, 1)
    assert not constraints.allows((DataRange(0, 384, 512),), 1, 2, 1)
    assert constraints.allows((DataRange(0, 512, 640),), 1, 2, 1)


def test_a_forbidden_directed_edge_does_not_forbid_a_legal_detour(tmp_path):
    from vericcl.semantics.ranges import DataRange
    constraints = constraint_set(tmp_path)
    payload = (DataRange(0, 256, 512),)
    assert not constraints.allows(payload, 1, 2, 1)
    assert constraints.allows(payload, 1, 3, 1)
    assert constraints.allows(payload, 3, 2, 1)
    assert constraints.allows(payload, 2, 1, 1)
    assert constraints.allows(payload, 1, 2, 0)


def test_stage_remapping_must_preserve_every_restricted_logical_stage(tmp_path):
    from vericcl.semantics.ranges import DataRange
    constraints = constraint_set(tmp_path)
    with pytest.raises(SemanticError, match="mapping"):
        constraints.remap_stages({0: (0,)}, new_stage_count=3)
    mapped = constraints.remap_stages({1: (2, 3)}, new_stage_count=4)
    for stage in (2, 3):
        assert not mapped.allows((DataRange(0, 256, 512),), 1, 2, stage)
    assert mapped.allows((DataRange(0, 256, 512),), 1, 2, 1)


def test_fixed_stage_count_survives_structural_candidates(tmp_path):
    from vericcl.input.loader import load_request
    from vericcl.semantics.ranges import bind_constraints
    docs = request_documents()
    docs["atom"].update(stage_num=2, forbidden_transfers=[[0, 0, 1, 1]])
    constraints = bind_constraints(load_request(write_request(tmp_path, docs)))
    with pytest.raises(SemanticError, match="stage count"):
        constraints.remap_stages({1: (2,)}, new_stage_count=3)


def test_reference_ids_cover_the_same_original_ranges_after_repartition():
    from vericcl.semantics.ranges import DataRange, reference_slice_ids
    assert reference_slice_ids(DataRange(2, 384, 768), input_bytes=1536, reference_slice_bytes=256) == (13, 14)


def test_storage_versions_allow_multiple_results_and_only_overwrite_the_target():
    from vericcl.semantics.ranges import DataRange
    from vericcl.semantics.versions import BufferRegion, StateVersion, VersionLedger
    ledger = VersionLedger(scratch_budgets={0: 1024})
    p = StateVersion("p", 0, 0, 64, (DataRange(0, 0, 64),), BufferRegion(0, "A", 0, 64), 0)
    q = StateVersion("q", 0, 0, 64, (DataRange(1, 0, 64),), BufferRegion(0, "B", 0, 64), 0)
    ledger.add_initial(p)
    ledger.add_initial(q)
    result = ledger.reduce("p", "q", state_id="sum", destination=p.storage, start_us=1, end_us=2)
    assert not ledger.is_active("p")
    assert ledger.is_active("q")
    assert result.contributors == (DataRange(0, 0, 64), DataRange(1, 0, 64))
    copied = ledger.copy("sum", state_id="sum-copy", destination=BufferRegion(0, "C", 0, 64), start_us=2, end_us=3)
    assert ledger.is_active("sum") and ledger.is_active(copied.state_id)
    ledger.release("q", at_us=2)
    assert not ledger.is_active("q")


def test_a_later_reader_prevents_early_overwrite_or_release():
    from vericcl.semantics.ranges import DataRange
    from vericcl.semantics.versions import BufferRegion, StateVersion, VersionLedger
    ledger = VersionLedger(scratch_budgets={0: 1024})
    p = StateVersion("p", 0, 0, 64, (DataRange(0, 0, 64),), BufferRegion(0, "A", 0, 64), 0)
    q = StateVersion("q", 0, 0, 64, (DataRange(1, 0, 64),), BufferRegion(0, "B", 0, 64), 0)
    ledger.add_initial(p)
    ledger.add_initial(q)
    ledger.reserve_read("p", "send", start_us=1, end_us=4)
    with pytest.raises(SemanticError, match="reader"):
        ledger.reduce("p", "q", state_id="sum", destination=p.storage, start_us=2, end_us=3)
    with pytest.raises(SemanticError, match="reader"):
        ledger.release("p", at_us=3)
    result = ledger.reduce("p", "q", state_id="sum", destination=p.storage, start_us=4, end_us=5)
    assert ledger.is_active(result.state_id)


def test_overlapping_storage_and_scratch_budget_are_checked():
    from vericcl.semantics.ranges import DataRange
    from vericcl.semantics.versions import BufferRegion, StateVersion, VersionLedger
    ledger = VersionLedger(scratch_budgets={0: 64})
    p = StateVersion("p", 0, 0, 64, (DataRange(0, 0, 64),), BufferRegion(0, "A", 0, 64), 0)
    ledger.add_initial(p)
    ledger.copy("p", state_id="copy", destination=BufferRegion(0, "scratch", 0, 64, "scratch"), start_us=0, end_us=1)
    with pytest.raises(SemanticError, match="scratch"):
        ledger.copy("p", state_id="copy2", destination=BufferRegion(0, "scratch", 64, 64, "scratch"), start_us=1, end_us=2)
    with pytest.raises(SemanticError, match="overlap"):
        ledger.copy("p", state_id="partial", destination=BufferRegion(0, "A", 16, 64), start_us=2, end_us=3)


def test_external_reader_names_cannot_collide_with_internal_operation_reads():
    from vericcl.semantics.ranges import DataRange
    from vericcl.semantics.versions import BufferRegion, StateVersion, VersionLedger
    ledger = VersionLedger(scratch_budgets={0: 0})
    p = StateVersion("p", 0, 0, 4, (DataRange(0, 0, 4),), BufferRegion(0, "A", 0, 4), 0)
    q = StateVersion("q", 0, 0, 4, (DataRange(1, 0, 4),), BufferRegion(0, "B", 0, 4), 0)
    ledger.add_initial(p)
    ledger.add_initial(q)
    ledger.reserve_read("q", "sum:input:0", start_us=4, end_us=5)
    ledger.reduce("p", "q", state_id="sum", destination=p.storage, start_us=0, end_us=1)
    with pytest.raises(SemanticError, match="reader"):
        ledger.release("q", at_us=2)


def test_an_inflight_write_cannot_be_overwritten_before_it_finishes():
    from vericcl.semantics.ranges import DataRange
    from vericcl.semantics.versions import BufferRegion, StateVersion, VersionLedger
    ledger = VersionLedger(scratch_budgets={0: 0})
    source = StateVersion("p", 0, 0, 4, (DataRange(0, 0, 4),), BufferRegion(0, "input", 0, 4), 0)
    ledger.add_initial(source)
    destination = BufferRegion(0, "output", 0, 4)
    ledger.copy("p", state_id="first", destination=destination, start_us=0, end_us=2)
    with pytest.raises(SemanticError, match="write"):
        ledger.copy("p", state_id="second", destination=destination, start_us=1, end_us=3)
