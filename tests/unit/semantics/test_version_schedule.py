from dataclasses import replace

import pytest

from vericcl.errors import SemanticError

from tests.unit.input.v5_support import request_documents, write_request


def allreduce_schedule(tmp_path):
    from vericcl.input.loader import load_request
    from vericcl.semantics.ranges import DataRange
    from vericcl.semantics.versions import BufferRegion, StateVersion
    from vericcl.semantics.version_schedule import OutputBinding, VersionOperation, VersionSchedule
    docs = request_documents("allreduce", ranks=2)
    docs["sketch"]["hyperparameters"].update(total_size_bytes=4, slice_size_bytes=4)
    inputs = load_request(write_request(tmp_path, docs))
    initial = tuple(StateVersion("i" + str(rank), rank, 0, 4, (DataRange(rank, 0, 4),),
                                BufferRegion(rank, "input", 0, 4, "input"), 0) for rank in range(2))
    operations = (
        VersionOperation("r", "reduce", ("i0", "i1"), "sum", BufferRegion(0, "scratch", 0, 4, "scratch"), 0, 1, 0),
        VersionOperation("c0", "copy", ("sum",), "out0", BufferRegion(0, "output", 0, 4, "output"), 1, 2, 1),
        VersionOperation("c1", "copy", ("sum",), "out1", BufferRegion(1, "output", 0, 4, "output"), 1, 2, 1),
    )
    schedule = VersionSchedule(2, 4, 4, 2, initial, operations,
                               (OutputBinding(0, 0, "out0"), OutputBinding(1, 0, "out1")))
    return inputs, schedule


def test_complete_schedule_replays_states_and_checks_exact_final_contributions(tmp_path):
    from vericcl.verification.semantics import check_semantics
    inputs, schedule = allreduce_schedule(tmp_path)
    result = check_semantics(schedule, inputs)
    assert result.status.value == "valid", result
    assert result.evidence["output_bytes"] == {0: 4, 1: 4}


@pytest.mark.parametrize("defect", ["missing_output", "duplicate_output", "early_read", "invented_input", "forbidden_edge"])
def test_schedule_checker_reports_concrete_semantic_counterexamples(tmp_path, defect):
    from vericcl.input.models import AtomConstraints, ForbiddenTransfer
    from vericcl.verification.semantics import check_semantics
    inputs, schedule = allreduce_schedule(tmp_path)
    if defect == "missing_output":
        schedule = replace(schedule, outputs=schedule.outputs[:1])
    elif defect == "duplicate_output":
        schedule = replace(schedule, outputs=schedule.outputs + schedule.outputs[:1])
    elif defect == "early_read":
        operations = list(schedule.operations)
        operations[1] = replace(operations[1], start_us=0.5)
        schedule = replace(schedule, operations=tuple(operations))
    elif defect == "invented_input":
        schedule = replace(schedule, initial_states=schedule.initial_states[:1])
    elif defect == "forbidden_edge":
        inputs = replace(inputs, atom_constraints=AtomConstraints(None, (ForbiddenTransfer(1, 1, 0, 0),)))
    result = check_semantics(schedule, inputs)
    assert result.status.value == "invalid"
    assert result.message and result.evidence


def test_initial_state_ranges_must_use_the_resolved_slice_partition(tmp_path):
    from vericcl.verification.semantics import check_semantics
    inputs, schedule = allreduce_schedule(tmp_path)
    inputs = replace(inputs, hyperparameters=replace(inputs.hyperparameters, slice_size_bytes=2))
    schedule = replace(schedule, slice_size_bytes=2)
    result = check_semantics(schedule, inputs)
    assert result.status.value == "invalid"
    assert "slice" in result.message


@pytest.mark.parametrize("operator", ["broadcast", "reduce", "allgather", "allreduce", "alltoall", "reduce_scatter"])
def test_output_byte_positions_and_sources_match_each_external_operator(operator):
    from vericcl.semantics.collective import CollectiveKind, CollectiveSpec
    from vericcl.semantics.ranges import DataRange
    from vericcl.semantics.versions import BufferRegion, StateVersion, VersionLedger
    from vericcl.semantics.version_schedule import OutputBinding, check_version_outputs
    # Explicit small examples independent of the checker's interval constructor.
    cases = {
        "broadcast": [(0, 0, 0, 8, (1,)), (1, 0, 0, 8, (1,))],
        "reduce": [(1, 0, 0, 8, (0, 1))],
        "allgather": [(0, 0, 0, 8, (0,)), (0, 8, 0, 8, (1,)),
                      (1, 0, 0, 8, (0,)), (1, 8, 0, 8, (1,))],
        "allreduce": [(0, 0, 0, 8, (0, 1)), (1, 0, 0, 8, (0, 1))],
        "alltoall": [(0, 0, 0, 4, (0,)), (0, 4, 0, 4, (1,)),
                     (1, 0, 4, 8, (0,)), (1, 4, 4, 8, (1,))],
        "reduce_scatter": [(0, 0, 0, 4, (0, 1)), (1, 0, 4, 8, (0, 1))],
    }
    ledger = VersionLedger(scratch_budgets={0: 0, 1: 0})
    outputs = []
    for index, (rank, offset, start, end, sources) in enumerate(cases[operator]):
        state_id = "s" + str(index)
        ledger.add_initial(StateVersion(state_id, rank, start, end,
                           tuple(DataRange(source, start, end) for source in sources),
                           BufferRegion(rank, "output", offset, end - start), 0))
        outputs.append(OutputBinding(rank, offset, state_id))
    kind = CollectiveKind(operator)
    spec = CollectiveSpec(kind, "float32", "sum" if operator in {"reduce", "allreduce", "reduce_scatter"} else None,
                          1 if operator in {"broadcast", "reduce"} else None)
    totals = check_version_outputs(spec, 2, 8, outputs, ledger, at_us=0)
    assert sum(totals.values()) == sum(end - start for _, _, start, end, _ in cases[operator])
    with pytest.raises(SemanticError, match="incomplete"):
        check_version_outputs(spec, 2, 8, outputs[:-1], ledger, at_us=0)
