from dataclasses import replace

import pytest

from tests.integration.test_lowering_v5 import candidate_fixture
from vericcl.xml.lower import lower_and_check


def average_bundle(tmp_path, operator='allreduce', *, measured=False):
    candidate, context = candidate_fixture(tmp_path, operator)
    summed = lower_and_check(candidate, context)
    inputs = replace(context.inputs, collective=replace(context.inputs.collective, reduction_op='avg'))
    model = context.model
    if measured:
        model = replace(model, version='measured-components', services={key: tuple(replace(value,
            source='measured', measured_at='2026-11-04T00:00:00Z') for value in pieces)
            for key, pieces in model.services.items()})
    context = replace(context, inputs=inputs, model=model)
    return summed, lower_and_check(candidate, context), context


@pytest.mark.parametrize('operator', ['reduce', 'allreduce', 'reduce_scatter'])
def test_avg_completion_includes_native_postdivision_after_all_rank_work(tmp_path, operator):
    summed, average, context = average_bundle(tmp_path, operator)
    assert average.can_execute and average.xml_sha256 == summed.xml_sha256
    tail = [op for op in average.executable.operations if op.operation_id.startswith('native-postscale:')]
    expected_ranks = {rank for rank, count in average.program.output_slots.items() if count}
    assert {int(op.operation_id.split(':')[-1]) for op in tail} == expected_ranks
    assert average.prediction.t_pred_us > summed.prediction.t_pred_us
    assert set(average.executable.output_operations) == {op.operation_id for op in tail}
    for op in tail:
        rank = int(op.operation_id.split(':')[-1])
        local_work = {step.operation_id for block in average.program.blocks if block.rank == rank
                      for step in block.steps if step.kind != 'nop'}
        assert set(op.dependencies) == local_work
        assert op.payload_bytes == average.program.output_slots[rank] * context.inputs.hyperparameters.slice_size_bytes
        assert op.source_override == 'estimated'


def test_avg_memory_proxy_remains_estimated_and_causal_analysis_handles_native_tail(tmp_path):
    from vericcl.verification.bottlenecks import analyze, AnalysisContext, SearchScope
    _, bundle, context = average_bundle(tmp_path, measured=True)
    assert bundle.prediction.model_coverage == 'partial'
    assert {segment.source for segment in bundle.prediction.segments
            if segment.operation_id.startswith('native-postscale:')} == {'estimated'}
    batch = analyze(bundle, context.model, SearchScope('avg', batch_size=2), AnalysisContext(context))
    assert batch.status == 'opportunities'
    assert any(cause.kind == 'fixed_native_postprocess' for cause in batch.causes)


def test_avg_native_tail_survives_repair_event_model_and_xml_reload(tmp_path):
    from vericcl.tuning.strategies import RepairContext
    from vericcl.tuning.event_columns import solve_event_columns
    from vericcl.xml.v5_lower import check_xml_bundle
    _, bundle, context = average_bundle(tmp_path)
    reloaded = check_xml_bundle(bundle.xml_text, bundle.sidecar, context)
    assert reloaded.can_execute and reloaded.prediction.exact_completion_ratio == bundle.prediction.exact_completion_ratio
    selected, evidence = solve_event_columns((bundle,), RepairContext(context))
    assert selected == 0 and evidence.primal == pytest.approx(bundle.prediction.t_pred_us)
