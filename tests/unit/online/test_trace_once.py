from dataclasses import replace
from pathlib import Path

import pytest

from tests.unit.online.test_trace_format import _two_rank_sidecar, _rank_records
from vericcl.verification.online.clock_sync import ClockSyncSample
from vericcl.verification.online.trace_format import encode_raw_trace
from vericcl.verification.online.trace_once import RawTraceCapture, TraceContext, capture_once


def capture_fixture(tmp_path, *, missing=(), overflow=(), twice=(), no_clock=(), reset=True):
    sidecar = _two_rank_sidecar()
    paths, metadata, clocks = {}, {}, []
    for rank in (0, 1):
        if rank in missing:
            continue
        records = _rank_records(sidecar, rank)
        if rank in twice:
            records += tuple(replace(record, iteration=1) for record in records)
        path = tmp_path / f'rank-{rank}.bin'
        path.write_bytes(encode_raw_trace(records, rank=rank, overflow=rank in overflow))
        paths[rank] = path
        metadata[rank] = dict(warmup_iterations=5, reset_after_warmup=reset, executed_after_reset=1,
            correct=True, gpu_completed=True, clock_domain='globaltimer_ns', target_event_time_us=2)
        if rank not in no_clock:
            clocks += [ClockSyncSample(rank, tick, tick / 1000 - .01, tick / 1000 + .01, 0, .01)
                       for tick in (1, 100)]
    return sidecar, RawTraceCapture(paths, tuple(clocks), metadata, 'unit_fixture')


class FixtureDriver:
    def __init__(self, result):
        self.result, self.plans = result, []
    def capture_once(self, plan):
        self.plans.append(plan)
        return self.result


def test_one_post_warmup_call_and_preallocated_capacity(tmp_path):
    sidecar, raw = capture_fixture(tmp_path)
    driver = FixtureDriver(raw)
    result = capture_once(sidecar, TraceContext(True, (0, 1), driver, allow_test_driver=True))
    assert result.status == 'complete' and result.global_complete
    assert driver.plans[0].timed_iterations == 1 and driver.plans[0].warmup_iterations == 5
    assert driver.plans[0].capacity_by_rank == {rank: len(_rank_records(sidecar, rank)) for rank in (0, 1)}
    assert set(result.local_steps) == {0, 1}
    assert result.global_span_us is not None


def test_disabled_trace_does_not_launch(tmp_path):
    sidecar, raw = capture_fixture(tmp_path)
    driver = FixtureDriver(raw)
    assert capture_once(sidecar, TraceContext(False, (0, 1), driver)).status == 'disabled'
    assert not driver.plans


def test_capacity_shortage_is_reported_before_launch(tmp_path):
    sidecar, raw = capture_fixture(tmp_path)
    driver = FixtureDriver(raw)
    result = capture_once(sidecar, TraceContext(True, (0, 1), driver,
        capacity_by_rank={0: 1, 1: 1}, allow_test_driver=True))
    assert result.status == 'capacity_insufficient' and not driver.plans


@pytest.mark.parametrize('option', ['missing', 'overflow', 'twice', 'no_clock'])
def test_partial_rank_or_clock_failure_preserves_other_rank(tmp_path, option):
    sidecar, raw = capture_fixture(tmp_path, **{option: (1,)})
    result = capture_once(sidecar, TraceContext(True, (0, 1), FixtureDriver(raw), allow_test_driver=True))
    assert not result.global_complete and result.status == 'partial'
    assert result.local_steps[0]
    assert any(issue['rank'] == 1 for issue in result.issues)
    if option == 'no_clock':
        assert result.local_steps[1] and 1 not in result.aligned_ranks


def test_reset_receipt_is_required_for_target_attribution(tmp_path):
    sidecar, raw = capture_fixture(tmp_path, reset=False)
    result = capture_once(sidecar, TraceContext(True, (0, 1), FixtureDriver(raw), allow_test_driver=True))
    assert not result.local_steps and not result.global_complete


def test_clock_uncertainty_retains_local_durations_but_marks_cross_rank_order(tmp_path):
    sidecar, raw = capture_fixture(tmp_path)
    raw = replace(raw, clock_samples=tuple(replace(sample, reference_uncertainty_us=10) for sample in raw.clock_samples))
    result = capture_once(sidecar, TraceContext(True, (0, 1), FixtureDriver(raw), allow_test_driver=True))
    assert result.local_steps[0] and result.local_steps[1]
    assert result.uncertain_relations and all(step['transfer_us'] >= 0 for steps in result.local_steps.values() for step in steps)


def test_warmup_reset_and_finish_exports_are_in_distributable_runtime():
    root = Path(__file__).resolve().parents[3]
    header = root / 'runtime/include/vericcl_trace_control.h'
    assert header.is_file()
    text = header.read_text()
    for symbol in ('vericclTracePause', 'vericclTraceReset', 'vericclTraceFinish'):
        assert symbol in text
    assert (root / 'runtime/msccl-trace/patches/0004-vericcl-trace-once-control.patch').is_file()


def test_v5_sidecar_maps_executable_steps_bytes_and_state_dependencies(tmp_path):
    from tests.integration.test_lowering_v5 import candidate_fixture
    from vericcl.xml.lower import lower_and_check
    from vericcl.xml.trace_sidecar import build_v5_trace_sidecar, TraceSidecar
    candidate, context = candidate_fixture(tmp_path)
    bundle = lower_and_check(candidate, context)
    sidecar = build_v5_trace_sidecar(bundle)
    native_steps = [(block, step) for block in bundle.program.blocks for step in block.steps if step.kind != 'nop']
    assert len(sidecar.entries) == len(native_steps)
    assert sidecar.xml_sha256 == bundle.xml_sha256
    assert all(entry.payload_bytes == context.inputs.hyperparameters.slice_size_bytes for entry in sidecar.entries.values())
    assert TraceSidecar.from_json_text(sidecar.to_json_text()) == sidecar
    for block in bundle.program.blocks:
        entries = sorted((entry for entry in sidecar.entries.values() if entry.rank == block.rank and entry.tb_id == block.tb_id), key=lambda entry: entry.step_index)
        assert [entry.step_index for entry in entries] == list(range(len(entries)))


def test_different_rank_invocation_ids_do_not_form_a_global_trace(tmp_path):
    sidecar, raw = capture_fixture(tmp_path)
    records = tuple(replace(record, iteration=1) for record in _rank_records(sidecar, 1))
    raw.paths[1].write_bytes(encode_raw_trace(records, rank=1))
    result = capture_once(sidecar, TraceContext(True, (0, 1), FixtureDriver(raw), allow_test_driver=True))
    assert result.status == 'partial' and result.global_span_us is None
    assert result.local_steps[0] and result.local_steps[1]
    assert any(issue['reason'] == 'target_invocation_identity_differs_across_ranks' for issue in result.issues)


def test_missing_single_step_retains_other_records_on_that_rank(tmp_path):
    sidecar, raw = capture_fixture(tmp_path)
    records = _rank_records(sidecar, 1)
    assert len(records) > 1
    raw.paths[1].write_bytes(encode_raw_trace(records[:-1], rank=1, capacity=len(records)))
    result = capture_once(sidecar, TraceContext(True, (0, 1), FixtureDriver(raw), allow_test_driver=True))
    assert result.status == 'partial' and len(result.local_steps[1]) == len(records) - 1
    assert any(issue['rank'] == 1 and issue['reason'] == 'missing_step' for issue in result.issues)
