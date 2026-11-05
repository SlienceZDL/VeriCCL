from dataclasses import replace

import pytest

from tests.integration.test_lowering_v5 import candidate_fixture
from vericcl.errors import SemanticError
from vericcl.performance.events import WorkOperation
from vericcl.verification.online.clock_sync import ClockSyncSample
from vericcl.verification.online.trace_format import RawStepTraceRecord, encode_raw_trace
from vericcl.verification.online.trace_once import RawTraceCapture, TraceContext, capture_once
from vericcl.xml.lower import lower_and_check
from vericcl.xml.trace_sidecar import build_v5_trace_sidecar


def trace_fixture(tmp_path, *, bundle=None, finish_us=None, missing_keys=(), no_clock=(), uncertainty_us=0):
    """Synthetic timestamps exercise attribution; no GPU/performance claim."""
    if bundle is None:
        candidate, lowering = candidate_fixture(tmp_path, 'broadcast', ranks=3, channels=2)
        bundle = lower_and_check(candidate, lowering)
    else:
        lowering = None
    sidecar = build_v5_trace_sidecar(bundle)
    paths, metadata, clocks = {}, {}, []
    tmp_path.mkdir(parents=True, exist_ok=True)
    for rank in range(bundle.program.rank_count):
        entries = [entry for entry in sidecar.entries.values() if entry.rank == rank]
        records = []
        for entry in entries:
            if entry.key in missing_keys:
                continue
            key = entry.transfer_id
            start = 10000 + round(bundle.prediction.start_times[key] * 1000)
            end = 10000 + round((finish_us or {}).get(key, bundle.prediction.end_times[key]) * 1000)
            records.append(RawStepTraceRecord(rank, entry.tb_id, entry.step_index,
                entry.runtime_endpoint_type, entry.peer, entry.runtime_channel, 0,
                start, start, start, end, 0, 0))
        path = tmp_path / ('rank-%d.bin' % rank)
        path.write_bytes(encode_raw_trace(records, rank=rank, capacity=max(1, len(entries))))
        paths[rank] = path
        metadata[rank] = dict(warmup_iterations=5, reset_after_warmup=True, executed_after_reset=1,
            correct=True, gpu_completed=True, clock_domain='globaltimer_ns', target_event_time_us=2000)
        if rank not in no_clock:
            clocks.extend(ClockSyncSample(rank, tick, tick / 1000, tick / 1000, 0, uncertainty_us)
                          for tick in (0, 1000000000))
    raw = RawTraceCapture(paths, tuple(clocks), metadata, 'unit_fixture')
    class Driver:
        calls = 0
        def capture_once(self, plan):
            self.calls += 1
            return raw
    driver = Driver()
    trace = capture_once(sidecar, TraceContext(True, tuple(paths), driver, allow_test_driver=True))
    assert driver.calls == 1
    return bundle, lowering, trace, raw


def output_choices(bundle):
    storage = {op.operation_id: op for op in bundle.storage_plan.instructions}
    choices = [key for key in bundle.executable.output_operations if storage[key].rank != 0]
    assert len(choices) >= 2
    return choices[0], next(key for key in choices if storage[key].rank != storage[choices[0]].rank)


def test_trace_tail_uses_required_outputs_and_retains_observed_durations(tmp_path):
    from vericcl.verification.online.trace_causality import analyze_trace_causality
    bundle, _, _, _ = trace_fixture(tmp_path)
    first, _ = output_choices(bundle)
    _, _, trace, _ = trace_fixture(tmp_path / 'late', bundle=bundle, finish_us={first: 1000})
    report = analyze_trace_causality(bundle, trace)
    assert report['completion_tails'] == [first]
    assert report['definite_completion_tails'] == [first]
    assert report['completion_scope'] == 'all_necessary_outputs'
    assert report['operations'][first]['local_endpoints'][0]['transfer_us'] > 900
    assert report['priority_operations'][0] == first
    assert report['global_no_improvement_proven'] is False
    assert report['device_durations_us'] == {'0': 2000., '1': 2000., '2': 2000.}


def test_clock_bounds_preserve_all_possible_tails_without_certain_order(tmp_path):
    from vericcl.verification.online.trace_causality import analyze_trace_causality
    bundle, _, _, _ = trace_fixture(tmp_path)
    first, second = output_choices(bundle)
    _, _, trace, _ = trace_fixture(tmp_path / 'uncertain', bundle=bundle,
        finish_us={first: 1000, second: 1100}, uncertainty_us=10000)
    report = analyze_trace_causality(bundle, trace)
    assert {first, second} <= set(report['completion_tails'])
    assert not report['definite_completion_tails']
    assert report['operations'][second]['local_endpoints'][0]['transfer_us'] > 1000


def test_missing_endpoint_keeps_independent_observations_and_local_regions(tmp_path):
    from vericcl.verification.online.trace_causality import analyze_trace_causality
    bundle, _, _, _ = trace_fixture(tmp_path)
    sidecar = build_v5_trace_sidecar(bundle)
    lost = next(entry for entry in sidecar.entries.values() if entry.peer >= 0)
    _, _, trace, _ = trace_fixture(tmp_path / 'partial', bundle=bundle, missing_keys=(lost.key,))
    report = analyze_trace_causality(bundle, trace)
    assert report['status'] == 'partial'
    assert not report['operations'][lost.transfer_id]['endpoint_complete']
    assert report['operations'][lost.transfer_id]['local_endpoints']
    assert any(value['endpoint_complete'] for key, value in report['operations'].items() if key != lost.transfer_id)
    assert report['completion_scope'] != 'all_necessary_outputs'
    assert not report['global_no_improvement_proven']


def test_missing_clock_preserves_local_tail_and_never_invents_global_interval(tmp_path):
    from vericcl.verification.online.trace_causality import analyze_trace_causality
    bundle, _, _, _ = trace_fixture(tmp_path)
    first, _ = output_choices(bundle)
    storage = {op.operation_id: op for op in bundle.storage_plan.instructions}
    rank = storage[first].rank
    _, _, trace, _ = trace_fixture(tmp_path / 'local', bundle=bundle, no_clock=(rank,))
    report = analyze_trace_causality(bundle, trace)
    assert report['operations'][first]['end_us_bounds'] is None
    assert report['local_completion_tails'][str(rank)]
    assert report['operations'][first]['local_endpoints']
    assert first in report['priority_operations']


def test_trace_xml_and_independent_endpoint_mapping_must_match(tmp_path):
    from vericcl.verification.online.trace_causality import analyze_trace_causality
    bundle, _, trace, _ = trace_fixture(tmp_path)
    with pytest.raises(SemanticError, match='XML'):
        analyze_trace_causality(bundle, replace(trace, xml_sha256='0' * 64))
    rank = next(rank for rank, rows in trace.records_by_rank.items() if rows)
    rows = list(trace.records_by_rank[rank])
    rows[0] = replace(rows[0], metadata=replace(rows[0].metadata, transfer_id='unrelated-operation'))
    changed = dict(trace.records_by_rank, **{})
    changed[rank] = tuple(rows)
    with pytest.raises(SemanticError, match='mapping'):
        analyze_trace_causality(bundle, replace(trace, records_by_rank=changed))


def test_untraced_native_postprocess_has_no_fabricated_completion_time(tmp_path):
    from vericcl.verification.online.trace_causality import analyze_trace_causality
    bundle, _, trace, _ = trace_fixture(tmp_path)
    native = WorkOperation('native:avg', 'reduce', 'reduce:0', 1024,
                           dependencies=bundle.executable.output_operations)
    bundle = replace(bundle, executable=replace(bundle.executable,
        operations=bundle.executable.operations + (native,), output_operations=(native.operation_id,)))
    report = analyze_trace_causality(bundle, trace)
    assert report['unobserved_output_operations'] == ['native:avg']
    assert report['operations']['native:avg']['end_us_bounds'] is None
    assert report['completion_scope'] == 'observed_output_frontier'
    assert 'native:avg' not in report['completion_tails']
    assert report['completion_tails'] and report['device_durations_us']


def test_report_binding_includes_required_output_contract(tmp_path):
    from vericcl.verification.online.trace_causality import analyze_trace_causality, validate_trace_causality
    bundle, _, trace, _ = trace_fixture(tmp_path)
    report = analyze_trace_causality(bundle, trace)
    changed = replace(bundle, executable=replace(bundle.executable,
        output_operations=bundle.executable.output_operations[:1]))
    with pytest.raises(SemanticError, match='executable'):
        validate_trace_causality(report, changed)


def test_different_invocations_preserve_local_data_without_pairing_endpoints(tmp_path):
    from vericcl.verification.online.trace_causality import analyze_trace_causality
    bundle, _, trace, _ = trace_fixture(tmp_path)
    records = dict(trace.records_by_rank)
    records[1] = tuple(replace(row, raw=replace(row.raw, iteration=1)) for row in records[1])
    changed = replace(trace, records_by_rank=records, global_complete=False, status='partial')
    report = analyze_trace_causality(bundle, changed)
    mixed = {row.transfer_id for row in records[1] if row.metadata.peer >= 0}
    assert mixed and all(not report['operations'][key]['endpoint_complete'] for key in mixed)
    assert not report['completion_tails'] and not report['single_invocation']
    assert report['local_completion_tails']['1']


def test_interval_tail_candidates_match_independent_corner_enumeration():
    from itertools import product
    from random import Random
    from vericcl.verification.online.trace_causality import _tails
    random = Random(35)
    for _ in range(100):
        intervals = {str(key): sorted((random.randrange(8), random.randrange(8))) for key in range(4)}
        outcomes = [{key for key, value in zip(intervals, corner) if value == max(corner)}
                    for corner in product(*intervals.values())]
        possible, definite = _tails(intervals)
        assert set(possible) == set.union(*outcomes)
        assert set(definite) == set.intersection(*outcomes)


def skewed_shared_fixture(tmp_path):
    candidate, lowering = candidate_fixture(tmp_path, 'alltoall', ranks=4)
    bundle = lower_and_check(candidate, lowering)
    storage = bundle.storage_plan
    a = next(op.operation_id for op in storage.instructions if op.kind == 'send'
             and storage.values[op.inputs[0]].rank == 0 and op.rank == 1)
    b = next(op.operation_id for op in storage.instructions if op.kind == 'send'
             and storage.values[op.inputs[0]].rank == 2 and op.rank == 3)
    bundle = replace(bundle, executable=replace(bundle.executable, output_operations=(a,),
        operations=tuple(replace(op, resources=('shared-nic',)) if op.operation_id in {a, b} else op
                         for op in bundle.executable.operations)))
    _, _, trace, _ = trace_fixture(tmp_path / 'skewed', bundle=bundle)
    intervals = {(a, 0): (0, 100), (a, 1): (90, 110), (b, 2): (10, 30), (b, 3): (15, 35)}
    records = {}
    for rank, rows in trace.records_by_rank.items():
        changed = []
        for row in rows:
            interval = intervals.get((row.transfer_id, rank))
            if interval is not None:
                start, end = (value * 1000 for value in interval)
                row = replace(row, raw=replace(row.raw, tb_reach=start, dependency_done=start,
                    transfer_start=start, transfer_end=end))
            changed.append(row)
        records[rank] = tuple(changed)
    return bundle, replace(trace, records_by_rank=records), a, b


def test_endpoint_start_skew_keeps_cross_gpu_shared_resource_competitor(tmp_path):
    from vericcl.verification.online.trace_causality import analyze_trace_causality
    bundle, trace, a, b = skewed_shared_fixture(tmp_path)
    report = analyze_trace_causality(bundle, trace)
    assert b in report['priority_operations']
    groups = report['resource_overlap_groups'].values()
    assert any({a, b} <= set(group['operation_ids']) and group['clock_domain'] == 'aligned_global' for group in groups)
    assert report['operations'][a]['start_us_bounds'] == [0., 0.]


def test_missing_endpoint_uncertainty_stays_scoped_to_shared_resource(tmp_path):
    from vericcl.verification.online.trace_causality import analyze_trace_causality
    bundle, trace, a, b = skewed_shared_fixture(tmp_path)
    records = {rank: tuple(row for row in rows if not (row.transfer_id == b and rank == 3))
               for rank, rows in trace.records_by_rank.items()}
    report = analyze_trace_causality(bundle, replace(trace, records_by_rank=records,
        global_complete=False, status='partial'))
    assert b in report['priority_operations']
    groups = [row for row in report['resource_overlap_groups'].values()
              if row.get('uncertainty') == 'missing_endpoint_or_clock']
    assert groups and all(set(row['operation_ids']) <= {a, b} for row in groups)


def test_native_postscale_frontier_waits_for_local_sender_completion(tmp_path):
    from vericcl.verification.online.trace_causality import analyze_trace_causality
    bundle, trace, a, _ = skewed_shared_fixture(tmp_path)
    storage = bundle.storage_plan
    b = next(op.operation_id for op in storage.instructions if op.kind == 'send'
             and storage.values[op.inputs[0]].rank == 2 and op.rank == 0)
    native = WorkOperation('native-postscale:0', 'copy', 'copy:0', 4096, dependencies=(a, b))
    bundle = replace(bundle, executable=replace(bundle.executable,
        operations=bundle.executable.operations + (native,), output_operations=(native.operation_id,)))
    finishes = {(a, 0): 100, (a, 1): 10, (b, 2): 90, (b, 0): 95}
    records = {rank: tuple(replace(row, raw=replace(row.raw, tb_reach=0, dependency_done=0,
                    transfer_start=0, transfer_end=finishes[row.transfer_id, rank] * 1000))
                if (row.transfer_id, rank) in finishes else row for row in rows)
               for rank, rows in trace.records_by_rank.items()}
    report = analyze_trace_causality(bundle, replace(trace, records_by_rank=records))
    assert report['completion_tails'] == [a]
    assert report['definite_completion_tails'] == [a]
    assert report['operations'][a]['output_ready_us_bounds'] == [10., 10.]
    assert report['operations'][a]['frontier_us_bounds'] == [100., 100.]
    assert report['operations'][native.operation_id]['end_us_bounds'] is None
