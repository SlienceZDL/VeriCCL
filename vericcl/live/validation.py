"""Trace and paired sampling at idle boundaries of the same persistent ranks."""
import statistics

from vericcl.verification.online.acceptance import AcceptanceContext, LaunchSample, compare_measured
from vericcl.verification.online.clock_sync import parse_clock_sync_output
from vericcl.verification.online.statistical_ledger import StatisticalLedger
from vericcl.verification.online.trace_once import RawTraceCapture, TraceContext, capture_once
from vericcl.verification.online.trace_format import RAW_HEADER_STRUCT, RAW_RECORD_STRUCT
from vericcl.workflow_state import plain
from vericcl.xml.trace_sidecar import build_v5_trace_sidecar


def check_sample_group(result, inputs, bundle, *, warmup, timed, trace):
    records = result['ranks']
    if len(records) != inputs.rank_count or {item.get('rank') for item in records} != set(range(inputs.rank_count)):
        raise ValueError('live sample group is incomplete')
    for record in records:
        expected = dict(version_id=bundle.xml_sha256, correct=True, gpu_completed=True,
            gpu_work_quiescent=True, resources_retained=True, qualified=True, light_telemetry_enabled=False,
            trace_enabled=trace, warmup_iterations=warmup, timed_iterations=timed)
        if any(record.get(key) != value for key, value in expected.items()):
            raise ValueError('live sample lacks qualified all-rank quiescent evidence')
        if len(record['device_times_us']) != timed or record['request_ids'] != records[0]['request_ids'] or len(record['request_ids']) != timed:
            raise ValueError('live sample invocation identities differ')
    return records


class LiveBoundaryValidator:
    def __init__(self, session, *, current_inputs=None):
        self.session = session
        self.current_inputs = current_inputs or (lambda: session.inputs)

    def trace_driver(self, bundle, inputs):
        session = self.session
        class Driver:
            def capture_once(self, plan):
                result, files, _, root = session.sample(bundle, inputs, warmup=plan.warmup_iterations,
                    timed=1, capture_trace=True, lease=plan.lease)
                records = check_sample_group(result, inputs, bundle, warmup=plan.warmup_iterations, timed=1, trace=True)
                paths, clocks = {}, []
                for rank in inputs.request_config['runtime']['rank_map']:
                    index, host = rank['rank'], rank['host']
                    paths[index] = files[host, 'trace.rank-%d.bin' % index]
                    clocks.extend(parse_clock_sync_output(files[host, 'clock.rank-%d.log' % index].read_text()))
                return RawTraceCapture(paths, tuple(clocks), {r['rank']: r for r in records}, 'native_msccl')
        return Driver()

    def measure(self, bundle, inputs, launch):
        result, _, identity, root = self.session.sample(bundle, inputs, warmup=launch.warmup,
            timed=launch.timed, capture_trace=False, lease=launch.lease)
        records = check_sample_group(result, inputs, bundle, warmup=launch.warmup, timed=launch.timed, trace=False)
        values = [max(record['device_times_us'][index] for record in records) for index in range(launch.timed)]
        return LaunchSample(statistics.fmean(values), True, False, False, launch.launch_id, 'native_msccl_live', identity,
                           (str(root / 'result.json'), str(root / 'inventory.json')), True, True)

    def __call__(self, old, new, inputs, lease):
        settings = inputs.request_config['sketch']['trace_and_measure']; context = self.session.context
        old_inputs = self.current_inputs()
        capacity = {rank: (settings['trace_capacity_bytes'] - RAW_HEADER_STRUCT.size) // RAW_RECORD_STRUCT.size for rank in range(inputs.rank_count)}
        trace = capture_once(build_v5_trace_sidecar(new), TraceContext(True, tuple(range(inputs.rank_count)),
            self.trace_driver(new, inputs), capacity, lease, context, warmup_iterations=settings['warmup_iterations']))
        evidence = dict(measured_required=True, trace_complete=trace.global_complete,
            trace_evidence_kind=trace.evidence_kind, trace_status=trace.status, trace_issues=list(trace.issues))
        context.write_json('reports/live-trace-' + (trace.capture_id or 'unavailable') + '.json', plain(trace))
        try:
            if old is not None and trace.global_complete:
                ledger = StatisticalLedger(context.artifact('statistics/attempts.jsonl'), task_id=context.task_id,
                    task_alpha=settings['task_alpha'], single_attempt_cap=settings['single_attempt_alpha_cap'], run_context=context)
                acceptance = AcceptanceContext(True,
                    driver=lambda identity, launch: self.measure(new if identity == new.xml_sha256 else old,
                        inputs if identity == new.xml_sha256 else old_inputs, launch),
                    seed=settings['seed'], minimum_gain=settings['min_relative_gain'], pair_count=settings['pairs'], lease=lease,
                    warmup=settings['warmup_iterations'], timed=settings['timed_iterations'], max_rounds=settings['max_rounds'], max_cv=settings['max_cv'])
                measured = compare_measured(old.xml_sha256, new.xml_sha256, ledger, acceptance)
                evidence.update(measured_accepted=measured.accepted, measured=plain(measured))
            return evidence
        finally:
            if old is not None:
                # Validation calls use generated data and their own request ids.
                # Restore the application version until the controller accepts.
                self.session.prepare_version(old, old_inputs, lease=lease)
