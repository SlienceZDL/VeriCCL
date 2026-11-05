"""Qualify independent native launches and fit exact mode/resource components."""
from dataclasses import asdict, dataclass, replace
from datetime import datetime
import math
import statistics

from vericcl.performance.profiles import RateProfile
from vericcl.verification.search_coverage import digest
from .cost_fitting import fit_curve
from .model_publication import ComponentUpdates, profile_key


@dataclass(frozen=True)
class FitPolicy:
    max_relative_fit_error: float
    minimum_payloads: int = 3
    lease: object = None

    def __post_init__(self):
        if isinstance(self.max_relative_fit_error, bool) or not isinstance(self.max_relative_fit_error, (float, int)) or not math.isfinite(self.max_relative_fit_error) or self.max_relative_fit_error < 0:
            raise ValueError('fit policy needs a finite nonnegative residual limit')
        if isinstance(self.minimum_payloads, bool) or not isinstance(self.minimum_payloads, int) or self.minimum_payloads < 3:
            raise ValueError('fit policy needs at least three payloads')


def _qualification(job, result):
    if result.job_id != job.job_id or result.driver_kind != 'native_msccl' or result.status != 'completed':
        raise ValueError('only completed native measurements can update physical costs')
    if not result.cleanup_confirmed or not result.gpu_released:
        raise ValueError('native measurement has unresolved execution ownership')
    provenance = result.provenance
    for field in ('measurement_id', 'measured_at', 'runtime_identity', 'selected_round'):
        if not provenance.get(field):
            raise ValueError('native measurement lacks original provenance: ' + field)
    stamp = datetime.fromisoformat(provenance['measured_at'].replace('Z', '+00:00'))
    if stamp.tzinfo is None:
        raise ValueError('native measurement timestamp is not timezone-aware')
    chosen = provenance['selected_round']
    if isinstance(chosen, bool) or not isinstance(chosen, int) or not 1 <= chosen <= job.max_rounds:
        raise ValueError('invalid selected measurement round')
    wanted = {flow.flow_id for flow in job.flows}
    summary = None
    for number in range(1, chosen + 1):
        samples = [sample for sample in result.samples if sample.get('round') == number]
        if len(samples) != job.launches or {sample.get('launch') for sample in samples} != set(range(1, job.launches + 1)):
            raise ValueError('measurement round lacks distinct configured process launches')
        for sample in samples:
            if sample.get('correct') is not True or sample.get('gpu_released') is not True or set(sample.get('flow_us', {})) != wanted:
                raise ValueError('measurement sample is incomplete or failed correctness')
            for value in sample['flow_us'].values():
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                    raise ValueError('measurement service time must be finite and positive')
        summary = {}
        for flow in wanted:
            values = [sample['flow_us'][flow] for sample in samples]
            ordered = sorted(values)
            summary[flow] = {'median_us': statistics.median(values), 'mean_us': statistics.mean(values),
                              'p95_us': ordered[math.ceil(.95 * len(ordered)) - 1],
                              'cv': statistics.pstdev(values) / statistics.mean(values), 'launch_count': len(values)}
        stable = all(values['cv'] <= job.max_cv for values in summary.values())
        if stable != (number == chosen):
            raise ValueError('selected round is not the first independently stable round')
    if any(sample.get('round', 0) > chosen for sample in result.samples):
        raise ValueError('measurement continued after its first stable round')
    return summary


def _identity(job, result):
    return digest({'mode': [f.flow_id for f in job.flows], 'resources': job.resources,
                   'flow_resources': job.flow_resources, 'rank_map': job.rank_map,
                   'topology_facts': job.topology_facts,
                   'datatype': job.datatype, 'reduction_op': job.reduction_op,
                   'runtime': result.provenance['runtime_identity']})


def _service_key(flow):
    return ('link:%d:%d' % (flow.src_rank, flow.dst_rank) if flow.kind == 'send' else
            'rrc:%d:%d' % (flow.src_rank, flow.dst_rank) if flow.kind == 'rrc' else
            '%s:%d' % (flow.kind, flow.src_rank))


def fit_components(observations, policy, *, confirmations=()):
    if isinstance(policy, dict):
        policy = FitPolicy(policy['max_relative_fit_error'])
    if not isinstance(policy, FitPolicy):
        raise ValueError('component fitting requires FitPolicy')
    grouped, confirmed, issues, records, measurement_ids = {}, {}, [], [], set()
    def collect(values, target, confirmation):
        for job, result in values:
            if policy.lease is not None:
                policy.lease.check()
            try:
                summary = _qualification(job, result)
                identity = _identity(job, result)
                measurement = result.provenance['measurement_id']
                if measurement in measurement_ids:
                    raise ValueError('confirmation or payload reused an existing independent measurement identity')
                measurement_ids.add(measurement)
                group = target.setdefault(identity, {})
                if job.payload_bytes in group:
                    raise ValueError('duplicate primary payload; choose its independent result explicitly')
                group[job.payload_bytes] = (job, result, summary)
                records.append({'job_id': job.job_id, 'measurement_id': measurement, 'identity': identity,
                                'payload_bytes': job.payload_bytes, 'confirmation': confirmation, 'summary': summary})
            except (ValueError, KeyError, TypeError) as error:
                issues.append({'job_id': job.job_id, 'reason': str(error), 'status': result.status,
                               'driver_kind': result.driver_kind})
    collect(observations, grouped, False)
    collect(confirmations, confirmed, True)
    component_identities = {}
    for identity, sizes in grouped.items():
        if policy.lease is not None:
            policy.lease.check()
        job = next(iter(sizes.values()))[0]
        keys = []
        if len(job.flows) == 1 and job.flows[0].channel == 0:
            keys.append('service:' + _service_key(job.flows[0]))
        if all(flow.kind in {'send', 'rrc'} for flow in job.flows):
            keys.append('profile:' + digest({'mode': sorted(f.flow_id for f in job.flows), 'resources': sorted(job.resources)}))
        for key in keys:
            component_identities.setdefault(key, set()).add(identity)
    ambiguous = {key for key, identities in component_identities.items() if len(identities) > 1}
    issues.extend({'component': key, 'reason': 'multiple_runtime_identities_require_explicit_selection'} for key in sorted(ambiguous))
    services, profiles, signatures = {}, [], {}
    for identity, sizes in grouped.items():
        ordered = sorted(sizes)
        job = sizes[ordered[0]][0]
        fits = {}
        for flow in job.flows:
            points = [(size, sizes[size][2][flow.flow_id]['median_us']) for size in ordered]
            confirmation = [(size, record[2][flow.flow_id]['median_us']) for size, record in sorted(confirmed.get(identity, {}).items())]
            fit = fit_curve(points, policy.max_relative_fit_error, confirmation=confirmation, minimum_sizes=policy.minimum_payloads, lease=policy.lease)
            records.append({'identity': identity, 'flow_id': flow.flow_id, 'fit_status': fit.status,
                            'diagnostics': fit.diagnostics, 'proposed_segments': [asdict(v) for v in fit.proposed_segments]})
            if fit.status != 'qualified':
                followup = {'confirmation_payloads': ordered, 'requires_new_measurement_ids': True} if fit.status == 'needs_confirmation' else {}
                if fit.status in {'unidentifiable', 'poor_fit', 'insufficient_sizes'}:
                    from vericcl.input.schema import DATATYPE_BYTES
                    from vericcl.xml.capacity import simple_payload_capacity
                    environment = sizes[ordered[0]][1].provenance['runtime_identity'].get('environment', {})
                    if 'NCCL_BUFFSIZE' in environment:
                        maximum = simple_payload_capacity(int(environment['NCCL_BUFFSIZE']), DATATYPE_BYTES[job.datatype])
                        followup['additional_payloads'] = sorted({min(maximum, ordered[-1] * scale) for scale in (2, 4, 8)} - set(ordered))
                issues.append({'identity': identity, 'flow_id': flow.flow_id, 'reason': fit.status, **followup})
                continue
            values = []
            for piece in fit.segments:
                timestamps = [sizes[size][1].provenance['measured_at'] for size in ordered if piece.covers(size)]
                # Confirmation never refreshes an older primary measurement.
                values.append(replace(piece, source='measured', measured_at=min(timestamps)))
            fits[flow.flow_id] = tuple(values)
        if len(job.flows) == 1 and job.flows[0].channel == 0 and job.flows[0].flow_id in fits:
            key = _service_key(job.flows[0])
            if 'service:' + key not in ambiguous:
                services[key] = fits[job.flows[0].flow_id]
                signatures['service:' + key] = identity
        if any(flow.kind not in {'send', 'rrc'} for flow in job.flows) or len(fits) != len(job.flows):
            continue
        membership = dict(job.flow_resources)
        if set(membership) != set(fits):
            issues.append({'identity': identity, 'reason': 'missing_per_flow_resource_membership'})
            continue
        boundaries = sorted({x for values in fits.values() for piece in values for x in (piece.min_bytes, piece.max_bytes + 1)})
        for begin, after in zip(boundaries, boundaries[1:]):
            pieces = {flow: next((value for value in values if value.covers(begin)), None) for flow, values in fits.items()}
            if any(value is None for value in pieces.values()):
                continue
            profile = RateProfile('fit-' + digest({'identity': identity, 'begin': begin, 'end': after - 1})[:24],
                tuple(fits), {flow: 1 / value.beta_us_per_byte for flow, value in pieces.items()}, job.resources,
                begin, after - 1, 'measured', min(value.measured_at for value in pieces.values()),
                {flow: value.alpha_us for flow, value in pieces.items()}, membership)
            key = profile_key(profile)
            if key in ambiguous:
                break
            profiles.append(profile); signatures[key] = identity
    return ComponentUpdates(services, tuple(profiles), signatures, tuple(issues), tuple(records))
