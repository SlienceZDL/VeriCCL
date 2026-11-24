"""Observed software summaries and explicit external telemetry envelopes."""
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import math
import statistics

from vericcl.live.triggers import TriggerEvent


@dataclass(frozen=True)
class RequestSummary:
    group_id: str
    request_id: int
    attempt: int
    version: str
    qualified: bool
    metrics: dict
    observations: tuple
    observed_at: str


def _positive(value, name, *, zero=False):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError('invalid telemetry ' + name)
    return value


def summarize_request(group, receipt, *, ranks):
    records = receipt.get('ranks', ())
    if receipt.get('logical_success') is not True or len(records) != ranks or {r.get('rank') for r in records} != set(range(ranks)):
        raise ValueError('telemetry requires one successful complete group receipt')
    request, attempt, version = receipt['request_id'], receipt['attempt'], receipt['version_id']
    if type(request) is not int or request < 1 or type(attempt) is not int or attempt < 0 or not version:
        raise ValueError('invalid request telemetry identity')
    operations, observations, times = defaultdict(list), [], []
    qualified = attempt == 0
    for record in records:
        if (record.get('request_id'), record.get('attempt'), record.get('version_id')) != (request, attempt, version):
            raise ValueError('telemetry rank identity differs')
        if record.get('gpu_completed') is not True or record.get('correct') is not True:
            raise ValueError('telemetry lacks numerical/GPU completion')
        times.append(_positive(record['device_time_us'], 'device time'))
        if not record.get('telemetry'):
            qualified = False
        for step in record.get('telemetry', ()):
            if step.get('source') != 'software_device_counters' or step.get('samples') != 1 or step.get('invalid_samples') != 0:
                qualified = False; continue
            service = _positive(step['service_ns'], 'service time') / 1000
            wait = _positive(step['dependency_wait_ns'], 'dependency wait', zero=True) / 1000
            payload = _positive(step['payload_bytes'], 'payload')
            local = dict(step, rank=record['rank'], service_us=service, dependency_wait_us=wait)
            observations.append(local); operations[step['operation_id']].append(local)
    services = defaultdict(list)
    for steps in operations.values():
        send = next((s for s in steps if s['kind'] == 's'), None)
        recv = next((s for s in steps if s['kind'] in {'r', 'rrc'}), None)
        if send and recv and len(steps) == 2 and send['send_peer'] == recv['rank'] and recv['recv_peer'] == send['rank']:
            key = ('rrc:' if recv['kind'] == 'rrc' else 'link:') + '%d:%d' % (send['rank'], recv['rank'])
        elif len(steps) == 1 and steps[0]['kind'] in {'cpy', 're'}:
            key = ('copy:' if steps[0]['kind'] == 'cpy' else 'reduce:') + str(steps[0]['rank'])
        else:
            qualified = False; continue
        services['service:' + key].append(max(s['service_us'] for s in steps))
    metrics = {'request': max(times)} | {key: statistics.median(values) for key, values in services.items()}
    return RequestSummary(group, request, attempt, version, qualified, metrics, tuple(observations), datetime.now(timezone.utc).isoformat())


def parse_external(value, *, group_id, legal_links, current_version, request_id):
    fields = {'schema_version', 'event_id', 'group_id', 'source', 'observed_at', 'kind', 'links', 'metrics'}
    if not isinstance(value, dict) or set(value) != fields or type(value['schema_version']) is not int or value['schema_version'] != 1:
        raise ValueError('external telemetry envelope fields or schema differ')
    if value['group_id'] != group_id or not isinstance(value['event_id'], str) or not value['event_id']:
        raise ValueError('external telemetry identity differs')
    if value['source'] not in {'external_inband', 'external_health', 'fault_injection'}:
        raise ValueError('external telemetry must name its source')
    if not isinstance(value['observed_at'], str):
        raise ValueError('external telemetry timestamp must be an ISO string')
    stamp = datetime.fromisoformat(value['observed_at'].replace('Z', '+00:00'))
    if stamp.tzinfo is None:
        raise ValueError('external telemetry timestamp needs a timezone')
    if not isinstance(value['links'], list) or any(not isinstance(edge, list) for edge in value['links']):
        raise ValueError('external telemetry links must be an array of directed pairs')
    links = tuple(tuple(edge) for edge in value['links'])
    if any(len(edge) != 2 or any(type(rank) is not int for rank in edge) or edge not in legal_links for edge in links):
        raise ValueError('external telemetry names an unknown directed link')
    if len(set(links)) != len(links) or not isinstance(value['metrics'], dict):
        raise ValueError('invalid external telemetry scope or metrics')
    if value['kind'] == 'link_failed':
        if not links or value['metrics']:
            raise ValueError('link_failed needs explicit failed links and no inferred costs')
        return TriggerEvent('fault', request_id, current_version, invalid_links=links, evidence=dict(value))
    if value['kind'] != 'cost_update' or not value['metrics']:
        raise ValueError('unsupported external telemetry kind')
    for key, point in value['metrics'].items():
        if not isinstance(point, dict) or set(point) != {'alpha_us', 'beta_us_per_byte', 'min_bytes', 'max_bytes'}:
            raise ValueError('cost_update requires explicit alpha/beta units and payload scope')
        _positive(point['alpha_us'], 'alpha', zero=True); _positive(point['beta_us_per_byte'], 'beta')
        if type(point['min_bytes']) is not int or type(point['max_bytes']) is not int or not 1 <= point['min_bytes'] <= point['max_bytes']:
            raise ValueError('invalid external cost payload scope')
        parts = key.split(':')
        if len(parts) != 3 or parts[0] not in {'link', 'rrc'} or not all(p.isdigit() for p in parts[1:]) or tuple(map(int, parts[1:])) not in links:
            raise ValueError('external cost key is outside the stated directed-link scope')
    return TriggerEvent('model_update', request_id, current_version, evidence=dict(value))
