from datetime import datetime, timezone, timedelta
import json

import pytest


def envelope(**changes):
    return dict(schema_version=1, event_id='e1', group_id='group', source='external_health',
        observed_at=datetime.now(timezone.utc).isoformat(), kind='link_failed', links=[[0, 1]], metrics={}) | changes


def test_append_feed_waits_for_whole_record_and_deduplicates_identity(tmp_path):
    from vericcl.live.external import ExternalTelemetryFeed
    path = tmp_path / 'telemetry.jsonl'; raw = json.dumps(envelope()).encode() + b'\n'
    feed = ExternalTelemetryFeed(path, group_id='group', legal_links={(0, 1)}, max_age_s=60)
    path.write_bytes(raw[:20])
    assert feed.poll(version='A', request_id=0) == ()
    with path.open('ab') as stream: stream.write(raw[20:])
    result = feed.poll(version='A', request_id=0)
    assert len(result) == 1 and result[0].invalid_links == ((0, 1),)
    with path.open('ab') as stream: stream.write(raw)
    assert feed.poll(version='A', request_id=0) == ()
    with path.open('ab') as stream: stream.write(json.dumps(envelope(source='fault_injection')).encode() + b'\n')
    with pytest.raises(ValueError, match='identity'):
        feed.poll(version='A', request_id=0)


@pytest.mark.parametrize('seconds', [-120, 120])
def test_feed_rejects_expired_or_future_observations(tmp_path, seconds):
    from vericcl.live.external import ExternalTelemetryFeed
    path = tmp_path / 'telemetry.jsonl'
    stamp = (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()
    path.write_text(json.dumps(envelope(observed_at=stamp)) + '\n')
    feed = ExternalTelemetryFeed(path, group_id='group', legal_links={(0, 1)}, max_age_s=60)
    with pytest.raises(ValueError, match='timestamp'):
        feed.poll(version='A', request_id=0)


def test_feed_rejects_truncation_and_never_accepts_partial_oversized_record(tmp_path):
    from vericcl.live.external import ExternalTelemetryFeed
    path = tmp_path / 'telemetry.jsonl'; path.write_bytes(b'x' * (65536 + 1))
    feed = ExternalTelemetryFeed(path, group_id='group', legal_links={(0, 1)}, max_age_s=60)
    with pytest.raises(ValueError, match='64 KiB'):
        feed.poll(version='A', request_id=0)


@pytest.mark.parametrize('changes', [{'observed_at': None}, {'links': None}, {'links': [1]}, {'schema_version': True}])
def test_external_malformed_types_have_clear_validation_error(changes):
    from vericcl.live.telemetry import parse_external
    with pytest.raises(ValueError):
        parse_external(envelope(**changes), group_id='group', legal_links={(0, 1)}, current_version='A', request_id=0)
