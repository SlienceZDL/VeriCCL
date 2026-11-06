"""Large completed sessions must not hide durable execution/release receipts."""
import base64
import json
from pathlib import Path
from types import SimpleNamespace
import time

import pytest

from vericcl.execution.context import RunContext
from vericcl.verification.online import remote


def fixture(root, monkeypatch):
    monkeypatch.setattr(remote, 'owned_home_path', lambda value: Path(value).resolve())
    monkeypatch.setattr(remote, '_COLLECTION_PAGE_MIN_BYTES', 4096, raising=False)
    root.mkdir()
    (root / 'prepared-worker-a.json').write_text(json.dumps({'expected_ranks': [0]}))
    (root / 'launch.json').write_text(json.dumps({'kind': 'schedule'}))
    (root / 'native/rank-0').mkdir(parents=True)
    (root / 'ranks/0').mkdir(parents=True)
    (root / 'native/rank-0.json').write_text(json.dumps({'rank': 0, 'gpu_released': True}))
    (root / 'ranks/0/rank-result.json').write_text(json.dumps(
        {'rank': 0, 'cleanup_complete': True, 'gpu_released': True}))
    for index in range(20):
        (root / ('native/rank-0/request-%03d.json' % index)).write_text(json.dumps({'payload': 'x' * 800}))
    return root


def test_receipt_page_precedes_bounded_complete_artifact_pages(tmp_path, monkeypatch):
    root = fixture(tmp_path / 'launch', monkeypatch)
    first = remote._collect(root, 'worker-a', cursor=0)
    assert set(first['files']) == {'native/rank-0.json', 'ranks/0/rank-result.json'}
    assert first['next_cursor'] == 1
    assert first['snapshot']['scope'] == 'receipts'
    files, cursor, digest = dict(first['files']), first['next_cursor'], None
    while cursor is not None:
        page = remote._collect(root, 'worker-a', cursor=cursor, inventory_digest=digest)
        digest = page['snapshot']['digest']
        assert page['snapshot']['total_bytes'] > 4096
        assert sum(len(base64.b64decode(value)) for value in page['files'].values()) <= 4096
        assert not set(files) & set(page['files'])
        files.update(page['files'])
        assert page['next_cursor'] is None or page['next_cursor'] > cursor
        cursor = page['next_cursor']
    assert len(files) == 22
    assert all(base64.b64decode(data) == (root / name).read_bytes() for name, data in files.items())


def test_oversized_artifact_fails_after_receipts_are_available(tmp_path, monkeypatch):
    root = fixture(tmp_path / 'launch', monkeypatch)
    (root / 'native/rank-0/request-000.json').write_text('x' * 5000)
    first = remote._collect(root, 'worker-a', cursor=0)
    assert remote.collected_cleanup_receipts([first])[0]['gpu_released'] is True
    with pytest.raises(ValueError, match='bound|limit'):
        remote._collect(root, 'worker-a', cursor=first['next_cursor'])


def test_paging_rejects_changed_snapshot(tmp_path, monkeypatch):
    root = fixture(tmp_path / 'launch', monkeypatch)
    remote._collect(root, 'worker-a', cursor=0)
    first = remote._collect(root, 'worker-a', cursor=1)
    (root / 'native/rank-0/request-010.json').write_text('changed')
    with pytest.raises(ValueError, match='snapshot|inventory'):
        remote._collect(root, 'worker-a', cursor=first['next_cursor'], inventory_digest=first['snapshot']['digest'])


def test_all_hosts_receipts_are_collected_before_any_bulk_page(tmp_path, monkeypatch):
    run = RunContext.create(tmp_path, 'pages', 'ordered')
    driver = remote.NativeProbeDriver(project_root=tmp_path)
    calls = []
    def helper(context, host, args, *rest):
        bulk = '--cursor' in args and args[args.index('--cursor') + 1] != '0'
        calls.append((host, bulk))
        if bulk:
            raise ValueError('injected archive failure')
        return dict(host=host, files={}, prepared={}, next_cursor=1,
                    snapshot={'digest': 'snapshot-' + host})
    monkeypatch.setattr(driver, '_helper', helper)
    collected, errors = driver._collect_evidence(SimpleNamespace(run_context=run), ['worker-b', 'worker-a'],
        tmp_path / 'job', time.monotonic() + 30, 'collect', 'reports/collected')
    assert calls[:2] == [('worker-b', False), ('worker-a', False)]
    assert len(collected) == 2
    assert len(errors) == 2


def test_legacy_coordinator_retains_complete_bounded_response(tmp_path, monkeypatch):
    root = fixture(tmp_path / 'launch', monkeypatch)
    monkeypatch.setattr(remote, '_COLLECTION_PAGE_MIN_BYTES', 32 * 1024)
    collection = remote._collect(root, 'worker-a')
    assert collection['next_cursor'] is None and len(collection['files']) == 22
    monkeypatch.setattr(remote, '_COLLECTION_PAGE_MIN_BYTES', 4096)
    with pytest.raises(ValueError, match='bound'):
        remote._collect(root, 'worker-a')


def test_expired_collection_deadline_starts_no_new_helper(tmp_path, monkeypatch):
    run = RunContext.create(tmp_path, 'pages', 'expired')
    driver = remote.NativeProbeDriver(project_root=tmp_path)
    calls = []
    def helper(*args, **kwargs):
        calls.append(args)
        raise RuntimeError('helper must not start after the parent deadline')
    monkeypatch.setattr(driver, '_helper', helper)
    collected, errors = driver._collect_evidence(SimpleNamespace(run_context=run), ['worker-b', 'worker-a'],
        tmp_path / 'job', time.monotonic() - 1, 'collect', 'reports/collected')
    assert calls == []
    assert collected == [] and len(errors) == 2


def test_persisted_bulk_data_is_not_retained_in_receipt_memory(tmp_path, monkeypatch):
    run = RunContext.create(tmp_path, 'pages', 'bulk-memory')
    driver = remote.NativeProbeDriver(project_root=tmp_path)
    data = b'large telemetry record'
    def helper(context, host, args, *rest):
        bulk = args[args.index('--cursor') + 1] != '0'
        return dict(host=host, prepared={}, page_cursor=1 if bulk else 0,
            files={'native/rank-0/request-1.json': base64.b64encode(data).decode()} if bulk else {},
            next_cursor=None if bulk else 1, snapshot={'digest': 'fixed-snapshot'})
    monkeypatch.setattr(driver, '_helper', helper)
    pages, errors = driver._collect_evidence(SimpleNamespace(run_context=run), ['worker-a'],
        tmp_path / 'job', time.monotonic() + 30, 'collect', 'reports/collected')
    assert errors == [] and len(pages) == 2
    assert pages[1]['files'] == {}
    assert pages[1]['archived_files']['native/rank-0/request-1.json']['bytes'] == len(data)
    assert run.artifact('reports/collected/worker-a/native/rank-0/request-1.json').read_bytes() == data


def test_registry_uses_the_receipt_bytes_actually_returned(tmp_path, monkeypatch):
    from vericcl.execution.ownership import GPUOwnershipRegistry
    root = fixture(tmp_path / 'launch', monkeypatch)
    registry = GPUOwnershipRegistry(tmp_path / 'ownership')
    registry.acquire(['GPU-test'], str(root), {})
    (root / 'prepared-worker-a.json').write_text(json.dumps({'expected_ranks': [0],
        'ownership_registry': str(registry.root), 'device_uuids': ['GPU-test']}))
    # The early inspection saw a pending rank, but a valid final wrapper
    # receipt is present by the time the returned snapshot is read.
    monkeypatch.setattr(remote, '_rank_cleanup_receipt', lambda *args:
        {'rank': 0, 'cleanup_complete': False, 'gpu_released': False})
    first = remote._collect(root, 'worker-a', cursor=0)
    assert remote.collected_cleanup_receipts([first])[0]['gpu_released'] is True
    assert registry.snapshot()['GPU-test']['state'] == 'released'


def test_bulk_change_cannot_discard_unchanged_receipt_page(tmp_path, monkeypatch):
    root = fixture(tmp_path / 'launch', monkeypatch)
    original = Path.read_bytes
    def changing_read(path):
        data = original(path)
        if path == root / 'native/rank-0.json':
            (root / 'native/rank-0/request-000.json').write_text('changed while receipts were collected')
        return data
    monkeypatch.setattr(Path, 'read_bytes', changing_read)
    first = remote._collect(root, 'worker-a', cursor=0)
    assert remote.collected_cleanup_receipts([first])[0]['gpu_released'] is True


def test_bulk_inventory_error_preserves_receipts(tmp_path, monkeypatch):
    root = fixture(tmp_path / 'launch', monkeypatch)
    original = Path.stat
    def denied_stat(path, *args, **kwargs):
        if path.name == 'request-000.json':
            raise PermissionError('injected telemetry access failure')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'stat', denied_stat)
    first = remote._collect(root, 'worker-a', cursor=0)
    assert remote.collected_cleanup_receipts([first])[0]['gpu_released'] is True
    with pytest.raises(PermissionError, match='injected telemetry access failure'):
        remote._collect(root, 'worker-a', cursor=1)


def test_receipt_page_never_enters_bulk_inventory(tmp_path, monkeypatch):
    root = fixture(tmp_path / 'launch', monkeypatch)
    original = remote._collection_inventory
    calls = []
    def potentially_blocking_inventory(*args, **kwargs):
        calls.append(kwargs.get('receipts_only', False))
        if not kwargs.get('receipts_only', False):
            raise RuntimeError('bulk traversal could block beyond the receipt deadline')
        return original(*args, **kwargs)
    monkeypatch.setattr(remote, '_collection_inventory', potentially_blocking_inventory)
    first = remote._collect(root, 'worker-a', cursor=0)
    assert first['next_cursor'] == 1 and all(calls)
    assert remote.collected_cleanup_receipts([first])[0]['gpu_released'] is True
