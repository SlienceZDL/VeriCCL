import json
from pathlib import Path


def evidence(tmp_path, *, gpu_released=True):
    root = tmp_path
    job = root / 'ranks/0/jobs/native'; job.mkdir(parents=True)
    (root / 'native').mkdir()
    payload = {'kind': 'schedule', 'binary': '/home/example-user/native', 'job': {'rank_map': [{'rank': 0, 'host': 'worker-a', 'device': 2}]}}
    (root / 'launch.json').write_text(json.dumps(payload))
    (root / 'prepared-worker-a.json').write_text(json.dumps({'binary_sha256': 'binary-hash', 'expected_ranks': [0],
        'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}))
    (job / 'identity.json').write_text(json.dumps({'pid': 999999999, 'pgid': 999999999, 'start_ticks': 1,
        'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}))
    (job / 'request.json').write_text(json.dumps({'rank': 0, 'program': {'path': '/home/example-user/native', 'sha256': 'binary-hash'}}))
    (root / 'native/rank-0.json').write_text(json.dumps({'rank': 0, 'gpu_released': gpu_released, 'correct': False}))
    return root


def test_wrapper_loss_can_join_actual_native_release_with_exact_os_cleanup(tmp_path):
    from vericcl.verification.online.remote import _reconcile_rank_cleanup
    root = evidence(tmp_path)
    receipt = _reconcile_rank_cleanup(root, 0)
    assert receipt['cleanup_complete'] and receipt['gpu_released']
    assert receipt['status'] == 'reconciled_cleanup' and receipt['native_receipt_sha256']


def test_process_absence_alone_never_becomes_a_gpu_release(tmp_path):
    from vericcl.verification.online.remote import _reconcile_rank_cleanup
    root = evidence(tmp_path, gpu_released=False)
    receipt = _reconcile_rank_cleanup(root, 0)
    assert receipt['cleanup_complete'] and not receipt['gpu_released']


def test_mismatched_program_evidence_cannot_release_the_claimed_gpu(tmp_path):
    from vericcl.verification.online.remote import _reconcile_rank_cleanup
    root = evidence(tmp_path)
    (root / 'ranks/0/jobs/native/request.json').write_text(json.dumps({'rank': 0, 'program': {'path': '/home/example-user/unrelated', 'sha256': 'different'}}))
    assert not _reconcile_rank_cleanup(root, 0).get('gpu_released')


def test_host_restart_cannot_reuse_an_old_release_receipt(tmp_path):
    from vericcl.verification.online.remote import _reconcile_rank_cleanup
    root = evidence(tmp_path)
    path = root / 'ranks/0/jobs/native/identity.json'
    identity = json.loads(path.read_text()); identity['boot_id'] = 'another-boot'
    path.write_text(json.dumps(identity))
    assert not _reconcile_rank_cleanup(root, 0).get('gpu_released')


def test_live_process_group_prevents_cleanup_reconciliation(tmp_path, monkeypatch):
    from vericcl.verification.online import remote
    root = evidence(tmp_path)
    monkeypatch.setattr(remote, '_active_group_members', lambda identity: (identity.pid,))
    assert not remote._reconcile_rank_cleanup(root, 0).get('gpu_released')


def test_other_rank_release_cannot_release_this_rank(tmp_path):
    from vericcl.verification.online.remote import _reconcile_rank_cleanup
    root = evidence(tmp_path)
    (root / 'native/rank-0.json').write_text(json.dumps({'rank': 1, 'gpu_released': True}))
    receipt = _reconcile_rank_cleanup(root, 0)
    assert receipt['cleanup_complete'] and not receipt['gpu_released']


def test_collection_deduplicates_original_and_reconciled_receipts():
    import base64
    from vericcl.verification.online.remote import collected_cleanup_receipts
    def encoded(**value):
        return base64.b64encode(json.dumps(value).encode()).decode()
    collection = {'files': {
        'ranks/0/rank-result.json': encoded(rank=0, cleanup_complete=True, gpu_released=False),
        'ranks/0/cleanup-reconciliation.json': encoded(rank=0, cleanup_complete=True, gpu_released=True),
        'ranks/1/rank-result.json': encoded(rank=1, cleanup_complete=True, gpu_released=True),
    }}
    receipts = collected_cleanup_receipts([collection])
    assert set(receipts) == {0, 1} and all(item['gpu_released'] for item in receipts.values())
