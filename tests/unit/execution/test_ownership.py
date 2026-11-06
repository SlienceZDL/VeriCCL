import pytest


def test_gpu_lease_survives_owner_exit_and_requires_explicit_release_receipts(tmp_path):
    from vericcl.execution.ownership import GPUOwnershipRegistry
    registry = GPUOwnershipRegistry(tmp_path)
    registry.acquire(['GPU-one'], 'run-one', {'pid': 123, 'boot_id': 'boot'})
    with pytest.raises(RuntimeError, match='retained'):
        registry.acquire(['GPU-one'], 'run-two', {'pid': 456, 'boot_id': 'boot'})
    registry.finish(['GPU-one'], 'run-one', process_clean=True, gpu_released=False)
    with pytest.raises(RuntimeError, match='retained'):
        registry.acquire(['GPU-one'], 'run-two', {'pid': 456, 'boot_id': 'boot'})
    registry.finish(['GPU-one'], 'run-one', process_clean=True, gpu_released=True)
    registry.acquire(['GPU-one'], 'run-two', {'pid': 456, 'boot_id': 'boot'})


def test_multi_device_acquisition_is_atomic_and_cannot_release_another_owner(tmp_path):
    from vericcl.execution.ownership import GPUOwnershipRegistry
    registry = GPUOwnershipRegistry(tmp_path)
    registry.acquire(['GPU-two'], 'first', {})
    with pytest.raises(RuntimeError):
        registry.acquire(['GPU-one', 'GPU-two'], 'second', {})
    registry.acquire(['GPU-one'], 'third', {})
    with pytest.raises(RuntimeError):
        registry.finish(['GPU-two'], 'third', process_clean=True, gpu_released=True)


def test_quarantine_is_shared_between_registry_instances(tmp_path):
    from vericcl.execution.ownership import GPUOwnershipRegistry
    a = GPUOwnershipRegistry(tmp_path)
    a.acquire(['GPU-one'], 'old', {})
    a.finish(['GPU-one'], 'old', process_clean=False, gpu_released=False)
    b = GPUOwnershipRegistry(tmp_path)
    assert b.snapshot()['GPU-one']['state'] == 'quarantined'
    with pytest.raises(RuntimeError):
        b.acquire(['GPU-one'], 'new', {})
