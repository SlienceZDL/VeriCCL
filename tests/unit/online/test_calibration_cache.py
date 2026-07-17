from dataclasses import replace

import pytest

from vericcl.errors import SemanticError
from vericcl.verification.online.cache import (
    CalibrationCache,
    EnvironmentSignature,
    environment_signature_sha256,
)
from vericcl.verification.online.calibration import CalibrationPoint
from vericcl.verification.online.statistics import summarize_runs


pytestmark = pytest.mark.phase06


def _signature(**changes):
    values = {
        "link_class": "intra_node",
        "topology_signature": "a" * 64,
        "gpu_model": "NVIDIA V100",
        "nic_model": "none",
        "cuda_version": "11.8",
        "nccl_version": "2.18.5",
        "msccl_version": "0.7.4",
        "protocol": "Simple",
        "slice_size_bytes": 1024 * 1024,
        "benchmark_size_bytes": 128 * 1024 * 1024,
        "concurrency": 4,
        "nccl_buffsize_bytes": 2 * 1024 * 1024,
        "chunk_steps": 4,
        "slice_steps": 4,
        "path_variables": (
            ("LD_LIBRARY_PATH", "/opt/msccl/lib"),
            ("MSCCL_XML_FILES", "/tmp/calibration.xml"),
        ),
    }
    values.update(changes)
    return EnvironmentSignature(**values)


def _point(concurrency=4):
    return CalibrationPoint(
        concurrency,
        summarize_runs((10.0,) * 20),
        full_wave_count=8,
        tail_transfer_count=0,
    )


def test_cache_requires_exact_environment_signature_and_force_bypasses():
    signature = _signature()
    point = _point()
    cache = CalibrationCache()
    cache.put(signature, point)

    assert cache.get(signature) == point
    assert cache.get(signature, force_recalibrate=True) is None
    assert len(environment_signature_sha256(signature)) == 64

    mismatches = (
        {"link_class": "inter_node"},
        {"topology_signature": "b" * 64},
        {"gpu_model": "NVIDIA A100"},
        {"nic_model": "ConnectX-6"},
        {"cuda_version": "12.1"},
        {"nccl_version": "2.19.0"},
        {"msccl_version": "0.8.0"},
        {"protocol": "LL"},
        {"slice_size_bytes": 2 * 1024 * 1024},
        {"benchmark_size_bytes": 64 * 1024 * 1024},
        {"concurrency": 3},
        {"nccl_buffsize_bytes": 4 * 1024 * 1024},
        {"chunk_steps": 2},
        {"slice_steps": 2},
        {"path_variables": (("LD_LIBRARY_PATH", "/other"),)},
    )
    for changes in mismatches:
        assert cache.get(replace(signature, **changes)) is None


def test_signature_normalizes_path_order_and_cache_validates_values():
    first = _signature()
    second = _signature(path_variables=tuple(reversed(first.path_variables)))
    assert first == second
    assert environment_signature_sha256(first) == (
        environment_signature_sha256(second)
    )

    cache = CalibrationCache()
    with pytest.raises(SemanticError, match="signature"):
        cache.put(object(), _point())
    with pytest.raises(SemanticError, match="point"):
        cache.put(first, object())
    with pytest.raises(SemanticError, match="concurrency"):
        cache.put(first, _point(concurrency=3))
    with pytest.raises(SemanticError, match="boolean"):
        cache.get(first, force_recalibrate="yes")


def test_environment_signature_rejects_invalid_fields():
    value = _signature()
    for changes in (
        {"topology_signature": "short"},
        {"slice_size_bytes": 0},
        {"concurrency": 0},
        {"path_variables": (("A", "1"), ("A", "2"))},
        {"path_variables": (("", "1"),)},
    ):
        with pytest.raises(SemanticError):
            replace(value, **changes)
