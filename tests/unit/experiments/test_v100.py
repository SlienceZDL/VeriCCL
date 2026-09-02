import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vericcl.errors import SemanticError
from vericcl.experiments.model import ExperimentCase
from vericcl.experiments.state import TaskStatus
from vericcl.experiments.v100 import (
    V100ExperimentConfig,
    load_v100_config,
    preflight,
    solve_case,
    write_hostfile,
)


def _touch(path, text="x", *, executable=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="ascii")
    if executable:
        path.chmod(0o755)
    return path


def _config(tmp_path, *, hostfile_text=None, environment=None):
    root = tmp_path / "experiment"
    repo = root / "repo"
    baseline = tmp_path / "baseline"
    baseline.mkdir(parents=True, exist_ok=True)
    manifest = _touch(repo / "exp" / "v100-k16-manifest.json", "{}")
    launcher = _touch(tmp_path / "opt" / "mpirun", executable=True)
    library = root / "msccl" / "build" / "lib"
    library.mkdir(parents=True, exist_ok=True)
    tests = root / "nccl-tests" / "build"
    tests.mkdir(parents=True, exist_ok=True)
    clock = _touch(root / "bin" / "vericcl_clock_sync", executable=True)
    hostfile_8 = root / "hostfile-2x4"
    hostfile_16 = root / "hostfile-2x8"
    write_hostfile(hostfile_8, 8)
    write_hostfile(hostfile_16, 16)
    if hostfile_text is not None:
        hostfile_8.write_text(hostfile_text, encoding="ascii")
    values = {
        "NCCL_ALGO": "MSCCL,RING",
        "NCCL_IB_DISABLE": "1",
        "NCCL_PROTO": "Simple",
        "VERICCL_CALIBRATION_LINK_CLASS": "inter_node",
        "VERICCL_CUDA_VERSION": "12.8",
        "VERICCL_FORCE_RECALIBRATE": "0",
        "VERICCL_GPU_MODEL": "NVIDIA-V100",
        "VERICCL_MSCCL_VERSION": "vericcl-runtime-v0.1.0",
        "VERICCL_NCCL_VERSION": "2.12.12",
        "VERICCL_NIC_MODEL": "ethernet",
        "VERICCL_ONLINE_INTER_NODE": "1",
    }
    if environment is not None:
        values.update(environment)
    return V100ExperimentConfig(
        experiment_root=root,
        repo_root=repo,
        manifest_path=manifest,
        baseline_source=baseline,
        remote_host="10.0.0.104",
        mpi_launcher=launcher,
        msccl_library_path=library,
        nccl_tests_binary_directory=tests,
        clock_sync_binary=clock,
        calibration_cache_path=root / "calibration" / "cache.json",
        hostfile_8=hostfile_8,
        hostfile_16=hostfile_16,
        environment=values,
    )


def _case(tmp_path):
    topology = _touch(tmp_path / "input" / "topology.json", "{}")
    sketch = _touch(tmp_path / "input" / "sketch.json", "{}")
    return ExperimentCase(
        task_id="v100-n2g4-ag-4m",
        topology_name="v100-n2g4",
        collective_label="ag",
        size_label="4m",
        topology_path=topology,
        sketch_path=sketch,
        rank_count=8,
        message_size_bytes=4 * 1024 * 1024,
        slice_size_bytes=512 * 1024,
    )


def test_preflight_requires_node4_first_and_allowed_root(tmp_path):
    config = _config(
        tmp_path,
        hostfile_text=(
            "10.0.0.102 slots=4\n10.0.0.104 slots=4\n"
        ),
    )

    with pytest.raises(SemanticError, match="node4 must be first"):
        preflight(config)


def test_preflight_rejects_ib_enabled(tmp_path):
    config = _config(tmp_path, environment={"NCCL_IB_DISABLE": "0"})

    with pytest.raises(SemanticError, match="NCCL_IB_DISABLE"):
        preflight(config)


def test_hostfiles_encode_node4_before_node2(tmp_path):
    path = tmp_path / "hostfile"

    write_hostfile(path, 16)

    assert path.read_text(encoding="ascii") == (
        "10.0.0.104 slots=8\n10.0.0.102 slots=8\n"
    )


def test_solve_case_requests_online_validation_and_tuning(tmp_path):
    config = _config(tmp_path)
    case = _case(tmp_path)
    atom = _touch(config.repo_root / "vericcl/examples/atom/default.json", "{}")
    calls = []

    def execute(context):
        calls.append(context)
        output = context.output_base / "fake"
        output.mkdir(parents=True, exist_ok=True)
        final_xml = _touch(output / "final.xml", "<algo/>")
        final_report = output / "final.validation.json"
        final_report.write_text(
            json.dumps(
                {"validation": {"online": {"status": "valid"}}}
            ),
            encoding="ascii",
        )
        return SimpleNamespace(
            final_xml=final_xml,
            final_report=final_report,
            final_candidate_id="candidate",
            status="feasible",
            layout=SimpleNamespace(summary=output / "summary.json"),
        )

    result = solve_case(
        case,
        config,
        atom_path=atom,
        execute=execute,
    )

    assert calls[0].online is True
    assert calls[0].tune is True
    assert calls[0].timeout_s == 10800.0
    assert calls[0].online_context_factory is not None
    assert result.status is TaskStatus.PASSED


def test_load_v100_config_rejects_unknown_fields(tmp_path):
    config = _config(tmp_path)
    path = tmp_path / "config.json"
    payload = {
        "schema_version": 1,
        "experiment_root": str(config.experiment_root),
        "repo_root": str(config.repo_root),
        "manifest_path": str(config.manifest_path),
        "baseline_source": str(config.baseline_source),
        "remote_host": config.remote_host,
        "mpi_launcher": str(config.mpi_launcher),
        "msccl_library_path": str(config.msccl_library_path),
        "nccl_tests_binary_directory": str(
            config.nccl_tests_binary_directory
        ),
        "clock_sync_binary": str(config.clock_sync_binary),
        "calibration_cache_path": str(config.calibration_cache_path),
        "hostfiles": {
            "8": str(config.hostfile_8),
            "16": str(config.hostfile_16),
        },
        "max_clock_uncertainty_us": 50.0,
        "environment": dict(config.environment),
        "unexpected": True,
    }
    path.write_text(json.dumps(payload), encoding="ascii")

    with pytest.raises(SemanticError, match="fields"):
        load_v100_config(path)
