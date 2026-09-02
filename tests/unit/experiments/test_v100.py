import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import vericcl.experiments.v100 as v100_module
from vericcl.errors import SemanticError
from vericcl.experiments.model import ExperimentCase
from vericcl.experiments.state import TaskStatus
from vericcl.verification.online.runner import ProcessResult
from vericcl.experiments.v100 import (
    V100ExperimentConfig,
    benchmark_xml,
    build_mpi_prefix,
    build_performance_environment,
    load_v100_config,
    preflight,
    smoke,
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


def test_smoke_preserves_runner_error_details(tmp_path, monkeypatch):
    config = _config(tmp_path)
    case = _case(tmp_path)
    atom = _touch(config.repo_root / "vericcl/examples/atom/default.json", "{}")
    inputs = SimpleNamespace(
        hyperparameters=SimpleNamespace(slice_size_bytes=512 * 1024)
    )
    benchmark = SimpleNamespace(
        artifact=SimpleNamespace(xml_text="<algo/>"),
        schedule=object(),
        inputs=inputs,
    )
    _touch(
        config.experiment_root / "smoke" / "inter-node-k01-128m.xml",
        benchmark.artifact.xml_text,
    )
    monkeypatch.setattr(
        v100_module,
        "load_experiment_manifest",
        lambda *args, **kwargs: SimpleNamespace(
            cases=(case,), atom_path=atom
        ),
    )
    monkeypatch.setattr(v100_module, "resolve_inputs", lambda *args: inputs)
    monkeypatch.setattr(v100_module, "load_topology", lambda value: object())
    monkeypatch.setattr(
        v100_module,
        "representative_calibration_topology",
        lambda *args: object(),
    )
    monkeypatch.setattr(
        v100_module,
        "build_calibration_benchmark",
        lambda *args, **kwargs: benchmark,
    )

    def fail_factory(*args):
        raise RuntimeError("launcher failed")

    result = smoke(config, factory_builder=fail_factory)

    assert result.status is TaskStatus.FAILED
    assert result.log_path is not None
    error_path = Path(result.log_path)
    assert error_path.name == "runner-error.log"
    assert error_path.read_text(encoding="ascii") == "launcher failed\n"


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


def test_mpi_prefix_uses_tcp_and_node4_first_hostfile(tmp_path):
    config = _config(tmp_path)

    prefix = build_mpi_prefix(config, 8, ())

    assert prefix == (
        str(config.mpi_launcher),
        "--allow-run-as-root",
        "--prefix",
        str(config.mpi_launcher.parent.parent),
        "-np",
        "8",
        "--hostfile",
        str(config.hostfile_8),
        "-mca",
        "pml",
        "ob1",
        "-mca",
        "btl",
        "tcp,self,vader",
        "-mca",
        "btl_vader_single_copy_mechanism",
        "none",
        "-mca",
        "btl_tcp_if_include",
        "10.0.0.0/24",
    )


def test_performance_environment_uses_xml_specific_buffer(tmp_path):
    config = _config(tmp_path)
    case = _case(tmp_path)
    xml = _touch(config.experiment_root / "schedule.xml", "<algo/>")

    generated = build_performance_environment(
        config,
        case,
        xml,
        source="vericcl",
    )
    baseline = build_performance_environment(
        config,
        case,
        xml,
        source="baseline",
    )

    assert generated["NCCL_BUFFSIZE"] == str(2 * case.slice_size_bytes)
    assert baseline["NCCL_BUFFSIZE"] == str(2 * 1024 * 1024)
    assert generated["NCCL_IB_DISABLE"] == "1"
    assert "NCCL_IB_HCA" not in generated


def test_benchmark_xml_preserves_raw_and_structured_results(tmp_path):
    config = _config(tmp_path)
    case = _case(tmp_path)
    xml = _touch(config.experiment_root / "schedule.xml", "<algo/>")

    class Executor:
        requests = []

        def run(self, request):
            self.requests.append(request)
            return ProcessResult(
                0,
                (
                    "4194304 1048576 float none -1 "
                    "100 40 35 0 90 45 39 0\n"
                ),
                "NCCL INFO Connected 1 MSCCL algorithms\n",
            )

    executor = Executor()
    result = benchmark_xml(
        case,
        config,
        source="vericcl",
        xml_path=xml,
        begin="4M",
        end="4M",
        output_directory=config.experiment_root / "performance" / "case",
        executor=executor,
    )

    assert result.runs[0].in_place.algorithm_bandwidth_gbps == 45.0
    assert result.activation[0].confirmed is True
    assert result.stdout_path.read_text(encoding="utf-8").startswith(
        "4194304"
    )
    output = result.stdout_path.parent
    assert (output / "command.txt").is_file()
    assert (output / "environment.json").is_file()
    assert (output / "measurements.json").is_file()
    assert (output / "activation.json").is_file()
