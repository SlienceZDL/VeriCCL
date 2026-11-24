from dataclasses import replace
import json
import shutil
from pathlib import Path

import pytest

from vericcl.errors import SemanticError
from vericcl.execution.status import RunResult
from vericcl.experiments.model import ExperimentCase, load_experiment_manifest
from vericcl.experiments.state import ExperimentStateStore, TaskStatus
from vericcl.verification.online.runner import ProcessResult
from vericcl.experiments.v100 import (
    BENCHMARK_TRANSPORT,
    SMOKE_TASK_ID,
    V100ExperimentConfig,
    accepted_xml,
    benchmark_xml,
    bind_runtime,
    build_mpi_prefix,
    build_performance_environment,
    load_v100_config,
    preflight,
    smoke,
    solve_case,
    write_hostfile,
)


REPO_ROOT = Path(__file__).parents[3]
FIXTURES = "tests/fixtures/experiments-v5"
CASE_FILES = {
    "topology": FIXTURES + "/topology/v100-n2g4.json",
    "sketch": FIXTURES + "/sketch/v100-n2g4/ag-4m.json",
    "runtime": FIXTURES + "/runtime/v100-n2g4/ag-4m.json",
}


def _touch(path, text="x", *, executable=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="ascii")
    if executable:
        path.chmod(0o755)
    return path


def _config(tmp_path, *, hostfile_text=None, environment=None, runtime_paths=None):
    root = tmp_path / "experiment"
    repo = root / "repo"
    baseline = tmp_path / "baseline"
    baseline.mkdir(parents=True, exist_ok=True)
    manifest = _touch(repo / "exp" / "v100-k16-manifest.json", "{}")
    launcher = _touch(tmp_path / "opt" / "bin" / "mpirun", executable=True)
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
        "VERICCL_MPI_TCP_IF_INCLUDE": "192.0.2.0/24",
    }
    if environment is not None:
        values.update(environment)
    return V100ExperimentConfig(
        experiment_root=root,
        repo_root=repo,
        manifest_path=manifest,
        baseline_source=baseline,
        remote_host="worker-a",
        mpi_launcher=launcher,
        msccl_library_path=library,
        nccl_tests_binary_directory=tests,
        clock_sync_binary=clock,
        calibration_cache_path=root / "calibration" / "cache.json",
        hostfile_8=hostfile_8,
        hostfile_16=hostfile_16,
        environment=values,
        runtime_paths=runtime_paths or {},
    )


def _install_manifest(config):
    """Copy one converted v5 case into the config's repository."""
    case = {"topology_name": "v100-n2g4", "collective_label": "ag", "size_label": "4m"}
    for name, relative in CASE_FILES.items():
        target = config.repo_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / relative, target)
        case[name] = relative
    atom = FIXTURES + "/atom.json"
    shutil.copyfile(REPO_ROOT / atom, config.repo_root / atom)
    config.manifest_path.write_text(
        json.dumps({"schema_version": 2, "atom": atom, "cases": [case]}),
        encoding="ascii",
    )
    return load_experiment_manifest(config.manifest_path, repo_root=config.repo_root)


def _case(tmp_path):
    runtime = {
        "rank_map": [
            {"rank": rank, "host": "worker-a" if rank < 4 else "worker-b", "device": rank % 4}
            for rank in range(8)
        ],
        "environment": {"NCCL_BUFFSIZE": "1048576", "NCCL_PROTO": "Simple"},
        "scratch_bytes_per_rank": 4194304,
    }
    return ExperimentCase(
        task_id="v100-n2g4-ag-4m",
        topology_name="v100-n2g4",
        collective_label="ag",
        size_label="4m",
        topology_path=_touch(tmp_path / "input" / "topology.json", "{}"),
        sketch_path=_touch(tmp_path / "input" / "sketch.json", "{}"),
        runtime_path=_touch(tmp_path / "input" / "runtime.json", json.dumps(runtime)),
        rank_count=8,
        message_size_bytes=4 * 1024 * 1024,
        slice_size_bytes=512 * 1024,
        nccl_buffsize_bytes=1048576,
    )


def _accepting_execute(calls, *, xml_text="<algo/>"):
    def execute(request, context):
        calls.append((request, context))
        context.write_bytes("candidates/c1/schedule.xml", xml_text.encode())
        context.write_json("best/accepted.json", {
            "task_id": context.task_id,
            "candidate": {"xml": "candidates/c1/schedule.xml", "sidecar": "candidates/c1/sidecar.json"},
        })
        return RunResult(True, "completed", acceptance="accepted", stop_reason="tuning_rounds_exhausted")
    return execute


def test_preflight_requires_worker_a_first_and_allowed_root(tmp_path):
    config = _config(
        tmp_path,
        hostfile_text=(
            "worker-b slots=4\nworker-a slots=4\n"
        ),
    )

    with pytest.raises(SemanticError, match="remote_host must be first"):
        preflight(config)


def test_preflight_rejects_ib_enabled(tmp_path):
    config = _config(tmp_path, environment={"NCCL_IB_DISABLE": "0"})

    with pytest.raises(SemanticError, match="NCCL_IB_DISABLE"):
        preflight(config)


def test_preflight_requires_native_drivers(tmp_path):
    with pytest.raises(SemanticError, match="runtime_paths.vericcl_driver"):
        preflight(_config(tmp_path))
    driver = _touch(tmp_path / "runtime" / "vericcl-probe", executable=True)
    native = _touch(tmp_path / "runtime" / "vericcl-native")
    config = _config(tmp_path, runtime_paths={"vericcl_driver": str(driver), "vericcl_native": str(native)})
    with pytest.raises(SemanticError, match="vericcl_native is not executable"):
        preflight(config)
    native.chmod(0o755)
    preflight(config)


def test_hostfiles_encode_worker_a_before_worker_b(tmp_path):
    path = tmp_path / "hostfile"

    write_hostfile(path, 16)

    assert path.read_text(encoding="ascii") == (
        "worker-a slots=8\nworker-b slots=8\n"
    )


def test_bind_runtime_maps_logical_hosts_and_keeps_case_values(tmp_path):
    driver = tmp_path / "runtime" / "vericcl-probe"
    config = _config(
        tmp_path,
        environment={"NCCL_SOCKET_IFNAME": "eth0", "NCCL_BUFFSIZE": "999"},
        runtime_paths={"vericcl_driver": str(driver)},
    )
    for ranks in (8, 16):
        write_hostfile(config.hostfile(ranks), ranks, hostnames=("worker-a", "compute-b"))
    runtime = json.loads((REPO_ROOT / CASE_FILES["runtime"]).read_text())

    bound = bind_runtime(config, runtime)

    assert [(m["rank"], m["host"], m["device"]) for m in bound["rank_map"]] == [
        (rank, "worker-a" if rank < 4 else "compute-b", rank % 4) for rank in range(8)
    ]
    environment = bound["environment"]
    assert environment["NCCL_BUFFSIZE"] == runtime["environment"]["NCCL_BUFFSIZE"]
    assert environment["NCCL_SOCKET_IFNAME"] == "eth0"
    assert environment["NCCL_ALGO"] == "MSCCL,RING"
    assert environment["OMPI_MCA_btl_tcp_if_include"] == "192.0.2.0/24"
    # Calibration and trace runs measure under the benchmark's transport.
    assert {key: environment[key] for key in BENCHMARK_TRANSPORT} == dict(BENCHMARK_TRANSPORT)
    assert "VERICCL_GPU_MODEL" not in environment
    assert bound["paths"] == {"mpi_bin": str(config.mpi_launcher.parent),
                              "vericcl_driver": config.runtime_paths["vericcl_driver"]}
    assert runtime["rank_map"][4]["host"] == "worker-b"


def test_solve_case_runs_v5_solve_and_records_accepted_xml(tmp_path):
    config = _config(tmp_path)
    manifest = _install_manifest(config)
    case = manifest.cases[0]
    store = ExperimentStateStore(config.state_path)
    calls = []

    result = solve_case(case, config, atom_path=manifest.atom_path,
                        execute=_accepting_execute(calls), state_store=store, run_id="r1")

    assert result.status is TaskStatus.PASSED
    request, context = calls[0]
    assert request.purpose == "solve"
    assert request.inputs.rank_count == case.rank_count == 8
    project = config.experiment_root / "runs" / case.task_id
    assert context.root == project / "examples" / case.task_id / "output" / "r1"
    runtime = request.inputs.request_config["runtime"]
    assert {m["host"] for m in runtime["rank_map"]} == {"worker-a", "worker-b"}
    assert runtime["environment"]["NCCL_BUFFSIZE"] == str(case.nccl_buffsize_bytes)
    assert runtime["paths"]["mpi_bin"] == str(config.mpi_launcher.parent)
    saved = json.loads((project / "inputs" / "runtime.json").read_text())
    assert saved["environment"] == dict(runtime["environment"])
    assert [m["host"] for m in saved["rank_map"]] == [m["host"] for m in runtime["rank_map"]]
    xml = context.root / "candidates" / "c1" / "schedule.xml"
    assert result.log_path == str(xml)
    assert accepted_xml(context.root) == xml
    summary = json.loads((context.root / "run-summary.json").read_text())
    assert summary["execution"] == "completed"
    assert (context.root / "input_snapshot" / "runtime.json").is_file()
    assert store.load()[case.task_id] == result


def test_solve_case_requires_accepted_nonempty_xml(tmp_path):
    config = _config(tmp_path)
    manifest = _install_manifest(config)

    result = solve_case(manifest.cases[0], config, atom_path=manifest.atom_path,
                        execute=_accepting_execute([], xml_text=""), run_id="r1")

    assert result.status is TaskStatus.FAILED
    assert result.output_sha256 is None
    assert "missing or empty" in Path(result.log_path).read_text()


def test_solve_case_rejects_incomplete_run(tmp_path):
    config = _config(tmp_path)
    manifest = _install_manifest(config)

    def execute(request, context):
        return RunResult(False, "incomplete", stop_reason="no_qualified_baseline",
                         failure_category="online", optional_failures=("probe failed",))

    result = solve_case(manifest.cases[0], config, atom_path=manifest.atom_path,
                        execute=execute, run_id="r1")

    assert result.status is TaskStatus.FAILED
    message = Path(result.log_path).read_text()
    assert "stop_reason=no_qualified_baseline" in message
    assert "probe failed" in message


def test_solve_case_finalizes_executor_exception(tmp_path):
    config = _config(tmp_path)
    manifest = _install_manifest(config)

    def execute(request, context):
        raise RuntimeError("solver exploded")

    result = solve_case(manifest.cases[0], config, atom_path=manifest.atom_path,
                        execute=execute, run_id="r1")

    assert result.status is TaskStatus.FAILED
    assert result.failure_code == "SemanticError"
    run = (config.experiment_root / "runs" / manifest.cases[0].task_id
           / "examples" / manifest.cases[0].task_id / "output" / "r1")
    summary = json.loads((run / "run-summary.json").read_text())
    assert summary["execution"] == "failed"
    assert "solver exploded" in Path(result.log_path).read_text()


def test_smoke_runs_v5_calibration_on_two_inter_node_ranks(tmp_path):
    config = _config(tmp_path)
    _install_manifest(config)
    store = ExperimentStateStore(config.state_path)
    calls = []

    def execute(request, context):
        calls.append((request, context))
        return RunResult(True, "completed", model_coverage="qualified_components",
                         acceptance="not_required", stop_reason="calibration_finished")

    result = smoke(config, execute=execute, state_store=store, run_id="r1")

    assert result.status is TaskStatus.PASSED
    request, context = calls[0]
    assert request.purpose == "calibrate"
    inputs = request.inputs
    assert inputs.rank_count == 2
    config_ = inputs.request_config
    assert [(m["rank"], m["host"], m["device"]) for m in config_["runtime"]["rank_map"]] == [
        (0, "worker-a", 0), (1, "worker-b", 0)]
    assert {(e["src"], e["dst"]) for e in config_["topology"]["directed_links"]} == {(0, 1), (1, 0)}
    assert all(e["alpha_us"] == 4 for e in config_["topology"]["directed_links"])
    assert {r["id"] for r in config_["topology"]["shared_resources"]} == {
        "ib-n0-to-n1", "ib-n1-to-n0", "nic-n0-egress", "nic-n0-ingress", "nic-n1-egress", "nic-n1-ingress"}
    assert config_["sketch"]["hyperparameters"]["total_size_bytes"] == 128 * 1024 * 1024
    assert config_["sketch"]["solver"]["max_channels"] == 1
    assert config_["sketch"]["calibration"]["enabled"] is True
    assert config_["sketch"]["trace_and_measure"]["enabled"] is False
    assert config_["sketch"]["budgets"]["max_online_validation_total_s"] == 10800.0
    assert context.root == config.experiment_root / "smoke" / "examples" / SMOKE_TASK_ID / "output" / "r1"
    assert result.log_path == str(context.root / "run-summary.json")
    assert store.load()[SMOKE_TASK_ID] == result


def test_smoke_preserves_runner_error_details(tmp_path):
    config = _config(tmp_path)
    _install_manifest(config)
    _touch(
        config.experiment_root / "smoke" / "runner-error.log",
        "previous failure\n",
    )

    def execute(request, context):
        raise RuntimeError("launcher failed")

    result = smoke(config, execute=execute, run_id="r1")

    assert result.status is TaskStatus.FAILED
    assert result.log_path is not None
    error_path = Path(result.log_path)
    assert error_path.name == "runner-error.log"
    text = error_path.read_text(encoding="utf-8")
    assert "previous failure" not in text
    assert "stop_reason=exception" in text
    assert text.endswith("launcher failed\n")


def test_load_v100_config_rejects_unknown_fields(tmp_path):
    drivers = {"vericcl_driver": "/opt/vericcl/bin/vericcl-probe",
               "vericcl_native": "/opt/vericcl/bin/vericcl-native"}
    config = _config(tmp_path, runtime_paths=drivers)
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
        "runtime_paths": dict(drivers),
    }
    path.write_text(json.dumps(payload), encoding="ascii")
    assert load_v100_config(path) == config

    payload["runtime_paths"] = {"vericcl_native": drivers["vericcl_native"]}
    path.write_text(json.dumps(payload), encoding="ascii")
    with pytest.raises(SemanticError, match="runtime_paths requires vericcl_driver"):
        load_v100_config(path)

    payload["runtime_paths"] = dict(drivers, vericcl_native="relative/vericcl-native")
    path.write_text(json.dumps(payload), encoding="ascii")
    with pytest.raises(SemanticError, match="must be absolute"):
        load_v100_config(path)

    payload["runtime_paths"] = dict(drivers, unknown_tool="/opt/tool")
    path.write_text(json.dumps(payload), encoding="ascii")
    with pytest.raises(SemanticError, match="unknown entries"):
        load_v100_config(path)

    del payload["runtime_paths"]
    path.write_text(json.dumps(payload), encoding="ascii")
    with pytest.raises(SemanticError, match="fields"):
        load_v100_config(path)

    payload["runtime_paths"] = dict(drivers)
    payload["unexpected"] = True
    path.write_text(json.dumps(payload), encoding="ascii")
    with pytest.raises(SemanticError, match="fields"):
        load_v100_config(path)


def test_example_config_loads_with_required_runtime_paths():
    config = load_v100_config(REPO_ROOT / "examples" / "v100-experiments" / "config.example.json")
    assert set(config.runtime_paths) >= {"vericcl_driver", "vericcl_native"}
    assert config.environment["NCCL_IB_DISABLE"] == "1"


def test_mpi_prefix_uses_tcp_and_worker_a_first_hostfile(tmp_path):
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
        "192.0.2.0/24",
    )


def test_performance_environment_uses_case_runtime_buffer(tmp_path):
    config = _config(tmp_path)
    case = replace(_case(tmp_path), nccl_buffsize_bytes=4194304)
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

    assert generated["NCCL_BUFFSIZE"] == str(case.nccl_buffsize_bytes)
    assert generated["NCCL_BUFFSIZE"] != str(2 * case.slice_size_bytes)
    assert baseline["NCCL_BUFFSIZE"] == str(2 * 1024 * 1024)
    assert generated["NCCL_IB_DISABLE"] == "1"
    assert "NCCL_IB_HCA" not in generated


def test_manifest_cases_carry_explicit_runtime_buffer():
    manifest = load_experiment_manifest(
        REPO_ROOT / FIXTURES / "v100-k16-manifest.json", repo_root=REPO_ROOT)
    for case in manifest.cases:
        runtime = json.loads(case.runtime_path.read_text())
        assert case.nccl_buffsize_bytes == int(runtime["environment"]["NCCL_BUFFSIZE"])


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
    assert executor.requests[0].environment["NCCL_BUFFSIZE"] == "1048576"


def test_hostfile_writer_accepts_explicit_aliases(tmp_path):
    path = tmp_path / 'hosts'
    write_hostfile(path, 8, hostnames=('compute-a', 'compute-b'))
    assert path.read_text() == 'compute-a slots=4\ncompute-b slots=4\n'


def test_mpi_interface_follows_explicit_configuration(tmp_path):
    config = _config(tmp_path, environment={'VERICCL_MPI_TCP_IF_INCLUDE': 'eth9'})
    assert build_mpi_prefix(config, 8, ())[-1] == 'eth9'
