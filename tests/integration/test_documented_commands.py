"""Execute the current Chinese tutorial and compare published schemas to code."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
PATTERN = re.compile(r"<!-- vericcl-v5-command: ([a-z-]+) -->\s*```bash\n(.*?)\n```", re.S)


def commands():
    pairs = PATTERN.findall((ROOT / "README.md").read_text())
    assert len(pairs) == len(dict(pairs)), "duplicate tutorial command identity"
    return dict(pairs)


def test_current_tutorial_has_complete_workflow_and_local_links():
    text = (ROOT / "README.md").read_text()
    assert "## 环境" in text and "## 输入" in text and "## 结果与故障" in text
    for path in ("input-schema.md", "architecture.md", "supported-configuration.md"):
        assert (ROOT / "docs" / path).is_file()
        assert "docs/" + path in text
    for path in re.findall(r"\]\(([^)]+)\)", text):
        if "://" not in path and not path.startswith("#"):
            assert (ROOT / path.split("#")[0]).exists(), path
    assert "import gurobipy" not in text
    assert "--output-dir output" not in text


def test_published_input_schemas_and_every_field_match_implementation():
    from vericcl.input.schema import public_schemas, SCHEMAS
    text = (ROOT / "docs/input-schema.md").read_text()
    def paths(spec, prefix):
        for key, child in spec.get("properties", {}).items():
            path = prefix + "." + key
            yield path
            yield from paths(child, path)
            if child.get("type") == "array":
                yield from paths(child["items"], path + "[]")
    for name, schema in public_schemas().items():
        assert json.loads((ROOT / "docs/input-schemas" / (name + ".json")).read_text()) == schema
        for field in paths(SCHEMAS[name], name):
            assert "`" + field + "`" in text, field


def tutorial_project(tmp_path, *, layout, operator="allgather", method="hierarchy"):
    shutil.copytree(ROOT / "vericcl", tmp_path / "vericcl", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "tools", tmp_path / "tools", ignore=shutil.ignore_patterns("__pycache__"))
    (tmp_path / ".venv").symlink_to(Path(sys.prefix), target_is_directory=True)
    return dict(os.environ, PYTHONPATH=str(tmp_path), VERICCL_ROOT=str(tmp_path),
        VERICCL_PYTHON=sys.executable, VERICCL_CASE="documented-case",
        VERICCL_INPUT=str(tmp_path / "examples/documented-case/input/base"),
        VERICCL_LAYOUT=json.dumps(layout), VERICCL_OPERATOR=operator, VERICCL_METHOD=method,
        VERICCL_M="1024", VERICCL_S="1024", VERICCL_CHANNELS="7", VERICCL_CPU_BUDGET="2",
        VERICCL_WALL_S="120", VERICCL_TUNING_ROUNDS="0", VERICCL_SCRATCH_BYTES="16777216", VERICCL_NCCL_BUFFSIZE="16777216")


def run_documented(script, tmp_path, environment, timeout=120):
    result = subprocess.run(["bash", "-euo", "pipefail", "-c", script], cwd=tmp_path,
        env=environment, capture_output=True, text=True, timeout=timeout)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.mark.parametrize("operator", ["allgather", "allreduce"])
@pytest.mark.parametrize("method", ["hierarchy", "constructive_trees"])
def test_cpu_tutorial_commands_execute_in_a_clean_project(tmp_path, operator, method):
    script = commands()
    environment = tutorial_project(tmp_path, layout={"localhost": [1, 3]},
                                   operator=operator, method=method)
    # The override deliberately differs from the input file. Verification must
    # recover the original effective request, including it. Parameter selection
    # may choose fewer channels while retaining that request identity.
    run_documented("\n".join([script["input-generate"], script["common-args"],
        'VERICCL_ARGS+=(--override-input --set sketch.solver.max_channels=2)',
        script["solve"], script["inspect"], script["verify"]]), tmp_path, environment)
    root = tmp_path / "examples/documented-case/output"
    runs = list(root.iterdir())
    assert len(runs) == 2
    for run in runs:
        metadata = json.loads((run / "run.json").read_text())
        summary = metadata["result"]
        assert summary["contract_met"] is True
        accepted = metadata["best"]
        config = accepted["candidate"]["effective_config"]
        assert config["topology"]["ranks"] == 2
        assert config["sketch"]["collective"]["operator"] == operator
        assert config["atom"]["synthesis_method"] == method
        assert config["sketch"]["solver"]["max_channels"] in (1, 2)
        effective = metadata["effective_request"]
        assert effective["sketch"]["solver"]["max_channels"] == 2
    source = next(path for path in runs if path.name.startswith("solve-"))
    original = json.loads((source / "process/input_snapshot/sketch.json").read_text())
    assert original["solver"]["max_channels"] == 7


LAYOUTS = [
    {"localhost": list(range(4))},
    {"host-a": list(range(4)), "host-b": list(range(4))},
    {"host-a": list(range(8)), "host-b": list(range(8))},
    {f"host-{i}": list(range(4)) for i in range(4)},
    {"host-a": [1, 3, 6], "host-b": [2]},
]


@pytest.mark.parametrize("layout", LAYOUTS, ids=["1x4", "2x4", "2x8", "4x4", "unequal-devices"])
@pytest.mark.parametrize("operator,method", [(o, m) for o in ("allgather", "allreduce")
                                           for m in ("hierarchy", "constructive_trees")])
def test_general_templates_preserve_scale_and_build_matching_gpu_commands(tmp_path, layout, operator, method):
    from vericcl.input.loader import load_request
    from vericcl.input.json_codec import canonical_json
    from vericcl.topology.loader import load_topology
    script = commands()
    environment = tutorial_project(tmp_path, layout=layout, operator=operator, method=method)
    environment.update(VERICCL_GPU_INPUT=str(tmp_path / "examples/documented-case/input/gpu"),
        VERICCL_PROBE=str(tmp_path / "runtime/probe"), VERICCL_NATIVE=str(tmp_path / "runtime/native"),
        MPI_HOME=str(tmp_path / "mpi"), LD_LIBRARY_PATH=str(tmp_path / "lib"),
        VERICCL_CONTROL_ADDRESS="192.0.2.10", VERICCL_CONTROL_PORT="24950")
    run_documented(script["input-generate"] + "\n" + script["gpu-setup"], tmp_path, environment)
    files = {name: Path(environment["VERICCL_GPU_INPUT"]) / (name + ".json")
             for name in ("topology", "sketch", "atom", "runtime")}
    count = sum(len(devices) for devices in layout.values())
    inputs = load_request(files)
    load_topology(inputs)
    config = json.loads(canonical_json(inputs.request_config))
    assert inputs.rank_count == count
    expected_mapping = [{"rank": rank, "host": host, "device": device}
                        for rank, (host, device) in enumerate((h, d) for h, ds in layout.items() for d in ds)]
    assert config["runtime"]["rank_map"] == expected_mapping
    assert sorted(rank for node in config["topology"]["nodes"] for rank in node["ranks"]) == list(range(count))
    assert len(config["topology"]["directed_links"]) == count * (count - 1)
    resources = config["topology"]["shared_resources"]
    assert len(resources) == (2 * len(layout) if len(layout) > 1 else 0)
    for edge in config["topology"]["directed_links"]:
        memberships = sum([edge["src"], edge["dst"]] in item["links"] for item in resources)
        same_host = expected_mapping[edge["src"]]["host"] == expected_mapping[edge["dst"]]["host"]
        assert memberships == (0 if same_host else 2)
        assert set(edge["resources"]) == {item["id"] for item in resources if [edge["src"], edge["dst"]] in item["links"]}
    # Check every optional entry's required fields without starting any GPU job.
    load_request(files, {"sketch.calibration.enabled": True}, override_input=True, purpose="calibrate")
    load_request(files, {"sketch.trace_and_measure.enabled": True}, override_input=True)
    # A fake MPI executable captures the exact documented launch arguments.
    # It never invokes MPI, SSH, a GPU runtime, or the benchmark binary.
    mpi = Path(environment["MPI_HOME"]) / "bin/mpirun"
    mpi.parent.mkdir(parents=True)
    capture = tmp_path / "mpi-arguments.json"
    mpi.write_text("#!" + sys.executable + "\nimport json,sys\nfrom pathlib import Path\n" +
                   "Path(" + repr(str(capture)) + ").write_text(json.dumps(sys.argv[1:]))\n")
    mpi.chmod(0o755)
    source = tmp_path / "examples/documented-case/output/source"
    source.mkdir(parents=True)
    (source / "run.json").write_text(json.dumps({"best": {"candidate": {"effective_config": config}}}))
    environment.update(VERICCL_RUN=str(source), VERICCL_NCCL_TESTS=str(tmp_path / "test-binaries"))
    run_documented(script["nccl-tests"], tmp_path, environment)
    argv = json.loads(capture.read_text())
    assert argv[:2] == ["--bind-to", "none"]
    contexts, context = [], []
    for word in argv[2:] + [":"]:
        if word == ":":
            contexts.append(context)
            context = []
        else:
            context.append(word)
    assert len(contexts) == count
    size = 1024 * (count if operator == "allgather" else 1)
    for context, entry in zip(contexts, expected_mapping):
        assert context[:4] == ["-np", "1", "--host", entry["host"] + ":" + str(len(layout[entry["host"]]))]
        assert "NCCL_TESTS_DEVICE=" + str(entry["device"]) in context
        assert context[context.index("-b") + 1] == str(size)
        assert context[context.index("-e") + 1] == str(size)
        assert context[context.index("-g") + 1] == "1"
        assert context[context.index("-d") + 1] == "float"
        binary = "all_gather_perf" if operator == "allgather" else "all_reduce_perf"
        assert str(tmp_path / "test-binaries" / binary) in context
        if operator == "allreduce":
            assert context[context.index("-o") + 1] == "sum"


@pytest.mark.parametrize("name", ["solve", "verify", "calibrate", "inspect", "resume"])
def test_every_public_command_has_help_and_documentation(name):
    text = (ROOT / "README.md").read_text()
    assert "`" + name + "`" in text
    result = subprocess.run([sys.executable, "-m", "vericcl", name, "--help"], cwd=ROOT,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0 and "usage:" in result.stdout


@pytest.mark.parametrize("operator", ["allgather", "allreduce"])
@pytest.mark.parametrize("layout", LAYOUTS, ids=["1x4", "2x4", "2x8", "4x4", "unequal-devices"])
def test_manual_template_matches_operator_and_scale(tmp_path, operator, layout):
    from vericcl.input.loader import load_request
    from vericcl.planner.manual import plan_manual
    from vericcl.topology import load_topology
    environment = tutorial_project(tmp_path, layout=layout, operator=operator)
    script = commands()
    run_documented(script["input-generate"] + "\n" + script["manual-generate"], tmp_path, environment)
    directory = tmp_path / "examples/documented-case/input/manual"
    inputs = load_request({key: directory / (key + ".json") for key in ("topology", "sketch", "atom", "runtime")})
    plan = plan_manual(inputs, load_topology(inputs))
    assert plan.rank_count == sum(len(devices) for devices in layout.values())
    assert plan.collective.kind.value == operator
    assert inputs.atom_constraints.forbidden_transfers == ()
