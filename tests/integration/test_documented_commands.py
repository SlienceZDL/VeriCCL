"""Execute the current tutorial and compare published schemas to code."""
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
PATTERN = re.compile(r"```bash\n(.*?)\n```", re.S)


def commands():
    result = {}
    for block in PATTERN.findall((ROOT / "README.md").read_text()):
        arguments = shlex.split(block.replace("\\\n", ""))
        if arguments[:3] == [".venv/bin/python", "-m", "vericcl"]:
            assert arguments[3] not in result, "duplicate command template"
            result[arguments[3]] = arguments
    return result


def test_current_tutorial_has_complete_workflow_and_local_links():
    text = (ROOT / "README.md").read_text()
    assert "## Environment" in text and "## Prepare inputs" in text and "## Results" in text
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


def test_cpu_tutorial_commands_execute_in_a_clean_project(tmp_path):
    from tools.generate_examples import main as generate_examples

    script = commands()
    names = ("solve", "inspect", "verify")
    assert set(names) <= script.keys()
    # Isolate all user-visible input/output paths while using the already
    # validated development environment. Clean wheel installation is separate.
    shutil.copytree(ROOT / "vericcl", tmp_path / "vericcl", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "tools", tmp_path / "tools", ignore=shutil.ignore_patterns("__pycache__"))
    environment = dict(os.environ, PYTHONPATH=str(tmp_path))
    generate_examples(["--project-root", str(tmp_path), "--case", "readme"])
    inputs = tmp_path / "examples/readme/input/allreduce-hierarchy-outofplace"
    root = tmp_path / "examples/readme/output"
    values = {".venv/bin/python": sys.executable, "/path/to/VeriCCL": str(tmp_path),
              "CASE": "readme", "/path/to/run": str(root / "solve")}
    values.update({"/path/to/input/" + name + ".json": str(inputs / (name + ".json"))
                   for name in ("topology", "sketch", "atom", "runtime")})
    for name in names:
        if name == "verify":
            accepted = json.loads((root / "solve/best/accepted.json").read_text())
            values.update({"/path/to/schedule.xml": str(root / "solve" / accepted["candidate"]["xml"]),
                "/path/to/sidecar.json": str(root / "solve" / accepted["candidate"]["sidecar"]),
                "/path/to/source-run/models/MODEL_VERSION.json": str(root / "solve/models" / (accepted["model_version"] + ".json"))})
        values["RUN_ID"] = name
        arguments = [values.get(value, value) for value in script[name]]
        completed = subprocess.run(arguments, cwd=tmp_path, env=environment,
            capture_output=True, text=True, timeout=80)
        assert completed.returncode == 0, completed.stdout + completed.stderr
    for run in ("solve", "verify"):
        summary = json.loads((root / run / "run-summary.json").read_text())
        assert summary["contract_met"] is True
    assert (root / "solve/input_snapshot/sketch.json").is_file()


@pytest.mark.parametrize("name", ["solve", "verify", "calibrate", "inspect", "resume", "live"])
def test_every_public_command_has_help_and_documentation(name):
    from vericcl.cli.main import build_parser

    text = (ROOT / "README.md").read_text()
    assert "`" + name + "`" in text
    values = {"REQUESTS": "1", "SECONDS": "60"}
    parsed = build_parser().parse_args([values.get(value, value) for value in commands()[name][3:]])
    assert parsed.command == name
    result = subprocess.run([sys.executable, "-m", "vericcl", name, "--help"], cwd=ROOT,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0 and "usage:" in result.stdout
