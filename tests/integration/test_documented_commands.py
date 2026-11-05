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


def test_cpu_tutorial_commands_execute_in_a_clean_project(tmp_path):
    script = commands()
    names = ("quickstart-setup", "quickstart-generate", "quickstart-solve", "quickstart-inspect", "quickstart-verify")
    assert set(names) <= script.keys()
    # Isolate all user-visible input/output paths while using the already
    # validated development environment. Clean wheel installation is separate.
    shutil.copytree(ROOT / "vericcl", tmp_path / "vericcl", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "tools", tmp_path / "tools", ignore=shutil.ignore_patterns("__pycache__"))
    (tmp_path / ".venv").symlink_to(Path(sys.prefix), target_is_directory=True)
    environment = dict(os.environ, PYTHONPATH=str(tmp_path))
    completed = subprocess.run(["bash", "-euo", "pipefail", "-c", "\n".join(script[name] for name in names)],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=80)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    root = tmp_path / "examples/tutorial/output"
    for run in ("first", "checked"):
        summary = json.loads((root / run / "run-summary.json").read_text())
        assert summary["contract_met"] is True
    assert (root / "first/input_snapshot/sketch.json").is_file()


@pytest.mark.parametrize("name", ["solve", "verify", "calibrate", "inspect", "resume", "live"])
def test_every_public_command_has_help_and_documentation(name):
    text = (ROOT / "README.md").read_text()
    assert "`" + name + "`" in text
    result = subprocess.run([sys.executable, "-m", "vericcl", name, "--help"], cwd=ROOT,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0 and "usage:" in result.stdout
