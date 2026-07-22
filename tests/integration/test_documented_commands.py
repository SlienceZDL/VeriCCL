from __future__ import annotations

import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

import pytest

from vericcl.input.loader import resolve_inputs


pytestmark = pytest.mark.phase07


PROJECT_ROOT = Path(__file__).parents[2]
README_EN = PROJECT_ROOT / "README.md"
README_ZH = PROJECT_ROOT / "README.zh-CN.md"
RUNTIME_GUIDE = PROJECT_ROOT / "docs" / "runtime-configuration.md"
REPORT_GUIDE = PROJECT_ROOT / "docs" / "validation-report.md"
COMMAND_PATTERN = re.compile(
    r"<!-- vericcl-doc-test: ([a-z-]+) -->\s*"
    r"```bash\s*\n([^\n]+)\n```"
)
BASH_BLOCK_PATTERN = re.compile(r"```bash\s*\n(.*?)\n```", re.DOTALL)
OUTPUT_SETUP_PATTERN = re.compile(
    r'^export VERICCL_OUTPUT_DIR="([^"\n]+)"$',
    re.MULTILINE,
)


def _commands_from(path):
    return dict(COMMAND_PATTERN.findall(path.read_text(encoding="utf-8")))


def _commands():
    commands = {}
    for path in (README_EN, RUNTIME_GUIDE, REPORT_GUIDE):
        text = path.read_text(encoding="utf-8")
        for name, command in COMMAND_PATTERN.findall(text):
            assert name not in commands
            commands[name] = command
    return commands


def test_bilingual_readmes_have_identical_tested_commands():
    assert _commands_from(README_EN) == _commands_from(README_ZH)


def test_bilingual_readmes_have_identical_shell_command_blocks():
    english = README_EN.read_text(encoding="utf-8")
    chinese = README_ZH.read_text(encoding="utf-8")
    assert BASH_BLOCK_PATTERN.findall(english) == BASH_BLOCK_PATTERN.findall(chinese)


@pytest.mark.parametrize("path", (README_EN, README_ZH))
def test_smoke_output_directory_is_initialized_before_solve(path):
    text = path.read_text(encoding="utf-8")
    solve_marker = text.index("<!-- vericcl-doc-test: solve -->")
    setup = OUTPUT_SETUP_PATTERN.search(text, 0, solve_marker)
    assert setup is not None
    assert setup.group(1)
    assert "$VERICCL_ROOT" in setup.group(1)
    assert 'mkdir -p "$VERICCL_OUTPUT_DIR"' in text[setup.end() : solve_marker]


def _run(command, output_dir):
    expanded = command.replace(
        "${VERICCL_OUTPUT_DIR}",
        str(output_dir),
    )
    arguments = shlex.split(expanded)
    assert arguments[:1] == [".venv/bin/python"]
    arguments[0] = sys.executable
    return subprocess.run(
        arguments,
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )


def test_documented_repository_commands_execute_in_order(tmp_path):
    commands = _commands()
    assert set(commands) == {
        "help",
        "solve",
        "verify",
        "example-validation",
    }

    for name in ("help", "solve", "verify", "example-validation"):
        completed = _run(commands[name], tmp_path / "runs")
        assert completed.returncode == 0, (
            "{} failed:\nstdout={}\nstderr={}".format(
                name,
                completed.stdout,
                completed.stderr,
            )
        )


def test_documented_inputs_resolve_and_runtime_formula_is_exact():
    inputs = resolve_inputs(
        PROJECT_ROOT / "vericcl/examples/topo/two_rank.json",
        PROJECT_ROOT / "vericcl/examples/sketch/allreduce_8m_1m.json",
        PROJECT_ROOT / "vericcl/examples/atom/constructive.json",
    )
    runtime_text = RUNTIME_GUIDE.read_text(encoding="utf-8")
    report_text = REPORT_GUIDE.read_text(encoding="utf-8")

    assert inputs.rank_count == 2
    assert inputs.hyperparameters.total_size_bytes == 8 * 1024 * 1024
    assert inputs.hyperparameters.slice_size_bytes == 1024 * 1024
    assert "NCCL_BUFFSIZE=2*slice_size_bytes" in runtime_text
    assert "MSCCL_CHUNKSTEPS 4" in runtime_text
    assert "MSCCL_SLICESTEPS 4" in runtime_text
    for slice_size in (1024, 1024 * 1024, 4 * 1024 * 1024):
        assert 2 * slice_size == slice_size + slice_size
    dimensions = {
        "input",
        "semantic",
        "state",
        "topology",
        "timing",
        "resource",
        "buffer",
        "endpoint",
        "deadlock",
        "xml",
        "bdd",
        "simulation",
        "runtime",
        "online",
    }
    documented = set(
        json.loads(
            re.search(
                r"<!-- validation-dimensions: (\[.*\]) -->",
                report_text,
            ).group(1)
        )
    )
    assert documented == dimensions
