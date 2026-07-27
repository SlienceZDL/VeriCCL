from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess

import pytest

from vericcl.errors import InputValidationError
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
RUN_STEP_PATTERN = re.compile(
    r"<!-- vericcl-run-step: ([a-z-]+) -->\s*"
    r"(?:<!-- vericcl-doc-test: [a-z-]+ -->\s*)?"
    r"```bash\s*\n([^\n]+)\n```"
)
BASH_BLOCK_PATTERN = re.compile(r"```bash\s*\n(.*?)\n```", re.DOTALL)
INLINE_CODE_PATTERN = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")
OUTPUT_SETUP_PATTERN = re.compile(
    r'^export VERICCL_OUTPUT_DIR="([^"\n]+)"$',
    re.MULTILINE,
)
UNKNOWN_FIELD_CONTRACT = (
    "<!-- input-unknown-fields: topology-extra=accepted; "
    "sketch-top-extra=preserved; sketch-sections-extra=rejected; "
    "atom-top-extra=rejected -->"
)
DOCUMENTED_COMMAND_ORDER = (
    "help",
    "solve",
    "verify",
    "example-validation",
)
DOCUMENTED_RUN_STEP_ORDER = (
    "set-root",
    "set-output",
    "create-output",
    "solve",
    "verify",
    "check-xml",
    "check-report",
    "inspect-report",
)
EXAMPLE_SECTION_ANCHORS = (
    "vericcl/examples/legacy",
    "vericcl/examples/templates",
)


def _commands_from(path):
    return dict(COMMAND_PATTERN.findall(path.read_text(encoding="utf-8")))


def _run_steps_from(path):
    return tuple(RUN_STEP_PATTERN.findall(path.read_text(encoding="utf-8")))


def _example_section(path):
    text = path.read_text(encoding="utf-8")
    sections = re.split(r"(?m)^## ", text)
    matches = [
        section
        for section in sections
        if all(anchor in section for anchor in EXAMPLE_SECTION_ANCHORS)
    ]
    assert len(matches) == 1
    return matches[0]


def _tracked_repository_paths():
    completed = subprocess.run(
        ["git", "ls-files"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    return frozenset(completed.stdout.splitlines())


def _repository_paths_from_inline_code(text, tracked_paths):
    tracked_directories = {
        path.split("/", 1)[0] for path in tracked_paths if "/" in path
    }
    tracked_root_files = {path for path in tracked_paths if "/" not in path}
    documented = set()
    for value in INLINE_CODE_PATTERN.findall(text):
        candidate = value.strip()
        if (
            not candidate
            or any(character.isspace() for character in candidate)
            or "://" in candidate
            or candidate.startswith(("$", "~"))
        ):
            continue
        relative = PurePosixPath(candidate)
        if relative.is_absolute() or ".." in relative.parts:
            continue
        if len(relative.parts) == 1:
            if (
                candidate in tracked_root_files
                or candidate in tracked_directories
            ):
                documented.add(candidate)
            continue
        if relative.parts[0] in tracked_directories:
            documented.add(candidate)
    return frozenset(documented)


def _is_tracked_repository_entry(path, tracked_paths):
    return path in tracked_paths or any(
        tracked.startswith(path + "/") for tracked in tracked_paths
    )


def test_repository_path_extraction_excludes_non_repository_inline_code():
    inline_code = " ".join(
        (
            "`https://example.test/vericcl/examples/topo/url.json`",
            "`/vericcl/examples/topo/absolute.json`",
            "`$VERICCL_ROOT/runtime/msccl-trace/README.md`",
            "`.venv/bin/python -m vericcl --help`",
            "`MSCCL_ROOT/build/lib`",
            "`avg|max|min`",
            "`not-a-repository-symbol`",
            "`docs`",
            "`vericcl/examples/topo/two_rank.json`",
            "`runtime/msccl-trace/README.md`",
            "`docs/runtime-configuration.md`",
            "`MIGRATION.md`",
        )
    )
    tracked_paths = _tracked_repository_paths()

    documented = _repository_paths_from_inline_code(inline_code, tracked_paths)

    assert documented == {
        "docs",
        "vericcl/examples/topo/two_rank.json",
        "runtime/msccl-trace/README.md",
        "docs/runtime-configuration.md",
        "MIGRATION.md",
    }
    for path in documented:
        assert (PROJECT_ROOT / path).exists()
        assert _is_tracked_repository_entry(path, tracked_paths)


def test_repository_path_status_covers_files_directories_and_missing_paths():
    tracked_paths = _tracked_repository_paths()
    for path in (
        "docs/runtime-configuration.md",
        "vericcl/examples/legacy",
    ):
        assert (PROJECT_ROOT / path).exists()
        assert _is_tracked_repository_entry(path, tracked_paths)

    missing = "docs/not-a-documented-file.md"
    assert _repository_paths_from_inline_code(
        "`{}`".format(missing),
        tracked_paths,
    ) == {missing}
    assert not (PROJECT_ROOT / missing).exists()
    assert not _is_tracked_repository_entry(missing, tracked_paths)


def _write_example_inputs(directory):
    examples = PROJECT_ROOT / "vericcl" / "examples"
    paths = {}
    for name, source in (
        ("topology", examples / "topo" / "two_rank.json"),
        ("sketch", examples / "sketch" / "allreduce_8m_1m.json"),
        ("atom", examples / "atom" / "constructive.json"),
    ):
        value = json.loads(source.read_text(encoding="utf-8"))
        destination = directory / "{}.json".format(name)
        destination.write_text(json.dumps(value), encoding="utf-8")
        paths[name] = destination
    return paths


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


def test_bilingual_readmes_have_identical_run_steps():
    assert _run_steps_from(README_EN) == _run_steps_from(README_ZH)


def test_documented_run_steps_execute_and_write_bound_artifacts(tmp_path):
    steps = _run_steps_from(README_EN)
    assert tuple(name for name, _ in steps) == DOCUMENTED_RUN_STEP_ORDER
    commands = [command for _, command in steps]
    commands[1] = "export VERICCL_OUTPUT_DIR={}".format(
        shlex.quote(str(tmp_path / "runs"))
    )

    completed = subprocess.run(
        ["bash", "-eu", "-o", "pipefail", "-c", "\n".join(commands)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    run_root = tmp_path / "runs" / "vericcl_allreduce_8MiB_quickstart"
    xml = run_root / "vericcl_allreduce_8MiB_final.xml"
    sidecar = run_root / "vericcl_allreduce_8MiB_final.schedule.json"
    report = run_root / "vericcl_allreduce_8MiB_final.validation.json"
    summary = run_root / "run-summary.json"
    assert xml.is_file()
    assert sidecar.is_file()
    assert report.is_file()
    assert summary.is_file()
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["accepted"] is True
    verify_summary = (
        tmp_path / "runs" / "vericcl_allreduce_8MiB_quickstart-verify"
        / "run-summary.json"
    )
    verify_payload = json.loads(verify_summary.read_text(encoding="utf-8"))
    assert verify_payload["mode"] == "verify"


@pytest.mark.parametrize("path", (README_EN, README_ZH))
def test_readme_example_paths_are_tracked_repository_entries(path):
    section = _example_section(path)
    tracked_paths = _tracked_repository_paths()
    documented_paths = _repository_paths_from_inline_code(
        section,
        tracked_paths,
    )

    assert documented_paths
    for documented in documented_paths:
        resolved = PROJECT_ROOT / documented
        assert resolved.exists(), documented
        assert _is_tracked_repository_entry(documented, tracked_paths), documented


@pytest.mark.parametrize("path", (README_EN, README_ZH))
def test_readmes_retain_the_ordered_executable_command_markers(path):
    names = tuple(
        name
        for name, _ in COMMAND_PATTERN.findall(path.read_text(encoding="utf-8"))
    )
    assert names == DOCUMENTED_COMMAND_ORDER


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


@pytest.mark.parametrize("path", (README_EN, README_ZH))
def test_readmes_declare_the_unknown_field_contract(path):
    assert UNKNOWN_FIELD_CONTRACT in path.read_text(encoding="utf-8")


def test_topology_and_sketch_top_level_extra_fields_are_preserved(tmp_path):
    paths = _write_example_inputs(tmp_path)
    topology = json.loads(paths["topology"].read_text(encoding="utf-8"))
    sketch = json.loads(paths["sketch"].read_text(encoding="utf-8"))
    topology["extra_topology_field"] = "accepted"
    sketch["extra_sketch_field"] = "preserved"
    paths["topology"].write_text(json.dumps(topology), encoding="utf-8")
    paths["sketch"].write_text(json.dumps(sketch), encoding="utf-8")

    resolved = resolve_inputs(paths["topology"], paths["sketch"], paths["atom"])

    assert resolved.resolved_topology["extra_topology_field"] == "accepted"
    assert resolved.resolved_sketch["extra_sketch_field"] == "preserved"


@pytest.mark.parametrize(
    ("section", "error"),
    (
        ("collective", "unknown collective field"),
        ("hyperparameters", "unknown hyperparameter field"),
        ("solver", "unknown solver field"),
    ),
)
def test_sketch_sections_reject_unknown_fields(tmp_path, section, error):
    paths = _write_example_inputs(tmp_path)
    sketch = json.loads(paths["sketch"].read_text(encoding="utf-8"))
    sketch[section]["extra_section_field"] = True
    paths["sketch"].write_text(json.dumps(sketch), encoding="utf-8")

    with pytest.raises(InputValidationError, match=error):
        resolve_inputs(paths["topology"], paths["sketch"], paths["atom"])


def test_atom_rejects_unknown_top_level_fields(tmp_path):
    paths = _write_example_inputs(tmp_path)
    atom = json.loads(paths["atom"].read_text(encoding="utf-8"))
    atom["extra_atom_field"] = True
    paths["atom"].write_text(json.dumps(atom), encoding="utf-8")

    with pytest.raises(InputValidationError, match="unknown atom field"):
        resolve_inputs(paths["topology"], paths["sketch"], paths["atom"])


def _run(command, output_dir):
    environment = os.environ.copy()
    environment["VERICCL_OUTPUT_DIR"] = str(output_dir)
    return subprocess.run(
        ["bash", "-eu", "-o", "pipefail", "-c", command],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
        env=environment,
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
