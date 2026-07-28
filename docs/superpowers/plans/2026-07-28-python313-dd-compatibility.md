# Python 3.13 dd Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a fresh VeriCCL editable install select a compatible `dd` release on CPython 3.10–3.13 and fail early with a clear message outside that range.

**Architecture:** Packaging metadata owns the supported interpreter range and mutually exclusive `dd` environment markers. The bilingual README exposes one executable preflight command before environment creation, while tests execute the documented command with controlled version tuples and inspect the real `setup.py` metadata.

**Tech Stack:** Python 3.10–3.13, setuptools, PEP 508 environment markers, `dd.autoref.BDD`, pytest, Markdown, Bash, uv.

## Global Constraints

- Supported interpreter range is exactly CPython `>=3.10,<3.14`.
- Python 3.10 resolves `dd>=0.5.7,<0.6`.
- Python 3.11–3.13 resolve `dd>=0.6,<0.7`.
- BDD variables, set operations, enumeration, error mapping, and validation results do not change.
- English and Chinese README Bash blocks remain byte-identical.
- Code, tests, setup metadata, and machine-readable files contain no Chinese characters.
- CUDA/MSCCL compilation and GPU execution remain `not_run` unless actually executed.

---

### Task 1: Bind packaging metadata to supported Python and dd versions

**Files:**
- Create: `tests/unit/test_packaging_metadata.py`
- Modify: `setup.py:53-62`

**Interfaces:**
- Consumes: setuptools `setup()` keyword arguments and PEP 508 requirement markers.
- Produces: `python_requires=">=3.10,<3.14"` and two mutually exclusive `dd` requirements selected by `python_version`.

- [ ] **Step 1: Write failing metadata tests**

Create `tests/unit/test_packaging_metadata.py` with a helper that executes the real setup file while capturing the setuptools call:

```python
from pathlib import Path
import runpy
from unittest import mock

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet


PROJECT_ROOT = Path(__file__).parents[2]
SETUP_PATH = PROJECT_ROOT / "setup.py"


def _setup_metadata():
    with mock.patch("setuptools.setup") as setup:
        runpy.run_path(str(SETUP_PATH), run_name="__vericcl_setup_test__")
    return setup.call_args.kwargs


def _selected_dd(version):
    environment = default_environment()
    environment["python_version"] = version
    requirements = (
        Requirement(value)
        for value in _setup_metadata()["install_requires"]
        if Requirement(value).name == "dd"
    )
    return tuple(
        requirement
        for requirement in requirements
        if requirement.marker.evaluate(environment)
    )


def test_setup_declares_supported_python_range():
    assert _setup_metadata()["python_requires"] == ">=3.10,<3.14"


def test_setup_selects_one_dd_series_for_each_supported_python():
    expected = {
        "3.10": ">=0.5.7,<0.6",
        "3.11": ">=0.6,<0.7",
        "3.12": ">=0.6,<0.7",
        "3.13": ">=0.6,<0.7",
    }
    for version, specifier in expected.items():
        selected = _selected_dd(version)
        assert len(selected) == 1
        assert selected[0].specifier == SpecifierSet(specifier)
```

- [ ] **Step 2: Run the tests and verify the current metadata fails**

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_packaging_metadata.py -q
```

Expected: failure because the current Python range is `>=3.9` and no `dd 0.6.x` marker exists.

- [ ] **Step 3: Implement the minimal metadata change**

Replace the single `dd` requirement and Python range in `setup.py` with:

```python
        'dd>=0.5.7,<0.6; python_version == "3.10"',
        'dd>=0.6,<0.7; python_version >= "3.11"',
```

and:

```python
    python_requires=">=3.10,<3.14",
```

- [ ] **Step 4: Run packaging and BDD tests**

Run:

```bash
.venv/bin/python -m pytest \
  tests/unit/test_packaging_metadata.py \
  tests/unit/verification/test_bdd_flow.py \
  tests/unit/verification/test_bdd_order.py -q
```

Expected: all tests pass with the existing BDD semantics.

- [ ] **Step 5: Commit Task 1**

```bash
git add setup.py tests/unit/test_packaging_metadata.py
git commit -m "fix: select dd by python version"
```

### Task 2: Add an executable Python-version preflight to both READMEs

**Files:**
- Modify: `README.md:13-32`
- Modify: `README.zh-CN.md:13-32`
- Modify: `tests/integration/test_documented_commands.py:1-70,455-490`

**Interfaces:**
- Consumes: the host `python3` interpreter before virtual-environment creation.
- Produces: one `vericcl-doc-test: python-version` command that exits zero for 3.10–3.13 and nonzero with the detected version otherwise.

- [ ] **Step 1: Extend the documented-command test contract**

Import `sys` in `tests/integration/test_documented_commands.py`, add
`"python-version"` to the expected documented command names, and add:

```python
def _run_version_check(command, version):
    arguments = shlex.split(command)
    assert arguments[:2] == ["python3", "-c"]
    script = "import sys; sys.version_info = {!r}; ".format(version)
    return subprocess.run(
        [sys.executable, "-c", script + arguments[2]],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )


@pytest.mark.parametrize("version", ((3, 10), (3, 11), (3, 12), (3, 13)))
def test_documented_python_version_check_accepts_supported_versions(version):
    command = _commands_from(README_EN)["python-version"]
    completed = _run_version_check(command, version)
    assert completed.returncode == 0, completed.stderr
    assert "{}.{}".format(*version) in completed.stdout


@pytest.mark.parametrize("version", ((3, 9), (3, 14)))
def test_documented_python_version_check_rejects_unsupported_versions(version):
    command = _commands_from(README_EN)["python-version"]
    completed = _run_version_check(command, version)
    assert completed.returncode != 0
    assert "{}.{}".format(*version) in completed.stderr
    assert "3.10-3.13" in completed.stderr
```

Keep the existing repository-command loop limited to `help`, `solve`,
`verify`, and `example-validation`; the new tests execute the version command
with controlled interpreter versions.

- [ ] **Step 2: Run the focused test and verify the command is absent**

Run:

```bash
.venv/bin/python -m pytest \
  tests/integration/test_documented_commands.py \
  -k 'python_version or repository_commands' -q
```

Expected: failure because neither README defines `python-version`.

- [ ] **Step 3: Add the same executable check to both READMEs**

Before the clone/install block, state that CPython 3.10–3.13 is supported and
add the following byte-identical Bash block to both files:

````markdown
<!-- vericcl-doc-test: python-version -->
```bash
python3 -c 'import sys; v = sys.version_info[:2]; sys.exit("VeriCCL requires Python 3.10-3.13; found {}.{}.".format(*v)) if not (3, 10) <= v < (3, 14) else print("VeriCCL Python version check passed: {}.{}".format(*v))'
```
````

In the English prose, describe this as a preflight before virtual-environment
creation. In the Chinese prose, describe the same contract in Chinese without
changing the Bash block.

- [ ] **Step 4: Run the documentation behavior tests**

Run:

```bash
.venv/bin/python -m pytest \
  tests/integration/test_documented_commands.py \
  tests/integration/test_workflow_artifacts.py -q
```

Expected: supported versions pass, unsupported versions fail with the required
message, README Bash blocks match, and the quickstart commands still produce
their artifacts.

- [ ] **Step 5: Commit Task 2**

```bash
git add README.md README.zh-CN.md tests/integration/test_documented_commands.py
git commit -m "docs: check supported python before install"
```

### Task 3: Verify fresh Python 3.13 installation and record evidence

**Files:**
- Modify: `docs/final-validation-report.md`

**Interfaces:**
- Consumes: the Task 1 packaging markers and Task 2 installation command.
- Produces: fresh installation and test evidence with unsupported hardware work explicitly retained as `not_run`.

- [ ] **Step 1: Create a clean Python 3.13 environment and install**

Run from the worktree:

```bash
export VERICCL_PY313_ROOT="$(mktemp -d /tmp/vericcl-py313.XXXXXX)"
uv venv --python 3.13 --seed "$VERICCL_PY313_ROOT/.venv"
uv pip install --python "$VERICCL_PY313_ROOT/.venv/bin/python" \
  -r requirements-dev.txt
uv pip install --python "$VERICCL_PY313_ROOT/.venv/bin/python" -e .
"$VERICCL_PY313_ROOT/.venv/bin/python" -m pip check
"$VERICCL_PY313_ROOT/.venv/bin/python" -c \
  'import dd, vericcl; print(dd.__version__, vericcl.__version__)'
```

Expected: editable installation succeeds, `pip check` reports no broken
requirements, `dd` is 0.6.x, and VeriCCL is `0.1.0`.

- [ ] **Step 2: Run BDD and full tests under Python 3.13**

Run:

```bash
"$VERICCL_PY313_ROOT/.venv/bin/python" -m pytest \
  tests/unit/verification/test_bdd_flow.py \
  tests/unit/verification/test_bdd_order.py -q
"$VERICCL_PY313_ROOT/.venv/bin/python" -m pytest -q
```

Expected: BDD tests and the complete test suite pass; hardware tests may skip.

- [ ] **Step 3: Verify published Linux wheels for every supported version**

Query the official PyPI JSON endpoints and assert the required wheel filenames:

```bash
for version in 0.5.7 0.6.0; do
  curl -fsSL "https://pypi.org/pypi/dd/$version/json" \
    > "$VERICCL_PY313_ROOT/dd-$version.json"
done
"$VERICCL_PY313_ROOT/.venv/bin/python" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["VERICCL_PY313_ROOT"])
files_057 = {
    item["filename"]
    for item in json.loads((root / "dd-0.5.7.json").read_text())["urls"]
}
files_060 = {
    item["filename"]
    for item in json.loads((root / "dd-0.6.0.json").read_text())["urls"]
}
assert any("cp310" in name and "manylinux" in name for name in files_057)
for abi in ("cp311", "cp312", "cp313"):
    assert any(abi in name and "manylinux" in name for name in files_060)
print("supported Linux dd wheels verified")
PY
```

Expected: `supported Linux dd wheels verified`. This proves artifact
availability, not execution on Linux; Linux execution remains `not_run` unless
rerun on the server.

- [ ] **Step 4: Run final software and static gates**

Run:

```bash
.venv/bin/python -m pytest -q
python3 -m compileall -q vericcl tests runtime/msccl-trace/tools
git diff --check
rg -n '[\p{Han}]' vericcl tests runtime setup.py \
  --glob '*.py' --glob '*.c' --glob '*.cc' --glob '*.cu' \
  --glob '*.h' --glob '*.json'
```

Expected: tests and compileall pass, `git diff --check` is clean, and `rg`
returns 1 because no Chinese characters are present in source or
machine-readable files.

- [ ] **Step 5: Record exact results and limitations**

Append a dated section to `docs/final-validation-report.md` containing:

- the Python 3.13 editable-install result and resolved `dd` version;
- Python 3.13 BDD and complete pytest counts;
- existing-environment complete pytest counts;
- Linux wheel artifact checks;
- explicit `not_run` status for Linux execution, CUDA/MSCCL compilation, GPU
  execution, and nccl-tests performance.

- [ ] **Step 6: Commit Task 3**

```bash
git add docs/final-validation-report.md
git commit -m "docs: record python 3.13 installation validation"
```

### Task 4: Review, publish, merge, and clean up

**Files:**
- No source files; Git and GitHub state only.

**Interfaces:**
- Consumes: all verified commits from Tasks 1–3.
- Produces: a merged default branch with no remaining temporary local or remote branch.

- [ ] **Step 1: Review the complete branch diff**

Run:

```bash
git diff --check feature/vericcl-implementation..HEAD
git diff --stat feature/vericcl-implementation..HEAD
git log --oneline feature/vericcl-implementation..HEAD
```

Expected: only the approved packaging, README, tests, design, plan, and
validation-report changes are present.

- [ ] **Step 2: Request an independent code review**

Use `superpowers:requesting-code-review` with the approved design, this plan,
the complete branch diff, and the validation report. Resolve every Critical or
Important finding and rerun its covering tests before publishing.

- [ ] **Step 3: Push and create a draft PR**

Push `fix/python313-dd-compatibility`, create a draft PR targeting
`feature/vericcl-implementation`, and include the root cause, compatibility
matrix, exact test results, and hardware limitations.

- [ ] **Step 4: Re-check the remote PR scope and mark ready**

Verify the PR base/head SHAs, changed-file list, and mergeability. Mark it ready
only after the branch and remote tree hashes match.

- [ ] **Step 5: Merge and update the local default branch**

Merge with a merge commit, delete the remote feature branch, fast-forward the
local `feature/vericcl-implementation`, and rerun the complete test suite on the
merged tree.

- [ ] **Step 6: Delete the local worktree and branch**

From `/Users/zdl/work/code/VeriCCL`, remove
`.worktrees/python313-dd-compat`, prune worktrees, and delete
`fix/python313-dd-compatibility` only after merged-tree verification passes.
