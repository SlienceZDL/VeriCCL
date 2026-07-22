# VeriCCL Installation Readmes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver equivalent English and Chinese end-to-end installation and usage guides, plus two reproducible VeriCCL-compatible MSCCL installation paths for Ubuntu 22.04/24.04.

**Architecture:** Pin the official MSCCL source revision in machine-readable metadata, rebase the fixed-buffer trace patch onto that clean revision, and publish the identical patched source as `SlienceZDL/VeriCCL-MSCCL`. Keep `README.md` and `README.zh-CN.md` command-identical, and validate all hardware-independent commands automatically.

**Tech Stack:** Python 3.10-3.12, setuptools, pytest, Git, GitHub, Ubuntu 22.04/24.04, CUDA, MSCCL, NCCL Tests, Open MPI, Gurobi.

## Global Constraints

- `README.md` is English; `README.zh-CN.md` is Chinese; each links to the other at the top.
- Both README files must contain identical shell commands, version pins, paths, environment variables, and expected results.
- Pin official MSCCL commit `b23e9cd5dd63f82ee1c5aae7e0a2042079be903a`.
- Publish the pre-integrated source at `SlienceZDL/VeriCCL-MSCCL` with tag `vericcl-runtime-v0.1.0`.
- Strategy A and Strategy C must have identical runtime-relevant source file hashes.
- Preserve `MSCCL_CHUNKSTEPS=4`, `MSCCL_SLICESTEPS=4`, `NCCL_PROTO=Simple`, `cnt=1`, and `NCCL_BUFFSIZE=2*slice_size_bytes`.
- Do not modify solver, collective semantics, XML lowering, or offline validation behavior.
- Do not add Chinese characters to Python, C/CUDA, JSON metadata, tests, patches, or generated XML/JSON.
- Do not claim CUDA compilation or hardware execution unless those commands are run on an Ubuntu GPU server.

---

### Task 1: Pin MSCCL provenance and strengthen the patch verifier

**Files:**
- Create: `runtime/msccl-trace/upstream.json`
- Modify: `runtime/msccl-trace/tools/verify_patch.py`
- Modify: `setup.py`
- Modify: `tests/unit/online/test_runtime_patch.py`

**Interfaces:**
- Consumes: official MSCCL Git checkout and the bundled trace patch/header.
- Produces: `_load_metadata(path)`, pinned source validation, base-tree patch
  validation, and already-patched tree validation.

- [ ] **Step 1: Add failing metadata and revision tests**

Add tests that require the metadata file, remove the hard-coded macOS reference path, and verify rejection of a wrong Git revision:

```python
METADATA_FILE = RUNTIME_ROOT / "upstream.json"
REFERENCE_ROOT = os.environ.get("VERICCL_MSCCL_REFERENCE_ROOT")

def test_runtime_metadata_pins_reproducible_sources():
    metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    assert metadata["schema_version"] == 1
    assert metadata["upstream_repository"] == "https://github.com/microsoft/msccl.git"
    assert metadata["upstream_commit"] == "b23e9cd5dd63f82ee1c5aae7e0a2042079be903a"
    assert metadata["fork_repository"] == "https://github.com/SlienceZDL/VeriCCL-MSCCL.git"
    assert metadata["fork_tag"] == "vericcl-runtime-v0.1.0"

def test_verifier_rejects_wrong_git_revision(tmp_path):
    subprocess.run(("git", "init", str(tmp_path)), check=True)
    subprocess.run(
        ("git", "-C", str(tmp_path), "config", "user.name", "VeriCCL Test"),
        check=True,
    )
    subprocess.run(
        ("git", "-C", str(tmp_path), "config", "user.email", "vericcl@example.invalid"),
        check=True,
    )
    subprocess.run(
        ("git", "-C", str(tmp_path), "commit", "--allow-empty", "-m", "wrong revision"),
        check=True,
    )
    completed = subprocess.run(
        (sys.executable, str(VERIFY_PATCH), "--source-root", str(tmp_path)),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1
    assert "pinned MSCCL revision" in completed.stderr
```

- [ ] **Step 2: Run tests and confirm the new contract fails**

Run:

```bash
.venv/bin/python -m pytest tests/unit/online/test_runtime_patch.py -q
```

Expected: failure because `upstream.json` and pinned-revision validation do not exist.

- [ ] **Step 3: Add exact upstream metadata**

Create:

```json
{
  "schema_version": 1,
  "upstream_repository": "https://github.com/microsoft/msccl.git",
  "upstream_commit": "b23e9cd5dd63f82ee1c5aae7e0a2042079be903a",
  "fork_repository": "https://github.com/SlienceZDL/VeriCCL-MSCCL.git",
  "fork_tag": "vericcl-runtime-v0.1.0"
}
```

`patched_commit` and `patched_files` are added in Task 3 after the final fork
commit exists; no README may reference the fork tag before then.

- [ ] **Step 4: Implement metadata, Git revision, and file hash validation**

Add these interfaces to `verify_patch.py`:

```python
METADATA_FILE = RUNTIME_ROOT / "upstream.json"

def _load_metadata(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise ValueError("unsupported MSCCL metadata schema")
    return value

def _git_head(source_root: Path) -> str:
    completed = subprocess.run(
        ("git", "-C", str(source_root), "rev-parse", "HEAD"),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ValueError("source root must be a Git checkout")
    return completed.stdout.strip()

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
```

Extend the CLI with mutually exclusive `--base-tree` and `--patched-tree` modes. Base-tree mode is the default, requires the pinned upstream commit, applies the patch in a temporary copy, and validates final hashes when present. Patched-tree mode skips patch application and requires every metadata hash to match.

- [ ] **Step 5: Package the metadata file**

Add `runtime/msccl-trace/upstream.json` to the existing `share/vericcl/runtime/msccl-trace` `data_files` entry in `setup.py`.

- [ ] **Step 6: Run focused tests**

Run:

```bash
.venv/bin/python -m pytest tests/unit/online/test_runtime_patch.py -q
```

Expected: layout/metadata tests pass; the real upstream patch test skips unless `VERICCL_MSCCL_REFERENCE_ROOT` is set.

- [ ] **Step 7: Commit Task 1**

```bash
git add runtime/msccl-trace/upstream.json runtime/msccl-trace/tools/verify_patch.py setup.py tests/unit/online/test_runtime_patch.py
git commit -m "build: pin vericcl msccl provenance"
```

---

### Task 2: Rebase the trace patch onto clean official MSCCL

**Files:**
- Modify: `runtime/msccl-trace/patches/0001-vericcl-fixed-step-trace.patch`
- Modify: `runtime/msccl-trace/README.md`
- Modify: `tests/unit/online/test_runtime_patch.py`

**Interfaces:**
- Consumes: pinned official commit and `vericcl_trace_format.h`.
- Produces: a patch that applies to the clean official checkout and records complete fixed-buffer traces without device `printf`.

- [ ] **Step 1: Prepare a clean pinned source checkout**

```bash
git clone https://github.com/microsoft/msccl.git /tmp/vericcl-msccl-base
git -C /tmp/vericcl-msccl-base checkout --detach b23e9cd5dd63f82ee1c5aae7e0a2042079be903a
git -C /tmp/vericcl-msccl-base status --short
```

Expected: detached pinned commit and empty status.

- [ ] **Step 2: Demonstrate that the old patch fails on clean upstream**

```bash
python3 runtime/msccl-trace/tools/verify_patch.py --source-root /tmp/vericcl-msccl-base
```

Expected before rebasing: nonzero exit with patch hunk failures.

- [ ] **Step 3: Recreate the fixed-buffer implementation on a working branch**

Create branch `vericcl-runtime` in a second checkout and implement the existing trace contract directly against clean upstream:

```bash
git -C /tmp/vericcl-msccl-base switch -c vericcl-runtime
cp runtime/msccl-trace/include/vericcl_trace_format.h /tmp/vericcl-msccl-base/src/include/vericcl_trace_format.h
```

The implementation must make these exact source changes:

- `src/include/msccl.h`: set both step constants to `4`; add record pointers,
  count, overflow flag, and capacity to `mscclDevCommInfo`.
- `src/collectives/device/primitives.h`: include the trace format header and
  define device clock/resource-mask helpers without `printf`.
- `src/collectives/device/msccl_interpreter.h`: reserve one record per XML
  step, capture `tb_reach`, `dependency_done`, `transfer_start`, and
  `transfer_end`, store `workIndex` in `iteration`, and make tracing a no-op
  when disabled.
- `src/init.cc`: parse `VERICCL_TRACE_ENABLE`, `VERICCL_TRACE_RECORDS`,
  `VERICCL_TRACE_FILE_PREFIX`, and expected step constants; allocate/copy/free
  device buffers; write one rank file during communicator teardown; reject
  incompatible step constants during communicator initialization.

Use `apply_patch` for each source edit. Do not carry over the earlier
`MSCCLTRACE|` or aggregate timing `printf` instrumentation.

- [ ] **Step 4: Generate the clean upstream patch**

Exclude the separately copied header and export only the four modified upstream files:

```bash
git -C /tmp/vericcl-msccl-base diff -- src/collectives/device/msccl_interpreter.h src/collectives/device/primitives.h src/include/msccl.h src/init.cc
```

Replace `runtime/msccl-trace/patches/0001-vericcl-fixed-step-trace.patch` with that exact diff through `apply_patch`.

- [ ] **Step 5: Verify clean application and post-apply invariants**

```bash
VERICCL_MSCCL_REFERENCE_ROOT=/tmp/vericcl-msccl-base .venv/bin/python -m pytest tests/unit/online/test_runtime_patch.py -q
python3 runtime/msccl-trace/tools/verify_patch.py --source-root /tmp/vericcl-msccl-base
```

Expected: all focused tests pass and verifier prints `verification passed`.

- [ ] **Step 6: Update the runtime patch guide**

Replace the private `/Users/zdl/work/code/MSCCL_TIME` reference with the official repository, pinned commit, both installation strategies, exact `verify_patch.py` command, and the statement that local patch verification is not CUDA compilation evidence.

- [ ] **Step 7: Commit Task 2**

```bash
git add runtime/msccl-trace/README.md runtime/msccl-trace/patches/0001-vericcl-fixed-step-trace.patch tests/unit/online/test_runtime_patch.py
git commit -m "build: rebase trace patch onto pinned msccl"
```

---

### Task 3: Publish and bind the pre-integrated VeriCCL-MSCCL fork

**Files:**
- Modify: `runtime/msccl-trace/upstream.json`
- Modify: `tests/unit/online/test_runtime_patch.py`
- External: public GitHub repository `SlienceZDL/VeriCCL-MSCCL`

**Interfaces:**
- Consumes: the verified patched MSCCL working tree from Task 2.
- Produces: readable SSH/HTTPS repository, branch `vericcl-runtime`, tag `vericcl-runtime-v0.1.0`, and immutable commit/hash binding in VeriCCL.

- [ ] **Step 1: Create the GitHub fork**

Using the signed-in GitHub browser, open `https://github.com/microsoft/msccl/fork`, select owner `SlienceZDL`, set repository name `VeriCCL-MSCCL`, retain public visibility, and create the fork. Do not create a second repository if the name becomes visible during the operation.

- [ ] **Step 2: Configure remotes without overwriting upstream**

```bash
git -C /tmp/vericcl-msccl-base remote rename origin upstream
git -C /tmp/vericcl-msccl-base remote add origin git@github.com:SlienceZDL/VeriCCL-MSCCL.git
git -C /tmp/vericcl-msccl-base remote --verbose
```

Expected: `origin` is the user fork and `upstream` is `microsoft/msccl`.

- [ ] **Step 3: Add fork provenance and commit the patched runtime**

Add an English `VERICCL_RUNTIME.md` to the fork with the upstream commit, the
VeriCCL patch source path, required step constants, build command, trace/release
separation, and upstream synchronization policy. Commit all patched files and
the trace header:

```bash
git -C /tmp/vericcl-msccl-base add src/include/vericcl_trace_format.h src/include/msccl.h src/collectives/device/primitives.h src/collectives/device/msccl_interpreter.h src/init.cc VERICCL_RUNTIME.md
git -C /tmp/vericcl-msccl-base commit -m "feat: integrate vericcl fixed-step tracing"
```

- [ ] **Step 4: Push the branch and immutable tag**

```bash
git -C /tmp/vericcl-msccl-base push -u origin vericcl-runtime
git -C /tmp/vericcl-msccl-base tag -a vericcl-runtime-v0.1.0 -m "VeriCCL MSCCL runtime v0.1.0"
git -C /tmp/vericcl-msccl-base push origin vericcl-runtime-v0.1.0
```

Set `vericcl-runtime` as the fork's default branch in GitHub repository settings.

- [ ] **Step 5: Record final commit and source hashes**

Populate `patched_commit` with:

```bash
git -C /tmp/vericcl-msccl-base rev-parse vericcl-runtime-v0.1.0^{}
```

Populate `patched_files` with SHA-256 values for the trace header and four
runtime files. Add a test requiring a 40-character lowercase
`patched_commit`, exactly those file keys, and 64-character lowercase hashes.

- [ ] **Step 6: Verify Strategy A and Strategy C equivalence**

Clone the tag into `/tmp/vericcl-msccl-fork` and run patched-tree verification:

```bash
git clone --branch vericcl-runtime-v0.1.0 --depth 1 https://github.com/SlienceZDL/VeriCCL-MSCCL.git /tmp/vericcl-msccl-fork
python3 runtime/msccl-trace/tools/verify_patch.py --source-root /tmp/vericcl-msccl-fork --patched-tree
```

Expected: `verification passed`. Also confirm SSH visibility with
`git ls-remote git@github.com:SlienceZDL/VeriCCL-MSCCL.git vericcl-runtime-v0.1.0`.

- [ ] **Step 7: Commit Task 3 metadata binding**

```bash
git add runtime/msccl-trace/upstream.json tests/unit/online/test_runtime_patch.py
git commit -m "build: bind preintegrated vericcl msccl runtime"
```

---

### Task 4: Write command-equivalent English and Chinese installation guides

**Files:**
- Modify: `README.md`
- Create: `README.zh-CN.md`
- Modify: `docs/runtime-configuration.md`

**Interfaces:**
- Consumes: final metadata, fork tag, actual CLI, examples, and runtime environment contract.
- Produces: complete English and Chinese server workflows with identical command blocks.

- [ ] **Step 1: Add bilingual command-equivalence tests first**

Extend `tests/integration/test_documented_commands.py` with separate English and Chinese marker extraction:

```python
README_EN = PROJECT_ROOT / "README.md"
README_ZH = PROJECT_ROOT / "README.zh-CN.md"

def _commands_from(path):
    return dict(COMMAND_PATTERN.findall(path.read_text(encoding="utf-8")))

def test_bilingual_readmes_have_identical_tested_commands():
    assert _commands_from(README_EN) == _commands_from(README_ZH)
```

Run the test and expect failure because `README.zh-CN.md` does not exist.

- [ ] **Step 2: Replace the English README and add the Chinese README**

Use identical headings and command blocks in both files. The required sections are:

```text
Language link
Overview and supported collectives
Installation modes
Ubuntu prerequisites
CUDA/NCCL/MPI preflight
Clone and Python install
Gurobi license
Offline smoke test
MSCCL Strategy A
MSCCL Strategy C
NCCL Tests and clock helper build
Input schemas and examples
Solve and verify
Overrides, hierarchy, tuning, and timeout
Online validation
Single-node XML execution
Multi-node XML execution
Outputs, exit codes, troubleshooting, tests, references
```

The installation commands must include these exact core sequences in both files:

```bash
sudo apt update
sudo apt install -y build-essential git patch python3 python3-dev python3-pip python3-venv openmpi-bin libopenmpi-dev wget ca-certificates
git clone git@github.com:SlienceZDL/VeriCCL.git
cd VeriCCL
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pip install -e .
```

Document HTTPS clone as the alternative to SSH. Refer to NVIDIA's official
Ubuntu package-manager guide for CUDA installation rather than embedding an
unverified latest CUDA version.

- [ ] **Step 3: Document Gurobi licensing precisely**

Include `gurobipy` import and one-variable model checks, default license file locations, WLS/local/network license choices, and a warning that credentials must not be committed. Distinguish the size-limited bundled license from a license suitable for full VeriCCL MILP models.

- [ ] **Step 4: Document both MSCCL strategies with exact pins**

Strategy A must clone official MSCCL, detach at the pinned commit, copy the
header, apply the patch, verify, and build. Strategy C must clone the tagged
fork and run patched-tree verification before the same build command. Both use
`CUDA_HOME` and produce `<MSCCL_ROOT>/build/lib`.

- [ ] **Step 5: Document NCCL Tests and clock helper builds**

Use the official MPI build form:

```bash
make -C "$NCCL_TESTS_ROOT" -j MPI=1 MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi CUDA_HOME="$CUDA_HOME" NCCL_HOME="$MSCCL_ROOT/build"
nvcc -ccbin mpicxx -O2 -std=c++11 "$VERICCL_ROOT/runtime/msccl-trace/tools/vericcl_clock_sync.cu" -o "$VERICCL_ROOT/runtime/msccl-trace/tools/vericcl_clock_sync"
```

Verify the actual Ubuntu Open MPI prefix during implementation; if the package layout does not expose that prefix consistently, document `MPI_HOME` discovery using `dirname "$(dirname "$(readlink -f "$(command -v mpicxx)")")"`.

- [ ] **Step 6: Document all three input contracts and example selection**

Explain exact units and constraints for topology links/resources,
sketch collective/hyperparameter/solver fields, atom strategy/manual hierarchy,
and forbidden transfer tuples. Mark `vericcl/examples/legacy` and templates as
reference-only. Include the two-rank constructive example and the two-node
gateway hierarchy example.

- [ ] **Step 7: Document solve, verify, overrides, online mode, and XML execution**

Preserve the existing four `vericcl-doc-test` commands exactly in both files.
Add `--override-input`, `--tune`, `--timeout-s`, `--online`, single-node
`all_reduce_perf -g 2`, and multi-node `mpirun -np 8 -N 4 ... -g 1` examples.
For every XML execution set:

```bash
export NCCL_ALGO=MSCCL
export NCCL_PROTO=Simple
export NCCL_BUFFSIZE=2097152
export MSCCL_XML_FILES=/absolute/path/to/one-schedule.xml
export VERICCL_EXPECTED_MSCCL_CHUNKSTEPS=4
export VERICCL_EXPECTED_MSCCL_SLICESTEPS=4
export VERICCL_TRACE_ENABLE=0
```

- [ ] **Step 8: Synchronize runtime configuration documentation**

Update `docs/runtime-configuration.md` to link both READMEs and both MSCCL
installation strategies. Remove any instruction that assumes a private local
MSCCL tree.

- [ ] **Step 9: Run bilingual and documented-command tests**

```bash
.venv/bin/python -m pytest tests/integration/test_documented_commands.py -q
```

Expected: all tests pass and English/Chinese command maps are identical.

- [ ] **Step 10: Commit Task 4**

```bash
git add README.md README.zh-CN.md docs/runtime-configuration.md tests/integration/test_documented_commands.py
git commit -m "docs: add bilingual installation and usage guides"
```

---

### Task 5: Add documentation consistency and example coverage

**Files:**
- Modify: `tests/integration/test_documented_commands.py`
- Modify: `tests/integration/test_workflow_artifacts.py`

**Interfaces:**
- Consumes: both README files and repository example inputs.
- Produces: regression gates for command parity, required sections, real paths, metadata references, and output geometry.

- [ ] **Step 1: Add failing structural assertions**

Assert that both READMEs contain:

```python
required_fragments = {
    "Ubuntu 22.04",
    "Ubuntu 24.04",
    "b23e9cd5dd63f82ee1c5aae7e0a2042079be903a",
    "vericcl-runtime-v0.1.0",
    "NCCL_BUFFSIZE=2097152",
    "VERICCL_CALIBRATION_LINK_CLASS",
    "vericcl/examples/topo/two_rank.json",
    "vericcl/examples/topo/two_node_gateway.json",
    "vericcl/examples/atom/constructive.json",
    "vericcl/examples/atom/default.json",
}
```

Also extract all repository-relative paths in backticks under the examples
section and assert they exist.

- [ ] **Step 2: Run the focused tests and fix only genuine documentation mismatches**

```bash
.venv/bin/python -m pytest tests/integration/test_documented_commands.py tests/integration/test_workflow_artifacts.py -q
```

Expected: pass after correcting missing or inconsistent README content.

- [ ] **Step 3: Commit Task 5**

```bash
git add README.md README.zh-CN.md tests/integration/test_documented_commands.py tests/integration/test_workflow_artifacts.py
git commit -m "test: validate installation documentation"
```

---

### Task 6: Final verification and evidence report

**Files:**
- Modify if needed: `docs/final-validation-report.md`

**Interfaces:**
- Consumes: completed local repository and published fork.
- Produces: fresh software validation evidence and an explicit hardware `not_run` status unless an Ubuntu GPU server is available.

- [ ] **Step 1: Verify patch and fork provenance**

```bash
python3 runtime/msccl-trace/tools/verify_patch.py --source-root /tmp/vericcl-msccl-base
python3 runtime/msccl-trace/tools/verify_patch.py --source-root /tmp/vericcl-msccl-fork --patched-tree
git ls-remote https://github.com/SlienceZDL/VeriCCL-MSCCL.git vericcl-runtime vericcl-runtime-v0.1.0
git ls-remote git@github.com:SlienceZDL/VeriCCL-MSCCL.git vericcl-runtime vericcl-runtime-v0.1.0
```

Expected: both verifier runs pass and both protocols return the branch/tag refs.

- [ ] **Step 2: Run focused documentation and runtime tests**

```bash
.venv/bin/python -m pytest tests/unit/online/test_runtime_patch.py tests/integration/test_documented_commands.py tests/integration/test_workflow_artifacts.py -q
```

Expected: all selected tests pass.

- [ ] **Step 3: Run the full hardware-independent gate**

```bash
.venv/bin/python -m pytest -m 'not hardware and not gurobi' --cov=vericcl --cov-report=term-missing --cov-fail-under=90 -q
```

Expected: zero failures and total coverage at least 90%.

- [ ] **Step 4: Run Gurobi and hardware gates separately**

```bash
.venv/bin/python -m pytest -m gurobi -q
.venv/bin/python -m pytest -m hardware -q
```

Record exact pass/skip counts. Hardware skips remain `not_run`; do not convert
them into validation success.

- [ ] **Step 5: Run static repository checks**

```bash
python3 -m compileall -q vericcl tests runtime/msccl-trace/tools
git diff --check
rg -n '[\p{Han}]' vericcl tests runtime --glob '*.py' --glob '*.c' --glob '*.cc' --glob '*.cu' --glob '*.h' --glob '*.json'
```

Expected: compile and diff checks exit zero; Han scan produces no matches.

- [ ] **Step 6: Update evidence without overstating hardware status**

If the test counts or patch/fork evidence are materially new, append an
installation-documentation section to `docs/final-validation-report.md` with
the exact commands and results. State CUDA/MSCCL compilation and nccl-tests
execution as `not_run` unless executed on the target Ubuntu GPU server.

- [ ] **Step 7: Commit final evidence changes**

```bash
git add docs/final-validation-report.md
git commit -m "docs: record installation guide validation"
```

Skip this commit if the report does not require a change.
