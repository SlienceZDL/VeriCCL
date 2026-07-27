# VeriCCL README Project Guide Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the English and Chinese VeriCCL READMEs into executable project guides that install VeriCCL, explain every run command before it appears, document inputs and outputs, and retain the verified MSCCL evaluation workflow.

**Architecture:** Treat the two READMEs as one bilingual interface: prose is localized, while commands, paths, versions, environment variables, and result literals remain identical. Extend the existing documentation tests before each content change, then rewrite the guide in two reviewable parts and finish with fresh software-only validation evidence.

**Tech Stack:** Markdown, Bash examples, Python 3, pytest, VeriCCL CLI, MSCCL, NCCL Tests.

## Global Constraints

- Do not mention SyCCL in either README.
- Do not add diagrams or other graphics.
- Do not include Ubuntu package-installation or operating-system preflight sections in either README.
- Link detailed server, CUDA, MPI, and runtime environment setup to `docs/runtime-configuration.md`.
- Retain concrete VeriCCL, MSCCL Strategy A/C, NCCL Tests, online validation, and single-node/multi-node XML commands.
- In `Running VeriCCL`, put each command in a separate Bash block and explain its purpose, input path, output path, and important parameters before the block.
- Keep the four `vericcl-doc-test` markers and keep their order: `help`, `solve`, `verify`, `example-validation`.
- Keep English and Chinese Bash blocks byte-identical and in the same order.
- Keep the topology/sketch/atom unknown-field contract accurate.
- Test documented behavior by executing commands and checking artifacts; do not add exact prose, heading, required-word, or forbidden-word assertions.
- Do not modify CLI behavior, input schemas, solver logic, runtime patches, or generated formats.
- State `License: To be determined.` and `Citation: To be determined.` exactly.
- CUDA/MSCCL compilation and GPU/nccl-tests execution remain `not_run` unless run on a target Ubuntu GPU server.

## File Structure

- Modify `README.md`: English project guide and executable workflow.
- Modify `README.zh-CN.md`: Chinese guide with the same structure and command literals.
- Modify `tests/integration/test_documented_commands.py`: executable run-step, artifact, path, bilingual command-parity, and input-behavior regression gates.
- Modify `docs/final-validation-report.md`: append fresh README validation evidence without changing prior results.

---

### Task 1: Install and staged run guide

**Files:**
- Modify: `README.md`
- Modify: `README.zh-CN.md`
- Test: `tests/integration/test_documented_commands.py`

**Interfaces:**
- Consumes: the confirmed design in `docs/superpowers/specs/2026-07-27-vericcl-readme-project-guide-design.md`, the existing four `vericcl-doc-test` markers, and packaged examples under `vericcl/examples`.
- Produces: ordered `Overview`, `Building and Installing VeriCCL`, and `Running VeriCCL` sections in both languages, plus a test helper that extracts the running section.

- [ ] **Step 1: Remove obsolete prose change-detector tests**

In `tests/integration/test_documented_commands.py`, delete `REQUIRED_README_FRAGMENTS` and `test_readmes_retain_the_installation_and_example_contract`. Keep the existing tests that execute marked commands, resolve real inputs, validate repository paths, validate the machine-readable unknown-field contract, verify workflow artifacts, and compare bilingual Bash blocks.

- [ ] **Step 2: Add a failing executable workflow test**

Add a machine-readable marker for the eight commands that form the runnable README workflow:

```python
RUN_STEP_PATTERN = re.compile(
    r"<!-- vericcl-run-step: ([a-z-]+) -->\s*"
    r"(?:<!-- vericcl-doc-test: [a-z-]+ -->\s*)?"
    r"```bash\s*\n([^\n]+)\n```"
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
```

Add helpers that extract the same marked commands from both READMEs and execute the English sequence in one Bash process. Override only the dynamic output-root assignment so the test writes under `tmp_path`:

```python
def _run_steps_from(path):
    return tuple(RUN_STEP_PATTERN.findall(path.read_text(encoding="utf-8")))


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
    run_root = (
        tmp_path / "runs" / "vericcl_allreduce_8MiB_quickstart"
    )
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
    verify_payload = json.loads(
        verify_summary.read_text(encoding="utf-8")
    )
    assert verify_payload["mode"] == "verify"
```

Import `os` and replace the existing `_run` implementation so documented commands use real shell expansion:

```python
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
```

- [ ] **Step 3: Run the executable workflow test and confirm it fails for the expected reason**

Run:

```bash
.venv/bin/python -m pytest tests/integration/test_documented_commands.py -q
```

Expected: failure because the current READMEs do not expose the eight `vericcl-run-step` commands.

- [ ] **Step 4: Rewrite the English overview, installation, and running sections**

In `README.md`, replace the content from `# VeriCCL` through the end of the current offline smoke-test section. Preserve the existing MSCCL Strategy A/C and NCCL Tests material below that boundary for Task 2.

- `## Overview`: project purpose, three JSON inputs, MSCCL XML/sidecar/report outputs, six directly solved operators, eight semantic operators, and the software-versus-hardware validation boundary.
- `## Building and Installing VeriCCL`: SSH clone, HTTPS fallback, venv creation, dependency installation, editable install, `pip check`, import check, version check, and the existing one-variable Gurobi license check. Keep the installation commands concrete.
- `## Running VeriCCL`: the eight ordered one-command Bash blocks from the confirmed design specification.

Place these markers immediately before their corresponding Bash blocks:

```text
vericcl-run-step: set-root
vericcl-run-step: set-output
vericcl-run-step: create-output
vericcl-run-step: solve
vericcl-run-step: verify
vericcl-run-step: check-xml
vericcl-run-step: check-report
vericcl-run-step: inspect-report
```

For `solve` and `verify`, place the existing `vericcl-doc-test` marker between the run-step marker and the Bash block so both extractors select the same executable command.

Before the solve block, explicitly describe:

```text
two_rank.json: two ranks and their directed links.
allreduce_8m_1m.json: an 8 MiB AllReduce split into 1 MiB software slices.
constructive.json: the constructive strategy with MILP disabled.
VERICCL_OUTPUT_DIR: the parent directory for this run.
quickstart: the stable run identifier.
```

Before the verify block, state that `--xml` points to the final XML from the solve run and that the verify output is written under `vericcl_allreduce_8MiB_quickstart-verify/`.

Before each `test -f` and `json.tool` block, name the exact file and explain whether it is the executable XML or the offline validation report. Then describe these two non-command artifacts:

```text
$VERICCL_OUTPUT_DIR/vericcl_allreduce_8MiB_quickstart/vericcl_allreduce_8MiB_final.schedule.json
$VERICCL_OUTPUT_DIR/vericcl_allreduce_8MiB_quickstart/run-summary.json
```

Keep the `help`, `solve`, and `verify` `vericcl-doc-test` markers immediately before the one-line command block they mark.

- [ ] **Step 5: Apply the same structure to the Chinese README**

Translate only prose and headings in `README.zh-CN.md`. Copy every Bash block from `README.md` byte-for-byte and preserve all file paths, versions, environment variables, option names, result literals, and `vericcl-doc-test` markers.

- [ ] **Step 6: Run the focused test and fix only contract mismatches**

Run:

```bash
.venv/bin/python -m pytest tests/integration/test_documented_commands.py -q
```

Expected: the Task 1 tests pass, including staged command order, marker order, input resolution, command execution, and bilingual Bash parity.

- [ ] **Step 7: Commit Task 1**

```bash
git add README.md README.zh-CN.md tests/integration/test_documented_commands.py
git commit -m "docs: add staged vericcl install and run guide"
```

---

### Task 2: Input, advanced use, runtime evaluation, and extension guide

**Files:**
- Modify: `README.md`
- Modify: `README.zh-CN.md`
- Test: `tests/integration/test_documented_commands.py`

**Interfaces:**
- Consumes: the top-level section headings and running-command contract from Task 1.
- Produces: the complete nine-section bilingual project guide and a tested MSCCL activation/extension contract.

- [ ] **Step 1: Complete the English guide**

Reorganize the remaining English material into these exact sections:

```markdown
## Input Configuration
## Advanced Usage
## MSCCL Runtime Evaluation
## Extending VeriCCL
## Outputs, Limitations, and Troubleshooting
## License and Citation
```

`Input Configuration` must retain the current topology/sketch/atom field semantics, unknown-field contract comment, executable example-path inspection, and the `legacy`/`templates` reference-only warning.

`Advanced Usage` must retain semantic overrides, `--override-input`, `--tune`, `--timeout-s`, automatic/manual hierarchy, six direct operators, eight semantic operators, and the distinction between `solve --online` and `verify --online`.

`MSCCL Runtime Evaluation` must retain:

- official commit `b23e9cd5dd63f82ee1c5aae7e0a2042079be903a`;
- fork tag `vericcl-runtime-v0.1.0` and commit `782ee5f72cf48c1ae1a2365bcf525019f5620175`;
- both verifier commands and build commands;
- NCCL Tests and clock-helper build commands;
- online environment variables;
- Simple protocol, `cnt=1`, `NCCL_BUFFSIZE=2*slice_size_bytes`, expected step constants `4/4`, and one-XML restriction;
- single-node and multi-node execution commands;
- `NCCL INFO Connected 1 MSCCL algorithms` as the positive XML-load signal;
- a warning that missing activation evidence or NCCL fallback is not VeriCCL schedule validation.

At the beginning of this section, link `docs/runtime-configuration.md` for server/CUDA/MPI setup instead of embedding Ubuntu package installation.

`Extending VeriCCL` must describe the responsibility of `vericcl/input`, `vericcl/topology`, `vericcl/planner`, `vericcl/solver`, `vericcl/composer`, `vericcl/xml`, `vericcl/verification`, `vericcl/tuning`, and `vericcl/verification/online`, and state that these internal modules are development entry points, not a stable plugin API.

`Outputs, Limitations, and Troubleshooting` must combine output geometry, exit codes `0/2/3/4/5`, Gurobi license limits, `offline-valid` versus `runtime-compatible`, MSCCL activation, MPI, trace, clock, and input diagnostics.

`License and Citation` must contain exactly:

```text
License: To be determined.
Citation: To be determined.
```

- [ ] **Step 2: Complete the Chinese guide with byte-identical commands**

Mirror the English section order and technical claims in `README.zh-CN.md`. Translate prose only. Ensure all Bash blocks, code literals, external references, commits, tags, expected log text, and paths match `README.md`.

- [ ] **Step 3: Run the executable documentation and artifact tests**

Run:

```bash
.venv/bin/python -m pytest tests/integration/test_documented_commands.py tests/integration/test_workflow_artifacts.py -q
```

Expected: all selected tests pass; the eight-step README workflow and the existing help/example commands execute, repository paths resolve, workflow artifacts are valid, and Bash blocks are byte-identical.

- [ ] **Step 4: Perform the prose and scope self-review**

Read both READMEs in full and record the checklist result in the task report:

- no SyCCL references;
- no diagrams;
- no Ubuntu package-installation or operating-system preflight section;
- each `Running VeriCCL` command has a preceding purpose/file/parameter explanation;
- all nine sections appear in the confirmed order;
- MSCCL activation and fallback boundaries are explicit;
- extension modules are described as development entry points, not a stable plugin API;
- License/Citation status is present;
- no CUDA compilation or GPU execution claim is made.

This is a review checklist, not an automated exact-text test.

- [ ] **Step 5: Commit Task 2**

```bash
git add README.md README.zh-CN.md tests/integration/test_documented_commands.py
git commit -m "docs: complete vericcl project guide"
```

---

### Task 3: Final validation and evidence

**Files:**
- Modify: `docs/final-validation-report.md`

**Interfaces:**
- Consumes: the complete README contract and existing validation report.
- Produces: fresh software evidence with explicit CUDA/GPU `not_run` status.

- [ ] **Step 1: Run the complete documentation gate**

Run:

```bash
.venv/bin/python -m pytest tests/integration/test_documented_commands.py tests/integration/test_workflow_artifacts.py tests/unit/online/test_runtime_patch.py -q
```

Expected: zero failures.

- [ ] **Step 2: Run the full hardware-independent coverage gate**

Run:

```bash
.venv/bin/python -m pytest -m 'not hardware and not gurobi' --cov=vericcl --cov-report=term-missing --cov-fail-under=90 -q
```

Expected: zero failures and at least 90% total coverage.

- [ ] **Step 3: Run Gurobi and hardware markers separately**

Run:

```bash
.venv/bin/python -m pytest -m gurobi -q
```

Expected: zero failures; record the exact pass/deselection counts.

Run:

```bash
.venv/bin/python -m pytest -m hardware -q
```

Expected on the current macOS host: hardware tests skip. Record the exact skip/deselection counts as `not_run`, not as hardware validation.

- [ ] **Step 4: Run static checks**

Run:

```bash
python3 -m compileall -q vericcl tests runtime/msccl-trace/tools
```

Run:

```bash
git diff --check
```

Run:

```bash
rg -n '[\p{Han}]' vericcl tests runtime --glob '*.py' --glob '*.c' --glob '*.cc' --glob '*.cu' --glob '*.h' --glob '*.json'
```

Expected: compile and diff checks exit zero; the Han scan returns no matches.

- [ ] **Step 5: Append exact README validation evidence**

Append a dated `2026-07-27` subsection to `docs/final-validation-report.md` with:

- exact commands and observed pass/skip/deselection/coverage counts;
- confirmation that both READMEs contain the same Bash blocks and the staged commands execute;
- confirmation that excluded Ubuntu/SyCCL content is absent;
- confirmation that repository-relative paths exist and are tracked;
- CUDA/MSCCL compilation and nccl-tests GPU execution marked `not_run`.

Do not replace or reinterpret earlier validation evidence.

- [ ] **Step 6: Commit Task 3**

```bash
git add docs/final-validation-report.md
git commit -m "docs: record project guide validation"
```
