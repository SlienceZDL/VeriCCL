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
- Do not modify CLI behavior, input schemas, solver logic, runtime patches, or generated formats.
- State `License: To be determined.` and `Citation: To be determined.` exactly.
- CUDA/MSCCL compilation and GPU/nccl-tests execution remain `not_run` unless run on a target Ubuntu GPU server.

## File Structure

- Modify `README.md`: English project guide and executable workflow.
- Modify `README.zh-CN.md`: Chinese guide with the same structure and command literals.
- Modify `tests/integration/test_documented_commands.py`: README structure, absence, step order, path, bilingual parity, and executable-command regression gates.
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

- [ ] **Step 1: Replace the obsolete installation-fragment contract with the new README contract**

In `tests/integration/test_documented_commands.py`, replace the Ubuntu fragments in `REQUIRED_README_FRAGMENTS`, add localized ordered headings, and add forbidden fragments:

```python
REQUIRED_README_FRAGMENTS = {
    "b23e9cd5dd63f82ee1c5aae7e0a2042079be903a",
    "vericcl-runtime-v0.1.0",
    "782ee5f72cf48c1ae1a2365bcf525019f5620175",
    "NCCL_BUFFSIZE=2097152",
    "VERICCL_CALIBRATION_LINK_CLASS",
    "vericcl/examples/topo/two_rank.json",
    "vericcl/examples/topo/two_node_gateway.json",
    "vericcl/examples/sketch/allreduce_8m_1m.json",
    "vericcl/examples/atom/constructive.json",
    "vericcl/examples/atom/default.json",
    "docs/runtime-configuration.md",
}
README_PRIMARY_HEADINGS = {
    README_EN: (
        "## Overview",
        "## Building and Installing VeriCCL",
        "## Running VeriCCL",
    ),
    README_ZH: (
        "## 概述",
        "## 构建与安装VeriCCL",
        "## 运行VeriCCL",
    ),
}
FORBIDDEN_README_FRAGMENTS = {
    "SyCCL",
    "```mermaid",
    "sudo apt update",
    "sudo apt install",
    "## Ubuntu prerequisites",
    "## CUDA, NCCL, and MPI preflight",
    "## Ubuntu前置依赖",
    "## CUDA、NCCL与MPI预检",
}
INSTALL_COMMAND_FRAGMENTS = (
    "git clone git@github.com:SlienceZDL/VeriCCL.git",
    "git clone https://github.com/SlienceZDL/VeriCCL.git",
    "python3 -m venv .venv",
    ".venv/bin/python -m pip install --upgrade pip setuptools wheel",
    ".venv/bin/python -m pip install -r requirements-dev.txt",
    ".venv/bin/python -m pip install -e .",
    ".venv/bin/python -m pip check",
    ".venv/bin/python -m vericcl --version",
)
```

Add the ordered-heading and forbidden-content tests:

```python
@pytest.mark.parametrize("path", (README_EN, README_ZH))
def test_readmes_have_the_ordered_project_guide_sections(path):
    text = path.read_text(encoding="utf-8")
    positions = [text.index(heading) for heading in README_PRIMARY_HEADINGS[path]]
    assert positions == sorted(positions)


@pytest.mark.parametrize("path", (README_EN, README_ZH))
def test_readmes_exclude_out_of_scope_content(path):
    text = path.read_text(encoding="utf-8")
    assert not {
        fragment for fragment in FORBIDDEN_README_FRAGMENTS
        if fragment in text
    }


@pytest.mark.parametrize("path", (README_EN, README_ZH))
def test_readmes_contain_the_concrete_install_commands(path):
    text = path.read_text(encoding="utf-8")
    assert not {
        command for command in INSTALL_COMMAND_FRAGMENTS
        if command not in text
    }
```

- [ ] **Step 2: Add a failing contract for individually explained run commands**

Add exact one-command Bash blocks and a section extractor:

```python
RUNNING_STEP_COMMANDS = (
    'export VERICCL_ROOT="$(pwd)"',
    'export VERICCL_OUTPUT_DIR="$VERICCL_ROOT/runs/readme-$(date +%Y%m%dT%H%M%S)"',
    'mkdir -p "$VERICCL_OUTPUT_DIR"',
    ".venv/bin/python -m vericcl solve --topology "
    "vericcl/examples/topo/two_rank.json --sketch "
    "vericcl/examples/sketch/allreduce_8m_1m.json --atoms "
    "vericcl/examples/atom/constructive.json --output-dir "
    '"$VERICCL_OUTPUT_DIR" --run-id quickstart',
    ".venv/bin/python -m vericcl verify --topology "
    "vericcl/examples/topo/two_rank.json --sketch "
    "vericcl/examples/sketch/allreduce_8m_1m.json --atoms "
    "vericcl/examples/atom/constructive.json --output-dir "
    '"$VERICCL_OUTPUT_DIR" --run-id quickstart-verify --xml '
    '"$VERICCL_OUTPUT_DIR/vericcl_allreduce_8MiB_quickstart/'
    'vericcl_allreduce_8MiB_final.xml"',
    'test -f "$VERICCL_OUTPUT_DIR/vericcl_allreduce_8MiB_quickstart/'
    'vericcl_allreduce_8MiB_final.xml"',
    'test -f "$VERICCL_OUTPUT_DIR/vericcl_allreduce_8MiB_quickstart/'
    'vericcl_allreduce_8MiB_final.validation.json"',
    ".venv/bin/python -m json.tool "
    '"$VERICCL_OUTPUT_DIR/vericcl_allreduce_8MiB_quickstart/'
    'vericcl_allreduce_8MiB_final.validation.json"',
)


def _heading_section(text, heading):
    start = text.index(heading)
    end = text.find("\n## ", start + len(heading))
    return text[start:] if end == -1 else text[start:end]


@pytest.mark.parametrize(
    ("path", "heading"),
    (
        (README_EN, "## Running VeriCCL"),
        (README_ZH, "## 运行VeriCCL"),
    ),
)
def test_running_commands_are_individual_ordered_blocks(path, heading):
    section = _heading_section(path.read_text(encoding="utf-8"), heading)
    blocks = BASH_BLOCK_PATTERN.findall(section)
    positions = []
    for command in RUNNING_STEP_COMMANDS:
        assert command in blocks
        assert all(
            line == command
            for block in blocks if command in block
            for line in block.splitlines()
        )
        positions.append(section.index(command))
    assert positions == sorted(positions)
```

Add prose anchors that force the three input paths and output artifacts to be explained before the solve/verify/check commands:

```python
RUNNING_EXPLANATION_FRAGMENTS = {
    "vericcl/examples/topo/two_rank.json",
    "vericcl/examples/sketch/allreduce_8m_1m.json",
    "vericcl/examples/atom/constructive.json",
    "vericcl_allreduce_8MiB_final.xml",
    "vericcl_allreduce_8MiB_final.schedule.json",
    "vericcl_allreduce_8MiB_final.validation.json",
    "run-summary.json",
    "vericcl_allreduce_8MiB_quickstart-verify",
}
```

Extend the parametrized test to assert each fragment occurs in the running section and that the first prose occurrence of each solve input path precedes the solve command.

- [ ] **Step 3: Run the new tests and confirm the current README fails**

Run:

```bash
.venv/bin/python -m pytest tests/integration/test_documented_commands.py -q
```

Expected: failures for missing primary project-guide headings, forbidden Ubuntu sections, and missing staged quickstart commands.

- [ ] **Step 4: Rewrite the English overview, installation, and running sections**

In `README.md`, replace the content from `# VeriCCL` through the end of the current offline smoke-test section. Preserve the existing MSCCL Strategy A/C and NCCL Tests material below that boundary for Task 2.

- `## Overview`: project purpose, three JSON inputs, MSCCL XML/sidecar/report outputs, six directly solved operators, eight semantic operators, and the software-versus-hardware validation boundary.
- `## Building and Installing VeriCCL`: SSH clone, HTTPS fallback, venv creation, dependency installation, editable install, `pip check`, import check, version check, and the existing one-variable Gurobi license check. Keep the installation commands concrete.
- `## Running VeriCCL`: eight ordered one-command Bash blocks matching `RUNNING_STEP_COMMANDS`.

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

- [ ] **Step 1: Add failing assertions for runtime and extension content**

Define the extension module paths and add them, the MSCCL activation line, and the License/Citation status to `REQUIRED_README_FRAGMENTS`:

```python
EXTENSION_MODULE_PATHS = (
    "vericcl/input",
    "vericcl/topology",
    "vericcl/planner",
    "vericcl/solver",
    "vericcl/composer",
    "vericcl/xml",
    "vericcl/verification",
    "vericcl/tuning",
    "vericcl/verification/online",
)
REQUIRED_README_FRAGMENTS.update({
    "NCCL INFO Connected 1 MSCCL algorithms",
    "License: To be determined.",
    "Citation: To be determined.",
})
REQUIRED_README_FRAGMENTS.update(EXTENSION_MODULE_PATHS)
```

Add the remaining localized headings and the localized extension-boundary phrase:

```python
README_REMAINING_HEADINGS = {
    README_EN: (
        "## Input Configuration",
        "## Advanced Usage",
        "## MSCCL Runtime Evaluation",
        "## Extending VeriCCL",
        "## Outputs, Limitations, and Troubleshooting",
        "## License and Citation",
    ),
    README_ZH: (
        "## 输入配置",
        "## 高级用法",
        "## MSCCL运行时评测",
        "## 扩展VeriCCL",
        "## 输出、限制与故障诊断",
        "## 许可证与引用",
    ),
}
EXTENSION_BOUNDARY = {
    README_EN: "not a stable plugin API",
    README_ZH: "不是稳定的插件API",
}
```

Add:

```python
@pytest.mark.parametrize("path", (README_EN, README_ZH))
def test_readmes_define_the_extension_boundary(path):
    assert EXTENSION_BOUNDARY[path] in path.read_text(encoding="utf-8")


@pytest.mark.parametrize("path", (README_EN, README_ZH))
def test_readmes_have_the_complete_ordered_project_guide(path):
    text = path.read_text(encoding="utf-8")
    headings = README_PRIMARY_HEADINGS[path] + README_REMAINING_HEADINGS[path]
    positions = [text.index(heading) for heading in headings]
    assert positions == sorted(positions)
```

- [ ] **Step 2: Run the focused test and confirm the extension contract fails**

Run:

```bash
.venv/bin/python -m pytest tests/integration/test_documented_commands.py -q
```

Expected: failure because the current READMEs do not contain the complete module navigation and stable-API boundary.

- [ ] **Step 3: Complete the English guide**

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

`Extending VeriCCL` must describe the responsibility of every module named in `EXTENSION_MODULE_PATHS` and state that the internal modules are development entry points, not a stable plugin API.

`Outputs, Limitations, and Troubleshooting` must combine output geometry, exit codes `0/2/3/4/5`, Gurobi license limits, `offline-valid` versus `runtime-compatible`, MSCCL activation, MPI, trace, clock, and input diagnostics.

`License and Citation` must contain exactly:

```text
License: To be determined.
Citation: To be determined.
```

- [ ] **Step 4: Complete the Chinese guide with byte-identical commands**

Mirror the English section order and technical claims in `README.zh-CN.md`. Translate prose only. Ensure all Bash blocks, code literals, external references, commits, tags, expected log text, and paths match `README.md`.

- [ ] **Step 5: Run focused documentation and artifact tests**

Run:

```bash
.venv/bin/python -m pytest tests/integration/test_documented_commands.py tests/integration/test_workflow_artifacts.py -q
```

Expected: all selected tests pass; the documented solve/verify commands execute, repository paths resolve, and Bash blocks are byte-identical.

- [ ] **Step 6: Scan the two READMEs for excluded content**

Run:

```bash
rg -n 'SyCCL|sudo apt update|sudo apt install|Ubuntu prerequisites|Ubuntu前置依赖|CUDA, NCCL, and MPI preflight|CUDA、NCCL与MPI预检' README.md README.zh-CN.md
```

Expected: no matches.

- [ ] **Step 7: Commit Task 2**

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
