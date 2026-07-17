# VeriCCL End-to-End Integration, Migration, and Acceptance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 接通 `vericcl solve/verify` 全流程，生成规范化任务目录和逐 XML 报告，完成 TACCL 内部标识迁移，并通过纯软件、Gurobi、XML 与可选硬件验收。

**Architecture:** CLI 仅负责请求装配和退出码；工作流服务按“输入→拓扑→计划→求解→合成→验证→XML→可选在线/调优→制品”执行。迁移清理放在全部新功能稳定后，避免在实现中途破坏原代码参考。

**Tech Stack:** Python、argparse、pathlib、json、pytest、coverage、现有 VeriCCL 全部模块。

## Global Constraints

- 继承全部前序阶段约束。
- 每次运行创建独立输出目录，不覆盖用户输入或历史候选。
- `solve` 和 `verify` 共享相同 ResolvedInput、验证器、报告器和超时语义。
- 只有完整验证且可执行的候选输出普通 `.xml`；不兼容但离线有效的候选输出 `.candidate.xml`。
- 移除内部 `taccl` 标识时保留 MSCCL 固定 schema、第三方版权和明确来源说明，并逐项列入 allowlist。
- 删除旧生成产物前先确认新包不依赖它们；不得删除用户新增文件或模板数据。
- 本计划创建的每个测试模块声明 `pytestmark = pytest.mark.phase07`，并按环境附加 `gurobi` 或 `hardware` 标记。

---

### Task 1: Workflow Service and Deterministic Artifact Layout

**Files:**
- Create: `vericcl/workflow.py`
- Create: `vericcl/artifacts/layout.py`
- Create: `vericcl/artifacts/writer.py`
- Create: `vericcl/artifacts/summary.py`
- Test: `tests/unit/artifacts/test_layout.py`
- Test: `tests/unit/artifacts/test_writer.py`
- Test: `tests/integration/test_workflow_artifacts.py`

**Interfaces:**
- Produces: `RunContext`, `RunArtifacts`, `execute_solve(context: RunContext) -> RunArtifacts`, `execute_verify(context: RunContext) -> RunArtifacts`
- Produces: `create_run_layout(base: Path, inputs: ResolvedInput, run_id: str) -> RunLayout`

- [x] **Step 1: Write deterministic layout and no-overwrite tests**

```python
def test_run_layout_matches_spec(tmp_path):
    layout = create_run_layout(tmp_path, resolved_allreduce(), run_id="0001")
    assert layout.root.name == "vericcl_allreduce_256MiB_0001"
    assert layout.resolved_input.name == "resolved-input.json"
    assert layout.summary.name == "run-summary.json"
    assert layout.schedules.name == "schedules"
    assert layout.reports.name == "reports"
    assert layout.traces.name == "traces"
```

Test ordinary XML, candidate XML, selected-best iteration naming, final alias output, report SHA binding, and failure when an existing non-empty run directory would be overwritten.

- [x] **Step 2: Run tests and confirm missing workflow/layout**

Run: `python3 -m pytest tests/unit/artifacts/test_layout.py tests/unit/artifacts/test_writer.py tests/integration/test_workflow_artifacts.py -q`

Expected: collection fails.

- [x] **Step 3: Implement atomic writes and complete lineage**

Write each JSON/XML to a temporary sibling, fsync, then rename. `run-summary.json` lists candidate ID, parent, iteration, XML/report paths, hashes, validation statuses, runtime compatibility, acceptance/rejection, selected_best, proven_optimal, restrictions, and final selection. Never label a file `optimal`.

- [x] **Step 4: Implement solve and verify workflow services**

`execute_solve` resolves inputs, loads topology, builds plan, solves, composes, runs pre-lowering validation, lowers only valid schedules, completes validation, writes every candidate result, and optionally tunes/validates online according to input. Invalid pre-lowering candidates receive reports but no XML. `execute_verify` parses a supplied XML and sidecar, reconstructs its typed program, runs the same validation pipeline, and optionally tunes. Both enforce wall-clock budgets.

- [x] **Step 5: Run artifact tests**

Run: `python3 -m pytest tests/unit/artifacts/test_layout.py tests/unit/artifacts/test_writer.py tests/integration/test_workflow_artifacts.py -q`

Expected: all tests pass.

### Task 2: Final CLI Wiring and Exit Codes

**Files:**
- Modify: `vericcl/cli/main.py`
- Create: `vericcl/cli/solve.py`
- Create: `vericcl/cli/verify.py`
- Create: `vericcl/cli/overrides.py`
- Test: `tests/unit/cli/test_solve.py`
- Test: `tests/unit/cli/test_verify.py`
- Test: `tests/integration/test_cli_end_to_end.py`

**Interfaces:**
- Produces executable commands `vericcl solve --topology TOPOLOGY --sketch SKETCH --atoms ATOMS` and `vericcl verify --topology TOPOLOGY --sketch SKETCH --atoms ATOMS --xml XML [--online] [--tune] [--timeout-s 10800]`

- [x] **Step 1: Write parser, override, and exit-code tests**

Test missing files, conflicting semantic CLI values without explicit override, accepted explicit override, fatal input exit, invalid schedule exit, runtime warning success with candidate output, online failure status, and clean solve/verify success.

- [x] **Step 2: Run tests and confirm missing handlers**

Run: `python3 -m pytest tests/unit/cli/test_solve.py tests/unit/cli/test_verify.py tests/integration/test_cli_end_to_end.py -q`

Expected: collection fails.

- [x] **Step 3: Implement thin handlers and stable exit codes**

Use `0` for completed valid offline work including runtime warnings, `2` for input/usage fatal errors, `3` for no semantic-valid candidate, `4` for requested online validation failure, and `5` for unexpected internal errors. Print one concise English summary to stdout and diagnostics to stderr; full details remain in JSON reports.

- [x] **Step 4: Run CLI integration tests**

Run: `python3 -m pytest tests/unit/cli tests/integration/test_cli_end_to_end.py -q`

Expected: all tests pass.

### Task 3: End-to-End Six-Collective and Hierarchical Acceptance Cases

**Files:**
- Create: `tests/e2e/test_six_collectives.py`
- Create: `tests/e2e/test_inplace_outofplace.py`
- Create: `tests/e2e/test_hierarchical_allreduce.py`
- Create: `tests/e2e/test_candidate_xml.py`
- Create: `tests/e2e/test_reproducibility.py`

**Interfaces:**
- Consumes: public CLI and artifact files only

- [x] **Step 1: Add six direct-operator end-to-end cases**

For each operator, run a 2-rank tiny solve, parse final XML/report, replay semantic outputs, check `cnt=1`, exact chunk counts/offsets, endpoint pairs, lane order, no deadlock, and required validation dimensions. Run both in-place and out-of-place where supported.

- [x] **Step 2: Add confirmed gateway hierarchy case**

Use two 4-rank nodes with only ranks 0 and 4 connected to NIC. Assert local Reduce to gateways, gateway RS+AG, local AG, no `[1,5]` link, no stage barrier, and exact final AllReduce contributors at every rank.

- [x] **Step 3: Add incompatibility and reproducibility cases**

Force a 257-step TB and verify candidate XML plus recommendations while semantic/BDD analysis still completes. Repeat a deterministic pure-software solve with seed 0 and identical environment signature; assert canonical schedule/report sections and hashes match, while solver metadata states the documented reproducibility limits.

- [x] **Step 4: Run end-to-end tests**

Run: `python3 -m pytest tests/e2e -q`

Expected: all pure-software cases pass; Gurobi-specific variants are separately marked.

### Task 4: TACCL-to-VeriCCL Source Migration and Provenance Allowlist

**Files:**
- Move: `taccl/examples/` to `vericcl/examples/legacy/`
- Move: `template/` to `vericcl/examples/templates/`
- Move: `Allgather.n16-1MB_i8_v1.xml` to `vericcl/examples/legacy/Allgather.n16-1MB_i8_v1.xml`
- Delete after dependency proof: `taccl/`
- Delete generated artifacts: `build/`
- Delete generated artifacts: `taccl.egg-info/`
- Modify: `setup.py`
- Create: `MIGRATION.md`
- Create: `vericcl/provenance.py`
- Create: `tests/unit/test_provenance.py`
- Create: `tests/integration/test_no_legacy_imports.py`

**Interfaces:**
- Produces: `ALLOWED_TACCL_REFERENCES: Mapping[str, str]`

- [ ] **Step 1: Prove the new package has no runtime dependency on old source**

Run: `rg -n '(^|[ .])taccl([ .]|$)|from taccl|import taccl' vericcl tests setup.py`

Expected before allowlist cleanup: only explicit legacy format/provenance references; no Python import of `taccl`.

Add a test-only import finder that raises `ImportError` for `taccl` and every `taccl.*` module, clear any such modules from `sys.modules`, then run the complete public workflow in that process. Expected: all tests pass without renaming or modifying the old directory.

- [ ] **Step 2: Move data references and update every path**

Move examples/templates/reference XML into the new package tree, update README, tests, setup package data, and migration documentation. Do not copy `__pycache__`, `.DS_Store`, old egg-info, or build outputs.

- [ ] **Step 3: Remove old source and generated artifacts**

After Step 1 passes, delete the old `taccl` source tree and generated build/egg-info directories through reviewed file deletions. Update `setup.py` so only `vericcl` packages and required example/runtime patch data are installed.

- [ ] **Step 4: Create and test the explicit provenance allowlist**

Allow only external `sccl_type` schema compatibility, legacy input format names, third-party copyright text, citation/source descriptions, and migration documentation. Every retained source-code string containing `taccl` maps to an English reason; unlisted occurrences fail the test.

- [ ] **Step 5: Run migration tests**

Run: `python3 -m pytest tests/unit/test_provenance.py tests/integration/test_no_legacy_imports.py -q`

Expected: all tests pass.

### Task 5: Documentation, Runtime Parameter Guide, and Final Acceptance

**Files:**
- Modify: `README.md`
- Modify: `Vericcl-work-document.md`
- Modify: `MIGRATION.md`
- Create: `docs/runtime-configuration.md`
- Create: `docs/validation-report.md`
- Create: `tests/integration/test_documented_commands.py`

**Interfaces:**
- Documents: installation, input schemas, solve/verify commands, MSCCL build parameters, runtime environment, output layout, validation statuses, online prerequisites, and migration.

- [ ] **Step 1: Write executable documentation tests**

Extract repository-local `vericcl --help`, solve, verify, and example-validation commands from Markdown and run them against temporary output directories. Validate that documented JSON examples resolve and documented environment formulas compute `NCCL_BUFFSIZE=2*S`.

- [ ] **Step 2: Update runtime configuration instructions**

Document exact locations and values: `src/include/msccl.h`, `MSCCL_CHUNKSTEPS 4`, `MSCCL_SLICESTEPS 4`, rebuild command `make -j src.build`, `NCCL_BUFFSIZE=2*slice_size_bytes`, Simple protocol, one XML per run, exact message range, and candidate/runtime warnings. Include release versus trace build instructions and trace environment variables.

- [ ] **Step 3: Update workflow and report documentation**

Document the three input files, immutable resolved input, output directory, candidate lineage, validation dimensions, selected_best/proven_optimal distinction, BDD opportunity semantics, online statistics, trace uncertainty, calibration cache, and all `not_run` conditions.

- [ ] **Step 4: Run documented-command and complete pure-software tests**

Run: `python3 -m pytest tests/integration/test_documented_commands.py -q`

Expected: all documented local commands pass.

Run: `python3 -m pytest -m 'not hardware and not gurobi' --cov=vericcl --cov-report=term-missing --cov-fail-under=90 -q`

Expected: all pure-software tests pass and total new-code coverage is at least 90%.

- [ ] **Step 5: Run optional integration matrices and report evidence**

Run: `python3 -m pytest -m gurobi -q`

Expected: pass with licensed Gurobi; otherwise explicit `not_run`.

Run: `python3 -m pytest -m hardware -q`

Expected: pass on configured hardware; otherwise explicit `not_run`.

- [ ] **Step 6: Run final source, artifact, and naming scans**

Run: `rg -n '[\p{Han}]' vericcl tests runtime -g '*.{py,c,cc,cu,cuh,h,json,xml}'`

Expected: no output.

Run: `rg -n 'taccl' vericcl tests setup.py -g '*.{py,json,xml}'`

Expected: every result is accepted by `tests/unit/test_provenance.py`; there are no legacy imports, command names, log prefixes, cache prefixes, or generated file prefixes.

Run: `python3 -m compileall -q vericcl`

Expected: exit code 0.

- [ ] **Step 7: Produce the final validation report**

Record exact commands, pass/fail/not_run counts, coverage, Gurobi status, hardware status, retained provenance strings, MSCCL patch verification, known environment limitations, and links to final XML/report examples. Do not claim completion before these fresh checks finish.
