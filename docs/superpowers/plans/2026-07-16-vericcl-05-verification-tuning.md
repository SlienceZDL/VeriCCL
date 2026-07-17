# VeriCCL Offline Verification and Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现离线正确性验证、动态并发事件模拟、BDD 优化机会分析和受约束的增量调优，并为每个 XML 生成结构化验证报告。

**Architecture:** 正确性验证与优化机会分析严格分离。验证器先检查语义、状态、资源、BufferPlan、端点、TB 和 XML；BDD 只生成 flow/order hint；调优器在 copy-on-write overlay 中执行替换、依赖修复和时间重算，候选必须重新通过完整验证后才能成为 selected_best。

**Tech Stack:** Python、BDD 包装层、Gurobi 局部模型、dataclasses、pytest、hypothesis。

## Global Constraints

- 继承前序阶段全部约束。
- BDD 发现机会不代表候选无效；BDD 自身失败记录 `analysis_error`，该候选不得最终入选。
- BDD 不查找同一 lane 的重叠，因为求解硬约束已禁止重叠；它查找 ready 状态等待与替代 lane 空闲窗口。
- flow 替换只从首个分歧 Rank 后修改后缀；公共前缀和其他 flow 引用的 transfer 保持不变。
- 单个 BDD hint 不触发全局重新求解；顺序为增量贪心修复、局部 MILP、拒绝。
- 调优不得修改 CollectiveSpec、总大小、slice 大小/ID、用户 forbidden atom、手动分层、拓扑连通性或共享资源定义。
- 本计划创建的每个测试模块声明 `pytestmark = pytest.mark.phase05`；局部 MILP 测试同时声明 `pytest.mark.gurobi`。

---

### Task 1: Validation Status Model and Core Correctness Checks

**Files:**
- Create: `vericcl/verification/__init__.py`
- Create: `vericcl/verification/model.py`
- Create: `vericcl/verification/semantics.py`
- Create: `vericcl/verification/constraints.py`
- Test: `tests/unit/verification/test_model.py`
- Test: `tests/unit/verification/test_semantics.py`
- Test: `tests/unit/verification/test_constraints.py`

**Interfaces:**
- Produces: `ValidationStatus`, `CheckResult`, `ValidationReport`
- Produces: `verify_schedule_semantics(schedule: Schedule, inputs: ResolvedInput) -> CheckResult`
- Produces: `verify_schedule_constraints(schedule: Schedule, inputs: ResolvedInput, topology: Topology) -> CheckResult`
- Produces: `verify_schedule_pre_lowering(schedule: Schedule, inputs: ResolvedInput, topology: Topology) -> tuple[CheckResult, ...]`

- [x] **Step 1: Write status-separation tests**

```python
def test_warning_does_not_replace_semantic_status():
    report = report_with(semantic="valid", runtime="warning")
    assert report.overall_status == ValidationStatus.VALID
    assert not report.runtime_compatible


def test_bdd_analysis_error_blocks_selection_without_marking_semantics_invalid():
    report = report_with(semantic="valid", bdd="analysis_error")
    assert report.semantic.status == ValidationStatus.VALID
    assert not report.eligible_for_selection
```

- [x] **Step 2: Write semantic and constraint negative tests**

Cover missing final contributor, duplicate reduction contributor, inactive state reuse, wrong logical address, forbidden member in shared transfer, absent directed link, `st_time < ready_time`, same-lane overlap, exceeded fixed shared-resource K, and missing paired endpoint metadata.

- [x] **Step 3: Run tests and confirm missing verification modules**

Run: `python3 -m pytest tests/unit/verification/test_model.py tests/unit/verification/test_semantics.py tests/unit/verification/test_constraints.py -q`

Expected: collection fails.

- [x] **Step 4: Implement dimensioned results and exact state replay**

`ValidationReport` contains independent input, semantic, state, topology, timing, resource, buffer, endpoint, deadlock, XML, BDD, simulation, runtime, and online results. Replay Schedule transfers through a fresh PayloadLedger and compare exact final states with `required_outputs()`.

- [x] **Step 5: Implement structural constraint checks**

Validate topology and forbidden items, transfer pairing metadata, ready/start/end causality, non-overlapping lane intervals, resource occupancy, path-prefix continuity, unique IDs, and no unaccounted physical duplicate for member atoms. Diagnostics include stable IDs and English messages.

- [x] **Step 6: Run focused tests**

Run: `python3 -m pytest tests/unit/verification/test_model.py tests/unit/verification/test_semantics.py tests/unit/verification/test_constraints.py -q`

Expected: all tests pass.

### Task 2: Dynamic Concurrency Event Simulator

**Files:**
- Create: `vericcl/verification/simulator.py`
- Create: `vericcl/verification/resource_events.py`
- Test: `tests/unit/verification/test_simulator.py`
- Test: `tests/property/test_simulator_resources.py`

**Interfaces:**
- Produces: `SimulationEvent`, `SimulationResult`, `ResourceTimeline`
- Produces: `simulate_schedule(schedule: Schedule, topology: Topology) -> SimulationResult`

- [x] **Step 1: Write concurrency and direction tests**

```python
def test_opposite_directions_progress_in_parallel():
    result = simulate_schedule(opposite_direction_schedule(), calibrated_two_rank_topology())
    assert result.completion_time_us == pytest.approx(single_transfer_time_us())


def test_two_channels_share_total_directed_link_bandwidth():
    result = simulate_schedule(two_channel_same_direction_schedule(), calibrated_two_rank_topology())
    assert result.completion_time_us >= conservative_shared_bandwidth_time_us()
```

- [x] **Step 2: Add deterministic event-order and dependency tests**

Test equal-time event tie-breaking, REDUCE join readiness as max predecessor end, shared NIC occupancy, idle lane with unavailable semantic data, and transfer duration changes when active concurrency K changes.

- [x] **Step 3: Run tests and verify missing simulator**

Run: `python3 -m pytest tests/unit/verification/test_simulator.py tests/property/test_simulator_resources.py -q`

Expected: collection fails.

- [x] **Step 4: Implement a deterministic discrete-event engine**

Maintain semantic-ready queues, lane heads, directed-link active sets, and shared-resource active sets. At every event, recompute conservative duration using current calibrated K, advance the earliest stable event, and update downstream readiness. Deduplicate member atoms by transfer ID. Record resource busy/idle intervals, queue waits, and completion time in microseconds.

- [x] **Step 5: Run simulator tests**

Run: `python3 -m pytest tests/unit/verification/test_simulator.py tests/property/test_simulator_resources.py -q`

Expected: all tests pass.

### Task 3: Flow and TB-Order BDD Opportunity Analysis

**Files:**
- Modify: `setup.py`
- Create: `vericcl/verification/flow_index.py`
- Create: `vericcl/verification/bdd_backend.py`
- Create: `vericcl/verification/bdd_flow.py`
- Create: `vericcl/verification/bdd_order.py`
- Test: `tests/unit/verification/test_flow_index.py`
- Test: `tests/unit/verification/test_bdd_flow.py`
- Test: `tests/unit/verification/test_bdd_order.py`

**Interfaces:**
- Produces: `FlowRecord`, `LaneState`, `FlowReplacementHint`, `TBOrderHint`, `BDDAnalysisResult`
- Produces: `analyze_flow_congestion(schedule: Schedule, topology: Topology, inputs: ResolvedInput) -> BDDAnalysisResult`
- Produces: `analyze_tb_order(tb_program: ThreadblockProgram, schedule: Schedule) -> BDDAnalysisResult`

- [x] **Step 1: Add the BDD dependency through an internal adapter**

Add runtime dependency `dd` to `setup.py`, but expose it only through `bdd_backend.py`. The adapter declares compact bit-vector variables, builds relations from integer IDs, performs union/intersection/difference/complement, enumerates satisfying ID tuples, and converts backend exceptions to `analysis_error`.

- [x] **Step 2: Write flow truncation and cross-endpoint resource tests**

```python
def test_member_flows_stop_comparing_after_first_aggregate_merge():
    index = build_flow_index(two_members_with_shared_suffix())
    assert index.flow("f0").comparison_end == index.flow("f1").comparison_end
    assert index.shared_suffix_transfer_ids == frozenset({"tx-shared"})


def test_different_root_leaf_flows_share_lane_index():
    result = analyze_flow_congestion(crossing_flows_with_ready_wait(), topology(), inputs())
    assert result.hints[0].bottleneck_lane == LaneKey(1, 2, 0)
```

- [x] **Step 3: Write waiting-opportunity and TB inversion tests**

Construct a non-leaf state with `ready_time < st_time` and an earlier compatible idle lane; assert a FlowReplacementHint includes source flow, candidate flow IDs, divergence rank, waiting transfer, wait interval, and earliest candidate start. For TB order, assert a later step that became ready first produces a swap hint only when no necessary order exists.

- [x] **Step 4: Run tests and confirm missing analysis modules**

Run: `python3 -m pytest tests/unit/verification/test_flow_index.py tests/unit/verification/test_bdd_flow.py tests/unit/verification/test_bdd_order.py -q`

Expected: collection fails.

- [x] **Step 5: Implement compact relations with external metadata**

BDD variables contain only encoded `flow_id`, `candidate_flow_id`, `demand_id`, and `lane_id` for flow analysis, or `tb_id`, `op_id`, and `step_index` for order analysis. Rank, time, paths, resource intervals, and transfer metadata stay in Python records. Candidate compatibility prefilters topology, forbidden atoms, logical demand, stage interface, and target semantics; it does not modify schedules.

- [x] **Step 6: Run BDD tests**

Run: `python3 -m pytest tests/unit/verification/test_flow_index.py tests/unit/verification/test_bdd_flow.py tests/unit/verification/test_bdd_order.py -q`

Expected: all tests pass, including explicit `analysis_error` behavior.

### Task 4: TuningOverlay, Impact Closure, and Safe Suffix Repair

**Files:**
- Modify: `vericcl/tuning/model.py`
- Create: `vericcl/tuning/impact.py`
- Create: `vericcl/tuning/repair.py`
- Create: `vericcl/tuning/local_milp.py`
- Test: `tests/unit/tuning/test_overlay.py`
- Test: `tests/unit/tuning/test_impact.py`
- Test: `tests/unit/tuning/test_repair.py`
- Test: `tests/gurobi/test_local_repair_milp.py`

**Interfaces:**
- Produces: `TuningOverlay`, `RepairResult`, `ImpactClosure`
- Produces: `compute_impact_closure(schedule: Schedule, changed_transfer_ids: frozenset[str], topology: Topology) -> ImpactClosure`
- Produces: `repair_flow_suffix(schedule: Schedule, hint: FlowReplacementHint, overlay: TuningOverlay, topology: Topology, inputs: ResolvedInput) -> RepairResult`
- Produces: `solve_local_repair(schedule: Schedule, hint: FlowReplacementHint, impact: ImpactClosure, overlay: TuningOverlay, topology: Topology, inputs: ResolvedInput, budget: ModelBudget) -> RepairResult`

- [x] **Step 1: Write immutability and boundary tests**

Assert overlay changes do not mutate ResolvedInput or the parent Schedule. Reject overlay changes to slice size, CollectiveSpec, manual hierarchy, user forbidden atoms, topology links, or shared resource membership.

- [x] **Step 2: Write fixed-point impact closure tests**

Start from one changed transfer and assert the closure includes downstream semantic dependencies, later operations on the same lane, operations whose directed-link concurrency changes, shared-NIC operations, and recursively affected descendants until no new item is added.

- [x] **Step 3: Write suffix repair tests**

Test preservation of the common prefix, replacement after the first divergence rank, missing leaf delivery repair, AggregateState contributor repair, shared-suffix deduplication, forbidden candidate rejection, and deterministic earliest-time recomputation.

- [x] **Step 4: Run pure-software tuning tests and confirm failure**

Run: `python3 -m pytest tests/unit/tuning/test_overlay.py tests/unit/tuning/test_impact.py tests/unit/tuning/test_repair.py -q`

Expected: collection fails.

- [x] **Step 5: Implement greedy repair and local MILP fallback**

Greedy cost orders legal suffixes by added transfer time, hops, lane wait, shared-resource load, repair count, then stable ID. If it fails and the hint has positive expected gain, build a local MILP containing only affected states, lanes, resources, and descendants while fixing the common prefix and unrelated order. Return success, infeasible, timeout, or invalid with evidence; never trigger global solve here.

- [x] **Step 6: Run tuning tests**

Run: `python3 -m pytest tests/unit/tuning -q`

Expected: all pure-software tests pass.

Run: `python3 -m pytest tests/gurobi/test_local_repair_milp.py -q`

Expected: pass with Gurobi, otherwise explicit `not_run`.

### Task 5: Iterative Tuning Engine, Reports, and Artifact Binding

**Files:**
- Create: `vericcl/tuning/engine.py`
- Create: `vericcl/artifacts/__init__.py`
- Create: `vericcl/artifacts/reports.py`
- Create: `vericcl/artifacts/hashing.py`
- Create: `vericcl/verification/pipeline.py`
- Test: `tests/unit/tuning/test_engine.py`
- Test: `tests/unit/artifacts/test_reports.py`
- Test: `tests/integration/test_verify_and_tune.py`

**Interfaces:**
- Produces: `verify_candidate(schedule: Schedule, artifact: XmlArtifact, inputs: ResolvedInput, topology: Topology) -> ValidationReport`
- Produces: `tune(initial: SolveCandidate, context: TuningContext) -> TuningResult`
- Produces: `build_validation_json(report: ValidationReport) -> str`

- [x] **Step 1: Write candidate acceptance and rejection tests**

Test strict simulated improvement without online data, statistical threshold with online medians/CVs, rejection on any correctness failure, rejection on BDD `analysis_error`, retention of runtime-incompatible candidates for offline analysis only, exact candidate signature deduplication, and selection of best historical candidate rather than last candidate.

- [x] **Step 2: Write report completeness and SHA binding tests**

Assert every report includes normalized input hash, requested/applied strategies and parameters, overlay, hierarchy plan, channels, BufferPlan summary, solver metrics, all validation dimensions, candidate lineage, rejection reason, selected_best, proven_optimal, search_space_restricted, runtime_compatible, XML SHA-256, and BDD/simulation evidence.

- [x] **Step 3: Run tests and confirm missing pipeline**

Run: `python3 -m pytest tests/unit/tuning/test_engine.py tests/unit/artifacts/test_reports.py tests/integration/test_verify_and_tune.py -q`

Expected: collection fails.

- [x] **Step 4: Implement the required validation order**

Run semantic/state/topology/timing/resource checks before XML lowering; only a pre-lowering-valid schedule may build BufferPlan and endpoints. Then run BufferPlan/liveness, endpoint/TB/deadlock/XML checks, compatibility warning, BDD analysis, and dynamic simulation. Retain structured failures at the stage where they occur. A BDD opportunity is a successful analysis result.

- [x] **Step 5: Implement bounded iterative tuning**

Use at most 20 iterations and the remaining verification wall-clock budget. Generate candidates from BDD flow/order hints and permitted overlay dimensions, run incremental simulation first, then complete validation only for improved candidates. Preserve every rejected candidate and reason. Mark selected_best only after comparing all fully validated history.

- [x] **Step 6: Run Phase 05 regression and coverage**

Run: `python3 -m pytest -m 'phase05 and not gurobi' --cov=vericcl.verification --cov=vericcl.tuning --cov=vericcl.artifacts --cov-report=term-missing -q`

Expected: all pure-software Phase 05 tests pass with at least 90% coverage for new modules.

Run: `rg -n '[\p{Han}]' vericcl/verification vericcl/tuning vericcl/artifacts tests/unit/verification tests/unit/tuning -g '*.{py,json}'`

Expected: no output.
