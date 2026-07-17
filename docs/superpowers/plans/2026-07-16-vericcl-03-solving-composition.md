# VeriCCL Solving and Composition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 PlanDAG 节点生成构造式和 MILP 候选，支持 latency/throughput/auto、固定并发度外层搜索、AG 对偶归约以及无阶段 barrier 的全局调度合成。

**Architecture:** 求解器把每个逻辑传输需求建模为确定的有向树或链，MILP 选择路径、channel、开始时间和资源 slot；AggregateState 的归约通过语义正确的 AG 树反向重建。所有后端返回同一 `SolveCandidate`，Orchestrator 负责预算、缓存、并行模型、回退与最终候选比较。

**Tech Stack:** Python、Gurobi、concurrent.futures、dataclasses、pytest；纯软件测试使用小型 fake backend，Gurobi 测试独立标记。

## Global Constraints

- 继承索引计划、Phase 01 和 Phase 02 全部约束。
- Gurobi 不可用或无许可证时，构造式与纯软件测试仍必须运行；MILP 集成测试报告 `not_run`。
- 每个固定 `K` 和 objective 形成独立模型；最多并行 4 个模型、每模型最多 12 线程，总线程不超过 CPU 核数。
- `search_space_restricted` 必须记录最短路径、对称性、批量构造等限制，不得把受限空间最优写成全局最优。
- 同一状态版本不能作为多次 REDUCE 源；归约树必须不相交，SEND 分支只允许来自完整状态。
- 本计划创建的每个测试模块声明 `pytestmark = pytest.mark.phase03`；Gurobi 测试同时声明 `pytest.mark.gurobi`。

---

### Task 1: Solver Requests, Results, Budgets, and Exact Cache Keys

**Files:**
- Create: `vericcl/solver/__init__.py`
- Create: `vericcl/solver/model.py`
- Create: `vericcl/solver/budget.py`
- Create: `vericcl/solver/cache.py`
- Create: `vericcl/tuning/__init__.py`
- Create: `vericcl/tuning/model.py`
- Test: `tests/unit/solver/test_model.py`
- Test: `tests/unit/solver/test_budget.py`
- Test: `tests/unit/solver/test_cache.py`

**Interfaces:**
- Produces: `SolveStatus`, `SolveRequest`, `SolveCandidate`, `SolveResult`, `SolverMetrics`
- Produces: `SolveBudget`, `ModelBudget`
- Produces: `candidate_cache_key(request: SolveRequest) -> str`
- Produces: initial immutable `TuningOverlay`; Phase 05 adds validation and repair behavior without changing its fields

- [x] **Step 1: Write failing result and budget tests**

```python
def test_selected_best_is_distinct_from_proven_optimal():
    candidate = make_candidate(selected_best=True, proven_optimal=False)
    assert candidate.selected_best
    assert not candidate.proven_optimal


def test_model_budget_is_bounded_by_both_deadlines():
    budget = SolveBudget(total_seconds=100, per_model_seconds=30, started_at=0)
    assert budget.model_budget(now=85).seconds == 15


def test_solver_seed_changes_cache_key():
    assert candidate_cache_key(request(seed=0)) != candidate_cache_key(request(seed=1))
```

- [x] **Step 2: Run tests and confirm solver package is absent**

Run: `python3 -m pytest tests/unit/solver/test_model.py tests/unit/solver/test_budget.py tests/unit/solver/test_cache.py -q`

Expected: collection fails.

- [x] **Step 3: Implement exact request and result records**

```python
@dataclass(frozen=True)
class TuningOverlay:
    overlay_id: str
    parent_candidate_id: Optional[str]
    channel_count: Optional[int] = None
    path_weights: tuple[tuple[str, float], ...] = ()
    temporary_forbidden: frozenset[ForbiddenTransfer] = frozenset()
    batch_size: Optional[int] = None
    tree_roots: tuple[tuple[int, int], ...] = ()
    tree_edges: tuple[tuple[int, int, int], ...] = ()
    lane_order: tuple[tuple[str, str], ...] = ()
    milp_parameters: tuple[tuple[str, float], ...] = ()
    warm_start_candidate_id: Optional[str] = None
    resolve_scope: tuple[str, ...] = ()
    hierarchy_template: Optional[str] = None


@dataclass(frozen=True)
class SolveRequest:
    inputs: ResolvedInput
    topology: Topology
    plan: PlanDAG
    overlay: Optional[TuningOverlay] = None


@dataclass(frozen=True)
class SolveCandidate:
    candidate_id: str
    node_schedules: Mapping[str, Schedule]
    objective_mode: ObjectiveMode
    channel_count: int
    metrics: SolverMetrics
    selected_best: bool
    proven_optimal: bool
    search_space_restricted: bool
    restrictions: tuple[str, ...]
    parent_candidate_id: Optional[str]
```

`SolverMetrics` includes status, objective values, best bound, MIP gap, `within_requested_gap`, solve time, model count, operation count, hop count, makespan, maximum normalized resource load, and solver version/seed/thread metadata.

- [x] **Step 4: Implement monotonic wall-clock budgets and two-level cache keys**

The structural cache key covers normalized inputs, topology structure, PlanDAG, enabled restrictions, objective, K, solver seed, and solver/model version. The performance cache additionally includes alpha/beta/invbw, calibrated B_link points, slice size, and environment signature. Expired or partial results are never returned as proven.

- [x] **Step 5: Run focused tests**

Run: `python3 -m pytest tests/unit/solver/test_model.py tests/unit/solver/test_budget.py tests/unit/solver/test_cache.py -q`

Expected: all tests pass.

### Task 2: Demand Expansion, Pruning, and Constructive Trees

**Files:**
- Create: `vericcl/solver/demands.py`
- Create: `vericcl/solver/pruning.py`
- Create: `vericcl/solver/constructive.py`
- Test: `tests/unit/solver/test_demands.py`
- Test: `tests/unit/solver/test_constructive.py`

**Interfaces:**
- Produces: `TransferDemand`, `CandidateEdge`, `SolverProblem`
- Produces: `build_solver_problem(node: PlanNode, inputs: ResolvedInput, topology: Topology) -> SolverProblem`
- Produces: `construct_candidate(problem: SolverProblem, channel_count: int) -> Schedule`

- [x] **Step 1: Write tests for forbidden members, shortest paths, and tree branching**

```python
def test_shared_transfer_is_removed_if_any_member_is_forbidden():
    problem = aggregate_problem(member_ids={0, 4}, forbidden=[ForbiddenTransfer(4, 0, 1, 2)])
    assert CandidateEdge(0, 1, 0) not in problem.candidate_edges


def test_complete_broadcast_state_can_branch():
    schedule = construct_candidate(broadcast_problem(4), channel_count=2)
    assert outgoing_destinations(schedule, src=0) == {1, 2, 3}


def test_constructive_schedule_respects_lane_order():
    schedule = construct_candidate(two_flow_problem(), channel_count=1)
    assert lane_intervals(schedule, LaneKey(0, 1, 0)).are_non_overlapping()


def test_local_contributor_does_not_create_self_transfer():
    schedule = construct_candidate(reduce_problem_with_root_local_value(), channel_count=1)
    assert all(transfer.src_rank != transfer.dst_rank for transfer in schedule.transfers)
```

- [x] **Step 2: Run tests and verify missing implementation**

Run: `python3 -m pytest tests/unit/solver/test_demands.py tests/unit/solver/test_constructive.py -q`

Expected: collection fails.

- [x] **Step 3: Expand PlanNode interfaces into deterministic transfer demands**

Each demand identifies stage, root, required leaf, logical position, contributors, member slice IDs, allowed links, forbidden members, and candidate paths. AllGather expands into Broadcast demands; AllToAll creates one source-to-owner chain per slice. Reduction dual nodes remain marked for Phase 03 Task 5.

- [x] **Step 4: Port only reusable heuristics into a deterministic constructive backend**

Build trees by earliest resource-ready time, then link duration, hop count, rank, and channel. Batch mode groups demands with identical root, legal path set, stage interface, and size; excess resource occupancy starts a new batch. The backend outputs a valid schedule or a typed infeasibility reason and may be used as MILP warm start.

- [x] **Step 5: Run focused tests**

Run: `python3 -m pytest tests/unit/solver/test_demands.py tests/unit/solver/test_constructive.py -q`

Expected: all tests pass.

### Task 3: Gurobi Adapter and MILP Hard Constraints

**Files:**
- Create: `vericcl/solver/gurobi_api.py`
- Create: `vericcl/solver/milp.py`
- Create: `vericcl/solver/scheduling.py`
- Test: `tests/unit/solver/test_gurobi_api.py`
- Test: `tests/gurobi/test_milp_feasibility.py`
- Test: `tests/gurobi/test_milp_infeasible.py`

**Interfaces:**
- Produces: `GurobiAdapter.available() -> bool`
- Produces: `solve_milp(problem: SolverProblem, channel_count: int, objective: ObjectiveMode, budget: ModelBudget, warm_start: Optional[Schedule]) -> SolveCandidate`

- [ ] **Step 1: Write an adapter test that does not require Gurobi**

```python
def test_missing_gurobi_is_reported_without_import_failure(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None if name == "gurobipy" else object())
    assert not GurobiAdapter.available()
    with pytest.raises(SolverUnavailableError):
        GurobiAdapter.require()
```

- [ ] **Step 2: Write marked 2-rank feasibility and infeasibility tests**

The feasible test solves a two-rank, two-slice Broadcast and asserts final delivery, `st_time >= ready_time`, and non-overlapping lane intervals. The infeasible test forbids the only directed link and asserts `SolveStatus.INFEASIBLE` without emitting a schedule.

- [ ] **Step 3: Implement lazy Gurobi import and exact parameter configuration**

Set `Seed=solver_seed`, `Threads=effective_threads`, `TimeLimit=model_budget.seconds`, `MIPGap=mip_gap`, deterministic output names, and no global module state. When `require_proven_optimal` is true, set `MIPGap=0.0`. Preserve solver status, incumbent, best bound, and actual gap independently.

- [ ] **Step 4: Implement path, causality, state, lane, and shared-resource constraints**

Use binary edge/channel selection, continuous start/end/arrival variables, indicator constraints for selected edges, exact one-parent constraints for required non-root tree nodes, and flow conservation for chains. Assign every selected transfer to one of `K` slots for each shared resource; operations in the same resource slot use an ordering binary and cannot overlap. Transfer duration is fixed by the outer K model. Add final reachability constraints before objectives.

- [ ] **Step 5: Extract a typed schedule and reject numerically invalid incumbents**

After solving, reconstruct paths and times, re-evaluate every hard constraint in Python with a documented tolerance, and reject any incumbent that violates topology, forbidden atoms, state semantics, lane order, resource slot capacity, or final reachability.

- [ ] **Step 6: Run adapter and available MILP tests**

Run: `python3 -m pytest tests/unit/solver/test_gurobi_api.py -q`

Expected: pass on all hosts.

Run: `python3 -m pytest tests/gurobi -q`

Expected: pass when Gurobi and a license are available; otherwise tests are reported as `not_run` through an explicit skip reason.

### Task 4: Objectives, Lower Bounds, K Search, and Parallel Models

**Files:**
- Create: `vericcl/solver/objectives.py`
- Create: `vericcl/solver/lower_bounds.py`
- Create: `vericcl/solver/search.py`
- Test: `tests/unit/solver/test_objectives.py`
- Test: `tests/unit/solver/test_lower_bounds.py`
- Test: `tests/unit/solver/test_search.py`

**Interfaces:**
- Produces: `throughput_time_lower_bound(problem: SolverProblem, max_channels: int) -> LowerBound`
- Produces: `search_models(problem: SolverProblem, config: SolverConfig, objective: ObjectiveMode, warm_start: Optional[Schedule]) -> tuple[SolveCandidate, ...]`
- Produces: `rank_candidates(candidates: Iterable[SolveCandidate]) -> tuple[SolveCandidate, ...]`

- [ ] **Step 1: Write objective and lower-bound tests**

```python
def test_latency_tie_breaks_by_operations_then_hops():
    ranked = rank_candidates([candidate(10, 8, 9), candidate(10, 7, 11), candidate(10, 7, 8)])
    assert ranked[0].hop_count == 8


def test_throughput_lower_bound_is_max_of_resource_and_dependency():
    bound = LowerBound(resource_us=80.0, dependency_us=95.0)
    assert bound.total_us == 95.0


def test_thread_allocation_never_exceeds_cpu_count():
    allocation = allocate_model_threads(model_count=4, requested_per_model=12, cpu_count=16)
    assert sum(allocation) <= 16
```

- [ ] **Step 2: Run tests and confirm missing implementations**

Run: `python3 -m pytest tests/unit/solver/test_objectives.py tests/unit/solver/test_lower_bounds.py tests/unit/solver/test_search.py -q`

Expected: collection fails.

- [ ] **Step 3: Implement lexicographic latency and throughput objectives**

Latency uses priorities makespan, physical transfer count, and hop count. Throughput minimizes maximum normalized steady load over directed links and shared resources, then makespan. Stable IDs are the final deterministic tie-break outside Gurobi.

- [ ] **Step 4: Implement the continuous resource LP and dependency bound**

The LP keeps legal topology, shared resources, forbidden transfers, semantic demands, and chosen hierarchy while relaxing integral paths, slice indivisibility, channel allocation, and TB order. Resource capacity uses `max(K * b_safe(K))`. The dependency bound ignores contention but keeps chain and join causality. Return both components and the maximum in microseconds.

- [ ] **Step 5: Implement bounded K-model execution**

Create one independent model per `K=1..max_channels` and objective. Run at most four concurrently, assign threads without exceeding CPU count, stop launching new models when the total budget expires, and retain complete incumbents from finished models. Seed, model order, solver version, threads, and wall-clock termination are written to metrics.

- [ ] **Step 6: Run focused tests**

Run: `python3 -m pytest tests/unit/solver/test_objectives.py tests/unit/solver/test_lower_bounds.py tests/unit/solver/test_search.py -q`

Expected: all tests pass.

### Task 5: AG Dual Conversion and Event-Driven Composition

**Files:**
- Create: `vericcl/planner/dual.py`
- Create: `vericcl/composer/__init__.py`
- Create: `vericcl/composer/dual.py`
- Create: `vericcl/composer/compose.py`
- Create: `vericcl/composer/timing.py`
- Test: `tests/unit/composer/test_dual.py`
- Test: `tests/unit/composer/test_compose.py`
- Test: `tests/property/test_ag_rs_duality.py`

**Interfaces:**
- Produces: `reverse_allgather_schedule(ag_schedule: Schedule, reduce_spec: CollectiveSpec, target_interface: StageInterface) -> Schedule`
- Produces: `compose(plan: PlanDAG, candidates: Mapping[str, SolveCandidate]) -> Schedule`
- Produces: `recompute_earliest_times(schedule: Schedule, topology: Topology) -> Schedule`

- [ ] **Step 1: Write dual and no-barrier tests**

```python
def test_ag_edge_becomes_reduce_with_rebuilt_state():
    rs = reverse_allgather_schedule(two_rank_ag_tree(), reduce_scatter_spec(), rs_target_interface())
    assert rs.transfers[0].kind == "REDUCE"
    assert final_contributors(rs) == {0: frozenset({0, 2}), 1: frozenset({1, 3})}
    assert {atom.slice_id for atom in rs.transfers[0].atoms} == rs.transfers[0].member_slice_ids
    assert rs.transfers[0].physical_bytes == rs.slice_size_bytes


def test_composer_pipelines_ready_slice_without_stage_barrier():
    schedule = compose(two_stage_plan(), independently_solved_nodes())
    assert stage_start(schedule, stage=1, slice_id=0) < stage_end(schedule, stage=0, slice_id=1)
```

- [ ] **Step 2: Run tests and observe missing composer modules**

Run: `python3 -m pytest tests/unit/composer/test_dual.py tests/unit/composer/test_compose.py tests/property/test_ag_rs_duality.py -q`

Expected: collection fails.

- [ ] **Step 3: Reverse AG trees by semantic state propagation**

Reverse each physical edge, replace SEND with REDUCE, create target-local initial states, merge only disjoint contributors, rebuild predecessor IDs, and recompute ready times from leaves to owners. Never reverse XML step order or reuse AG buffer offsets. Reject an AG tree whose reversal cannot satisfy exact target contributors.

- [ ] **Step 4: Compose local schedules through exact interfaces**

Map local ranks and values to global IDs, deduplicate exact reused physical transfers by transfer ID, connect producer final states to consumer initial states, and recompute earliest start times from actual state readiness and resource availability. No stage-wide edge or barrier is inserted.

- [ ] **Step 5: Run dual, composition, and property tests**

Run: `python3 -m pytest tests/unit/composer tests/property/test_ag_rs_duality.py -q`

Expected: all tests pass.

### Task 6: Solve Orchestrator and Auto Objective

**Files:**
- Create: `vericcl/solver/orchestrator.py`
- Test: `tests/unit/solver/test_orchestrator.py`
- Test: `tests/integration/test_solve_six_collectives.py`

**Interfaces:**
- Consumes: construct backend, MILP backend, lower bounds, composer, cache
- Produces: `solve(request: SolveRequest) -> SolveResult`

- [ ] **Step 1: Write orchestrator branch tests with fake backends**

Test constructive-only success, MILP timeout with constructive fallback, both backends disabled, manual hierarchy conflict, `require_proven_optimal=True` rejecting an unproven incumbent, `force_resolve=True` bypassing result caches, and `auto` skipping throughput when `gain_upper < min_expected_improvement` after CV adjustment.

- [ ] **Step 2: Write six-collective 2-rank integration tests**

For each direct operator, solve a tiny topology through the constructive backend, compose the global schedule, and assert exact final output contributors plus non-overlapping lane intervals. Mark only true Gurobi variants with `gurobi`.

- [ ] **Step 3: Run tests and confirm missing orchestrator**

Run: `python3 -m pytest tests/unit/solver/test_orchestrator.py tests/integration/test_solve_six_collectives.py -q`

Expected: collection fails.

- [ ] **Step 4: Implement the fixed strategy pipeline**

Normalize request, apply manual or automatic hierarchy, apply forbidden/topology/symmetry/shortest-path pruning, generate batched/tree candidates, run enabled MILP models with warm starts, compose global schedules, and return all complete candidates with explicit restrictions. No strategy may silently override another hard constraint.

- [ ] **Step 5: Implement auto-mode gating and selection**

Solve latency first. Compute `gain_upper=max(0,(T_latency-lower_bound)/T_latency)` and apply the configured threshold adjusted by measurement CV when available. Only then solve throughput. Compare candidates using the Phase 05 simulator interface when available and the conservative schedule makespan before Phase 05; record which comparison method was used.

- [ ] **Step 6: Run Phase 03 regression and coverage**

Run: `python3 -m pytest -m 'phase03 and not gurobi' --cov=vericcl.solver --cov=vericcl.composer --cov-report=term-missing -q`

Expected: all pure-software Phase 03 tests pass with at least 90% coverage for new modules.

Run: `rg -n '[\p{Han}]' vericcl/solver vericcl/composer tests/unit/solver tests/unit/composer -g '*.py'`

Expected: no output.
