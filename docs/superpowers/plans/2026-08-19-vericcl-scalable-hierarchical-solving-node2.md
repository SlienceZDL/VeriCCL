# VeriCCL Scalable Hierarchical Solving — node2 Implementation Plan

> **For Codex:** Use `superpowers:subagent-driven-development` to execute this plan one task at a time. Each task requires strict TDD, a requirements review, a code-quality review, fixes, focused verification, and a task commit before the next task begins.

**Goal:** Make hierarchical VeriCCL solving scale with topology and exact routing templates instead of the number of real slices, while preserving full slice semantics, deterministic scheduling, verification, XML lowering, and compatibility with existing inputs.

**Architecture:** Planning produces an effective hierarchy mode and exact local communication domains. Solver problems are split into semantic routing units, exact-isomorphic units are represented by one routing template, and a route-only MILP chooses a topology path/tree for each representative. The route patterns are instantiated for every real slice, then one deterministic global scheduler assigns channels, shared-resource slots, dependencies, and times across the complete transfer DAG. The old full timing MILP remains available only for requests that require a global optimality proof.

**Tech Stack:** Python 3.13, immutable dataclasses, Gurobi through the existing adapter, pytest, existing VeriCCL semantic/BDD/simulation/XML verification.

**Workspace:** `/home/zdl/.codex/worktrees/d889/VeriCCL`

**Reference checkout:** `/home/zdl/VeriCCL` is an old user checkout and must not be modified.

**Primary specification:** `Vericcl-work-document.md`

## Global constraints

- Preserve `slice_id = source_rank * slice_count + logical_slice_index` and all existing Atom, AggregateState, contributor, buffer, and XML semantics.
- Do not add required topology, sketch, or atom JSON fields. New cache/report fields must have backward-compatible defaults.
- Production code, tests, diagnostics, JSON, and XML must contain no Chinese characters.
- Exact template reuse may share only route structure. It must never copy representative channel assignments, resource slots, start times, or end times to real slices.
- Exact isomorphism must compare directed links, direction, performance curves, channel limits, shared-resource membership and capacity, semantic roles, demands, contributor/reduction state, slice size, allowed links, and forbidden transfers.
- A failed exact member mapping falls back only that member to its own routing model; it must not disable template reuse for unrelated members.
- All channel counts `K = 1..K_max` remain eligible unless an existing hard input constraint removes them. Do not introduce heuristic channel pruning.
- Template composition and independent PlanNode composition restrict the search space and must never report global optimality.
- `require_proven_optimal=true` must use the existing full timing MILP path; the scalable template path is ineligible for that request.
- Every final candidate must pass existing semantic, state, topology, timing, resource, buffer, endpoint, deadlock, XML, BDD, and simulation validation.
- Follow strict TDD: observe RED before production edits, implement the smallest behavior, observe GREEN, run the focused suite, scan for CJK, and commit the task.

## Completed Task 1: Effective planning metadata

Task 1 is complete at commits `3114fb2` and `3300d41`.

- Added `PlanningMode` and stable `planning_reason` metadata.
- Recorded requested versus effective hierarchy in caches and reports.
- Fixed incomplete gateway domains to fall back to direct planning.
- Final verification: `1167 passed, 1 skipped, 8 deselected`.

---

## Task 2: Plan hierarchical multi-rail AllGather on real gateway domains

**Files:**

- Modify: `vericcl/planner/groups.py`
- Modify: `vericcl/planner/hierarchy.py`
- Modify: `vericcl/planner/build.py`
- Modify: `vericcl/topology/isomorphism.py`
- Test: `tests/unit/planner/test_groups.py`
- Test: `tests/unit/planner/test_hierarchy.py`
- Create: `tests/integration/test_plan_gateway_allgather.py`
- Modify: `tests/property/test_plan_interfaces.py`

**Required interface:**

```python
def build_gateway_allgather_plan(
    inputs: ResolvedInput,
    topology: Topology,
    groups: CommunicationGroups,
) -> PlanDAG:
    ...
```

Use stable node IDs:

```text
local-gather-node-{node_id}-rail-{rail_index}
gateway-allgather-rail-{rail_index}
local-allgather-node-{node_id}-rail-{rail_index}
```

### Steps

1. Add failing tests for:
   - Two nodes with one gateway: local Gather, gateway AllGather, and local dissemination form a three-phase DAG.
   - Two nodes with four corresponding gateways: four independent rails; slice assignment is `slice_id % rail_count`.
   - A topology where only ranks 0 and 4 reach the NIC never creates `[1, 5]` or any other nonexistent gateway pair.
   - Unequal gateway counts, missing reverse logical links, incomplete node coverage, or non-exact-isomorphic local domains produce a direct plan with `no_eligible_gateway_domain` or another stable specific English reason.
   - Every `PlanEdge` interface exactly matches its producer and consumer, and the final interface equals direct AllGather semantics.
2. Run and observe RED:

```bash
.venv/bin/python -m pytest \
  tests/unit/planner/test_groups.py \
  tests/unit/planner/test_hierarchy.py \
  tests/integration/test_plan_gateway_allgather.py \
  tests/property/test_plan_interfaces.py -q
```

3. Tighten gateway discovery and eligibility:
   - Preserve sorted rank correspondence across nodes.
   - Require real bidirectional logical connectivity for every rail.
   - Use `exact_domain_signature` to compare directed links, performance, channel limits, and shared resources.
   - Never infer or synthesize a GPU-to-GPU logical link.
4. Build the three-phase per-rail DAG:
   - Local Gather moves original slices to each gateway.
   - Gateway AllGather handles only the slices assigned to that rail.
   - Local dissemination distributes all global slices to every local rank.
   - Preserve global slice IDs and generate only actual producer/consumer interfaces.
   - Do not add a global phase barrier.
   - On success set `GATEWAY_ALLGATHER` and `eligible_gateway_domain`; on failure return a direct plan with a stable reason.
5. Re-run the focused suite, then CJK and whitespace checks.
6. Commit:

```text
feat: plan hierarchical gateway allgather
```

---

## Task 3: Split routing units and build exact solver templates

**Files:**

- Create: `vericcl/solver/templates.py`
- Modify: `vericcl/solver/demands.py`
- Modify: `vericcl/solver/__init__.py`
- Create: `tests/unit/solver/test_templates.py`
- Modify: `tests/unit/solver/test_demands.py`
- Create: `tests/property/test_template_isomorphism.py`

**Required interfaces:**

```python
@dataclass(frozen=True)
class RoutingUnit:
    unit_id: str
    node: PlanNode
    demands: tuple[TransferDemand, ...]


@dataclass(frozen=True)
class TemplateMember:
    unit_id: str
    node_id: str
    rank_map: tuple[tuple[int, int], ...]
    contributor_map: tuple[tuple[int, int], ...]
    logical_position_map: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class SolverTemplate:
    template_id: str
    representative: RoutingUnit
    members: tuple[TemplateMember, ...]
    exact_signature: str


def split_routing_units(problem: SolverProblem) -> tuple[RoutingUnit, ...]:
    ...


def build_solver_templates(
    problems: tuple[SolverProblem, ...],
    planning_mode: PlanningMode,
) -> tuple[SolverTemplate, ...]:
    ...
```

### Steps

1. Add tests proving:
   - Direct 8-rank, 128-slice AllGather splits into 1024 routing units but only one exact structural template per source root, for eight templates total.
   - Different logical positions with the same root reuse one identity-rank template.
   - A slice-specific forbidden transfer separates only the impacted unit.
   - Direction, maximum channels, performance curve, shared-resource membership, semantic root role, contributors, or reduction-dual differences prevent merging.
   - Gather/Scatter/AllToAll units preserve chain semantics, while Broadcast/AllGather/reduction-dual units preserve tree semantics.
   - Exact rank renumbering yields an invertible member mapping; any resource change splits the classes.
2. Run the three focused files and observe RED.
3. Extract the existing MILP tree identity logic into pure routing-unit helpers without changing old MILP behavior.
4. Build a canonical exact signature containing the complete structural and semantic facts from the global constraints.
5. Allow same-group logical translation and verified exact cross-group rank mapping. Do not use approximate symmetry, and do not disable exact deduplication when the optional symmetry strategy is false.
6. Recheck mapped paths, forbidden transfers, contributor maps, and logical-position maps. Emit a standalone template for a failed member only.
7. Run focused tests, CJK scan, and whitespace checks.
8. Commit:

```text
feat: deduplicate exact routing templates
```

---

## Task 4: Solve representative route-only MILPs

**Files:**

- Create: `vericcl/solver/routing.py`
- Create: `vericcl/solver/routing_milp.py`
- Modify: `vericcl/solver/gurobi_api.py`
- Modify: `vericcl/solver/lower_bounds.py`
- Modify: `vericcl/solver/__init__.py`
- Create: `tests/unit/solver/test_routing.py`
- Create: `tests/gurobi/test_routing_milp.py`
- Create: `tests/gurobi/test_routing_model_size.py`

**Required interfaces:**

```python
@dataclass(frozen=True)
class RoutingModelStats:
    variable_count: int
    constraint_count: int
    general_constraint_count: int
    build_time_s: float
    optimize_time_s: float


@dataclass(frozen=True)
class RoutePattern:
    template_id: str
    channel_count: int
    objective_mode: ObjectiveMode
    selected_edges: tuple[LinkKey, ...]
    parent_edges: tuple[tuple[int, int], ...]
    model_stats: RoutingModelStats


def solve_route_milp(
    template: SolverTemplate,
    channel_count: int,
    objective_mode: ObjectiveMode,
    budget: ModelBudget,
) -> RoutePattern:
    ...
```

### Steps

1. Add immutable model and serialization tests; reject `ObjectiveMode.AUTO` at this layer.
2. Add Gurobi tests for leaf reachability, one parent, acyclic levels, flow conservation, forbidden/directed/allowed links, latency objective, throughput load objective, and model statistics.
3. Add the central scalability test: at fixed topology and `K=4`, route-model variables and constraints are exactly equal for 8, 16, 64, and 128 real slices.
4. Implement only route/tree variables and structural constraints. Do not create real-slice channel assignment, start/end time, overlap, or disjunctive ordering variables.
5. Preserve the existing full `solve_milp` implementation unchanged for proof-required fallback.
6. Extract selected edges and revalidate path continuity, single-parent structure, acyclicity, and all forbidden/allowed constraints in Python before returning.
7. Run focused unit/Gurobi tests and checks.
8. Commit:

```text
feat: solve representative routing models
```

---

## Task 5: Instantiate route patterns for all real slices

**Files:**

- Create: `vericcl/solver/instantiate.py`
- Modify: `vericcl/solver/scheduling.py`
- Modify: `vericcl/composer/dual.py`
- Create: `tests/unit/solver/test_instantiate.py`
- Modify: `tests/unit/composer/test_dual.py`
- Modify: `tests/property/test_ag_rs_duality.py`
- Modify: `tests/property/test_collective_semantics.py`

**Required interfaces:**

```python
@dataclass(frozen=True)
class InstantiationFailure:
    unit_id: str
    node_id: str
    reason: str


@dataclass(frozen=True)
class InstantiationResult:
    node_schedules: Mapping[str, Schedule]
    failures: tuple[InstantiationFailure, ...]


def instantiate_route_patterns(
    templates: tuple[SolverTemplate, ...],
    patterns: Mapping[str, RoutePattern],
    problems: tuple[SolverProblem, ...],
) -> InstantiationResult:
    ...
```

### Steps

1. Add tests for real member slice IDs, contributors, path prefixes, unique deterministic `transfer_id` values, and complete node schedules.
2. Prove representative time, channel, and resource-slot assignments are never copied. Instantiated schedules are marked `routing_only=True`, use deterministic provisional channel zero, empty resource slots, and zero-based provisional timing.
3. Add reduction-dual tests: reverse AG paths into REDUCE semantics, account for each contributor exactly once, prohibit repeated source-state reduction, and make a post-reduction SEND depend on the final RRC contributor state.
4. Extract a pure semantic reconstruction helper shared by old and new paths.
5. Revalidate every member mapping against allowed links and slice-specific forbidden transfers. Record only the failed unit for standalone fallback.
6. Run focused and property tests, CJK scan, and whitespace checks.
7. Commit:

```text
feat: instantiate routes for real slices
```

---

## Task 6: Deterministically schedule the complete transfer DAG

**Files:**

- Create: `vericcl/solver/global_scheduler.py`
- Modify: `vericcl/solver/scheduling.py`
- Modify: `vericcl/composer/compose.py`
- Modify: `vericcl/composer/timing.py`
- Create: `tests/unit/solver/test_global_scheduler.py`
- Modify: `tests/unit/composer/test_compose.py`
- Modify: `tests/unit/composer/test_timing.py`
- Modify: `tests/property/test_simulator_resources.py`

**Required interfaces:**

```python
def assign_global_resources(
    schedule: Schedule,
    topology: Topology,
    channel_count: int,
) -> Schedule:
    ...


def compose_routes(
    plan: PlanDAG,
    node_schedules: Mapping[str, Schedule],
    topology: Topology,
    channel_count: int,
) -> Schedule:
    ...
```

### Steps

1. Add tests proving reverse directions can overlap, different channels can overlap, the same `LaneKey` cannot overlap, and shared NIC/resource slots serialize only when they truly share a slot.
2. Add tests for semantic ready times, AggregateState fan-in, no global stage barrier, deterministic output independent of provisional transfer order, and explicit errors for cycles or unavailable capacity.
3. Implement a ready-only deterministic list scheduler:
   - Consider only transfers whose semantic predecessors have completed.
   - Enumerate channel `0..min(K, link.max_channels)-1`.
   - Enumerate every required resource slot `0..min(K, resource.max_channels)-1`.
   - For each combination, compute start as the maximum semantic, lane, and resource availability time and end from the conservative link duration.
   - Select earliest end, then earliest start, stable topology/resource tuple, and `transfer_id` as tie breakers.
4. Rebuild predecessor IDs, atom symbol ready times, transfer start/end times, channel assignments, and `resource_slots` metadata from scratch.
5. Keep the existing `compose` and `_retime` compatibility APIs. Add `compose_routes` for routing-only schedules and ensure it discards provisional allocations before calling the global scheduler.
6. Run focused/property tests and checks.
7. Commit:

```text
feat: schedule complete transfer DAG
```

---

## Task 7: Orchestrate template route search, member fallback, and proof fallback

**Files:**

- Modify: `vericcl/solver/model.py`
- Modify: `vericcl/solver/search.py`
- Modify: `vericcl/solver/orchestrator.py`
- Modify: `vericcl/solver/cache.py`
- Modify: `vericcl/solver/__init__.py`
- Modify: `tests/unit/solver/test_search.py`
- Modify: `tests/unit/solver/test_orchestrator.py`
- Modify: `tests/unit/solver/test_cache.py`
- Create: `tests/integration/test_scalable_solver_pipeline.py`

**Required diagnostics:**

```python
@dataclass(frozen=True)
class SearchDiagnostics:
    requested_problem_count: int = 0
    template_count: int = 0
    template_member_count: int = 0
    route_model_count: int = 0
    fallback_member_model_count: int = 0
    route_model_build_time_s: float = 0.0
    route_model_optimize_time_s: float = 0.0
    expansion_time_s: float = 0.0
    scheduling_time_s: float = 0.0
    maximum_variable_count: int = 0
    maximum_constraint_count: int = 0
    maximum_general_constraint_count: int = 0
```

Add `diagnostics: SearchDiagnostics = field(default_factory=SearchDiagnostics)` to `SolveResult` for backward compatibility.

### Steps

1. Add tests proving the scalable path is selected when `require_proven_optimal` is false and the existing full timing MILP is selected when it is true.
2. Add tests for complete `(template, K, objective)` search:
   - Up to `max_parallel_models` independent jobs may run concurrently.
   - CPU threads are allocated from actual concurrently running jobs without oversubscription.
   - One global candidate for a given `K` is created only when every required template and standalone fallback member completed for that `K`.
   - A missing template result invalidates only that `K`, not other channel counts.
   - AUTO still solves latency first and applies the existing throughput lower-bound gate.
3. Add exact cache signatures for planning mode, template signature, member-map digest, route-model version, global-scheduler version, objective, and `K`. Old cache payloads remain readable with default diagnostics.
4. Implement parallel route search using the existing global wall-clock and per-model budgets. Aggregate model counts once per actual route model; do not multiply counts by generated global candidates.
5. Instantiate every successful template set, solve failed members independently, compose and globally schedule the full real-slice candidate, and add `template_route_composition` plus `independent_node_composition` where applicable.
6. Mark every scalable candidate restricted and non-proven. Preserve verified constructive candidates and timeout incumbents as existing fallbacks.
7. Run focused tests and the scalable integration test, then checks.
8. Commit:

```text
feat: orchestrate scalable template solving
```

---

## Task 8: Expose structural scaling and effective solving diagnostics

**Files:**

- Modify: `vericcl/artifacts/reports.py`
- Modify: `vericcl/artifacts/summary.py`
- Modify: `vericcl/artifacts/writer.py`
- Modify: `vericcl/workflow.py`
- Modify: `tests/unit/artifacts/test_reports.py`
- Modify: `tests/unit/artifacts/test_writer.py`
- Modify: `tests/integration/test_workflow_artifacts.py`

### Steps

1. Add failing tests for all `SearchDiagnostics` fields in candidate validation reports and run summaries.
2. Report planning mode, requested PlanNode/problem count, exact template count, template members, actual route-model count, fallback count, build/optimize/expansion/scheduling times, and maximum model sizes.
3. Preserve schema compatibility: missing diagnostics decode as zeros, and unrelated existing report fields remain byte-stable after canonical serialization where possible.
4. Distinguish requested hierarchy, applied planning mode, restricted template composition, selected best, requested-gap status, and global proof status.
5. Ensure every emitted XML/candidate report records the exact scalable strategy and `TuningOverlay` used.
6. Run focused artifact/workflow tests and checks.
7. Commit:

```text
feat: report scalable solver diagnostics
```

---

## Task 9: End-to-end acceptance and scalability regression protection

**Files:**

- Modify: `tests/e2e/test_hierarchical_allreduce.py`
- Create: `tests/e2e/test_hierarchical_allgather.py`
- Modify: `tests/e2e/test_reproducibility.py`
- Modify: `tests/e2e/test_six_collectives.py`
- Modify: `tests/integration/test_solve_six_collectives.py`
- Modify: `tests/integration/test_workflow_artifacts.py`
- Update: `Vericcl-work-document.md`
- Update: `README.md` only if the public behavior or documented commands changed

### Steps

1. Add acceptance tests for gateway AllGather and AllReduce on representative two-node topologies, including one-gateway and multi-rail domains.
2. For 8/16/64/128 slices at fixed topology, assert exact template/route model counts stay constant while instantiated transfer count grows with real work.
3. Assert a practical solve smoke test no longer creates one full timing MILP per real slice and completes within a generous CI structural timeout; use diagnostics as the primary non-flaky acceptance criterion.
4. Validate full global semantics, AggregateState contributors, no forbidden/fake links, deterministic schedules, no stage barrier, BDD, event simulation, buffer lowering, endpoint pairing, deadlock freedom, XML readback, and runtime compatibility warnings.
5. Re-run all six directly supported collectives so the scalable path does not regress non-hierarchical behavior.
6. Update the work document with the final scalable architecture, exact fallback/optimality contract, diagnostic meanings, and Ubuntu/node2 verification evidence.
7. Run final verification:

```bash
.venv/bin/python -m pytest -m 'not hardware' -q
rg -n '[一-龥]' vericcl tests || true
git diff --check
```

8. Run any available Gurobi-marked tests separately and record hardware tests as `not_run` unless the required hardware environment is explicitly available.
9. Request a final requirements and code-quality review across the complete Task 1–9 range. Fix all Critical and Important findings before branch handoff.
10. Commit:

```text
test: validate scalable hierarchical solving
```

## Final acceptance criteria

- Route-model structural size is independent of real slice count for an unchanged exact template.
- All real slices still receive distinct, semantically complete paths and deterministic global resource assignments.
- Hierarchical AllGather and AllReduce use only real bidirectional gateway rails and exact-isomorphic domains.
- Existing full timing MILP remains the only path capable of claiming unrestricted global optimality.
- Every scalable candidate is fully validated and truthfully marked as search-space restricted.
- Full non-hardware regression, CJK scan, and whitespace checks pass.
