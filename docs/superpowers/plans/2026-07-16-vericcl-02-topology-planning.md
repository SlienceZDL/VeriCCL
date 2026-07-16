# VeriCCL Topology and Planning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将拓扑输入规范化为有向链路、lane 和共享资源模型，并把六类直接算子及可选分层策略展开为语义接口严格的全局 PlanDAG。

**Architecture:** 拓扑模块只描述可用资源及性能，不包含算子语义；Planner 只生成局部求解任务及输入输出映射，不决定具体传输时间。PlanDAG 中的每个节点都携带合法子拓扑和精确 contributors 接口，Composer 后续据此执行事件驱动合成。

**Tech Stack:** Python dataclasses、heapq、collections、pytest、hypothesis；兼容现有 TACCL 拓扑示例并扩展 SyCCL 风格共享资源。

## Global Constraints

- 继承索引计划和 Phase 01 全部约束。
- 所有链路均有方向；不得根据反向链路存在推导正向链路。
- 现有链路默认允许，唯一用户传输禁用来源是 atom forbidden items。
- 通信组内 Rank 升序；同构节点间只按实际网关位置一一对应，不创建虚拟链路。
- 局部计划复用要求通信域、方向、容量、共享资源和语义接口完全同构。
- 本计划创建的每个测试模块声明 `pytestmark = pytest.mark.phase02`。

---

### Task 1: Directed Topology and Performance Model

**Files:**
- Create: `vericcl/topology/__init__.py`
- Create: `vericcl/topology/model.py`
- Create: `vericcl/topology/performance.py`
- Test: `tests/unit/topology/test_model.py`
- Test: `tests/unit/topology/test_performance.py`

**Interfaces:**
- Produces: `LinkKey`, `LaneKey`, `PerformanceCurve`, `DirectedLink`, `SharedResource`, `Topology`
- Produces: `safe_per_channel_bandwidth(curve: PerformanceCurve, concurrency: int) -> float`
- Produces: `transfer_duration_us(link: DirectedLink, slice_size_bytes: int, concurrency: int) -> float`

- [x] **Step 1: Write failing direction, lane, and duration tests**

```python
def test_links_are_directional():
    topology = two_rank_topology(links=[link(0, 1)])
    assert topology.has_link(0, 1)
    assert not topology.has_link(1, 0)


def test_uncalibrated_duration_is_conservative():
    edge = link(0, 1, alpha_us=2.0, invbw_us=5.0)
    assert transfer_duration_us(edge, 1024, concurrency=3) == 11.0


def test_calibrated_curve_uses_prefix_minimum():
    curve = PerformanceCurve(alpha_us=2.0, invbw_us=5.0, bandwidth_bytes_per_us={1: 100.0, 2: 170.0})
    assert safe_per_channel_bandwidth(curve, 2) == 85.0
```

- [x] **Step 2: Run tests and observe missing topology modules**

Run: `python3 -m pytest tests/unit/topology/test_model.py tests/unit/topology/test_performance.py -q`

Expected: collection fails on missing imports.

- [x] **Step 3: Implement immutable topology records**

```python
@dataclass(frozen=True, order=True)
class LinkKey:
    src_rank: int
    dst_rank: int


@dataclass(frozen=True, order=True)
class LaneKey:
    src_rank: int
    dst_rank: int
    channel: int


@dataclass(frozen=True)
class PerformanceCurve:
    alpha_us: float
    invbw_us: float
    bandwidth_bytes_per_us: Mapping[int, float]


@dataclass(frozen=True)
class DirectedLink:
    key: LinkKey
    max_channels: int
    performance: PerformanceCurve
    resource_ids: tuple[str, ...]
```

`Topology` stores rank count, links by LinkKey, shared resources, node membership, gateways, and an isomorphism signature. It exposes `destinations(src)`, `sources(dst)`, `lanes(link, channel_count)`, and `resources_for(link)` in deterministic order.

- [x] **Step 4: Implement parameter consistency and duration formulas**

If input alpha, beta, and invbw disagree, retain invbw, compute `beta_effective=invbw-alpha`, and append an English warning. Uncalibrated duration is `alpha + K*beta_effective`. Calibrated duration uses `min(B_link(k)/k for 1 <= k <= K)` and `alpha + S/b_safe(K)`; missing intermediate k is an input error, not interpolation.

- [x] **Step 5: Run focused tests**

Run: `python3 -m pytest tests/unit/topology/test_model.py tests/unit/topology/test_performance.py -q`

Expected: all tests pass.

### Task 2: Topology Loader, Legacy Compatibility, and Shared Resources

**Files:**
- Create: `vericcl/topology/loader.py`
- Create: `vericcl/topology/legacy.py`
- Create: `vericcl/examples/topo/two_node_gateway.json`
- Test: `tests/unit/topology/test_loader.py`
- Test: `tests/unit/topology/test_shared_resources.py`

**Interfaces:**
- Consumes: `ResolvedInput.resolved_topology`
- Produces: `load_topology(inputs: ResolvedInput) -> Topology`
- Produces: `convert_legacy_topology(raw_topology: Mapping[str, object], raw_sketch: Mapping[str, object]) -> Mapping[str, object]`

- [x] **Step 1: Write loader tests for explicit and legacy inputs**

```python
def test_gateway_topology_contains_only_real_cross_node_links():
    topology = load_example("two_node_gateway.json")
    assert topology.has_link(0, 4)
    assert topology.has_link(4, 0)
    assert not topology.has_link(1, 5)


def test_channels_share_directed_link_resource():
    topology = load_example("two_node_gateway.json")
    resources = topology.resources_for(LinkKey(0, 4))
    assert "inter-node-0-to-1" in resources
    assert "nic-node-0-egress" in resources


def test_legacy_ndv2_example_resolves_rank_count():
    topology = load_legacy_pair("topo-ndv2-1MB.json", "sk2-ndv2-n2.json")
    assert topology.rank_count == 16
```

- [x] **Step 2: Run tests and confirm loader is absent**

Run: `python3 -m pytest tests/unit/topology/test_loader.py tests/unit/topology/test_shared_resources.py -q`

Expected: collection fails on missing loader functions.

- [x] **Step 3: Implement explicit schema parsing**

The explicit topology schema contains `ranks`, `nodes`, `directed_links`, and `shared_resources`. Each directed link names src, dst, max_channels, alpha, beta or invbw, optional calibrated bandwidth points, and resource IDs. Each shared resource contains an ID, direction-specific membership, and the same performance representation used by links.

- [x] **Step 4: Implement legacy conversion without mutating source dictionaries**

Port only the necessary matrix and `internode_conn` interpretation from `taccl/topologies/generic.py` and `taccl/cli/common.py`. Convert switch and NIC hyperedges into explicit shared resources. Preserve the legacy source snapshot and add provenance `legacy_format="taccl_topology_v2"`; do not expose TACCL names in internal class names or log prefixes.

- [x] **Step 5: Run topology loader tests**

Run: `python3 -m pytest tests/unit/topology -q`

Expected: all topology tests pass and parameter mismatch warnings are deterministic.

### Task 3: Legal Paths, Exact Isomorphism, and Communication Groups

**Files:**
- Create: `vericcl/topology/paths.py`
- Create: `vericcl/topology/isomorphism.py`
- Create: `vericcl/planner/groups.py`
- Test: `tests/unit/topology/test_paths.py`
- Test: `tests/unit/planner/test_groups.py`

**Interfaces:**
- Produces: `shortest_path_set(topology: Topology, src: int, dst: int) -> tuple[tuple[int, ...], ...]`
- Produces: `exact_domain_signature(topology: Topology, ranks: tuple[int, ...]) -> str`
- Produces: `discover_communication_groups(topology: Topology) -> CommunicationGroups`

- [x] **Step 1: Write tests for all equal shortest paths and gateway grouping**

```python
def test_shortest_path_set_keeps_equal_cost_routes():
    topology = diamond_topology()
    assert shortest_path_set(topology, 0, 3) == ((0, 1, 3), (0, 2, 3))


def test_gateway_groups_do_not_invent_rank_pairs():
    groups = discover_communication_groups(two_node_gateway_topology())
    assert groups.intra_node == ((0, 1, 2, 3), (4, 5, 6, 7))
    assert groups.inter_node == ((0, 4),)
    assert (1, 5) not in groups.inter_node
```

- [x] **Step 2: Run tests and verify missing path/group functions**

Run: `python3 -m pytest tests/unit/topology/test_paths.py tests/unit/planner/test_groups.py -q`

Expected: collection fails.

- [x] **Step 3: Implement deterministic all-shortest-path enumeration**

Use Dijkstra over `invbw_us` with predecessor sets, then enumerate predecessor DAG paths in rank order. A path is legal only if every directed edge exists; forbidden atom filtering remains per slice and stage in the solver.

- [x] **Step 4: Implement exact communication-domain signatures**

The signature includes sorted rank-relative directed links, max channels, performance parameters, shared-resource membership, node/gateway roles, and direction. Two groups are reusable only when their signatures and logical input/output interfaces match exactly.

- [x] **Step 5: Run focused tests**

Run: `python3 -m pytest tests/unit/topology/test_paths.py tests/unit/planner/test_groups.py -q`

Expected: all tests pass.

### Task 4: PlanDAG Types and Direct Collective Plans

**Files:**
- Create: `vericcl/planner/__init__.py`
- Create: `vericcl/planner/model.py`
- Create: `vericcl/planner/direct.py`
- Test: `tests/unit/planner/test_direct.py`
- Test: `tests/property/test_plan_interfaces.py`

**Interfaces:**
- Consumes: `CollectiveSpec`, `Topology`, `required_outputs()`
- Produces: `LogicalValue`, `StageInterface`, `PlanNode`, `PlanEdge`, `PlanDAG`
- Produces: `build_direct_plan(inputs: ResolvedInput, topology: Topology) -> PlanDAG`
- Produces: `build_internal_scatter(root: int, group: tuple[int, ...], values: StageInterface, topology: Topology) -> tuple[PlanNode, ...]`
- Produces: `build_internal_gather(root: int, group: tuple[int, ...], values: StageInterface, topology: Topology) -> tuple[PlanNode, ...]`

- [ ] **Step 1: Write tests for direct plan semantics**

```python
def test_allgather_plan_is_sum_of_broadcasts():
    plan = build_direct_plan(resolved_allgather(ranks=2, slices=2), full_duplex_topology(2))
    assert [node.local_collective.kind for node in plan.nodes] == [CollectiveKind.BROADCAST] * 4
    assert plan.final_outputs == required_outputs(plan.collective, 2, 2)


def test_scatter_and_gather_are_not_direct_targets():
    with pytest.raises(InputValidationError):
        build_direct_plan(resolved_operator("scatter"), full_duplex_topology(2))


def test_scatter_and_gather_are_available_as_internal_nodes():
    scatter_nodes = build_internal_scatter(0, (0, 1), scattered_interface(), full_duplex_topology(2))
    gather_nodes = build_internal_gather(0, (0, 1), gathered_interface(), full_duplex_topology(2))
    assert scatter_nodes[-1].logical_output == scattered_interface()
    assert gather_nodes[-1].logical_output == gathered_interface()
```

- [ ] **Step 2: Run tests and observe missing planner**

Run: `python3 -m pytest tests/unit/planner/test_direct.py tests/property/test_plan_interfaces.py -q`

Expected: collection fails on missing planner types.

- [ ] **Step 3: Implement immutable PlanDAG records**

```python
@dataclass(frozen=True)
class StageInterface:
    values: Mapping[OutputSlot, frozenset[int]]


@dataclass(frozen=True)
class PlanNode:
    node_id: str
    stage_id: int
    local_collective: CollectiveSpec
    communication_group: tuple[int, ...]
    logical_input: StageInterface
    logical_output: StageInterface
    allowed_links: frozenset[LinkKey]
    shared_resource_ids: frozenset[str]
    dual_of_node_id: Optional[str] = None
```

`PlanDAG` validates unique IDs, acyclicity, every edge's producer/consumer interface equality, and exact final outputs. Stage IDs express semantic scope only and never create an implicit barrier.

- [ ] **Step 4: Implement six direct operator decompositions**

Broadcast creates one propagation node per logical slice; AllGather creates one Broadcast per source slice; Reduce and ReduceScatter create dual descriptors that Phase 03 will materialize from solved AG trees; AllReduce creates a ReduceScatter dual stage followed by AllGather; AllToAll creates source/destination demands directly. Scatter and Gather builders are exposed only for internal PlanDAG composition. No direct full-global AllReduce candidate is emitted when hierarchy is explicitly enabled and the dominated gateway template applies.

- [ ] **Step 5: Run direct planner and property tests**

Run: `python3 -m pytest tests/unit/planner/test_direct.py tests/property/test_plan_interfaces.py -q`

Expected: all tests pass.

### Task 5: Hierarchical Planner and Stage Interface Validation

**Files:**
- Create: `vericcl/planner/hierarchy.py`
- Create: `vericcl/planner/build.py`
- Test: `tests/unit/planner/test_hierarchy.py`
- Test: `tests/integration/test_plan_gateway_allreduce.py`

**Interfaces:**
- Consumes: communication groups, direct planner, manual hierarchy input
- Produces: `build_plan(inputs: ResolvedInput, topology: Topology) -> PlanDAG`
- Produces: `validate_manual_hierarchy(plan_spec: object, topology: Topology, collective: CollectiveSpec) -> None`

- [ ] **Step 1: Write the confirmed two-node gateway AllReduce test**

```python
def test_gateway_allreduce_uses_only_real_gateways():
    plan = build_plan(resolved_hierarchical_allreduce(), two_node_gateway_topology())
    assert [(node.local_collective.kind, node.communication_group) for node in plan.nodes] == [
        (CollectiveKind.REDUCE, (0, 1, 2, 3)),
        (CollectiveKind.REDUCE, (4, 5, 6, 7)),
        (CollectiveKind.REDUCE_SCATTER, (0, 4)),
        (CollectiveKind.ALL_GATHER, (0, 4)),
        (CollectiveKind.ALL_GATHER, (0, 1, 2, 3)),
        (CollectiveKind.ALL_GATHER, (4, 5, 6, 7)),
    ]
    assert all(LinkKey(1, 5) not in node.allowed_links for node in plan.nodes)
```

- [ ] **Step 2: Add illegal manual hierarchy tests**

Test rejection of a non-existent communication edge, mismatched contributors between adjacent stages, a group with unsorted duplicate ranks, and a final interface that does not equal the global CollectiveSpec.

- [ ] **Step 3: Run tests and confirm missing hierarchy builder**

Run: `python3 -m pytest tests/unit/planner/test_hierarchy.py tests/integration/test_plan_gateway_allreduce.py -q`

Expected: collection fails.

- [ ] **Step 4: Implement manual-first and automatic hierarchy selection**

If a manual plan exists, validate and use it; otherwise, when hierarchy is enabled, derive gateway templates from actual topology groups. Local ranks are sorted numerically. Add edges only for exact logical interfaces, and allow independent stages to remain unordered so Phase 03 can pipeline ready slices.

- [ ] **Step 5: Run Phase 02 regression and coverage**

Run: `python3 -m pytest -m phase02 --cov=vericcl.topology --cov=vericcl.planner --cov-report=term-missing -q`

Expected: all Phase 02 tests pass and new modules reach at least 90% line coverage.

Run: `rg -n '[\p{Han}]' vericcl/topology vericcl/planner tests/unit/topology tests/unit/planner -g '*.py'`

Expected: no output.
