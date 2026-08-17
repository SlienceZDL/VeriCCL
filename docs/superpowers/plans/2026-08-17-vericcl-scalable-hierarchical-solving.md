# VeriCCL 可扩展分层求解实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在保持真实 slice 粒度、完整 atom 路径、集合通信语义、禁用项和全局资源约束正确的前提下，使默认求解规模由真实 slice 数量转为由非同构局部通信模式数量决定，并使分层策略与模型规模报告真实反映实际执行路径。

**Architecture:** Planner 用显式 `PlanningMode` 记录直接、手动和网关分层计划，并为 AllGather 构造真实多 rail 网关 PlanDAG。求解器把 PlanNode 拆为单棵语义树或单条语义链的 `RoutingUnit`，仅对精确同构类的代表 unit 建立路由 MILP；随后将代表路径映射到全部真实 slice，并在完整 TransferDAG 上统一分配 channel、共享资源 slot 和起止时间。默认路径保留模板组合限制并禁止声明全局最优；`require_proven_optimal=true` 明确回退现有完整时间 MILP。

**Tech Stack:** Python 3.10+、immutable dataclasses、Gurobi、`concurrent.futures`、pytest、现有 VeriCCL Planner/Solver/Composer/Verifier/XML 管线。

## Global Constraints

- 设计依据为 `docs/superpowers/specs/2026-08-17-vericcl-scalable-hierarchical-solving-design.md` 和 `Vericcl-work-document.md`；若实现细节发生冲突，以两份文档中更严格的语义与资源约束为准。
- 不改变 `slice_id = source_rank * N + logical_slice_index`、atom 外部结构、AggregateState 语义、MSCCL buffer offset 或 XML 输入格式。
- 输入 JSON 不增加必填字段；旧报告和旧缓存数据必须可读取，新字段使用显式默认值。
- 生产代码、测试代码、JSON 字段、诊断字符串和 XML 中不得出现中文字符；本实施计划和其他设计文档可使用中文。
- 默认求解路径只复用路由结构，不复用代表实例的 channel、resource slot、`st_time` 或 `ed_time`。
- 每个最终候选仍必须通过现有语义、依赖、资源、BDD、动态模拟、BufferPlan、XML 和死锁检查；本计划不放宽任何验证条件。
- 不使用未经证明的 channel 剪枝。第一版仍完整搜索 `K=1..K_max`，并保持 latency、throughput 和 auto 的现有对外语义。
- 无法证明精确同构时必须拆分模板；成员映射失败时仅回退该成员，不得输出近似映射。
- 默认模板路径始终包含 `template_route_composition` 限制；多 PlanNode 合成时同时保留 `independent_node_composition`。受限候选不得设置 `proven_optimal=true`。
- `require_proven_optimal=true` 只调用保留的完整时间 MILP；超时或没有严格最优解时按现有契约失败，不得用模板候选代替证明。
- 每项生产代码修改前先添加失败测试；完成后运行任务级测试并独立提交。不得顺带修改用户当前未提交的 `README.zh-CN.md`、`docs/figures/`、`docs/vericcl-paper-story.md` 或 `exp/`。

---

### Task 1: PlanningMode、规划原因与真实策略报告

**Files:**
- Modify: `vericcl/planner/model.py`
- Modify: `vericcl/planner/direct.py`
- Modify: `vericcl/planner/hierarchy.py`
- Modify: `vericcl/planner/build.py`
- Modify: `vericcl/planner/__init__.py`
- Modify: `vericcl/solver/cache.py`
- Modify: `vericcl/workflow.py`
- Modify: `vericcl/artifacts/reports.py`
- Test: `tests/unit/planner/test_direct.py`
- Test: `tests/unit/planner/test_hierarchy.py`
- Test: `tests/unit/solver/test_cache.py`
- Test: `tests/unit/artifacts/test_reports.py`
- Test: `tests/integration/test_workflow_artifacts.py`

**Interfaces:**

```python
class PlanningMode(str, Enum):
    DIRECT = "direct"
    MANUAL = "manual"
    GATEWAY_ALLREDUCE = "gateway_allreduce"
    GATEWAY_ALLGATHER = "gateway_allgather"


@dataclass(frozen=True)
class PlanDAG:
    ...
    planning_mode: PlanningMode = PlanningMode.DIRECT
    planning_reason: str = "direct_request"


def _applied_strategies(inputs: ResolvedInput, plan: PlanDAG) -> dict:
    ...
```

- [ ] **Step 1: 添加规划模式与报告行为的失败测试**

在 `tests/unit/planner/test_direct.py` 断言直接计划为 `PlanningMode.DIRECT`；在 `tests/unit/planner/test_hierarchy.py` 断言手动和网关 AllReduce 分别设置 `MANUAL`、`GATEWAY_ALLREDUCE`。在 `tests/unit/artifacts/test_reports.py` 和 `tests/integration/test_workflow_artifacts.py` 构造“请求 hierarchy 但实际直接计划”的场景，断言：

```python
assert report["requested_strategies"]["hierarchy"] is True
assert report["applied_strategies"]["hierarchy"] is False
assert report["hierarchy_plan"]["planning_mode"] == "direct"
assert report["hierarchy_plan"]["planning_reason"] == "no_eligible_gateway_domain"
```

同时在 `tests/unit/solver/test_cache.py` 断言仅改变 `planning_mode` 或 `planning_reason` 会改变 cache key。

- [ ] **Step 2: 运行测试并确认失败原因**

Run:

```bash
.venv/bin/python -m pytest \
  tests/unit/planner/test_direct.py \
  tests/unit/planner/test_hierarchy.py \
  tests/unit/solver/test_cache.py \
  tests/unit/artifacts/test_reports.py \
  tests/integration/test_workflow_artifacts.py -q
```

Expected: 因 `PlanningMode`、`planning_reason` 或新的 `_applied_strategies` 契约不存在而失败。

- [ ] **Step 3: 实现不可变规划模式并更新所有 PlanDAG 构造点**

在 `vericcl/planner/model.py` 中验证枚举类型和非空英文原因；为兼容直接构造 `PlanDAG` 的旧测试提供上述默认值。显式更新 `direct.py`、`hierarchy.py` 内所有生产构造点，使自动回退记录可判定原因，而不是依靠节点数量推断模式。

- [ ] **Step 4: 让缓存、workflow 和报告读取实际 PlanDAG**

在 `vericcl/solver/cache.py::_plan` 序列化 `planning_mode.value` 与 `planning_reason`。将 `workflow._applied_strategies` 改为同时接收 `inputs` 和 `plan`：`hierarchy` 由 `plan.planning_mode is not PlanningMode.DIRECT` 推导，其余策略继续表示实际启用状态。`_hierarchy_plan` 输出模式和原因；报告生成器不得再用请求值覆盖实际值。

- [ ] **Step 5: 运行任务测试并检查英文代码约束**

Run:

```bash
.venv/bin/python -m pytest \
  tests/unit/planner/test_direct.py \
  tests/unit/planner/test_hierarchy.py \
  tests/unit/solver/test_cache.py \
  tests/unit/artifacts/test_reports.py \
  tests/integration/test_workflow_artifacts.py -q
rg -n '[一-龥]' vericcl tests || true
```

Expected: 测试全部通过，`rg` 不显示本任务新增的生产或测试代码中文字符。

- [ ] **Step 6: 提交规划元数据修改**

```bash
git add vericcl/planner/model.py vericcl/planner/direct.py \
  vericcl/planner/hierarchy.py vericcl/planner/build.py \
  vericcl/planner/__init__.py vericcl/solver/cache.py \
  vericcl/workflow.py vericcl/artifacts/reports.py \
  tests/unit/planner/test_direct.py tests/unit/planner/test_hierarchy.py \
  tests/unit/solver/test_cache.py tests/unit/artifacts/test_reports.py \
  tests/integration/test_workflow_artifacts.py
git commit -m "feat: record effective planning mode"
```

---

### Task 2: 真实网关域上的多 rail AllGather 分层计划

**Files:**
- Modify: `vericcl/planner/groups.py`
- Modify: `vericcl/planner/hierarchy.py`
- Modify: `vericcl/planner/build.py`
- Modify: `vericcl/topology/isomorphism.py`
- Test: `tests/unit/planner/test_groups.py`
- Test: `tests/unit/planner/test_hierarchy.py`
- Create: `tests/integration/test_plan_gateway_allgather.py`
- Modify: `tests/property/test_plan_interfaces.py`

**Interfaces:**

```python
def build_gateway_allgather_plan(
    inputs: ResolvedInput,
    topology: Topology,
    groups: CommunicationGroups,
) -> PlanDAG:
    ...
```

稳定节点 ID 使用：

```text
local-gather-node-{node_id}-rail-{rail_index}
gateway-allgather-rail-{rail_index}
local-allgather-node-{node_id}-rail-{rail_index}
```

- [ ] **Step 1: 添加单网关、多网关和无效网关域失败测试**

覆盖以下断言：

1. 2 节点、每节点 1 网关生成 Gather → gateway AllGather → local AllGather 三阶段 DAG。
2. 2 节点、每节点 4 个对应网关生成 4 条 rail；每个 slice 仅出现在 `slice_id % 4` 对应 rail。
3. 只有 Rank 0 和 Rank 4 连接 NIC 时，不生成 `[1, 5]` 等虚拟跨节点组。
4. 网关数量不一致、缺少双向逻辑链路、未覆盖全部节点或局部域不精确同构时回退直接计划，并记录 `no_eligible_gateway_domain` 或具体稳定英文原因。
5. 每个 PlanEdge 接口都由 producer 产生、被 consumer 需要，最终输出等于直接 AllGather 语义。

- [ ] **Step 2: 运行失败测试**

Run:

```bash
.venv/bin/python -m pytest \
  tests/unit/planner/test_groups.py \
  tests/unit/planner/test_hierarchy.py \
  tests/integration/test_plan_gateway_allgather.py \
  tests/property/test_plan_interfaces.py -q
```

Expected: 因 AllGather 网关计划构造器和 rail 分配不存在而失败。

- [ ] **Step 3: 收紧通信组发现与网关资格验证**

复用 `discover_communication_groups` 的节点内域与按数字顺序建立的对应网关组，但只接受拓扑中真实存在的双向逻辑连通关系。使用 `exact_domain_signature` 比较每个对应局部域的有向链路、性能曲线、channel 上限和共享资源结构；不得从节点索引推导不存在的逻辑链路。

- [ ] **Step 4: 构造三阶段 AllGather PlanDAG**

对每个 rail：局部 Gather 把本节点属于该 rail 的原始 slice 送到对应网关；跨节点 AllGather 只交换该 rail 的 slice；局部 AllGather 将网关持有的全局 slice 分发到本节点所有 Rank。阶段接口始终使用全局 slice ID；PlanEdge 只连接真实 producer-consumer 值，不建立整阶段 barrier。

每个 PlanNode 的 `allowed_links` 只包含其通信组内部真实有向链路，`shared_resource_ids` 只包含这些链路实际引用的资源。成功计划设置 `GATEWAY_ALLGATHER` 和 `eligible_gateway_domain`；失败时 `build_plan` 返回带原因的直接计划。

- [ ] **Step 5: 运行规划与属性测试**

Run:

```bash
.venv/bin/python -m pytest \
  tests/unit/planner/test_groups.py \
  tests/unit/planner/test_hierarchy.py \
  tests/integration/test_plan_gateway_allgather.py \
  tests/property/test_plan_interfaces.py -q
```

Expected: 全部通过。

- [ ] **Step 6: 提交 AllGather 分层计划**

```bash
git add vericcl/planner/groups.py vericcl/planner/hierarchy.py \
  vericcl/planner/build.py vericcl/topology/isomorphism.py \
  tests/unit/planner/test_groups.py tests/unit/planner/test_hierarchy.py \
  tests/integration/test_plan_gateway_allgather.py \
  tests/property/test_plan_interfaces.py
git commit -m "feat: plan hierarchical gateway allgather"
```

---

### Task 3: RoutingUnit 拆分与精确 SolverTemplate 等价类

**Files:**
- Create: `vericcl/solver/templates.py`
- Modify: `vericcl/solver/demands.py`
- Modify: `vericcl/solver/__init__.py`
- Create: `tests/unit/solver/test_templates.py`
- Modify: `tests/unit/solver/test_demands.py`
- Create: `tests/property/test_template_isomorphism.py`

**Interfaces:**

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

- [ ] **Step 1: 添加拆分与精确复用失败测试**

在 `tests/unit/solver/test_templates.py` 覆盖：

- 8 Rank、每 Rank 128 个 slice 的直接 AllGather 先拆成 1024 个 unit，但只形成每个 root 一个模板，共 8 个模板。
- 同一 root 下仅逻辑位置不同的 128 个 unit 使用恒等 Rank 映射。
- 一个 slice 专属 `ForbiddenTransfer` 只把受影响 unit 拆为独立模板。
- 链路方向、`max_channels`、性能曲线、共享资源成员、root 角色、contributors 或 reduction-dual 标记任一不同均禁止合并。
- Gather、Scatter、AllToAll 按不可共享物理前缀的语义链拆分；Broadcast、AllGather 和 reduction-dual 按语义树拆分。

在属性测试中对精确 Rank 重编号的两个通信域生成映射，断言映射前后的 demand、候选边和禁用项逐项相等；修改任一资源后必须产生两个模板。

- [ ] **Step 2: 运行失败测试**

Run:

```bash
.venv/bin/python -m pytest \
  tests/unit/solver/test_demands.py \
  tests/unit/solver/test_templates.py \
  tests/property/test_template_isomorphism.py -q
```

Expected: 因模板模型与精确签名实现不存在而失败。

- [ ] **Step 3: 将现有 `_tree_key` 语义提升为 RoutingUnit 边界**

从 `vericcl/solver/milp.py` 中现有 `_tree_key` 提取稳定的语义 key 到 `demands.py` 或 `templates.py`。每个 unit 必须完整包含一棵树或一条链所需的全部 leaf demand，且 unit 拆分前后 demand ID 集合完全相同、无重复、无遗漏。

- [ ] **Step 4: 实现精确规范化签名和成员映射**

签名包括规划模式、stage、局部算子、通信组角色、root/owner、接口形状、相对 Rank 后的全部有向链路、性能、channel、共享资源、candidate paths、slice size、限制、禁用项和归约标记。第一版只允许：

1. 同一通信组与同一 root/owner 的逻辑位置平移；
2. 不同组按数字顺序建立并逐项验证的 Rank 双射。

`symmetry=false` 不关闭精确去重；它只关闭代表模型内额外的对称性搜索限制。

- [ ] **Step 5: 实现模板映射的二次合法性检查**

为每个 `TemplateMember` 验证映射后的每条路径属于成员 `allowed_links`、不命中永久或临时禁用项、contributors 和 logical position 映射可逆。失败成员从当前模板移除并成为独立模板，且模板成员总数保持等于 unit 总数。

- [ ] **Step 6: 运行模板测试并提交**

Run:

```bash
.venv/bin/python -m pytest \
  tests/unit/solver/test_demands.py \
  tests/unit/solver/test_templates.py \
  tests/property/test_template_isomorphism.py -q
```

Expected: 全部通过，128 slice 用例的模板数不随逻辑位置增长。

```bash
git add vericcl/solver/templates.py vericcl/solver/demands.py \
  vericcl/solver/__init__.py tests/unit/solver/test_templates.py \
  tests/unit/solver/test_demands.py \
  tests/property/test_template_isomorphism.py
git commit -m "feat: deduplicate exact routing templates"
```

---

### Task 4: RoutePattern 与代表路由 MILP

**Files:**
- Create: `vericcl/solver/routing.py`
- Create: `vericcl/solver/routing_milp.py`
- Modify: `vericcl/solver/gurobi_api.py`
- Modify: `vericcl/solver/lower_bounds.py`
- Modify: `vericcl/solver/__init__.py`
- Create: `tests/unit/solver/test_routing.py`
- Create: `tests/gurobi/test_routing_milp.py`
- Create: `tests/gurobi/test_routing_model_size.py`

**Interfaces:**

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
    selected_edges: tuple[tuple[int, int], ...]
    member_paths: tuple[tuple[str, tuple[tuple[int, int], ...]], ...]
    metrics: SolverMetrics
    model_stats: RoutingModelStats


def solve_route_milp(
    template: SolverTemplate,
    inputs: ResolvedInput,
    topology: Topology,
    channel_count: int,
    objective: ObjectiveMode,
    budget: ModelBudget,
    warm_start: RoutePattern | None = None,
) -> RoutePattern:
    ...
```

- [ ] **Step 1: 添加纯软件数据模型与 Gurobi 失败测试**

纯软件测试验证模型不可变性、路径连续性、稳定序列化和 `AUTO` 被拒绝。Gurobi 测试使用小型 Broadcast 与 reduction-dual unit，断言：

- 每个 required leaf 可达且非 root 节点只有一个父节点；
- 流守恒、禁用边、方向和 `allowed_links` 被严格满足；
- latency 目标使用无竞争路径时间；throughput 目标使用有向链路与共享资源归一化负载；
- 结果包含 incumbent、bound、gap、求解状态和模型结构计数。

- [ ] **Step 2: 添加模型规模不随 slice 数增长的失败测试**

分别为 8、16、64、128 个仅逻辑位置不同的 slice 建立模板，取一个代表模板和固定 `K=4`；断言 `variable_count`、`constraint_count`、`general_constraint_count` 完全相同。另断言模型中不存在 channel assignment、完整 `st_time/ed_time` 或任意两真实 slice 操作间的析取顺序变量。

- [ ] **Step 3: 运行失败测试**

Run:

```bash
.venv/bin/python -m pytest tests/unit/solver/test_routing.py -q
.venv/bin/python -m pytest \
  tests/gurobi/test_routing_milp.py \
  tests/gurobi/test_routing_model_size.py -q
```

Expected: 首个命令因新模块不存在而失败；Gurobi 可用时同样失败，不可用时按现有显式 skip 规则跳过。

- [ ] **Step 4: 实现路由变量和硬约束**

为代表 unit 建立 edge selection、共享树、单父节点、严格层级和链路流守恒约束。不得创建真实成员的重复变量。继续使用拓扑有向边、永久/临时禁用和 candidate paths。保留现有 `milp.py` 不变，作为完整时间 MILP 兼容后端。

- [ ] **Step 5: 实现两个路由目标和结构计数**

固定 K 下，从拓扑 `D(K)` 计算 edge duration。latency 最小化代表路径无竞争完成时间，再以 operation/hop 数为次级目标；throughput 最小化有向链路和共享资源的归一化负载，再最小化无竞争完成时间。模型创建后、优化前记录变量与约束数量，分别记录 build 和 optimize 时间。

- [ ] **Step 6: 提取 RoutePattern 并在 Python 中复核**

从 incumbent 提取代表树边与每个 leaf 的唯一根路径，复核路径连续、无环、root/leaf 正确、边合法且未命中禁用项。无 incumbent 返回类型化失败；TIME_LIMIT 有 incumbent 时允许返回 pattern，但 `proven_optimal` 语义不在此层设置。

- [ ] **Step 7: 运行路由模型测试并提交**

Run:

```bash
.venv/bin/python -m pytest tests/unit/solver/test_routing.py -q
.venv/bin/python -m pytest \
  tests/gurobi/test_routing_milp.py \
  tests/gurobi/test_routing_model_size.py -q
```

Expected: 纯软件测试通过；Gurobi 可用时全部通过，否则仅显示明确 skip。

```bash
git add vericcl/solver/routing.py vericcl/solver/routing_milp.py \
  vericcl/solver/gurobi_api.py vericcl/solver/lower_bounds.py \
  vericcl/solver/__init__.py tests/unit/solver/test_routing.py \
  tests/gurobi/test_routing_milp.py \
  tests/gurobi/test_routing_model_size.py
git commit -m "feat: solve representative routing models"
```

---

### Task 5: 路由模板实例化与真实 slice 语义重建

**Files:**
- Create: `vericcl/solver/instantiate.py`
- Modify: `vericcl/solver/scheduling.py`
- Modify: `vericcl/composer/dual.py`
- Create: `tests/unit/solver/test_instantiate.py`
- Modify: `tests/unit/composer/test_dual.py`
- Modify: `tests/property/test_ag_rs_duality.py`
- Modify: `tests/property/test_collective_semantics.py`

**Interfaces:**

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
    plan: PlanDAG,
    templates: tuple[SolverTemplate, ...],
    patterns: Mapping[str, RoutePattern],
    inputs: ResolvedInput,
    topology: Topology,
) -> InstantiationResult:
    ...
```

实例化结果为每个 PlanNode 的 provisional `Schedule`，包含完整真实 transfer 和语义依赖，但其 channel/resource slot/time 仅为待全局调度的初始值，metadata 明确设置 `routing_only=True`。

- [ ] **Step 1: 添加 Broadcast、AllGather 和 reduction-dual 实例化失败测试**

覆盖：

- 代表 logical position 0 的路径映射到 position 1 后，使用新的全局 slice ID、member slice IDs 和稳定唯一 `transfer_id`。
- 每个 slice 的 atom path prefix 从唯一源 Rank 连续延伸到当前 Rank；映射不得复制代表 `st_time`、`ed_time`、channel 或 resource slot。
- 归约对偶按 AG 树反向产生 `r` 与最终 `rrc`，contributors 只聚合一次，同一源状态不参与多次 Reduce。
- 三个贡献到达同一 Rank 后的后续 send 依赖最终 `rrc`，而不是任一单独贡献。
- 一个映射后命中禁用项的成员被报告给独立求解回退，其他成员仍复用模板。

- [ ] **Step 2: 运行失败测试**

Run:

```bash
.venv/bin/python -m pytest \
  tests/unit/solver/test_instantiate.py \
  tests/unit/composer/test_dual.py \
  tests/property/test_ag_rs_duality.py \
  tests/property/test_collective_semantics.py -q
```

Expected: 因实例化模块不存在或现有 dual helper 不接受 route pattern 而失败。

- [ ] **Step 3: 从现有 MILP schedule 提取代码中复用纯语义重建逻辑**

把 transfer ID、path prefix、member contributors 和 reduction-dual 重建提取为不依赖 Gurobi 变量的纯函数。保留现有 `solve_milp` 行为；新函数同时服务旧后端和模板实例化，避免形成两套归约语义。

- [ ] **Step 4: 实例化每个 TemplateMember**

只通过 `rank_map`、`contributor_map`、`logical_position_map` 变换 RoutePattern 的拓扑结构；从成员 RoutingUnit 读取真实 demand ID、slice ID、stage 接口和禁用项。生成后再次验证每条边存在、方向正确、属于 PlanNode 域且没有禁用。失败成员返回稳定错误记录供 orchestrator 单独求解，禁止产生部分 schedule。

- [ ] **Step 5: 重建跨阶段语义元数据**

在 provisional schedule 中写入真实 `semantic_predecessors`、`path_prefixes`、`final_outputs` 和 AggregateState consumption metadata。时间字段使用确定性零基准与真实 edge duration，不将其视为最终时间；`resource_slots` 为空，channel 统一为 0，仅用于通过不可变数据模型构造。

- [ ] **Step 6: 运行实例化测试并提交**

Run:

```bash
.venv/bin/python -m pytest \
  tests/unit/solver/test_instantiate.py \
  tests/unit/composer/test_dual.py \
  tests/property/test_ag_rs_duality.py \
  tests/property/test_collective_semantics.py -q
```

Expected: 全部通过。

```bash
git add vericcl/solver/instantiate.py vericcl/solver/scheduling.py \
  vericcl/composer/dual.py tests/unit/solver/test_instantiate.py \
  tests/unit/composer/test_dual.py tests/property/test_ag_rs_duality.py \
  tests/property/test_collective_semantics.py
git commit -m "feat: instantiate routes for real slices"
```

---

### Task 6: 完整 TransferDAG 的确定性全局资源调度

**Files:**
- Create: `vericcl/solver/global_scheduler.py`
- Modify: `vericcl/solver/scheduling.py`
- Modify: `vericcl/composer/compose.py`
- Modify: `vericcl/composer/timing.py`
- Create: `tests/unit/solver/test_global_scheduler.py`
- Modify: `tests/unit/composer/test_compose.py`
- Modify: `tests/unit/composer/test_timing.py`
- Modify: `tests/property/test_simulator_resources.py`

**Interfaces:**

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

- [ ] **Step 1: 添加 lane、共享资源和依赖调度失败测试**

构造以下最小用例：

- Rank 0→1 与 1→0 为独立有向链路，可在相同时间运行。
- 相同有向链路不同 channel 可并行；相同 `LaneKey(src,dst,channel)` 区间不得重叠。
- 两条不同逻辑链路引用同一 NIC shared resource 时，同一 slot 不得重叠，不同 slot 可并行。
- 下游 transfer 只有在全部 semantic predecessors 完成后进入 ready set。
- 没有 stage barrier：stage 1 中已满足 slice 依赖的 transfer 可以与 stage 0 的其他 slice 并行。
- 输入 provisional 时间顺序被刻意打乱时，输出只由依赖、资源和稳定 tie-break 决定。

- [ ] **Step 2: 运行失败测试**

Run:

```bash
.venv/bin/python -m pytest \
  tests/unit/solver/test_global_scheduler.py \
  tests/unit/composer/test_compose.py \
  tests/unit/composer/test_timing.py \
  tests/property/test_simulator_resources.py -q
```

Expected: 因全局 scheduler 不存在，或现有 compose 仅保留局部 channel/slot 后 retime 而失败。

- [ ] **Step 3: 实现确定性最早结束时间列表调度**

从 `semantic_predecessors` 建立完整 DAG。对 ready transfer 枚举：

```text
channel = 0 .. min(K, directed_link.max_channels) - 1
resource_slot(resource_id) = 0 .. min(K, resource.max_channels) - 1
```

对该 transfer 涉及的少量 shared resource 取合法 slot 组合，计算 `start=max(semantic_ready, lane_ready, every_resource_slot_ready)` 与 `end=start+D(K)`，选择最早结束组合。并列依次比较 semantic ready time、route priority、stage、logical position、src、dst、transfer ID、channel 和 slot tuple，保证相同 seed 和输入得到字节级稳定 sidecar。

- [ ] **Step 4: 重建资源前驱、atom ready time 和最终时间**

每次分配后更新 lane/resource 可用时间，并将最近占用者加入调度前驱。全部 transfer 分配完成后，用扩展后的 `_retime` 统一重建 `predecessor_ids`、symbol ready time、`st_time` 和 `ed_time`。检测 DAG 环、无合法 channel/slot 或不完整 schedule 时抛出类型化错误，不返回部分结果。

- [ ] **Step 5: 将模板路径接入 Composer，同时保持旧 compose 兼容**

现有 `compose(plan, candidates)` 继续用于完整时间 MILP。新增 `compose_routes` 先执行现有跨 PlanNode 语义合成，再清除局部资源分配并调用全局 scheduler。两条路径最终产生相同 Schedule 数据模型，供后续验证和 XML lowering 使用。

- [ ] **Step 6: 运行调度、Composer 与资源属性测试**

Run:

```bash
.venv/bin/python -m pytest \
  tests/unit/solver/test_global_scheduler.py \
  tests/unit/composer/test_compose.py \
  tests/unit/composer/test_timing.py \
  tests/property/test_simulator_resources.py -q
```

Expected: 全部通过；每个 lane/resource slot 区间无重叠，反向链路与不同 channel 保持并行。

- [ ] **Step 7: 提交全局调度器**

```bash
git add vericcl/solver/global_scheduler.py vericcl/solver/scheduling.py \
  vericcl/composer/compose.py vericcl/composer/timing.py \
  tests/unit/solver/test_global_scheduler.py \
  tests/unit/composer/test_compose.py tests/unit/composer/test_timing.py \
  tests/property/test_simulator_resources.py
git commit -m "feat: schedule instantiated routes globally"
```

---

### Task 7: 模板模型并行搜索、构造式回退与最优性路径隔离

**Files:**
- Create: `vericcl/solver/template_search.py`
- Modify: `vericcl/solver/search.py`
- Modify: `vericcl/solver/constructive.py`
- Modify: `vericcl/solver/orchestrator.py`
- Modify: `vericcl/solver/model.py`
- Modify: `vericcl/solver/objectives.py`
- Modify: `vericcl/solver/lower_bounds.py`
- Test: `tests/unit/solver/test_search.py`
- Create: `tests/unit/solver/test_template_search.py`
- Modify: `tests/unit/solver/test_orchestrator.py`
- Modify: `tests/unit/solver/test_lower_bounds.py`
- Modify: `tests/gurobi/test_lower_bounds.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class SearchDiagnostics:
    requested_problem_count: int = 0
    routing_unit_count: int = 0
    template_count: int = 0
    template_member_count: int = 0
    route_model_count: int = 0
    fallback_member_model_count: int = 0
    search_model_count_total: int = 0
    route_model_build_time_s: float = 0.0
    route_model_optimize_time_s: float = 0.0
    template_expansion_time_s: float = 0.0
    global_scheduling_time_s: float = 0.0
    model_variables_max: int = 0
    model_constraints_max: int = 0
    model_general_constraints_max: int = 0


@dataclass(frozen=True)
class TemplateSearchResult:
    candidates: tuple[SolveCandidate, ...]
    diagnostics: SearchDiagnostics


def search_route_models(
    request: SolveRequest,
    problems: tuple[SolverProblem, ...],
    objective: ObjectiveMode,
    deadline: float,
) -> TemplateSearchResult:
    ...


@dataclass(frozen=True)
class SolveResult:
    ...
    diagnostics: SearchDiagnostics = field(default_factory=SearchDiagnostics)
```

- [ ] **Step 1: 添加工作队列、模型计数和预算失败测试**

使用 fake `solve_route_milp` 验证：

- 并行单位是 ready `(template, K, objective)`；worker 数为 `min(ready_models, max_parallel_models, cpu_count)`。
- `threads_per_model=max(1,min(requested,floor(cpu_count/J)))`，所有运行模型总线程不超过 CPU 数。
- 每个 K 必须完成所有模板后才能实例化并产生一个全局候选；缺少任一模板时该 K 不产生候选。
- `model_count` 只等于直接形成当前候选的模板模型数；每个 `TemplateSearchResult` 的 `search_model_count_total` 等于该 objective 全部实际启动模型数，`SolveResult` 再对实际执行的 objective 汇总，不能在每个候选中重复累计。
- 总预算到期后不再启动新模型，但保留已经完成且可形成完整 K 候选的结果。
- 路由模型无 incumbent 时，构造式后端只为代表 unit 生成 path；成员映射失败时只增加 `fallback_member_model_count`。

- [ ] **Step 2: 添加默认路径和严格最优路径的 orchestrator 失败测试**

通过 monkeypatch 断言：

```python
assert default_request_calls == ["template_route_pipeline"]
assert proven_request_calls == ["legacy_full_time_milp"]
```

默认候选必须包含 `template_route_composition` 且 `proven_optimal is False`。`require_proven_optimal=true` 时不得调用 `build_solver_templates`、`solve_route_milp` 或 `compose_routes`；现有完整 MILP 只有严格 OPTIMAL、无限制时才可证明最优。

- [ ] **Step 3: 添加 auto 与 multiplicity-aware lower bound 测试**

同一模板的 128 个成员必须按成员实际 bytes/operations 计入吞吐下界，而不是只计一个代表 unit；路由求解仍只启动一个代表模型。auto 先执行 latency 候选，再依据全局 schedule 和保守下界决定是否执行 throughput，保持现有选择规则和最终动态模拟选择边界。

- [ ] **Step 4: 运行失败测试**

Run:

```bash
.venv/bin/python -m pytest \
  tests/unit/solver/test_search.py \
  tests/unit/solver/test_template_search.py \
  tests/unit/solver/test_orchestrator.py \
  tests/unit/solver/test_lower_bounds.py -q
```

Expected: 因新搜索入口、诊断模型和 orchestrator 分支不存在而失败。

- [ ] **Step 5: 实现共享 deadline 的模板模型工作队列**

延续 `search.py` 的单调时钟、`ModelBudget` 和 lazy Gurobi error 处理，但以全部模板与 K 的笛卡尔工作项建立队列。新模型启动前重新计算剩余总预算；单模型预算为 `min(per_model_timeout, remaining)`。结果按 objective、K、template ID 稳定排序，线程池异常必须关联到具体工作项。

- [ ] **Step 6: 为每个 K 执行实例化和一次全局调度**

当该 K 的全部模板获得 RoutePattern 或合法构造式 pattern 后，调用 `instantiate_route_patterns` 和 `compose_routes` 生成一个全局候选；用全局 makespan、资源负载、operation count 和 hop count 重算目标值。限制集合加入 `template_route_composition`，多 PlanNode 再加入 `independent_node_composition`。

- [ ] **Step 7: 隔离默认和严格最优后端**

在 `orchestrator.solve` 中：

- `require_proven_optimal=false`：调用模板路径，保留构造式代表回退。
- `require_proven_optimal=true`：调用重命名为明确 legacy/full-time 语义的现有 `_solve_objective` 路径，不允许模板候选进入结果。

两条路径共用 cache、candidate ranking 和错误契约，但使用不同 model version/cache signature。

- [ ] **Step 8: 实现 multiplicity-aware lower bound 和 auto 汇总**

每个 template 的代表负载乘以映射后实际成员 multiplicity，并按真实 link/resource 映射累加，再求合法下界。不得简单对 per-node bound 取最大后忽略多个 slice 或多个并行域的共享资源负载。

- [ ] **Step 9: 运行搜索与 orchestrator 测试并提交**

Run:

```bash
.venv/bin/python -m pytest \
  tests/unit/solver/test_search.py \
  tests/unit/solver/test_template_search.py \
  tests/unit/solver/test_orchestrator.py \
  tests/unit/solver/test_lower_bounds.py -q
.venv/bin/python -m pytest tests/gurobi/test_lower_bounds.py -q
```

Expected: 纯软件测试全部通过；Gurobi 测试可用时通过，否则明确 skip。

```bash
git add vericcl/solver/template_search.py vericcl/solver/search.py \
  vericcl/solver/constructive.py vericcl/solver/orchestrator.py \
  vericcl/solver/model.py vericcl/solver/objectives.py \
  vericcl/solver/lower_bounds.py tests/unit/solver/test_search.py \
  tests/unit/solver/test_template_search.py \
  tests/unit/solver/test_orchestrator.py \
  tests/unit/solver/test_lower_bounds.py tests/gurobi/test_lower_bounds.py
git commit -m "feat: orchestrate scalable template search"
```

---

### Task 8: 缓存版本、结构诊断与报告契约

**Files:**
- Modify: `vericcl/solver/cache.py`
- Modify: `vericcl/solver/model.py`
- Modify: `vericcl/artifacts/reports.py`
- Modify: `vericcl/artifacts/summary.py`
- Modify: `vericcl/artifacts/writer.py`
- Modify: `vericcl/workflow.py`
- Modify: `vericcl/__main__.py`
- Modify: `tests/unit/solver/test_cache.py`
- Modify: `tests/unit/solver/test_model.py`
- Modify: `tests/unit/artifacts/test_reports.py`
- Modify: `tests/unit/artifacts/test_writer.py`
- Modify: `tests/integration/test_workflow_artifacts.py`
- Modify: `tests/integration/test_cli_end_to_end.py`

**Report fields:**

```text
planning_mode
requested_problem_count
routing_unit_count
template_count
template_member_count
route_model_count
fallback_member_model_count
search_model_count_total
route_model_build_time_s
route_model_optimize_time_s
template_expansion_time_s
global_scheduling_time_s
model_variables_max
model_constraints_max
model_general_constraints_max
```

- [ ] **Step 1: 添加旧数据兼容和新诊断失败测试**

覆盖：

- 旧 `SolverMetrics`/报告缺少新字段时全部读取为零，不报错。
- 候选级 `model_count`、run 级 `search_model_count_total` 和 `route_model_count` 各自保持定义，不互相复制。
- 报告分别显示 route build、route optimize、template expansion、global scheduling、verification 和总墙钟时间。
- 同一输入在 route model version、global scheduler version、template exact signature 或成员映射摘要改变时 cache key 改变。
- 旧完整时间 MILP cache entry 不会命中新 route template request。

- [ ] **Step 2: 运行失败测试**

Run:

```bash
.venv/bin/python -m pytest \
  tests/unit/solver/test_cache.py \
  tests/unit/solver/test_model.py \
  tests/unit/artifacts/test_reports.py \
  tests/unit/artifacts/test_writer.py \
  tests/integration/test_workflow_artifacts.py \
  tests/integration/test_cli_end_to_end.py -q
```

Expected: 因诊断字段和新 cache signature 不存在而失败。

- [ ] **Step 3: 扩展不可变诊断数据并保持默认值兼容**

将结构计数和分阶段耗时集中在 `SearchDiagnostics`，由 `SolveResult` 持有 run 级诊断；`SolverMetrics.model_count` 继续表示候选直接贡献模型数。所有新增字段置于 dataclass 有默认值的尾部，并执行非负、有限数值验证。

- [ ] **Step 4: 版本化缓存和稳定序列化**

cache payload 加入 `planning_mode`、template exact signature、成员映射摘要、`route_model_version="1"`、`global_scheduler_version="1"` 和后端类型。映射和集合排序后再 hash，保证重启后 key 稳定。保留旧 cache 读取代码，但后端/version 不匹配时视为 miss。

- [ ] **Step 5: 将真实结构计数与耗时写入候选报告和 run summary**

workflow 从 `SolveResult.diagnostics` 传递 run 级字段；候选报告从 metrics/diagnostics 读取候选直接贡献信息。不得由 slice_count、K_max 或请求布尔值推算“看起来合理”的数字。XML 所用调优策略继续写入验证报告，且新增模板策略名称使用英文。

- [ ] **Step 6: 运行报告、CLI 和 cache 测试并提交**

Run:

```bash
.venv/bin/python -m pytest \
  tests/unit/solver/test_cache.py \
  tests/unit/solver/test_model.py \
  tests/unit/artifacts/test_reports.py \
  tests/unit/artifacts/test_writer.py \
  tests/integration/test_workflow_artifacts.py \
  tests/integration/test_cli_end_to_end.py -q
```

Expected: 全部通过，新旧报告均能读取。

```bash
git add vericcl/solver/cache.py vericcl/solver/model.py \
  vericcl/artifacts/reports.py vericcl/artifacts/summary.py \
  vericcl/artifacts/writer.py vericcl/workflow.py vericcl/__main__.py \
  tests/unit/solver/test_cache.py tests/unit/solver/test_model.py \
  tests/unit/artifacts/test_reports.py tests/unit/artifacts/test_writer.py \
  tests/integration/test_workflow_artifacts.py \
  tests/integration/test_cli_end_to_end.py
git commit -m "feat: report template solver diagnostics"
```

---

### Task 9: 全链路正确性、结构性能与回归验收

**Files:**
- Create: `tests/integration/test_scalable_template_solving.py`
- Create: `tests/e2e/test_hierarchical_allgather.py`
- Modify: `tests/e2e/test_hierarchical_allreduce.py`
- Modify: `tests/e2e/test_reproducibility.py`
- Modify: `tests/e2e/test_candidate_xml.py`
- Modify: `tests/unit/verification/test_semantics.py`
- Modify: `tests/unit/verification/test_constraints.py`
- Modify: `tests/unit/verification/test_bdd_flow.py`
- Modify: `docs/final-validation-report.md`

- [ ] **Step 1: 添加结构性能失败测试**

不以固定秒数作为断言，改为检查求解对象规模：

- 8 Rank、128 slice 直接 AllGather 每个固定 root 只有一个模板，固定 objective 下每个 K 启动 8 个 route model，而不是 1024 个。
- 具备网关域的分层 AllGather 模型数只由非同构局部域与 rail 决定。
- 分层 AllReduce 在 8、16、64、128 slice 时代表模型的变量、普通约束和广义约束计数保持不变。
- 默认候选报告含 `template_route_composition`；严格最优请求只出现 full-time backend。

- [ ] **Step 2: 添加完整语义和 XML 回读失败测试**

对 2 节点网关 AllGather 与 AllReduce 的小规模实例执行：plan → solve → compose → verify → buffer plan → XML lowering → XML 回读。断言最终 contributors、output offsets、`r/rrc`、`depid/deps`、单向 TB、channel 和死锁检查全部正确。BDD 必须读取实例化后的真实 flow，不得读取 template ID 代替 slice/Rank/time。

- [ ] **Step 3: 运行新验收测试并修复最小集成问题**

Run:

```bash
.venv/bin/python -m pytest \
  tests/integration/test_scalable_template_solving.py \
  tests/e2e/test_hierarchical_allgather.py \
  tests/e2e/test_hierarchical_allreduce.py \
  tests/e2e/test_reproducibility.py \
  tests/e2e/test_candidate_xml.py \
  tests/unit/verification/test_semantics.py \
  tests/unit/verification/test_constraints.py \
  tests/unit/verification/test_bdd_flow.py -q
```

Expected: 全部通过；Gurobi 不可用的精确后端用例遵循已有 skip 规则，构造式默认路径仍执行。

- [ ] **Step 4: 运行所有非硬件测试**

Run:

```bash
.venv/bin/python -m pytest tests/unit tests/property tests/integration tests/e2e -q
.venv/bin/python -m pytest tests/gurobi -q
```

Expected: unit/property/integration/e2e 全部通过；Gurobi 可用时全部通过，否则只有带明确原因的 skip。不得运行 `tests/hardware`，因为该目录需要目标 GPU/NIC 环境和在线工具链。

- [ ] **Step 5: 执行静态检查和输入规模结构检查**

Run:

```bash
git diff --check
rg -n '[一-龥]' vericcl tests || true
.venv/bin/python -m vericcl --help
.venv/bin/python -m pytest tests/integration/test_documented_commands.py -q
```

Expected: 无 whitespace error；生产与测试代码没有新增中文；CLI 可启动；README 现有命令测试继续通过。

使用 `exp/topo/v100-n2g4.json` 与 `exp/sketch/v100-n2g4/ag/ag-1g.json` 只运行规划和模板构建的结构检查，不启动长时间全量 Gurobi 求解。记录 `requested_problem_count`、`routing_unit_count`、`template_count` 和单代表模型结构计数，确认模板数与 1 GiB 对应 slice 数解耦。

- [ ] **Step 6: 更新最终验证报告**

在 `docs/final-validation-report.md` 记录：测试命令、通过/skip 数、Gurobi/license 环境、结构计数、默认路径的最优性限制、未运行硬件测试的原因。只记录本次实际获得的数据，不写估算墙钟性能。

- [ ] **Step 7: 进行提交前代码审查**

使用 `superpowers:requesting-code-review` 检查：设计覆盖、语义不回退、模板映射精确性、资源调度无重叠、严格最优路径隔离、报告真实性和用户未提交文件未被纳入。处理审查意见后重新执行 Step 4 与 Step 5。

- [ ] **Step 8: 提交验收测试与验证报告**

```bash
git add tests/integration/test_scalable_template_solving.py \
  tests/e2e/test_hierarchical_allgather.py \
  tests/e2e/test_hierarchical_allreduce.py \
  tests/e2e/test_reproducibility.py tests/e2e/test_candidate_xml.py \
  tests/unit/verification/test_semantics.py \
  tests/unit/verification/test_constraints.py \
  tests/unit/verification/test_bdd_flow.py \
  docs/final-validation-report.md
git commit -m "test: validate scalable hierarchical solving"
```

---

## Final Verification Gate

- [ ] 从干净测试进程重新运行：

```bash
.venv/bin/python -m pytest tests/unit tests/property tests/integration tests/e2e -q
.venv/bin/python -m pytest tests/gurobi -q
git diff --check
git status --short --branch
```

- [ ] 确认 `README.zh-CN.md`、`docs/figures/`、`docs/vericcl-paper-story.md` 和 `exp/` 的用户改动未被本计划各提交纳入。
- [ ] 确认默认模板候选从未声明 `proven_optimal=true`，且 `require_proven_optimal=true` 从未调用模板后端。
- [ ] 确认 128 slice 结构测试没有构造真实 slice 两两间的 lane/resource interval disjunction。
- [ ] 确认所有最终候选仍经过现有完整离线验证与 XML 回读，而不是只验证 RoutePattern。
- [ ] 使用 `superpowers:verification-before-completion` 核验最新测试输出后，才可报告实现完成。
- [ ] 使用 `superpowers:finishing-a-development-branch` 向用户提供合并、推送或保留分支选项；未经用户选择不得自行合并或删除分支。
