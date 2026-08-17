# VeriCCL 可扩展分层求解设计

**日期：** 2026-08-17

**目标：** 在保持真实 slice 粒度、完整 atom 路径、精确集合通信语义和全局资源约束的前提下，将路由求解规模从“随 slice 数增长”改为“随非同构局部通信模式数量增长”，并修复分层策略报告与实际计划不一致的问题。

## 1. 问题与证据

当前实现只复用了分层 PlanDAG 的组织形式，没有实现 SyCCL 式同构子需求复用。

1. 自动分层仅覆盖 AllReduce。AllGather 即使请求 `hierarchy=true`，仍生成全局直接 Broadcast 节点；报告却直接复制输入布尔值并声称层次化策略已经应用。
2. 直接 AllGather 为每个 `(source_rank, logical_address)` 生成独立 PlanNode。8 Rank、每 Rank 128 个 slice 时共有 1024 个 PlanNode。
3. 每个 PlanNode 对 `K=1..32` 分别构造 MILP；`auto` 还可能完整执行 latency 和 throughput 两轮。上述 AllGather 最多触发 65536 次局部 MILP。
4. 分层 AllReduce 虽只有少量 PlanNode，但每个 PlanNode 把全部 slice 树放入同一 MILP。lane 和共享资源上的操作两两建立区间不重叠析取约束，模型规模随 slice 数近似二次增长。
5. 一个 8 Rank、单 slice Broadcast 在 `K=32` 时已经产生 58290 个变量和 53764 个广义约束；分层 AllReduce 的一个节点内 Reduce 在 128 个 slice、`K=4` 时产生 889602 个变量和 882432 个广义约束。

该问题不是单纯的超时参数、Gurobi 许可证或线程数问题，而是求解对象展开方式与模型边界不正确。

## 2. 方案比较

### 方案 A：只调整参数

降低 `max_channels`、增大 `slice_size_bytes`、关闭 MILP或提高 timeout 可以减少个别实验的运行时间，但会缩小搜索空间、改变用户指定粒度或延长失败等待，不能修复重复求解和二次建模。

### 方案 B：继续优化单体精确 MILP

可通过对称性破缺、延迟生成冲突约束和 Gurobi 参数调优降低常数开销，并保留当前模型内的全局最优性范围。但真实 slice、可选路径、channel、时间顺序和共享资源共同构成大规模离散调度问题，模型仍随 slice 数快速增长，无法达到 SyCCL 级求解规模。

### 方案 C：代表路由求解与全局调度分离

先对精确同构的局部通信问题只求解代表路由，再将路由映射到全部真实 slice，最后统一分配 channel、共享资源 slot 和起止时间。BDD 与局部调优继续负责修复模板复用后出现的拥塞机会。

采用方案 C。它与现有工作文档中的“同构通信组只求解一次”“Composer 事件驱动合成”“BDD 仅输出调优机会”和“局部 MILP 修复”一致。该方案保证语义与约束正确，但不把模板路由和启发式全局调度声明为全局最优。

## 3. 设计边界

本次修改包括：

- 修复请求策略、实际策略和 PlanDAG 类型的报告。
- 为具备真实网关通信域的 AllGather 增加自动分层计划。
- 建立局部求解问题的精确等价类和代表映射。
- 将 MILP 的职责缩小为代表通信模式的路径/树选择。
- 将代表路由展开为全部真实 slice，并统一执行确定性全局调度。
- 保留 `K=1..K_max` 的外层离散搜索和 latency/throughput/auto 语义。
- 输出实际模型数、等价类数、模型规模和各阶段耗时。

本次不修改：

- `slice_id = source_rank * N + logical_slice_index` 和 atom 外部定义。
- AggregateState、归约状态消费、贡献集合和 AG 反向生成归约的语义。
- BDD 关系定义、在线校准协议、MSCCL buffer/XML 地址规则和运行时限制。
- 用户输入的 `total_size_bytes`、`slice_size_bytes`、禁用项和手动分层。
- 在线校准对 `k=1..K_effective` 的完整测试要求。

## 4. PlanDAG 与实际策略

`PlanDAG`增加稳定的规划模式和原因字段：

```python
class PlanningMode(str, Enum):
    DIRECT = "direct"
    MANUAL = "manual"
    GATEWAY_ALLREDUCE = "gateway_allreduce"
    GATEWAY_ALLGATHER = "gateway_allgather"


planning_mode: PlanningMode
planning_reason: str
```

允许值为：

- `direct`
- `manual`
- `gateway_allreduce`
- `gateway_allgather`

报告中的 `requested_strategies`继续记录输入；`applied_strategies.hierarchy`由 `planning_mode != "direct"` 推导，不再复制请求值。缓存签名和 `hierarchy_plan`必须包含 `planning_mode`。

如果请求了自动分层，但拓扑不存在覆盖全部节点的真实网关通信组，则使用直接计划，并记录：

```text
hierarchy=false
hierarchy_reason=no_eligible_gateway_domain
```

不得生成虚拟跨节点通信组。

## 5. AllGather 自动分层

当拓扑包含每个节点的局部通信组和覆盖全部节点的真实网关组时，全局 AllGather 生成以下 PlanDAG：

1. 每个节点执行局部 Gather，将本节点的全部原始 slice 收集到该节点网关。
2. 网关组执行 AllGather，使每个网关获得全部节点的原始 slice。
3. 每个节点从本地网关执行若干局部 Broadcast，将全部原始 slice 分发到节点内所有 Rank。

如果每个节点具有 `G>1` 个对应网关，则使用全部 `G` 个真实网关组。每个原始 slice 按 `slice_id mod G`分配到唯一 rail：节点内 Gather 将该 slice 送到对应位置的网关，网关间 AllGather 只处理该 rail 的 slice，最后由同一网关在节点内分发。该分配不复制 slice，并允许不同 rail 并行。`G=1`时退化为单网关模板。

阶段接口始终使用全局 slice ID 和最终 AllGather 输出 offset `source_rank * N + logical_slice_index`。阶段之间只建立状态因果依赖，不加入统一 barrier。每个局部节点的 `allowed_links`和`shared_resource_ids`只包含该通信域中的真实拓扑对象。

自动分层仅在以下条件全部满足时启用：

- 用户请求 `hierarchy=true`。
- 未提供 `manual_hierarchy`。
- 每个节点的网关数量一致。
- 网关组由真实双向逻辑连通分量构成并覆盖全部节点。
- 对应局部通信域精确同构。

条件不满足时回退直接 AllGather，不改变最终算子语义，并在报告中说明原因。

## 6. 精确问题等价类

Planner 的一个 PlanNode 可以包含多个逻辑位置。求解器先将它拆成只包含一棵语义树或一条语义链的 `RoutingUnit`，再对这些 unit 建立等价类。新增不可变模型：

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


@dataclass(frozen=True)
class RoutePattern:
    template_id: str
    channel_count: int
    objective_mode: ObjectiveMode
    selected_edges: tuple[tuple[int, int], ...]
    member_paths: tuple[tuple[str, tuple[tuple[int, int], ...]], ...]
    metrics: SolverMetrics
```

Broadcast、AllGather 和归约对偶节点按当前 `_tree_key` 的语义等价物拆分，每个 unit 包含同一 root、logical position、contributors 和 reduction-dual 标记的全部叶需求。Gather、Scatter 和 AllToAll 按不可再共享物理前缀的独立语义链拆分。拆分不改变 PlanNode 的阶段接口。

等价类只接受可验证的精确映射。签名至少包含：

- `planning_mode`、stage ID 和局部算子。
- 通信组中的 Rank 角色、root/owner 角色和局部接口形状。
- 相对 Rank 编号后的全部单向链路、性能曲线、channel 上限和共享资源结构。
- demand 的根、叶、contributors 关系、归约对偶标记和 slice 大小。
- 最短路、对称性、batching 等搜索限制。
- 映射后的用户禁用项与候选级临时禁用项。

第一版执行两级复用：

1. 同一通信组、同一 root/owner、仅逻辑 slice 位置不同的问题使用恒等 Rank 映射复用。这一步将每 Rank 的重复 slice 问题压缩为一个代表问题。
2. 不同通信组只有在按 Rank 数字顺序映射后，全部链路、方向、资源、性能和角色逐项相等时才复用。

不同 root 之间只有存在显式且逐项验证通过的 Rank 双射时才能合并；否则保留为不同模板。不得依赖近似拓扑标签或未经验证的对称性。

任何 slice 专属禁用项映射不一致时，相关 slice 自动进入独立等价类。模板实例化后再次检查所有成员的合法链路和禁用项；检查失败时只对该成员执行独立求解，不允许输出错误映射。

精确等价类复用属于求解去重，始终启用，不受用户的 `symmetry`布尔值控制。`symmetry=true`仍表示进一步限制代表路由模型的搜索空间；`batching=true`仍表示允许多个不同但兼容的 unit 共享一棵批量树。两者必须继续独立记录。

## 7. 代表路由 MILP

新增路由模型，将路径选择与全量时间调度分离。每个固定 `K`、objective 和 `SolverTemplate`对应一个路由模型。

模型保留：

- demand 流守恒。
- 同一 payload 的共享树边。
- 单父节点和严格递增树层级约束。
- 拓扑、方向、禁用项和共享资源成员关系。
- latency 的无竞争路径完成时间。
- throughput 的有向链路/共享资源归一化负载。
- 操作数和 hop 数次级目标。

模型移除：

- 每个真实 slice 的重复路径变量。
- channel slot 的逐操作二进制选择。
- 全量 `st_time/ed_time`变量。
- lane 和共享资源操作之间的两两区间析取约束。

路由模型输出 `RoutePattern`，其中包含代表树边、成员路径、目标值、bound、gap 和求解状态。它不直接输出最终 `Schedule`。

`K`仍影响保守传输开销和资源负载计算。第一版继续完整搜索 `K=1..K_max`，避免通过未经证明的 channel 剪枝遗漏合法候选。不同 K 和 objective 可并行，但总线程数与墙钟预算规则保持不变。

代表路由目标只负责产生候选。每个 K 的模板路由全部实例化并完成全局调度后，Orchestrator 使用全局调度的保守 makespan 和资源负载重新排序；验证阶段仍以动态事件模拟结果执行最终性能选择。

## 8. 模板实例化与全局调度

代表路由通过 `TemplateMember`映射为所有真实 slice 的物理传输。实例化必须重建：

- 全局 slice ID。
- `member_slice_ids`和 `semantic_contributors`。
- atom 的完整路径前缀。
- stage 接口和 producer-consumer 依赖。
- AggregateState 汇合与消费依赖。
- 稳定且唯一的 `transfer_id`。

只允许复用路径结构，禁止复制代表实例的 `st_time/ed_time`、channel 或共享资源 slot。

全局调度器以完整实例化 TransferDAG 为输入，在固定 K 下执行确定性最早完成时间列表调度：

1. 只将语义前驱全部完成的传输放入就绪集合。
2. 为每个就绪传输枚举合法 channel 和共享资源 slot 组合。
3. 选择最早结束的组合；并列时依次比较语义 ready time、原路由优先级、stage、逻辑位置、源 Rank、目标 Rank 和 transfer ID。
4. 使用固定 K 的保守 `D(K)`计算持续时间，并更新 lane 和共享资源 slot 的可用时间。
5. 重建 `predecessor_ids`、atom symbol ready time、`st_time`和`ed_time`。

调度结果必须满足同一 `LaneKey`和同一共享资源 slot 区间不重叠。不同有向方向和不同 channel 保持可并行。调度器不加入 stage barrier。

模板复用和确定性调度仍属于受限全局组合。候选继续记录 `independent_node_composition`，新增 `template_route_composition`；即使全部代表路由模型最优，也不得设置全局 `proven_optimal=true`。

默认 `require_proven_optimal=false`时使用模板路由路径。用户显式设置 `require_proven_optimal=true`时，Orchestrator 必须绕过模板组合并调用保留的完整时间 MILP；只有完整模型在未受限搜索空间中获得严格最优状态时才能返回结果。完整模型超时或无严格最优解时按现有契约失败，不得用模板候选代替最优性证明。

## 9. 与 BDD 和调优的关系

BDD 继续分析实例化后的完整真实 slice 调度，不分析抽象模板。若同一路由模板使大量 slice 集中到一条链路，BDD 应输出具体 flow 和候选替换 flow。

调优器首先重新分配受影响 slice 的现有候选路径；没有合法路径时，为受影响的模板或 slice 生成新 RoutePattern。只有分歧点后的后缀、相关依赖和时间闭包允许改变。局部修复失败不得触发所有 slice 的单体全局 MILP。

## 10. 预算、并行与缓存

并行单位从“单个 PlanNode 的不同 K”扩展为“已经就绪的 `(template, K, objective)`模型”。调度器仍满足：

```text
J = min(ready_models, max_parallel_models, cpu_count)
threads_per_model = max(1, min(12, floor(cpu_count / J)))
```

外层总预算和单模型预算定义不变。预算耗尽后停止启动新模型，并保留已完成的代表路由和构造式候选。

缓存签名新增：

- `planning_mode`
- template exact signature
- 完整成员映射摘要
- route model version
- global scheduler version

旧的完整时间 MILP 缓存不得作为新路由模板结果使用。

## 11. 报告和诊断

每个候选报告新增：

```text
planning_mode
requested_problem_count
template_count
template_member_count
route_model_count
fallback_member_model_count
route_model_build_time_s
route_model_optimize_time_s
template_expansion_time_s
global_scheduling_time_s
model_variables_max
model_constraints_max
model_general_constraints_max
```

候选级 `model_count`表示直接生成该候选的 `(template, K, objective)`模型数量；run summary 另设 `search_model_count_total`记录本次搜索在所有 K 和 objective 下实际启动的模型总数。不得在每个 K 候选中重复累计整轮搜索的模型数量。报告必须能区分路由模型耗时、展开耗时、全局调度耗时、验证耗时和总墙钟时间。

## 12. 错误处理与回退

- 无法证明精确同构：拆分为独立模板。
- 成员映射违反禁用项或拓扑：只对该成员独立求解。
- 路由 MILP 超时但有 incumbent：实例化后执行完整验证，并保持 `proven_optimal=false`。
- 路由 MILP 无 incumbent：使用构造式代表路由。
- 全局调度无法满足语义或资源：拒绝该候选，不输出部分调度。
- 分层模板无法构造：回退直接计划并在报告中记录原因。
- 任一最终候选仍必须通过语义、状态、约束、BDD、动态模拟、BufferPlan、XML 和死锁验证。

## 13. 测试与验收

### 单元测试

- AllGather 请求层次化策略时生成三段网关计划；无真实网关域时回退直接计划。
- `applied_strategies.hierarchy`与 `planning_mode`一致。
- 128 个只改变逻辑地址的 slice 归入一个模板；一个 slice 专属禁用项只拆分受影响成员。
- 不同链路方向、共享资源或性能参数的通信域不得复用。
- 模板实例化重建正确 slice ID、contributors、路径和依赖。
- 全局调度器不复制代表时间，并保证 lane/resource slot 无重叠。

### 属性与集成测试

- Rank 重编号后的精确同构模板生成语义等价调度。
- 2 节点网关 AllGather 和 AllReduce 的最终 contributors 与直接语义完全一致。
- 归约模板实例化后不存在贡献丢失、重复贡献或状态重复消费。
- 8 Rank、128 slice 的路由模型数量不随 slice 数增长。
- 分层 AllReduce 代表路由模型的变量和约束数量对 8、16、64、128 个 slice 保持不变。
- 所有生成候选通过现有离线验证和 XML 回读。

### 性能结构验收

对于 8 Rank AllGather：

- 直接模式中，每个固定 source/root 的 128 个逻辑位置只启动一个代表路由模型。
- 分层模式中，模型数量由非同构局部通信模式决定，不得等于 `rank_count * slice_count * K_max`。
- 任何候选均不得构造随全部真实 slice 两两增长的 lane/resource 析取约束。

墙钟时间受 Gurobi 版本和硬件影响，不以固定秒数作为正确性测试；报告必须提供足以解释耗时的结构计数和分阶段时间。

## 14. 代码与兼容性约束

- 所有生产代码、测试代码、JSON 字段、诊断和生成 XML 不包含中文字符。
- 文档可以使用中文。
- 输入 JSON 格式保持兼容，不增加必填字段。
- 现有直接计划仍可通过 `hierarchy=false`显式选择。
- 旧报告读取保持兼容；新增字段使用明确默认值。
- 新模型和调度器使用独立版本号，避免错误命中旧缓存。
