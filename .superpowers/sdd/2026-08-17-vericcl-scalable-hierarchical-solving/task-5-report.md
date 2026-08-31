# Task 5 实施报告

## 状态

已完成路由模板实例化、真实 slice 语义重建、归约对偶状态汇合，以及旧 MILP schedule 语义构造逻辑的纯函数复用。实现保持 `solve_milp` 的外部行为不变，并将 provisional schedule 明确标记为 `routing_only=True`。审查修复轮次 1 已处理真实网关 AllGather demand ID 冲突、routing-only 归约状态版本链，以及非 routing-only dual metadata 兼容性。

## RED 证据

首次运行计划指定测试：

```text
.venv/bin/python -m pytest tests/unit/solver/test_instantiate.py tests/unit/composer/test_dual.py tests/property/test_ag_rs_duality.py tests/property/test_collective_semantics.py -q
```

测试在收集阶段失败，错误为：

```text
ModuleNotFoundError: No module named 'vericcl.solver.instantiate'
```

初版实现后的首次垂直测试为 1 passed、2 failed：

- 禁用边测试先返回 `mapped_route_outside_legal_domain`，而不是稳定的禁用项失败。根因是成员 demand 的合法域已提前剪枝。修复为先对完整物理路径检查禁用项，再检查各层路由域。
- reduction-dual schedule 缺少 `final_dependencies`。修复为由共享 dual 逻辑重建最终汇合依赖和 AggregateState consumption metadata。

AllGather 补充测试再次得到 RED：3 Rank、每 Rank 2 slice 时，期望 18 个最终输出，实际仅 14 个。根因是本地 passthrough 使用完全相同的 `OutputSlot` 匹配，而 AllGather 输出 offset 已重编号为全局 slice 地址。修复为按 Rank 和 contributors 匹配阶段输入。

公共 API 导出测试也先得到 `ImportError`，随后在 `vericcl.solver` 中导出 `InstantiationFailure`、`InstantiationResult` 和 `instantiate_route_patterns`。

## 实现摘要

- 新增不可变的 `InstantiationFailure`、`InstantiationResult` 和 `instantiate_route_patterns`。
- 仅将 `RoutePattern` 的 rank 拓扑结构通过 `TemplateMember` 的三个映射展开；真实 demand、slice、stage、禁用项和候选路径域均从重新构建的成员 RoutingUnit 读取。
- 逐边复核物理方向、Topology、PlanNode 域、demand 域、禁用项、共享资源成员关系及 Task 4 candidate-path-domain。
- 单个成员失败时不提交任何该成员 transfer；其他成员继续复用，并返回排序稳定的失败记录。
- provisional schedule 固定 `channel=0`、空 `resource_slots`，使用零基准、真实边持续时间和完整路径前缀，并设置 `routing_only=True`。
- 将旧 MILP 的 transfer、atom、路径、前驱和元数据构造提取为 `materialize_route_schedule`；MILP 变量提取、数值校验、求解入口和外部输出保持不变。
- reduction-dual 复用现有 AG 反向转换，重建 REDUCE、最终依赖、路径前缀和 AggregateState consumption metadata。routing-only 模式不继承占位 lane 或共享资源串行关系。
- AllReduce 垂直测试通过 compose、语义验证、BufferPlan、endpoint lowering 和 TransferDAG；最终完整 AggregateValue 对应唯一最终 rrc，后续 SEND 依赖该状态。

## GREEN 与回归

```text
计划指定聚焦测试：23 passed in 0.39s
tests/unit/solver：210 passed in 36.66s
tests/unit/composer + 相关 property：27 passed in 0.33s
完整非硬件测试（含 Gurobi）：1249 passed, 1 skipped, 8 deselected in 54.50s
tests/gurobi 单独验证：30 passed in 0.47s
```

覆盖率版完整非硬件测试通过；补充负向契约测试后的关键模块覆盖率为：

```text
vericcl/solver/instantiate.py  92%
vericcl/solver/scheduling.py   94%
vericcl/composer/dual.py       96%
```

旧 MILP 兼容性除回归测试外，还将基线提交中的 `_build_schedule` 动态加载，与新实现对 Broadcast 多值、共享资源多跳和 reduction-dual 输入逐对象比较，三类结果均相等。

## 审查修复轮次 1

### RED 证据

- 标准 `two_node_gateway.json` AllGather 走完整 `build_solver_problem -> split/templates -> instantiate -> compose -> semantic verifier -> BufferPlan` 链时，首次在 `SolverProblem` 构造阶段得到 `SemanticError: solver problem demand IDs must be unique`。根因是多个 stage demand 共享 node、logical position、root 和 leaf，但具有不同真实 contributors/member slice 集。
- routing-only 4 Rank star 的判别测试初次运行有 3 个失败：三个 REDUCE 没有 accumulator 前驱链；`aggregate_consumptions` 仍是 `final_dependencies` 的别名；非 routing-only dual 比基线多出 routing metadata。
- 对重复 final state 的精确版本测试加入 `(correct, unknown)` 和 `(correct, wrong-contributor-producer)` 后，inplace 与 out-of-place 两个用例均错误返回 `VALID`，证明仅过滤未知依赖会放宽语义守恒。
- 多跳 routing-only reduction 的 path closure 判别测试先选择按 ID 排序的子树 transfer，得到 `reduce-a-child`，而不是包含完整成员路径的 terminal transfer。

### 修复边界

- demand ID 继续保留 node、logical position、root 和 leaf 的可读前缀，并追加 contributors 与 member slice IDs 的完整、排序、定长十进制编码；不使用截断哈希。candidate path 域、剪枝逻辑和模板签名未改变，模板签名仍由相对 rank/contributor/logical-position 语义构造，不读取原始 demand ID。
- routing-only dual 按同一 tree、目标 Rank 和 logical position 建立确定性 accumulator chain。每个 REDUCE 同时依赖其源子树与上一个 accumulator producer；`_schedule_drafts` 因该语义链重算 provisional 时间，因此后继开始时间不早于前驱结束时间，不同 logical position 仍可并行。
- `aggregate_consumptions[transfer_id]` 记录 `consumed_state_ids=(source, accumulator)` 与 `produced_state_id`；`aggregate_states[state_id]` 记录 rank、logical position、contributors 和 producer ID。状态 ID 本身完整编码 tree、rank、logical position、version 和 contributors。输入 contributors 必须不相交，输出必须为并集，同一状态版本不得消费两次。
- 每个 final output 的 `final_dependencies` 只含生成完整 contributors 的 terminal rrc，`final_ready_time` 等于该 rrc 的结束时间。Composer 仅把 terminal rrc 作为下游 SEND 的直接语义前驱，同时沿其语义闭包重建成员路径，并优先使用 terminal transfer 中已有的完整路径。
- BufferPlan 将最终完整 `AggregateValue` 绑定到唯一 terminal rrc；endpoint lowering 产生对应 `RECV_REDUCE_COPY`。TransferDAG 对 `AggregateValue` 的 path 边使用相同 value 与 source ref 的精确 producer，因此早期 rrc 只通过 terminal rrc 形成传递依赖。
- final-state 消歧不是放宽语义守恒：仅当某 output slot 的 `final_dependencies` 恰有一个字符串 ID，且该 ID 对应 replay 后仍活动、rank/slot/contributors 完全匹配的状态版本时，重复候选才可消歧。空依赖、未知 ID、错误 contributor、混合正确与错误 ID，以及多个匹配 producer 均保持 `INVALID/duplicate_final_state`。该判定发生在逻辑 payload replay 层，与后续 buffer placement 无关；测试分别覆盖 inplace 与 out-of-place。
- `final_dependencies`、`aggregate_consumptions`、`aggregate_states` 和 routing path metadata 仅在 `routing_only=True` 分支生成。非 routing-only `reverse_allgather_schedule` 通过硬编码基线完整 `Schedule` 与 metadata 的逐对象相等测试。

### 最新 GREEN 与回归

```text
审查修复定向链：7 passed in 2.45s
Task 5 计划指定测试：28 passed in 2.51s
tests/unit/solver + tests/unit/composer + tests/property：252 passed in 21.23s
受影响 verification + XML 与聚焦模块：306 passed in 2.76s
完整非硬件测试：1223 passed, 1 skipped in 29.89s
tests/gurobi：30 passed in 0.24s
文档命令回归：29 passed in 2.53s
```

最终静态门禁：

```text
.venv/bin/python -m compileall -q vericcl tests: passed
git diff --check: passed
.venv/bin/python -m vericcl --help: passed
changed production/test Han-character scan: no matches
```

静态检查：

```text
python -m compileall -q vericcl tests: passed
git diff --check: passed
changed production/test Han-character scan: no matches
```

## 未解决关注项

- 8 个硬件测试按 marker 排除，未在本机执行；这不计为通过。
- provisional schedule 的 channel、资源 slot 和最终时间仍需由后续全局调度任务确定，这是既定接口语义，不是当前缺陷。
- 未发现 Task 5 范围内的功能性遗留问题。
