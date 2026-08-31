# Task 5 实施报告

## 状态

已完成路由模板实例化、真实 slice 语义重建、归约对偶状态汇合，以及旧 MILP schedule 语义构造逻辑的纯函数复用。实现保持 `solve_milp` 的外部行为不变，并将 provisional schedule 明确标记为 `routing_only=True`。

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
