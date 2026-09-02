# VeriCCL验证报告

每个候选对应一个`.validation.json`和一个`.schedule.json`；能够降低为XML的候选还对应`.xml`或`.candidate.xml`。最终别名复制被选候选的三类制品，`run-summary.json`记录全部候选及选择结果。

<!-- validation-dimensions: ["input","semantic","state","topology","timing","resource","buffer","endpoint","deadlock","xml","bdd","simulation","runtime","online"] -->

## 验证维度

- `input`：规范化输入与哈希。
- `semantic`：最终输出Rank、逻辑偏移及贡献slice集合。
- `state`：RawValue/AggregateState版本、归约不相交及消费规则。
- `topology`：单向链路、禁用项、channel和通信组。
- `timing`：直接依赖、ready time、有向链路每channel区间及共享资源。
- `resource`：链路和NIC等共享资源的保守并发模型。
- `buffer`：输入、输出、scratch地址、原地别名和活性区间。
- `endpoint`：每个物理传输的`s`与`r/rrc`配对。
- `deadlock`：单向TB顺序、依赖step和端点同步执行模拟。
- `xml`：schema字段、偏移、chunk数、依赖坐标和哈希。
- `bdd`：基于flow、等待区间和LaneState查找可调优候选。
- `simulation`：动态并发事件、完成时间和资源利用率。
- `runtime`：MSCCL step、TB、channel、偏移和依赖编号限制。
- `online`：release统计、逐step trace、时钟误差和在线调优资格。

BDD发现的是潜在的非必要等待或可替换flow，不表示语义错误，也不直接修改调度。实际替换由调优模块完成，并重新计算后缀依赖、时间和TB顺序；只有全部约束、语义和验证再次通过的候选才能被接受。

## 状态

- `valid`：该维度已完成且满足要求。
- `fatal`：输入或问题定义不可用，不能生成候选XML。
- `invalid`：语义、依赖、缓冲区、死锁或XML错误。
- `warning`：离线有效但MSCCL运行时不兼容，输出candidate XML。
- `analysis_error`：必需分析未完成，候选不能进入最终选择。
- `failed`：请求的在线步骤失败，不否定此前离线有效结果。
- `not_run`：该维度未请求或前置条件未满足。

常见`not_run`条件包括：未指定`--online`；候选在降低前已失败，导致XML、BDD、模拟、runtime和online均无法执行；128 MiB不能被slice大小整除而跳过校准；硬件、MPI、MSCCL、nccl-tests或时钟同步工具未配置；可选Gurobi模型未安装或无许可证。

## 可复现性与最优性

`solver_metrics`记录求解器名称、版本、seed、线程数、模型数、best bound、MIP gap和终止原因。`solver_seed=0`固定伪随机起点，但不同求解器版本、硬件环境或并行执行仍可能改变候选。`reproducibility.limits`明确列出这些边界。

`selected_best=true`仅表示该候选在当前已生成且验证通过的集合中被选择；`proven_optimal=true`要求求解器在当前完整模型上给出最优性证明。构造式、受限搜索空间、调优候选或超时incumbent不得被描述为全局最优。

## 路径、哈希与调优记录

报告包含规范化输入哈希、拓扑签名、候选签名、XML哈希及绑定哈希。`lineage.parent_candidate_id`和`iteration`记录候选关系；`overlay`记录临时禁用项、channel、路径权重、lane顺序、树根/边和局部重求解范围；`tuning_strategy`记录该XML采用的初始求解、flow后缀修复或其他调优策略。

原始`topo.json`、`sketch.json`和`atom.json`不会被修改。实际使用的不可变副本写入`resolved-input.json`，所有调优只产生新候选和TuningOverlay。

## 在线证据

在线报告记录每轮样本数、中位数、P95、均值、总体标准差、CV、稳定性、trace文件、时钟误差上界、`online_operator_validation`及瓶颈优先级。`trace_analysis.step_waits`保存逐step等待分解，`bottlenecks`和`tuning_evidence.bottleneck_priorities`关联transfer、atom、flow、Rank、TB、step、lane和等待类型。只有release结果稳定、trace完整且时钟不确定性在阈值内时，在线数据才可用于候选比较；任何`failure_code`都会使online维度为`failed`。若仅release性能不稳定，而正确性与完整trace均通过，则online维度为`warning`、问题码为`online_release_unstable`，XML仍可保留，但不得执行在线调优或形成性能结论。

链路校准缓存签名覆盖链路类别、拓扑签名、GPU/NIC、CUDA/NCCL/MSCCL版本、Simple协议、slice大小、128 MiB、并发度、`NCCL_BUFFSIZE`、chunk/slice steps及相关路径变量。任一字段变化或`force_recalibrate=true`都会绕过缓存。
