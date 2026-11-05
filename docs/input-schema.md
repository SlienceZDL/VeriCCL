# 输入字段参考

此表由 `vericcl/input/schema.py` 生成。四份 JSON 都拒绝重复键、未知字段和非有限数；原文件不会被覆盖。省略或 null 表示未指定：有默认值时补入默认，无默认但本次启用功能需要的字段报错。字段在新旧位置重复出现时，即使值相同也报错。

M 必须是 S 的整数倍，S 必须包含完整的数据类型元素；AllToAll/ReduceScatter 的 N=M/S 还须能被 rank 数整除。禁止直接输入 N 或 input_chunkup。不适用算子的 root/reduction_op 使用省略或 null。

命令行 `--set 路径=JSON` 可以补入未指定值。覆盖已有不同值须加 `--override-input`。原文件保存在 `input_snapshot`，最终有效配置和来源分别写入 `effective-request.json`、`value-sources.json`。配置字段不读取 shell 占位符，请在写 JSON 时使用实际绝对路径。

solve/live 必须给初始合成预算和最大轮数；轮数大于 0 时还须给修复预算。所有处理入口需要单模型/CPU、离线每轮/累计及清理预算。校准启用或使用 calibrate 时，需要在线预算、拟合误差和单任务期限；trace/实测启用时还需要 trace 容量和任务期限。历史复用各自要求索引及对应容差/年龄。live 要求全部七个阶段期限和控制端点。运行时数值检查使用显式绝对/相对容差。

JSON Schema 描述标准结构；跨字段算术、启用条件、重复键和明确的旧写法迁移由 load_request 检查。算子名称忽略大小写；兼容别名为 all_gather、all_reduce、all_to_all、reducescatter。旧 hyperparameters 中明确列出的 solver 参数可迁移；atom 顶层的 symmetry、shortest_paths、batching、objective_mode 可迁移。旧 total_solve_timeout_s 仅迁移为初始合成预算。force_resolve、force_recalibrate 及旧算法布尔开关已删除。

下面的 object/array 行描述所属配置段，后续行描述其成员。array 中的 [] 表示每一个元素。更多运行方式见 [README](../README.md)，证据范围见 [适用边界](supported-configuration.md)。


## topology.json

[机器可读 Schema](input-schemas/topology.json)

| 字段 | 类型 | 默认值 | 要求与取值 | 作用 |
|---|---|---|---|---|
| `topology.name` | string | `未指定` | 按启用条件检查；— | 拓扑名称，仅用于识别。 |
| `topology.ranks` | integer | `未指定` | 必填；≥1，≤9223372036854775807 | 参与任务的逻辑 rank 数；节点内列表则列出其 rank。 |
| `topology.nodes` | array | `未指定` | 必填；— | 节点分组，完整覆盖 rank；节点编号不代替实际主机地址。 |
| `topology.nodes[].id` | integer | `未指定` | 必填；≥0，≤9223372036854775807 | 本配置内唯一标识。 |
| `topology.nodes[].ranks` | array | `未指定` | 必填；— | 参与任务的逻辑 rank 数；节点内列表则列出其 rank。 |
| `topology.nodes[].gateways` | array | `[]` | 可省略或 null；— | 可作为该节点对外连接端点的 rank。 |
| `topology.directed_links` | array | `未指定` | 必填；— | 有向直连或交换机抽象链路；反方向须单独提供。 |
| `topology.directed_links[].src` | integer | `未指定` | 必填；≥0，≤9223372036854775807 | 发送端 rank。 |
| `topology.directed_links[].dst` | integer | `未指定` | 必填；≥0，≤9223372036854775807 | 接收端 rank。 |
| `topology.directed_links[].alpha_us` | number | `未指定` | 必填；≥0 | 每次服务固定开销，单位微秒。 |
| `topology.directed_links[].beta_us_per_byte` | number | `未指定` | 必填；>0 | 每字节服务开销，单位微秒/字节，带宽为其倒数。 |
| `topology.directed_links[].max_channels` | integer | `16` | 可省略或 null；≥1，≤16 | 允许的 channel 数上限；任务默认搜索 1 到 7，显式上限可到 16。 |
| `topology.directed_links[].resources` | array | `[]` | 可省略或 null；— | 该链路占用的共享资源标识。 |
| `topology.directed_links[].physical_path` | array | `[]` | 可省略或 null；— | 抽象链路对应的物理路径说明。 |
| `topology.shared_resources` | array | `[]` | 可省略或 null；— | 声明不同抽象链路的共同瓶颈，防止重复计算物理带宽。 |
| `topology.shared_resources[].id` | string | `未指定` | 必填；— | 本配置内唯一标识。 |
| `topology.shared_resources[].capacity_bytes_per_us` | number | `未指定` | 必填；>0 | 资源总有效带宽，单位字节/微秒。 |
| `topology.shared_resources[].links` | array | `[]` | 可省略或 null；— | 属于资源的有向链路 [src,dst]。 |
| `topology.shared_resources[].kind` | string | `"other"` | 可省略或 null；nic / pcie / switch_port / link / other | 共享资源类别。 |
| `topology.groups` | array | `[]` | 可省略或 null；— | 用户提供的候选分层组。 |
| `topology.local_costs` | array | `[]` | 可省略或 null；— | 当前数据类型和归约操作下的本地开销初值。 |
| `topology.local_costs[].rank` | integer | `未指定` | 必填；≥0，≤9223372036854775807 | 逻辑 rank 编号，从 0 开始。 |
| `topology.local_costs[].copy` | object | `未指定` | 按启用条件检查；— | 本地读取、写入或复制的初始代价。 |
| `topology.local_costs[].copy.alpha_us` | number | `未指定` | 必填；≥0 | 每次服务固定开销，单位微秒。 |
| `topology.local_costs[].copy.beta_us_per_byte` | number | `未指定` | 必填；>0 | 每字节服务开销，单位微秒/字节，带宽为其倒数。 |
| `topology.local_costs[].reduce` | object | `未指定` | 按启用条件检查；— | 本地归约的初始代价。 |
| `topology.local_costs[].reduce.alpha_us` | number | `未指定` | 必填；≥0 | 每次服务固定开销，单位微秒。 |
| `topology.local_costs[].reduce.beta_us_per_byte` | number | `未指定` | 必填；>0 | 每字节服务开销，单位微秒/字节，带宽为其倒数。 |

## sketch.json

[机器可读 Schema](input-schemas/sketch.json)

| 字段 | 类型 | 默认值 | 要求与取值 | 作用 |
|---|---|---|---|---|
| `sketch.collective` | object | `未指定` | 必填；— | 对应用可见的算子语义。 |
| `sketch.collective.operator` | string | `未指定` | 必填；broadcast / reduce / allgather / allreduce / alltoall / reduce_scatter | 集合通信算子；手工内部节点还可使用 Scatter/Gather。 |
| `sketch.collective.datatype` | string | `"float32"` | 可省略或 null；float16 / float32 / float64 / bfloat16 / int8 / uint8 / int32 / uint32 / int64 / uint64 | 数据类型；默认 float32，实际支持需通过运行时检查。 |
| `sketch.collective.root` | integer | `适用时 0` | 按启用条件检查；≥0，≤9223372036854775807 | Broadcast/Reduce 的外部根，默认 0；手工节点表示该节点的根。 |
| `sketch.collective.reduction_op` | string | `适用时 sum` | 按启用条件检查；sum / prod / min / max / avg | Reduce/AllReduce/ReduceScatter 的归约操作，默认 sum。其他外部算子须省略或为 null。 |
| `sketch.collective.inplace` | boolean | `false` | 可省略或 null；— | 是否使用合法的原位缓冲关系，默认关闭。 |
| `sketch.hyperparameters` | object | `未指定` | 必填；— | 应用输入规模与最小 slice。 |
| `sketch.hyperparameters.total_size_bytes` | integer | `未指定` | 必填；≥1，≤9223372036854775807 | 每 rank 输入字节数 M；Broadcast 指 root 消息大小。 |
| `sketch.hyperparameters.slice_size_bytes` | integer | `null` | 可省略或 null；≥1，≤9223372036854775807 | 固定最小 slice 字节数 S；省略时先搜索合法分区。N 由 M/S 派生。 |
| `sketch.hyperparameters.reference_slice_size_bytes` | integer | `null` | 可省略或 null；≥1，≤9223372036854775807 | 解释原始 slice_id 硬约束的参考分区；重分区时仍绑定原字节范围。 |
| `sketch.solver` | object | `{}` | 可省略或 null；— | 求解器、CPU 配额和求解策略。 |
| `sketch.solver.backend` | string | `"scip"` | 可省略或 null；scip | 仅支持 SCIP；缺失时直接报错。 |
| `sketch.solver.per_model_timeout_s` | number | `未指定` | 按启用条件检查；>0 | 单个求解模型的时间上限，仍受父阶段剩余额度约束。 |
| `sketch.solver.cpu_budget` | integer | `未指定` | 按启用条件检查；≥1，≤9223372036854775807 | 整个任务可使用的 CPU 配额。 |
| `sketch.solver.max_threads_per_model` | integer | `12` | 可省略或 null；≥1，≤9223372036854775807 | 单模型线程上限，默认 12；实际还受 CPU 配额限制。 |
| `sketch.solver.max_parallel_models` | integer | `未指定` | 按启用条件检查；≥1，≤9223372036854775807 | 同时运行模型数的上限；省略时由预算管理器控制。 |
| `sketch.solver.max_channels` | integer | `7` | 可省略或 null；≥1，≤16 | 允许的 channel 数上限；任务默认搜索 1 到 7，显式上限可到 16。 |
| `sketch.solver.solver_seed` | integer | `0` | 可省略或 null；≥0，≤9223372036854775807 | SCIP 随机种子。 |
| `sketch.solver.mip_gap` | number | `0` | 可省略或 null；≥0，≤1 | 允许的数值 MIP gap；不构成完整修复空间证明。 |
| `sketch.solver.require_proven_optimal` | boolean | `false` | 可省略或 null；— | 是否要求当前求解模型达到数值最优状态。 |
| `sketch.solver.bdd_backend` | string | `"cudd"` | 可省略或 null；cudd / autoref | 分析使用的 BDD 后端；CUDD 默认，autoref 仅显式选择。 |
| `sketch.budgets` | object | `{}` | 可省略或 null；— | 各阶段时间上限和最大调优轮数；时间均为秒。 |
| `sketch.budgets.max_initial_synthesis_s` | number | `未指定` | 按启用条件检查；>0 | 本次初始合成总时间上限。 |
| `sketch.budgets.max_online_validation_per_round_s` | number | `未指定` | 按启用条件检查；>0 | 一轮在线校准、trace 和实测的累计上限。 |
| `sketch.budgets.max_online_validation_total_s` | number | `未指定` | 按启用条件检查；>0 | 整个任务在线验证累计上限。 |
| `sketch.budgets.max_offline_validation_per_round_s` | number | `未指定` | 按启用条件检查；>0 | 一轮离线检查与分析累计上限。 |
| `sketch.budgets.max_offline_validation_total_s` | number | `未指定` | 按启用条件检查；>0 | 整个任务离线检查与分析累计上限。 |
| `sketch.budgets.max_repair_per_round_s` | number | `未指定` | 按启用条件检查；>0 | 一轮所有局部/结构/分区修复累计上限。 |
| `sketch.budgets.max_repair_total_s` | number | `未指定` | 按启用条件检查；>0 | 整个任务修复累计上限。 |
| `sketch.budgets.max_run_wall_s` | number | `未指定` | 按启用条件检查；>0 | 可选的整个任务墙钟上限。 |
| `sketch.budgets.cleanup_reserve_s` | number | `未指定` | 按启用条件检查；>0 | 从剩余时间中保留的收尾时间；不会延长父期限。 |
| `sketch.budgets.max_tuning_rounds` | integer | `未指定` | 按启用条件检查；≥0，≤9223372036854775807 | 最多开始多少轮调优；0 表示建立初始结果后结束。 |
| `sketch.tolerances` | object | `{}` | 可省略或 null；— | 浮点数值比较容差，执行时须明确提供。 |
| `sketch.tolerances.absolute` | number | `未指定` | 按启用条件检查；≥0 | 绝对误差容差。 |
| `sketch.tolerances.relative` | number | `未指定` | 按启用条件检查；≥0 | 相对误差容差。 |
| `sketch.calibration` | object | `{}` | 可省略或 null；— | 独立原语测量与代价模型校准。 |
| `sketch.calibration.enabled` | boolean | `true` | 可省略或 null；— | 启用本配置段的功能。 |
| `sketch.calibration.launches` | integer | `20` | 可省略或 null；≥1，≤9223372036854775807 | 每轮独立进程启动数；同次启动内的迭代不视为独立启动。 |
| `sketch.calibration.warmup_iterations` | integer | `5` | 可省略或 null；≥0，≤9223372036854775807 | 每次启动的预热次数；详细 trace 在预热后清零再记录一次。 |
| `sketch.calibration.timed_iterations` | integer | `20` | 可省略或 null；≥1，≤9223372036854775807 | 无详细 trace 时每次启动的计时次数。 |
| `sketch.calibration.max_rounds` | integer | `3` | 可省略或 null；≥1，≤9223372036854775807 | 稳定性不满足时最多进行的测量轮数。 |
| `sketch.calibration.max_cv` | number | `0.05` | 可省略或 null；>0 | 同轮样本变异系数上限。 |
| `sketch.calibration.max_relative_fit_error` | number | `未指定` | 按启用条件检查；≥0 | 代价拟合允许的最大相对误差。 |
| `sketch.calibration.job_timeout_s` | number | `未指定` | 按启用条件检查；>0 | 完整单次测量任务上限，含启动、重试、收集与清理。 |
| `sketch.calibration.payload_sizes` | array | `[]` | 可省略或 null；— | 额外探针大小；不改变应用的 M/S。 |
| `sketch.trace_and_measure` | object | `{}` | 可省略或 null；— | 同一开关控制详细 trace 和候选正式实测收益门槛，默认关闭。 |
| `sketch.trace_and_measure.enabled` | boolean | `false` | 可省略或 null；— | 启用本配置段的功能。 |
| `sketch.trace_and_measure.pairs` | integer | `20` | 可省略或 null；≥1，≤9223372036854775807 | 预先确定的 AB/BA 成对启动数，须为至少 2 的偶数。 |
| `sketch.trace_and_measure.warmup_iterations` | integer | `5` | 可省略或 null；≥0，≤9223372036854775807 | 每次启动的预热次数；详细 trace 在预热后清零再记录一次。 |
| `sketch.trace_and_measure.timed_iterations` | integer | `20` | 可省略或 null；≥1，≤9223372036854775807 | 无详细 trace 时每次启动的计时次数。 |
| `sketch.trace_and_measure.max_rounds` | integer | `3` | 可省略或 null；≥1，≤9223372036854775807 | 稳定性不满足时最多进行的测量轮数。 |
| `sketch.trace_and_measure.max_cv` | number | `0.05` | 可省略或 null；>0 | 同轮样本变异系数上限。 |
| `sketch.trace_and_measure.min_relative_gain` | number | `0` | 可省略或 null；≥0，≤1 | 实测要求的最低相对改善。 |
| `sketch.trace_and_measure.task_alpha` | number | `0.05` | 可省略或 null；>0，≤0.05 | 整个任务允许的统计错误概率预算，最大 0.05。 |
| `sketch.trace_and_measure.single_attempt_alpha_cap` | number | `0.05` | 可省略或 null；>0，≤0.05 | 单次比较的错误概率上限，仍受任务账本分配约束。 |
| `sketch.trace_and_measure.seed` | integer | `0` | 可省略或 null；≥0，≤9223372036854775807 | 预先确定采样顺序的种子。 |
| `sketch.trace_and_measure.job_timeout_s` | number | `未指定` | 按启用条件检查；>0 | 完整单次测量任务上限，含启动、重试、收集与清理。 |
| `sketch.trace_and_measure.trace_capacity_bytes` | integer | `未指定` | 按启用条件检查；≥1，≤9223372036854775807 | 每 rank 的详细 trace 缓冲区字节数，至少容纳一次完整记录。 |
| `sketch.repair` | object | `{}` | 可省略或 null；— | 修复阶段的分析/求解分工。 |
| `sketch.repair.mode` | string | `"strategy_first"` | 可省略或 null；strategy_first / joint | strategy_first 先选策略，joint 联合选择合法决策。 |
| `sketch.repair.candidate_batch_size` | integer | `4` | 可省略或 null；≥1，≤9223372036854775807 | 每批进入局部 SCIP 的候选数；不表示总搜索上限。 |
| `sketch.repair.analysis_batch_size` | integer | `16` | 可省略或 null；≥1，≤9223372036854775807 | 每批分析建议数；耗尽后继续扩展。 |
| `sketch.runtime_compatibility` | object | `{}` | 可省略或 null；— | 可关闭的完整静态运行时兼容检查；基础启动安全检查始终保留。 |
| `sketch.runtime_compatibility.enabled` | boolean | `true` | 可省略或 null；— | 启用本配置段的功能。 |
| `sketch.history` | object | `{}` | 可省略或 null；— | 跨任务调度与绝对校准缓存，两个开关互相独立，默认关闭。 |
| `sketch.history.schedule_reuse` | boolean | `false` | 可省略或 null；— | 检索同构且相对带宽接近的历史调度，再适配并重新验证。 |
| `sketch.history.calibration_reuse` | boolean | `false` | 可省略或 null；— | 仅复用精确设备/运行时/类型/模式身份下仍有效的测量组件。 |
| `sketch.history.relative_bandwidth_tolerance` | number | `未指定` | 按启用条件检查；≥0 | 映射后的各带宽缩放比例最大值/最小值减一的允许上限。 |
| `sketch.history.max_schedule_candidates` | integer | `未指定` | 按启用条件检查；≥1，≤9223372036854775807 | 快速历史检索最多检查的候选数。 |
| `sketch.history.calibration_max_age_s` | number | `未指定` | 按启用条件检查；>0 | 校准缓存的最大年龄。 |
| `sketch.history.schedule_index` | string | `未指定` | 按启用条件检查；— | 历史调度索引路径；相对路径从案例目录解析。 |
| `sketch.history.calibration_index` | string | `未指定` | 按启用条件检查；— | 校准索引路径；相对路径从案例目录解析。 |
| `sketch.live` | object | `{}` | 可省略或 null；— | 辅助常驻在线模式，主离线闭环不依赖其开启。 |
| `sketch.live.enabled` | boolean | `false` | 可省略或 null；— | 启用本配置段的功能。 |
| `sketch.live.request_prepare_timeout_s` | number | `未指定` | 按启用条件检查；>0 | 请求准备和原输入保护期限。 |
| `sketch.live.execution_attempt_timeout_s` | number | `未指定` | 按启用条件检查；>0 | 每次 GPU 尝试期限。 |
| `sketch.live.background_tuning_task_timeout_s` | number | `未指定` | 按启用条件检查；>0 | 一次后台调优任务期限。 |
| `sketch.live.version_prepare_timeout_s` | number | `未指定` | 按启用条件检查；>0 | 新版本准备期限。 |
| `sketch.live.control_phase_timeout_s` | number | `未指定` | 按启用条件检查；>0 | 一个控制阶段的期限。 |
| `sketch.live.recovery_timeout_s` | number | `未指定` | 按启用条件检查；>0 | 整个失败恢复过程的固定总期限。 |
| `sketch.live.close_cleanup_timeout_s` | number | `未指定` | 按启用条件检查；>0 | 关闭、后台结束和资源回收的共同期限。 |
| `sketch.live.window_requests` | integer | `8` | 可省略或 null；≥1，≤9223372036854775807 | 一个合格遥测窗口的请求数。 |
| `sketch.live.consecutive_windows` | integer | `2` | 可省略或 null；≥1，≤9223372036854775807 | 触发前要求连续同方向变化的窗口数。 |
| `sketch.live.relative_change_threshold` | number | `0.1` | 可省略或 null；>0，≤0.99 | 相对基线的变化阈值。 |
| `sketch.live.baseline_max_cv` | number | `0.05` | 可省略或 null；≥0，≤0.05 | 建立基线所需稳定性上限。 |
| `sketch.live.min_residence_requests` | integer | `32` | 可省略或 null；≥0，≤9223372036854775807 | 采用版本后，普通更新前至少保留的成功请求数。 |
| `sketch.live.max_retries` | integer | `2` | 可省略或 null；≥0，≤4294967295 | 失败后允许的额外完整请求重试次数；默认 2。 |
| `sketch.live.external_telemetry_path` | string | `未指定` | 按启用条件检查；— | 调用者维护的带内遥测 JSONL 文件路径。 |
| `sketch.live.external_max_age_s` | number | `60` | 可省略或 null；>0 | 接受外部遥测的最大年龄。 |
| `sketch.live.fault_injection` | object | `{}` | 可省略或 null；— | 仅显式启用的开发故障注入。 |
| `sketch.live.fault_injection.request_id` | integer | `0` | 可省略或 null；≥0，≤9223372036854775807 | 需要注入失败的逻辑请求号；0 关闭。 |
| `sketch.live.fault_injection.attempts` | integer | `0` | 可省略或 null；≥0，≤4294967295 | 该请求前多少次尝试在实际完成后被标记失败。 |

## atom.json

[机器可读 Schema](input-schemas/atom.json)

| 字段 | 类型 | 默认值 | 要求与取值 | 作用 |
|---|---|---|---|---|
| `atom.synthesis_method` | string | `"auto"` | 可省略或 null；auto / hierarchy / constructive_trees | auto 按 M_switch 选择；小消息 hierarchy，大消息 constructive_trees。 |
| `atom.M_switch` | integer | `未指定` | 按启用条件检查；≥1，≤9223372036854775807 | auto 模式必须显式给出的消息大小分界，单位字节。 |
| `atom.stage_num` | integer | `null` | 可省略或 null；≥1，≤9223372036854775807 | 逻辑阶段数硬约束；省略时自动确定。 |
| `atom.forbidden_transfers` | array | `[]` | 可省略或 null；— | [原 slice_id,src,dst,stage]，只禁止该阶段的这条直连边，仍可合法绕路。 |
| `atom.strategies` | object | `{}` | 可省略或 null；— | 可选初始求解策略，默认不启用。 |
| `atom.strategies.symmetry` | boolean | `false` | 可省略或 null；— | 启用对称约简。 |
| `atom.strategies.shortest_paths` | boolean | `false` | 可省略或 null；— | 启用候选路径缩减策略。 |
| `atom.strategies.batching` | boolean | `false` | 可省略或 null；— | 显式启用分层求解的批处理开关；批量生成树方法按自身规则管理 batch。 |
| `atom.strategies.objective_mode` | string | `null` | 可省略或 null；latency / throughput / auto | 显式选择延迟/吞吐目标，或由消息规模自动选择。 |
| `atom.manual` | object | `{}` | 可省略或 null；— | 显式手工计划，仅在 enabled=true 时使用。 |
| `atom.manual.enabled` | boolean | `false` | 可省略或 null；— | 启用本配置段的功能。 |
| `atom.manual.nodes` | array | `[]` | 可省略或 null；— | 手工逻辑节点；每个节点给出组、算子、依赖及可选状态合同。 |
| `atom.manual.nodes[].id` | string | `未指定` | 必填；— | 本配置内唯一标识。 |
| `atom.manual.nodes[].stage_id` | integer | `未指定` | 必填；≥0，≤9223372036854775807 | 逻辑阶段编号，从 0 开始。 |
| `atom.manual.nodes[].operator` | string | `未指定` | 必填；broadcast / reduce / scatter / gather / allgather / allreduce / alltoall / reduce_scatter | 集合通信算子；手工内部节点还可使用 Scatter/Gather。 |
| `atom.manual.nodes[].group` | array | `未指定` | 必填；— | 该手工节点允许使用的 rank 集合。 |
| `atom.manual.nodes[].root` | integer | `未指定` | 按启用条件检查；≥0，≤9223372036854775807 | Broadcast/Reduce 的外部根，默认 0；手工节点表示该节点的根。 |
| `atom.manual.nodes[].relay_ranks` | array | `[]` | 可省略或 null；— | 允许中转的组内 rank。 |
| `atom.manual.nodes[].depends_on` | array | `[]` | 可省略或 null；— | 该手工节点依赖的节点 ID。 |
| `atom.manual.nodes[].input_states` | array | `未指定` | 按启用条件检查；— | 手工节点的输入状态合同。 |
| `atom.manual.nodes[].input_states[].rank` | integer | `未指定` | 必填；≥0，≤9223372036854775807 | 逻辑 rank 编号，从 0 开始。 |
| `atom.manual.nodes[].input_states[].offset` | integer | `未指定` | 必填；≥0，≤9223372036854775807 | 逻辑数据位置编号。 |
| `atom.manual.nodes[].input_states[].contributors` | array | `未指定` | 必填；— | 该状态包含的原始 slice_id 集合。 |
| `atom.manual.nodes[].output_states` | array | `未指定` | 按启用条件检查；— | 手工节点的输出状态合同。 |
| `atom.manual.nodes[].output_states[].rank` | integer | `未指定` | 必填；≥0，≤9223372036854775807 | 逻辑 rank 编号，从 0 开始。 |
| `atom.manual.nodes[].output_states[].offset` | integer | `未指定` | 必填；≥0，≤9223372036854775807 | 逻辑数据位置编号。 |
| `atom.manual.nodes[].output_states[].contributors` | array | `未指定` | 必填；— | 该状态包含的原始 slice_id 集合。 |

## runtime.json

[机器可读 Schema](input-schemas/runtime.json)

| 字段 | 类型 | 默认值 | 要求与取值 | 作用 |
|---|---|---|---|---|
| `runtime.rank_map` | array | `未指定` | 必填；— | 每个 rank 的实际主机和设备位置，必须完整且无重复设备。 |
| `runtime.rank_map[].rank` | integer | `未指定` | 必填；≥0，≤9223372036854775807 | 逻辑 rank 编号，从 0 开始。 |
| `runtime.rank_map[].host` | string | `未指定` | 必填；— | SSH/MPI 均可访问的主机名或地址。 |
| `runtime.rank_map[].device` | integer | `未指定` | 必填；≥0，≤9223372036854775807 | 该 host 进程可见的 CUDA 设备编号。 |
| `runtime.environment` | object | `未指定` | 必填；— | 运行时环境变量白名单，值均为字符串。 |
| `runtime.environment.CUDA_HOME` | string | `未指定` | 按启用条件检查；— | CUDA 根目录。 |
| `runtime.environment.CUDA_PATH` | string | `未指定` | 按启用条件检查；— | CUDA 根目录兼容变量。 |
| `runtime.environment.CUDA_VISIBLE_DEVICES` | string | `未指定` | 按启用条件检查；— | 设备可见范围，会改变 device 编号含义。 |
| `runtime.environment.LD_LIBRARY_PATH` | string | `未指定` | 按启用条件检查；— | 动态库搜索路径，优先目标运行时与 MPI。 |
| `runtime.environment.MSCCL_ROOT` | string | `未指定` | 按启用条件检查；— | 运行时根目录。 |
| `runtime.environment.MSCCL_XML_FILES` | string | `未指定` | 按启用条件检查；— | 既有 MSCCL XML 加载接口；原生版本接口显式绑定计划。 |
| `runtime.environment.NCCL_ALGO` | string | `未指定` | 按启用条件检查；— | 保留的 NCCL 算法接口。 |
| `runtime.environment.NCCL_BUFFSIZE` | string | `未指定` | 必填；— | 每 channel 通信缓冲配置，须是 1 到 2147483647 的十进制字节字符串。 |
| `runtime.environment.NCCL_DEBUG` | string | `未指定` | 按启用条件检查；— | 运行时日志级别。 |
| `runtime.environment.NCCL_DEBUG_SUBSYS` | string | `未指定` | 按启用条件检查；— | 运行时日志子系统。 |
| `runtime.environment.NCCL_IB_GID_INDEX` | string | `未指定` | 按启用条件检查；— | 显式 GID 编号。 |
| `runtime.environment.NCCL_IB_HCA` | string | `未指定` | 按启用条件检查；— | 明确使用的 IB/RoCE 设备。 |
| `runtime.environment.NCCL_P2P_DISABLE` | string | `未指定` | 按启用条件检查；— | 是否禁用 GPU P2P 路径。 |
| `runtime.environment.NCCL_PROTO` | string | `未指定` | 按启用条件检查；— | 运行协议，任务使用 Simple。 |
| `runtime.environment.NCCL_SHM_DISABLE` | string | `未指定` | 按启用条件检查；— | 是否禁用主机共享内存路径。 |
| `runtime.environment.NCCL_SOCKET_IFNAME` | string | `未指定` | 按启用条件检查；— | 通信网卡选择。 |
| `runtime.environment.NCCL_TESTS_ROOT` | string | `未指定` | 按启用条件检查；— | 可选 nccl-tests 根目录。 |
| `runtime.environment.OMPI_MCA_btl` | string | `未指定` | 按启用条件检查；— | Open MPI BTL 组件。 |
| `runtime.environment.OMPI_MCA_btl_tcp_if_include` | string | `未指定` | 按启用条件检查；— | Open MPI TCP 数据通道使用的接口或网段；各节点须互相可达。 |
| `runtime.environment.OMPI_MCA_oob_tcp_if_include` | string | `未指定` | 按启用条件检查；— | Open MPI 控制通道使用的接口或网段；可与 TCP 数据通道显式使用同一网段。 |
| `runtime.environment.OMPI_MCA_pml` | string | `未指定` | 按启用条件检查；— | Open MPI PML 组件。 |
| `runtime.environment.OMP_NUM_THREADS` | string | `未指定` | 按启用条件检查；— | OpenMP 线程数。 |
| `runtime.environment.PATH` | string | `未指定` | 按启用条件检查；— | 子进程工具搜索路径。 |
| `runtime.environment.PYTHONPATH` | string | `未指定` | 按启用条件检查；— | Python 模块搜索路径。 |
| `runtime.environment.UCX_NET_DEVICES` | string | `未指定` | 按启用条件检查；— | UCX 网络设备。 |
| `runtime.environment.UCX_TLS` | string | `未指定` | 按启用条件检查；— | UCX 传输选择。 |
| `runtime.paths` | object | `{}` | 可省略或 null；— | 显式工具链和可执行文件路径。 |
| `runtime.paths.cuda_home` | string | `未指定` | 按启用条件检查；— | 已有 CUDA 安装根目录。 |
| `runtime.paths.mpi_bin` | string | `未指定` | 按启用条件检查；— | 含 mpirun、mpicxx 的目录。 |
| `runtime.paths.msccl_root` | string | `未指定` | 按启用条件检查；— | 编译后的 MSCCL 根目录，含 include/lib。 |
| `runtime.paths.nccl_tests_root` | string | `未指定` | 按启用条件检查；— | 可选 nccl-tests 根目录。 |
| `runtime.paths.vericcl_driver` | string | `未指定` | 按启用条件检查；— | 独立原语测量程序 vericcl-probe 的绝对路径。 |
| `runtime.paths.vericcl_native` | string | `未指定` | 按启用条件检查；— | 完整调度及常驻程序 vericcl-native 的绝对路径。 |
| `runtime.device_memory_budget_bytes` | integer | `0` | 可省略或 null；≥0，≤9223372036854775807 | 设备总预算；0 使用启动时可用显存界，仍检查实际分配。 |
| `runtime.protocol` | string | `"Simple"` | 可省略或 null；Simple | 当前固定使用 Simple。 |
| `runtime.scratch_bytes_per_rank` | integer | `未指定` | 必填；≥0，≤9223372036854775807 | 每 rank 允许的 scratch 字节数。 |
| `runtime.limits` | object | `{}` | 可省略或 null；— | 运行时结构上限，取值至多为编译支持值。 |
| `runtime.limits.steps_per_tb` | integer | `256` | 可省略或 null；≥1，≤256 | 每 TB 最大指令数。 |
| `runtime.limits.tbs_per_rank` | integer | `216` | 可省略或 null；≥1，≤216 | 每 rank 最大 TB 数；还须满足实际设备驻留能力。 |
| `runtime.limits.send_tbs_per_channel` | integer | `32` | 可省略或 null；≥1，≤32 | 每 channel 发送 TB 上限。 |
| `runtime.limits.recv_tbs_per_channel` | integer | `32` | 可省略或 null；≥1，≤32 | 每 channel 接收 TB 上限。 |
| `runtime.limits.dependency_id` | integer | `127` | 可省略或 null；≥0，≤127 | 依赖编号上限。 |
| `runtime.limits.buffer_offset` | integer | `32767` | 可省略或 null；≥0，≤32767 | 缓冲区 slice 偏移上限。 |
| `runtime.tb_layout` | string | `"separate"` | 可省略或 null；separate / fused | 默认按发送/接收分开编排；fused 须显式选择。 |
| `runtime.control` | object | `{}` | 可省略或 null；— | 常驻控制面配置。 |
| `runtime.control.bind_address` | string | `未指定` | 按启用条件检查；— | rank 0 的可达监听地址。 |
| `runtime.control.port` | integer | `未指定` | 按启用条件检查；≥1，≤65535 | 控制 TCP 端口。 |
| `runtime.control.unix_socket` | string | `未指定` | 按启用条件检查；— | 控制器所在节点的 Unix socket 绝对路径。 |
