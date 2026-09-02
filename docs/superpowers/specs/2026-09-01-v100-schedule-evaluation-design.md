# VeriCCL V100 调度合成与性能评测设计

**日期：** 2026-09-01

**修订：** 2026-09-02，根据 Gurobi 许可证只读预检调整控制端与 MPI 启动端。

**目标：** 基于现有 V100 2×4 与 2×8 拓扑、AllGather/AllReduce sketch 和默认 atom，完成 24 个调度的合成、离线验证、在线校准、在线验证与调优，并在 node2/node4 上使用同一套 MSCCL 运行时测试 VeriCCL XML 和 `/home/zdl/MSCCL/test` 中的对照 XML。保存完整日志，最终比较仅使用 in-place `algbw`。

## 1. 范围与成功标准

本任务覆盖：

- 拓扑：`v100-n2g4`、`v100-n2g8`。
- 算子：AllGather、AllReduce。
- atom：固定使用 `vericcl/examples/atom/default.json`。
- nccl-tests 消息大小：4 MiB、16 MiB、64 MiB、256 MiB、1 GiB、2 GiB。
- 每个 VeriCCL 输入生成一个最终 XML，共 `2 × 2 × 6 = 24` 个 XML。
- 对每个最终 XML 保存求解、离线验证、BDD 分析、在线校准、在线算子验证和调优结果。
- 在 node2/node4 上测试全部 24 个 VeriCCL XML，并测试 `/home/zdl/MSCCL/test` 中与 Rank 数和算子匹配的全部对照 XML。
- 原始 nccl-tests 输出必须完整保留，包括 out-of-place 与 in-place 的 `time/algbw/busbw/#wrong` 列。
- 汇总报告只以 in-place `algbw` 为主要性能指标，同时记录 XML 是否被确认走 MSCCL。

“最优调度”定义为：在全局 `K_max=16`、当前候选策略和校准后链路模型形成的受限搜索空间中，通过全部强制验证并取得最佳实测结果的候选。只有完整模型返回严格最优证明时，报告才允许写 `proven_optimal=true`；超时后的可行解、模板组合和构造式候选不得表述为数学全局最优。

## 2. 安全边界

必须满足以下约束：

- 严禁读取、写入、列举或修改 `/home/cc` 及其任何子路径。
- 不修改 GPU、NIC、InfiniBand、以太网、驱动、固件、内核、路由、接口状态或系统权限。
- 只允许读取硬件状态和运行用户态程序，例如 `nvidia-smi`、`mpirun`、nccl-tests 和 MSCCL。
- 远程写入仅限用户已授权的 `/home/zdl` 及其子路径。
- `/home/zdl/MSCCL/test` 作为只读基线源；测试前复制到隔离实验目录，绝不改写原文件。
- node2 当前 InfiniBand 链路不可用，因此不得尝试启用或修复。node2/node4 测试使用现有 10.0.0.0/24 以太网路径，并设置 `NCCL_IB_DISABLE=1`。最终报告必须明确：该结果验证的是当前 TCP/Ethernet 环境，不是 IB 性能。

## 3. 仓库与提交顺序

现有可扩展分层求解工作树中包含已经通过完整测试的在线运行时修复，但尚未提交；主工作区还包含用户的 README、论文叙事文件和未跟踪的 `exp/` 输入。集成顺序固定为：

1. 在当前功能分支只提交已经验证的在线运行时修复和相应文档、测试。
2. 再次执行完整测试、`git diff --check`、Python 编译检查和代码非 ASCII 检查。
3. 在主工作区使用包含未跟踪文件的安全 stash 保存用户改动。
4. 将功能分支快进合并到 `feature/vericcl-implementation`，再次测试并推送。
5. 恢复 stash；若 README 存在重叠，保留用户路径修改并合并已验证内容，不覆盖 `docs/figures/`、`docs/vericcl-paper-story.md` 或 `exp/`。
6. 将现有 `exp/topo` 与 `exp/sketch` 输入纳入仓库，排除 `.DS_Store`。
7. 后续实验支持代码和结果分别在独立提交中完成；只有通过对应验证的提交才允许合并和推送。

不使用重置工作区、强制检出或其他破坏性 Git 命令。

## 4. 全局并发上限

全局并发上限固定为 16：

- 求解器外层 channel 搜索范围为 `K=1..16`。
- 在线校准范围为
  `k=1..min(16, max_calibration_channels, 128 MiB / S, link_max_channels)`。
- `vericcl/input/models.py`、输入加载默认值、校准常量、示例 sketch、运行文档和工作文档中的软件默认上限统一改为 16。
- 24 个实验 sketch 的 `solver.max_channels` 与 `hyperparameters.max_calibration_channels` 都改为 16。
- topo 中链路和共享资源的 `max_channels=32` 仍表示物理输入容量，不改写为 16；所有软件搜索和校准还会受到全局 16 的限制。
- 旧报告、历史示例结果和已冻结的旧计划文档不做追溯性改写。

## 5. 输入大小与运行大小

AllGather 的 `total_size_bytes` 表示每 Rank 输入大小，因此 nccl-tests 的全局 `size` 为 `rank_count × total_size_bytes`。AllReduce 的 `total_size_bytes` 直接等于 nccl-tests 的 `size`。`slice_size_bytes` 保持现有真实软件传输粒度，不为校准或测试自动改变。

| 拓扑 | 算子 | nccl-tests size | `total_size_bytes` | `slice_size_bytes` |
|---|---|---:|---:|---:|
| 2×4 | AG | 4 MiB | 512 KiB | 512 KiB |
| 2×4 | AG | 16 MiB | 2 MiB | 1 MiB |
| 2×4 | AG | 64 MiB | 8 MiB | 1 MiB |
| 2×4 | AG | 256 MiB | 32 MiB | 1 MiB |
| 2×4 | AG | 1 GiB | 128 MiB | 1 MiB |
| 2×4 | AG | 2 GiB | 256 MiB | 2 MiB |
| 2×8 | AG | 4 MiB | 256 KiB | 256 KiB |
| 2×8 | AG | 16 MiB | 1 MiB | 1 MiB |
| 2×8 | AG | 64 MiB | 4 MiB | 1 MiB |
| 2×8 | AG | 256 MiB | 16 MiB | 1 MiB |
| 2×8 | AG | 1 GiB | 64 MiB | 1 MiB |
| 2×8 | AG | 2 GiB | 128 MiB | 1 MiB |
| 2×4/2×8 | AR | 4 MiB | 4 MiB | 512/256 KiB |
| 2×4/2×8 | AR | 16 MiB | 16 MiB | 1 MiB |
| 2×4/2×8 | AR | 64 MiB | 64 MiB | 1 MiB |
| 2×4/2×8 | AR | 256 MiB | 256 MiB | 2 MiB |
| 2×4/2×8 | AR | 1 GiB | 1 GiB | 8 MiB |
| 2×4/2×8 | AR | 2 GiB | 2 GiB | 16 MiB |

表中 AR 的 4 MiB slice 分别为：2×4 使用 512 KiB，2×8 使用 256 KiB。全部 sketch 保持 `inplace=true`。

## 6. 校准策略

在线链路校准只测试 128 MiB、2 机×1 卡的 inter-node 链路。原因是 `v100-n2g8` 的机内链路包含 NVLink 与较慢的非 NVLink 两类性能，当前按单一 `intra_node` 类应用校准会把非同构链路错误合并。机内链路继续使用 topo 中的保守参数。

两个拓扑的不同 slice 集合分别为：

- `v100-n2g4`：512 KiB、1 MiB、2 MiB、8 MiB、16 MiB，对应 `16+16+16+16+8=72` 个校准点。
- `v100-n2g8`：256 KiB、1 MiB、2 MiB、8 MiB、16 MiB，对应 `16+16+16+16+8=72` 个校准点。

总计最多 144 个 inter-node 校准点。相同拓扑、slice、协议、运行时和路径参数下，AG 与 AR 复用稳定缓存；不稳定点必须重新测量。每个 `k` 仍使用全部 128 MiB 数据，不插值、不外推、不提前停止。

跨节点端点时钟不确定性不能阻断校准。`PhysicalTransferInterval` 保留现有跨端点 `physical_start/physical_end`，同时保存发送端本地的开始和结束时间。校准波次中的全部发送均来自同一 Rank，因此以发送端本地时间计算
`D_wave = max(send_end) - min(send_start)`，偏移量相消，不需要比较不同主机的绝对时钟。该调整只用于链路校准；算子依赖等待、端点配对和在线调优仍使用现有全局对齐时间与不确定性规则。

## 7. 求解、离线验证与调优流程

每个输入执行同一流程：

1. 解析 topo、sketch、atom，验证输入和分层计划。
2. 使用模板路由与全局调度，在 `K=1..16` 内搜索候选。
3. 完成语义、状态回放、拓扑、资源、时间、buffer、endpoint、deadlock、XML 和动态事件模拟验证。
4. BDD 查找 flow 分流和 TB 顺序调优机会；只在保持目标约束与最终算子语义正确时执行局部替换、后缀依赖修复、时间重排和比较选择。
5. 首次在线阶段测量 inter-node 曲线并要求重新求解；稳定缓存命中时直接使用已有曲线。
6. 对重新求解的最终候选执行 nccl-tests 正确性、稳定性和 step trace 验证。
7. 仅当 trace 完整、时钟不确定性满足阈值且比较可信时，允许在线调优；否则保留有效 XML，并在报告中记录 `tuning_eligible=false` 及原因。
8. 选择所有强制验证通过且实测性能最好的候选作为最终 XML。

实验环境将 `VERICCL_MAX_CLOCK_UNCERTAINTY_US` 设为 50。提高阈值不等于忽略不确定性；若端点比较仍不确定，在线调优必须禁用。离线 BDD 验证不受该状态影响。

## 8. 非共享文件系统支持

node2 与 node4 的 `/home/zdl` 不是共享文件系统。2026-09-02 的只读预检确认：node4 的 Gurobi 13.0.3 许可证 `2634413` 已过期，node2 的 Gurobi 13.0.2 许可证 `2802355` 可正常求解。不得复制、修改或更新许可证。完整 VeriCCL/Gurobi 工作流因此在 node2 运行，但所有 MPI/nccl-tests 命令仍由 node2 通过 SSH 委托 node4 发起。默认单机行为保持不变，并增加可注入依赖：

```python
build_online_context_factory(
    environment,
    *,
    executor=None,
    trace_collector=None,
)
```

- `executor=None` 时仍创建 `SubprocessCommandExecutor`。
- `trace_collector=None` 时仍使用 `collect_trace_files`。
- 实验运行器提供 remote staging executor：发现 `MSCCL_XML_FILES` 后，先将 node2 上的 XML 原子复制到 node4 的同一路径，再通过 SSH 在 node4 执行完整命令并原样返回 stdout、stderr 和退出状态。
- 2×4 hostfile 中 node4 对应 Rank 0–3，node2 对应 Rank 4–7；2×8 中 node4 对应 Rank 0–7，node2 对应 Rank 8–15。
- 实验 trace collector 在分析前将 node4 生成的前半 Rank trace 复制回 node2 的本地 trace 目录；node2 生成的后半 Rank trace 已经位于控制端。全部文件齐备后再调用标准 collector。
- 所有 SSH/SCP 命令使用参数数组和固定主机清单，不拼接 shell 字符串。
- node2 与 node4 使用相同的 `/home/zdl/VeriCCL-experiments/<run-id>` 路径。remote executor 只写入该目录，不允许写入其他远程路径。

对应单元测试覆盖默认依赖、注入依赖、XML staging、缺失远程文件、trace 汇集和失败传播。

## 9. MSCCL 运行时与 MPI 环境

VeriCCL 和对照 XML 使用同一套已验证的 patched MSCCL runtime，避免把运行时差异误当作调度差异。运行时满足：

- device 与 host proxy 的 `MSCCL_CHUNKSTEPS=4`、`MSCCL_SLICESTEPS=4` 一致。
- `NCCL_PROTO=Simple`。
- `NCCL_ALGO=MSCCL,RING`，使匹配的 in-place 区段使用 MSCCL，另一放置模式回退到 Ring。
- VeriCCL XML 使用 `NCCL_BUFFSIZE=2 × slice_size_bytes`。
- 对照 XML 按用户给出的参考配置使用 `NCCL_BUFFSIZE=2097152`。

MPI 测试由 node2 控制端通过 SSH 在 node4 发起，生成两个隔离 hostfile，并在两台机器的相同路径保存相同内容：

- 2×4：`10.0.0.104 slots=4` 与 `10.0.0.102 slots=4`。
- 2×8：`10.0.0.104 slots=8` 与 `10.0.0.102 slots=8`。

不得使用现有 `/home/zdl/MSCCL/hostfile`，因为其中包含不属于本次 node2/node4 组合的 `10.0.0.101`。MPI 保留 OpenMPI 4.1.8、`pml ob1`、`btl tcp,self,vader` 和 `btl_tcp_if_include=10.0.0.0/24`。NCCL 设置 `NCCL_SOCKET_IFNAME=eno0,enp4s0`、`NCCL_IB_DISABLE=1`、`NCCL_P2P_LEVEL=NVL`、`NCCL_IGNORE_DISABLED_P2P=1`、`NCCL_NET_GDR_LEVEL=0`、`NCCL_NET_GDR_READ=0`，并按拓扑设置可见 GPU 数量。node2 与 node4 的 MSCCL、nccl-tests、clock helper、hostfile 和实验根目录必须具有相同绝对路径；预检不通过时停止实验，不创建系统级软链接或修改系统路径。

## 10. 性能测试矩阵

每个 VeriCCL XML 只测试其对应的一个消息大小，调用相应的 `all_gather_perf` 或 `all_reduce_perf`，使用 `-g 1 -n 15`，并保留完整 stdout/stderr。

对照 XML 先读取根属性，按 `coll` 与 `ngpus` 分组。只有与当前算子和 Rank 数匹配的 XML 才运行：

- 2×4：5 个 AG 对照、1 个 AR 对照。
- 2×8：9 个 AG 对照、8 个 AR 对照。

每个对照 XML 使用一次 `-b 4M -e 2G -f 2 -g 1 -n 15` 运行，原始日志保留全部输出行；最终汇总只提取 4 MiB、16 MiB、64 MiB、256 MiB、1 GiB、2 GiB 六个大小。由此共有 24 次 VeriCCL 性能运行和 23 次对照性能运行，共 47 次主性能运行；在线验证与校准运行另行统计。

所有运行必须满足 `#wrong=0`。失败、超时或不支持某个大小时不删除日志，也不以其他运行替代，而是在矩阵中记录明确状态。

## 11. MSCCL 激活判定

每次测试同时检查：

1. NCCL INFO 中存在 XML 成功加载/连接的证据。
2. 对于 `inplace=true` XML，in-place 与 out-of-place 的 `busbw` 相对差异满足
   `abs(B_ip - B_oop) / max(B_ip, B_oop) >= 0.05`。

两项都满足时标记 `msccl_activation=confirmed`。任一项不满足时，保留性能数据但标记 `msccl_activation=unconfirmed`，不得把该数据用于“VeriCCL 优于对照”的结论。内部在线验证还要求选定区段存在 MSCCL step trace；没有 trace 时不得允许在线调优。

## 12. 结果目录与可恢复执行

远程完整实验以 node2 为控制端，并在 node2/node4 的相同绝对路径保存控制文件或暂存文件：

```text
/home/zdl/VeriCCL-experiments/2026-09-01-v100-k16/
```

仓库内可提交结果保存在：

```text
exp/results/2026-09-01-v100-k16/
  manifest.json
  calibration/
  vericcl/<topology>/<collective>/<size>/
  baselines/<topology>/<collective>/<xml-name>/
  logs/
  summary/results.csv
  summary/report.md
```

仓库保存最终 XML、schedule、validation JSON、校准曲线、命令清单、输入与二进制哈希、全部 nccl-tests 文本日志、结构化 CSV/JSON 和总结报告。体积较大的原始二进制 step trace 保留在远程完整实验目录，仓库 manifest 记录路径、大小和 SHA-256；解析后的 trace 摘要进入仓库。

实验运行器为每个任务记录 `pending/running/passed/failed/skipped` 状态，并采用原子写入。重启后跳过输入哈希、代码提交、环境签名和输出哈希均匹配的已完成任务；不稳定校准点与未完成任务重新执行。空输出目录不能表示成功。

## 13. 报告内容

最终报告至少包含：

- VeriCCL、MSCCL runtime、nccl-tests 的提交或构建哈希。
- topo、sketch、atom 和每个 XML 的 SHA-256。
- 当前硬件与网络只读信息，以及 `NCCL_IB_DISABLE=1` 的原因。
- 每个输入的求解状态、超时、gap、候选数、选中 K、离线验证、BDD 调优、在线校准、在线调优和 XML 激活状态。
- 每个输出 XML 使用的调优策略。
- 每个大小的 VeriCCL 与对照 in-place `algbw`，以及相对提升。
- `#wrong`、in-place/out-of-place `busbw` 和激活判定只作为正确性与运行路径证据，不作为最终排序指标。
- 所有失败、未确认激活、时钟不确定、在线调优不可用和 Ethernet 环境限制。

## 14. 测试与验收

代码修改遵循测试驱动流程。最低验收包括：

- 新增/修改单元测试覆盖 K=16 默认值、校准有效上限、同源本地校准时长、可注入 executor/collector、远程 staging、结果解析和恢复执行。
- `resolve_inputs` 与 `build_plan` 验证全部 24 个输入。
- 完整 pytest 套件通过；Gurobi 专用测试按可用许可证执行并单独报告。
- `git diff --check`、Python `compileall` 和代码文件非 ASCII 扫描通过。
- 每个最终 XML 通过离线语义、资源、时间、buffer、endpoint、deadlock 和 XML 验证。
- node2/node4 nccl-tests 的目标区段 `#wrong=0`。
- 原始性能表、命令、环境白名单和退出状态全部可追溯。
- 合并和推送前重新执行与修改风险相称的完整验证；不以历史测试结果代替当前工作树验证。
- Python、C/C++、CUDA、shell 与测试代码中不得出现中文或其他非 ASCII 字符；中文仅用于文档和报告。
