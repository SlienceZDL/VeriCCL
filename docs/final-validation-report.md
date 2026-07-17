# VeriCCL最终验收报告

验收日期：2026-07-17。验收分支：`feature/vericcl-implementation`。

## 结论

公开CLI、八类算子语义链路、六类直接求解算子、分层组合、XML生成、离线验证、BDD机会分析、动态事件模拟、在线校准编排及报告链路均通过纯软件测试。纯软件覆盖率为91.62%。Gurobi矩阵通过；当前主机未配置GPU、MSCCL和`nccl-tests`硬件环境，因此硬件矩阵为`not_run`，不声明实机性能或逐step trace验证已完成。

## 测试矩阵

| 范围 | 命令 | 结果 |
|---|---|---|
| 文档命令 | `.venv/bin/python -m pytest tests/integration/test_documented_commands.py -q` | 2 passed |
| 纯软件与覆盖率 | `.venv/bin/python -m pytest -m 'not hardware and not gurobi' --cov=vericcl --cov-report=term-missing --cov-fail-under=90 -q` | 1105 passed, 22 deselected, 91.62% |
| Gurobi | `.venv/bin/python -m pytest -m gurobi -q` | 14 passed, 1113 deselected |
| 硬件 | `.venv/bin/python -m pytest -m hardware -q` | 8 skipped, 1119 deselected；`not_run` |

纯软件覆盖率配置仅排除由独立Gurobi矩阵负责的`vericcl/solver/milp.py`；该后端未从功能测试或Gurobi验收中移除。

## 静态与运行时检查

- `rg -n '[\p{Han}]' vericcl tests runtime -g '*.{py,c,cc,cu,cuh,h,json,xml}'`：无匹配，代码与生成格式中没有中文字符。
- `rg -n 'taccl' vericcl tests setup.py -g '*.{py,json,xml}'`：仅保留`vericcl/provenance.py`中的外部格式字面量`taccl_topology_v2`，由白名单测试约束。
- `.venv/bin/python -m compileall -q vericcl`：通过。
- `git diff --check`：通过。
- `.venv/bin/python runtime/msccl-trace/tools/verify_patch.py --source-root /Users/zdl/work/code/MSCCL_TIME`：`verification passed`。该检查验证补丁可干净应用及二进制trace布局，不等同于本机GPU构建或实机执行。

## 在线校准验收范围

纯软件测试已验证以下控制流：

1. 机内`nccl-tests`使用单进程`-g rank_count`，机间使用每Rank一个`-g 1`进程。
2. 对所选链路类别生成128 MiB、全部整数并发度的Broadcast基准XML。
3. 只用完整wave的逐step物理区间计算`D_safe(k)`；尾部wave执行但不参与估计。
4. 环境签名完全匹配时复用持久化校准点；`force_recalibrate`绕过缓存。
5. 稳定校准保留各链路或共享资源原有`alpha`，更新同构类别的`invbw`和`B_link(k)`。
6. 校准传播使用锚定有向边的完整通信域规范签名，并将channel上限限制到最大实测并发度。
7. `solve --online`更新拓扑、重建Plan并二次求解，同时保留首轮候选和父子血缘；`verify --online`保持输入XML不变并报告`requires_resolve=true`。
8. Trace以MSCCL `workIndex`标识集合调用，排除setup后严格分析20次正式调用，并按XML step数自动计算安全缓冲区下限。

本机缺少硬件前置条件，因此release统计、跨Rank时钟同步、真实trace完整性及线上瓶颈定位均记录为`not_run`。每次在线执行只校准`VERICCL_CALIBRATION_LINK_CLASS`指定的代表类别，其他类别继续使用topo输入中的保守参数。

## 可复核样例

接受样例使用2 Rank、8 MiB AllReduce、1 MiB slice及构造式后端：

- [最终XML](examples/vericcl_allreduce_8MiB_acceptance/vericcl_allreduce_8MiB_final.xml)
- [最终验证报告](examples/vericcl_allreduce_8MiB_acceptance/vericcl_allreduce_8MiB_final.validation.json)
- [最终调度sidecar](examples/vericcl_allreduce_8MiB_acceptance/vericcl_allreduce_8MiB_final.schedule.json)
- [运行摘要](examples/vericcl_allreduce_8MiB_acceptance/run-summary.json)

样例输出包含2个候选。最终候选的`tuning_strategy.kind`为`initial_solve`，`selected_best=true`，`proven_optimal=false`，`runtime_compatible=true`；BDD报告6个可调优flow提示但不改变正确性状态。最终XML SHA-256为`7cc9b62698ac68a44631d699925b036c089db5bd89c882b24adc8e8a637ec92f`，artifact binding SHA-256为`f3b5214a9ccd3f7e04380614643473bdbe52c5ccfe77d7d3da6b0e2074b47f26`。

## 环境限制

- 当前Gurobi为受限的非生产许可证；测试通过不构成生产许可声明。
- 硬件矩阵需要按[运行时配置](runtime-configuration.md)提供GPU、MSCCL、`nccl-tests`、MPI及trace工具。
- 接受样例是离线生成结果，`online`维度为`not_run`；不得将其作为实测性能证据。
