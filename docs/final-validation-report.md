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

## 2026-07-22 安装指南验收

本节保留本次执行的新鲜证据，验证对象为英文与中文安装指南、固定版本MSCCL补丁及公开集成仓库。

### 来源与公开引用

| 命令 | 结果 |
|---|---|
| `python3 runtime/msccl-trace/tools/verify_patch.py --source-root /tmp/vericcl-msccl-base` | 退出码0；`verification passed` |
| `python3 runtime/msccl-trace/tools/verify_patch.py --source-root /tmp/vericcl-msccl-fork --patched-tree` | 退出码0；`verification passed` |
| `git ls-remote https://github.com/SlienceZDL/VeriCCL-MSCCL.git vericcl-runtime vericcl-runtime-v0.1.0` | 退出码0；分支为`782ee5f72cf48c1ae1a2365bcf525019f5620175`，标注标签对象为`8f4185a4d0d917209c34e3c4710b784e7f7a2f64` |
| `git ls-remote git@github.com:SlienceZDL/VeriCCL-MSCCL.git vericcl-runtime vericcl-runtime-v0.1.0` | 退出码0；SSH返回与HTTPS相同的分支和标签引用 |

本地标签类型检查结果为`tag`，解引用后指向`782ee5f72cf48c1ae1a2365bcf525019f5620175`。

### 测试矩阵

| 范围 | 命令 | 结果 |
|---|---|---|
| 运行时补丁与文档工作流 | `.venv/bin/python -m pytest tests/unit/online/test_runtime_patch.py tests/integration/test_documented_commands.py tests/integration/test_workflow_artifacts.py -q` | 退出码0；51 passed, 1 skipped |
| 完整纯软件与覆盖率 | `.venv/bin/python -m pytest -m 'not hardware and not gurobi' --cov=vericcl --cov-report=term-missing --cov-fail-under=90 -q` | 退出码0；1134 passed, 1 skipped, 22 deselected；92.10% |
| Gurobi | `.venv/bin/python -m pytest -m gurobi -q` | 退出码0；14 passed, 1143 deselected |
| 硬件 | `.venv/bin/python -m pytest -m hardware -q` | 退出码0；8 skipped, 1149 deselected；`not_run` |

### 静态检查与限制

| 命令 | 结果 |
|---|---|
| `python3 -m compileall -q vericcl tests runtime/msccl-trace/tools` | 退出码0 |
| `git diff --check` | 退出码0；无格式错误 |
| `rg -n '[\p{Han}]' vericcl tests runtime --glob '*.py' --glob '*.c' --glob '*.cc' --glob '*.cu' --glob '*.h' --glob '*.json'` | 退出码1；无匹配，受检代码与机器可读文件未发现中文字符 |

验收主机为macOS/Darwin纯软件环境，覆盖率命令使用Python 3.9.6；该环境不替代安装指南规定的Ubuntu 22.04/24.04与Python 3.10-3.12目标环境。CUDA/MSCCL编译和`nccl-tests`执行均为`not_run`；上述补丁一致性、静态检查与跳过的硬件测试不构成Ubuntu GPU服务器上的编译、功能或性能验证证据。

## 2026-07-27 README项目指南验收

本节追加本次新鲜软件验证证据，不替代或重新解释前述验收记录。

### 文档与软件测试矩阵

| 范围 | 命令 | 结果 |
|---|---|---|
| README、工作流与运行时补丁文档门禁 | `.venv/bin/python -m pytest tests/integration/test_documented_commands.py tests/integration/test_workflow_artifacts.py tests/unit/online/test_runtime_patch.py -q` | 退出码0；54 passed, 1 skipped |
| 完整硬件无关覆盖率门禁 | `.venv/bin/python -m pytest -m 'not hardware and not gurobi' --cov=vericcl --cov-report=term-missing --cov-fail-under=90 -q` | 退出码0；1137 passed, 1 skipped, 22 deselected；总覆盖率92.10%（阈值90%） |
| Gurobi 标记矩阵 | `.venv/bin/python -m pytest -m gurobi -q` | 退出码0；14 passed, 1146 deselected |
| hardware 标记矩阵 | `.venv/bin/python -m pytest -m hardware -q` | 退出码0；8 skipped, 1152 deselected；`not_run` |

文档门禁包含中英文 README Bash 块逐字一致性检查、八个分阶段命令的执行，以及公开 `help`、`solve`、`verify` 与示例验证命令的依次执行。完整扫描两份 README 后，各解析出17个仓库相对路径，均存在且受 Git 跟踪；`Ubuntu` 和 `SyCCL` 的 README 扫描无匹配，排除内容未出现。

MSCCL文档行为测试在隔离临时目录中执行策略A与策略C命令块，确认二者分别调用`--base-tree`与`--patched-tree`验证器并生成预期构建目录。激活测试执行单节点与MPI探测控制流：算法数为1时通过，为2时拒绝；MPI探测传播`NCCL_DEBUG`和`NCCL_DEBUG_SUBSYS`，正式release命令清除这些变量。上述测试使用受控命令替身，不构成CUDA编译、MSCCL加载或GPU执行证据。

唯一跳过项为`tests/unit/online/test_runtime_patch.py::test_patch_dry_run_and_post_apply_source_scan`，原因是当前主机未提供有效的`VERICCL_MSCCL_REFERENCE_ROOT`参考源码树。

### 静态检查

| 命令 | 结果 |
|---|---|
| `python3 -m compileall -q vericcl tests runtime/msccl-trace/tools` | 退出码0 |
| `git diff --check` | 退出码0；无格式错误 |
| `rg -n '[\p{Han}]' vericcl tests runtime --glob '*.py' --glob '*.c' --glob '*.cc' --glob '*.cu' --glob '*.h' --glob '*.json'` | 退出码1；无匹配（`rg` 用退出码1表示未找到匹配），受检源码与 JSON 未发现中文字符 |

验收主机仍为 macOS/Darwin 纯软件环境。CUDA/MSCCL 编译和 `nccl-tests` GPU 执行均为`not_run`；hardware 标记测试的跳过结果不构成 GPU、CUDA、MSCCL 或 `nccl-tests` 的实机验证证据。

## 2026-07-28 Python 3.13与dd依赖兼容验收

本节记录 Python 3.13 隔离安装、BDD 回归、完整软件测试及 Linux wheel 制品检查的独立验证结果。验收主机为 macOS/Darwin；安装、导入和测试均成功，但 `pip check` 因既有第三方 `z3-solver` wheel 标签不兼容而失败，二者在结论中分别记录。

### Python 3.13隔离安装与测试

| 范围 | 命令 | 结果 |
|---|---|---|
| 开发依赖安装 | `uv pip install --python /tmp/vericcl-py313.zK7JUd/.venv/bin/python -r requirements-dev.txt` | 退出码0；Python 3.13.13 环境，4个包已满足 |
| VeriCCL可编辑安装 | `uv pip install --python /tmp/vericcl-py313.zK7JUd/.venv/bin/python -e .` | 退出码0；构建并安装 VeriCCL 0.1.0 |
| 依赖导入与版本 | `/tmp/vericcl-py313.zK7JUd/.venv/bin/python -c 'import dd, vericcl; print(dd.__version__, vericcl.__version__)'` | 退出码0；`0.6.0 0.1.0` |
| BDD定向回归 | `/tmp/vericcl-py313.zK7JUd/.venv/bin/python -m pytest tests/unit/verification/test_bdd_flow.py tests/unit/verification/test_bdd_order.py -q` | 退出码0；20 passed |
| Python 3.13完整测试 | `/tmp/vericcl-py313.zK7JUd/.venv/bin/python -m pytest -q` | 退出码0；1159 passed, 9 skipped |
| 既有环境完整测试 | `.venv/bin/python -m pytest -q` | 退出码0；Python 3.9.6；1159 passed, 9 skipped |

`/tmp/vericcl-py313.zK7JUd/.venv/bin/python -m pip check` 退出码为1，唯一输出为 `z3-solver 5.0.0.0 is not supported on this platform`。该发行包的 `WHEEL` 元数据声明 `py3-none-macosx_13_3_arm64`，而当前 pip 26.1.2 的兼容标签包含 `py3-none-macosx_13_0_arm64`、不包含 `py3-none-macosx_13_3_arm64`。`import z3` 成功且 `z3.get_version_string()` 返回 `5.0.0`；该问题属于既有第三方 `z3-solver` wheel 标签，不由本次 `dd` 依赖变更引入，也未在本次任务中修复。因此，本节不声明 `pip check` 通过，但保留可编辑安装、依赖导入和测试通过的独立证据。

### Linux wheel制品检查

通过官方 PyPI JSON 端点检查 `dd` 0.5.7 和 0.6.0，脚本输出 `supported Linux dd wheels verified`，并确认以下制品存在：

- CPython 3.10：`dd-0.5.7-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`
- CPython 3.11：`dd-0.6.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`
- CPython 3.12：`dd-0.6.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`
- CPython 3.13：`dd-0.6.0-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`

该检查仅证明官方制品可用性，不构成 Linux 安装或执行证据。

### 静态门禁与环境限制

| 命令 | 结果 |
|---|---|
| `python3 -m compileall -q vericcl tests runtime/msccl-trace/tools` | 退出码0 |
| `git diff --check` | 退出码0；无格式错误 |
| `rg -n '[\p{Han}]' vericcl tests runtime setup.py --glob '*.py' --glob '*.c' --glob '*.cc' --glob '*.cu' --glob '*.h' --glob '*.json'` | 退出码1；无匹配，受检源码与机器可读文件未发现中文字符 |

Linux 执行、CUDA/MSCCL 编译、GPU 执行和 `nccl-tests` 性能测试均为`not_run`。PyPI 元数据、macOS 安装及跳过的硬件测试不得解释为上述平台或硬件验证。
