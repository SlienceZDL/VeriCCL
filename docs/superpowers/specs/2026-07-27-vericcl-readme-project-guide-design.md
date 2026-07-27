# VeriCCL README项目指南设计

日期：2026-07-27

## 目标

将英文`README.md`和中文`README.zh-CN.md`整理为面向研究人员和工程使用者的项目指南。文档参考SyCCL的求解器安装、输入配置、运行与扩展说明，以及MSCCL的运行时构建、XML加载和实机评测说明。

README必须让新用户能够完成以下主流程：

1. 获取并安装VeriCCL。
2. 理解topology、sketch和atom三类JSON输入。
3. 使用仓库内示例执行`solve`。
4. 使用生成的XML和sidecar执行`verify`。
5. 识别主要输出、退出码和失败原因。
6. 在需要实机评测时继续配置MSCCL与NCCL Tests。

## 文档边界

- 主README不提供Ubuntu软件包安装和操作系统级预检命令。
- 服务器、CUDA、MPI和运行环境的详细配置由`docs/runtime-configuration.md`承载，主README提供明确链接。
- 主README继续保留VeriCCL安装、MSCCL策略A与策略C、NCCL Tests构建、在线验证及单机/多机XML执行说明。
- 不增加流程图或其他图形。
- 不修改VeriCCL命令行接口、输入schema、求解逻辑、运行时补丁或测试语义。
- 不宣称当前主机已完成CUDA编译、GPU运行或性能验证。

## README结构

### 1. Overview

说明VeriCCL的项目定位、输入、输出和能力边界：

- 输入为topology、sketch和atom JSON。
- 输出为MSCCL XML、schedule sidecar及验证报告。
- 直接求解Broadcast、Reduce、AllGather、AllReduce、AllToAll和ReduceScatter。
- Scatter和Gather作为分阶段组合内部算子，与前述六类算子共同构成八类完整集合通信语义。
- 功能覆盖规划、求解、阶段组合、XML生成、离线验证、BDD调优机会分析、动态事件模拟和可选在线验证。

### 2. Building and Installing VeriCCL

采用SyCCL式连续操作流程，提供SSH和HTTPS两种克隆方式。主路径依次执行：

1. `git clone`。
2. `cd VeriCCL`。
3. 创建Python虚拟环境。
4. 更新`pip`、`setuptools`和`wheel`。
5. 安装`requirements-dev.txt`。
6. 以editable模式安装VeriCCL。
7. 使用`pip check`、导入检查和`vericcl --version`验证安装。

Gurobi许可证作为VeriCCL求解依赖在本节之后单独说明。构造式后端的无MILP示例继续作为无需完整Gurobi许可证的基础检查路径。

### 3. Running VeriCCL

首先说明三类输入及现有示例位置，然后提供连续的最小运行流程：

1. 设置新的输出根目录。
2. 执行`vericcl solve`。
3. 定位生成的最终XML。
4. 执行`vericcl verify`。
5. 检查最终验证报告和运行摘要。

现有四条文档测试标记必须保留。命令必须使用仓库中真实存在的`two_rank.json`、`allreduce_8m_1m.json`和`constructive.json`，并保持可由测试直接执行。

### 4. Input Configuration

分别解释：

- topology：全局Rank、节点、网关、有向链路、channel上限、`alpha`、`invbw`和共享资源。
- sketch：算子语义、数据大小、slice大小、原地属性、求解超时和调优参数。
- atom：阶段约束、禁用传输、求解策略和可选手工分层计划。

本节列出可运行示例，并明确`legacy`与`templates`目录仅供参考。未知字段处理应与输入解析器的真实行为一致。

### 5. Advanced Usage

保留并整理以下高级功能：

- CLI语义覆盖与`--override-input`。
- `--tune`与`--timeout-s`。
- 自动和手工分层求解。
- 六类直接求解算子及八类最终语义。
- 在线求解与在线验证的差异。

所有说明必须使用现有代码、测试和示例能够验证的术语，不引入尚未实现的配置接口。

### 6. MSCCL Runtime Evaluation

保留以下内容：

- 策略A：官方MSCCL固定commit与仓库内补丁。
- 策略C：`VeriCCL-MSCCL`固定tag。
- NCCL Tests和时钟辅助程序构建。
- XML传递方式、`NCCL_ALGO=MSCCL`、Simple协议、buffer/slice约束和单XML约束。
- 单节点与多节点执行命令。
- 运行时加载自定义XML的确认方法。

文档必须区分静态补丁验证、CUDA编译、MSCCL加载、nccl-tests正确性检查和性能测量。未执行的硬件步骤标记为`not_run`，不得由纯软件测试推断成功。

### 7. Extending VeriCCL

参考SyCCL的扩展章节，按实际仓库模块提供开发导航：

- `vericcl/input`：输入解析和schema。
- `vericcl/topology`：拓扑与共享资源建模。
- `vericcl/planner`：通信组和阶段计划。
- `vericcl/solver`：构造式与MILP求解。
- `vericcl/composer`：阶段组合。
- `vericcl/xml`：MSCCL XML降低。
- `vericcl/verification`：语义、依赖、资源和BDD验证。
- `vericcl/tuning`：候选替换、修复和比较。
- `vericcl/verification/online`：在线校准、trace和执行编排。

本节仅说明修改入口和依赖边界，不承诺稳定插件API。

### 8. Outputs, Limitations, and Troubleshooting

集中说明：

- 输出目录、最终文件、候选文件及报告。
- 退出码`0`、`2`、`3`、`4`和`5`。
- Gurobi许可证限制。
- runtime-compatible与offline-valid的区别。
- 单XML、slice大小、`cnt=1`和固定step参数。
- MPI、MSCCL加载、trace、时钟同步和输入错误的诊断方向。

### 9. License and Citation

仓库当前没有`LICENSE`、`NOTICE`或`CITATION.cff`。README使用以下明确状态，不推断许可证或论文信息：

```text
License: To be determined.
Citation: To be determined.
```

## 双语一致性

英文和中文README采用相同章节顺序，并保持以下内容一致：

- Bash命令块的数量、顺序和字节内容。
- 文件路径、版本、commit、tag、环境变量、参数和预期结果字面量。
- 算子、输入schema、输出、退出码和运行时限制。
- License/Citation状态。

自然语言分别使用专业英文和简洁、连贯的中文。

## 验证策略

扩展`tests/integration/test_documented_commands.py`，验证：

1. 两份README包含新的核心章节。
2. 两份README的Bash命令块完全一致。
3. 文档内声明的仓库相对路径真实存在且被Git跟踪。
4. 四条文档测试标记顺序不变且命令可执行。
5. 安装、运行、扩展、MSCCL评测和License/Citation状态均存在。
6. 主README不再包含Ubuntu软件包安装或系统预检章节。

继续运行workflow artifact测试、runtime patch测试和完整非硬件测试门禁。CUDA/MSCCL编译与nccl-tests实机执行仅能在目标Ubuntu GPU服务器上验证。
