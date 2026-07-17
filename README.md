# VeriCCL

VeriCCL是一个面向GPU集群的集合通信调度生成与验证库。它读取拓扑、sketch和atom三类JSON输入，生成MSCCL XML，并执行语义、状态、时序、资源、缓冲区、死锁、BDD、事件模拟及可选在线验证。

当前直接求解Broadcast、Reduce、AllGather、AllReduce、AllToAll和ReduceScatter；Scatter与Gather作为其他算子的组合阶段使用。分层求解会将局部通信组结果合成为一个满足全局算子语义的调度。

## 安装

要求Python 3.9或更高版本。开发环境可按以下方式创建：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pip install -e .
```

## 输入

- `topo.json`：Rank、单向链路、channel、`alpha/beta/invbw`、节点、网关和共享资源。
- `sketch.json`：算子语义、消息大小、slice大小、目标模式、求解与验证预算。
- `atom.json`：求解策略、手动分层和禁用传输`(slice_id, src, dst, stage_id)`。

可直接使用以下纯软件示例：

- `vericcl/examples/topo/two_rank.json`
- `vericcl/examples/sketch/allreduce_8m_1m.json`
- `vericcl/examples/atom/constructive.json`

## 公共命令

<!-- vericcl-doc-test: help -->
```bash
.venv/bin/python -m vericcl --help
```

<!-- vericcl-doc-test: solve -->
```bash
.venv/bin/python -m vericcl solve --topology vericcl/examples/topo/two_rank.json --sketch vericcl/examples/sketch/allreduce_8m_1m.json --atoms vericcl/examples/atom/constructive.json --output-dir ${VERICCL_OUTPUT_DIR} --run-id docs
```

`solve`不修改输入文件。每次运行创建独立目录，并写入规范化输入、所有候选、逐候选报告、最终XML及摘要。

<!-- vericcl-doc-test: verify -->
```bash
.venv/bin/python -m vericcl verify --topology vericcl/examples/topo/two_rank.json --sketch vericcl/examples/sketch/allreduce_8m_1m.json --atoms vericcl/examples/atom/constructive.json --output-dir ${VERICCL_OUTPUT_DIR} --run-id docs-verify --xml ${VERICCL_OUTPUT_DIR}/vericcl_allreduce_8MiB_docs/vericcl_allreduce_8MiB_final.xml
```

<!-- vericcl-doc-test: example-validation -->
```bash
.venv/bin/python -m pytest tests/e2e/test_six_collectives.py -q
```

退出码为：`0`表示离线有效，包括仅有运行时兼容性警告的candidate XML；`2`表示输入或配置错误；`3`表示没有语义有效候选或离线超时；`4`表示请求的在线验证失败或超时；`5`表示内部错误。

指定`--online`时，CLI会执行所选链路类别的128 MiB校准、基础算子release测试和逐step trace诊断。稳定校准会使`solve`更新拓扑性能参数并二次求解；`verify`保持输入XML不变，并通过`requires_resolve`提示后续重新求解。完整环境变量与MSCCL参数见[运行时配置](docs/runtime-configuration.md)。

## 输出目录

```text
vericcl_<operator>_<scale>_<run_id>/
├── resolved-input.json
├── run-summary.json
├── schedules/
├── reports/
├── traces/
├── vericcl_<operator>_<scale>_final.xml
├── vericcl_<operator>_<scale>_final.schedule.json
└── vericcl_<operator>_<scale>_final.validation.json
```

MSCCL不可执行但离线有效的结果使用`.candidate.xml`后缀。报告会给出增大channel数或slice大小等重新求解建议，不修改当前调度。

## 进一步文档

- [工作设计文档](Vericcl-work-document.md)
- [运行时与MSCCL参数](docs/runtime-configuration.md)
- [验证报告说明](docs/validation-report.md)
- [最终验收报告](docs/final-validation-report.md)
- [迁移说明](MIGRATION.md)
- [MSCCL trace补丁](runtime/msccl-trace/README.md)

旧输入、模板和参考XML仅保存在`vericcl/examples/legacy`及`vericcl/examples/templates`，不参与新包的运行时导入。第三方来源与必须保留的外部格式标识由`MIGRATION.md`和`vericcl.provenance`记录。
