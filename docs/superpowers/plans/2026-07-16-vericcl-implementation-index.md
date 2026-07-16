# VeriCCL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将现有 TACCL/Canvas 原型迁移为满足 `Vericcl-work-document.md` 的 VeriCCL 集合通信求解、验证、XML 生成及在线诊断系统。

**Architecture:** 新建 `vericcl` Python 包，以不可变输入、统一语义状态和 `Schedule` 为核心边界；规划器、求解器、合成器、XML 降低器和验证器只能通过这些公开模型交换数据。实现按依赖顺序拆成七个可独立验收的阶段，前一阶段的公开接口是后一阶段唯一允许依赖的内部协议。

**Tech Stack:** Python 3.9+、dataclasses、argparse、Gurobi、lxml、Z3、BDD 包装层、pytest；MSCCL Simple 协议 XML；可选 CUDA/MSCCL 与 nccl-tests 硬件集成。

## Global Constraints

- 规范唯一来源为仓库根目录 `Vericcl-work-document.md`；实现不得自行改变其中已确认的算子、状态、缓冲区、求解或验证语义。
- 公开 atom 固定为 `[s, pt, t]`，`slice_id = source_rank * N + logical_slice_index`，`N = total_size_bytes / slice_size_bytes`。
- 仅支持等长 slice；`M > 0`、`S > 0`、`M % S == 0`，ReduceScatter 和 AllToAll 还要求 `N % P == 0`。
- 同一 `LaneKey(src_rank, dst_rank, channel)` 的传输区间不得重叠；不同方向与不同 channel 可并行，但共享有向链路及 NIC 总带宽。
- REDUCE 只能合并同一逻辑位置且 contributors 不相交的状态；不得创建新的外部 slice ID。
- 直接生成 XML 的算子为 Broadcast、Reduce、AllGather、AllReduce、AllToAll、ReduceScatter；Scatter 和 Gather 仅作为内部 PlanDAG 节点。
- XML 只生成 `s`、`r`、`rrc`、`cpy`、`nop`，每个 step 固定 `cnt="1"`，禁止融合和连续地址合并。
- XML 通信 TB 按 `(rank, direction, peer, channel)` 单向划分；每个 `transfer_id` 的两个端点必须同步排序并通过死锁模拟。
- `MSCCL_CHUNKSTEPS = 4`、`MSCCL_SLICESTEPS = 4`、`NCCL_BUFFSIZE = 2 * slice_size_bytes`，协议固定为 Simple。
- 默认 `K_max=32`、`solver_seed=0`、`total_solve_timeout_s=10800`、`per_model_timeout_s=1800`、`mip_gap=1e-4`。
- 默认 `max_tuning_iterations=20`、`total_verification_timeout_s=10800`，统计接受阈值遵循规范。
- 所有源代码、测试代码、生成 XML 与 JSON 诊断不得包含中文字符；第三方版权和明确的来源说明可以保留必要的 `TACCL` 名称。
- 每个阶段先写失败测试，再写最小实现，并在阶段结束时运行对应回归和中文字符扫描。
- 当前目录不是 Git 仓库，因此计划以“测试通过 + 文件清单复核”作为阶段检查点，不执行会失败的提交命令；用户建立 Git 仓库后再按阶段形成独立提交。
- 当前本机 Python 为 3.9.6，尚未安装 pytest 和 gurobipy；第一阶段负责建立开发依赖，Gurobi 或硬件不可用时必须显式报告 `not_run`。
- 所有 Python 类型注解兼容 3.9：可空类型和联合类型分别使用 `typing.Optional` 与 `typing.Union`，不得使用 Python 3.10 的 `X | Y` 语法。

---

## Phase Order

1. [01 Foundation, Inputs, and Semantics](2026-07-16-vericcl-01-foundation-input-semantics.md)
2. [02 Topology and Planning](2026-07-16-vericcl-02-topology-planning.md)
3. [03 Solving and Composition](2026-07-16-vericcl-03-solving-composition.md)
4. [04 Buffer Planning and MSCCL XML](2026-07-16-vericcl-04-xml-lowering.md)
5. [05 Offline Verification and Tuning](2026-07-16-vericcl-05-verification-tuning.md)
6. [06 Online Calibration and Step Tracing](2026-07-16-vericcl-06-online-validation.md)
7. [07 End-to-End Integration, Migration, and Acceptance](2026-07-16-vericcl-07-integration-acceptance.md)

阶段必须顺序执行。每个阶段只有在其公开接口测试和阶段回归通过后才能进入下一阶段；不得为满足后续测试而绕过当前阶段的不变量。

## Locked Public Interfaces

以下接口名在各阶段之间固定；若实现中必须改变，先更新全部受影响计划和规范映射，再修改代码。

```python
from vericcl.input.models import ResolvedInput
from vericcl.semantics.atom import Atom, Schedule, Transfer
from vericcl.semantics.collective import CollectiveSpec
from vericcl.semantics.state import PayloadState
from vericcl.topology.model import Topology
from vericcl.planner.model import PlanDAG
from vericcl.solver.model import SolveCandidate, SolveRequest, SolveResult
from vericcl.xml.model import BufferPlan, XmlArtifact
from vericcl.xml.endpoints import EndpointProgram
from vericcl.verification.model import ValidationReport
from vericcl.tuning.model import TuningOverlay
```

统一主流程签名固定为 `resolve_inputs(topology_path: Path, sketch_path: Path, atom_path: Path) -> ResolvedInput`、`build_plan(inputs: ResolvedInput, topology: Topology) -> PlanDAG`、`solve(request: SolveRequest) -> SolveResult`、`compose(plan: PlanDAG, candidates: Mapping[str, SolveCandidate]) -> Schedule`、`lower_to_xml(schedule: Schedule, inputs: ResolvedInput, topology: Topology) -> XmlArtifact` 和 `verify_candidate(schedule: Schedule, artifact: XmlArtifact, inputs: ResolvedInput, topology: Topology) -> ValidationReport`。

## Existing-to-New Code Map

| 现有代码 | 可迁移内容 | 新模块 | 迁移原则 |
| --- | --- | --- | --- |
| `taccl/collectives.py` | pre/post condition 与地址映射思路 | `vericcl/semantics/collective.py` | 以 contributors 集合重写，不沿用布尔“持有 chunk”模型 |
| `taccl/semantic/reduction.py` | 归约相交检查思路 | `vericcl/semantics/state.py` | 使用不可变 PayloadState、版本消费和精确错误类型 |
| `taccl/topologies/*.py` | 单向矩阵、switch/NIC 约束 | `vericcl/topology/` | 统一为显式 DirectedLink、LaneKey、SharedResource |
| `taccl/shortest_path_sets.py` | 最短路径候选 | `vericcl/topology/paths.py` | 保留所有等价最短路径并尊重 forbidden atom |
| `taccl/routing.py` | 路径选择变量和对称约束 | `vericcl/solver/milp.py` | 移除 big-M 常量散布，统一配置与状态语义 |
| `taccl/scheduler.py` | 传输时间和链路排序约束 | `vericcl/solver/scheduling.py` | 使用连续微秒、LaneKey 和共享资源并发模型 |
| `taccl/reduce_scheduler.py` | AG 反向构造归约的思路 | `vericcl/planner/dual.py` | 重建 REDUCE 状态与依赖，不反转 XML step |
| `taccl/heuristic_ordering.py` | 确定性初始排序 | `vericcl/solver/constructive.py` | 仅作为可行候选与 warm start |
| `taccl/ncclize.py` | MSCCL XML 字段与单依赖编码 | `vericcl/xml/` | 重写 BufferPlan、单向 TB、NOP 汇合和死锁验证 |
| `taccl/serialization.py` | 外部 `sccl_type` 兼容字段 | `vericcl/provenance.py` | 新主流程不依赖旧 Algorithm JSON；只在来源说明和 legacy 示例中保留固定字段 |
| `taccl/cli/*.py` | 入口参数及旧示例路径 | `vericcl/cli/` | 合并为 `solve`、`verify`，移除规模专用重复脚本 |
| `/Users/zdl/work/code/MSCCL_TIME` | 现有 global timer 与 interpreter 位置 | Phase 06 runtime patch | VeriCCL 仓库保存可复查 patch；硬件应用单独执行 |

## Cross-Phase Verification Matrix

| 不变量 | 首次实现 | 强制复验 |
| --- | --- | --- |
| slice 编号与逻辑位置 | Phase 01 | 03, 04, 05, 07 |
| AggregateState 不相交归约与状态消费 | Phase 01 | 03, 05, 07 |
| 单向链路、channel 与共享资源 | Phase 02 | 03, 05, 06, 07 |
| 六类直接算子最终语义 | Phase 01/02 | 03, 04, 05, 07 |
| 分层阶段接口及无全局 barrier | Phase 02 | 03, 05, 07 |
| lane 不重叠与性能模型 | Phase 03 | 05, 06, 07 |
| BufferPlan、端点配对、TB 顺序 | Phase 04 | 05, 06, 07 |
| BDD 机会分析与安全调优 | Phase 05 | 07 |
| 在线统计、校准、trace | Phase 06 | 07 |
| 无中文源代码与诊断 | Phase 01 | 每个阶段、最终验收 |

## Common Checkpoint Command

每个阶段除执行阶段文件列出的精确测试命令外，还执行以下共同检查：

```sh
python -m pytest -m "not hardware and not gurobi" -q
python -m pytest tests/unit -q
rg -n '[\p{Han}]' vericcl tests runtime -g '*.{py,c,cc,cu,cuh,h,json,xml}'
```

预期结果：pytest 全部通过；`rg` 无输出。Phase 03 的 Gurobi 测试和 Phase 06 的硬件测试若环境缺失，必须由测试报告明确列为 `not_run`，不得计为通过。
