# VeriCCL

[English](README.md)

## 概述

VeriCCL用于规划、求解和验证集合通信调度。它接收topology、sketch和atom三类JSON输入，生成MSCCL XML、调度sidecar及离线验证报告。

可直接求解`broadcast`、`reduce`、`allgather`、`allreduce`、`alltoall`和`reduce_scatter`。`scatter`与`gather`是内部阶段化算子；它们与六类直接求解算子共同构成八类集合通信语义。

硬件验证：`not_run`。下述流程仅在软件层面验证生成产物，不能推断CUDA编译、MSCCL加载、GPU执行或性能结果；准备这些环境前请阅读[运行时配置](docs/runtime-configuration.md)。

## 构建与安装 VeriCCL

通过SSH克隆，创建虚拟环境，安装开发依赖并以可编辑模式安装VeriCCL：

```bash
git clone git@github.com:SlienceZDL/VeriCCL.git
cd VeriCCL
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pip install -e .
```

若无法使用SSH认证，可通过HTTPS克隆，然后执行上一代码块中的相同虚拟环境命令：

```bash
git clone https://github.com/SlienceZDL/VeriCCL.git
cd VeriCCL
```

设置仓库根目录，并检查依赖、导入和CLI版本：

```bash
export VERICCL_ROOT="$(pwd)"
.venv/bin/python -m pip check
.venv/bin/python -c 'import vericcl, gurobipy, lxml, numpy, z3; print(vericcl.__version__)'
.venv/bin/python -m vericcl --version
```

预期版本字面量：`0.1.0`。

### Gurobi许可证检查

该单变量检查确认当前Gurobi许可证可以求解最小模型。下述构造式快速示例禁用MILP，因此不需要完整Gurobi许可证。

```bash
.venv/bin/python - <<'PY'
import gurobipy as gp

model = gp.Model("vericcl-license-check")
x = model.addVar(lb=0.0, name="x")
model.setObjective(x, gp.GRB.MINIMIZE)
model.optimize()
assert model.Status == gp.GRB.OPTIMAL
print("gurobi optimize check passed")
PY
```

预期结果字面量：`gurobi optimize check passed`。

## 运行 VeriCCL

仓库内示例输入为`vericcl/examples/topo/two_rank.json`、`vericcl/examples/sketch/allreduce_8m_1m.json`和`vericcl/examples/atom/constructive.json`。以下帮助命令列出可用CLI操作：

<!-- vericcl-doc-test: help -->
```bash
.venv/bin/python -m vericcl --help
```

将`VERICCL_ROOT`设为已安装VeriCCL仓库的根目录。后续相对路径均以该目录为起点。

<!-- vericcl-run-step: set-root -->
```bash
export VERICCL_ROOT="$(pwd)"
```

每次运行使用新的`VERICCL_OUTPUT_DIR`，它是本次运行的父目录。

<!-- vericcl-run-step: set-output -->
```bash
export VERICCL_OUTPUT_DIR="$VERICCL_ROOT/runs/readme-$(date +%Y%m%dT%H%M%S)"
```

创建输出根目录。VeriCCL会在其下创建包含算子、数据大小和run ID的目录。

<!-- vericcl-run-step: create-output -->
```bash
mkdir -p "$VERICCL_OUTPUT_DIR"
```

`two_rank.json`描述两个Rank及其有向链路。`allreduce_8m_1m.json`描述一个切分为1 MiB软件slice的8 MiB AllReduce。`constructive.json`选择禁用MILP的构造式策略。`VERICCL_OUTPUT_DIR`是本次运行的父目录，`quickstart`是稳定的run标识。

<!-- vericcl-run-step: solve -->
<!-- vericcl-doc-test: solve -->
```bash
.venv/bin/python -m vericcl solve --topology vericcl/examples/topo/two_rank.json --sketch vericcl/examples/sketch/allreduce_8m_1m.json --atoms vericcl/examples/atom/constructive.json --output-dir "$VERICCL_OUTPUT_DIR" --run-id quickstart
```

`--xml`指向solve阶段生成的最终XML。verify输出写入`vericcl_allreduce_8MiB_quickstart-verify/`。

<!-- vericcl-run-step: verify -->
<!-- vericcl-doc-test: verify -->
```bash
.venv/bin/python -m vericcl verify --topology vericcl/examples/topo/two_rank.json --sketch vericcl/examples/sketch/allreduce_8m_1m.json --atoms vericcl/examples/atom/constructive.json --output-dir "$VERICCL_OUTPUT_DIR" --run-id quickstart-verify --xml "$VERICCL_OUTPUT_DIR/vericcl_allreduce_8MiB_quickstart/vericcl_allreduce_8MiB_final.xml"
```

检查`$VERICCL_OUTPUT_DIR/vericcl_allreduce_8MiB_quickstart/vericcl_allreduce_8MiB_final.xml`是否存在；该文件是可执行的MSCCL XML。

<!-- vericcl-run-step: check-xml -->
```bash
test -f "$VERICCL_OUTPUT_DIR/vericcl_allreduce_8MiB_quickstart/vericcl_allreduce_8MiB_final.xml"
```

检查`$VERICCL_OUTPUT_DIR/vericcl_allreduce_8MiB_quickstart/vericcl_allreduce_8MiB_final.validation.json`是否存在；该文件是最终离线验证报告。

<!-- vericcl-run-step: check-report -->
```bash
test -f "$VERICCL_OUTPUT_DIR/vericcl_allreduce_8MiB_quickstart/vericcl_allreduce_8MiB_final.validation.json"
```

查看`$VERICCL_OUTPUT_DIR/vericcl_allreduce_8MiB_quickstart/vericcl_allreduce_8MiB_final.validation.json`这一最终离线验证报告；该文件不是可执行XML。

<!-- vericcl-run-step: inspect-report -->
```bash
.venv/bin/python -m json.tool "$VERICCL_OUTPUT_DIR/vericcl_allreduce_8MiB_quickstart/vericcl_allreduce_8MiB_final.validation.json"
```

`$VERICCL_OUTPUT_DIR/vericcl_allreduce_8MiB_quickstart/vericcl_allreduce_8MiB_final.schedule.json`是最终XML的sidecar。`$VERICCL_OUTPUT_DIR/vericcl_allreduce_8MiB_quickstart/run-summary.json`记录solve工作流摘要。

检查全部直接求解算子时，可单独运行软件回归示例：

<!-- vericcl-doc-test: example-validation -->
```bash
.venv/bin/python -m pytest tests/e2e/test_six_collectives.py -q
```

## MSCCL策略A：官方源码与内置补丁

策略A基于官方MSCCL仓库的不可变commit `b23e9cd5dd63f82ee1c5aae7e0a2042079be903a`。验证器在临时副本中检查干净的固定版本基础树及最终哈希，然后对实际checkout应用补丁并构建。

```bash
export VERICCL_ROOT="$(pwd)"
export CUDA_HOME=/usr/local/cuda
export MSCCL_ROOT="$(dirname "$VERICCL_ROOT")/msccl-official"
git clone https://github.com/microsoft/msccl.git "$MSCCL_ROOT"
git -C "$MSCCL_ROOT" checkout --detach b23e9cd5dd63f82ee1c5aae7e0a2042079be903a
python3 "$VERICCL_ROOT/runtime/msccl-trace/tools/verify_patch.py" --source-root "$MSCCL_ROOT" --base-tree
cp "$VERICCL_ROOT/runtime/msccl-trace/include/vericcl_trace_format.h" "$MSCCL_ROOT/src/include/vericcl_trace_format.h"
patch --directory="$MSCCL_ROOT" --strip=1 --input="$VERICCL_ROOT/runtime/msccl-trace/patches/0001-vericcl-fixed-step-trace.patch"
make -C "$MSCCL_ROOT" clean
make -C "$MSCCL_ROOT" -j src.build
test -d "$MSCCL_ROOT/build/lib"
```

验证器预期输出`verification passed`，库目录为`$MSCCL_ROOT/build/lib`。

## MSCCL策略C：预集成不可变tag

策略C使用公开tag `vericcl-runtime-v0.1.0`及commit `782ee5f72cf48c1ae1a2365bcf525019f5620175`。patched-tree验证器先检查commit及`runtime/msccl-trace/upstream.json`中的全部哈希，再执行相同构建。

```bash
export VERICCL_ROOT="$(pwd)"
export CUDA_HOME=/usr/local/cuda
export MSCCL_ROOT="$(dirname "$VERICCL_ROOT")/VeriCCL-MSCCL"
git clone --branch vericcl-runtime-v0.1.0 --depth 1 https://github.com/SlienceZDL/VeriCCL-MSCCL.git "$MSCCL_ROOT"
test "$(git -C "$MSCCL_ROOT" rev-parse HEAD)" = 782ee5f72cf48c1ae1a2365bcf525019f5620175
python3 "$VERICCL_ROOT/runtime/msccl-trace/tools/verify_patch.py" --source-root "$MSCCL_ROOT" --patched-tree
make -C "$MSCCL_ROOT" clean
make -C "$MSCCL_ROOT" -j src.build
test -d "$MSCCL_ROOT/build/lib"
```

验证器预期输出`verification passed`。策略A与策略C的trace头文件及运行时源码哈希均与`upstream.json`一致，并生成`$MSCCL_ROOT/build/lib`。静态验证不能证明CUDA编译或GPU执行成功。

## NCCL Tests与时钟辅助程序构建

使用当前[NCCL Tests](https://github.com/NVIDIA/nccl-tests)源码，并链接所选MSCCL树。`MPI_HOME`采用Ubuntu multiarch软件包位置，编译前需检查头文件和库目录。

```bash
export VERICCL_ROOT="$(pwd)"
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export NCCL_TESTS_ROOT="$(dirname "$VERICCL_ROOT")/nccl-tests"
export MPI_HOME="/usr/lib/$(dpkg-architecture -qDEB_HOST_MULTIARCH)/openmpi"
test -f "$MPI_HOME/include/mpi.h"
test -d "$MPI_HOME/lib"
git clone https://github.com/NVIDIA/nccl-tests "$NCCL_TESTS_ROOT"
make -C "$NCCL_TESTS_ROOT" -j MPI=1 MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi CUDA_HOME="$CUDA_HOME" NCCL_HOME="$MSCCL_ROOT/build"
nvcc -ccbin mpicxx -O2 -std=c++11 "$VERICCL_ROOT/runtime/msccl-trace/tools/vericcl_clock_sync.cu" -o "$VERICCL_ROOT/runtime/msccl-trace/tools/vericcl_clock_sync"
test -x "$NCCL_TESTS_ROOT/build/all_reduce_perf"
test -x "$VERICCL_ROOT/runtime/msccl-trace/tools/vericcl_clock_sync"
```

规定的官方构建形式使用Ubuntu x86_64 Open MPI前缀。其他Ubuntu架构应将`make`命令中的`MPI_HOME`字面量替换为已验证的`$MPI_HOME`值。

## 输入schema与示例

三类输入均为UTF-8 JSON对象。重复键、非有限数值及不一致维度会被拒绝。不同输入对未知字段的处理不同：topology验证已识别字段，但当前不拒绝额外键；sketch保留额外的顶层键，但会拒绝`collective`、`hyperparameters`和`solver`内部的未知字段；atom拒绝未知的顶层字段。

<!-- input-unknown-fields: topology-extra=accepted; sketch-top-extra=preserved; sketch-sections-extra=rejected; atom-top-extra=rejected -->

拓扑（`vericcl/examples/topo/two_rank.json`与`vericcl/examples/topo/two_node_gateway.json`）：

- `ranks`是正的全局Rank数。`nodes`必须完整且无重复地覆盖所有Rank；每个gateway必须属于其节点。
- `directed_links`包含唯一、非自环的`(src,dst)`链路。`max_channels`为正的并发上限，`resources`引用共享资源ID。
- `alpha`/`alpha_us`、`beta`/`beta_us`和`invbw`/`invbw_us`单位为微秒且非负。`invbw`为权威值，必须不小于`alpha`，并应等于`alpha + beta`。可选`bandwidth_bytes_per_us`将整数并发度映射为聚合字节/微秒。
- `shared_resources`包含ID、已有的有向`member_links`、正的`max_channels`及相同时序字段，用于描述NIC入口、出口或节点间链路等竞争资源。

Sketch（`vericcl/examples/sketch/allreduce_8m_1m.json`）：

- `collective.operator`为上述六个直接算子之一。仅`broadcast`与`reduce`要求`root`；归约算子要求`reduction_op`属于`avg|max|min|prod|sum`。`datatype`不能为空，`inplace`为Boolean。
- `total_size_bytes`与`slice_size_bytes`均为正字节数。总大小必须被slice大小整除；若存在`input_chunkup`，其值必须等于两者之商。`alltoall`与`reduce_scatter`还要求slice数可被`ranks`整除。
- 超参数控制目标选择、校准并发度、调优阈值与迭代数、验证超时及强制重新校准。示例给出接受的类型与默认值。
- `solver`控制总求解/单模型秒数、`[0,1]`范围内的`mip_gap`、最优性证明要求、确定性seed、`[1,32]`范围内的channel数、线程/模型并行度及强制重新求解。

Atom（`vericcl/examples/atom/constructive.json`与`vericcl/examples/atom/default.json`）：

- `stage_num`为`null`或正的精确阶段数。每个禁用传输为`[slice_id, src_rank, dst_rank, stage_id]`；源、目标Rank必须不同，且所有索引必须在有效范围内。
- 策略Boolean字段选择hierarchy、symmetry、shortest paths、batching、constructive trees及MILP。constructive文件禁用MILP，default文件启用MILP。
- 自动规划时`manual_hierarchy`为空。手动节点定义`node_id`、从零开始的`stage_id`、算子、排序且无重复的`communication_group`、可选的有根`root`、`[rank, offset, contributor_ids]`形式的`logical_input`与`logical_output`，以及唯一的`depends_on`节点ID。接口与依赖必须精确组合为全局集合通信。

可直接查看实际支持的示例：

```bash
.venv/bin/python -m json.tool vericcl/examples/topo/two_rank.json
.venv/bin/python -m json.tool vericcl/examples/topo/two_node_gateway.json
.venv/bin/python -m json.tool vericcl/examples/sketch/allreduce_8m_1m.json
.venv/bin/python -m json.tool vericcl/examples/atom/constructive.json
.venv/bin/python -m json.tool vericcl/examples/atom/default.json
```

`vericcl/examples/legacy`与`vericcl/examples/templates`仅供参考，不是受支持的运行时输入。

## 求解与验证

`solve`在`--output-dir`下创建新的确定性目录；`verify`检查XML及自动推导的`.schedule.json` sidecar，也可用`--sidecar`指定路径。前述四个冒烟命令展示真实语法。直接输入保留Broadcast、Reduce、AllGather、AllReduce、AllToAll和ReduceScatter语义；组合过程中的内部Scatter与Gather阶段保留另外两类语义。

## 覆盖、分层、调优与超时

若CLI语义参数与sketch不同，必须提供`--override-input`，否则拒绝执行。覆盖值写入临时effective sketch，不修改输入。`--tune`启用经验证的局部修复，正值`--timeout-s`限制当前工作流。

```bash
.venv/bin/python -m vericcl solve --topology vericcl/examples/topo/two_rank.json --sketch vericcl/examples/sketch/allreduce_8m_1m.json --atoms vericcl/examples/atom/constructive.json --operator allreduce --total-size-bytes 4194304 --slice-size-bytes 1048576 --out-of-place --override-input --tune --timeout-s 600 --output-dir runs --run-id override-tune
.venv/bin/python -m vericcl verify --topology vericcl/examples/topo/two_rank.json --sketch vericcl/examples/sketch/allreduce_8m_1m.json --atoms vericcl/examples/atom/constructive.json --timeout-s 600 --output-dir runs --run-id verify-existing --xml /absolute/path/to/schedule.xml --sidecar /absolute/path/to/schedule.schedule.json
```

对于实际的双节点gateway拓扑，可由内置default文件生成hierarchy策略，并求解八Rank AllReduce。gateway Rank为`0`与`4`。

```bash
cp vericcl/examples/atom/default.json /tmp/vericcl-hierarchy.json
.venv/bin/python - <<'PY'
import json
from pathlib import Path

path = Path("/tmp/vericcl-hierarchy.json")
value = json.loads(path.read_text(encoding="utf-8"))
value["strategies"]["hierarchy"] = True
path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
PY
.venv/bin/python -m vericcl solve --topology vericcl/examples/topo/two_node_gateway.json --sketch vericcl/examples/sketch/allreduce_8m_1m.json --atoms /tmp/vericcl-hierarchy.json --output-dir runs --run-id gateway --timeout-s 10800
```

自动分层会发现实际机内通信组与连通的gateway组。仅当已显式推导逻辑接口和依赖时，才使用`manual_hierarchy`。

## 在线验证

`--online`要求运行时兼容XML、每次执行仅加载一个XML，并设置下述环境变量。应填入实际本地版本字符串；这些字段属于校准缓存签名。GPU/NIC标签使用站点实际值，但不得包含秘密信息。

```bash
export VERICCL_MSCCL_BUILD_DIR="$MSCCL_ROOT/build/lib"
export VERICCL_NCCL_TESTS_BUILD_DIR="$NCCL_TESTS_ROOT/build"
export VERICCL_CLOCK_SYNC_BINARY="$VERICCL_ROOT/runtime/msccl-trace/tools/vericcl_clock_sync"
export VERICCL_ONLINE_INTER_NODE=0
export VERICCL_MAX_CLOCK_UNCERTAINTY_US=10
export VERICCL_CALIBRATION_LINK_CLASS=intra_node
export VERICCL_CALIBRATION_CACHE_PATH="$VERICCL_ROOT/runs/calibration-cache.json"
export VERICCL_GPU_MODEL="replace-with-gpu-model"
export VERICCL_NIC_MODEL="replace-with-nic-model"
export VERICCL_CUDA_VERSION="replace-with-cuda-version"
export VERICCL_NCCL_VERSION="replace-with-nccl-version"
export VERICCL_MSCCL_VERSION=vericcl-runtime-v0.1.0
export VERICCL_FORCE_RECALIBRATE=0
export VERICCL_TRACE_RECORDS=1048576
```

```bash
.venv/bin/python -m vericcl solve --topology vericcl/examples/topo/two_rank.json --sketch vericcl/examples/sketch/allreduce_8m_1m.json --atoms vericcl/examples/atom/constructive.json --online --output-dir runs --run-id online --timeout-s 10800
```

校准固定使用128 MiB与输入slice大小。slice大小必须整除128 MiB，否则校准状态为`not_run`。机内校准由一个进程以`-g 2`启动；节点间校准由两个MPI进程分别以`-g 1`启动。`solve --online`获得稳定校准后，更新匹配的`invbw`/并发上限并重新求解。`verify --online`保持输入XML不变；重新校准需要新求解时报告`requires_resolve=true`。

release测量与trace诊断必须分开运行。release使用5次预热、20次测量、正确性检查及`VERICCL_TRACE_ENABLE=0`；trace使用0次预热、20次测量、`-c 0`及`VERICCL_TRACE_ENABLE=1`。trace开销不能作为性能数据。

## 单节点XML执行

固定运行时契约为`MSCCL_CHUNKSTEPS=4`、`MSCCL_SLICESTEPS=4`、`NCCL_PROTO=Simple`、XML `cnt=1`及`NCCL_BUFFSIZE=2*slice_size_bytes`。内置示例的slice为1 MiB，因此缓冲区为2 MiB。每次仅设置一个XML，并确保消息大小、datatype、归约操作、root、Rank数及in-place模式与XML一致。

单节点双GPU release运行：

```bash
export LD_LIBRARY_PATH="$MSCCL_ROOT/build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NCCL_ALGO=MSCCL
export NCCL_PROTO=Simple
export NCCL_BUFFSIZE=2097152
export MSCCL_XML_FILES=/absolute/path/to/one-schedule.xml
export VERICCL_EXPECTED_MSCCL_CHUNKSTEPS=4
export VERICCL_EXPECTED_MSCCL_SLICESTEPS=4
export VERICCL_TRACE_ENABLE=0
"$NCCL_TESTS_ROOT/build/all_reduce_perf" -b 8388608 -e 8388608 -w 5 -n 20 -c 1 -d float -o sum -g 2
```

独立诊断运行：

```bash
export LD_LIBRARY_PATH="$MSCCL_ROOT/build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NCCL_ALGO=MSCCL
export NCCL_PROTO=Simple
export NCCL_BUFFSIZE=2097152
export MSCCL_XML_FILES=/absolute/path/to/one-schedule.xml
export VERICCL_EXPECTED_MSCCL_CHUNKSTEPS=4
export VERICCL_EXPECTED_MSCCL_SLICESTEPS=4
export VERICCL_TRACE_ENABLE=1
export VERICCL_TRACE_RECORDS=1048576
export VERICCL_TRACE_FILE_PREFIX=/absolute/path/to/vericcl-step
"$NCCL_TESTS_ROOT/build/all_reduce_perf" -b 8388608 -e 8388608 -w 0 -n 20 -c 0 -d float -o sum -g 2
```

## 多节点XML执行

每个Rank启动一个MPI进程，每个进程使用一个GPU。以下八Rank示例在两个主机上各启动四个Rank；hostfile与XML必须在所有节点的相同路径存在。

```bash
export LD_LIBRARY_PATH="$MSCCL_ROOT/build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NCCL_ALGO=MSCCL
export NCCL_PROTO=Simple
export NCCL_BUFFSIZE=2097152
export MSCCL_XML_FILES=/absolute/path/to/one-schedule.xml
export VERICCL_EXPECTED_MSCCL_CHUNKSTEPS=4
export VERICCL_EXPECTED_MSCCL_SLICESTEPS=4
export VERICCL_TRACE_ENABLE=0
export VERICCL_ONLINE_INTER_NODE=1
export VERICCL_MPI_LAUNCHER="$(command -v mpirun)"
export VERICCL_MPI_HOSTFILE=/absolute/path/to/hosts
export VERICCL_CALIBRATION_LINK_CLASS=inter_node
mpirun -np 8 -N 4 --hostfile "$VERICCL_MPI_HOSTFILE" -x LD_LIBRARY_PATH -x NCCL_ALGO -x NCCL_PROTO -x NCCL_BUFFSIZE -x MSCCL_XML_FILES -x VERICCL_EXPECTED_MSCCL_CHUNKSTEPS -x VERICCL_EXPECTED_MSCCL_SLICESTEPS -x VERICCL_TRACE_ENABLE "$NCCL_TESTS_ROOT/build/all_reduce_perf" -b 8388608 -e 8388608 -w 5 -n 20 -c 1 -d float -o sum -g 1
```

## 输出、退出码与故障诊断

对于冒烟输入及run ID `docs`，目录名为`vericcl_allreduce_8MiB_docs/`，其中包含`resolved-input.json`、`run-summary.json`、`schedules/`、`reports/`、`traces/`以及最终`.xml`、`.schedule.json`和`.validation.json`。超过MSCCL限制但离线有效的调度使用`.candidate.xml`，在重新求解前不得执行。

退出码：`0`表示离线有效完成，包括仅含运行时告警的candidate；`2`表示输入/配置错误；`3`表示没有语义有效候选或离线超时；`4`表示请求的在线验证失败或超时；`5`表示内部错误。

常见故障诊断：

- CUDA/MSCCL构建失败：重新检查驱动、Toolkit、主机编译器兼容性表、`CUDA_HOME`、GPU架构及完整编译输出。
- 缺少`mpi.h`或MPI链接失败：重新执行multiarch `MPI_HOME`检查，并确认已安装`libopenmpi-dev`。
- MSCCL验证失败：使用精确的上游commit或fork tag、干净的策略A checkout及未修改的内置头文件与补丁。
- `Model too large...`：激活完整Gurobi许可证；重新安装`gurobipy`不会扩大内置许可证限制。
- 在线binary/库缺失：检查三个`VERICCL_*_BUILD_DIR`/binary路径，并向每个MPI Rank传播`LD_LIBRARY_PATH`。
- 运行时不匹配：使用Simple协议、`NCCL_BUFFSIZE=2*slice_size_bytes`、step常量`4/4`、`cnt=1`、单个XML及匹配的NCCL Tests参数。
- Trace/时钟失败：分离release与trace运行，按需增大`VERICCL_TRACE_RECORDS`，并检查时钟误差和各Rank trace文件。

## 软件测试与参考资料

可在任意开发主机运行不依赖硬件的测试：

```bash
.venv/bin/python -m pytest tests/integration/test_documented_commands.py -q
.venv/bin/python -m pytest tests/unit/online/test_runtime_patch.py -q
git diff --check
```

本次文档工作未在macOS文档主机执行CUDA编译或GPU运行。报告硬件验证结果前，必须在目标Ubuntu GPU环境执行已标明的服务器构建与运行命令。

其他参考资料：[运行时配置](docs/runtime-configuration.md)、[验证报告](docs/validation-report.md)、[MSCCL trace补丁](runtime/msccl-trace/README.md)、[迁移说明](MIGRATION.md)、[NVIDIA CUDA安装](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)、[NCCL Tests](https://github.com/NVIDIA/nccl-tests)及[官方MSCCL](https://github.com/microsoft/msccl.git)。
