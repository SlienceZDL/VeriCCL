# VeriCCL

VeriCCL 用于合成 AllGather 和 AllReduce 集合通信调度。它根据用户提供的拓扑与通信配置生成初始调度。随后可通过在线测量修正代价模型，自动进行多轮“验证—局部修复”，逐轮比较并保存更优调度。

VeriCCL 提供 `hierarchy` 分层合成和 `constructive_trees` 批量生成树两种方法，使用 SCIP 求解调度。更多实现细节见 [架构说明](docs/architecture.md)。

## 环境

**安装合成器**

在 Linux 上使用 Python 3.12，依次安装依赖和 VeriCCL：

```bash
git clone --branch vericcl-v5 https://github.com/SlienceZDL/VeriCCL.git "$HOME/VeriCCL"
cd "$HOME/VeriCCL"
export VERICCL_ROOT="$(pwd -P)"
python3.12 -m venv .venv
export VERICCL_PYTHON="$VERICCL_ROOT/.venv/bin/python"
"$VERICCL_PYTHON" -m pip install -r requirements-lock/python312.txt
"$VERICCL_PYTHON" -m pip install --no-deps -e .
"$VERICCL_PYTHON" -m pip check
"$VERICCL_PYTHON" -m vericcl --version
```

安装完成后即可在 CPU 上合成调度。GPU 测量与执行还需配置下面的环境；多节点运行时，在各节点安装相同版本，并使用相同的绝对路径。以下命令使用 Bash。

**配置 CUDA 与 MPI**

设置 CUDA 路径和 GPU 架构，V100 对应 `70`。以下命令安装 Open MPI 4.1.8。已有 MPI 时，将 `MPI_HOME` 指向其目录并跳过下载和编译。

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export VERICCL_ARCH=70
export MPI_HOME="$VERICCL_ROOT/.deps/openmpi-4.1.8"
nvidia-smi
"$CUDA_HOME/bin/nvcc" --version
mkdir -p "$VERICCL_ROOT/.deps/src" "$VERICCL_ROOT/examples/mpi-build/output"
cd "$VERICCL_ROOT/.deps/src"
curl -fL https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.8.tar.gz -o openmpi-4.1.8.tar.gz
printf '%s  %s\n' fb41086bbed9300baa2f3d7572491facfe5257412fa524ec5a396aa9101d5c62 openmpi-4.1.8.tar.gz | sha256sum -c -
tar -xzf openmpi-4.1.8.tar.gz
cd openmpi-4.1.8
./configure --prefix="$MPI_HOME" --disable-mpi-fortran > "$VERICCL_ROOT/examples/mpi-build/output/configure.log" 2>&1
make -j8 > "$VERICCL_ROOT/examples/mpi-build/output/build.log" 2>&1
make install > "$VERICCL_ROOT/examples/mpi-build/output/install.log" 2>&1
cd "$VERICCL_ROOT"
export PATH="$MPI_HOME/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$MPI_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
"$MPI_HOME/bin/mpirun" --version
```

编译输出保存在 `examples/mpi-build/output/`；可在另一终端用 `tail -f` 查看相应日志。

**构建 GPU 运行环境**

以下命令依次构建 MSCCL、测量探针和调度执行程序。编译产物保存在各输出目录的 `build/`，构建记录位于 `process/result.json`。

```bash
export VERICCL_BUILD="$VERICCL_ROOT/examples/runtime-build/output/tutorial-runtime"
"$VERICCL_PYTHON" runtime/build_runtime.py --output-dir "$VERICCL_BUILD" \
  --cuda-home "$CUDA_HOME" --arch "$VERICCL_ARCH" --jobs 8 --timeout-s 900
export VERICCL_MSCCL="$VERICCL_BUILD/build"
"$VERICCL_PYTHON" runtime/build_probes.py \
  --output-dir "$VERICCL_ROOT/examples/runtime-build/output/tutorial-probes" \
  --cuda-home "$CUDA_HOME" --msccl-build "$VERICCL_MSCCL" \
  --mpi-bin "$MPI_HOME/bin" --arch "$VERICCL_ARCH" --timeout-s 180
"$VERICCL_PYTHON" runtime/build_native.py \
  --output-dir "$VERICCL_ROOT/examples/runtime-build/output/tutorial-native" \
  --cuda-home "$CUDA_HOME" --msccl-build "$VERICCL_MSCCL" \
  --mpi-bin "$MPI_HOME/bin" --arch "$VERICCL_ARCH" --timeout-s 180
export VERICCL_PROBE="$VERICCL_ROOT/examples/runtime-build/output/tutorial-probes/build/bin/vericcl-probe"
export VERICCL_NATIVE="$VERICCL_ROOT/examples/runtime-build/output/tutorial-native/build/bin/vericcl-native"
export LD_LIBRARY_PATH="$VERICCL_MSCCL/lib:$MPI_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
ldd "$VERICCL_PROBE"
ldd "$VERICCL_NATIVE"
```

`ldd` 用于检查程序使用的动态库路径。C 接口库和头文件位于 `tutorial-native/build/lib`、`tutorial-native/build/include`。

## 输入

每个参与通信的 GPU 对应一个 rank，编号为 `0` 到 `P-1`。任务规模由四份 JSON 配置决定，后续命令同时适用于单节点和多节点。

| 文件 | 内容 |
|---|---|
| `topology.json` | rank 数、节点分组、GPU 连接、共享资源与初始开销 |
| `sketch.json` | 算子、消息与 slice 大小、求解资源、时间限制和测量开关 |
| `atom.json` | 合成方法、通信阶段和用户约束 |
| `runtime.json` | rank 到主机、GPU 的映射，以及程序路径和环境变量 |

M 表示每个 rank 的输入字节数，S 表示每个 slice 的字节数。M 必须是 S 的整数倍，N=M/S 由程序计算。S 也应包含完整的数据类型元素，例如 FP32 要求 S 是 4 的整数倍。

| 算子 | 每 rank 输入 | 每 rank 输出 | nccl-tests 的 `-b`、`-e` |
|---|---|---|---|
| AllGather | M | P×M | P×M |
| AllReduce | M | M | M |

**设置任务规模**

先设置案例目录和通信参数。下面以两台主机、每台四张 GPU 为填值示例；将 `host-a`、`host-b` 替换为实际主机名或 SSH 别名，并按需增删主机与 GPU 编号。单节点可填写 `{"localhost":[0,1,2,3]}`。

<!-- vericcl-v5-command: case-setup -->
```bash
export VERICCL_CASE=my-case
export VERICCL_INPUT="$VERICCL_ROOT/examples/$VERICCL_CASE/input/base"
export VERICCL_LAYOUT='{"host-a":[0,1,2,3],"host-b":[0,1,2,3]}'
export VERICCL_OPERATOR=allgather
export VERICCL_METHOD=hierarchy
export VERICCL_M=1048576
export VERICCL_S=262144
export VERICCL_CHANNELS=7
export VERICCL_CPU_BUDGET=8
export VERICCL_WALL_S=10800
export VERICCL_TUNING_ROUNDS=3
export VERICCL_SCRATCH_BYTES=1073741824
export VERICCL_NCCL_BUFFSIZE=67108864
```

`VERICCL_OPERATOR` 可取 `allgather` 或 `allreduce`，`VERICCL_METHOD` 可取 `hierarchy` 或 `constructive_trees`。例如，2×8 规模填写两个各含八个 GPU 编号的列表，4×4 规模填写四个各含四个编号的列表；各节点使用不同数量或不连续编号的 GPU 时，直接列出这些编号。

**生成四份配置**

下面的模板根据上述参数生成节点分组、rank 映射和消息配置。模板采用节点内、节点间均可直接通信的抽象拓扑，每台主机的跨节点发送与接收分别共享一个带宽资源。开销数值是初值，请按实际设备修改；其他连接方式可直接修改 `directed_links` 和 `shared_resources`，后续命令保持相同。

<!-- vericcl-v5-command: input-generate -->
```bash
"$VERICCL_PYTHON" - <<'PY'
import json, os
from pathlib import Path
from tools.generate_examples import example_documents, write_documents
from vericcl.input.loader import load_request
from vericcl.topology.loader import load_topology

layout = json.loads(os.environ['VERICCL_LAYOUT'])
mapping, nodes = [], []
for node_id, (host, devices) in enumerate(layout.items()):
    assert devices and len(set(devices)) == len(devices)
    ranks = list(range(len(mapping), len(mapping) + len(devices)))
    nodes.append({'id': node_id, 'ranks': ranks})
    mapping.extend({'rank': r, 'host': host, 'device': gpu} for r, gpu in zip(ranks, devices))
p = len(mapping)
d = example_documents(os.environ['VERICCL_OPERATOR'], os.environ['VERICCL_METHOD'], ranks=p)
d['topology'].update(name='custom-topology', nodes=nodes, shared_resources=[])
# 单位：alpha 为微秒，bandwidth 为字节/微秒。
intra_alpha, intra_bandwidth = 1.0, 100000.0
inter_alpha, inter_bandwidth = 5.0, 12500.0
for link in d['topology']['directed_links']:
    same_host = mapping[link['src']]['host'] == mapping[link['dst']]['host']
    link.update(alpha_us=intra_alpha if same_host else inter_alpha,
                beta_us_per_byte=1 / (intra_bandwidth if same_host else inter_bandwidth))
for node in nodes:
    members = set(node['ranks'])
    for direction in ('tx', 'rx'):
        links = [[e['src'], e['dst']] for e in d['topology']['directed_links']
                 if (e['src'] in members and e['dst'] not in members if direction == 'tx'
                     else e['dst'] in members and e['src'] not in members)]
        if links:
            d['topology']['shared_resources'].append({'id': f"nic-{node['id']}-{direction}",
                'kind': 'nic', 'capacity_bytes_per_us': inter_bandwidth, 'links': links})
for link in d['topology']['directed_links']:
    link['resources'] = [r['id'] for r in d['topology']['shared_resources']
                         if [link['src'], link['dst']] in r['links']]
d['sketch']['hyperparameters'].update(total_size_bytes=int(os.environ['VERICCL_M']),
                                    slice_size_bytes=int(os.environ['VERICCL_S']))
d['sketch']['collective'].update(datatype='float32', inplace=False)
d['sketch']['solver'].update(per_model_timeout_s=300,
    cpu_budget=int(os.environ['VERICCL_CPU_BUDGET']), max_threads_per_model=8,
    max_parallel_models=1, max_channels=int(os.environ['VERICCL_CHANNELS']))
d['sketch']['budgets'].update(max_run_wall_s=float(os.environ['VERICCL_WALL_S']),
    max_initial_synthesis_s=3600, max_offline_validation_per_round_s=600,
    max_offline_validation_total_s=1800, max_repair_per_round_s=600,
    max_repair_total_s=1800, max_tuning_rounds=int(os.environ['VERICCL_TUNING_ROUNDS']), cleanup_reserve_s=30)
d['runtime'].update(rank_map=mapping, scratch_bytes_per_rank=int(os.environ['VERICCL_SCRATCH_BYTES']))
d['runtime']['environment']['NCCL_BUFFSIZE'] = os.environ['VERICCL_NCCL_BUFFSIZE']
files = write_documents(Path(os.environ['VERICCL_INPUT']), d)
request = load_request(files)
load_topology(request)
print('rank 数：', request.rank_count, '输入/输出字节数：', dict(request.size_accounting))
print('配置目录：', os.environ['VERICCL_INPUT'])
PY
```

生成后可编辑 JSON。改变规模时，`topology.ranks`、`nodes[].ranks` 与 `runtime.rank_map` 应覆盖同一组 rank；每台主机上的 GPU 编号应互不重复。链路是有向的，双向连接需要分别列出。多条链路共用网卡、交换机端口等资源时，在 `shared_resources[].links` 中列出这些链路，并在各链路的 `resources` 中填写对应资源 ID。

默认使用 FP32、非原位缓冲区，AllReduce 使用 sum。原位执行设置 `sketch.collective.inplace=true`。`scratch_bytes_per_rank` 是每 rank 的临时存储预算；`NCCL_BUFFSIZE` 是通信缓冲区字节数，取值范围为 1 到 2147483647。消息与 rank 数增加时，结合 GPU 显存调整这些预算。

`max_run_wall_s` 限制一次任务的总用时，各阶段还受各自时限约束；`max_tuning_rounds=0` 表示生成并检查初始调度后结束。增大规模时，可调整求解时间、CPU 配额和 slice 大小。完整字段与运行时容量见 [输入参考](docs/input-schema.md) 和 [支持配置](docs/supported-configuration.md)。

**可选：手工阶段与 Atom 变量组**

手工计划在 `atom.manual.enabled=true` 时使用。每个节点定义一个阶段中的通信组、输入状态、输出状态和前序节点；`atoms` 为该节点补充有向边禁用约束。自动合成默认使用 `manual.enabled=false`。

下面按当前算子和规模生成一个可编辑的手工计划，保存到新输入目录。阶段输入、输出和依赖由现有数据语义生成，随后可按需编辑 `atom.json`。

<!-- vericcl-v5-command: manual-generate -->
```bash
export VERICCL_MANUAL_INPUT="$VERICCL_ROOT/examples/$VERICCL_CASE/input/manual"
"$VERICCL_PYTHON" - <<'PY'
import json, os
from pathlib import Path
from tools.generate_examples import write_documents
from vericcl.input.loader import load_request
from vericcl.topology import load_topology
from vericcl.planner.direct import build_direct_plan
from vericcl.planner.manual import plan_manual
source = Path(os.environ['VERICCL_INPUT'])
files = {k: source / (k + '.json') for k in ('topology', 'sketch', 'atom', 'runtime')}
d = {k: json.loads(p.read_text()) for k, p in files.items()}
request = load_request(files)
plan = build_direct_plan(request, load_topology(request))
def states(interface):
    return [{'rank': slot.rank, 'offset': slot.offset, 'contributors': sorted(value)}
            for slot, value in interface.values.items()]
nodes = []
for node in plan.nodes:
    nodes.append({'id': node.node_id, 'stage_id': node.stage_id,
        'operator': node.local_collective.kind.value, 'root': node.local_collective.root,
        'group': list(node.communication_group), 'relay_ranks': list(node.communication_group),
        'depends_on': sorted({e.producer_id for e in plan.edges if e.consumer_id == node.node_id}),
        'input_states': states(node.logical_input), 'output_states': states(node.logical_output), 'atoms': []})
d['atom'].update(synthesis_method='hierarchy', stage_num=len({n.stage_id for n in plan.nodes}),
                 manual={'enabled': True, 'nodes': nodes})
paths = write_documents(Path(os.environ['VERICCL_MANUAL_INPUT']), d)
manual = load_request(paths)
plan_manual(manual, load_topology(manual))
print('手工输入目录：', os.environ['VERICCL_MANUAL_INPUT'])
PY
```

在所需节点的 `atoms` 中填写变量组，例如：

```json
"atoms": [
  {"slice_ids": [0, 1], "src_ranks": [0], "dst_ranks": [1, 2]}
]
```

这个组展开为四条约束：两个 slice 分别禁止从 rank 0 直传到 rank 1、2，阶段号取所在节点的 `stage_id`。各列表中的值自由组合；源、目标 rank 均属于该节点的 `group`，且每个组合的源和目标不同。只禁用指定有向边，其他合法中转路径继续可用。

原始 `slice_id=来源 rank×N+rank 内编号`，编号从 0 开始。Atom 变量只填写上述三个集合，阶段号从节点继承，时间、传输数据量和路径由求解器确定。多个节点、多个变量组和已有 `forbidden_transfers` 合并生效，重复约束自动去重。`input_states`、`output_states` 中的 `offset` 表示逻辑位置，`contributors` 列出该状态包含的原始 slice。

编辑完成后，将 `VERICCL_INPUT` 设为 `VERICCL_MANUAL_INPUT`，再执行下面的输入参数设置和 `solve` 命令。完整字段见 [输入参考](docs/input-schema.md)。

## 运行合成器

**指定输入与结果目录**

已有四份配置时，将 `VERICCL_INPUT` 设为其所在目录即可使用以下命令。通过 `--project-root` 固定结果位置，命令可从任意工作目录执行。

<!-- vericcl-v5-command: common-args -->
```bash
VERICCL_ARGS=(--project-root "$VERICCL_ROOT" --case "$VERICCL_CASE"
  --topology "$VERICCL_INPUT/topology.json" --sketch "$VERICCL_INPUT/sketch.json"
  --atom "$VERICCL_INPUT/atom.json" --runtime "$VERICCL_INPUT/runtime.json")
```

`solve`、`verify` 和 `calibrate` 的结果保存在 `examples/<case>/output/<run_id>/`。省略 `--run-id` 时按启动时间生成编号；指定时，每次运行使用新编号。

**合成调度：`solve`**

`solve` 根据四份配置合成调度，并执行已开启的测量与修复。通用格式如下，尖括号内容替换为实际值：

```text
python -m vericcl solve --project-root <项目目录> --case <案例名> \
  --topology <拓扑文件> --sketch <通信配置> --atom <算法配置> --runtime <运行配置>
```

使用当前配置合成，并保存结果目录变量，供后续查看、验证和测试使用：

<!-- vericcl-v5-command: solve -->
```bash
export VERICCL_RUN_ID="solve-$(date -u +%Y%m%dT%H%M%S)-$RANDOM"
export VERICCL_RUN="$VERICCL_ROOT/examples/$VERICCL_CASE/output/$VERICCL_RUN_ID"
"$VERICCL_PYTHON" -m vericcl solve "${VERICCL_ARGS[@]}" --run-id "$VERICCL_RUN_ID"
```

最终调度保存在 `schedule.xml`，对应说明和模型为 `schedule.json`、`model.json`。`run.json` 记录生效配置、当前最佳调度和结束原因。模板默认在 CPU 上完成合成、验证和多轮修复；GPU 测量按下一节开启。

**查看结果：`inspect`**

`inspect` 读取指定运行目录，将状态、最佳调度和检查点以 JSON 打印到终端：

```text
python -m vericcl inspect --run <运行结果目录>
```

查看上一步结果使用以下命令。查看其他任务时，先将 `VERICCL_RUN` 改为对应的绝对路径。

<!-- vericcl-v5-command: inspect -->
```bash
"$VERICCL_PYTHON" -m vericcl inspect --run "$VERICCL_RUN"
"$VERICCL_PYTHON" - <<'PY'
import json, os
from pathlib import Path
run = Path(os.environ['VERICCL_RUN'])
a = json.loads((run / 'run.json').read_text())['best']
print('T_pred（微秒）：', a['t_pred_us'])
print('XML：', run / 'schedule.xml')
print('调度说明：', run / 'schedule.json')
print('模型：', run / 'model.json')
PY
```

`T_pred` 是当前代价模型下的预测通信完成时间，用于比较同一任务、同一模型下的调度。

**验证已有调度：`verify`**

`verify` 使用与调度匹配的配置检查 XML。`--sidecar` 指定随 XML 生成的调度说明，`--model` 指定评估模型：

```text
python -m vericcl verify --project-root <项目目录> --case <案例名> \
  --topology <拓扑文件> --sketch <通信配置> --atom <算法配置> --runtime <运行配置> \
  --xml <XML路径> --sidecar <sidecar路径> --model <模型路径>
```

下面提取选定运行的生效配置，保留合成时的命令行覆盖值，再验证最佳 XML、sidecar 和模型。实际 slice 分区由 sidecar 恢复。提取的配置写入本案例的新输入目录，检查结果写入新的输出目录。

<!-- vericcl-v5-command: verify -->
```bash
export VERICCL_VERIFY_ID="verify-$(date -u +%Y%m%dT%H%M%S)-$RANDOM"
export VERICCL_VERIFY_INPUT="$VERICCL_ROOT/examples/$VERICCL_CASE/input/$VERICCL_VERIFY_ID"
"$VERICCL_PYTHON" - <<'PY'
import json, os
from pathlib import Path
from tools.generate_examples import write_documents
run = Path(os.environ['VERICCL_RUN'])
config = json.loads((run / 'run.json').read_text())['effective_request']
write_documents(Path(os.environ['VERICCL_VERIFY_INPUT']), config)
PY
"$VERICCL_PYTHON" -m vericcl verify --project-root "$VERICCL_ROOT" --case "$VERICCL_CASE" --run-id "$VERICCL_VERIFY_ID" \
  --topology "$VERICCL_VERIFY_INPUT/topology.json" --sketch "$VERICCL_VERIFY_INPUT/sketch.json" \
  --atom "$VERICCL_VERIFY_INPUT/atom.json" --runtime "$VERICCL_VERIFY_INPUT/runtime.json" \
  --xml "$VERICCL_RUN/schedule.xml" --sidecar "$VERICCL_RUN/schedule.json" \
  --model "$VERICCL_RUN/model.json"
```

验证会沿用配置中的测量开关。检查结果见新目录的 `run.json` 和 `report.md`，逐项记录保存在 `process/`。

**设置调优轮数**

在 `sketch.json` 中设置 `max_tuning_rounds`、`max_repair_per_round_s` 和 `max_repair_total_s`，分别控制轮数、每轮修复时间和累计修复时间。也可通过 `--override-input --set 字段=JSON值` 修改当前运行的参数，例如：

```bash
"$VERICCL_PYTHON" -m vericcl solve "${VERICCL_ARGS[@]}" --override-input \
  --set sketch.budgets.max_tuning_rounds=3 \
  --set sketch.budgets.max_repair_per_round_s=600 --set sketch.budgets.max_repair_total_s=1800
```

每轮先验证当前调度并分析瓶颈，再将建议转成约束，局部求解并验证候选。获得更优调度后自动进入下一轮；有限候选用尽时继续扩大搜索。达到时间或轮数上限，或在明确的合法修复空间中取得无改进证明时结束。

已有调度使用相同的 `verify` 命令并增加 `--tune`，即可进入这个多轮循环：

```text
python -m vericcl verify <四份输入与输出参数> \
  --xml <XML路径> --sidecar <调度说明路径> --model <模型路径> --tune
```

## GPU 测量与执行

**补充运行环境**

参与节点之间配置免密 SSH，并在 rank 0 所在主机启动。各节点使用相同的 VeriCCL、Python、MPI、CUDA 和执行程序路径；所有运行产物放在执行账户的用户目录中。`runtime.rank_map` 沿用前面的规模与 GPU 映射。

将控制地址设为 rank 0 主机上可被其他节点访问的 IPv4 地址，端口使用空闲端口。单节点可使用 `127.0.0.1`。下面复制当前输入，补充 GPU 设置；按任务规模调整测量时限、trace 容量和采样大小。

```bash
export VERICCL_CONTROL_ADDRESS=192.0.2.10
export VERICCL_CONTROL_PORT=24950
export VERICCL_GPU_INPUT="$VERICCL_ROOT/examples/$VERICCL_CASE/input/gpu"
```

<!-- vericcl-v5-command: gpu-setup -->
```bash
"$VERICCL_PYTHON" - <<'PY'
import json, os
from pathlib import Path
from tools.generate_examples import write_documents
source = Path(os.environ['VERICCL_INPUT'])
d = {key: json.loads((source / (key + '.json')).read_text()) for key in ('topology', 'sketch', 'atom', 'runtime')}
d['runtime']['paths'] = {'vericcl_driver': os.environ['VERICCL_PROBE'],
    'vericcl_native': os.environ['VERICCL_NATIVE'], 'mpi_bin': os.environ['MPI_HOME'] + '/bin'}
d['runtime']['environment']['LD_LIBRARY_PATH'] = os.environ['LD_LIBRARY_PATH']
d['runtime']['control'] = {'bind_address': os.environ['VERICCL_CONTROL_ADDRESS'],
    'port': int(os.environ['VERICCL_CONTROL_PORT'])}
d['sketch']['budgets'].update(max_online_validation_per_round_s=1800, max_online_validation_total_s=7200)
d['sketch']['trace_and_measure'] = {'enabled': False, 'job_timeout_s': 600, 'trace_capacity_bytes': 16777216}
d['sketch']['calibration'].update(max_relative_fit_error=0.1, job_timeout_s=600,
    payload_sizes=sorted({d['sketch']['hyperparameters']['slice_size_bytes'], d['sketch']['hyperparameters']['total_size_bytes']} - {None}))
write_documents(Path(os.environ['VERICCL_GPU_INPUT']), d)
PY
GPU_ARGS=(--project-root "$VERICCL_ROOT" --case "$VERICCL_CASE"
  --topology "$VERICCL_GPU_INPUT/topology.json" --sketch "$VERICCL_GPU_INPUT/sketch.json"
  --atom "$VERICCL_GPU_INPUT/atom.json" --runtime "$VERICCL_GPU_INPUT/runtime.json")
```

在 `runtime.environment` 中可补充 `NCCL_SOCKET_IFNAME`、`NCCL_IB_HCA` 等网络设置，JSON 中填写实际值与绝对路径。

**执行调度并记录 trace**

`sketch.trace_and_measure.enabled` 同时开启 trace 与实测比较。程序预热后记录一次完整调用，再单独测量执行时间。沿用 `solve` 的输入格式，开启此开关：

```bash
"$VERICCL_PYTHON" -m vericcl solve "${GPU_ARGS[@]}" \
  --override-input --set sketch.trace_and_measure.enabled=true
```

结果目录由终端输出。需要复验或运行 nccl-tests 时，将 `VERICCL_RUN` 设为这次运行的目录。

**测量代价模型：`calibrate`**

`calibrate` 独立测量链路、共享带宽、本地计算和访存开销，通用格式为：

```text
python -m vericcl calibrate --project-root <项目目录> --case <案例名> \
  --topology <拓扑文件> --sketch <通信配置> --atom <算法配置> --runtime <运行配置> \
  --override-input --set sketch.calibration.enabled=true
```

使用当前 GPU 配置进行测量：

```bash
"$VERICCL_PYTHON" -m vericcl calibrate "${GPU_ARGS[@]}" \
  --override-input --set sketch.calibration.enabled=true
```

最终模型为 `model.json`；模型版本与差异位于 `process/models/`，测量任务与日志位于 `process/jobs/`。要在合成中完成测量、验证和修复，使用：

```bash
"$VERICCL_PYTHON" -m vericcl solve "${GPU_ARGS[@]}" --override-input \
  --set sketch.calibration.enabled=true --set sketch.budgets.max_tuning_rounds=3 \
  --set sketch.budgets.max_repair_per_round_s=600 --set sketch.budgets.max_repair_total_s=1800
```

模型更新后，程序在同一模型下重新评估并比较调度。测量与修复流程见 [架构说明](docs/architecture.md)。

**可选：nccl-tests 对照**

先在各参与节点构建相同版本的 nccl-tests。以下命令生成测试程序，`--release-receipts` 为每个进程记录结束后的 GPU 释放情况：

```bash
"$VERICCL_PYTHON" "$VERICCL_ROOT/examples/nccl-tests-build/input/build.py" --run-id tutorial-tests \
  --cuda-home "$CUDA_HOME" --mpi-home "$MPI_HOME" \
  --msccl-build "$VERICCL_MSCCL" --arch "$VERICCL_ARCH" --release-receipts
export VERICCL_NCCL_TESTS="$VERICCL_ROOT/examples/nccl-tests-build/output/tutorial-tests/process/rebuildable-cache/source/build"
```

用 `mpirun` 启动测试。`-np` 设置进程数，`--host` 设置主机及进程位置，`-g 1` 表示每个进程使用一张 GPU；AllGather 使用 `all_gather_perf`，AllReduce 使用 `all_reduce_perf`：

```text
mpirun -np <rank数> --host <主机:进程数,...> <测试程序> \
  -b <测试字节数> -e <测试字节数> -g 1 -d <数据类型> -w 5 -n 20 -c 1
```

下面从 `VERICCL_RUN` 中读取最佳调度的配置，自动换算消息大小，并逐 rank 指定主机与 GPU。该模板也适用于各节点 GPU 数量不同、GPU 编号不连续的情况。运行 Ring 作为对照，日志保存在本案例的 `output/nccl-tests-*/nccl-tests.log`，各节点在同路径下保存本节点的释放记录。

<!-- vericcl-v5-command: nccl-tests -->
```bash
"$VERICCL_PYTHON" - <<'PY'
import json, os, shlex, subprocess, sys, uuid
from pathlib import Path
run = Path(os.environ['VERICCL_RUN'])
a = json.loads((run / 'run.json').read_text())['best']
c = a['candidate']['effective_config']
mapping = sorted(c['runtime']['rank_map'], key=lambda x: x['rank'])
collective = c['sketch']['collective']
op = collective['operator']
size = c['sketch']['hyperparameters']['total_size_bytes'] * (len(mapping) if op == 'allgather' else 1)
binary = Path(os.environ['VERICCL_NCCL_TESTS']) / {'allgather': 'all_gather_perf', 'allreduce': 'all_reduce_perf'}[op]
datatype = {'float16': 'half', 'float32': 'float', 'float64': 'double'}.get(collective['datatype'], collective['datatype'])
output = Path(os.environ['VERICCL_ROOT']) / 'examples' / os.environ['VERICCL_CASE'] / 'output' / ('nccl-tests-' + uuid.uuid4().hex[:12])
output.mkdir(parents=True)
environment = dict(c['runtime']['environment'], LD_LIBRARY_PATH=os.environ['LD_LIBRARY_PATH'], NCCL_ALGO='Ring')
environment.pop('MSCCL_XML_FILES', None)
# 每个 MPI 进程按配置选择 GPU；创建本节点的释放记录目录后启动测试。
wrapper = "import os,pathlib,sys; pathlib.Path(os.environ['VERICCL_NCCL_TESTS_RECEIPT_PREFIX']).parent.mkdir(parents=True,exist_ok=True); os.execv(sys.argv[1],sys.argv[1:])"
command = [os.environ['MPI_HOME'] + '/bin/mpirun', '--bind-to', 'none']
for item in mapping:
    if item['rank']:
        command.append(':')
    env = dict(environment, NCCL_TESTS_DEVICE=str(item['device']),
               VERICCL_NCCL_TESTS_RECEIPT_PREFIX=str(output / 'receipts' / 'rank-'))
    slots = sum(entry['host'] == item['host'] for entry in mapping)
    command += ['-np', '1', '--host', f"{item['host']}:{slots}", 'env', '-u', 'MSCCL_XML_FILES', '-u', 'CUDA_VISIBLE_DEVICES',
                *[f'{key}={value}' for key, value in env.items()], sys.executable, '-c', wrapper, str(binary),
                '-b', str(size), '-e', str(size), '-g', '1', '-d', datatype, '-w', '5', '-n', '20', '-c', '1']
    if op == 'allreduce':
        command += ['-o', collective['reduction_op']]
(output / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
print(shlex.join(command), flush=True)
print('日志：', output / 'nccl-tests.log', flush=True)
with (output / 'nccl-tests.log').open('w') as log:
    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
PY
```

nccl-tests 同时报告原位与非原位结果，按 `collective.inplace` 选取相应列。VeriCCL XML 的执行与 trace 使用前面的 `solve` / `verify` 流程。测试程序的设备选择规则见 [nccl-tests 源码](https://github.com/NVIDIA/nccl-tests/blob/0b4c4cb99fb1381edec1f78c37230688ea1ceb26/src/common.cu#L825-L831)。

## 结果与故障

结果统一保存在 `examples/<case>/output/<run_id>/`，输入位于同一案例的 `input/`。

| 顶层文件或目录 | 内容与生成条件 |
|---|---|
| `run.json` | 任务身份、命令、输入快照索引、生效配置、默认与覆盖来源、数据量换算、阶段耗时、结果和文件索引 |
| `report.md` | 本次结果、检查结论、预计时间、结束原因和问题记录 |
| `run.log` | 按时间排列的主流程日志 |
| `schedule.xml`、`schedule.json` | 已接受调度及其语义、缓冲区、TB 和事件映射；获得合格调度后生成 |
| `model.json` | 与最终调度对应的代价模型；独立校准时为本次发布的模型 |
| `checkpoint.json` | 最新检查点的状态、已用时间、轮数和过程文件引用 |
| `trace.json` | 已接受调度的 trace 分析；开启 trace 并取得完整结果后生成 |
| `process/` | 全部中间过程、历史版本和恢复所需文件 |
| `details.zip` | 结束时对 `process/` 的归档副本，便于整体保存；`process/` 继续用于恢复 |

`run.json` 的 `invocation`、`input_snapshot`、`effective_request`、`value_sources`、`size_accounting` 分别保存命令、原输入索引、生效配置、取值来源和字节数换算。`best` 指向已接受候选，`result` 记录状态与结束原因，`timing` 记录阶段时间和轮数，`artifacts` 列出本次生成的顶层文件。

| `process/` 内部位置 | 内容 |
|---|---|
| `input_snapshot/` | 原始输入文件，以及 verify 读取的 XML、调度说明和模型 |
| `models/` | 初始模型、校准模型、组件差异、探针输入与样本 |
| `candidates/<id>/` | 每个候选的 `candidate.json`、XML、调度说明、检查和预测结果 |
| `jobs/<id>/` | 子进程的请求、身份、标准输出、错误输出、退出与清理记录；远程任务另含逐 rank 回执 |
| `checkpoints/<id>/`、`checkpoints/latest.json` | 各阶段恢复快照，以及最新检查点位置和校验值 |
| `statistics/` | 实测比较的采样编号与统计记录 |
| `live/` | 原生执行的资源占用和释放状态 |
| `reports/`、`best/` | 分析、证明、时间与 trace 细节，以及最佳候选记录 |
| `tmp/`、`rebuildable-cache/` | 临时求解输入、构建源码等可重新生成的数据 |
| `events.jsonl`、`run-summary.json` | 结构化事件流与本次结束记录；逐 rank 日志也保存在此目录 |

`solve` 输出完整循环的结果；`verify` 输出已有调度的检查结果，`--tune` 还记录各轮修复；`calibrate` 输出模型及测量记录。`inspect` 只在终端显示现有结果，`resume` 在原目录继续记录。构建命令另保留 `build/` 存放库和程序，诊断记录仍在 `process/`。文件按实际执行阶段生成，详细模型、候选和任务保留各自的独立编号。

**恢复任务：`resume`**

`resume` 从指定运行目录的检查点继续执行：

```text
python -m vericcl resume --run <运行结果目录>
```

将 `VERICCL_RUN` 设为待恢复目录后运行：

```bash
"$VERICCL_PYTHON" -m vericcl resume --run "$VERICCL_RUN"
```

结果继续写入原目录，时间按原任务累计计算。每个指令可通过 `--help` 查看完整参数，例如：

```bash
"$VERICCL_PYTHON" -m vericcl solve --help
```

运行详情见 `run.log`。配置或调度检查失败时，先查看 `report.md`、`process/reports/` 与 `process/candidates/` 中的检查结果；动态库加载错误时，核对 CUDA、MPI 和 MSCCL 路径。trace 接口见 [运行时说明](runtime/msccl-trace/README.md)。
