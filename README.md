# VeriCCL

VeriCCL 用于合成 AllGather 和 AllReduce 集合通信调度。它根据用户提供的拓扑与通信配置生成初始调度。随后可通过在线测量修正开销估计，再分析瓶颈并局部修复调度。

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

安装完成后即可在 CPU 上合成调度。GPU 测量与执行还需要下面的 CUDA、MPI 和 MSCCL 环境。

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

**构建 GPU 运行环境**

以下命令依次构建 MSCCL、测量探针和调度执行程序。构建结果保存在各输出目录的 `result.json` 中。

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

每个参与通信的 GPU 对应一个 rank。VeriCCL 使用四份 JSON 文件描述任务，可从下文生成的示例修改：

| 文件 | 内容 |
|---|---|
| `topology.json` | GPU 连接、共享通信资源、链路与本地操作的开销 |
| `sketch.json` | 算子、消息与 slice 大小、运行时间限制、测量和修复开关 |
| `atom.json` | 合成方法、通信阶段和用户指定的约束 |
| `runtime.json` | 各 rank 使用的主机与 GPU、程序路径和环境变量 |

M 表示每个 rank 的输入字节数，S 表示每个 slice 的字节数。M 必须是 S 的整数倍，slice 数量由 N=M/S 自动计算。设参与通信的 rank 数为 P，AllGather 每个 rank 输出 P×M 字节，AllReduce 输出 M 字节。默认使用 FP32 和非原位缓冲区，AllReduce 默认执行求和（sum）。

链路开销按 `alpha_us + beta_us_per_byte × 字节数` 计算，`shared_resources` 描述多个连接共用网卡等资源的关系。配置中的文件路径填写绝对路径，完整字段见 [输入参考](docs/input-schema.md)。

## 运行合成器

以下命令在项目目录中运行。`solve`、`verify`、`calibrate` 和 `live` 使用四份 JSON 配置，并通过以下参数设置结果位置：

| 参数 | 用途 | 默认值 |
|---|---|---|
| `--project-root` | 项目目录 | 当前目录 |
| `--case` | 案例名称 | 由用户填写 |
| `--run-id` | 本次运行的编号 | 按启动时间生成 |

结果保存到 `<项目目录>/examples/<case>/output/<run_id>/`。下面的通用命令使用文件名表示输入，使用时将其换成实际路径。

**生成输入示例**

`tools.generate_examples` 生成可修改的输入配置，命令如下：

```bash
"$VERICCL_PYTHON" -m tools.generate_examples --project-root "$VERICCL_ROOT" --case my-case
```

配置保存在 `examples/my-case/input/`。下面生成两 rank 的示例，并选取 AllReduce 配置：

<!-- vericcl-v5-command: quickstart-setup -->
```bash
export VERICCL_ROOT="$(pwd -P)"
export VERICCL_PYTHON="$VERICCL_ROOT/.venv/bin/python"
export VERICCL_INPUT="$VERICCL_ROOT/examples/tutorial/input/allreduce-hierarchy-outofplace"
export VERICCL_RUN="$VERICCL_ROOT/examples/tutorial/output/first"
```

<!-- vericcl-v5-command: quickstart-generate -->
```bash
"$VERICCL_PYTHON" -m tools.generate_examples --project-root "$VERICCL_ROOT" --case tutorial
```

**合成调度：`solve`**

`solve` 根据配置合成调度，并执行已开启的测量与修复。通过四个参数指定输入文件：

```bash
"$VERICCL_PYTHON" -m vericcl solve --case my-case \
  --topology topology.json --sketch sketch.json \
  --atom atom.json --runtime runtime.json
```

结果目录中的 `best/accepted.json` 记录当前最佳调度及 XML 的位置。以下使用上面的 AllReduce 配置，将运行编号设为 `first`；该示例在 CPU 上完成合成与验证：

<!-- vericcl-v5-command: quickstart-solve -->
```bash
"$VERICCL_PYTHON" -m vericcl solve --project-root "$VERICCL_ROOT" --case tutorial --run-id first \
  --topology "$VERICCL_INPUT/topology.json" --sketch "$VERICCL_INPUT/sketch.json" \
  --atom "$VERICCL_INPUT/atom.json" --runtime "$VERICCL_INPUT/runtime.json"
```

AllGather 的输入目录为 `examples/tutorial/input/allgather-hierarchy-outofplace`，修改 `VERICCL_INPUT` 即可使用。将目录名中的 `hierarchy` 换为 `constructive_trees`，即可选择另一种合成方法。每次运行用新的 `--run-id` 保存结果。

**查看结果：`inspect`**

`inspect` 通过 `--run` 读取已有结果目录，将任务状态、当前最佳调度和检查点信息以 JSON 打印到终端：

```bash
"$VERICCL_PYTHON" -m vericcl inspect --run /path/to/run
```

例如，查看刚生成的调度并打印 XML 路径：

<!-- vericcl-v5-command: quickstart-inspect -->
```bash
"$VERICCL_PYTHON" -m vericcl inspect --run "$VERICCL_RUN"
"$VERICCL_PYTHON" - <<'PY'
import json, os
from pathlib import Path
run = Path(os.environ['VERICCL_RUN'])
s = json.loads((run / 'run-summary.json').read_text())
assert s['contract_met']
a = json.loads((run / 'best/accepted.json').read_text())
print('当前模型：', a['model_version'])
print('T_pred（微秒）：', a['t_pred_us'])
print('XML：', run / a['candidate']['xml'])
print('结束原因：', s['stop_reason'])
PY
```

**验证已有调度：`verify`**

`verify` 使用原配置检查已有 XML。`--sidecar` 指定随 XML 生成的调度说明文件，`--model` 指定本次评估使用的代价模型：

```bash
"$VERICCL_PYTHON" -m vericcl verify --case my-case \
  --topology topology.json --sketch sketch.json \
  --atom atom.json --runtime runtime.json \
  --xml schedule.xml --sidecar sidecar.json --model model.json
```

检查结果保存到新运行目录的 `run-summary.json` 和 `reports/`。下面读取刚才合成的 XML、sidecar 和模型，执行一次复验：

<!-- vericcl-v5-command: quickstart-verify -->
```bash
export VERICCL_XML="$("$VERICCL_PYTHON" -c 'import json,os; print(json.load(open(os.environ["VERICCL_RUN"]+"/best/accepted.json"))["candidate"]["xml"])')"
export VERICCL_SIDECAR="$("$VERICCL_PYTHON" -c 'import json,os; print(json.load(open(os.environ["VERICCL_RUN"]+"/best/accepted.json"))["candidate"]["sidecar"])')"
export VERICCL_MODEL="$("$VERICCL_PYTHON" -c 'import json,os; print(json.load(open(os.environ["VERICCL_RUN"]+"/best/accepted.json"))["model_version"])')"
"$VERICCL_PYTHON" -m vericcl verify --project-root "$VERICCL_ROOT" --case tutorial --run-id checked \
  --topology "$VERICCL_INPUT/topology.json" --sketch "$VERICCL_INPUT/sketch.json" \
  --atom "$VERICCL_INPUT/atom.json" --runtime "$VERICCL_INPUT/runtime.json" \
  --xml "$VERICCL_RUN/$VERICCL_XML" --sidecar "$VERICCL_RUN/$VERICCL_SIDECAR" \
  --model "$VERICCL_RUN/models/$VERICCL_MODEL.json"
```

合成时通过命令行修改的参数，在复验时也一并传入。

**设置调优时间与轮数**

`sketch.budgets.max_run_wall_s` 设置整项任务的最长运行时间。`T_pred` 表示调度的预测通信时间，单位为微秒，用于比较修复前后的调度。

修改输入参数时，使用 `--override-input --set 字段=JSON值`；多项修改可重复添加 `--set`。例如，开启两轮调优：

```bash
"$VERICCL_PYTHON" -m vericcl solve --project-root "$VERICCL_ROOT" --case tutorial --run-id tuned \
  --topology "$VERICCL_INPUT/topology.json" --sketch "$VERICCL_INPUT/sketch.json" \
  --atom "$VERICCL_INPUT/atom.json" --runtime "$VERICCL_INPUT/runtime.json" \
  --override-input --set sketch.budgets.max_tuning_rounds=2
```

程序根据时间、轮数和搜索结果结束调优，保存当前最佳调度。对于已设置调优轮数和时间的输入，也可在 `verify` 命令中增加 `--tune`，继续修复已有调度。

## GPU 测量与执行

**配置两 GPU 示例**

完成上述构建后，选择两张 GPU 并填写程序路径。下面使用两 rank AllReduce，开启 trace 记录和实测比较：

<!-- vericcl-v5-command: gpu-setup -->
```bash
export VERICCL_DEVICES=0,1
export VERICCL_GPU_INPUT="$VERICCL_ROOT/examples/gpu-tutorial/input"
"$VERICCL_PYTHON" - <<'PYGPU'
import os
from pathlib import Path
from tools.generate_examples import example_documents, write_documents
from vericcl.input.schema import LIVE_TIMEOUTS
root = Path(os.environ['VERICCL_ROOT'])
d = example_documents('allreduce')
devices = [int(x) for x in os.environ['VERICCL_DEVICES'].split(',')]
assert len(devices) == 2 and len(set(devices)) == 2
d['runtime']['rank_map'] = [{'rank': r, 'host': 'localhost', 'device': v} for r, v in enumerate(devices)]
d['runtime']['paths'] = {'vericcl_driver': os.environ['VERICCL_PROBE'],
    'vericcl_native': os.environ['VERICCL_NATIVE'], 'mpi_bin': os.environ['MPI_HOME'] + '/bin'}
d['runtime']['environment']['LD_LIBRARY_PATH'] = os.environ['LD_LIBRARY_PATH']
d['runtime']['control'] = {'bind_address': '127.0.0.1', 'port': 24950,
    'unix_socket': str(root / 'examples/gpu-tutorial/control.sock')}
d['sketch']['budgets'].update(max_run_wall_s=10800,
    max_online_validation_per_round_s=1800, max_online_validation_total_s=7200)
d['sketch']['trace_and_measure'] = {'enabled': True, 'job_timeout_s': 180, 'trace_capacity_bytes': 1048576}
d['sketch']['calibration'].update(max_relative_fit_error=0.1, job_timeout_s=600,
    payload_sizes=[512, 1024, 2048])
d['sketch']['live'] = {'enabled': False, **{key: 60 for key in LIVE_TIMEOUTS}, 'close_cleanup_timeout_s': 180}
write_documents(Path(os.environ['VERICCL_GPU_INPUT']), d)
PYGPU
GPU_ARGS=(--project-root "$VERICCL_ROOT" --case gpu-tutorial
  --topology "$VERICCL_GPU_INPUT/topology.json" --sketch "$VERICCL_GPU_INPUT/sketch.json"
  --atom "$VERICCL_GPU_INPUT/atom.json" --runtime "$VERICCL_GPU_INPUT/runtime.json")
```

使用上述 GPU 配置运行 `solve`，结果保存在 `examples/gpu-tutorial/output/first/`：

```bash
"$VERICCL_PYTHON" -m vericcl solve "${GPU_ARGS[@]}" --run-id first
```

`trace_and_measure.enabled` 同时控制 trace 记录和实测比较。程序预热后记录一次完整调用的 trace，再单独测量调度执行时间。`NCCL_BUFFSIZE` 设置通信缓冲区大小，本例为 16 MiB。

**测量代价模型：`calibrate`**

`calibrate` 独立测量链路、共享带宽、本地计算和访存开销。通过 `sketch.calibration.enabled` 开启测量：

```bash
"$VERICCL_PYTHON" -m vericcl calibrate --case my-case \
  --topology topology.json --sketch sketch.json \
  --atom atom.json --runtime runtime.json \
  --override-input --set sketch.calibration.enabled=true
```

模型保存在运行目录的 `models/`，测量任务与日志保存在 `jobs/`。例如，测量上述两 GPU 配置：

```bash
"$VERICCL_PYTHON" -m vericcl calibrate "${GPU_ARGS[@]}" --run-id calibrated \
  --override-input --set sketch.calibration.enabled=true
```

若要在测量后继续验证和修复调度，使用 `solve` 并设置调优轮数与修复时间：

```bash
"$VERICCL_PYTHON" -m vericcl solve "${GPU_ARGS[@]}" --run-id calibrated-tuned \
  --override-input --set sketch.calibration.enabled=true \
  --set sketch.budgets.max_tuning_rounds=3 \
  --set sketch.budgets.max_repair_per_round_s=600 --set sketch.budgets.max_repair_total_s=1800
```

模型更新后，程序使用同一模型比较原调度和修复后的调度。测量与修复流程见 [架构说明](docs/architecture.md)。

**常驻在线运行：`live`**

`live` 连续处理集合通信请求，并在两次请求之间更新调度。`--requests` 设置请求次数，`--session-timeout-s` 设置常驻阶段的时限，单位为秒：

```bash
"$VERICCL_PYTHON" -m vericcl live --case my-case \
  --topology topology.json --sketch sketch.json \
  --atom atom.json --runtime runtime.json \
  --requests 40 --session-timeout-s 900 \
  --override-input --set sketch.live.enabled=true
```

请求、版本更新和关闭记录保存在运行目录的 `live/` 中。下面沿用上述 GPU 配置，执行 40 次 AllReduce：

```bash
"$VERICCL_PYTHON" -m vericcl live "${GPU_ARGS[@]}" --run-id live \
  --requests 40 --session-timeout-s 900 --override-input \
  --set sketch.trace_and_measure.enabled=false --set sketch.live.enabled=true \
  --set sketch.budgets.max_tuning_rounds=3 \
  --set sketch.budgets.max_repair_per_round_s=60 --set sketch.budgets.max_repair_total_s=180
```

多节点运行时，各节点使用相同的安装路径，并配置节点间的免密 SSH。将控制地址和端口改为节点间可访问的值，在 rank 0 所在主机启动程序。在线模式和 C API 见 [架构说明](docs/architecture.md) 与 [接口头文件](runtime/include/vericcl.h)。

**可选：nccl-tests 对照**

通过 `mpirun` 启动 nccl-tests，`-np` 指定进程数，`--host` 指定主机，`-b` 和 `-e` 指定消息大小范围。命令格式如下，尖括号中的内容替换为实际值：

```text
mpirun -np <进程数> --host <主机列表> <nccl-tests目录>/all_reduce_perf -b <最小消息> -e <最大消息> -g 1
```

性能结果打印在终端。下面构建 nccl-tests，并运行两 GPU 的 AllReduce Ring 作为对照；`--release-receipts` 用于记录各进程结束后的 GPU 释放情况：

```bash
"$VERICCL_PYTHON" examples/nccl-tests-build/input/build.py --run-id tutorial-tests \
  --cuda-home "$CUDA_HOME" --mpi-home "$MPI_HOME" \
  --msccl-build "$VERICCL_MSCCL" --arch "$VERICCL_ARCH" --release-receipts
export VERICCL_NCCL_TESTS="$VERICCL_ROOT/examples/nccl-tests-build/output/tutorial-tests/rebuildable-cache/source/build"
mkdir "$VERICCL_ROOT/examples/nccl-tests-build/output/tutorial-receipts"
export VERICCL_NCCL_TESTS_RECEIPT_PREFIX="$VERICCL_ROOT/examples/nccl-tests-build/output/tutorial-receipts/rank-"
"$MPI_HOME/bin/mpirun" -np 2 --host localhost:2 \
  -x CUDA_VISIBLE_DEVICES="$VERICCL_DEVICES" -x LD_LIBRARY_PATH \
  -x NCCL_ALGO=Ring -x NCCL_BUFFSIZE=16777216 -x VERICCL_NCCL_TESTS_RECEIPT_PREFIX \
  "$VERICCL_NCCL_TESTS/all_reduce_perf" -b 2048 -e 2048 -g 1 -d float -o sum -w 5 -n 20 -c 1
```

VeriCCL 生成的 XML 通过前面的 `solve` 流程执行。跨节点运行 nccl-tests 时，在各节点创建结果目录，并将 `--host` 改为实际主机列表。每次运行使用新的结果目录。

## 结果与故障

结果统一保存在 `examples/<case>/output/<run_id>/`，输入位于同一案例的 `input/`。

| 文件 | 内容 |
|---|---|
| `run-summary.json`、`reports/timing.json` | 成功状态、结束原因、各阶段时间与 T_pred |
| `best/accepted.json`、`candidates/` | 当前最佳调度、XML、sidecar 和逐项检查 |
| `models/` | 初始/更新模型、测量和差异 |
| `effective-request.json`、`input_snapshot/` | 生效配置与原输入 |
| `run.log`、`events.jsonl`、`jobs/` | 流程日志、命令输出、进程与释放记录 |
| `checkpoints/`、`live/` | 恢复状态及在线请求记录 |

**恢复任务：`resume`**

`resume` 通过 `--run` 指定原运行目录，从保存的检查点继续执行：

```bash
"$VERICCL_PYTHON" -m vericcl resume --run /path/to/run
```

后续结果继续写入原目录，运行时间按原任务累计计算。例如，恢复一次中断的任务：

```bash
"$VERICCL_PYTHON" -m vericcl resume --run "$VERICCL_ROOT/examples/my-case/output/interrupted-run"
```

每个指令都可通过 `--help` 查看完整参数。例如，查看 `solve` 的用法：

```bash
"$VERICCL_PYTHON" -m vericcl solve --help
```

运行详情保存在 `run.log`，配置和结果说明见 [支持配置](docs/supported-configuration.md)。遇到动态库加载错误时，检查 CUDA、MPI 和 MSCCL 路径；trace 接口见 [运行时说明](runtime/msccl-trace/README.md)。
