# VeriCCL

VeriCCL 是一个集合通信调度合成器，支持 Broadcast、Reduce、AllGather、AllReduce、AllToAll 和 ReduceScatter。它根据用户提供的拓扑与运行配置生成调度，再通过可选的在线测量校准代价模型，分析瓶颈并局部修复。主流程以完整调度的预测完成时间 `T_pred` 比较候选，辅助在线模式支持在集合通信请求之间更新运行版本。

VeriCCL 提供 `hierarchy` 分层合成和 `constructive_trees` 批量生成树两种方法，使用 PySCIPOpt 调用 SCIP。求解器统一使用 SCIP，执行前检查其可用性。方法与实现详见 [架构说明](docs/architecture.md)。

## 环境

CPU 合成需要 Linux、Python 3.12、Git、GCC/G++ 和 Make。GPU 测量与执行另需 NVIDIA 驱动、CUDA 和 Open MPI。参考组合为 CUDA 12.8、Open MPI 4.1.8、V100；Python 依赖固定在 `requirements-lock/python312.txt`。远程执行将所有可写产物限制在执行账户的用户目录内。

以下命令在同一个 Bash 会话中依次执行。多节点时，各节点安装相同版本并保持相同绝对路径，参与节点之间须可免密 SSH。将 `YOUR_REPOSITORY_URL` 替换为本仓库地址。

**安装合成器**

```bash
git clone --branch vericcl-v5 "YOUR_REPOSITORY_URL" "$HOME/VeriCCL"
cd "$HOME/VeriCCL"
export VERICCL_ROOT="$(pwd -P)"
python3.12 -m venv .venv
export VERICCL_PYTHON="$VERICCL_ROOT/.venv/bin/python"
"$VERICCL_PYTHON" -m pip install -r requirements-lock/python312.txt
"$VERICCL_PYTHON" -m pip install --no-deps -e .
"$VERICCL_PYTHON" -m pip check
"$VERICCL_PYTHON" -m vericcl --version
"$VERICCL_PYTHON" -c 'from pyscipopt import Model; from dd.cudd import BDD; from gmpy2 import mpq; print("SCIP、CUDD、GMP 可导入")'
```

锁文件包含 SCIP 的 Python 接口、CUDD 和精确算术依赖，直接使用 Python 接口调用 SCIP。只做 CPU 合成可直接跳到“输入”。已有目录请先保存工作并确认版本，避免覆盖；缺少 Python venv 或系统编译工具时先补齐环境。

**配置 CUDA 与 MPI**

将 CUDA 路径和 GPU 架构改为实际值。V100 使用 `70`；下面在项目内安装 Open MPI 4.1.8。已有同版本 MPI 时可直接把 `MPI_HOME` 指向其安装前缀。

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

**构建 MSCCL、测量探针与原生执行器**

固定源码与补丁随仓库提供，构建时自动应用并校验。每次构建使用新的输出目录；查看其中的 `result.json` 和日志确认成功。

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

`ldd` 的 MPI、CUDA、NCCL 库应解析到上述安装目录。C 接口库和头文件位于 `tutorial-native/build/lib`、`tutorial-native/build/include`。

## 输入

每次显式提供四份 JSON 文件，可从下文生成的示例修改：

| 文件 | 内容 |
|---|---|
| `topology.json` | rank、节点、有向链路、共享资源及初始 α/β 和本地开销 |
| `sketch.json` | 算子、消息与 slice 大小、求解预算、测量和修复开关 |
| `atom.json` | `hierarchy` / `constructive_trees`、阶段与用户硬约束 |
| `runtime.json` | rank 到主机/GPU 的映射、工具路径、环境和运行时上限 |

M 是每 rank 的输入字节数，Broadcast 指 root 消息大小；S 是最小 slice 大小，N=M/S 自动计算。M 必须是 S 的整数倍；AllToAll、ReduceScatter 还要求 N 能被 rank 数整除。默认 FP32、sum、root=0、非原位。AllGather 每 rank 输出为 P×M，ReduceScatter 为 M/P；完整口径与字段见 [输入参考](docs/input-schema.md)。

拓扑和初始代价由用户提供，程序按输入中的连接与共享关系建模。`alpha_us + beta_us_per_byte × 字节数` 表示链路服务开销，`shared_resources` 声明共同使用的 NIC、交换端口等资源。JSON 中的路径使用实际绝对值，填写 shell 展开后的值。修改已有字段的命令行值需加 `--override-input`。

## 运行合成器

**生成示例并合成**

下面生成六算子、两种方法及原位/非原位示例，并运行一个两 rank AllReduce。教学代价仅用于演示；该例关闭硬件测量，调优轮数为 0。

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

<!-- vericcl-v5-command: quickstart-solve -->
```bash
"$VERICCL_PYTHON" -m vericcl solve --project-root "$VERICCL_ROOT" --case tutorial --run-id first \
  --topology "$VERICCL_INPUT/topology.json" --sketch "$VERICCL_INPUT/sketch.json" \
  --atom "$VERICCL_INPUT/atom.json" --runtime "$VERICCL_INPUT/runtime.json"
```

换用其他方法或算子时，修改 `VERICCL_INPUT` 指向对应示例目录。再次执行请更换 `--run-id`，每次运行使用独立目录。

**查看和复验结果**

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

`inspect` 只读取结果。`verify` 必须使用原输入、原命令行覆盖值及对应 XML/sidecar；`--model` 指定用于比较的代价模型。若合成调整了 S，复验会从产物恢复实际分区并重新检查。修改 trace、预算等输入字段后，应新建 `solve` 任务。

**设置预算与局部修复**

`sketch.budgets.max_run_wall_s` 限制从初始合成到测量、分析、修复和清理的完整流程；各阶段还受独立预算限制。`T_pred` 是预测通信时间，单位微秒；求解期限由预算字段控制。以下在教学输入上开启两轮 CPU 调优：

```bash
"$VERICCL_PYTHON" -m vericcl solve --project-root "$VERICCL_ROOT" --case tutorial --run-id tuned \
  --topology "$VERICCL_INPUT/topology.json" --sketch "$VERICCL_INPUT/sketch.json" \
  --atom "$VERICCL_INPUT/atom.json" --runtime "$VERICCL_INPUT/runtime.json" \
  --override-input --set sketch.budgets.max_tuning_rounds=2
```

已有调度可在上述 `verify` 命令增加 `--tune`，但原输入须已声明调优轮数和修复预算。默认先选修复策略；`sketch.repair.mode=joint` 改为联合选择。达到时间、轮数或指定合法空间内的无改进证明时结束，并保存当前最佳已接受调度。

## GPU 测量与执行

**配置实际两 GPU 示例**

先完成运行时构建，再选择两张可用 GPU。下列配置保留教学拓扑，替换了实际设备和路径，启用一次完整 trace，并提供在线预算。用于其他拓扑时须同时修改拓扑和 rank 映射。

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

```bash
"$VERICCL_PYTHON" -m vericcl solve "${GPU_ARGS[@]}" --run-id first
```

`trace_and_measure.enabled` 同时控制详细 trace 和候选实测门槛，默认关闭。trace 在预热后记录一次完整调用；正式收益比较采用另一次不含 trace 的采样。`NCCL_BUFFSIZE` 在 `runtime.environment` 中显式设置，本例为 16 MiB；它表示运行时缓冲容量，取值须处于底层有符号 32 位参数范围内。

**校准代价模型并迭代修复**

`calibration.enabled` 独立控制链路、共享并发和本地开销测量。单独运行 `calibrate` 输出测量及模型；在 `solve` 中开启它则进入完整流程：

```bash
"$VERICCL_PYTHON" -m vericcl calibrate "${GPU_ARGS[@]}" --run-id calibrated \
  --override-input --set sketch.calibration.enabled=true
"$VERICCL_PYTHON" -m vericcl solve "${GPU_ARGS[@]}" --run-id calibrated-tuned \
  --override-input --set sketch.calibration.enabled=true \
  --set sketch.budgets.max_tuning_rounds=3 \
  --set sketch.budgets.max_repair_per_round_s=600 --set sketch.budgets.max_repair_total_s=1800
```

测量失败时保留已合格组件和缺失范围；资源释放未确认时停止后续启动。模型更新后用同一新模型重评原调度与修复候选。详细采样、缓存复用和容量处理见 [架构说明](docs/architecture.md)。

**常驻在线模式**

`live` 在固定 rank 组中连续处理请求，并在请求边界准备和切换版本。沿用上述实际 GPU 配置，关闭详细 trace、显式启用在线模式：

```bash
"$VERICCL_PYTHON" -m vericcl live "${GPU_ARGS[@]}" --run-id live \
  --requests 40 --session-timeout-s 900 --override-input \
  --set sketch.trace_and_measure.enabled=false --set sketch.live.enabled=true \
  --set sketch.budgets.max_tuning_rounds=3 \
  --set sketch.budgets.max_repair_per_round_s=60 --set sketch.budgets.max_repair_total_s=180
```

多节点时设置可达内网地址与专用端口，在 rank 0 所在主机启动控制器；各节点运行时路径应一致。外部遥测、故障注入及 C API 接入见 [架构说明](docs/architecture.md) 和 [公开头文件](runtime/include/vericcl.h)。

**可选：nccl-tests 对照**

使用仓库脚本构建固定版本，`--release-receipts` 编入计时结束后的回执支持；再设置输出前缀，运行时便会写入逐 rank 释放记录：

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

此命令运行 Ring 参考。执行 VeriCCL XML 使用上述原生流程；跨节点参考测试须在每台主机安装相同路径、创建回执目录，并替换 `--host`。回执目录也使用新名称，避免覆盖旧结果。

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

`resume` 从受支持的检查点继续，保留原任务已消耗的预算。把下列路径替换为需要恢复的运行目录：

```bash
"$VERICCL_PYTHON" -m vericcl resume --run "$VERICCL_ROOT/examples/my-case/output/interrupted-run"
"$VERICCL_PYTHON" -m vericcl solve --help
```

退出码 0 表示本次启用的检查完成；是否证明最优另看 `stop_reason` 和 `proof_scope`。输入报错检查字段、整除和覆盖值；动态库错误检查 CUDA/MPI/MSCCL 路径；`gpu_released=false` 时保留现场并核对原始回执。配置范围与结果含义见 [支持配置](docs/supported-configuration.md)，详细 trace 接口见 [运行时说明](runtime/msccl-trace/README.md)。
