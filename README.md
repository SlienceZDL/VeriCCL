# VeriCCL

VeriCCL synthesizes collective communication schedules for multi-GPU clusters. This guide covers AllGather and AllReduce. It takes GPU connectivity, link costs, and a communication request, and produces executable MSCCL XML plus a sidecar that records data movement and provenance.

The workflow has three stages. Initial synthesis chooses routes and execution order using hierarchical planning and SCIP. Correctness checks validate the resulting schedule. The optional full symbolic mode independently parses the XML and models buffer states, TB order, dependency flags, FIFO capacity, and repeated execution; BDD analysis checks data contributions, ordering, progress, and routing constraints. GPU calibration then updates the cost model, and repeated analysis and local repair seek a faster schedule. When measurement is enabled, accepting a repair requires both predicted and paired measured improvement. Enable full symbolic execution with `sketch.symbolic.enabled`.

VeriCCL uses SCIP through PySCIPOpt. See the [architecture](docs/architecture.md) for implementation details and the [input reference](docs/input-schema.md) for every configuration field.

## Environment and build

1. Install the synthesizer

Clone the repository and install the dependencies with Python 3.12:

```bash
git clone https://github.com/SlienceZDL/VeriCCL.git "$HOME/VeriCCL"
cd "$HOME/VeriCCL"
export VERICCL_ROOT="$(pwd -P)"
python3.12 -m venv .venv
export VERICCL_PYTHON="$VERICCL_ROOT/.venv/bin/python"
"$VERICCL_PYTHON" -m pip install -r requirements-lock/python312.txt
"$VERICCL_PYTHON" -m pip install --no-deps -e .
"$VERICCL_PYTHON" -m pip check
"$VERICCL_PYTHON" -m vericcl --version
```

You can now synthesize and check schedules on the CPU.

2. Build the optional native accelerator

The C++ implementation accelerates schedule-time prediction and follows the Python implementation. It requires a C++20 compiler, CMake, and GMP headers. Build with the first command; use the second form when GMP headers and libraries are in a separate directory:

```bash
"$VERICCL_PYTHON" tools/build_native.py
"$VERICCL_PYTHON" tools/build_native.py --gmp-root "$VERICCL_ROOT/.deps/gmp-dev/root/usr"
```

The Python implementation is used when the accelerator is absent. Set `VERICCL_CORE=off` to select Python explicitly, or `VERICCL_CORE=require` to require the accelerator.

3. Configure CUDA and MPI

Set the CUDA path and GPU architecture (`70` for V100), then install Open MPI 4.1.8. If MPI is already installed, set `MPI_HOME` to its directory and skip the download and build:

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

4. Build the GPU runtime

These commands build the patched MSCCL library (`libnccl.so`), the measurement probe (`vericcl-probe`), and the schedule executor (`vericcl-native`). Each output directory contains a `result.json` build record:

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

Use `ldd` to check the loaded libraries. The C API library and headers are in `tutorial-native/build/lib` and `tutorial-native/build/include`.

5. Build nccl-tests for bandwidth measurements

```bash
"$VERICCL_PYTHON" examples/nccl-tests-build/input/build.py --run-id tutorial-tests \
  --cuda-home "$CUDA_HOME" --mpi-home "$MPI_HOME" \
  --msccl-build "$VERICCL_MSCCL" --arch "$VERICCL_ARCH" --release-receipts
export VERICCL_NCCL_TESTS="$VERICCL_ROOT/examples/nccl-tests-build/output/tutorial-tests/rebuildable-cache/source/build"
```

## Inputs

Each participating GPU is one rank. Four JSON files describe a task:

| File | Contents |
|---|---|
| `topology.json` | GPU links, shared resources such as NICs, and link and local-operation costs |
| `sketch.json` | Collective, input bytes M per rank, slice bytes S, budgets, and measurement and repair settings |
| `atom.json` | Synthesis method and user constraints, such as forbidden directed edges |
| `runtime.json` | Host and GPU for each rank, executable paths, and environment variables |

M must be divisible by S, giving N=M/S slices per rank. With P ranks, each rank produces P*M output bytes for AllGather or M bytes for AllReduce. The initial link model estimates the time for x bytes as `alpha_us + beta_us_per_byte * x`: alpha is the fixed cost per step, and beta is the cost per byte. `shared_resources` limits the combined traffic through shared components such as a NIC. Use absolute paths in configuration files.

## Run the synthesizer

Run the following commands from the project directory. `solve`, `verify`, `calibrate`, and `live` read the same four input files and write to `<project-root>/examples/<case>/output/<run_id>/`. These command forms apply to any configured rank count; adapt the topology, rank map, sizes, and budgets to your cluster.

| Argument | Purpose | Default |
|---|---|---|
| `--project-root` | Project directory | Current directory |
| `--case` | Case name | Required |
| `--run-id` | Run identifier | Generated from the start time |

1. Generate example inputs

`tools.generate_examples` writes editable inputs. This quickstart creates two-rank examples and selects AllReduce:

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

2. Synthesize a schedule: `solve`

`solve` synthesizes a schedule and runs the measurement and repair stages enabled in the inputs. This example uses CPU synthesis and checks:

<!-- vericcl-v5-command: quickstart-solve -->
```bash
"$VERICCL_PYTHON" -m vericcl solve --project-root "$VERICCL_ROOT" --case tutorial --run-id first \
  --topology "$VERICCL_INPUT/topology.json" --sketch "$VERICCL_INPUT/sketch.json" \
  --atom "$VERICCL_INPUT/atom.json" --runtime "$VERICCL_INPUT/runtime.json"
```

The AllGather inputs are in `examples/tutorial/input/allgather-hierarchy-outofplace`. Replace `hierarchy` with `constructive_trees` in the directory name to select batch tree construction. Use a new `--run-id` for each run.

3. Inspect the result: `inspect`

`inspect --run` reports task status, the best accepted schedule, and checkpoints. `best/accepted.json` identifies the accepted XML and sidecar:

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
print('Model:', a['model_version'])
print('T_pred (us):', a['t_pred_us'])
print('XML:', run / a['candidate']['xml'])
print('Stop reason:', s['stop_reason'])
PY
```

4. Check an existing schedule: `verify`

`verify` checks an XML using its original inputs. Pass the matching sidecar with `--sidecar` and the evaluation model with `--model`. The sidecar records the input hash, which must match:

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

5. Set tuning rounds and time limits

Repeat `--override-input --set field=JSON` to change individual fields. This example enables two tuning rounds. `sketch.budgets.max_run_wall_s` limits the complete run:

```bash
"$VERICCL_PYTHON" -m vericcl solve --project-root "$VERICCL_ROOT" --case tutorial --run-id tuned \
  --topology "$VERICCL_INPUT/topology.json" --sketch "$VERICCL_INPUT/sketch.json" \
  --atom "$VERICCL_INPUT/atom.json" --runtime "$VERICCL_INPUT/runtime.json" \
  --override-input --set sketch.budgets.max_tuning_rounds=2
```

Add `--tune` to `verify` to continue tuning from the supplied XML.

## GPU measurement and execution

1. Two-GPU example

After building the runtime, select two GPUs. This example enables tracing and measured comparisons:

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

`trace_and_measure.enabled` controls both tracing and measured comparisons. After warmup, the executor records step start and end times for one complete call, then repeatedly measures the full schedule.

2. Measure the cost model: `calibrate`

`calibrate` measures links, shared resources, and local operations without synthesizing a schedule. Models are saved in the run directory under `models/`:

```bash
"$VERICCL_PYTHON" -m vericcl calibrate "${GPU_ARGS[@]}" --run-id calibrated \
  --override-input --set sketch.calibration.enabled=true
```

3. Resident execution: `live`

`live` handles consecutive communication requests and adopts updated schedules between requests. Set the request count with `--requests` and the session deadline in seconds with `--session-timeout-s`:

```bash
"$VERICCL_PYTHON" -m vericcl live "${GPU_ARGS[@]}" --run-id live \
  --requests 40 --session-timeout-s 900 --override-input \
  --set sketch.trace_and_measure.enabled=false --set sketch.live.enabled=true \
  --set sketch.budgets.max_tuning_rounds=3 \
  --set sketch.budgets.max_repair_per_round_s=60 --set sketch.budgets.max_repair_total_s=180
```

Request and version records are saved in `live/`. See the [C API header](runtime/include/vericcl.h) for application integration.

## Two-server example: 2 x 8 GPUs

This example runs in-place AllGather with eight GPUs per host. Use identical installation paths and passwordless SSH between the hosts. Run the commands on the controller host. Set `$VERICCL_RUNTIME` to a directory available at the same path on both hosts, with the MSCCL library in `lib/` and both VeriCCL executables in `bin/`.

1. Generate inputs

`make_topology_2x8.py` provides an illustrative V100-style topology with NVLink, PCIe, NIC, and host-staging resources. Adapt its structure and initial cost estimates to your hardware. `make_case_2x8.py` writes four inputs with 16 MiB per rank (256 MiB nccl-tests size), S=4 MiB, four channels, and a 16 MiB `NCCL_BUFFSIZE`. Supply the host names, communication subnet, network interface, and IB/RoCE device names for your cluster:

```bash
cd "$VERICCL_ROOT/examples/two-server-2x8/input"
export HOSTS=host-a,host-b
export NETWORK=192.0.2.0/24 SOCKET_IFNAME=eth0
export IB_HCA_FIRST=mlx5_0 IB_HCA_SECOND=mlx5_0
"$VERICCL_PYTHON" make_case_2x8.py --output ag-2x8 --runtime "$VERICCL_RUNTIME" \
  --mpi-home "$MPI_HOME" --cuda-home "$CUDA_HOME" --hosts "$HOSTS" \
  --network "$NETWORK" --socket-ifname "$SOCKET_IFNAME" \
  --ib-hca "$IB_HCA_FIRST,$IB_HCA_SECOND"
cd "$VERICCL_ROOT"
export VERICCL_2X8="$VERICCL_ROOT/examples/two-server-2x8/input/ag-2x8"
ARGS_2X8=(--project-root "$VERICCL_ROOT" --case two-server-2x8
  --topology "$VERICCL_2X8/topology.json" --sketch "$VERICCL_2X8/sketch.json"
  --atom "$VERICCL_2X8/atom.json" --runtime "$VERICCL_2X8/runtime.json")
```

2. Synthesize, calibrate, and tune

```bash
"$VERICCL_PYTHON" -m vericcl solve "${ARGS_2X8[@]}" --run-id ag-solve
```

3. Continue tuning an accepted schedule

`verify --tune` checks the supplied schedule, measures it when enabled, and continues repair and refinement. This example starts from the previous result:

```bash
RUN="$VERICCL_ROOT/examples/two-server-2x8/output/ag-solve"
XML="$RUN/$("$VERICCL_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["candidate"]["xml"])' "$RUN/best/accepted.json")"
"$VERICCL_PYTHON" -m vericcl verify "${ARGS_2X8[@]}" --run-id ag-tune --tune \
  --xml "$XML" --sidecar "$(dirname "$XML")/sidecar.json"
```

4. Select XMLs by message size

MSCCL loads up to four colon-separated XML paths from `MSCCL_XML_FILES`. It selects the first XML satisfying `minBytes <= size < maxBytes`, where size is the nccl-tests size column: total output bytes for AllGather. The following example assigns [0, 64 MiB) to the first schedule and [64 MiB, 4 GiB) to the tuned schedule. Choose the boundary from measurements across the intended sizes:

```bash
TUNED="$VERICCL_ROOT/examples/two-server-2x8/output/ag-tune"
SMALL_XML="$XML"
LARGE_XML="$TUNED/$("$VERICCL_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["candidate"]["xml"])' "$TUNED/best/accepted.json")"
"$VERICCL_PYTHON" - "$SMALL_XML" "$LARGE_XML" <<'PY'
import sys
from pathlib import Path
from vericcl.bands import with_window
small, large = (Path(p).read_text() for p in sys.argv[1:3])
Path('band-0.xml').write_text(with_window(small, 0, 64 << 20))
Path('band-1.xml').write_text(with_window(large, 64 << 20, 4 << 30))
PY
```

Alternatively, `sketch.size_bands` runs a complete synthesis workflow for each band and writes `bands/MSCCL_XML_FILES`; see Extensions below.

5. Measure with nccl-tests

`bench_2x8.sh` launches eight processes on each host and runs `all_gather_perf`. Set `CHECK=1` for correctness checks; the default measures timing. Column 11 is in-place algorithm bandwidth in GB/s. Make the XMLs available at the same absolute paths on both hosts:

```bash
export LIB="$VERICCL_RUNTIME/lib" NCCL_TESTS="$VERICCL_NCCL_TESTS"
CHECK=1 examples/two-server-2x8/input/bench_2x8.sh "$PWD/band-0.xml:$PWD/band-1.xml"
examples/two-server-2x8/input/bench_2x8.sh "$PWD/band-0.xml:$PWD/band-1.xml" | awk '/^ +[0-9]+ /{print $1, $11}'
```

The script passes the configured network and library paths to MPI. `rank_2x8.sh` selects one GPU per local rank and the IB/RoCE devices configured for each host.



Use the same command, runtime library, inputs, and placement when comparing schedules.

## Extensions

**Message-size bands: `sketch.size_bands`**

Declare up to four bands with `min_bytes`, `max_bytes`, and `representative_bytes`. `solve` runs the full workflow at each representative size and carries the measured model into the next band. Results are written to `bands/band-<k>.xml`, `bands/manifest.json`, and `bands/MSCCL_XML_FILES`. Adjacent bands with identical programs share one XML:

```json
"size_bands": [
  {"min_bytes": 0, "max_bytes": 33554432, "representative_bytes": 16777216},
  {"min_bytes": 33554432, "max_bytes": 4294967296, "representative_bytes": 268435456}]
```

**Slice refinement: `sketch.repair.refinement_factors`**

When local repair finds no improvement, VeriCCL first tries splitting each slice into 2, 4, or 8 pieces while preserving routes, channels, and execution order. Smaller steps can improve overlap, at the cost of more fixed overhead. Refined schedules pass the same checks and acceptance criteria as other candidates; accepted schedules can be refined again. An empty list disables refinement.

**Calibration: `sketch.calibration`**

- Each probe flow executes `chain_steps` consecutive steps on one TB (default 8), capped by `chain_max_bytes` (default 4 MiB). Fitting uses average time per step to represent pipelined execution.
- Without explicit `payload_sizes`, probes cover S/2, S, 2S and refinement sizes such as S/4 and S/8. This lets refined candidates reuse measurements when the required sizes are covered.
- Each measurement uses one MPI launch. Cross-host probes use wall time per iteration to include network transfers staged through host memory when GPUDirect RDMA is unavailable.
- Shared-resource capacities, such as NIC or host-staging bandwidth, are reduced only when measurements provide saturation evidence.

**Symbolic execution and BDDs: `sketch.symbolic`**

Add this section to `sketch.json` to enable full symbolic execution and BDD analysis:

```json
"symbolic": {"enabled": true}
```

- Correctness: the independent XML interpreter versions buffer contents, recording contributor sets as BDDs, data paths as Symbols, and start/end time expressions. The progress model includes TB order, flags, send/receive pairing, FIFO capacity, and repeated MSCCL loops. Checks cover data contributions, ordered reads and writes, forbidden bytes, gateways, connection sequences, and agreement between XML and the planned schedule. Progress is checked at the configured window boundary and representative sizes, with counterexamples on failure.
- Hotspots: time-bucket counters track utilization, bytes, and concurrency. BDD analysis finds delayed transfers with alternative routes from ranks that already hold the data, then identifies resource competitors, ordering predecessors, and downstream dependencies. It also proposes reordering independent TB steps when a later step is ready earlier.
- Repair: symbolic analysis defines the affected region. BDD decisions filter alternative-path combinations against forbidden-byte, gateway, and shared-edge constraints. The remaining plans feed the local MILP solver.

**Historical schedules: `sketch.history`**

With `schedule_reuse` enabled, successful runs store their best schedule in `schedule_index`. Later tasks with the same topology structure and bandwidth differences within `relative_bandwidth_tolerance` adapt and recheck those schedules.

## Results and troubleshooting

Results are saved in `examples/<case>/output/<run_id>/`:

| File | Contents |
|---|---|
| `run-summary.json`, `reports/timing.json` | Status, stop reason, stage durations, and predicted time T_pred |
| `best/accepted.json`, `candidates/` | Best accepted schedule, candidate XMLs, sidecars, and checks |
| `models/` | Initial and calibrated models, with their differences |
| `effective-request.json`, `input_snapshot/` | Effective configuration and original inputs |
| `run.log`, `events.jsonl`, `jobs/` | Workflow logs, subprocess output, and GPU release records |
| `checkpoints/`, `live/` | Recovery checkpoints and resident-execution records |

**Resume an interrupted task: `resume`**

`resume --run` continues from the latest checkpoint in the original run directory. Previously consumed time remains charged to the budget:

```bash
"$VERICCL_PYTHON" -m vericcl resume --run "$VERICCL_ROOT/examples/my-case/output/interrupted-run"
```

Use `--help` for complete command options, for example `"$VERICCL_PYTHON" -m vericcl solve --help`. See [supported configurations](docs/supported-configuration.md) for scope and result interpretation. For library-loading errors, check CUDA, MPI, and MSCCL paths. The [runtime guide](runtime/msccl-trace/README.md) describes the trace format.
