# VeriCCL

VeriCCL synthesizes AllGather and AllReduce collective communication schedules from user-provided topology and communication settings. It can then update the cost model through online measurements and run multiple rounds of verification and local repair, comparing schedules and keeping improvements after each round.

VeriCCL provides two methods: `hierarchy` for hierarchical synthesis and `constructive_trees` for batched tree construction. Both use SCIP to solve for schedules. See the [architecture](docs/architecture.md) for implementation details.

## Environment

**Install the synthesizer**

Use Linux with Python 3.12, Git, a C/C++ compiler, Make, curl, and patch. Run the commands below in one Bash session. Install the Python dependencies and VeriCCL:

```bash
set -euo pipefail
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

You can now synthesize schedules on the CPU. GPU measurement and execution require the setup below. For multiple nodes, install the same versions at the same absolute paths on each node.

**Configure CUDA and MPI**

Set the CUDA path and GPU architecture; V100 uses `70`. The following commands install Open MPI 4.1.8. If MPI is already installed, set `MPI_HOME` to its directory and skip the download and build.

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
./configure --prefix="$MPI_HOME" --with-cuda="$CUDA_HOME" --disable-mpi-fortran > "$VERICCL_ROOT/examples/mpi-build/output/configure.log" 2>&1
make -j8 > "$VERICCL_ROOT/examples/mpi-build/output/build.log" 2>&1
make install > "$VERICCL_ROOT/examples/mpi-build/output/install.log" 2>&1
cd "$VERICCL_ROOT"
export PATH="$MPI_HOME/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$MPI_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
"$MPI_HOME/bin/mpirun" --version
"$MPI_HOME/bin/ompi_info" --parsable --all | grep mpi_built_with_cuda_support
```

Build output is saved in `examples/mpi-build/output/`. These commands print no build progress to the terminal; use `tail -f` in another terminal to follow the relevant log. The final MPI check reports `true` for CUDA support.

**Build the GPU runtime**

`build_runtime.py` builds the patched MSCCL library. `build_probes.py` builds the measurement program, and `build_native.py` builds the schedule executor and C interface. Run them in this order. Each build uses a new output directory, with binaries under `build/` and build records in `process/result.json`.

```bash
export VERICCL_BUILD_ID="build-$(date -u +%Y%m%dT%H%M%S)-$RANDOM"
export VERICCL_BUILD="$VERICCL_ROOT/examples/runtime-build/output/$VERICCL_BUILD_ID-runtime"
"$VERICCL_PYTHON" runtime/build_runtime.py --output-dir "$VERICCL_BUILD" \
  --cuda-home "$CUDA_HOME" --arch "$VERICCL_ARCH" --jobs 8 --timeout-s 900
export VERICCL_MSCCL="$VERICCL_BUILD/build"
"$VERICCL_PYTHON" runtime/build_probes.py \
  --output-dir "$VERICCL_ROOT/examples/runtime-build/output/$VERICCL_BUILD_ID-probes" \
  --cuda-home "$CUDA_HOME" --msccl-build "$VERICCL_MSCCL" \
  --mpi-bin "$MPI_HOME/bin" --arch "$VERICCL_ARCH" --timeout-s 180
"$VERICCL_PYTHON" runtime/build_native.py \
  --output-dir "$VERICCL_ROOT/examples/runtime-build/output/$VERICCL_BUILD_ID-native" \
  --cuda-home "$CUDA_HOME" --msccl-build "$VERICCL_MSCCL" \
  --mpi-bin "$MPI_HOME/bin" --arch "$VERICCL_ARCH" --timeout-s 180
export VERICCL_PROBE="$VERICCL_ROOT/examples/runtime-build/output/$VERICCL_BUILD_ID-probes/build/bin/vericcl-probe"
export VERICCL_NATIVE="$VERICCL_ROOT/examples/runtime-build/output/$VERICCL_BUILD_ID-native/build/bin/vericcl-native"
export LD_LIBRARY_PATH="$VERICCL_MSCCL/lib:$MPI_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
ldd "$VERICCL_PROBE"
ldd "$VERICCL_NATIVE"
```

Use `ldd` to check that CUDA, MPI, and MSCCL resolve to the selected installations. The C interface library and headers are in the native build's `build/lib` and `build/include`. On multiple nodes, use the same `VERICCL_BUILD_ID` so that executable paths match.

## Inputs

Each participating GPU corresponds to a rank numbered from `0` to `P-1`. Four JSON files define the task size and configuration. The commands below apply to both single-node and multi-node runs.

| File | Contents |
|---|---|
| `topology.json` | Rank count, node groups, GPU links, shared resources, and initial costs |
| `sketch.json` | Collective, message and slice sizes, solver resources, time limits, and measurement switches |
| `atom.json` | Synthesis method, communication stages, and user constraints |
| `runtime.json` | Rank-to-host and GPU mapping, program paths, and environment variables |

M is the input size in bytes per rank, and S is the size in bytes per slice. M must be a multiple of S; the program computes N=M/S. Each slice must also contain whole elements of the selected data type. For example, FP32 requires S to be a multiple of 4.

| Collective | Input per rank | Output per rank | nccl-tests `-b` and `-e` |
|---|---|---|---|
| AllGather | M | P×M | P×M |
| AllReduce | M | M | M |

**Set the task size**

First, set the case directory and communication parameters. This example uses two hosts with four GPUs each. Replace `host-a` and `host-b` with hostnames that resolve consistently on all participating nodes, and add or remove hosts and GPU IDs as needed. For a single node, use `{"localhost":[0,1,2,3]}`. Run configuration-generation commands from `VERICCL_ROOT` so Python can import the example helpers.

<!-- vericcl-v5-command: case-setup -->
```bash
cd "$VERICCL_ROOT"
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

Set `VERICCL_OPERATOR` to `allgather` or `allreduce`, and `VERICCL_METHOD` to `hierarchy` or `constructive_trees`. For example, a 2×8 layout uses two lists of eight GPU IDs, and a 4×4 layout uses four lists of four IDs. List the IDs directly when nodes have different GPU counts or nonconsecutive IDs.

**Generate the four configuration files**

The template below generates node groups, rank mappings, and message settings from these parameters. It uses an abstract topology with direct communication within and between nodes. Each host's inter-node sends share one bandwidth resource, and its receives share another. Adjust the initial costs to match your hardware. For other connection patterns, edit `directed_links` and `shared_resources`; the remaining commands stay the same.

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
# Units: alpha in microseconds, bandwidth in bytes per microsecond.
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
print('Rank count:', request.rank_count, 'Input/output bytes:', dict(request.size_accounting))
print('Configuration directory:', os.environ['VERICCL_INPUT'])
PY
```

You can edit the generated JSON. When changing the task size, ensure that `topology.ranks`, `nodes[].ranks`, and `runtime.rank_map` cover the same ranks, with unique GPU IDs on each host. Links are directed, so list both directions for a bidirectional connection. When links share a NIC, switch port, or another resource, list them in `shared_resources[].links` and add the resource ID to each link's `resources`.

The defaults are FP32 and out-of-place buffers, with sum for AllReduce. Set `sketch.collective.inplace=true` for in-place execution. `scratch_bytes_per_rank` is the temporary storage budget per rank. `NCCL_BUFFSIZE` accepts integer byte counts from 8 to 2147483647. For the Simple protocol, choose a multiple of 8192 that is at least twice S; the 64 MiB example supports slices up to 32 MiB. As message sizes and rank counts grow, adjust these budgets to fit GPU memory.

`max_run_wall_s` limits total wall time, and each stage also has its own time limit. With `max_tuning_rounds=0`, the run ends after generating and checking the initial schedule. For larger tasks, adjust solver time, CPU budgets, and slice size. See the [input reference](docs/input-schema.md) and [supported configurations](docs/supported-configuration.md) for all fields and runtime limits.

**Optional: manual stages and Atom variable groups**

Manual plans are enabled by `atom.manual.enabled=true`. Each node defines a communication group, input states, output states, and predecessors within a stage; `atoms` adds constraints that forbid directed edges for that node. Automatic synthesis defaults to `manual.enabled=false`. Manual plans use `hierarchy`; selecting `auto` also uses this solver path. Manual mode retains hierarchical subproblem reuse and keeps `sketch.history.schedule_reuse` at `false`, so the current plan defines the stages.

The following commands generate an editable manual plan for the current collective and task size in a new input directory. Stage inputs, outputs, and dependencies are generated from the existing data semantics. You can then edit `atom.json` as needed.

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
print('Manual input directory:', os.environ['VERICCL_MANUAL_INPUT'])
PY
```

Add variable groups to the target node's `atoms`, for example:

```json
"atoms": [
  {"slice_ids": [0, 1], "src_ranks": [0], "dst_ranks": [1, 2]}
]
```

This group expands into four constraints: each of the two slices is forbidden from transferring directly from rank 0 to ranks 1 and 2. The stage is taken from the node's `stage_id`. All combinations of the list values are used. Source and destination ranks must belong to the node's `group`, and each pair must have distinct endpoints. The specified directed edges are disabled; other valid relay paths remain available.

An original slice has `slice_id=source rank×N+index within the rank`, with indices starting at 0. Specify only the three sets above for Atom variables. The stage is inherited from the node, and the solver determines timing, transfer sizes, and paths. Constraints from all nodes, variable groups, and existing `forbidden_transfers` are combined, with duplicates removed. In `input_states` and `output_states`, `offset` is the logical position, and `contributors` lists the original slices contained in the state.

After editing, set `VERICCL_INPUT` to `VERICCL_MANUAL_INPUT`, then run the input argument setup and `solve` commands below. See the [input reference](docs/input-schema.md) for all fields.

## Run the synthesizer

**Set input and output directories**

If you already have the four configuration files, set `VERICCL_INPUT` to their directory. Set `--project-root` to keep output in a fixed location when running commands from any working directory.

<!-- vericcl-v5-command: common-args -->
```bash
VERICCL_ARGS=(--project-root "$VERICCL_ROOT" --case "$VERICCL_CASE"
  --topology "$VERICCL_INPUT/topology.json" --sketch "$VERICCL_INPUT/sketch.json"
  --atom "$VERICCL_INPUT/atom.json" --runtime "$VERICCL_INPUT/runtime.json")
```

Results from `solve`, `verify`, and `calibrate` are saved in `examples/<case>/output/<run_id>/`. If `--run-id` is omitted, an ID is generated from the start time. When setting it explicitly, use a new ID for each run.

**Synthesize a schedule: `solve`**

`solve` synthesizes a schedule from the four configuration files and runs the enabled measurement and repair steps. The general syntax is below; replace angle-bracket placeholders with actual values:

```text
"$VERICCL_PYTHON" -m vericcl solve --project-root <project-directory> --case <case-name> \
  --topology <topology-file> --sketch <communication-config> --atom <algorithm-config> --runtime <runtime-config>
```

Synthesize with the current configuration and save the output directory variable for inspection, verification, and testing:

<!-- vericcl-v5-command: solve -->
```bash
export VERICCL_RUN_ID="solve-$(date -u +%Y%m%dT%H%M%S)-$RANDOM"
export VERICCL_RUN="$VERICCL_ROOT/examples/$VERICCL_CASE/output/$VERICCL_RUN_ID"
"$VERICCL_PYTHON" -m vericcl solve "${VERICCL_ARGS[@]}" --run-id "$VERICCL_RUN_ID"
```

The final schedule is saved in `schedule.xml`, with its description in `schedule.json` and model in `model.json`. `run.json` records the effective configuration, current best schedule, and stop reason. By default, the template performs synthesis, verification, and multiple repair rounds on the CPU. The next section explains how to enable GPU measurements.

**Inspect results: `inspect`**

`inspect` reads the specified run directory and prints its status, best schedule, and checkpoints as JSON:

```text
"$VERICCL_PYTHON" -m vericcl inspect --run <run-directory>
```

Use the following commands to inspect the previous run. To inspect another run, first set `VERICCL_RUN` to its absolute path.

<!-- vericcl-v5-command: inspect -->
```bash
"$VERICCL_PYTHON" -m vericcl inspect --run "$VERICCL_RUN"
"$VERICCL_PYTHON" - <<'PY'
import json, os
from pathlib import Path
run = Path(os.environ['VERICCL_RUN'])
metadata = json.loads((run / 'run.json').read_text())
a = metadata['best']
if a is None:
    print('Run result:', metadata['result']['stop_reason'])
else:
    print('T_pred (microseconds):', a['t_pred_us'])
    print('XML:', run / 'schedule.xml')
    print('Schedule description:', run / 'schedule.json')
    print('Model:', run / 'model.json')
PY
```

`T_pred` is the predicted communication completion time under the current cost model. It is used to compare schedules for the same task under the same model.

**Verify an existing schedule: `verify`**

`verify` checks the XML using the configuration that matches the schedule. `--sidecar` specifies the schedule description generated with the XML, and `--model` specifies the evaluation model:

```text
"$VERICCL_PYTHON" -m vericcl verify --project-root <project-directory> --case <case-name> \
  --topology <topology-file> --sketch <communication-config> --atom <algorithm-config> --runtime <runtime-config> \
  --xml <xml-path> --sidecar <sidecar-path> --model <model-path>
```

The following commands extract the selected run's effective configuration, retaining command-line overrides used during synthesis, then verify the best XML, sidecar, and model. The actual slice partition is restored from the sidecar. The extracted configuration is written to a new input directory for this case, and verification results go to a new output directory.

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

Verification uses the measurement switches in the configuration. See `run.json` and `report.md` in the new output directory for results, and `process/` for individual records.

**Set the number of tuning rounds**

In `sketch.json`, set `max_tuning_rounds`, `max_repair_per_round_s`, and `max_repair_total_s` to control the round count, repair time per round, and total repair time. You can also change parameters for the current run with `--override-input --set field=JSON-value`, for example:

```bash
"$VERICCL_PYTHON" -m vericcl solve "${VERICCL_ARGS[@]}" --override-input \
  --set sketch.budgets.max_tuning_rounds=3 \
  --set sketch.budgets.max_repair_per_round_s=600 --set sketch.budgets.max_repair_total_s=1800
```

Each round verifies the current schedule and analyzes bottlenecks, converts suggestions into constraints, then solves locally and verifies candidates. An improved schedule starts the next round automatically. Exhausting a finite candidate set expands the search. The run ends at the time or round limit, or when a no-improvement proof is established over an explicitly defined legal repair space.

For an existing schedule, add `--tune` to the same `verify` command to enter this loop:

```text
"$VERICCL_PYTHON" -m vericcl verify <four-input-and-output-arguments> \
  --xml <xml-path> --sidecar <schedule-description-path> --model <model-path> --tune
```

## GPU measurement and execution

**Complete the runtime setup**

Configure passwordless SSH between participating nodes and launch from the rank 0 host. Use the same paths for VeriCCL, Python, MPI, CUDA, and executables on every node. Store all run artifacts under the executing account's home directory. `runtime.rank_map` retains the task size and GPU mapping defined earlier.

Set the control address on the rank 0 host to an IPv4 address that other nodes can reach, and choose an unused port. For a single node, use `127.0.0.1`. The following commands copy the current inputs and add GPU settings. Adjust measurement time limits, trace capacity, and sample sizes for the task.

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
    payload_sizes=[])
write_documents(Path(os.environ['VERICCL_GPU_INPUT']), d)
PY
GPU_ARGS=(--project-root "$VERICCL_ROOT" --case "$VERICCL_CASE"
  --topology "$VERICCL_GPU_INPUT/topology.json" --sketch "$VERICCL_GPU_INPUT/sketch.json"
  --atom "$VERICCL_GPU_INPUT/atom.json" --runtime "$VERICCL_GPU_INPUT/runtime.json")
```

An empty `payload_sizes` lets the calibrator select at least three legal sizes around the effective slice size. Custom probe sizes must fit one Simple-protocol transfer and contain whole elements; they are independent of the full message size. Add network settings such as `NCCL_SOCKET_IFNAME` and `NCCL_IB_HCA` to `runtime.environment` as needed. Use actual values and absolute paths in the JSON.

**Execute a schedule and capture a trace**

`sketch.trace_and_measure.enabled` enables both tracing and measured comparisons. After warmup, the program records one complete invocation. When comparing a repaired candidate with an accepted schedule, it measures their execution times separately with tracing disabled. Enable this switch with the usual `solve` inputs:

```bash
"$VERICCL_PYTHON" -m vericcl solve "${GPU_ARGS[@]}" \
  --override-input --set sketch.trace_and_measure.enabled=true
```

The output directory is printed to the terminal. Set `VERICCL_RUN` to that directory before verification or nccl-tests.

**Measure the cost model: `calibrate`**

`calibrate` independently measures link, shared-bandwidth, local-computation, and memory-access costs. The general syntax is:

```text
"$VERICCL_PYTHON" -m vericcl calibrate --project-root <project-directory> --case <case-name> \
  --topology <topology-file> --sketch <communication-config> --atom <algorithm-config> --runtime <runtime-config> \
  --override-input --set sketch.calibration.enabled=true
```

Measure with the current GPU configuration:

```bash
"$VERICCL_PYTHON" -m vericcl calibrate "${GPU_ARGS[@]}" \
  --override-input --set sketch.calibration.enabled=true
```

The final model is saved in `model.json`. A component is updated after its samples and fit pass the configured checks; `run.json` and `report.md` record the outcome and remaining estimates. Model versions and differences are in `process/models/`; measurement jobs and logs are in `process/jobs/`. To include measurement, verification, and repair in synthesis, run:

```bash
"$VERICCL_PYTHON" -m vericcl solve "${GPU_ARGS[@]}" --override-input \
  --set sketch.calibration.enabled=true --set sketch.budgets.max_tuning_rounds=3 \
  --set sketch.budgets.max_repair_per_round_s=600 --set sketch.budgets.max_repair_total_s=1800
```

After a model update, the program reevaluates and compares schedules under the same model. See the [architecture](docs/architecture.md) for the measurement and repair workflow.

**Optional: compare with nccl-tests**

First, build the same nccl-tests version on each participating node. The following commands build the test programs. `--release-receipts` records GPU release status after each process finishes:

```bash
export VERICCL_TEST_BUILD_ID="tests-$(date -u +%Y%m%dT%H%M%S)-$RANDOM"
"$VERICCL_PYTHON" "$VERICCL_ROOT/examples/nccl-tests-build/input/build.py" --run-id "$VERICCL_TEST_BUILD_ID" \
  --cuda-home "$CUDA_HOME" --mpi-home "$MPI_HOME" \
  --msccl-build "$VERICCL_MSCCL" --arch "$VERICCL_ARCH" --release-receipts
export VERICCL_NCCL_TESTS="$VERICCL_ROOT/examples/nccl-tests-build/output/$VERICCL_TEST_BUILD_ID/process/rebuildable-cache/source/build"
```

Use the same `VERICCL_TEST_BUILD_ID` on all participating nodes. Launch tests with `mpirun`. `-np` sets the process count, `--host` sets host placement, and `-g 1` uses one GPU per process. Use `all_gather_perf` for AllGather and `all_reduce_perf` for AllReduce:

```text
mpirun -np <rank-count> --host <host:process-count,...> <test-program> \
  -b <test-bytes> -e <test-bytes> -g 1 -d <data-type> -w 5 -n 20 -c 1
```

The following template reads the best schedule's configuration from `VERICCL_RUN`, converts message sizes, and assigns a host and GPU to each rank. It also supports different GPU counts per node and nonconsecutive GPU IDs. It runs Ring as the baseline and saves the log in this case's `output/nccl-tests-*/nccl-tests.log`. Each node saves its own release records under the same path.

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
# Each MPI process uses its configured GPU and creates the local release-record directory before testing.
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
print('Log:', output / 'nccl-tests.log', flush=True)
with (output / 'nccl-tests.log').open('w') as log:
    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
PY
```

nccl-tests reports both in-place and out-of-place results; use the column matching `collective.inplace`. Execute and trace VeriCCL XML through the `solve` / `verify` workflow above. See the [nccl-tests source](https://github.com/NVIDIA/nccl-tests/blob/0b4c4cb99fb1381edec1f78c37230688ea1ceb26/src/common.cu#L825-L831) for device selection rules.

## Results and troubleshooting

Results are saved in `examples/<case>/output/<run_id>/`, with inputs under the same case's `input/` directory.

| Top-level file or directory | Contents and generation conditions |
|---|---|
| `run.json` | Run identity, command, input snapshot index, effective configuration, sources of defaults and overrides, size accounting, stage timing, results, and file index |
| `report.md` | Run result, check conclusions, predicted time, stop reason, and issues |
| `run.log` | Chronological log of the main workflow |
| `schedule.xml`, `schedule.json` | Accepted schedule with semantics, buffers, TBs, and event mappings; generated after a schedule qualifies |
| `model.json` | Model matching the accepted schedule; otherwise the available current or checkpoint model, including standalone calibration output |
| `checkpoint.json` | Latest checkpoint state, elapsed time, round count, and process file references |
| `trace.json` | Trace analysis of the accepted schedule; generated when tracing is enabled and complete results are available |
| `process/` | All intermediate records, historical versions, and files needed for recovery |
| `details.zip` | Archive copy of `process/` created at the end for convenient storage; `process/` remains available for recovery |

In `run.json`, `invocation`, `input_snapshot`, `effective_request`, `value_sources`, and `size_accounting` store the command, original input index, effective configuration, value sources, and byte-size conversions, respectively. `best` points to the accepted candidate, `result` records status and stop reason, `timing` records stage times and round counts, and `artifacts` lists the generated top-level files.

| Location inside `process/` | Contents |
|---|---|
| `input_snapshot/` | Original input files and the XML, schedule description, and model read by verify |
| `models/` | Initial and calibrated models, component differences, probe inputs, and samples |
| `candidates/<id>/` | Each candidate's `candidate.json`, XML, schedule description, checks, and predictions |
| `jobs/<id>/` | Subprocess requests, identities, stdout, stderr, exit records, and cleanup records; remote jobs also include per-rank receipts |
| `checkpoints/<id>/`, `checkpoints/latest.json` | Recovery snapshots for each stage, plus the latest checkpoint location and checksum |
| `statistics/` | Sample IDs and statistics for measured comparisons |
| `live/` | Resource use and release status for native execution |
| `reports/`, `best/` | Analysis, proofs, timing and trace details, and best-candidate records |
| `tmp/`, `rebuildable-cache/` | Data that can be regenerated, such as temporary solver inputs and build sources |
| `events.jsonl`, `run-summary.json` | Structured event stream and final run record; per-rank logs are also stored in this directory |

`solve` outputs results for the full loop. `verify` outputs checks for an existing schedule, with `--tune` also recording each repair round. `calibrate` outputs the model and measurement records. `inspect` displays existing results in the terminal, and `resume` continues recording in the original directory. Build commands also keep libraries and programs in `build/`, with diagnostic records in `process/`. Files are generated for the stages actually executed; models, candidates, and jobs retain their own IDs.

**Resume a run: `resume`**

`resume` continues from the checkpoint in the specified run directory:

```text
"$VERICCL_PYTHON" -m vericcl resume --run <run-directory>
```

Set `VERICCL_RUN` to the directory to resume, then run:

```bash
"$VERICCL_PYTHON" -m vericcl resume --run "$VERICCL_RUN"
```

Results continue in the original directory, and elapsed time remains cumulative for that run. If solving finished but output finalization was interrupted, `resume` completes the exports and archive. Use `--help` with any command to see all arguments, for example:

```bash
"$VERICCL_PYTHON" -m vericcl solve --help
```

See `run.log` for execution details. If configuration or schedule checks fail, first inspect the results in `report.md`, `process/reports/`, and `process/candidates/`. For dynamic library loading errors, check the CUDA, MPI, and MSCCL paths. See the [runtime documentation](runtime/msccl-trace/README.md) for the trace interface.

**Run CPU regression tests**

The following command runs the CPU tests, including the documented input, synthesis, and verification examples. Keep its temporary directory under your home directory to exercise the same storage boundary as native jobs:

```bash
cd "$VERICCL_ROOT"
export VERICCL_TEST_ID="cpu-$(date -u +%Y%m%dT%H%M%S)-$RANDOM"
"$VERICCL_PYTHON" -m pytest -q -m "not hardware" \
  --basetemp "$VERICCL_ROOT/examples/cpu-tests/output/$VERICCL_TEST_ID/tmp"
```
