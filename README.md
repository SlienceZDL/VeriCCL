# VeriCCL

[Chinese](README.zh-CN.md)

## Overview and supported collectives

VeriCCL generates and validates MSCCL XML schedules from topology, sketch, and atom JSON inputs. It checks input, collective semantics, state, topology, timing, resources, buffers, endpoints, deadlock freedom, XML compatibility, BDD flow, event simulation, and optional online execution.

Direct solving supports `broadcast`, `reduce`, `allgather`, `allreduce`, `alltoall`, and `reduce_scatter`. `scatter` and `gather` remain fully defined collective semantics, but are internal composition stages rather than direct input operators. Hierarchical plans compose these eight semantics into a schedule that satisfies the requested global collective.

The supported server baseline is x86_64 Ubuntu 22.04 or 24.04, Python 3.10-3.12, and one or more NVIDIA GPUs. Multi-node runs additionally require passwordless SSH, synchronized clocks, and the same installation paths on every node.

## Installation modes

Offline use needs Python and Gurobi. Full online use additionally needs a compatible NVIDIA driver and CUDA Toolkit, the VeriCCL MSCCL runtime, Open MPI, MPI-enabled NCCL Tests, and the clock helper. Either MSCCL Strategy A or Strategy C below produces `$MSCCL_ROOT/build/lib` with the same runtime-relevant source hashes.

## Ubuntu prerequisites

Run the exact package sequence on Ubuntu 22.04 or 24.04:

```bash
sudo apt update
sudo apt install -y build-essential git patch python3 python3-dev python3-pip python3-venv openmpi-bin libopenmpi-dev wget ca-certificates
```

CUDA is intentionally not pinned here. Select the server driver and CUDA Toolkit from NVIDIA's current compatibility tables and the [CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html). Confirm that the CUDA-supported host compiler and the target GPU architecture are compatible before compiling MSCCL.

## CUDA, NCCL, and MPI preflight

Set `CUDA_HOME` to the installed toolkit. The Open MPI prefix below follows Ubuntu's multiarch package layout and avoids resolving the alternatives wrapper to `/usr`.

```bash
uname -m
. /etc/os-release && printf '%s %s\n' "$NAME" "$VERSION_ID"
python3 --version
gcc --version
nvidia-smi
export CUDA_HOME=/usr/local/cuda
test -x "$CUDA_HOME/bin/nvcc"
"$CUDA_HOME/bin/nvcc" --version
mpirun --version
mpicxx --showme:version
export MPI_HOME="/usr/lib/$(dpkg-architecture -qDEB_HOST_MULTIARCH)/openmpi"
test -f "$MPI_HOME/include/mpi.h"
test -d "$MPI_HOME/lib"
```

These commands verify discovery, not end-to-end CUDA or NCCL compatibility. The active NCCL implementation for later tests is built through MSCCL.

## Clone and Python install

SSH is the primary clone path:

```bash
git clone git@github.com:SlienceZDL/VeriCCL.git
cd VeriCCL
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pip install -e .
```

If SSH authentication is unavailable, use HTTPS and then run the same virtual-environment commands from the preceding block:

```bash
git clone https://github.com/SlienceZDL/VeriCCL.git
cd VeriCCL
```

Verify the editable package and its imports:

```bash
export VERICCL_ROOT="$(pwd)"
.venv/bin/python -m pip check
.venv/bin/python -c 'import vericcl, gurobipy, lxml, numpy, z3; print(vericcl.__version__)'
.venv/bin/python -m vericcl --version
```

Expected version literal: `0.1.0`.

## Gurobi license

The `gurobipy` wheel includes a size-limited license only. It is sufficient for the one-variable check below, but full VeriCCL MILP models require an appropriate academic, commercial, evaluation, WLS, local, or network license. Constructive-only runs can use `vericcl/examples/atom/constructive.json` without invoking the MILP solver.

On Linux, common default license locations are `~/gurobi.lic` and `/opt/gurobi/gurobi.lic`. For a non-default file, set `GRB_LICENSE_FILE` to its absolute path. Follow Gurobi's [Python installation guidance](https://support.gurobi.com/hc/en-us/articles/360044290292-How-do-I-install-Gurobi-for-Python) and [full-license guidance](https://support.gurobi.com/hc/en-us/articles/360051597492-How-do-I-resolve-a-Model-too-large-for-size-limited-Gurobi-license-error). Never commit a license file, WLS credentials, access IDs, secrets, or a site-specific token.

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

Expected result literal: `gurobi optimize check passed`. A later `Model too large for size-limited Gurobi license` error means the import works but the active license is not suitable for the requested MILP.

## Offline smoke test

The following four marker commands are hardware-independent and are executed in order by the documentation test. Use an empty `${VERICCL_OUTPUT_DIR}` because run directories are never overwritten.

<!-- vericcl-doc-test: help -->
```bash
.venv/bin/python -m vericcl --help
```

<!-- vericcl-doc-test: solve -->
```bash
.venv/bin/python -m vericcl solve --topology vericcl/examples/topo/two_rank.json --sketch vericcl/examples/sketch/allreduce_8m_1m.json --atoms vericcl/examples/atom/constructive.json --output-dir ${VERICCL_OUTPUT_DIR} --run-id docs
```

<!-- vericcl-doc-test: verify -->
```bash
.venv/bin/python -m vericcl verify --topology vericcl/examples/topo/two_rank.json --sketch vericcl/examples/sketch/allreduce_8m_1m.json --atoms vericcl/examples/atom/constructive.json --output-dir ${VERICCL_OUTPUT_DIR} --run-id docs-verify --xml ${VERICCL_OUTPUT_DIR}/vericcl_allreduce_8MiB_docs/vericcl_allreduce_8MiB_final.xml
```

<!-- vericcl-doc-test: example-validation -->
```bash
.venv/bin/python -m pytest tests/e2e/test_six_collectives.py -q
```

The constructive two-rank example uses an 8 MiB AllReduce split into eight 1 MiB software slices and does not modify its source inputs.

## MSCCL Strategy A: official source plus bundled patch

Strategy A starts from the official MSCCL repository at immutable commit `b23e9cd5dd63f82ee1c5aae7e0a2042079be903a`. The verifier checks the clean pinned base in a temporary copy, including the final hashes; the actual checkout is then patched and built.

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

`verification passed` is the expected verifier result. The output library directory is `$MSCCL_ROOT/build/lib`.

## MSCCL Strategy C: pre-integrated immutable tag

Strategy C uses public tag `vericcl-runtime-v0.1.0` at commit `782ee5f72cf48c1ae1a2365bcf525019f5620175`. The patched-tree verifier checks the commit and every hash in `runtime/msccl-trace/upstream.json` before the same build.

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

`verification passed` is the expected verifier result. Strategies A and C have the same hashes for the trace header and runtime source files recorded in `upstream.json`; both produce `$MSCCL_ROOT/build/lib`. Static verification is not evidence of CUDA compilation or GPU execution.

## NCCL Tests and clock helper build

Build the current [NCCL Tests](https://github.com/NVIDIA/nccl-tests) source against the selected MSCCL tree. `MPI_HOME` uses the Ubuntu multiarch package location and is checked before compilation.

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

The required official build form uses Ubuntu's x86_64 Open MPI prefix. On another Ubuntu architecture, replace the literal `MPI_HOME` in the `make` command with the verified value of `$MPI_HOME`.

## Input schemas and examples

All three inputs are UTF-8 JSON objects. Unknown fields, duplicate keys, non-finite numbers, and inconsistent dimensions are rejected.

Topology (`vericcl/examples/topo/two_rank.json` and `vericcl/examples/topo/two_node_gateway.json`):

- `ranks` is a positive global rank count. `nodes` must cover every rank exactly once; every gateway must belong to its node.
- `directed_links` contains unique non-self `(src,dst)` links. `max_channels` is a positive concurrency cap; `resources` names shared-resource IDs.
- `alpha`/`alpha_us`, `beta`/`beta_us`, and `invbw`/`invbw_us` are non-negative microsecond parameters. `invbw` is authoritative, must be at least `alpha`, and should equal `alpha + beta`. Optional `bandwidth_bytes_per_us` maps each integer concurrency to aggregate bytes per microsecond.
- `shared_resources` gives an ID, existing directed `member_links`, a positive `max_channels`, and the same timing fields. It models contention such as a NIC ingress, egress, or inter-node link.

Sketch (`vericcl/examples/sketch/allreduce_8m_1m.json`):

- `collective.operator` is one of the six direct operators above. `root` is required only for `broadcast` and `reduce`; reduction operators require `reduction_op` in `avg|max|min|prod|sum`. `datatype` is non-empty and `inplace` is Boolean.
- `total_size_bytes` and `slice_size_bytes` are positive byte counts. Total size must be divisible by slice size; `input_chunkup`, when present, must equal their quotient. `alltoall` and `reduce_scatter` additionally require the slice count to be divisible by `ranks`.
- Hyperparameters control objective selection, calibration concurrency, tuning thresholds/iterations, verification timeout, and forced recalibration. The example shows their accepted types and defaults.
- `solver` controls total/per-model seconds, `mip_gap` in `[0,1]`, proof requirement, deterministic seed, channels in `[1,32]`, thread/model parallelism, and forced re-solving.

Atom (`vericcl/examples/atom/constructive.json` and `vericcl/examples/atom/default.json`):

- `stage_num` is `null` or a positive exact stage count. Each forbidden transfer is `[slice_id, src_rank, dst_rank, stage_id]`; ranks must differ and all indexes must be in range.
- Strategy Booleans select hierarchy, symmetry, shortest paths, batching, constructive trees, and MILP. The constructive file disables MILP; the default file enables it.
- `manual_hierarchy` is empty for automatic planning. A manual node defines `node_id`, zero-based `stage_id`, an operator, a sorted unique `communication_group`, optional rooted `root`, `logical_input` and `logical_output` entries `[rank, offset, contributor_ids]`, plus unique `depends_on` node IDs. Interfaces and dependencies must compose exactly to the global collective.

Inspect the real supported examples directly:

```bash
.venv/bin/python -m json.tool vericcl/examples/topo/two_rank.json
.venv/bin/python -m json.tool vericcl/examples/topo/two_node_gateway.json
.venv/bin/python -m json.tool vericcl/examples/sketch/allreduce_8m_1m.json
.venv/bin/python -m json.tool vericcl/examples/atom/constructive.json
.venv/bin/python -m json.tool vericcl/examples/atom/default.json
```

`vericcl/examples/legacy` and `vericcl/examples/templates` are reference-only and are not supported runtime inputs.

## Solve and verify

`solve` creates a new deterministic directory under `--output-dir`; `verify` checks an XML and its inferred `.schedule.json` sidecar, or a path supplied with `--sidecar`. The four smoke-test commands above show the actual syntax. Direct inputs preserve the semantics of Broadcast, Reduce, AllGather, AllReduce, AllToAll, and ReduceScatter; internal Scatter and Gather stages preserve the remaining two semantics during composition.

## Overrides, hierarchy, tuning, and timeout

Semantic CLI values that differ from the sketch are rejected unless `--override-input` is present. Overrides are written to a temporary effective sketch and never mutate the input. `--tune` enables verified local repair, while positive `--timeout-s` bounds the requested workflow.

```bash
.venv/bin/python -m vericcl solve --topology vericcl/examples/topo/two_rank.json --sketch vericcl/examples/sketch/allreduce_8m_1m.json --atoms vericcl/examples/atom/constructive.json --operator allreduce --total-size-bytes 4194304 --slice-size-bytes 1048576 --out-of-place --override-input --tune --timeout-s 600 --output-dir runs --run-id override-tune
.venv/bin/python -m vericcl verify --topology vericcl/examples/topo/two_rank.json --sketch vericcl/examples/sketch/allreduce_8m_1m.json --atoms vericcl/examples/atom/constructive.json --timeout-s 600 --output-dir runs --run-id verify-existing --xml /absolute/path/to/schedule.xml --sidecar /absolute/path/to/schedule.schedule.json
```

For the real two-node gateway topology, create a hierarchy policy from the packaged default and solve an eight-rank AllReduce. The gateway ranks are `0` and `4`.

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

Automatic hierarchy discovers the real intra-node groups and the connected gateway group. Use `manual_hierarchy` only when its logical interfaces and dependencies have been derived explicitly.

## Online validation

`--online` requires a runtime-compatible XML, exactly one XML per execution, and the environment below. Use real local version strings; they are part of the calibration-cache signature. Set site-specific GPU/NIC labels without embedding secrets.

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

Calibration uses exactly 128 MiB and the input slice size. The slice size must divide 128 MiB or calibration is `not_run`. Intra-node calibration launches one process with `-g 2`; inter-node calibration launches two MPI processes with `-g 1`. A stable `solve --online` calibration updates matching `invbw`/concurrency limits and solves again. `verify --online` keeps the supplied XML unchanged and reports `requires_resolve=true` when recalibration indicates a new solve.

Release measurement and trace diagnosis are separate runs. Release uses five warmups, 20 measurements, correctness checks, and `VERICCL_TRACE_ENABLE=0`; trace uses zero warmups, 20 measurements, `-c 0`, and `VERICCL_TRACE_ENABLE=1`. Trace cost is never performance data.

## Single-node XML execution

The fixed runtime contract is `MSCCL_CHUNKSTEPS=4`, `MSCCL_SLICESTEPS=4`, `NCCL_PROTO=Simple`, XML `cnt=1`, and `NCCL_BUFFSIZE=2*slice_size_bytes`. For the packaged 1 MiB slice, the buffer is 2 MiB. Set exactly one XML and match its message size, datatype, reduction operation, root, rank count, and in-place mode.

Release run on one node with two GPUs:

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

Separate diagnostic run:

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

## Multi-node XML execution

Use one MPI process per rank and one GPU per process. This eight-rank example uses four ranks on each of two hosts. The hostfile and XML must exist at the same paths on all nodes.

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

## Outputs, exit codes, and troubleshooting

For the smoke input and run ID `docs`, the directory is `vericcl_allreduce_8MiB_docs/`. It contains `resolved-input.json`, `run-summary.json`, `schedules/`, `reports/`, `traces/`, the final `.xml`, `.schedule.json`, and `.validation.json`. Offline-valid schedules that exceed MSCCL limits use `.candidate.xml` and must not be executed until re-solved.

Exit codes are: `0` for offline-valid completion (including runtime-warning candidates), `2` for input/configuration errors, `3` for no semantic-valid candidate or offline timeout, `4` for requested online failure/timeout, and `5` for an internal error.

Common diagnoses:

- CUDA/MSCCL build failure: recheck the driver/toolkit/host-compiler compatibility table, `CUDA_HOME`, GPU architecture, and the full compiler output.
- `mpi.h` or MPI link failure: re-run the multiarch `MPI_HOME` checks and confirm `libopenmpi-dev` is installed.
- MSCCL verifier failure: use the exact upstream commit or fork tag, a clean Strategy A checkout, and an unmodified bundled header/patch.
- `Model too large...`: activate a full Gurobi license; reinstalling `gurobipy` does not expand the bundled limit.
- Missing online binary/library: check the three `VERICCL_*_BUILD_DIR`/binary paths and propagate `LD_LIBRARY_PATH` to every MPI rank.
- Runtime mismatch: use Simple protocol, `NCCL_BUFFSIZE=2*slice_size_bytes`, step constants `4/4`, `cnt=1`, one XML, and matching NCCL Tests arguments.
- Trace/clock failure: keep release and trace runs separate, increase `VERICCL_TRACE_RECORDS` if required, and inspect clock uncertainty and per-rank trace files.

## Software tests and references

Run hardware-independent tests on any development host:

```bash
.venv/bin/python -m pytest tests/integration/test_documented_commands.py -q
.venv/bin/python -m pytest tests/unit/online/test_runtime_patch.py -q
git diff --check
```

CUDA compilation and GPU execution were not performed on the macOS documentation host. Run the marked server build and execution commands on the target Ubuntu GPU environment before reporting hardware validation.

Further references: [runtime configuration](docs/runtime-configuration.md), [validation reports](docs/validation-report.md), [MSCCL trace patch](runtime/msccl-trace/README.md), [migration notes](MIGRATION.md), [NVIDIA CUDA installation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html), [NCCL Tests](https://github.com/NVIDIA/nccl-tests), and [official MSCCL](https://github.com/microsoft/msccl.git).
