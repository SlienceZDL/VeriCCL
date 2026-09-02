# VeriCCL

[Chinese](README.zh-CN.md)

## Overview

VeriCCL plans, solves, and validates collective-communication schedules. It consumes topology, sketch, and atom JSON inputs, then produces MSCCL XML, a schedule sidecar, and an offline validation report.

Direct solving supports `broadcast`, `reduce`, `allgather`, `allreduce`, `alltoall`, and `reduce_scatter`. `scatter` and `gather` are internal staged operators; together with the six directly solved operators, they define eight collective semantics.

Hardware validation: `not_run`. The workflow below validates generated artifacts in software and cannot establish CUDA compilation, MSCCL loading, GPU execution, or performance results; see [runtime configuration](docs/runtime-configuration.md) before preparing those environments.

## Building and Installing VeriCCL

VeriCCL supports CPython 3.10-3.13. Run this preflight before creating a virtual environment:

<!-- vericcl-doc-test: python-version -->
```bash
python3 -c 'import sys; v = sys.version_info[:2]; sys.exit("VeriCCL requires Python 3.10-3.13; found {}.{}.".format(*v)) if not (3, 10) <= v < (3, 14) else print("VeriCCL Python version check passed: {}.{}".format(*v))'
```

Clone with SSH, create a virtual environment, install the development dependencies, and install VeriCCL in editable mode:

```bash
git clone git@github.com:SlienceZDL/VeriCCL.git
cd VeriCCL
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pip install -e .
```

If SSH authentication is unavailable, clone with HTTPS and run the same virtual-environment commands from the preceding block:

```bash
git clone https://github.com/SlienceZDL/VeriCCL.git
cd VeriCCL
```

Set the repository root and verify dependencies, imports, and the CLI version:

```bash
export VERICCL_ROOT="$(pwd)"
.venv/bin/python -m pip check
.venv/bin/python -c 'import vericcl, gurobipy, lxml, numpy, z3; print(vericcl.__version__)'
.venv/bin/python -m vericcl --version
```

Expected version literal: `0.1.0`.

### Gurobi license check

The one-variable check confirms that the active Gurobi license can solve a minimal model. The constructive quickstart below disables MILP and does not require a full Gurobi license.

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

Expected result literal: `gurobi optimize check passed`.

## Running VeriCCL

The packaged inputs are `vericcl/examples/topo/two_rank.json`, `vericcl/examples/sketch/allreduce_8m_1m.json`, and `vericcl/examples/atom/constructive.json`. The help command lists the available CLI operations:

<!-- vericcl-doc-test: help -->
```bash
.venv/bin/python -m vericcl --help
```

Set `VERICCL_ROOT` to the root of the installed VeriCCL repository. All following relative paths start from this directory.

<!-- vericcl-run-step: set-root -->
```bash
export VERICCL_ROOT="$(pwd)"
```

Use a new `VERICCL_OUTPUT_DIR` for each run; it is the parent directory for this run.

<!-- vericcl-run-step: set-output -->
```bash
export VERICCL_OUTPUT_DIR="$VERICCL_ROOT/runs/readme-$(date +%Y%m%dT%H%M%S)"
```

Create the output root. VeriCCL creates the operator, size, and run-ID directory below it.

<!-- vericcl-run-step: create-output -->
```bash
mkdir -p "$VERICCL_OUTPUT_DIR"
```

`two_rank.json` describes two ranks and their directed links. `allreduce_8m_1m.json` describes an 8 MiB AllReduce split into 1 MiB software slices. `constructive.json` selects the constructive strategy with MILP disabled. `VERICCL_OUTPUT_DIR` is the parent directory for this run, and `quickstart` is the stable run identifier.

<!-- vericcl-run-step: solve -->
<!-- vericcl-doc-test: solve -->
```bash
.venv/bin/python -m vericcl solve --topology vericcl/examples/topo/two_rank.json --sketch vericcl/examples/sketch/allreduce_8m_1m.json --atoms vericcl/examples/atom/constructive.json --output-dir "$VERICCL_OUTPUT_DIR" --run-id quickstart
```

`--xml` points to the final XML from the solve run. The verify output is written under `vericcl_allreduce_8MiB_quickstart-verify/`.

<!-- vericcl-run-step: verify -->
<!-- vericcl-doc-test: verify -->
```bash
.venv/bin/python -m vericcl verify --topology vericcl/examples/topo/two_rank.json --sketch vericcl/examples/sketch/allreduce_8m_1m.json --atoms vericcl/examples/atom/constructive.json --output-dir "$VERICCL_OUTPUT_DIR" --run-id quickstart-verify --xml "$VERICCL_OUTPUT_DIR/vericcl_allreduce_8MiB_quickstart/vericcl_allreduce_8MiB_final.xml"
```

Check that `$VERICCL_OUTPUT_DIR/vericcl_allreduce_8MiB_quickstart/vericcl_allreduce_8MiB_final.xml`, the executable MSCCL XML, exists.

<!-- vericcl-run-step: check-xml -->
```bash
test -f "$VERICCL_OUTPUT_DIR/vericcl_allreduce_8MiB_quickstart/vericcl_allreduce_8MiB_final.xml"
```

Check that `$VERICCL_OUTPUT_DIR/vericcl_allreduce_8MiB_quickstart/vericcl_allreduce_8MiB_final.validation.json`, the final offline validation report, exists.

<!-- vericcl-run-step: check-report -->
```bash
test -f "$VERICCL_OUTPUT_DIR/vericcl_allreduce_8MiB_quickstart/vericcl_allreduce_8MiB_final.validation.json"
```

Inspect `$VERICCL_OUTPUT_DIR/vericcl_allreduce_8MiB_quickstart/vericcl_allreduce_8MiB_final.validation.json`, the final offline validation report rather than the executable XML.

<!-- vericcl-run-step: inspect-report -->
```bash
.venv/bin/python -m json.tool "$VERICCL_OUTPUT_DIR/vericcl_allreduce_8MiB_quickstart/vericcl_allreduce_8MiB_final.validation.json"
```

`$VERICCL_OUTPUT_DIR/vericcl_allreduce_8MiB_quickstart/vericcl_allreduce_8MiB_final.schedule.json` is the final XML sidecar. `$VERICCL_OUTPUT_DIR/vericcl_allreduce_8MiB_quickstart/run-summary.json` records the solve workflow summary.

Run the software regression example independently when checking all directly solved collectives:

<!-- vericcl-doc-test: example-validation -->
```bash
.venv/bin/python -m pytest tests/e2e/test_six_collectives.py -q
```

## Input Configuration

All three inputs are UTF-8 JSON objects. Duplicate keys, non-finite numbers, and inconsistent dimensions are rejected. Unknown-field handling differs by input: topology validates recognized fields but currently does not reject extra keys; sketch preserves extra top-level keys but rejects unknown fields inside `collective`, `hyperparameters`, and `solver`; atom rejects unknown top-level fields.

<!-- input-unknown-fields: topology-extra=accepted; sketch-top-extra=preserved; sketch-sections-extra=rejected; atom-top-extra=rejected -->

Topology (`vericcl/examples/topo/two_rank.json` and `vericcl/examples/topo/two_node_gateway.json`):

- `ranks` is a positive global rank count. `nodes` must cover every rank exactly once; every gateway must belong to its node.
- `directed_links` contains unique non-self `(src,dst)` links. `max_channels` is a positive concurrency cap; `resources` names shared-resource IDs.
- `alpha`/`alpha_us`, `beta`/`beta_us`, and `invbw`/`invbw_us` are non-negative microsecond parameters. `invbw` is authoritative, must be at least `alpha`, and should equal `alpha + beta`. Optional `bandwidth_bytes_per_us` maps each integer concurrency to aggregate bytes per microsecond.
- `shared_resources` gives an ID, existing directed `member_links`, a positive `max_channels`, and the same timing fields. It models contention such as a NIC ingress, egress, or inter-node link.

Sketch (`vericcl/examples/sketch/allreduce_8m_1m.json`):

- `collective.operator` is one of the six direct operators above. `root` is required only for `broadcast` and `reduce`; reduction operators require `reduction_op` in `avg|max|min|prod|sum`. `datatype` is non-empty and `inplace` is Boolean.
- `total_size_bytes` and `slice_size_bytes` are positive byte counts. Total size must be divisible by slice size; `input_chunkup`, when present, must equal their quotient. `alltoall` and `reduce_scatter` additionally require the slice count to be divisible by `ranks`.
- Hyperparameters control objective selection, calibration concurrency, tuning thresholds/iterations, verification timeout, and forced recalibration. The example shows their accepted types and defaults.
- `solver` controls total/per-model seconds, `mip_gap` in `[0,1]`, proof requirement, deterministic seed, channels in `[1,16]`, thread/model parallelism, and forced re-solving.

Atom (`vericcl/examples/atom/constructive.json` and `vericcl/examples/atom/default.json`):

- `stage_num` is `null` or a positive exact stage count. Each forbidden transfer is `[slice_id, src_rank, dst_rank, stage_id]`; ranks must differ, slice and rank indexes must be in range, and `stage_id` must be non-negative. When `stage_num` is positive, `stage_id` must also be less than `stage_num`; when `stage_num` is `null`, no configured upper bound applies.
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

## Advanced Usage

`solve` creates a new deterministic directory under `--output-dir`; `verify` checks an XML and its inferred `.schedule.json` sidecar, or a path supplied with `--sidecar`. The four smoke-test commands above show the actual syntax. Direct inputs preserve the semantics of Broadcast, Reduce, AllGather, AllReduce, AllToAll, and ReduceScatter; internal Scatter and Gather stages preserve the remaining two semantics during composition.

### Semantic overrides, hierarchy, tuning, and timeout

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

For online operation, a stable `solve --online` calibration updates matching `invbw`/concurrency limits and solves again. In contrast, `verify --online` keeps the supplied XML unchanged and reports `requires_resolve=true` when recalibration indicates a new solve; its runtime environment and execution contract are specified below.

## MSCCL Runtime Evaluation

For CUDA, MPI, and server configuration, see [runtime configuration](docs/runtime-configuration.md). This section records the VeriCCL-specific source, activation, and evaluation contract; it does not provide an operating-system installation procedure.

The commands below target NVIDIA V100 (`compute_70`/`sm_70`). On NVIDIA A100, replace every `compute_70`/`sm_70` pair in `NVCC_GENCODE` and the clock-helper `nvcc` command with `compute_80`/`sm_80` before building.

### Strategy A: official source plus bundled patches

Strategy A starts from the official MSCCL repository at immutable commit `b23e9cd5dd63f82ee1c5aae7e0a2042079be903a`. The verifier checks the clean pinned base in a temporary copy, including both patches and the recorded hashes; the actual checkout is then patched and built.

```bash
export VERICCL_ROOT="$(pwd)"
export CUDA_HOME=/usr/local/cuda
export NVCC_GENCODE="-gencode=arch=compute_70,code=sm_70"
export MSCCL_ROOT="$(dirname "$VERICCL_ROOT")/msccl-official"
git clone https://github.com/microsoft/msccl.git "$MSCCL_ROOT"
git -C "$MSCCL_ROOT" checkout --detach b23e9cd5dd63f82ee1c5aae7e0a2042079be903a
python3 "$VERICCL_ROOT/runtime/msccl-trace/tools/verify_patch.py" --source-root "$MSCCL_ROOT" --base-tree
cp "$VERICCL_ROOT/runtime/msccl-trace/include/vericcl_trace_format.h" "$MSCCL_ROOT/src/include/vericcl_trace_format.h"
patch --directory="$MSCCL_ROOT" --strip=1 --input="$VERICCL_ROOT/runtime/msccl-trace/patches/0001-vericcl-fixed-step-trace.patch"
patch --directory="$MSCCL_ROOT" --strip=1 --input="$VERICCL_ROOT/runtime/msccl-trace/patches/0002-vericcl-host-step-signature.patch"
make -C "$MSCCL_ROOT" clean
make -C "$MSCCL_ROOT" -j NVCC_GENCODE="$NVCC_GENCODE" src.build
test -d "$MSCCL_ROOT/build/lib"
```

`verification passed` is the expected verifier result. The output library directory is `$MSCCL_ROOT/build/lib`.

### Strategy C: pre-integrated immutable tag

Strategy C uses public tag `vericcl-runtime-v0.1.0` at commit `782ee5f72cf48c1ae1a2365bcf525019f5620175`. The patched-tree verifier checks the commit and every hash in `runtime/msccl-trace/upstream.json` before the same build.

```bash
export VERICCL_ROOT="$(pwd)"
export CUDA_HOME=/usr/local/cuda
export NVCC_GENCODE="-gencode=arch=compute_70,code=sm_70"
export MSCCL_ROOT="$(dirname "$VERICCL_ROOT")/VeriCCL-MSCCL"
git clone --branch vericcl-runtime-v0.1.0 --depth 1 https://github.com/SlienceZDL/VeriCCL-MSCCL.git "$MSCCL_ROOT"
test "$(git -C "$MSCCL_ROOT" rev-parse HEAD)" = 782ee5f72cf48c1ae1a2365bcf525019f5620175
python3 "$VERICCL_ROOT/runtime/msccl-trace/tools/verify_patch.py" --source-root "$MSCCL_ROOT" --patched-tree
patch --directory="$MSCCL_ROOT" --strip=1 --input="$VERICCL_ROOT/runtime/msccl-trace/patches/0002-vericcl-host-step-signature.patch"
make -C "$MSCCL_ROOT" clean
make -C "$MSCCL_ROOT" -j NVCC_GENCODE="$NVCC_GENCODE" src.build
test -d "$MSCCL_ROOT/build/lib"
```

`verification passed` is the expected verifier result. Strategies A and C use the same trace implementation and both apply the host-side step-signature supplement before producing `$MSCCL_ROOT/build/lib`. The supplement makes the MSCCL network proxy use the same `4/4` chunk/slice signature as the device interpreter; without it, inter-node byte counts and proxy credits are inconsistent. Static verification is not evidence of CUDA compilation or GPU execution.

### NCCL Tests and clock helper build

Build the current [NCCL Tests](https://github.com/NVIDIA/nccl-tests) source against the selected MSCCL tree after completing the site-specific MPI setup in [runtime configuration](docs/runtime-configuration.md). `MPI_HOME` must name the active MPI installation prefix.

```bash
export VERICCL_ROOT="$(pwd)"
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export NCCL_TESTS_ROOT="$(dirname "$VERICCL_ROOT")/nccl-tests"
export MPI_HOME="$(dirname "$(mpicxx --showme:incdirs | awk '{print $1}')")"
test -f "$MPI_HOME/include/mpi.h"
test -d "$MPI_HOME/lib"
git clone https://github.com/NVIDIA/nccl-tests "$NCCL_TESTS_ROOT"
make -C "$NCCL_TESTS_ROOT" -j MPI=1 MPI_HOME="$MPI_HOME" CUDA_HOME="$CUDA_HOME" NCCL_HOME="$MSCCL_ROOT/build"
nvcc -ccbin mpicxx -O2 -std=c++11 -gencode=arch=compute_70,code=sm_70 "$VERICCL_ROOT/runtime/msccl-trace/tools/vericcl_clock_sync.cu" -o "$VERICCL_ROOT/runtime/msccl-trace/tools/vericcl_clock_sync"
test -x "$NCCL_TESTS_ROOT/build/all_reduce_perf"
test -x "$VERICCL_ROOT/runtime/msccl-trace/tools/vericcl_clock_sync"
```

`$MPI_HOME` is derived from the first public include directory reported by the active Open MPI `mpicxx` wrapper.

### Online validation environment

`--online` requires a runtime-compatible XML, exactly one XML per execution, and the environment below. Use real local version strings; they are part of the calibration-cache signature. Set site-specific GPU/NIC labels without embedding secrets.

```bash
export VERICCL_MSCCL_BUILD_DIR="$MSCCL_ROOT/build/lib"
export VERICCL_NCCL_TESTS_BUILD_DIR="$NCCL_TESTS_ROOT/build"
export VERICCL_CLOCK_SYNC_BINARY="$VERICCL_ROOT/runtime/msccl-trace/tools/vericcl_clock_sync"
export VERICCL_MPI_LAUNCHER="$(command -v mpirun)"
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

Calibration uses exactly 128 MiB and the input slice size. The slice size must divide 128 MiB or calibration is `not_run`. Both intra-node and inter-node calibration launch one MPI process per GPU with `-g 1`; only inter-node execution additionally requires a hostfile.
The calibrated concurrency limit is `K_effective=min(16,max_calibration_channels,128MiB/S,link_max_channels)` for slice size `S`; VeriCCL does not infer unmeasured concurrency points.
For its two-rank representative benchmark, inter-node calibration also uses `-N 1`, placing exactly one process on each node.
Keep the documented `--timeout-s 10800` for an uncached 16-point calibration. Each calibration point starts one correctness-enabled release process with five warmups and 20 measurements, followed by one 20-iteration trace process. The trace iterations determine calibration stability and `B_link(k)`. Final operator validation remains stricter: it uses rounds of 20 independent release processes, with up to three rounds when measurements are unstable.

Release measurement and trace diagnosis are separate runs. Release uses five warmups, 20 measurements, correctness checks, and `VERICCL_TRACE_ENABLE=0`; trace uses zero warmups, 20 measurements, `-c 0`, and `VERICCL_TRACE_ENABLE=1`. The aggregate nccl-tests time reported by the trace run is not performance data; calibration uses the sender-local step intervals recorded for its 20 iterations.

### MSCCL activation boundary

The positive XML-load signal is `NCCL INFO Connected 1 MSCCL algorithms`. NCCL Tests executes both out-of-place and in-place timing blocks, while one VeriCCL XML matches exactly one placement mode. `NCCL_ALGO=MSCCL,RING` therefore runs the matching block with MSCCL and lets the non-matching block fall back to Ring. A missing load signal, missing MSCCL step trace for the selected block, or fallback of the selected block is not VeriCCL schedule validation.

### Single-node XML execution

The fixed runtime contract is `MSCCL_CHUNKSTEPS=4`, `MSCCL_SLICESTEPS=4`, the bundled host-side step-signature supplement, `NCCL_PROTO=Simple`, XML `cnt=1`, and `NCCL_BUFFSIZE=2*slice_size_bytes`. For the packaged 1 MiB slice, the buffer is 2 MiB. Set exactly one XML and match its message size, datatype, reduction operation, root, rank count, and in-place mode.

Run a short INFO-level activation probe on one node with two GPUs. The `sed` check requires every reported connected-algorithm count to be exactly one and fails when the signal is absent:

<!-- vericcl-msccl-run: single-node-activation -->
```bash
export LD_LIBRARY_PATH="$MSCCL_ROOT/build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NCCL_ALGO=MSCCL,RING
export NCCL_PROTO=Simple
export NCCL_BUFFSIZE=2097152
export MSCCL_XML_FILES=/absolute/path/to/one-schedule.xml
export VERICCL_EXPECTED_MSCCL_CHUNKSTEPS=4
export VERICCL_EXPECTED_MSCCL_SLICESTEPS=4
export VERICCL_TRACE_ENABLE=0
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT
export VERICCL_MPI_LAUNCHER="$(command -v mpirun)"
MSCCL_ACTIVATION_LOG="$(mktemp)"
"$VERICCL_MPI_LAUNCHER" --bind-to none -np 2 -x LD_LIBRARY_PATH -x NCCL_ALGO -x NCCL_PROTO -x NCCL_BUFFSIZE -x MSCCL_XML_FILES -x VERICCL_EXPECTED_MSCCL_CHUNKSTEPS -x VERICCL_EXPECTED_MSCCL_SLICESTEPS -x VERICCL_TRACE_ENABLE -x NCCL_DEBUG -x NCCL_DEBUG_SUBSYS "$NCCL_TESTS_ROOT/build/all_reduce_perf" -b 8388608 -e 8388608 -w 0 -n 1 -c 1 -d float -o sum -g 1 2>&1 | tee "$MSCCL_ACTIVATION_LOG"
test "$(sed -n 's/.*NCCL INFO Connected \([0-9][0-9]*\) MSCCL algorithms.*/\1/p' "$MSCCL_ACTIVATION_LOG" | sort -u)" = 1
rm -f "$MSCCL_ACTIVATION_LOG"
```

After the probe succeeds, run the formal release measurement with debug logging and tracing explicitly disabled:

<!-- vericcl-msccl-run: single-node-release -->
```bash
unset NCCL_DEBUG NCCL_DEBUG_SUBSYS
export LD_LIBRARY_PATH="$MSCCL_ROOT/build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NCCL_ALGO=MSCCL,RING
export NCCL_PROTO=Simple
export NCCL_BUFFSIZE=2097152
export MSCCL_XML_FILES=/absolute/path/to/one-schedule.xml
export VERICCL_EXPECTED_MSCCL_CHUNKSTEPS=4
export VERICCL_EXPECTED_MSCCL_SLICESTEPS=4
export VERICCL_TRACE_ENABLE=0
export VERICCL_MPI_LAUNCHER="$(command -v mpirun)"
"$VERICCL_MPI_LAUNCHER" --bind-to none -np 2 -x LD_LIBRARY_PATH -x NCCL_ALGO -x NCCL_PROTO -x NCCL_BUFFSIZE -x MSCCL_XML_FILES -x VERICCL_EXPECTED_MSCCL_CHUNKSTEPS -x VERICCL_EXPECTED_MSCCL_SLICESTEPS -x VERICCL_TRACE_ENABLE "$NCCL_TESTS_ROOT/build/all_reduce_perf" -b 8388608 -e 8388608 -w 5 -n 20 -c 1 -d float -o sum -g 1
```

Separate diagnostic run:

<!-- vericcl-msccl-run: single-node-trace -->
```bash
unset NCCL_DEBUG NCCL_DEBUG_SUBSYS
export LD_LIBRARY_PATH="$MSCCL_ROOT/build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NCCL_ALGO=MSCCL,RING
export NCCL_PROTO=Simple
export NCCL_BUFFSIZE=2097152
export MSCCL_XML_FILES=/absolute/path/to/one-schedule.xml
export VERICCL_EXPECTED_MSCCL_CHUNKSTEPS=4
export VERICCL_EXPECTED_MSCCL_SLICESTEPS=4
export VERICCL_TRACE_ENABLE=1
export VERICCL_TRACE_RECORDS=1048576
export VERICCL_TRACE_FILE_PREFIX=/absolute/path/to/vericcl-step
export VERICCL_MPI_LAUNCHER="$(command -v mpirun)"
"$VERICCL_MPI_LAUNCHER" --bind-to none -np 2 -x LD_LIBRARY_PATH -x NCCL_ALGO -x NCCL_PROTO -x NCCL_BUFFSIZE -x MSCCL_XML_FILES -x VERICCL_EXPECTED_MSCCL_CHUNKSTEPS -x VERICCL_EXPECTED_MSCCL_SLICESTEPS -x VERICCL_TRACE_ENABLE -x VERICCL_TRACE_RECORDS -x VERICCL_TRACE_FILE_PREFIX "$NCCL_TESTS_ROOT/build/all_reduce_perf" -b 8388608 -e 8388608 -w 0 -n 20 -c 0 -d float -o sum -g 1
```

For inter-node validation, `VERICCL_TRACE_FILE_PREFIX` must reside on storage mounted at the same absolute path on every node so the collector can read every per-rank file. Raising the clock-uncertainty threshold permits parsing but does not make uncertain endpoint orderings eligible for tuning.

### Multi-node XML execution

Use one MPI process per rank and one GPU per process. This eight-rank example uses four ranks on each of two hosts. The hostfile and XML must exist at the same paths on all nodes.

Run the INFO-level activation probe first. Both debug variables are propagated to every MPI rank, and the same exact-one check is applied to the combined log:

<!-- vericcl-msccl-run: multi-node-activation -->
```bash
export LD_LIBRARY_PATH="$MSCCL_ROOT/build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NCCL_ALGO=MSCCL,RING
export NCCL_PROTO=Simple
export NCCL_BUFFSIZE=2097152
export MSCCL_XML_FILES=/absolute/path/to/one-schedule.xml
export VERICCL_EXPECTED_MSCCL_CHUNKSTEPS=4
export VERICCL_EXPECTED_MSCCL_SLICESTEPS=4
export VERICCL_TRACE_ENABLE=0
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT
export VERICCL_MPI_HOSTFILE=/absolute/path/to/hosts
MSCCL_ACTIVATION_LOG="$(mktemp)"
mpirun -np 8 -N 4 --hostfile "$VERICCL_MPI_HOSTFILE" -x LD_LIBRARY_PATH -x NCCL_ALGO -x NCCL_PROTO -x NCCL_BUFFSIZE -x MSCCL_XML_FILES -x VERICCL_EXPECTED_MSCCL_CHUNKSTEPS -x VERICCL_EXPECTED_MSCCL_SLICESTEPS -x VERICCL_TRACE_ENABLE -x NCCL_DEBUG -x NCCL_DEBUG_SUBSYS "$NCCL_TESTS_ROOT/build/all_reduce_perf" -b 8388608 -e 8388608 -w 0 -n 1 -c 1 -d float -o sum -g 1 2>&1 | tee "$MSCCL_ACTIVATION_LOG"
test "$(sed -n 's/.*NCCL INFO Connected \([0-9][0-9]*\) MSCCL algorithms.*/\1/p' "$MSCCL_ACTIVATION_LOG" | sort -u)" = 1
rm -f "$MSCCL_ACTIVATION_LOG"
```

Run the formal multi-node release measurement only after the probe succeeds. Debug logging remains disabled and is not propagated:

<!-- vericcl-msccl-run: multi-node-release -->
```bash
unset NCCL_DEBUG NCCL_DEBUG_SUBSYS
export LD_LIBRARY_PATH="$MSCCL_ROOT/build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NCCL_ALGO=MSCCL,RING
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

## Extending VeriCCL

The internal modules are development entry points, not a stable plugin API. `vericcl/input` resolves and validates the topology, sketch, and atom inputs; `vericcl/topology` represents links, nodes, shared resources, and timing constraints; and `vericcl/planner` constructs staged collective plans.

`vericcl/solver` searches schedule candidates under those constraints, `vericcl/composer` composes staged operators into collective semantics, and `vericcl/xml` lowers accepted schedules to MSCCL XML and sidecars. `vericcl/verification` performs offline semantic, structural, and XML validation; `vericcl/tuning` repairs and revalidates candidate schedules; and `vericcl/verification/online` calibrates hardware and validates a runtime execution.

## Outputs, Limitations, and Troubleshooting

For the quickstart input and run ID `quickstart`, the directory is `vericcl_allreduce_8MiB_quickstart/`. It contains `resolved-input.json`, `run-summary.json`, `schedules/`, `reports/`, `traces/`, the final `.xml`, `.schedule.json`, and `.validation.json`. Offline-valid schedules that exceed MSCCL limits use `.candidate.xml` and must not be executed until re-solved.

Exit codes are: `0` for offline-valid completion (including runtime-warning candidates), `2` for input/configuration errors, `3` for no semantic-valid candidate or offline timeout, `4` for requested online failure/timeout, and `5` for an internal error.

Common diagnoses:

- CUDA/MSCCL build failure: recheck the driver/toolkit/host-compiler compatibility table, `CUDA_HOME`, GPU architecture, and the full compiler output.
- `mpi.h` or MPI link failure: confirm that the active Open MPI wrapper reports the intended include directory and derived `MPI_HOME`, then recheck the compiler and linker output.
- MSCCL verifier failure: use the exact upstream commit or fork tag, a clean Strategy A checkout, and the unmodified bundled header and patches.
- `Model too large...`: activate a full Gurobi license; reinstalling `gurobipy` does not expand the bundled limit.
- Missing online binary/library: check the three `VERICCL_*_BUILD_DIR`/binary paths and propagate `LD_LIBRARY_PATH` to every MPI rank.
- Runtime mismatch: apply both bundled patches, use Simple protocol, `NCCL_BUFFSIZE=2*slice_size_bytes`, step constants `4/4`, `cnt=1`, one XML, and matching NCCL Tests arguments.
- Trace/clock failure: keep release and trace runs separate, increase `VERICCL_TRACE_RECORDS` if required, and inspect clock uncertainty and per-rank trace files.

### Software tests and references

Run hardware-independent tests on any development host:

```bash
.venv/bin/python -m pytest tests/integration/test_documented_commands.py -q
.venv/bin/python -m pytest tests/unit/online/test_runtime_patch.py -q
git diff --check
```

CUDA compilation and GPU execution were not performed on the macOS documentation host. Run the marked server build and execution commands in the configured target GPU environment before reporting hardware validation.

Further references: [V100 K<=16 experiment workflow](docs/experiments/v100-k16.md), [runtime configuration](docs/runtime-configuration.md), [validation reports](docs/validation-report.md), [MSCCL trace patch](runtime/msccl-trace/README.md), [migration notes](MIGRATION.md), [NCCL Tests](https://github.com/NVIDIA/nccl-tests), and [official MSCCL](https://github.com/microsoft/msccl.git).

## License and Citation

License: To be determined.
Citation: To be determined.
