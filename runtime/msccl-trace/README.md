# VeriCCL MSCCL Step Trace Patches

The first patch replaces device-side per-step `printf` tracing with a
fixed-size device record buffer. The second aligns the host network proxy's
step signature with the device interpreter. Their official source is
`https://github.com/microsoft/msccl.git` at commit
`b23e9cd5dd63f82ee1c5aae7e0a2042079be903a`.

The build commands below target NVIDIA V100 (`compute_70`/`sm_70`). For
NVIDIA A100, replace that pair consistently with `compute_80`/`sm_80`.

## Strategy A: official source plus the bundled patches

<!-- vericcl-msccl-strategy: strategy-a -->
```bash
export VERICCL_ROOT="$(pwd)"
export MSCCL_SRC="${TMPDIR:-/tmp}/vericcl-msccl-base"
export NVCC_GENCODE="-gencode=arch=compute_70,code=sm_70"
git clone https://github.com/microsoft/msccl.git "$MSCCL_SRC"
git -C "$MSCCL_SRC" checkout --detach \
  b23e9cd5dd63f82ee1c5aae7e0a2042079be903a
python3 "$VERICCL_ROOT/runtime/msccl-trace/tools/verify_patch.py" \
  --source-root "$MSCCL_SRC" --base-tree
cp "$VERICCL_ROOT/runtime/msccl-trace/include/vericcl_trace_format.h" \
  "$MSCCL_SRC/src/include/vericcl_trace_format.h"
patch --directory="$MSCCL_SRC" --strip=1 \
  --input="$VERICCL_ROOT/runtime/msccl-trace/patches/0001-vericcl-fixed-step-trace.patch"
patch --directory="$MSCCL_SRC" --strip=1 \
  --input="$VERICCL_ROOT/runtime/msccl-trace/patches/0002-vericcl-host-step-signature.patch"
make -C "$MSCCL_SRC" clean
make -C "$MSCCL_SRC" -j NVCC_GENCODE="$NVCC_GENCODE" src.build
```

The verifier copies the required files to a temporary directory, dry-runs and
applies the patch there, and leaves the official checkout unchanged.

## Strategy C: pre-integrated VeriCCL-MSCCL source

The immutable public tag `vericcl-runtime-v0.1.0` resolves to commit
`782ee5f72cf48c1ae1a2365bcf525019f5620175`. The verifier checks that revision
and every file hash recorded in `upstream.json` before the build.

<!-- vericcl-msccl-strategy: strategy-c -->
```bash
export VERICCL_ROOT="$(pwd)"
export MSCCL_SRC="${TMPDIR:-/tmp}/vericcl-msccl-runtime"
export NVCC_GENCODE="-gencode=arch=compute_70,code=sm_70"
git clone --branch vericcl-runtime-v0.1.0 --depth 1 \
  https://github.com/SlienceZDL/VeriCCL-MSCCL.git "$MSCCL_SRC"
test "$(git -C "$MSCCL_SRC" rev-parse HEAD)" = \
  782ee5f72cf48c1ae1a2365bcf525019f5620175
python3 "$VERICCL_ROOT/runtime/msccl-trace/tools/verify_patch.py" \
  --source-root "$MSCCL_SRC" --patched-tree
patch --directory="$MSCCL_SRC" --strip=1 \
  --input="$VERICCL_ROOT/runtime/msccl-trace/patches/0002-vericcl-host-step-signature.patch"
make -C "$MSCCL_SRC" clean
make -C "$MSCCL_SRC" -j NVCC_GENCODE="$NVCC_GENCODE" src.build
test -d "$MSCCL_SRC/build/lib"
```

Strategy A verification proves the pinned base revision, both patches'
applicability, layout, host/device step-signature invariant, and recorded
hashes. Strategy C first proves the published trace-enabled revision, clean
tracked state, source invariants, and recorded hashes; it then applies the same
host-side supplement as Strategy A. Neither mode compiles CUDA sources or
provides evidence of a successful CUDA build or GPU execution.

## Trace controls

- `VERICCL_TRACE_ENABLE=1` enables fixed-buffer tracing. Missing or `0`
  disables all record reservations and writes.
- `VERICCL_TRACE_RECORDS=<count>` sets the record capacity per rank. The
  default is `1048576`.
- `VERICCL_TRACE_FILE_PREFIX=<path-prefix>` selects the output prefix. The
  default is `vericcl-trace`; rank `r` writes
  `<path-prefix>.rank-<r>.bin` during communicator teardown.
- `VERICCL_EXPECTED_MSCCL_CHUNKSTEPS=4` and
  `VERICCL_EXPECTED_MSCCL_SLICESTEPS=4` make communicator initialization
  reject a runtime compiled with a different step signature.

An overflowed buffer is still written, but its header has the overflow flag
set and later online analysis must reject it. Release performance measurements
must run with `VERICCL_TRACE_ENABLE=0`; trace diagnostics are a separate run
and their overhead is not performance data.

The raw `iteration` field stores the MSCCL `workIndex`, which identifies one
NCCL collective invocation. VeriCCL runs trace diagnostics with zero warmups,
20 timed iterations, and correctness checks disabled. The collector excludes
the setup invocation from each out-of-place or in-place timing block and sends
exactly 20 measured invocations to trace analysis.

The patches fix `MSCCL_CHUNKSTEPS=4` and `MSCCL_SLICESTEPS=4` on the device and
force the MSCCL host proxy to use the same values. The host-side override is
required for inter-node transfers; otherwise the public NCCL API supplies the
standard AllReduce slice signature (`4/2`) while the MSCCL interpreter consumes
`4/4`, which mismatches network proxy byte counts and credits. Use the runtime
parameter guidance in `Vericcl-work-document.md` to keep each XML step at the
intended software slice granularity.

## Clock synchronization helper

Build and run the MPI/CUDA helper before parsing cross-rank traces:

```bash
nvcc -ccbin mpicxx -O2 -std=c++11 \
  runtime/msccl-trace/tools/vericcl_clock_sync.cu \
  -o runtime/msccl-trace/tools/vericcl_clock_sync
mpirun -np "$NRANKS" runtime/msccl-trace/tools/vericcl_clock_sync 16 \
  > vericcl-clock-sync.txt
```

Each process emits host-bracketed GPU timer samples and an MPI-derived offset
to rank 0's monotonic clock. Python fitting retains both bracket and MPI
round-trip uncertainty; comparisons within the combined bound remain
unordered.
