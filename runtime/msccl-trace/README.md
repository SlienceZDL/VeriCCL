# VeriCCL MSCCL Step Trace Patch

This patch replaces device-side per-step `printf` tracing with a fixed-size
device record buffer. It targets the read-only reference tree at
`/Users/zdl/work/code/MSCCL_TIME`.

## Apply and build

```bash
export MSCCL_SRC=/path/to/msccl
cp runtime/msccl-trace/include/vericcl_trace_format.h \
  "$MSCCL_SRC/src/include/vericcl_trace_format.h"
patch --directory="$MSCCL_SRC" --strip=1 \
  --input="$PWD/runtime/msccl-trace/patches/0001-vericcl-fixed-step-trace.patch"
make -C "$MSCCL_SRC" clean
make -C "$MSCCL_SRC" -j src.build
```

Verify compatibility without modifying the source tree:

```bash
python3 runtime/msccl-trace/tools/verify_patch.py \
  --source-root /path/to/msccl
```

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

The patch fixes `MSCCL_CHUNKSTEPS=4` and `MSCCL_SLICESTEPS=4`. Use the runtime
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
