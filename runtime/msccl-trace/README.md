# MSCCL measurement and trace runtime

The runtime uses the repository's pinned source and nine patches. In order, the patches provide step-level tracing, host/device step consistency, lightweight measurement, single-invocation trace control, explicit versions, preparation-time admission, dispatch for six collectives, bounded abort, and repeated trace cycles. For regular builds, use `runtime/build_runtime.py` as described in the root README. Source files, patches, and final files are all checked by hash.

To reproduce the build from upstream source, run the following commands from the project root, using the CUDA path and GPU architecture for your environment:

<!-- vericcl-msccl-strategy: strategy-a -->
```bash
export VERICCL_ROOT="$(pwd -P)"
export MSCCL_SRC="${TMPDIR:-$VERICCL_ROOT/examples/runtime-build/output}/vericcl-msccl-base"
git clone https://github.com/microsoft/msccl.git "$MSCCL_SRC"
git -C "$MSCCL_SRC" checkout --detach b23e9cd5dd63f82ee1c5aae7e0a2042079be903a
python3 "$VERICCL_ROOT/runtime/msccl-trace/tools/verify_patch.py" --source-root "$MSCCL_SRC" --base-tree
cp "$VERICCL_ROOT/runtime/msccl-trace/include/vericcl_trace_format.h" "$MSCCL_SRC/src/include/"
cp "$VERICCL_ROOT/runtime/include/vericcl_probe.h" "$MSCCL_SRC/src/include/"
cp "$VERICCL_ROOT/runtime/include/vericcl_plan.h" "$MSCCL_SRC/src/include/"
for VERICCL_PATCH in "$VERICCL_ROOT"/runtime/msccl-trace/patches/*.patch; do
  patch --directory="$MSCCL_SRC" --strip=1 --input="$VERICCL_PATCH"
done
make -C "$MSCCL_SRC" -j8 CUDA_HOME=/usr/local/cuda-12.8 \
  NVCC_GENCODE="-gencode=arch=compute_70,code=sm_70" src.build
```

## Capture one complete invocation

Lightweight cost measurement and detailed tracing use different modes. Detailed tracing is disabled by default and remains disabled for formal paired performance measurements.

Set the following before creating the communicator:

- `VERICCL_TRACE_ENABLE=1`: allocate trace buffers. Tracing is disabled when omitted or set to `0`.
- `VERICCL_TRACE_RECORDS`: record capacity per rank. Each record is 64 bytes, and the file header is 40 bytes. Estimate capacity from the XML steps that the rank executes; NOPs do not produce records.
- `VERICCL_TRACE_FILE_PREFIX`: full output path prefix. Rank `r` writes to `<prefix>.rank-<r>.bin`.
- `VERICCL_EXPECTED_MSCCL_CHUNKSTEPS=4` and `VERICCL_EXPECTED_MSCCL_SLICESTEPS=4`: check the fixed step configuration when creating the communicator.

The caller includes `runtime/include/vericcl_trace_control.h`, keeps the communicator's CUDA device current, and avoids concurrent work submission. Use this sequence:

1. After creating the communicator, call `vericclTracePause(comm)`, then run warmup.
2. Collect clock reference samples before the target invocation, then call `vericclTraceReset(comm, expected_records)`. This waits for GPU completion, checks capacity, and clears the count and overflow flag.
3. Execute the target collective once and wait for completion.
4. Call `vericclTraceFinish(comm)` to save records and disable further recording, then collect clock reference samples after the target invocation.
5. Check data correctness, GPU completion, and cleanup receipts for every rank before parsing the files. The raw `iteration` field stores MSCCL's `workIndex` to identify the captured invocation.

`runtime/probes/main.cu --capture-trace 1 --timed 1` implements this sequence for development checks of the standalone link measurement program. It also saves the device's full invocation time measured by CUDA events. This time includes tracing overhead; performance acceptance checks use a separate run with tracing disabled. Repeated diagnostic capture in the legacy `runner.py` remains for compatibility. The v5 single-invocation interface is `capture_once`.

Overflow or missing steps fail the completeness check. The parser retains other individually verifiable records and marks missing locations, so these fragments can support local analysis. Bottleneck conclusions for a complete schedule use records with full coverage and determinable causal relationships.

## Clocks and total completion time

The v5 single-invocation capture uses the nanosecond `globaltimer` on SM 7.0 and newer devices. The capture program reads GPU time before and after the target invocation, then maps it to rank 0's monotonic clock using host time intervals and MPI round-trip measurements. This measures clock relationships while leaving system clocks unchanged.

The parser requires reference samples to bracket the target invocation and retains mapping errors. Step durations within a device are computed directly from that device's time differences. Cross-device order is determined when error intervals do not overlap; otherwise, it is marked uncertain. The first and last steps of a complete trace define a global time interval with error bounds. Each rank's CUDA event time is recorded separately to preserve the meaning of both measurements.

`runtime/msccl-trace/tools/vericcl_clock_sync.cu` is an earlier standalone reference tool. v5 uses the before-and-after reference measurements in `runtime/probes/clock_reference.cuh`, avoiding direct extrapolation from an older sample set outside the target interval.
