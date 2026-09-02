# VeriCCL runtime configuration

This document defines the runtime contract that makes one XML step represent one complete software slice. Network and GPU runtimes may still packetize a slice internally; that hardware behavior is outside VeriCCL's software-granularity contract.

Start with the [English installation guide](../README.md) or the [Chinese installation guide](../README.zh-CN.md). Both document [Strategy A, the pinned official MSCCL source plus the bundled patches](../README.md#strategy-a-official-source-plus-bundled-patches), and [Strategy C, the pre-integrated immutable tag](../README.md#strategy-c-pre-integrated-immutable-tag). Do not substitute a private or unpinned MSCCL tree.

## Fixed MSCCL build parameters

The verified runtime source fixes these definitions in `src/include/msccl.h`:

```c
#define MSCCL_CHUNKSTEPS 4
#define MSCCL_SLICESTEPS 4
```

Therefore `SlicePerChunk = MSCCL_CHUNKSTEPS / MSCCL_SLICESTEPS = 1`. The first bundled patch also checks the expected values during communicator initialization. The second patch forces the MSCCL host proxy to use the same `4/4` signature. This is required for network transfers because the standard AllReduce API otherwise provides `4/2` host-side steps while the device interpreter uses `4/4`, producing inconsistent proxy byte counts and credits. Both installation strategies apply both patches before building with:

```bash
export NVCC_GENCODE="-gencode=arch=compute_70,code=sm_70"
make -C "$MSCCL_ROOT" -j NVCC_GENCODE="$NVCC_GENCODE" src.build
```

This example targets NVIDIA V100 (`compute_70`/`sm_70`). For NVIDIA A100, use `compute_80`/`sm_80` consistently for the MSCCL build and the clock-helper build.

The resulting runtime library directory is `$MSCCL_ROOT/build/lib`. Source verification proves the revision, patch applicability, invariants, and recorded hashes; it does not prove successful CUDA compilation or GPU execution.

## Per-execution environment

For a schedule with `slice_size_bytes=S`, the exact formula is:

```text
NCCL_BUFFSIZE=2*slice_size_bytes
```

For `S=1048576`, set `NCCL_BUFFSIZE=2097152` and load exactly one XML:

```bash
export NCCL_ALGO=MSCCL,RING
export NCCL_PROTO=Simple
export NCCL_BUFFSIZE=2097152
export MSCCL_XML_FILES=/absolute/path/to/one-schedule.xml
export VERICCL_EXPECTED_MSCCL_CHUNKSTEPS=4
export VERICCL_EXPECTED_MSCCL_SLICESTEPS=4
export VERICCL_TRACE_ENABLE=0
```

Every XML step has `cnt=1`. Together, the device and host-side `MSCCL_CHUNKSTEPS 4`/`MSCCL_SLICESTEPS 4` contract, Simple protocol, `SlicePerChunk=1`, and `NCCL_BUFFSIZE=2*S` prevent MSCCL from splitting one XML step into multiple software slices. `NCCL_ALGO=MSCCL,RING` is required because NCCL Tests always evaluates both placement modes: the XML-matching mode uses MSCCL and the other mode falls back to Ring. These settings do not constrain PCIe, NVLink, InfiniBand, Ethernet, or GPU hardware packetization.

## Exact message range

XML uses Simple protocol and an exact half-open `[minBytes,maxBytes)` range. For all direct collectives except AllGather, `minBytes=total_size_bytes` and `maxBytes=minBytes+1`. For AllGather, the MSCCL interface requires `minBytes=rank_count*total_size_bytes`. NCCL Tests must use the one matching message size, datatype, reduction operation, root, rank count, and in-place mode.

An offline-valid XML may exceed MSCCL limits for thread-block steps, thread blocks, channels, dependency thread-block IDs, or offsets. Such output uses `.candidate.xml`, includes `runtime_recommendations`, and must not be executed. Re-solve with the recommended minimum channel count or a divisible larger slice size.

## Release and trace runs

Release performance measurement must disable tracing:

```bash
export VERICCL_TRACE_ENABLE=0
```

Run step diagnosis separately:

```bash
export VERICCL_TRACE_ENABLE=1
export VERICCL_TRACE_RECORDS=1048576
export VERICCL_TRACE_FILE_PREFIX=/absolute/path/to/vericcl-step
```

For inter-node validation, every host uses the same absolute trace prefix, but
the path does not need shared storage. Each rank writes
`<prefix>.rank-<rank>.bin` on its local host. With the split-host V100 executor,
node4 owns the first half of the ranks and node2 owns the second half. The
remote collector copies node4 files to node2 before parsing all rank files.

The raw trace `iteration` field stores the MSCCL `workIndex`, which distinguishes repeated collective invocations. Trace diagnosis uses `-w 0 -n 20 -c 0`; the collector discards each timing block's setup invocation and analyzes exactly 20 measured invocations. `VERICCL_TRACE_RECORDS` is a user floor; preflight raises the effective capacity to at least `42 * max_steps_per_rank`.

Buffer overflow, missing `s` or `r/rrc` endpoints, the wrong invocation count, clock synchronization failure, or excessive clock uncertainty makes online operator validation fail. It does not invalidate offline collective semantics or XML syntax. Trace overhead must never be reported as release performance.

## Online environment

The CLI reads:

- `VERICCL_MSCCL_BUILD_DIR`: the selected `$MSCCL_ROOT/build/lib` directory.
- `VERICCL_NCCL_TESTS_BUILD_DIR`: the directory containing the six direct-collective NCCL Tests binaries.
- `VERICCL_CLOCK_SYNC_BINARY`: the compiled GPU/MPI clock helper.
- The online validator requests 256 samples per rank by default. The helper uses
  the same count for GPU timer fitting and MPI reference-clock estimation to
  reduce transient cross-node round-trip jitter without changing host clocks.
- `VERICCL_ONLINE_INTER_NODE=0|1`: single-node or inter-node operator execution.
- `VERICCL_MPI_LAUNCHER`: required for all online execution; VeriCCL launches one MPI process per GPU.
- `VERICCL_MPI_HOSTFILE`: additionally required for inter-node execution.
- `VERICCL_MAX_CLOCK_UNCERTAINTY_US`: allowed clock uncertainty, default `10` microseconds.
- `VERICCL_CALIBRATION_LINK_CLASS=intra_node|inter_node`: representative link class to calibrate.
- `VERICCL_CALIBRATION_CACHE_PATH`: absolute path to persistent calibration JSON.
- `VERICCL_GPU_MODEL` and `VERICCL_NIC_MODEL`: hardware labels in the environment signature.
- `VERICCL_CUDA_VERSION`, `VERICCL_NCCL_VERSION`, and `VERICCL_MSCCL_VERSION`: actual software versions in the signature.
- `VERICCL_FORCE_RECALIBRATE=0|1`: ignore a matching cache when set to `1`; input `force_recalibrate=true` has the same effect.
- `VERICCL_TRACE_RECORDS`: per-rank trace-capacity floor, default `1048576`.

Single-node and inter-node execution both start one MPI process per rank and use `-g 1`. Inter-node execution additionally uses the configured hostfile. Intra-node and inter-node calibration follow the same one-process-per-GPU rule. Calibration and operator launchers remain logically independent.

Increasing `VERICCL_MAX_CLOCK_UNCERTAINTY_US` only permits trace collection in
a noisier environment. It does not make endpoint orderings reliable:
comparisons within the combined uncertainty remain unordered, and the analysis
can set `tuning_eligible=false`. Prefer an existing lower-latency MPI path when
available; do not report uncertain comparisons as tuning evidence.

Each release round starts 20 independent NCCL Tests processes. Every process performs five warmups and 20 timed iterations. A coefficient of variation above 5% triggers up to three rounds. Reports retain the median, P95, mean, standard deviation, coefficient of variation, and stability for every round rather than selecting a best single run.

Calibration executes concurrency points in order. Before each uncached point, VeriCCL divides the remaining wall-clock budget across that point and the remaining points, so time saved by earlier measurements rolls forward without extending the outer verification deadline.

Intra-node calibration launches two MPI processes on one node. Inter-node calibration adds `-N 1` to `-np 2` with the configured hostfile, placing exactly one process and one GPU on each of two nodes. Global operator validation uses the global rank count and the hostfile slot distribution instead.

Use the documented 10800-second workflow budget for a first uncached 16-point calibration. A point contains 20 independent release processes and one trace process, so a 1800-second budget may expire without indicating an XML or hardware failure.

The cache reuses only stable points with an exact environment-signature match. Unstable points remain in the cache file for audit but are automatically measured again on the next run.

## Calibration contract

Link calibration uses exactly 128 MiB and the current slice size. Intra-node calibration covers one host and two GPUs; inter-node calibration covers two hosts and one GPU each. The limit is `K_effective=min(16,max_calibration_channels,128MiB/S,link_max_channels)`, and calibration evaluates every `k=1..K_effective`. If `S` does not divide 128 MiB, calibration is `not_run` and never changes the slice size.

Each concurrency point receives a separate Broadcast XML. Complete-wave duration comes from per-step traces; an incomplete final wave executes but does not contribute to `D_safe(k)`.

The environment signature covers the link class, topology, GPU/NIC, CUDA/NCCL/MSCCL versions, Simple protocol, slice size, 128 MiB, concurrency, `NCCL_BUFFSIZE`, chunk/slice steps, GPU visibility, library paths, and input `NCCL_*`/`UCX_*` variables. Hostfiles and NCCL topology files include their content SHA-256. Only an exact signature match reuses cached measurements; cache writes use an inter-process lock, atomic replacement, and `fsync`.

After stable calibration, `solve --online` preserves each link or resource `alpha`, updates `invbw` and `B_link(k)` only for the exact isomorphic link class, caps channels at the largest measured concurrency, and solves again. Reports retain both candidate generations with `parent_candidate_id` links. Unmeasured link classes keep their conservative input values.

`verify --online` never silently rewrites the supplied XML. It validates that XML after calibration and sets `requires_resolve=true` when the new measurements should be consumed by a later `solve`.
