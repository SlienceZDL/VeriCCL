# VeriCCL runtime configuration

This document defines the runtime contract that makes one XML step represent one complete software slice. Network and GPU runtimes may still packetize a slice internally; that hardware behavior is outside VeriCCL's software-granularity contract.

Start with the [English installation guide](../README.md) or the [Chinese installation guide](../README.zh-CN.md). Both document [Strategy A, the pinned official MSCCL source plus the bundled patch](../README.md#msccl-strategy-a-official-source-plus-bundled-patch), and [Strategy C, the pre-integrated immutable tag](../README.md#msccl-strategy-c-pre-integrated-immutable-tag). Do not substitute a private or unpinned MSCCL tree.

## Fixed MSCCL build parameters

The verified runtime source fixes these definitions in `src/include/msccl.h`:

```c
#define MSCCL_CHUNKSTEPS 4
#define MSCCL_SLICESTEPS 4
```

Therefore `SlicePerChunk = MSCCL_CHUNKSTEPS / MSCCL_SLICESTEPS = 1`. The bundled patch also checks the expected values during communicator initialization. Both installation strategies build with:

```bash
make -C "$MSCCL_ROOT" -j src.build
```

The resulting runtime library directory is `$MSCCL_ROOT/build/lib`. Source verification proves the revision, patch applicability, invariants, and recorded hashes; it does not prove successful CUDA compilation or GPU execution.

## Per-execution environment

For a schedule with `slice_size_bytes=S`, the exact formula is:

```text
NCCL_BUFFSIZE=2*slice_size_bytes
```

For `S=1048576`, set `NCCL_BUFFSIZE=2097152` and load exactly one XML:

```bash
export NCCL_ALGO=MSCCL
export NCCL_PROTO=Simple
export NCCL_BUFFSIZE=2097152
export MSCCL_XML_FILES=/absolute/path/to/one-schedule.xml
export VERICCL_EXPECTED_MSCCL_CHUNKSTEPS=4
export VERICCL_EXPECTED_MSCCL_SLICESTEPS=4
export VERICCL_TRACE_ENABLE=0
```

Every XML step has `cnt=1`. Together, `MSCCL_CHUNKSTEPS 4`, `MSCCL_SLICESTEPS 4`, Simple protocol, `SlicePerChunk=1`, and `NCCL_BUFFSIZE=2*S` prevent MSCCL from splitting one XML step into multiple software slices. They do not constrain PCIe, NVLink, InfiniBand, Ethernet, or GPU hardware packetization.

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

The raw trace `iteration` field stores the MSCCL `workIndex`, which distinguishes repeated collective invocations. Trace diagnosis uses `-w 0 -n 20 -c 0`; the collector discards each timing block's setup invocation and analyzes exactly 20 measured invocations. `VERICCL_TRACE_RECORDS` is a user floor; preflight raises the effective capacity to at least `42 * max_steps_per_rank`.

Buffer overflow, missing `s` or `r/rrc` endpoints, the wrong invocation count, clock synchronization failure, or excessive clock uncertainty makes online operator validation fail. It does not invalidate offline collective semantics or XML syntax. Trace overhead must never be reported as release performance.

## Online environment

The CLI reads:

- `VERICCL_MSCCL_BUILD_DIR`: the selected `$MSCCL_ROOT/build/lib` directory.
- `VERICCL_NCCL_TESTS_BUILD_DIR`: the directory containing the six direct-collective NCCL Tests binaries.
- `VERICCL_CLOCK_SYNC_BINARY`: the compiled GPU/MPI clock helper.
- `VERICCL_ONLINE_INTER_NODE=0|1`: single-node or inter-node operator execution.
- `VERICCL_MPI_LAUNCHER` and `VERICCL_MPI_HOSTFILE`: required for inter-node execution.
- `VERICCL_MAX_CLOCK_UNCERTAINTY_US`: allowed clock uncertainty, default `10` microseconds.
- `VERICCL_CALIBRATION_LINK_CLASS=intra_node|inter_node`: representative link class to calibrate.
- `VERICCL_CALIBRATION_CACHE_PATH`: absolute path to persistent calibration JSON.
- `VERICCL_GPU_MODEL` and `VERICCL_NIC_MODEL`: hardware labels in the environment signature.
- `VERICCL_CUDA_VERSION`, `VERICCL_NCCL_VERSION`, and `VERICCL_MSCCL_VERSION`: actual software versions in the signature.
- `VERICCL_FORCE_RECALIBRATE=0|1`: ignore a matching cache when set to `1`; input `force_recalibrate=true` has the same effect.
- `VERICCL_TRACE_RECORDS`: per-rank trace-capacity floor, default `1048576`.

Single-node execution starts all local GPUs from one NCCL Tests process with `-g rank_count`. Inter-node execution starts one `-g 1` process per rank through MPI. Inter-node calibration always uses two MPI processes; intra-node calibration uses one process with `-g 2`. Calibration and operator launchers are independent.

Each release round starts 20 independent NCCL Tests processes. Every process performs five warmups and 20 timed iterations. A coefficient of variation above 5% triggers up to three rounds. Reports retain the median, P95, mean, standard deviation, coefficient of variation, and stability for every round rather than selecting a best single run.

## Calibration contract

Link calibration uses exactly 128 MiB and the current slice size. Intra-node calibration covers one host and two GPUs; inter-node calibration covers two hosts and one GPU each. It evaluates `k=1..min(max_calibration_channels,32,128MiB/S,link_max_channels)`. If `S` does not divide 128 MiB, calibration is `not_run` and never changes the slice size.

Each concurrency point receives a separate Broadcast XML. Complete-wave duration comes from per-step traces; an incomplete final wave executes but does not contribute to `D_safe(k)`.

The environment signature covers the link class, topology, GPU/NIC, CUDA/NCCL/MSCCL versions, Simple protocol, slice size, 128 MiB, concurrency, `NCCL_BUFFSIZE`, chunk/slice steps, GPU visibility, library paths, and input `NCCL_*`/`UCX_*` variables. Hostfiles and NCCL topology files include their content SHA-256. Only an exact signature match reuses cached measurements; cache writes use an inter-process lock, atomic replacement, and `fsync`.

After stable calibration, `solve --online` preserves each link or resource `alpha`, updates `invbw` and `B_link(k)` only for the exact isomorphic link class, caps channels at the largest measured concurrency, and solves again. Reports retain both candidate generations with `parent_candidate_id` links. Unmeasured link classes keep their conservative input values.

`verify --online` never silently rewrites the supplied XML. It validates that XML after calibration and sets `requires_resolve=true` when the new measurements should be consumed by a later `solve`.
