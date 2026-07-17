# VeriCCL Online Calibration and Step Tracing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 使用 nccl-tests 执行基础算子验证和 128 MiB 链路校准，并通过固定 GPU 事件缓冲区获得逐 step 时间线、等待分解和在线瓶颈报告。

**Architecture:** Python 在线模块负责命令构造、环境预检、统计、校准 XML、缓存、trace 解析和分析；MSCCL 修改以独立 patch 集保存在 VeriCCL 仓库。正式性能测试使用 release 运行时，trace 诊断另运行一次，不把 trace 开销计入性能结果。

**Tech Stack:** Python subprocess/statistics/json/struct、nccl-tests、MPI、CUDA/MSCCL patch、pytest；硬件测试独立标记。

## Global Constraints

- 继承前序阶段全部约束。
- 在线验证只运行当前 XML 对应的六类 nccl-tests 基础程序；每次只加载一个 XML。
- 正式统计为 5 次预热、20 次测量，中位数为主，同时记录 P95、均值、标准差和 CV；CV>5% 最多重试 3 轮。
- 链路校准只使用 1 机×2 卡和 2 机×1 卡、128 MiB、当前 slice 大小以及 `k=1..min(32,N_bench)` 的全部整数并发度。
- trace 必须使用固定 GPU 记录缓冲区，不使用逐 step device `printf`。
- 在线 trace 失败不否定离线有效 XML，但设置 `online_operator_validation=failed`，并禁止基于该数据在线调优。
- 本机无 CUDA GPU、nccl-tests 和 MPI 环境时，硬件任务必须报告 `not_run`。
- 本计划创建的每个测试模块声明 `pytestmark = pytest.mark.phase06`；硬件测试同时声明 `pytest.mark.hardware`。

---

### Task 1: nccl-tests Command Builder, Parser, and Statistics

**Files:**
- Create: `vericcl/verification/online/__init__.py`
- Create: `vericcl/verification/online/model.py`
- Create: `vericcl/verification/online/nccl_tests.py`
- Create: `vericcl/verification/online/statistics.py`
- Test: `tests/unit/online/test_nccl_tests.py`
- Test: `tests/unit/online/test_statistics.py`

**Interfaces:**
- Produces: `NcclTestRequest`, `NcclTestRun`, `PerformanceStatistics`
- Produces: `build_nccl_tests_command(request: NcclTestRequest) -> tuple[str, ...]`
- Produces: `parse_nccl_tests_output(text: str, expected_bytes: int) -> tuple[NcclTestRun, ...]`
- Produces: `summarize_runs(samples_us: Sequence[float]) -> PerformanceStatistics`

- [x] **Step 1: Write exact command tests**

```python
def test_allreduce_command_uses_exact_size_and_statistics_counts():
    command = build_nccl_tests_command(request(kind="allreduce", bytes=268435456, datatype="float", reduction_op="sum"))
    assert command[-14:] == (
        "-b", "268435456", "-e", "268435456", "-w", "5", "-n", "20",
        "-c", "1", "-d", "float", "-o", "sum",
    )


def test_broadcast_command_sets_exact_root():
    command = build_nccl_tests_command(request(kind="broadcast", bytes=134217728, root=0))
    assert command[-2:] == ("-r", "0")
```

- [x] **Step 2: Write parser and robust-statistics tests**

Use captured stdout fixtures with 20 samples. Assert median, nearest-rank P95, mean, population standard deviation, CV, wrong-count rejection, and unstable CV retry decision. Test both in-place and out-of-place columns when present.

- [x] **Step 3: Run tests and confirm missing online modules**

Run: `python3 -m pytest tests/unit/online/test_nccl_tests.py tests/unit/online/test_statistics.py -q`

Expected: collection fails.

- [x] **Step 4: Implement operator-to-binary mapping and validated arguments**

Map to `broadcast_perf`, `reduce_perf`, `all_gather_perf`, `all_reduce_perf`, `alltoall_perf`, and `reduce_scatter_perf`. Use exact `-b/-e`, `-w 5`, `-n 20`, `-c 1`, `-g 1`, datatype, reduction op, and root where relevant. Before execution, run `<binary> --help` once per binary and reject a requested option not supported by the installed nccl-tests version.

- [x] **Step 5: Implement deterministic statistics and retry policy**

Retain every raw sample and round. Retry when CV exceeds 0.05, at most three rounds total. After the last unstable round, return all rounds and `stable=False`; do not choose the best single run.

- [x] **Step 6: Run focused tests**

Run: `python3 -m pytest tests/unit/online/test_nccl_tests.py tests/unit/online/test_statistics.py -q`

Expected: all tests pass.

### Task 2: 128 MiB Calibration XMLs, Full-Wave Samples, and Exact Cache

**Files:**
- Create: `vericcl/verification/online/calibration_xml.py`
- Create: `vericcl/verification/online/calibration.py`
- Create: `vericcl/verification/online/cache.py`
- Test: `tests/unit/online/test_calibration_xml.py`
- Test: `tests/unit/online/test_calibration.py`
- Test: `tests/unit/online/test_calibration_cache.py`

**Interfaces:**
- Produces: `CalibrationRequest`, `CalibrationPoint`, `CalibrationResult`, `EnvironmentSignature`
- Produces: `build_calibration_artifacts(request: CalibrationRequest, topology: Topology) -> tuple[XmlArtifact, ...]`
- Produces: `derive_calibrated_curve(alpha_us: float, slice_size_bytes: int, points: Sequence[CalibrationPoint]) -> PerformanceCurve`

- [x] **Step 1: Write channel/wave allocation tests**

```python
def test_calibration_assigns_round_robin_channels_and_full_waves():
    artifact = build_calibration_artifact(slice_size=1 << 20, concurrency=3)
    assert channel_assignments(artifact)[:6] == [0, 1, 2, 0, 1, 2]
    assert full_wave_count(artifact) == (128 // 3)
    assert tail_transfer_count(artifact) == (128 % 3)
```

- [x] **Step 2: Write calibration formula and invalid-point tests**

Assert `invbw=D_safe(1)`, `beta=max(invbw-alpha, epsilon)`, `B_link(k)=k*S/max(D_safe(k)-alpha,epsilon)`, and invalidity when `D_safe(k)<=alpha`. Assert `S` not dividing 128 MiB skips calibration without changing S.

- [x] **Step 3: Write exact environment cache tests**

Cover link class, topology signature, GPU/NIC, CUDA/NCCL/MSCCL versions, Simple protocol, slice size, 128 MiB, k, NCCL_BUFFSIZE, chunk/slice steps, and relevant path variables. Any mismatch invalidates the cache; `force_recalibrate=True` always bypasses it.

- [x] **Step 4: Run tests and confirm missing calibration modules**

Run: `python3 -m pytest tests/unit/online/test_calibration_xml.py tests/unit/online/test_calibration.py tests/unit/online/test_calibration_cache.py -q`

Expected: collection fails.

- [x] **Step 5: Implement one custom Broadcast XML per integer k**

Use root 0, current S, `N_bench=128 MiB/S`, full slice transfers, `channel=l mod k`, and `wave=floor(l/k)`. Generate every k through `min(max_calibration_channels,32,N_bench)` without interpolation or early stop. Only full-wave trace intervals contribute to `D_safe(k)`; tail transfers still execute.

- [x] **Step 6: Run calibration tests**

Run: `python3 -m pytest tests/unit/online/test_calibration_xml.py tests/unit/online/test_calibration.py tests/unit/online/test_calibration_cache.py -q`

Expected: all tests pass.

### Task 3: MSCCL Fixed Event Buffer Patch Set

**Files:**
- Create: `runtime/msccl-trace/README.md`
- Create: `runtime/msccl-trace/patches/0001-vericcl-fixed-step-trace.patch`
- Create: `runtime/msccl-trace/include/vericcl_trace_format.h`
- Create: `runtime/msccl-trace/tools/verify_patch.py`
- Test: `tests/unit/online/test_runtime_patch.py`

**Reference Source:**
- Read-only baseline: `/Users/zdl/work/code/MSCCL_TIME/src/include/msccl.h`
- Read-only baseline: `/Users/zdl/work/code/MSCCL_TIME/src/collectives/device/primitives.h`
- Read-only baseline: `/Users/zdl/work/code/MSCCL_TIME/src/collectives/device/msccl_interpreter.h`
- Read-only baseline: `/Users/zdl/work/code/MSCCL_TIME/src/init.cc`
- Read-only baseline: `/Users/zdl/work/code/MSCCL_TIME/src/enqueue.cc`

**Interfaces:**
- Produces binary `VericclRawTraceHeader` and `VericclRawStepTraceRecord` format shared by C++ and Python
- Produces environment controls `VERICCL_TRACE_ENABLE`, `VERICCL_TRACE_RECORDS`, and `VERICCL_TRACE_FILE_PREFIX`

- [x] **Step 1: Write a patch contract test**

The test parses the patch and asserts it removes per-step `MSCCLTRACE` printf use, sets literal `MSCCL_SLICESTEPS 4` and `MSCCL_CHUNKSTEPS 4`, adds fixed record/counter/overflow fields, records the four timestamps, checks buffer bounds atomically, and frees trace allocations on communicator teardown.

- [x] **Step 2: Define a stable raw binary record**

```c
typedef struct {
  uint32_t rank;
  uint16_t tb_id;
  uint16_t step_index;
  uint16_t endpoint_type;
  int16_t peer;
  uint16_t channel;
  uint32_t iteration;
  uint64_t tb_reach;
  uint64_t dependency_done;
  uint64_t transfer_start;
  uint64_t transfer_end;
  uint32_t flags;
  uint32_t reserved;
} VericclRawStepTraceRecord;
```

The final Python StepTraceRecord obtains `transfer_id` and full semantic metadata from the XML sidecar keyed by rank/TB/step; the raw runtime format does not duplicate long string IDs.

- [x] **Step 3: Build the patch against the reference interpreter locations**

Patch `msccl.h` to add trace buffers to host/device comm state and literal chunk/slice steps. Patch `init.cc` to allocate the fixed device record buffer/counter when enabled and free it on destroy. Patch `msccl_interpreter.h` so thread 0 records TB reach before dependency polling, dependency completion after its barrier, primitive start immediately before the operation, and end after completion. Patch host teardown or an explicit flush path to copy records and header to `<prefix>.rank-<rank>.bin`. Remove aggregate printf timing from the trace build.

- [x] **Step 4: Add overflow and release-build behavior**

An atomic index reserves each record; indexes beyond capacity only set the overflow flag. When tracing is disabled, pointers are null and the interpreter executes no record writes. Patch documentation includes exact MSCCL rebuild commands and states that release performance measurements must use tracing disabled.

- [x] **Step 5: Verify patch application without modifying the reference tree**

`verify_patch.py` copies only required baseline files to a temporary directory, applies the patch with `git apply --check` or `patch --dry-run`, verifies the shared struct layout, and scans patched source for per-step trace printf.

Run: `python3 -m pytest tests/unit/online/test_runtime_patch.py -q`

Expected: all patch contract and dry-run tests pass.

### Task 4: Trace Sidecar, Clock Alignment, Pairing, and Wait Decomposition

**Files:**
- Create: `vericcl/xml/trace_sidecar.py`
- Create: `vericcl/verification/online/trace_format.py`
- Create: `vericcl/verification/online/clock_sync.py`
- Create: `vericcl/verification/online/trace_analysis.py`
- Create: `runtime/msccl-trace/tools/vericcl_clock_sync.cu`
- Test: `tests/unit/online/test_trace_format.py`
- Test: `tests/unit/online/test_clock_sync.py`
- Test: `tests/unit/online/test_trace_analysis.py`

**Interfaces:**
- Produces: `TraceSidecar`, `RawStepTraceRecord`, `StepTraceRecord`, `ClockTransform`, `TraceAnalysis`
- Produces: `parse_trace(path: Path, sidecar: TraceSidecar) -> tuple[StepTraceRecord, ...]`
- Produces: `align_clocks(records_by_rank: Mapping[int, Sequence[StepTraceRecord]], samples: Sequence[ClockSyncSample]) -> ClockAlignment`
- Produces: `analyze_trace(records: Sequence[StepTraceRecord], sidecar: TraceSidecar, alignment: ClockAlignment) -> TraceAnalysis`

- [x] **Step 1: Write sidecar and binary round-trip tests**

Assert `(rank,tb_id,step_index)` maps to transfer ID, endpoint type, atom IDs, flow ID, lane, and full semantic predecessor IDs. Assert truncated records, bad magic/version, missing sidecar entries, and overflow flags invalidate the trace.

- [x] **Step 2: Write endpoint pairing and wait-formula tests**

```python
def test_physical_interval_uses_both_endpoints():
    pair = pair_endpoints(send(start=10, end=20), recv(start=12, end=25))
    assert pair.physical_start == 12
    assert pair.physical_end == 25


def test_wait_decomposition_uses_semantic_predecessors():
    result = analyze_step(step(tb_reach=15, dependency_done=19, physical_start=24, physical_end=34), semantic_ready=10)
    assert result.head_of_line_wait == 5
    assert result.dependency_wait == 4
    assert result.peer_resource_wait == 5
    assert result.transfer_duration == 10
```

- [x] **Step 3: Write clock uncertainty tests**

Use multi-point CPU/GPU samples to fit an affine transform per rank. Assert cross-rank comparisons whose difference is below the combined uncertainty are marked unordered, not forced into a false sequence.

- [x] **Step 4: Run tests and confirm missing trace modules**

Run: `python3 -m pytest tests/unit/online/test_trace_format.py tests/unit/online/test_clock_sync.py tests/unit/online/test_trace_analysis.py -q`

Expected: collection fails.

- [x] **Step 5: Implement semantic-ready reconstruction and pair analysis**

Compute semantic readiness as the maximum physical end of every sidecar predecessor, not only XML `depid/deps`. Pair send and receive/rrc by transfer ID, require both endpoints, and compute physical start/end with max. Associate every bottleneck with transfer, atom, flow, rank, TB, step, lane, and wait class.

- [x] **Step 6: Implement clock-sync helper and uncertainty propagation**

The CUDA helper records multiple host-before/GPU-timer/host-after samples per process and exchanges host reference samples across MPI ranks. Python fits per-rank transforms, records residual and round-trip uncertainty, and refuses ordering decisions below that bound.

- [x] **Step 7: Run trace tests**

Run: `python3 -m pytest tests/unit/online/test_trace_format.py tests/unit/online/test_clock_sync.py tests/unit/online/test_trace_analysis.py -q`

Expected: all tests pass.

### Task 5: Online Validation Pipeline and Hardware Gates

**Files:**
- Create: `vericcl/verification/online/runner.py`
- Create: `vericcl/verification/online/pipeline.py`
- Test: `tests/unit/online/test_pipeline.py`
- Test: `tests/hardware/test_intra_node_calibration.py`
- Test: `tests/hardware/test_inter_node_calibration.py`
- Test: `tests/hardware/test_six_collectives.py`

**Interfaces:**
- Produces: `run_online_validation(context: OnlineContext) -> OnlineValidationResult`

- [ ] **Step 1: Write mocked pipeline tests**

Cover runtime incompatibility blocking launch, missing nccl-tests binary, trace failure after successful release run, unstable calibration blocking online tuning, cache reuse, force recalibration, and successful release/trace separation.

- [ ] **Step 2: Run unit tests and confirm missing pipeline**

Run: `python3 -m pytest tests/unit/online/test_pipeline.py -q`

Expected: collection fails.

- [ ] **Step 3: Implement strict preflight and environment propagation**

Require runtime-compatible XML, exact message size/type/op/root/inplace, one XML path, `NCCL_ALGO=MSCCL`, `NCCL_BUFFSIZE=2*S`, expected chunk/slice-step signature, MSCCL library path, nccl-tests binary, and MPI launcher for inter-node runs. Pass all variables to every process.

- [ ] **Step 4: Implement release measurement, optional calibration, and trace diagnostic**

Run cached or requested link calibration first, update only beta/invbw/B_link, and request a new solve rather than mutating an existing schedule. Run release nccl-tests statistics. Then run one trace-enabled diagnostic of the same XML and parameters, parse all rank files, and produce the online bottleneck report. Trace data never enters release performance samples. When online tuning is requested and trace is valid, pass the measured waits and bottleneck priorities into the existing tuning context; never create online tuning candidates from an incomplete or uncertain trace.

- [ ] **Step 5: Implement hardware markers and not-run reporting**

Hardware tests read explicit environment variables for MSCCL build, nccl-tests build, GPU count, hostfile, and MPI launcher. Missing prerequisites call `pytest.skip()` with a precise reason; the artifact report maps skips to `not_run`.

- [ ] **Step 6: Run Phase 06 pure-software tests**

Run: `python3 -m pytest -m 'phase06 and not hardware' --cov=vericcl.verification.online --cov-report=term-missing -q`

Expected: all pure-software Phase 06 tests pass with at least 90% coverage.

Run: `python3 -m pytest -m hardware -q`

Expected: pass on configured GPU hosts; otherwise every test is explicitly skipped and reported `not_run`.

Run: `rg -n '[\p{Han}]' vericcl/verification/online runtime/msccl-trace tests/unit/online -g '*.{py,c,cc,cu,cuh,h,json}'`

Expected: no output.
