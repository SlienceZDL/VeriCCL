# VeriCCL Buffer Planning and MSCCL XML Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将缓冲区无关的全局 Schedule 安全降低为显式 BufferPlan、成对 EndpointAtom、单向 TB 和可验证的 MSCCL Simple XML。

**Architecture:** XML 模块分四层：逻辑值活跃期与物理地址分配、物理传输端点化、同步 TB 列表调度、XML 序列化与执行兼容性检查。每层输出不可变中间表示，并在进入下一层前独立验证。

**Tech Stack:** Python dataclasses、lxml、heapq、pytest、XML golden files。

## Global Constraints

- 继承前序阶段全部约束。
- 求解器不接触 `i/o/s` 缓冲区；BufferPlan 只能在全局调度合成后生成。
- 非原地输入不得被修改；原地别名严格按具体算子契约建立。
- 每个物理 Transfer 产生且只产生一对端点；SEND 为 `s/r`，REDUCE 为 `s/rrc`。
- 通信 TB 严格单向；local `cpy` 使用 `send=-1, recv=-1` 的独立 TB。
- XML 不使用顶层 `<copy>`、`rcs`、`rrs`、`rrcs` 或 `re`，不生成 `cnt>1`。
- 本计划创建的每个测试模块声明 `pytestmark = pytest.mark.phase04`。

---

### Task 1: Value Keys, Buffer Address Mapping, and Liveness

**Files:**
- Create: `vericcl/xml/__init__.py`
- Create: `vericcl/xml/model.py`
- Create: `vericcl/xml/buffers.py`
- Create: `vericcl/xml/liveness.py`
- Test: `tests/unit/xml/test_buffer_offsets.py`
- Test: `tests/unit/xml/test_buffer_liveness.py`
- Test: `tests/property/test_buffer_aliasing.py`

**Interfaces:**
- Produces: `RawValue`, `AggregateValue`, `PhysicalRef`, `LocalCopy`, `BufferPlan`
- Produces: `build_buffer_plan(schedule: Schedule, inputs: ResolvedInput) -> BufferPlan`
- Produces: `verify_buffer_liveness(schedule: Schedule, plan: BufferPlan, inputs: ResolvedInput) -> None`

- [x] **Step 1: Write exact six-operator offset tests**

```python
@pytest.mark.parametrize(
    "kind,rank,source,logical,expected",
    [
        (CollectiveKind.BROADCAST, 1, 0, 1, ("o", 1)),
        (CollectiveKind.REDUCE, 0, 1, 1, ("o", 1)),
        (CollectiveKind.ALL_GATHER, 1, 1, 1, ("o", 3)),
        (CollectiveKind.ALL_REDUCE, 1, 1, 1, ("o", 1)),
        (CollectiveKind.ALL_TO_ALL, 1, 0, 1, ("o", 0)),
        (CollectiveKind.REDUCE_SCATTER, 1, 1, 1, ("o", 0)),
    ],
)
def test_final_offsets(kind, rank, source, logical, expected):
    plan = build_buffer_plan(final_schedule(kind, ranks=2, slices=2), resolved(kind))
    assert plan.final_ref(rank, source, logical).buffer_offset == expected
```

- [x] **Step 2: Write in-place and out-of-place hazard tests**

Test AllReduce `i[l]`/`o[l]` alias, AllGather `i[l]`/`o[r*N+l]` alias, Reduce root-only alias, ReduceScatter owner alias, AllToAll scratch preservation on overwrite, and non-in-place reduction input preservation through explicit `cpy`.

- [x] **Step 3: Run tests and confirm missing XML models**

Run: `python3 -m pytest tests/unit/xml/test_buffer_offsets.py tests/unit/xml/test_buffer_liveness.py tests/property/test_buffer_aliasing.py -q`

Expected: collection fails.

- [x] **Step 4: Implement value identities and deterministic interval allocation**

```python
@dataclass(frozen=True, order=True)
class RawValue:
    slice_id: int


@dataclass(frozen=True, order=True)
class AggregateValue:
    logical_slice_index: int
    contributors: frozenset[int]
    state_version: int


@dataclass(frozen=True)
class PhysicalRef:
    rank: int
    buffer: str
    offset: int
    valid_from: float
    valid_until: float

    @property
    def buffer_offset(self) -> tuple[str, int]:
        return self.buffer, self.offset


@dataclass(frozen=True)
class LocalCopy:
    copy_id: str
    rank: int
    src_ref: PhysicalRef
    dst_ref: PhysicalRef
    predecessor_state_id: str
    st_time: float
    ed_time: float
    reason: str


@dataclass(frozen=True)
class BufferPlan:
    value_locations: Mapping[Union[RawValue, AggregateValue], tuple[PhysicalRef, ...]]
    aliases: tuple[tuple[PhysicalRef, PhysicalRef], ...]
    local_copies: tuple[LocalCopy, ...]
    i_chunks: Mapping[int, int]
    o_chunks: Mapping[int, int]
    s_chunks: Mapping[int, int]
```

Allocate final output addresses first, then required input aliases, then scratch by first-fit over sorted live intervals. Two active ValueKeys may share a PhysicalRef only when their intervals do not overlap. Each LocalCopy records source, destination, predecessor state, and an English reason.

- [x] **Step 5: Implement mandatory liveness checks**

Verify every read follows initialization or write, every `rrc` destination accumulator exists before reduction, live values do not collide, in-place data is not overwritten before its final send, out-of-place input remains unchanged, final addresses match CollectiveSpec, and all offsets fit declared chunk counts.

- [x] **Step 6: Run buffer tests**

Run: `python3 -m pytest tests/unit/xml/test_buffer_offsets.py tests/unit/xml/test_buffer_liveness.py tests/property/test_buffer_aliasing.py -q`

Expected: all tests pass.

### Task 2: EndpointAtom Lowering and Physical Dependency Graph

**Files:**
- Create: `vericcl/xml/endpoints.py`
- Create: `vericcl/xml/dependencies.py`
- Test: `tests/unit/xml/test_endpoints.py`
- Test: `tests/unit/xml/test_dependencies.py`

**Interfaces:**
- Produces: `EndpointAtom`, `EndpointType`, `TransferNode`, `EndpointProgram`
- Produces: `lower_endpoints(schedule: Schedule, buffers: BufferPlan) -> EndpointProgram`
- Produces: `build_transfer_dag(program: EndpointProgram, schedule: Schedule, buffers: BufferPlan) -> TransferDAG`

- [x] **Step 1: Write endpoint pairing tests**

```python
def test_send_creates_s_and_r_pair():
    program = lower_endpoints(one_send_schedule(), matching_buffers())
    endpoints = program.by_transfer_id["tx-0"]
    assert {endpoint.xml_type for endpoint in endpoints} == {EndpointType.SEND, EndpointType.RECV}


def test_reduce_creates_s_and_rrc_pair():
    program = lower_endpoints(one_reduce_schedule(), matching_buffers())
    endpoints = program.by_transfer_id["tx-0"]
    assert {endpoint.xml_type for endpoint in endpoints} == {EndpointType.SEND, EndpointType.RECV_REDUCE_COPY}
```

- [x] **Step 2: Write relay and multi-contributor dependency tests**

Assert a relay has a receive TransferNode followed by a distinct send TransferNode. Assert a consumer of three reduce contributors has three semantic predecessors before XML single-dependency lowering.

- [x] **Step 3: Run tests and verify missing endpoint modules**

Run: `python3 -m pytest tests/unit/xml/test_endpoints.py tests/unit/xml/test_dependencies.py -q`

Expected: collection fails.

- [x] **Step 4: Implement endpoint records and exact pairs**

```python
@dataclass(frozen=True)
class EndpointAtom:
    atom: Atom
    transfer_id: str
    xml_type: EndpointType
    rank: int
    peer: int
    channel: int
    src_ref: Optional[PhysicalRef]
    dst_ref: Optional[PhysicalRef]
```

Pair validation requires exactly two endpoints, opposite ranks, identical channel/time interval, compatible types, and identical transfer/member metadata. Capacity accounting remains on TransferNode, never on endpoints.

- [x] **Step 5: Build the complete pre-lowering dependency DAG**

Include path dependencies, AggregateState joins, buffer initialization/copies, final copy dependencies, and all semantic predecessors. Reject missing IDs, cycles, or a dependency that crosses state versions incorrectly.

- [x] **Step 6: Run endpoint tests**

Run: `python3 -m pytest tests/unit/xml/test_endpoints.py tests/unit/xml/test_dependencies.py -q`

Expected: all tests pass.

### Task 3: Synchronized TB List Scheduling, NOP Joins, and Deadlock Detection

**Files:**
- Create: `vericcl/xml/threadblocks.py`
- Create: `vericcl/xml/list_scheduler.py`
- Create: `vericcl/xml/deadlock.py`
- Test: `tests/unit/xml/test_threadblocks.py`
- Test: `tests/unit/xml/test_list_scheduler.py`
- Test: `tests/unit/xml/test_deadlock.py`

**Interfaces:**
- Produces: `ThreadblockKey`, `XmlStep`, `Threadblock`, `ThreadblockProgram`
- Produces: `schedule_threadblocks(program: EndpointProgram, dag: TransferDAG) -> ThreadblockProgram`
- Produces: `simulate_endpoint_execution(program: ThreadblockProgram) -> DeadlockResult`

- [x] **Step 1: Write unidirectional TB and synchronized-order tests**

```python
def test_send_and_receive_use_different_threadblocks():
    tb_program = schedule_threadblocks(two_way_endpoint_program(), transfer_dag())
    for tb in tb_program.threadblocks:
        types = {step.xml_type for step in tb.steps if step.xml_type not in {EndpointType.NOP}}
        assert not ({EndpointType.SEND} & types and {EndpointType.RECV, EndpointType.RECV_REDUCE_COPY} & types)


def test_paired_endpoints_have_matching_lane_order():
    tb_program = schedule_threadblocks(crossing_candidate_program(), transfer_dag())
    assert send_order(tb_program, 0, 1, 0) == recv_order(tb_program, 1, 0, 0)
```

- [x] **Step 2: Write three-contributor NOP and deadlock tests**

Assert the latest `ed_time` predecessor is the direct dependency; two earlier contributors create NOPs in the consumer TB. Create a deliberately crossed send/receive head order and assert the simulator reports blocked transfer IDs and TB heads.

- [x] **Step 3: Run tests and verify missing scheduling implementation**

Run: `python3 -m pytest tests/unit/xml/test_threadblocks.py tests/unit/xml/test_list_scheduler.py tests/unit/xml/test_deadlock.py -q`

Expected: collection fails.

- [x] **Step 4: Implement synchronized TransferNode list scheduling**

Group communication TBs by `(rank, direction, peer, channel)` and copy TBs by rank. Select only nodes whose semantic predecessors are scheduled; order by smaller `st_time`, longer remaining critical path, smaller `ed_time`, then `(stage_id, logical_slice_index, src, dst, channel, transfer_id)`. Append both endpoints atomically.

- [x] **Step 5: Implement join lowering and minimal inversion repair**

Choose the critical predecessor by latest end time, then critical path and stable ID. Generate one NOP per remaining cross-TB predecessor in every consuming TB. Add TB serial edges, detect cycles, and only swap semantically independent nodes while minimizing inversions relative to solver order. Reject an unrepaired cycle.

- [x] **Step 6: Implement endpoint-head event simulation**

Expose only each TB head. A communication step executes only when both paired endpoints are heads and all dependencies are complete; local copies and NOPs execute when their dependencies and TB order permit. If unfinished steps remain with no executable event, return a deadlock result and reject XML generation.

- [x] **Step 7: Run scheduling and deadlock tests**

Run: `python3 -m pytest tests/unit/xml/test_threadblocks.py tests/unit/xml/test_list_scheduler.py tests/unit/xml/test_deadlock.py -q`

Expected: all tests pass.

### Task 4: Deterministic MSCCL XML Emitter and Parser Validation

**Files:**
- Create: `vericcl/xml/emitter.py`
- Create: `vericcl/xml/parser.py`
- Create: `vericcl/xml/lower.py`
- Create: `vericcl/xml/granularity.py`
- Create: `tests/golden/xml/two_rank_allreduce_out_of_place.xml`
- Create: `tests/golden/xml/two_rank_allgather_in_place.xml`
- Test: `tests/unit/xml/test_emitter.py`
- Test: `tests/golden/test_xml_golden.py`

**Interfaces:**
- Produces: `XmlArtifact(xml_text: str, buffer_plan: BufferPlan, endpoint_program: EndpointProgram, tb_program: ThreadblockProgram, sha256: str, runtime_compatible: bool)`
- Produces: `emit_xml(program: ThreadblockProgram, buffers: BufferPlan, inputs: ResolvedInput) -> str`
- Produces: `lower_to_xml(schedule: Schedule, inputs: ResolvedInput, topology: Topology) -> XmlArtifact`
- Produces: `verify_atom_granularity(runtime_count: int, size_multiplier: int, datatype_size_bytes: int, nchunks_per_loop: int, slice_size_bytes: int, nccl_buffsize_bytes: int) -> None`

- [ ] **Step 1: Write XML field and forbidden-op tests**

```python
def test_emitter_uses_exact_atom_granularity():
    root = parse_xml(emit_tiny_allreduce())
    assert root.attrib["proto"] == "Simple"
    assert all(step.attrib["cnt"] == "1" for step in root.xpath(".//step"))


def test_emitter_never_uses_fused_operations_or_top_level_copy():
    root = parse_xml(emit_tiny_allreduce())
    assert not root.xpath(".//copy")
    assert {step.attrib["type"] for step in root.xpath(".//step")} <= {"s", "r", "rrc", "cpy", "nop"}


def test_runtime_geometry_preserves_one_atom_per_step():
    verify_atom_granularity(
        runtime_count=67_108_864,
        size_multiplier=1,
        datatype_size_bytes=4,
        nchunks_per_loop=256,
        slice_size_bytes=1_048_576,
        nccl_buffsize_bytes=2_097_152,
    )
```

- [ ] **Step 2: Write exact chunk-count and byte-range tests**

Test the six operator table, including AllGather `nchunksperloop=P*N` and `runtime_bytes=P*M`; test all other direct operators use their specified chunk counts and `runtime_bytes=M`. Assert `minBytes=runtime_bytes` and `maxBytes=runtime_bytes+1`.

- [ ] **Step 3: Run tests and confirm missing emitter**

Run: `python3 -m pytest tests/unit/xml/test_emitter.py tests/golden/test_xml_golden.py -q`

Expected: collection fails.

- [ ] **Step 4: Implement deterministic XML emission**

Emit `<algo name="vericcl">`, exact coll/inplace/ngpus/nchannels/nchunksperloop/minBytes/maxBytes attributes, sorted GPUs, TB IDs, and sequential step indices. Emit `srcbuf/srcoff/dstbuf/dstoff/cnt/depid/deps/hasdep` for every step. Use `-1` offsets for NOP endpoints as required by the MSCCL schema. Never mutate the intermediate program during serialization.

- [ ] **Step 5: Implement parse-back structural validation and golden normalization**

Parse emitted XML, verify all references, contiguous TB/step IDs, single-direction TBs, known operations, offsets within declared buffers, one pair per transfer sidecar, and no `cnt>1`. Golden comparison ignores only insignificant XML whitespace; it does not reorder TBs or steps.

- [ ] **Step 6: Run emitter and golden tests**

Run: `python3 -m pytest tests/unit/xml/test_emitter.py tests/golden/test_xml_golden.py -q`

Expected: all tests pass and golden files contain no Chinese characters.

### Task 5: MSCCL Compatibility Checks and Candidate Recommendations

**Files:**
- Create: `vericcl/xml/compatibility.py`
- Create: `vericcl/xml/recommendations.py`
- Test: `tests/unit/xml/test_compatibility.py`
- Test: `tests/unit/xml/test_recommendations.py`

**Interfaces:**
- Produces: `CompatibilityIssue`, `CompatibilityReport`
- Produces: `check_msccl_compatibility(artifact: XmlArtifact) -> CompatibilityReport`
- Produces: `recommend_runtime_compatible_inputs(inputs: ResolvedInput, artifact: XmlArtifact) -> tuple[Recommendation, ...]`

- [ ] **Step 1: Write one test per confirmed MSCCL limit**

Cover 257 steps/TB, 33 send TBs/channel, 33 receive TBs/channel, 217 TBs/rank, 33 channels, offset 32768, and a dependent TB that cannot be renumbered into `0..127`. Assert each issue reports rank, TB/channel, current value, limit, and transfer IDs.

- [ ] **Step 2: Write candidate naming and recommendation tests**

Assert a compatible artifact uses `.xml`; an incompatible artifact uses `.candidate.xml`, remains available for offline analysis, and recommends TB renumbering before channel or larger-slice changes. Larger slice candidates must divide total size and preserve `N % P == 0` where required.

- [ ] **Step 3: Run tests and verify missing compatibility module**

Run: `python3 -m pytest tests/unit/xml/test_compatibility.py tests/unit/xml/test_recommendations.py -q`

Expected: collection fails.

- [ ] **Step 4: Implement checks without turning them into solve constraints**

Compatibility failures set `runtime_compatible=False` but do not invalidate logical schedules, stop BDD analysis, or mutate current slice size/channel count. Renumber dependent TBs only when content, order, and dependencies remain unchanged. Recommendations are next-run inputs, not current artifact edits.

- [ ] **Step 5: Run Phase 04 regression and coverage**

Run: `python3 -m pytest -m phase04 --cov=vericcl.xml --cov-report=term-missing -q`

Expected: all Phase 04 tests pass and XML modules reach at least 90% line coverage.

Run: `rg -n '[\p{Han}]' vericcl/xml tests/unit/xml tests/golden -g '*.{py,xml}'`

Expected: no output.
