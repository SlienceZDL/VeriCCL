# VeriCCL Foundation, Input, and Semantics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立可安装的 `vericcl` 包、确定性输入规范化、slice/atom/PayloadState 数据模型以及六类直接算子的最终语义。

**Architecture:** 先冻结跨模块数据类型，再实现无求解器依赖的输入与语义核心。输入层只负责解析和规范化，语义层只负责状态转换与最终输出集合，不读取文件、不生成 XML，也不依赖 Gurobi。

**Tech Stack:** Python 3.9+、dataclasses、Enum、argparse、hashlib、json、pytest、hypothesis。

## Global Constraints

- 继承索引计划的全部全局约束。
- 所有 dataclass 默认 `frozen=True`；需要状态消费时由 `PayloadLedger` 维护版本状态，不原地修改 PayloadState。
- JSON 规范化使用排序键和紧凑分隔符，SHA-256 必须对等价输入稳定。
- 错误通过 `VeriCCLError` 子类报告，诊断消息只使用英文。
- 此阶段不移除旧 `taccl` 包；最终迁移在 Phase 07 完成。
- 本计划创建的每个测试模块声明 `pytestmark = pytest.mark.phase01`。

---

### Task 1: Package Skeleton and Test Harness

Before Step 1, create the isolated environment with `python3 -m venv .venv` and install only the test bootstrap dependency with `.venv/bin/python -m pip install pytest`. In every command below, use `.venv/bin/python` in place of `python3`; later installation steps must not modify the system Python environment.

**Files:**
- Modify: `setup.py`
- Create: `vericcl/__init__.py`
- Create: `vericcl/__main__.py`
- Create: `vericcl/errors.py`
- Create: `vericcl/cli/__init__.py`
- Create: `vericcl/cli/main.py`
- Create: `requirements-dev.txt`
- Create: `pytest.ini`
- Create: `tests/conftest.py`
- Test: `tests/unit/test_cli.py`

**Interfaces:**
- Produces: `vericcl.cli.main.build_parser() -> argparse.ArgumentParser`
- Produces: `vericcl.cli.main.main(argv: Optional[Sequence[str]] = None) -> int`
- Produces: `VeriCCLError`, `InputValidationError`, `SemanticError`, `SolverUnavailableError`, `RuntimeCompatibilityError`

- [x] **Step 1: Write the failing CLI and package tests**

```python
from vericcl.cli.main import build_parser, main


def test_parser_exposes_solve_and_verify():
    parser = build_parser()
    assert parser.parse_args(["solve", "--topology", "t.json", "--sketch", "s.json", "--atoms", "a.json"]).command == "solve"
    assert parser.parse_args(["verify", "--topology", "t.json", "--sketch", "s.json", "--atoms", "a.json", "--xml", "x.xml"]).command == "verify"


def test_main_returns_zero_for_help():
    assert main(["--version"]) == 0
```

- [x] **Step 2: Run the test and confirm the package is absent**

Run: `python3 -m pytest tests/unit/test_cli.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'vericcl'`.

- [x] **Step 3: Add the package, errors, entry point, and development dependencies**

Implement `setup.py` with distribution name `vericcl`, `find_packages(include=["vericcl", "vericcl.*"])`, Python requirement `>=3.9`, console entry `vericcl=vericcl.cli.main:console_main`, and existing runtime dependencies. Add `pytest`, `pytest-cov`, and `hypothesis` to `requirements-dev.txt`. Configure markers `phase01` through `phase07`, `gurobi`, and `hardware` in `pytest.ini`.

```python
class VeriCCLError(Exception):
    pass


class InputValidationError(VeriCCLError):
    pass


class SemanticError(VeriCCLError):
    pass
```

`main()` must parse `--version`, `solve`, and `verify`; handler imports remain local so this phase does not create circular imports.

- [x] **Step 4: Install the editable package and run the focused test**

Run: `python3 -m pip install -e . -r requirements-dev.txt`

Expected: installation succeeds; if `gurobipy` cannot be installed on the current platform, install the editable package with `--no-deps`, install the pure-software dependencies explicitly, and record Gurobi as `not_run`.

Fallback commands:

```sh
python3 -m pip install -e . --no-deps
python3 -m pip install argcomplete lxml numpy ply z3-solver pytest pytest-cov hypothesis
```

Run: `python3 -m pytest tests/unit/test_cli.py -q`

Expected: `2 passed`.

- [x] **Step 5: Check the phase files for forbidden characters and review the file list**

Run: `rg -n '[\p{Han}]' vericcl tests setup.py requirements-dev.txt pytest.ini -g '*.{py,txt,ini}'`

Expected: no output.

Checkpoint files: `setup.py`, `vericcl/`, `requirements-dev.txt`, `pytest.ini`, `tests/conftest.py`, `tests/unit/test_cli.py`.

### Task 2: Immutable Input Models and Deterministic JSON

**Files:**
- Create: `vericcl/input/__init__.py`
- Create: `vericcl/input/models.py`
- Create: `vericcl/input/json_codec.py`
- Create: `vericcl/semantics/__init__.py`
- Create: `vericcl/semantics/collective.py`
- Test: `tests/unit/input/test_models.py`
- Test: `tests/unit/input/test_json_codec.py`

**Interfaces:**
- Produces from `vericcl.semantics.collective`: `CollectiveKind`, `CollectiveSpec`
- Produces from `vericcl.input.models`: `ObjectiveMode`, `Hyperparameters`, `SolverConfig`, `StrategyConfig`, `ForbiddenTransfer`, `AtomConstraints`, `ResolvedInput`
- Produces: `canonical_json(value: object) -> str`
- Produces: `sha256_json(value: object) -> str`

- [x] **Step 1: Write failing model and canonicalization tests**

```python
from vericcl.input.json_codec import canonical_json, sha256_json
from vericcl.input.models import Hyperparameters
from vericcl.semantics.collective import CollectiveKind, CollectiveSpec


def test_hyperparameters_derive_slice_count():
    hp = Hyperparameters(total_size_bytes=8 * 1024 * 1024, slice_size_bytes=1024 * 1024)
    assert hp.slice_count == 8


def test_canonical_json_is_order_independent():
    left = {"b": 2, "a": [1, 3]}
    right = {"a": [1, 3], "b": 2}
    assert canonical_json(left) == canonical_json(right)
    assert sha256_json(left) == sha256_json(right)


def test_collective_spec_defaults_to_out_of_place():
    spec = CollectiveSpec(kind=CollectiveKind.ALL_REDUCE, datatype="float32", reduction_op="sum")
    assert spec.inplace is False
```

- [x] **Step 2: Run tests and verify missing modules fail**

Run: `python3 -m pytest tests/unit/input/test_models.py tests/unit/input/test_json_codec.py -q`

Expected: collection fails because `vericcl.input` is absent.

- [x] **Step 3: Implement exact immutable input types**

Use these field contracts:

```python
class CollectiveKind(str, Enum):
    BROADCAST = "broadcast"
    REDUCE = "reduce"
    ALL_GATHER = "allgather"
    ALL_REDUCE = "allreduce"
    ALL_TO_ALL = "alltoall"
    REDUCE_SCATTER = "reduce_scatter"


class ObjectiveMode(str, Enum):
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    AUTO = "auto"


@dataclass(frozen=True)
class CollectiveSpec:
    kind: CollectiveKind
    datatype: str
    reduction_op: Optional[str] = None
    root: Optional[int] = None
    inplace: bool = False


@dataclass(frozen=True)
class Hyperparameters:
    total_size_bytes: int
    slice_size_bytes: int
    objective_mode: ObjectiveMode = ObjectiveMode.AUTO
    max_calibration_channels: int = 32
    min_expected_improvement: float = 0.01
    min_tuning_improvement: float = 0.01
    max_tuning_iterations: int = 20
    total_verification_timeout_s: int = 10800
    force_recalibrate: bool = False

    @property
    def slice_count(self) -> int:
        return self.total_size_bytes // self.slice_size_bytes


@dataclass(frozen=True)
class SolverConfig:
    total_solve_timeout_s: int = 10800
    per_model_timeout_s: int = 1800
    mip_gap: float = 1e-4
    require_proven_optimal: bool = False
    solver_seed: int = 0
    max_channels: int = 32
    max_threads_per_model: int = 12
    max_parallel_models: int = 4
    force_resolve: bool = False


@dataclass(frozen=True)
class ForbiddenTransfer:
    slice_id: int
    src_rank: int
    dst_rank: int
    stage_id: int


@dataclass(frozen=True)
class AtomConstraints:
    stage_num: Optional[int]
    forbidden_transfers: tuple[ForbiddenTransfer, ...]


@dataclass(frozen=True)
class StrategyConfig:
    hierarchy: bool
    symmetry: bool
    shortest_paths: bool
    batching: bool
    constructive_trees: bool
    milp: bool
    manual_hierarchy: tuple[Mapping[str, object], ...]


@dataclass(frozen=True)
class ResolvedInput:
    collective: CollectiveSpec
    hyperparameters: Hyperparameters
    solver: SolverConfig
    strategies: StrategyConfig
    atom_constraints: AtomConstraints
    rank_count: int
    resolved_topology: Mapping[str, object]
    resolved_sketch: Mapping[str, object]
    resolved_atom: Mapping[str, object]
    input_sha256: str
```

- [x] **Step 4: Implement recursive JSON conversion without serializing Python repr strings**

`canonical_json()` must convert dataclasses, enums, frozensets, tuples, mappings, and paths to JSON-native values, sort mapping keys and set elements, then call `json.dumps(..., sort_keys=True, separators=(",", ":"), ensure_ascii=True)`.

- [x] **Step 5: Run focused tests**

Run: `python3 -m pytest tests/unit/input/test_models.py tests/unit/input/test_json_codec.py -q`

Expected: all tests pass.

### Task 3: Three-File Input Loader and Validation

**Files:**
- Create: `vericcl/input/loader.py`
- Create: `vericcl/input/validation.py`
- Create: `vericcl/examples/topo/two_rank.json`
- Create: `vericcl/examples/sketch/allreduce_8m_1m.json`
- Create: `vericcl/examples/atom/default.json`
- Test: `tests/unit/input/test_loader.py`
- Test: `tests/unit/input/test_validation.py`

**Interfaces:**
- Consumes: Phase 01 Task 2 input models
- Produces: `resolve_inputs(topology_path: Path, sketch_path: Path, atom_path: Path) -> ResolvedInput`
- Produces: `validate_collective(spec: CollectiveSpec, rank_count: int, slice_count: int) -> None`

- [x] **Step 1: Write positive and negative loader tests**

```python
def test_resolve_inputs_derives_global_rank_count_and_chunkup(tmp_path):
    paths = write_three_inputs(tmp_path, ranks=2, total=8_388_608, size=1_048_576)
    resolved = resolve_inputs(*paths)
    assert resolved.rank_count == 2
    assert resolved.hyperparameters.slice_count == 8
    assert resolved.resolved_sketch["hyperparameters"]["input_chunkup"] == 8


@pytest.mark.parametrize("total,size", [(0, 1), (8, 0), (10, 4)])
def test_invalid_slice_geometry_is_rejected(tmp_path, total, size):
    paths = write_three_inputs(tmp_path, ranks=2, total=total, size=size)
    with pytest.raises(InputValidationError):
        resolve_inputs(*paths)


def test_reduce_scatter_requires_divisible_slice_count(tmp_path):
    paths = write_three_inputs(tmp_path, ranks=4, total=6, size=1, operator="reduce_scatter")
    with pytest.raises(InputValidationError, match="slice count must be divisible"):
        resolve_inputs(*paths)
```

- [x] **Step 2: Run tests and observe missing loader failures**

Run: `python3 -m pytest tests/unit/input/test_loader.py tests/unit/input/test_validation.py -q`

Expected: collection fails on missing loader functions.

- [x] **Step 3: Implement strict loading and normalization**

`topo.json` rank count is derived from explicit `ranks`, or from `nnodes * gpus_per_node` for legacy examples. `sketch.json` is the sole source of CollectiveSpec. Command-line semantic overrides are not implemented in this task. `atom.json` accepts `stage_num`, a list of four-field forbidden transfers, strategy booleans, and optional manual hierarchy.

Validation must reject unknown operators, missing or out-of-range roots, missing reduction operations for reduction collectives, non-positive sizes, non-divisible sizes, inconsistent `input_chunkup`, and AllToAll/ReduceScatter `N % P != 0`.

- [x] **Step 4: Emit normalized examples and resolved input content**

The three examples must contain only English keys and diagnostics. The normalized sketch must always contain explicit defaults for `inplace`, solver budgets, objective mode, calibration options, and tuning limits so cache signatures never depend on implicit defaults.

- [x] **Step 5: Run loader tests and inspect deterministic hash stability**

Run: `python3 -m pytest tests/unit/input -q`

Expected: all input tests pass, including equal hashes for reordered JSON keys.

### Task 4: Slice, Atom, Transfer, and Schedule Models

**Files:**
- Create: `vericcl/semantics/slice.py`
- Create: `vericcl/semantics/atom.py`
- Test: `tests/unit/semantics/test_slice.py`
- Test: `tests/unit/semantics/test_atom.py`

**Interfaces:**
- Produces: `source_rank(slice_id: int, slice_count: int) -> int`
- Produces: `logical_slice_index(slice_id: int, slice_count: int) -> int`
- Produces: `Symbol`, `PathStage`, `Atom`, `Transfer`, `Schedule`

- [x] **Step 1: Write failing identity, path-prefix, and transfer dedup tests**

```python
def test_slice_identity_uses_global_n():
    assert source_rank(7, 4) == 1
    assert logical_slice_index(7, 4) == 3


def test_atom_path_ends_at_transfer_source():
    atom = make_atom(slice_id=0, symbols=(Symbol(0, 1, 2.0), Symbol(1, 2, 4.0)), st_time=4.0, ed_time=6.0)
    atom.validate_path_prefix(current_rank=2)


def test_shared_transfer_counts_physical_bytes_once():
    transfer = make_transfer(member_slice_ids=frozenset({0, 4}), size_bytes=1024)
    assert transfer.physical_bytes == 1024
```

- [x] **Step 2: Run tests and confirm missing semantics modules**

Run: `python3 -m pytest tests/unit/semantics/test_slice.py tests/unit/semantics/test_atom.py -q`

Expected: collection fails on missing imports.

- [x] **Step 3: Implement exact atom and transfer records**

```python
@dataclass(frozen=True)
class Symbol:
    src_rank: int
    dst_rank: int
    ready_time: float


@dataclass(frozen=True)
class PathStage:
    stage_id: int
    operator: str
    symbols: tuple[Symbol, ...]

    @property
    def operation_count(self) -> int:
        return len(self.symbols)


@dataclass(frozen=True)
class Atom:
    slice_id: int
    slice_size_bytes: int
    path: tuple[PathStage, ...]
    st_time: float
    ed_time: float

    @property
    def stage_num(self) -> int:
        return len(self.path)


@dataclass(frozen=True)
class Transfer:
    transfer_id: str
    kind: str
    src_rank: int
    dst_rank: int
    channel: int
    stage_id: int
    member_slice_ids: frozenset[int]
    atoms: tuple[Atom, ...]
    st_time: float
    ed_time: float
    predecessor_ids: frozenset[str]

    @property
    def physical_bytes(self) -> int:
        return self.atoms[0].slice_size_bytes


@dataclass(frozen=True)
class Schedule:
    schedule_id: str
    transfers: tuple[Transfer, ...]
    final_state_ids: tuple[str, ...]
    rank_count: int
    slice_count: int
    slice_size_bytes: int
    metadata: Mapping[str, object]
```

`Schedule` contains a deterministic tuple of transfers, final state IDs, rank count, slice count, slice size, and metadata. Constructor validation enforces non-negative IDs, exact atom size, `st_time <= ed_time`, unique transfer IDs, and path-prefix consistency.

- [x] **Step 4: Implement forbidden-member matching**

Add `Transfer.is_forbidden(forbidden: Collection[ForbiddenTransfer]) -> bool`; return true when any member slice matches `(slice_id, src_rank, dst_rank, stage_id)`.

- [x] **Step 5: Run all atom tests**

Run: `python3 -m pytest tests/unit/semantics/test_slice.py tests/unit/semantics/test_atom.py -q`

Expected: all tests pass.

### Task 5: PayloadState and AggregateState Ledger

**Files:**
- Create: `vericcl/semantics/state.py`
- Test: `tests/unit/semantics/test_state.py`

**Interfaces:**
- Consumes: slice identity helpers and atom path types
- Produces: `PayloadState`, `PayloadLedger`, `initial_payload_states(rank_count: int, slice_count: int) -> tuple[PayloadState, ...]`
- Produces: `PayloadLedger.reduce(left_id: str, right_id: str, dst_rank: int, ready_time: float) -> PayloadState`
- Produces: `PayloadLedger.send(state_id: str, dst_rank: int, ready_time: float, required_contributors: frozenset[int]) -> PayloadState`

- [ ] **Step 1: Write state transition tests including negative cases**

```python
def test_reduce_unions_disjoint_contributors():
    ledger = ledger_with_states(state("a", 0, 0, {0}), state("b", 0, 0, {4}))
    result = ledger.reduce("a", "b", dst_rank=0, ready_time=3.0)
    assert result.contributors == frozenset({0, 4})


def test_reduce_rejects_intersecting_contributors():
    ledger = ledger_with_states(state("a", 0, 0, {0, 4}), state("b", 1, 0, {4}))
    with pytest.raises(SemanticError, match="contributors must be disjoint"):
        ledger.reduce("a", "b", dst_rank=1, ready_time=3.0)


def test_consumed_reduce_source_cannot_be_reused():
    ledger = ledger_with_three_singletons()
    ledger.reduce("a", "b", dst_rank=0, ready_time=2.0)
    with pytest.raises(SemanticError, match="state version is inactive"):
        ledger.reduce("a", "c", dst_rank=0, ready_time=4.0)


def test_incomplete_state_has_at_most_one_outbound_send():
    ledger = ledger_with_states(state("a", 0, 0, {0}))
    ledger.send("a", dst_rank=1, ready_time=1.0, required_contributors=frozenset({0, 4}))
    with pytest.raises(SemanticError, match="incomplete state already sent"):
        ledger.send("a", dst_rank=2, ready_time=1.0, required_contributors=frozenset({0, 4}))
```

- [ ] **Step 2: Run the tests and verify failure**

Run: `python3 -m pytest tests/unit/semantics/test_state.py -q`

Expected: collection fails because state types are absent.

- [ ] **Step 3: Implement the immutable state record and mutable ledger boundary**

```python
@dataclass(frozen=True)
class PayloadState:
    state_id: str
    version: int
    rank: int
    logical_address: int
    contributors: frozenset[int]
    ready_time: float
    active: bool
    member_paths: tuple[tuple[int, tuple[PathStage, ...]], ...]
```

The ledger owns state versions, inactive IDs, incomplete outbound counts, and the active `(rank, logical_address)` aggregate index. REDUCE verifies equal logical address, disjoint contributors, active versions, target uniqueness, and `ready_time >= max(input.ready_time)`; it deactivates both inputs and creates one new version. SEND creates a destination version, preserves member paths, and deactivates only an incomplete source after its single allowed outbound operation. Complete states may branch without consuming the source.

- [ ] **Step 4: Add local-contributor merge without a self-transfer**

Implement `merge_local(state_id, local_state_id, ready_time)` as a ledger reduction that records no network Transfer. Test that the resulting state has the union contributors and the caller's transfer list remains unchanged.

- [ ] **Step 5: Run state tests**

Run: `python3 -m pytest tests/unit/semantics/test_state.py -q`

Expected: all positive and negative tests pass.

### Task 6: Collective Output Semantics

**Files:**
- Modify: `vericcl/semantics/collective.py`
- Create: `vericcl/semantics/checker.py`
- Test: `tests/unit/semantics/test_collective.py`
- Test: `tests/property/test_collective_semantics.py`

**Interfaces:**
- Consumes: `CollectiveSpec`, PayloadState, slice identity helpers
- Produces: `OutputSlot(rank: int, offset: int)`
- Produces: `required_outputs(spec: CollectiveSpec, rank_count: int, slice_count: int) -> Mapping[OutputSlot, frozenset[int]]`
- Produces: `check_final_states(spec: CollectiveSpec, rank_count: int, slice_count: int, states: Iterable[PayloadState]) -> None`

- [ ] **Step 1: Write table-driven tests for all six direct operators**

```python
@pytest.mark.parametrize(
    "kind,slot,contributors",
    [
        (CollectiveKind.BROADCAST, OutputSlot(1, 0), frozenset({0})),
        (CollectiveKind.REDUCE, OutputSlot(0, 0), frozenset({0, 2})),
        (CollectiveKind.ALL_GATHER, OutputSlot(1, 2), frozenset({2})),
        (CollectiveKind.ALL_REDUCE, OutputSlot(1, 0), frozenset({0, 2})),
        (CollectiveKind.ALL_TO_ALL, OutputSlot(1, 0), frozenset({1})),
        (CollectiveKind.REDUCE_SCATTER, OutputSlot(1, 0), frozenset({1, 3})),
    ],
)
def test_required_output_mapping(kind, slot, contributors):
    spec = make_spec(kind)
    assert required_outputs(spec, rank_count=2, slice_count=2)[slot] == contributors
```

- [ ] **Step 2: Add property tests for complete and duplicate-free contributor sets**

Generate rank counts 2 to 4 and slice counts divisible by rank count. Assert that AllReduce has `P*N` output slots, each logical slot contains exactly one slice from every source rank, and ReduceScatter partitions the `N` logical positions without overlap.

- [ ] **Step 3: Run tests and confirm missing implementation failure**

Run: `python3 -m pytest tests/unit/semantics/test_collective.py tests/property/test_collective_semantics.py -q`

Expected: collection fails on missing collective functions.

- [ ] **Step 4: Implement final output mappings exactly as specified**

Broadcast uses root contributions at every rank; Reduce emits only root outputs; AllGather maps source `r`, logical `l` to offset `r*N+l`; AllReduce maps every full aggregate to `l`; AllToAll uses `q=N/P`, destination `floor(l/q)`, offset `r*q+(l mod q)`; ReduceScatter uses owner `floor(l/q)` and offset `l mod q`.

`check_final_states()` must compare exact output keys and exact contributor sets, rejecting missing, extra, duplicated, or misaddressed outputs.

- [ ] **Step 5: Run the complete Phase 01 suite and coverage**

Run: `python3 -m pytest -m phase01 --cov=vericcl.input --cov=vericcl.semantics --cov-report=term-missing -q`

Expected: all Phase 01 tests pass and new modules reach at least 90% line coverage.

Run: `rg -n '[\p{Han}]' vericcl tests -g '*.py'`

Expected: no output.
