# VeriCCL V100 调度合成与评测 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不修改 GPU、NIC、网络或许可证配置的前提下，实现可恢复的双机实验支持，合成并验证 24 个 VeriCCL V100 调度，在 node2/node4 上完成 47 次主性能运行，并提交可复现的 XML、日志和 in-place `algbw` 报告。

**Architecture:** node2 运行 VeriCCL、Gurobi、离线验证和结果管理，通过可注入的 SSH executor 在 node4 发起 MPI/nccl-tests；两台机器使用相同实验路径，XML 从 node2 原子暂存到 node4，node4 的前半 Rank trace 回收到 node2 后统一分析。实验模块按远程传输、矩阵模型、性能解析、状态恢复和报告生成拆分，现有单机在线验证默认行为保持不变。

**Tech Stack:** Python 3.10+、pytest、Gurobi 13、lxml、dd、OpenMPI 4.1.8、CUDA 12.8、patched MSCCL、nccl-tests、SSH/SCP、JSON/CSV/Markdown。

**Spec:** `docs/superpowers/specs/2026-09-01-v100-schedule-evaluation-design.md`

## Global Constraints

- 严禁读取、写入、列举或修改 `/home/cc` 及其子路径。
- 严禁修改 GPU、NIC、InfiniBand、以太网、驱动、固件、内核、接口状态、系统权限或 Gurobi 许可证。
- 远程写入仅限 `/home/zdl/VeriCCL-experiments/2026-09-01-v100-k16` 及为构建隔离副本所需的 `/home/zdl` 用户目录。
- `/home/zdl/MSCCL/test` 只读；基线 XML 必须复制到实验目录后运行。
- node2 是 VeriCCL/Gurobi 控制端；node4 是 MPI 启动端；hostfile 中 node4 必须排在 node2 前面。
- node2 当前 IB 不可用；运行时固定 `NCCL_IB_DISABLE=1`，只测试现有 10.0.0.0/24 TCP/Ethernet 路径。
- 全局软件并发上限固定为 16；topo 中表示物理容量的 `max_channels=32` 不修改。
- AllGather 的 nccl-tests size 等于 `rank_count × total_size_bytes`；AllReduce 的 nccl-tests size 等于 `total_size_bytes`。
- 所有实验 sketch 保持 `inplace=true`，atom 固定为 `vericcl/examples/atom/default.json`。
- VeriCCL XML 使用 `NCCL_BUFFSIZE=2 × slice_size_bytes`；基线 XML 使用 `NCCL_BUFFSIZE=2097152`。
- Python、C/C++、CUDA、shell 与测试代码只允许 ASCII；中文仅用于文档和报告。
- 不覆盖用户现有 README、`docs/figures/`、`docs/vericcl-paper-story.md` 或其他未提交修改。
- 不使用 `git reset --hard`、`git checkout --` 或强制推送。

---

### Task 1: 提交已经验证的在线运行时修复

**Files:**
- Modify: `README.md`
- Modify: `README.zh-CN.md`
- Modify: `Vericcl-work-document.md`
- Modify: `setup.py`
- Modify: `docs/runtime-configuration.md`
- Modify: `runtime/msccl-trace/README.md`
- Modify: `runtime/msccl-trace/tools/vericcl_clock_sync.cu`
- Modify: `runtime/msccl-trace/tools/verify_patch.py`
- Create: `runtime/msccl-trace/patches/0002-vericcl-host-step-signature.patch`
- Modify: `tests/hardware/_support.py`
- Modify: `tests/unit/cli/test_online.py`
- Modify: `tests/unit/online/test_calibration.py`
- Modify: `tests/unit/online/test_calibration_cache.py`
- Modify: `tests/unit/online/test_clock_sync.py`
- Modify: `tests/unit/online/test_nccl_tests.py`
- Modify: `tests/unit/online/test_pipeline.py`
- Modify: `tests/unit/online/test_runtime_patch.py`
- Modify: `vericcl/cli/online.py`
- Modify: `vericcl/verification/online/cache.py`
- Modify: `vericcl/verification/online/calibration.py`
- Modify: `vericcl/verification/online/clock_sync.py`
- Modify: `vericcl/verification/online/nccl_tests.py`
- Modify: `vericcl/verification/online/pipeline.py`

**Interfaces:**
- Consumes: 当前工作树中已实现的 MPI 一进程一卡、滚动校准预算、稳定缓存、clock fit、MSCCL 4/4 host proxy 修复。
- Produces: 一个独立、完整、可合并的运行时修复提交；后续任务以此为基线。

- [ ] **Step 1: 确认只包含已知修改**

Run:

```bash
git status --short
git diff --name-only
```

Expected: 文件集合与本任务 `Files` 完全一致，另有已经提交的设计文档；不得出现 `exp/`、论文文件或未知用户文件。

- [ ] **Step 2: 添加第二个 runtime patch 的安装包失败测试**

Add to `tests/unit/online/test_runtime_patch.py`:

```python
def test_setup_installs_both_runtime_patches():
    text = Path("setup.py").read_text(encoding="ascii")
    assert "0001-vericcl-fixed-step-trace.patch" in text
    assert "0002-vericcl-host-step-signature.patch" in text
```

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/unit/online/test_runtime_patch.py::test_setup_installs_both_runtime_patches
```

Expected: FAIL because `setup.py` currently installs only patch 0001.

- [ ] **Step 3: 将 patch 0002 加入发行文件**

Update the runtime patch `data_files` list in `setup.py` to contain both exact paths:

```python
[
    "runtime/msccl-trace/patches/0001-vericcl-fixed-step-trace.patch",
    "runtime/msccl-trace/patches/0002-vericcl-host-step-signature.patch",
]
```

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/unit/online/test_runtime_patch.py::test_setup_installs_both_runtime_patches
```

Expected: PASS.

- [ ] **Step 4: 重新运行针对性测试**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/unit/cli/test_online.py \
  tests/unit/online/test_calibration.py \
  tests/unit/online/test_calibration_cache.py \
  tests/unit/online/test_clock_sync.py \
  tests/unit/online/test_nccl_tests.py \
  tests/unit/online/test_pipeline.py \
  tests/unit/online/test_runtime_patch.py
```

Expected: all selected tests pass.

- [ ] **Step 5: 运行完整本地验证**

Run:

```bash
.venv/bin/python -m pytest -q
git diff --check
.venv/bin/python -m compileall -q vericcl tests runtime/msccl-trace/tools
if rg -n '[^\x00-\x7F]' vericcl tests runtime/msccl-trace/tools \
  --glob '*.py' --glob '*.cu' --glob '*.cc' --glob '*.h' --glob '*.sh'; \
then exit 1; fi
```

Expected: pytest 全部通过；其余命令退出 0；ASCII 扫描无输出。

- [ ] **Step 6: 只暂存本任务文件并复核**

Run:

```bash
git add README.md README.zh-CN.md Vericcl-work-document.md setup.py \
  docs/runtime-configuration.md runtime/msccl-trace \
  tests/hardware/_support.py tests/unit/cli/test_online.py \
  tests/unit/online vericcl/cli/online.py \
  vericcl/verification/online
git diff --cached --check
git diff --cached --name-only
```

Expected: staged names 仅为本任务文件，不包含设计/计划之外的用户文件。

- [ ] **Step 7: 提交运行时修复**

```bash
git commit -m "fix: harden MSCCL online validation"
```

Expected: commit succeeds and working tree retains no Task 1 implementation diff.

---

### Task 2: 安全集成已验证修复并导入实验输入

**Files:**
- Add: `exp/topo/a100-n2g8.json`
- Add: `exp/topo/v100-n2g4.json`
- Add: `exp/topo/v100-n2g8.json`
- Add: `exp/sketch/a100-n2g8/{ag,ar}/*.json`
- Add: `exp/sketch/v100-n2g4/{ag,ar}/*.json`
- Add: `exp/sketch/v100-n2g8/{ag,ar}/*.json`
- Preserve: `README.zh-CN.md`
- Preserve: `docs/figures/`
- Preserve: `docs/vericcl-paper-story.md`

**Interfaces:**
- Consumes: Task 1 的已验证提交和当前主工作区中用户创建的 39 个 JSON 文件。
- Produces: `feature/vericcl-implementation` 上可直接 clone 的已验证代码与实验输入；当前功能分支同步到同一基线。

- [ ] **Step 1: 记录两个工作树与分支关系**

Run in the feature worktree:

```bash
git rev-parse --show-toplevel
git branch --show-current
git merge-base --is-ancestor feature/vericcl-implementation HEAD
```

Expected: 当前分支为 `feature/vericcl-scalable-hierarchical-solving`，最后一条命令退出 0。

- [ ] **Step 2: 在主工作区保存全部用户改动**

Run in `/Users/zdl/work/code/VeriCCL`:

```bash
git status --short
git stash push -u -m "preserve-user-work-before-v100-integration-20260902"
git status --short
```

Expected: stash 创建成功，主工作区临时干净；不要删除该 stash。

- [ ] **Step 3: 快进合并已验证修复**

Run in `/Users/zdl/work/code/VeriCCL`:

```bash
git switch feature/vericcl-implementation
git merge --ff-only feature/vericcl-scalable-hierarchical-solving
.venv/bin/python -m pytest -q
git push origin feature/vericcl-implementation
```

Expected: fast-forward succeeds, tests pass, remote update succeeds.

- [ ] **Step 4: 恢复用户改动但保留 stash 作为备份**

Run:

```bash
git stash apply 'stash@{0}'
git status --short
```

Expected: `exp/`、README 和论文文件恢复。若命令返回冲突，停止后续 Git 操作，保留 stash，逐文件比较 stage 1/2/3 并使用 `apply_patch` 合并两侧内容；不得用检出命令覆盖任一侧。

- [ ] **Step 5: 验证全部静态实验输入**

Run:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from vericcl.input.loader import resolve_inputs
from vericcl.planner import build_plan

root = Path(".").resolve()
atom = root / "vericcl/examples/atom/default.json"
count = 0
for topology in sorted((root / "exp/topo").glob("*.json")):
    sketch_root = root / "exp/sketch" / topology.stem
    for sketch in sorted(sketch_root.glob("*/*.json")):
        resolved = resolve_inputs(topology, sketch, atom)
        build_plan(resolved)
        count += 1
assert count == 36, count
print("validated", count)
PY
```

Expected: `validated 36`.

- [ ] **Step 6: 仅提交 JSON，排除 macOS 元数据**

Run:

```bash
git add ':(glob)exp/topo/*.json' ':(glob)exp/sketch/**/*.json'
git diff --cached --check
git diff --cached --name-only
```

Expected: 39 个 JSON staged；没有 `.DS_Store`、README 或论文文件。

- [ ] **Step 7: 提交并推送实验输入**

```bash
git commit -m "testdata: add collective experiment inputs"
git push origin feature/vericcl-implementation
```

- [ ] **Step 8: 同步功能工作树并确认用户改动仍保留**

Run in the feature worktree:

```bash
git merge --ff-only feature/vericcl-implementation
```

Run in `/Users/zdl/work/code/VeriCCL`:

```bash
git status --short
```

Expected: 功能分支包含 JSON；主工作区仍显示用户的未提交 README/论文文件。确认后才执行 `git stash drop 'stash@{0}'`。

---

### Task 3: 将软件并发契约统一为 16

**Files:**
- Create: `vericcl/constants.py`
- Modify: `vericcl/input/models.py`
- Modify: `vericcl/input/loader.py`
- Modify: `vericcl/verification/online/calibration.py`
- Modify: `vericcl/verification/online/calibration_xml.py`
- Modify: `vericcl/verification/online/pipeline.py`
- Modify: `vericcl/cli/online.py`
- Modify: `vericcl/examples/sketch/allreduce_8m_1m.json`
- Modify: `exp/sketch/v100-n2g4/{ag,ar}/*.json`
- Modify: `exp/sketch/v100-n2g8/{ag,ar}/*.json`
- Modify: `exp/sketch/a100-n2g8/{ag,ar}/*.json`
- Modify: `README.md`
- Modify: `README.zh-CN.md`
- Modify: `Vericcl-work-document.md`
- Modify: `docs/runtime-configuration.md`
- Test: `tests/unit/input/test_models.py`
- Test: `tests/unit/input/test_loader.py`
- Test: `tests/unit/online/test_calibration_xml.py`
- Test: `tests/unit/online/test_pipeline.py`
- Create: `tests/integration/test_experiment_inputs.py`

**Interfaces:**
- Consumes: topo 物理 `max_channels` 和 sketch 软件参数。
- Produces: `SOFTWARE_MAX_CONCURRENCY: int = 16`；所有默认求解与校准路径共享该常量。

- [ ] **Step 1: 写默认值和输入上限的失败测试**

```python
from vericcl.constants import SOFTWARE_MAX_CONCURRENCY

def test_public_concurrency_defaults_are_sixteen():
    assert SOFTWARE_MAX_CONCURRENCY == 16
    assert Hyperparameters(8, 1).max_calibration_channels == 16
    assert SolverConfig().max_channels == 16

def test_values_above_global_concurrency_limit_are_rejected(tmp_path):
    paths = write_three_inputs(tmp_path)
    sketch = read_json(paths[1])
    sketch["solver"] = {"max_channels": 17}
    write_json(paths[1], sketch)
    with pytest.raises(InputValidationError, match="max_channels"):
        resolve_inputs(*paths)
```

- [ ] **Step 2: 运行失败测试**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/unit/input/test_models.py \
  tests/unit/input/test_loader.py
```

Expected: FAIL because `vericcl.constants` does not exist or defaults are 32.

- [ ] **Step 3: 实现唯一的软件上限常量**

Create `vericcl/constants.py`:

```python
SOFTWARE_MAX_CONCURRENCY = 16
```

Use it in defaults and loader boundaries:

```python
from vericcl.constants import SOFTWARE_MAX_CONCURRENCY

max_calibration_channels: int = SOFTWARE_MAX_CONCURRENCY
max_channels: int = SOFTWARE_MAX_CONCURRENCY
```

Replace calibration literals with the same constant, while retaining the compatibility name:

```python
MAX_CALIBRATION_CONCURRENCY = SOFTWARE_MAX_CONCURRENCY
```

- [ ] **Step 4: 更新静态 JSON 和文档契约**

For every sketch, set both fields exactly:

```json
"max_calibration_channels": 16
```

```json
"max_channels": 16
```

Do not modify any topo `max_channels`. Update current README/work-document/runtime text from software limit 32 to 16 and the formula to:

```text
k=1..min(max_calibration_channels,16,128MiB/S,link_max_channels)
```

- [ ] **Step 5: 添加 36 个输入的集成合同测试**

```python
def test_all_experiment_inputs_resolve_and_build():
    root = Path(__file__).parents[2]
    atom = root / "vericcl/examples/atom/default.json"
    cases = []
    for topology in sorted((root / "exp/topo").glob("*.json")):
        for sketch in sorted(
            (root / "exp/sketch" / topology.stem).glob("*/*.json")
        ):
            resolved = resolve_inputs(topology, sketch, atom)
            assert resolved.solver.max_channels == 16
            assert resolved.hyperparameters.max_calibration_channels == 16
            build_plan(resolved)
            cases.append((topology, sketch))
    assert len(cases) == 36
```

- [ ] **Step 6: 运行针对性测试**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/unit/input/test_models.py \
  tests/unit/input/test_loader.py \
  tests/unit/online/test_calibration_xml.py \
  tests/unit/online/test_pipeline.py \
  tests/integration/test_experiment_inputs.py
```

Expected: all pass.

- [ ] **Step 7: 提交软件并发契约**

```bash
git add vericcl/constants.py vericcl/input vericcl/verification/online \
  vericcl/cli/online.py vericcl/examples/sketch exp/sketch \
  README.md README.zh-CN.md Vericcl-work-document.md \
  docs/runtime-configuration.md tests/unit/input \
  tests/unit/online tests/integration/test_experiment_inputs.py
git commit -m "feat: cap VeriCCL concurrency at sixteen"
```

---

### Task 4: 使用发送端本地时间计算校准波次

**Files:**
- Modify: `vericcl/verification/online/trace_analysis.py`
- Modify: `vericcl/verification/online/calibration.py`
- Test: `tests/unit/online/test_trace_analysis.py`
- Test: `tests/unit/online/test_calibration.py`

**Interfaces:**
- Consumes: `StepTraceRecord` 的 `transfer_start/transfer_end` 和现有 `ClockAlignment`。
- Produces: `PhysicalTransferInterval.sender_start/sender_end: Optional[AlignedTimestamp]` 与 `sender_start_us/sender_end_us`；校准只使用同源发送端时间。

- [ ] **Step 1: 写端点不确定但发送端时长可用的失败测试**

```python
def test_pair_retains_sender_local_interval():
    send, recv = _pair(
        "x", start_send=10, end_send=20,
        start_recv=100, end_recv=110,
    )
    interval = pair_endpoints(send, recv, _alignment(uncertainty=50.0))
    assert interval.endpoint_order_uncertain is True
    assert interval.sender_start_us == 10.0
    assert interval.sender_end_us == 20.0

def test_calibration_uses_sender_local_wave_when_endpoints_are_uncertain():
    analysis = _calibration_analysis(
        sender_intervals=((0.0, 5.0), (0.5, 4.5)),
        receiver_offset_us=100.0,
        endpoint_order_uncertain=True,
    )
    point = calibration_point_from_trace(
        CalibrationRequest("inter_node", 64 * 1024 * 1024, 2, "float"),
        2,
        analysis,
    )
    assert point.duration_statistics.p95_us == 5.0
```

- [ ] **Step 2: 运行失败测试**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/unit/online/test_trace_analysis.py::test_pair_retains_sender_local_interval \
  tests/unit/online/test_calibration.py::test_calibration_uses_sender_local_wave_when_endpoints_are_uncertain
```

Expected: FAIL because sender fields do not exist or uncertain endpoints are rejected.

- [ ] **Step 3: 扩展物理区间但保持现有全局语义**

Add fields at the end of `PhysicalTransferInterval`:

```python
sender_start: Optional[AlignedTimestamp] = None
sender_end: Optional[AlignedTimestamp] = None

@property
def sender_start_us(self) -> float:
    if self.sender_start is None:
        raise SemanticError("network interval has no sender start")
    return self.sender_start.value_us

@property
def sender_end_us(self) -> float:
    if self.sender_end is None:
        raise SemanticError("network interval has no sender end")
    return self.sender_end.value_us
```

Set both fields in `pair_endpoints`; set both to `None` for local copies. Do not change `physical_start` or `physical_end`.

Update every manually constructed network interval in `tests/unit/online/test_calibration.py` with explicit sender timestamps. For an existing fixture whose physical interval is `(start, end)`, add:

```python
sender_start=AlignedTimestamp(start, 0.0),
sender_end=AlignedTimestamp(end, 0.0),
```

- [ ] **Step 4: 改用发送端波次时间**

Replace calibration wave duration with:

```python
duration = max(
    interval.sender_end_us for interval in wave_intervals
) - min(
    interval.sender_start_us for interval in wave_intervals
)
```

Remove only the calibration-specific rejection of `endpoint_order_uncertain`. Continue rejecting local copies, unknown transfers, missing endpoints, incomplete iterations and non-positive waves.

- [ ] **Step 5: 运行在线 trace 与校准测试**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/unit/online/test_trace_analysis.py \
  tests/unit/online/test_calibration.py \
  tests/unit/online/test_pipeline.py
```

Expected: all pass.

- [ ] **Step 6: 提交同源校准计时**

```bash
git add vericcl/verification/online/trace_analysis.py \
  vericcl/verification/online/calibration.py \
  tests/unit/online/test_trace_analysis.py \
  tests/unit/online/test_calibration.py
git commit -m "fix: measure calibration with sender-local time"
```

---

### Task 5: 为在线上下文注入执行器与 trace collector

**Files:**
- Modify: `vericcl/cli/online.py`
- Test: `tests/unit/cli/test_online.py`

**Interfaces:**
- Consumes: `CommandExecutor` 协议和 `TraceCollector` callable。
- Produces: `representative_calibration_topology(topology, link_class)` 与 `build_online_context_factory(environment, *, executor=None, trace_collector=None)`；默认行为不变。

- [ ] **Step 1: 写默认与注入路径的失败测试**

```python
def test_online_factory_uses_injected_runtime_dependencies(tmp_path):
    executor = FakeExecutor()
    collector = lambda request: object()
    factory = build_online_context_factory(
        _environment(tmp_path),
        executor=executor,
        trace_collector=collector,
    )
    context = factory(*_factory_arguments(tmp_path))
    assert context.executor is executor
    assert context.trace_collector is collector

def test_online_factory_defaults_to_local_dependencies(tmp_path):
    context = build_online_context_factory(_environment(tmp_path))(
        *_factory_arguments(tmp_path)
    )
    assert isinstance(context.executor, SubprocessCommandExecutor)
    assert context.trace_collector is collect_trace_files

def test_representative_calibration_topology_is_public():
    topology = representative_calibration_topology(
        _two_node_topology(), "inter_node"
    )
    assert topology.rank_count == 2
    assert topology.node_membership == {0: 0, 1: 1}
```

- [ ] **Step 2: 运行失败测试**

```bash
.venv/bin/python -m pytest -q tests/unit/cli/test_online.py -k injected
```

Expected: FAIL because keyword arguments are unsupported.

- [ ] **Step 3: 实现依赖选择**

```python
def build_online_context_factory(
    environment: Mapping[str, str] = os.environ,
    *,
    executor=None,
    trace_collector=None,
):
    selected_executor = (
        SubprocessCommandExecutor() if executor is None else executor
    )
    selected_collector = (
        collect_trace_files if trace_collector is None else trace_collector
    )
```

Use `selected_executor` for operator and calibration contexts and assign `selected_collector` to each `OnlineContext`.

Rename the existing private `_representative_topology` function to `representative_calibration_topology`, update its internal call site, and keep its current input validation and two-Rank topology construction unchanged. Task 10 imports this public function for the hardware smoke benchmark.

- [ ] **Step 4: 运行 CLI 与 pipeline 测试**

```bash
.venv/bin/python -m pytest -q \
  tests/unit/cli/test_online.py \
  tests/unit/online/test_pipeline.py
```

- [ ] **Step 5: 提交注入接口**

```bash
git add vericcl/cli/online.py tests/unit/cli/test_online.py
git commit -m "feat: inject online runtime dependencies"
```

---

### Task 6: 实现受限 SSH 文件传输与远程命令执行

**Files:**
- Create: `vericcl/experiments/__init__.py`
- Create: `vericcl/experiments/remote.py`
- Create: `tests/unit/experiments/test_remote.py`

**Interfaces:**
- Consumes: `ProcessRequest`、`ProcessResult`、`CommandExecutor`。
- Produces: `ExperimentPathPolicy`, `SshFileStager`, `SshStagingCommandExecutor`。

- [ ] **Step 1: 写路径越界和 XML 暂存的失败测试**

```python
def test_path_policy_rejects_path_outside_experiment_root(tmp_path):
    policy = ExperimentPathPolicy(tmp_path / "allowed")
    with pytest.raises(SemanticError, match="experiment root"):
        policy.require_allowed(tmp_path / "outside.xml")

def test_remote_executor_stages_xml_and_executes_on_node4(tmp_path):
    delegate = RecordingExecutor()
    stager = RecordingStager()
    executor = SshStagingCommandExecutor(
        delegate=delegate,
        stager=stager,
        remote_host="10.0.0.104",
        path_policy=ExperimentPathPolicy(tmp_path),
    )
    xml = tmp_path / "case.xml"
    xml.write_text("<algo/>", encoding="ascii")
    result = executor.run(_request(xml, tmp_path / "trace/step"))
    assert stager.uploads == ((xml, xml),)
    assert result.returncode == 0
    assert delegate.calls[-1].command[:4] == (
        "ssh", "-o", "BatchMode=yes", "10.0.0.104"
    )
```

- [ ] **Step 2: 运行失败测试**

```bash
.venv/bin/python -m pytest -q tests/unit/experiments/test_remote.py
```

Expected: FAIL because the experiment package does not exist.

- [ ] **Step 3: 实现绝对路径策略**

```python
@dataclass(frozen=True)
class ExperimentPathPolicy:
    root: Path

    def require_allowed(self, value: Path) -> Path:
        root = self.root.resolve()
        candidate = Path(value)
        if not candidate.is_absolute():
            raise SemanticError("experiment path must be absolute")
        path = candidate.resolve()
        if not path.is_relative_to(root):
            raise SemanticError("path is outside the experiment root")
        return path
```

Reject empty host names, NUL/newline characters and any local or remote path outside the experiment root.

- [ ] **Step 4: 实现原子 upload/fetch**

`SshFileStager.upload(local, remote)` executes these argument arrays through the delegate:

```python
("ssh", "-o", "BatchMode=yes", host, "mkdir", "-p", str(remote.parent))
("scp", "-q", str(local), "{}:{}".format(host, temporary))
("ssh", "-o", "BatchMode=yes", host, "mv", "-f", temporary, str(remote))
```

`fetch(remote, local)` downloads to a local temporary sibling, verifies a regular non-empty file, then uses `Path.replace(local)`.

- [ ] **Step 5: 实现远程 executor 的环境白名单**

Forward only:

```python
REMOTE_ENVIRONMENT_NAMES = frozenset({
    "PATH",
    "LD_LIBRARY_PATH",
    "CUDA_VISIBLE_DEVICES",
})
REMOTE_ENVIRONMENT_PREFIXES = ("NCCL_", "MSCCL_", "VERICCL_")
```

Never forward `GRB_`, `GUROBI_`, `WLS`, SSH agent variables or unrelated controller environment. Stage `MSCCL_XML_FILES`, ensure the trace prefix parent exists remotely, then execute:

```python
("ssh", "-o", "BatchMode=yes", remote_host, "env") \
    + tuple("{}={}".format(key, value) for key, value in allowed_items) \
    + request.command
```

Return the delegate's `ProcessResult` unchanged.

- [ ] **Step 6: 运行远程单元测试**

```bash
.venv/bin/python -m pytest -q tests/unit/experiments/test_remote.py
```

Expected: all pass, including upload failure, remote non-zero exit, unsafe path and secret-environment tests.

- [ ] **Step 7: 提交远程执行组件**

```bash
git add vericcl/experiments tests/unit/experiments/test_remote.py
git commit -m "feat: add restricted remote experiment executor"
```

---

### Task 7: 汇集非共享文件系统上的 Rank trace

**Files:**
- Modify: `vericcl/experiments/remote.py`
- Test: `tests/unit/experiments/test_remote.py`

**Interfaces:**
- Consumes: `SshFileStager.fetch`, `TraceCollectionRequest`, `collect_trace_files`。
- Produces: `RemoteTraceCollector(stager, delegate=collect_trace_files)`；每次根据请求 Rank 数选择 node4 的前半 Rank。

- [ ] **Step 1: 写 2×4 前半 Rank 回收的失败测试**

```python
def test_remote_trace_collector_fetches_node4_ranks_before_analysis(tmp_path):
    stager = RecordingStager(create_downloads=True)
    delegated = []
    collector = RemoteTraceCollector(
        stager=stager,
        delegate=lambda request: delegated.append(request) or _trace_result(),
    )
    request = _trace_request(tmp_path, rank_count=8)
    result = collector(request)
    assert tuple(path.name for _, path in stager.downloads) == (
        "step.rank-0.bin", "step.rank-1.bin",
        "step.rank-2.bin", "step.rank-3.bin",
    )
    assert delegated == [request]
    assert result.complete is True

def test_remote_trace_collector_fetches_rank_zero_for_two_rank_calibration(tmp_path):
    stager = RecordingStager(create_downloads=True)
    collector = RemoteTraceCollector(
        stager=stager,
        delegate=lambda request: _trace_result(),
    )
    collector(_trace_request(tmp_path, rank_count=2))
    assert tuple(path.name for _, path in stager.downloads) == (
        "step.rank-0.bin",
    )
```

- [ ] **Step 2: 运行失败测试**

```bash
.venv/bin/python -m pytest -q \
  tests/unit/experiments/test_remote.py::test_remote_trace_collector_fetches_node4_ranks_before_analysis
```

- [ ] **Step 3: 实现严格 Rank 集合检查**

```python
class RemoteTraceCollector:
    def __init__(self, *, stager, delegate=collect_trace_files):
        self._stager = stager
        self._delegate = delegate

    def __call__(self, request: TraceCollectionRequest) -> TraceCollectionResult:
        if request.rank_count < 2 or request.rank_count % 2:
            raise SemanticError("split-host trace rank count must be even")
        remote_ranks = tuple(range(request.rank_count // 2))
        for rank in remote_ranks:
            path = Path("{}.rank-{}.bin".format(request.file_prefix, rank))
            self._stager.fetch(path, path)
        return self._delegate(request)
```

2 Rank 校准回收 `(0,)`；2×4 回收 `(0,1,2,3)`；2×8 回收 `(0,1,2,3,4,5,6,7)`。缺失、空文件或 fetch 失败必须阻止 delegate 执行。

- [ ] **Step 4: 运行远程与标准 collector 测试**

```bash
.venv/bin/python -m pytest -q \
  tests/unit/experiments/test_remote.py \
  tests/unit/online/test_pipeline.py \
  tests/unit/online/test_trace_format.py
```

- [ ] **Step 5: 提交 trace 汇集**

```bash
git add vericcl/experiments/remote.py tests/unit/experiments/test_remote.py
git commit -m "feat: collect split-host MSCCL traces"
```

---

### Task 8: 解析多尺寸性能表并判定 MSCCL 激活

**Files:**
- Modify: `vericcl/verification/online/nccl_tests.py`
- Create: `vericcl/experiments/performance.py`
- Modify: `tests/unit/online/test_nccl_tests.py`
- Create: `tests/unit/experiments/test_performance.py`

**Interfaces:**
- Consumes: nccl-tests 完整 stdout。
- Produces: `parse_nccl_tests_table(text, allow_unchecked=False)`, `ActivationEvidence`, `evaluate_msccl_activation(text, run, threshold=0.05)`。

- [ ] **Step 1: 写多尺寸解析和 5% 判据失败测试**

```python
def test_parse_nccl_tests_table_keeps_every_size():
    rows = parse_nccl_tests_table(_two_size_output())
    assert tuple(row.message_size_bytes for row in rows) == (4194304, 8388608)

def test_activation_requires_info_and_five_percent_busbw_difference():
    run = _run(out_busbw=70.0, in_busbw=75.0)
    evidence = evaluate_msccl_activation(
        "NCCL INFO Connected 1 MSCCL algorithms\n", run
    )
    assert evidence.info_loaded is True
    assert evidence.relative_busbw_difference == pytest.approx(5.0 / 75.0)
    assert evidence.confirmed is True

def test_activation_is_unconfirmed_without_info():
    assert evaluate_msccl_activation("", _run(70.0, 75.0)).confirmed is False
```

- [ ] **Step 2: 运行失败测试**

```bash
.venv/bin/python -m pytest -q \
  tests/unit/online/test_nccl_tests.py \
  tests/unit/experiments/test_performance.py
```

- [ ] **Step 3: 提取通用表解析器**

```python
def parse_nccl_tests_table(
    text: str,
    *,
    allow_unchecked: bool = False,
) -> Tuple[NcclTestRun, ...]:
    if not isinstance(text, str):
        raise SemanticError("nccl-tests output must be a string")
    runs = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = tuple(stripped.split())
        if len(fields) < 7:
            continue
        try:
            int(fields[0], 0)
        except ValueError:
            continue
        runs.append(
            _performance_row(
                fields,
                expected_bytes=None,
                allow_unchecked=allow_unchecked,
            )
        )
    if not runs:
        raise SemanticError("nccl-tests output contains no performance row")
    return tuple(runs)

def parse_nccl_tests_output(text, expected_bytes, *, allow_unchecked=False):
    if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int):
        raise SemanticError("expected_bytes must be a positive integer")
    if expected_bytes < 1:
        raise SemanticError("expected_bytes must be a positive integer")
    rows = parse_nccl_tests_table(text, allow_unchecked=allow_unchecked)
    selected = tuple(
        row for row in rows if row.message_size_bytes == expected_bytes
    )
    if not selected:
        raise SemanticError("nccl-tests output contains no requested size")
    return selected
```

Change `_performance_row` to accept `expected_bytes: Optional[int]`; compare the parsed size only when it is not `None`. The table parser must still reject malformed numeric fields and any nonzero `#wrong`.

- [ ] **Step 4: 实现激活证据**

```python
@dataclass(frozen=True)
class ActivationEvidence:
    info_loaded: bool
    relative_busbw_difference: float
    threshold: float
    confirmed: bool

def evaluate_msccl_activation(text, run, threshold=0.05):
    in_place = run.in_place
    if in_place is None:
        return ActivationEvidence(False, 0.0, threshold, False)
    denominator = max(
        run.out_of_place.bus_bandwidth_gbps,
        in_place.bus_bandwidth_gbps,
    )
    difference = (
        abs(
            in_place.bus_bandwidth_gbps
            - run.out_of_place.bus_bandwidth_gbps
        ) / denominator
        if denominator > 0.0 else 0.0
    )
    loaded = "Connected 1 MSCCL algorithms" in text
    return ActivationEvidence(
        loaded, difference, threshold, loaded and difference >= threshold
    )
```

- [ ] **Step 5: 运行解析测试**

```bash
.venv/bin/python -m pytest -q \
  tests/unit/online/test_nccl_tests.py \
  tests/unit/experiments/test_performance.py
```

- [ ] **Step 6: 提交性能解析**

```bash
git add vericcl/verification/online/nccl_tests.py \
  vericcl/experiments/performance.py \
  tests/unit/online/test_nccl_tests.py \
  tests/unit/experiments/test_performance.py
git commit -m "feat: parse experiment performance evidence"
```

---

### Task 9: 定义 24 个案例与可恢复状态

**Files:**
- Create: `exp/v100-k16-manifest.json`
- Create: `vericcl/experiments/model.py`
- Create: `vericcl/experiments/state.py`
- Create: `tests/unit/experiments/test_model.py`
- Create: `tests/unit/experiments/test_state.py`

**Interfaces:**
- Consumes: topo/sketch/atom 路径及输入解析器。
- Produces: `ExperimentCase`, `ExperimentManifest`, `TaskRecord`, `ExperimentStateStore`, `atomic_replace_text`。

- [ ] **Step 1: 写 24 案例和 size 语义失败测试**

```python
def test_v100_manifest_contains_exact_matrix(repo_root):
    manifest = load_experiment_manifest(
        repo_root / "exp/v100-k16-manifest.json", repo_root=repo_root
    )
    assert len(manifest.cases) == 24
    assert {case.size_label for case in manifest.cases} == {
        "4m", "16m", "64m", "256m", "1g", "2g"
    }
    for case in manifest.cases:
        resolved = resolve_inputs(
            case.topology_path, case.sketch_path, manifest.atom_path
        )
        expected = (
            resolved.rank_count * resolved.hyperparameters.total_size_bytes
            if resolved.collective.kind is CollectiveKind.ALL_GATHER
            else resolved.hyperparameters.total_size_bytes
        )
        assert case.message_size_bytes == expected
        assert resolved.solver.max_channels == 16
```

- [ ] **Step 2: 写原子恢复状态失败测试**

```python
def test_state_store_resumes_only_matching_completed_task(tmp_path):
    store = ExperimentStateStore(tmp_path / "state.json")
    record = TaskRecord(
        task_id="v100-n2g4-ag-4m",
        status=TaskStatus.PASSED,
        input_sha256="a" * 64,
        output_sha256="b" * 64,
        command=("vericcl", "solve"),
        returncode=0,
    )
    store.put(record)
    assert store.reusable(record.task_id, "a" * 64, "b" * 64) is True
    assert store.reusable(record.task_id, "c" * 64, "b" * 64) is False
```

- [ ] **Step 3: 运行失败测试**

```bash
.venv/bin/python -m pytest -q \
  tests/unit/experiments/test_model.py \
  tests/unit/experiments/test_state.py
```

- [ ] **Step 4: 创建精确 manifest**

The JSON contains:

```json
{
  "schema_version": 1,
  "atom": "vericcl/examples/atom/default.json",
  "topologies": ["v100-n2g4", "v100-n2g8"],
  "collectives": ["ag", "ar"],
  "sizes": ["4m", "16m", "64m", "256m", "1g", "2g"]
}
```

`load_experiment_manifest` expands paths deterministically and derives `message_size_bytes` from resolved inputs rather than duplicating byte values in JSON.

- [ ] **Step 5: 实现不可变模型与原子状态写入**

```python
import os
import uuid

@dataclass(frozen=True)
class ExperimentCase:
    task_id: str
    topology_name: str
    collective_label: str
    size_label: str
    topology_path: Path
    sketch_path: Path
    rank_count: int
    message_size_bytes: int
    slice_size_bytes: int

@dataclass(frozen=True)
class ExperimentManifest:
    atom_path: Path
    cases: Tuple[ExperimentCase, ...]

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    status: TaskStatus
    input_sha256: str
    output_sha256: Optional[str]
    command: Tuple[str, ...]
    returncode: Optional[int]
    failure_code: Optional[str] = None
    log_path: Optional[str] = None
    started_at_utc: Optional[str] = None
    finished_at_utc: Optional[str] = None

def atomic_replace_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.{}.tmp".format(path.name, uuid.uuid4().hex))
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()

class ExperimentStateStore:
    def put(self, record: TaskRecord) -> None:
        payload = _replace_record(self.load(), record)
        atomic_replace_text(self.path, canonical_json(payload) + "\n")
```

Validate SHA-256 width, nonnegative return codes, nonempty commands and legal status transitions. A `RUNNING` record from a previous process is not reusable.

- [ ] **Step 6: 运行模型与状态测试**

```bash
.venv/bin/python -m pytest -q \
  tests/unit/experiments/test_model.py \
  tests/unit/experiments/test_state.py \
  tests/integration/test_experiment_inputs.py
```

- [ ] **Step 7: 提交矩阵模型**

```bash
git add exp/v100-k16-manifest.json vericcl/experiments/model.py \
  vericcl/experiments/state.py tests/unit/experiments/test_model.py \
  tests/unit/experiments/test_state.py
git commit -m "feat: define resumable V100 experiment matrix"
```

---

### Task 10: 实现 V100 实验编排器

**Files:**
- Modify: `vericcl/experiments/model.py`
- Create: `vericcl/experiments/v100.py`
- Modify: `tests/unit/experiments/test_model.py`
- Create: `tests/unit/experiments/test_v100.py`

**Interfaces:**
- Consumes: `ExperimentManifest`, `ExperimentStateStore`, `SshStagingCommandExecutor`, `RemoteTraceCollector`, `build_online_context_factory`, `execute_solve`。
- Produces: `V100ExperimentConfig`, `python -m vericcl.experiments.v100 {preflight,smoke,solve,benchmark,summarize,all}`。

- [ ] **Step 1: 写 preflight 禁止错误 hostfile 与路径的失败测试**

```python
def test_preflight_requires_node4_first_and_allowed_root(tmp_path):
    config = _config(tmp_path, hostfile_text=(
        "10.0.0.102 slots=4\n10.0.0.104 slots=4\n"
    ))
    with pytest.raises(SemanticError, match="node4 must be first"):
        preflight(config)

def test_preflight_rejects_ib_enabled(tmp_path):
    config = _config(tmp_path, environment={"NCCL_IB_DISABLE": "0"})
    with pytest.raises(SemanticError, match="NCCL_IB_DISABLE"):
        preflight(config)
```

- [ ] **Step 2: 写 solve 调用完整工作流的失败测试**

```python
def test_solve_case_requests_online_validation_and_tuning(tmp_path):
    calls = []
    result = solve_case(
        _case(tmp_path),
        _config(tmp_path),
        execute=lambda context: calls.append(context) or _workflow_result(),
    )
    assert calls[0].online is True
    assert calls[0].tune is True
    assert calls[0].online_context_factory is not None
    assert result.status is TaskStatus.PASSED
```

- [ ] **Step 3: 运行失败测试**

```bash
.venv/bin/python -m pytest -q tests/unit/experiments/test_v100.py
```

- [ ] **Step 4: 定义完整 config schema**

Add the immutable model:

```python
@dataclass(frozen=True)
class V100ExperimentConfig:
    experiment_root: Path
    repo_root: Path
    manifest_path: Path
    baseline_source: Path
    remote_host: str
    mpi_launcher: Path
    msccl_library_path: Path
    nccl_tests_binary_directory: Path
    clock_sync_binary: Path
    calibration_cache_path: Path
    hostfile_8: Path
    hostfile_16: Path
    environment: Mapping[str, str]
    max_clock_uncertainty_us: float = 50.0
```

`load_v100_config` requires this exact JSON shape:

```json
{
  "schema_version": 1,
  "experiment_root": "/home/zdl/VeriCCL-experiments/2026-09-01-v100-k16",
  "repo_root": "/home/zdl/VeriCCL-experiments/2026-09-01-v100-k16/repo",
  "manifest_path": "/home/zdl/VeriCCL-experiments/2026-09-01-v100-k16/repo/exp/v100-k16-manifest.json",
  "baseline_source": "/home/zdl/MSCCL/test",
  "remote_host": "10.0.0.104",
  "mpi_launcher": "/opt/openmpi-4.1.8/bin/mpirun",
  "msccl_library_path": "/home/zdl/VeriCCL-experiments/2026-09-01-v100-k16/msccl/build/lib",
  "nccl_tests_binary_directory": "/home/zdl/VeriCCL-experiments/2026-09-01-v100-k16/nccl-tests/build",
  "clock_sync_binary": "/home/zdl/VeriCCL-experiments/2026-09-01-v100-k16/bin/vericcl_clock_sync",
  "calibration_cache_path": "/home/zdl/VeriCCL-experiments/2026-09-01-v100-k16/calibration/cache.json",
  "hostfiles": {
    "8": "/home/zdl/VeriCCL-experiments/2026-09-01-v100-k16/hostfile-2x4",
    "16": "/home/zdl/VeriCCL-experiments/2026-09-01-v100-k16/hostfile-2x8"
  },
  "max_clock_uncertainty_us": 50.0,
  "environment": {
    "NCCL_ALGO": "MSCCL,RING",
    "NCCL_IB_DISABLE": "1",
    "NCCL_IGNORE_DISABLED_P2P": "1",
    "NCCL_NET_GDR_LEVEL": "0",
    "NCCL_NET_GDR_READ": "0",
    "NCCL_P2P_LEVEL": "NVL",
    "NCCL_PROTO": "Simple",
    "NCCL_SOCKET_IFNAME": "eno0,enp4s0",
    "VERICCL_CALIBRATION_LINK_CLASS": "inter_node",
    "VERICCL_CUDA_VERSION": "12.8",
    "VERICCL_FORCE_RECALIBRATE": "0",
    "VERICCL_GPU_MODEL": "NVIDIA-V100",
    "VERICCL_MSCCL_VERSION": "vericcl-runtime-v0.1.0",
    "VERICCL_NCCL_VERSION": "2.12.12",
    "VERICCL_NIC_MODEL": "ethernet-10.0.0.0-24",
    "VERICCL_ONLINE_INTER_NODE": "1"
  }
}
```

All experiment-managed paths except the read-only `baseline_source` must be inside `experiment_root`; reject unknown keys and relative paths.

- [ ] **Step 5: 实现严格配置与 hostfile 生成**

```python
HOSTS = {
    8: (("10.0.0.104", 4), ("10.0.0.102", 4)),
    16: (("10.0.0.104", 8), ("10.0.0.102", 8)),
}

def write_hostfile(path: Path, rank_count: int) -> None:
    lines = tuple(
        "{} slots={}".format(host, slots)
        for host, slots in HOSTS[rank_count]
    )
    atomic_replace_text(path, "\n".join(lines) + "\n")
```

Write identical hostfiles to node2 and node4 via the stager. Preflight validates absolute paths, binaries, patched runtime, SSH, rank counts, `NCCL_IB_DISABLE=1`, `NCCL_ALGO=MSCCL,RING`, protocol Simple and 50 us clock threshold.

- [ ] **Step 6: 实现单案例在线 solve**

Build the factory environment without inheriting path decisions:

```python
environment = dict(config.environment)
environment.update({
    "VERICCL_CALIBRATION_CACHE_PATH": str(config.calibration_cache_path),
    "VERICCL_CLOCK_SYNC_BINARY": str(config.clock_sync_binary),
    "VERICCL_MAX_CLOCK_UNCERTAINTY_US": str(
        config.max_clock_uncertainty_us
    ),
    "VERICCL_MPI_HOSTFILE": str(config.hostfile(case.rank_count)),
    "VERICCL_MPI_LAUNCHER": str(config.mpi_launcher),
    "VERICCL_MSCCL_BUILD_DIR": str(config.msccl_library_path),
    "VERICCL_NCCL_TESTS_BUILD_DIR": str(
        config.nccl_tests_binary_directory
    ),
})
```

```python
online_factory = build_online_context_factory(
    environment,
    executor=remote_executor,
    trace_collector=RemoteTraceCollector(
        stager=stager,
    ),
)
result = execute_solve(RunContext(
    topology_path=case.topology_path,
    sketch_path=case.sketch_path,
    atom_path=manifest.atom_path,
    output_base=case_output,
    run_id=case.task_id,
    online=True,
    tune=True,
    timeout_s=10800.0,
    online_context_factory=online_factory,
))
```

Mark PASSED only when `final_xml` exists, is nonempty, final validation exists and selected candidate online status is valid. Preserve logs and mark FAILED on timeout or exception, then continue the remaining matrix.

- [ ] **Step 7: 实现命令行阶段、smoke、案例筛选与 resume**

```text
python -m vericcl.experiments.v100 preflight \
  --config /home/zdl/VeriCCL-experiments/2026-09-01-v100-k16/config.json
python -m vericcl.experiments.v100 smoke \
  --config /home/zdl/VeriCCL-experiments/2026-09-01-v100-k16/config.json
python -m vericcl.experiments.v100 solve \
  --config /home/zdl/VeriCCL-experiments/2026-09-01-v100-k16/config.json --resume
python -m vericcl.experiments.v100 solve \
  --config /home/zdl/VeriCCL-experiments/2026-09-01-v100-k16/config.json \
  --case v100-n2g4-ag-4m --case v100-n2g8-ag-4m --resume
python -m vericcl.experiments.v100 benchmark \
  --config /home/zdl/VeriCCL-experiments/2026-09-01-v100-k16/config.json --resume
python -m vericcl.experiments.v100 summarize \
  --config /home/zdl/VeriCCL-experiments/2026-09-01-v100-k16/config.json
python -m vericcl.experiments.v100 all \
  --config /home/zdl/VeriCCL-experiments/2026-09-01-v100-k16/config.json --resume
```

`smoke` loads the first manifest topology, calls `representative_calibration_topology(topology, "inter_node")`, builds the K=1 inter-node 128 MiB XML with `build_calibration_benchmark`, runs it once through the remote executor, and requires `#wrong=0`. Each `--case` must match one manifest task ID; omission selects all 24 cases. Every phase writes its exact argv, allowlisted environment, start/end UTC time, return code and output hashes to `state.json` and `manifest.json`.

- [ ] **Step 8: 运行编排测试**

```bash
.venv/bin/python -m pytest -q tests/unit/experiments/test_v100.py
```

Expected: all fake-executor tests pass without SSH, Gurobi or GPU access.

- [ ] **Step 9: 提交编排器**

```bash
git add vericcl/experiments/model.py vericcl/experiments/v100.py \
  tests/unit/experiments/test_model.py tests/unit/experiments/test_v100.py
git commit -m "feat: orchestrate V100 schedule experiments"
```

---

### Task 11: 实现 VeriCCL 与基线性能运行

**Files:**
- Modify: `vericcl/experiments/v100.py`
- Modify: `vericcl/experiments/performance.py`
- Modify: `tests/unit/experiments/test_v100.py`
- Modify: `tests/unit/experiments/test_performance.py`

**Interfaces:**
- Consumes: 最终 VeriCCL XML、`/home/zdl/MSCCL/test` 的复制品、MPI 配置。
- Produces: 24 个精确尺寸日志、23 个范围日志以及结构化测量 JSON。

- [ ] **Step 1: 写精确命令构造失败测试**

```python
def test_vericcl_benchmark_uses_exact_size_and_fifteen_iterations():
    command = build_performance_command(
        binary="/tests/all_gather_perf",
        begin="4M", end="4M", factor=2, iterations=15,
    )
    assert command[-10:] == (
        "-b", "4M", "-e", "4M", "-f", "2", "-g", "1", "-n", "15"
    )

def test_baseline_benchmark_uses_full_range():
    command = build_performance_command(
        binary="/tests/all_reduce_perf",
        begin="4M", end="2G", factor=2, iterations=15,
    )
    assert ("-b", "4M", "-e", "2G", "-f", "2") == command[-10:-4]
```

- [ ] **Step 2: 写基线 XML 分类失败测试**

```python
def test_baselines_are_selected_by_xml_contract(tmp_path):
    paths = _write_baseline_contracts(tmp_path)
    selected = select_baselines(paths, collective="allgather", rank_count=16)
    assert tuple(path.name for path in selected) == ("ag-16.xml",)
```

- [ ] **Step 3: 运行失败测试**

```bash
.venv/bin/python -m pytest -q \
  tests/unit/experiments/test_v100.py \
  tests/unit/experiments/test_performance.py
```

- [ ] **Step 4: 实现统一 MPI 前缀**

Define the performance result boundary in `vericcl/experiments/performance.py`:

```python
class XmlSource(str, Enum):
    VERICCL = "vericcl"
    BASELINE = "baseline"

@dataclass(frozen=True)
class PerformanceResult:
    task_id: str
    topology_name: str
    collective_label: str
    source: XmlSource
    xml_name: str
    runs: Tuple[NcclTestRun, ...]
    activation: Tuple[ActivationEvidence, ...]
    stdout_path: Path
    stderr_path: Path
```

Require one activation entry per run and matching stable size order.

The prefix must be exactly equivalent to:

```text
/opt/openmpi-4.1.8/bin/mpirun --allow-run-as-root
--prefix /opt/openmpi-4.1.8
-np 8 --hostfile /home/zdl/VeriCCL-experiments/2026-09-01-v100-k16/hostfile-2x4
-mca pml ob1
-mca btl tcp,self,vader
-mca btl_vader_single_copy_mechanism none
-mca btl_tcp_if_include 10.0.0.0/24
```

For 2×8, replace only the process and hostfile pair with:

```text
-np 16 --hostfile /home/zdl/VeriCCL-experiments/2026-09-01-v100-k16/hostfile-2x8
```

Export the approved PATH, LD_LIBRARY_PATH, NCCL, CUDA and MSCCL variables only. Do not export IB HCA/GID values because `NCCL_IB_DISABLE=1`.

- [ ] **Step 5: 实现 XML 特定 buffer 设置**

```python
buffsize = (
    2 * case.slice_size_bytes
    if source is XmlSource.VERICCL
    else 2 * 1024 * 1024
)
```

Set `NCCL_ALGO=MSCCL,RING`, `NCCL_PROTO=Simple`, `NCCL_DEBUG=INFO`, `NCCL_P2P_LEVEL=NVL`, `NCCL_IGNORE_DISABLED_P2P=1`, `NCCL_SOCKET_IFNAME=eno0,enp4s0`, `NCCL_IB_DISABLE=1`, `NCCL_NET_GDR_LEVEL=0`, `NCCL_NET_GDR_READ=0` and topology-specific `CUDA_VISIBLE_DEVICES`.

- [ ] **Step 6: 保存原始与结构化结果**

For every run, write:

```text
command.txt
environment.json
stdout.log
stderr.log
measurements.json
activation.json
```

`stdout.log` and `stderr.log` are never truncated. `measurements.json` includes both placements, while later ranking reads only `in_place.algorithm_bandwidth_gbps`.

- [ ] **Step 7: 运行性能模块测试**

```bash
.venv/bin/python -m pytest -q \
  tests/unit/experiments/test_v100.py \
  tests/unit/experiments/test_performance.py
```

- [ ] **Step 8: 提交性能运行支持**

```bash
git add vericcl/experiments/v100.py \
  vericcl/experiments/performance.py \
  tests/unit/experiments/test_v100.py \
  tests/unit/experiments/test_performance.py
git commit -m "feat: benchmark VeriCCL and baseline XMLs"
```

---

### Task 12: 生成 in-place `algbw` 汇总报告

**Files:**
- Modify: `.gitignore`
- Create: `vericcl/experiments/report.py`
- Create: `tests/unit/experiments/test_report.py`

**Interfaces:**
- Consumes: state、validation JSON、measurements、activation evidence。
- Produces: `summary/results.csv`, `summary/results.json`, `summary/report.md`。

- [ ] **Step 1: 写未确认激活不得参与结论的失败测试**

```python
def test_report_excludes_unconfirmed_activation_from_comparison(tmp_path):
    rows = build_report_rows((_measurement(confirmed=False, algbw=80.0),))
    assert rows[0].inplace_algbw_gbps == 80.0
    assert rows[0].eligible_for_comparison is False
    assert rows[0].relative_improvement is None
```

- [ ] **Step 2: 写最佳基线与相对提升失败测试**

```python
def test_report_compares_vericcl_with_best_confirmed_baseline():
    rows = build_report_rows((
        _measurement(source="vericcl", confirmed=True, algbw=90.0),
        _measurement(source="baseline-a", confirmed=True, algbw=75.0),
        _measurement(source="baseline-b", confirmed=True, algbw=80.0),
    ))
    selected = next(row for row in rows if row.source == "vericcl")
    assert selected.baseline_inplace_algbw_gbps == 80.0
    assert selected.relative_improvement == pytest.approx(0.125)
```

- [ ] **Step 3: 运行失败测试**

```bash
.venv/bin/python -m pytest -q tests/unit/experiments/test_report.py
```

- [ ] **Step 4: 实现稳定排序与完整字段**

Define:

```python
@dataclass(frozen=True)
class ReportRow:
    topology: str
    collective: str
    size_bytes: int
    source: str
    xml_name: str
    inplace_algbw_gbps: float
    out_of_place_busbw_gbps: float
    in_place_busbw_gbps: float
    busbw_relative_difference: float
    msccl_activation: str
    wrong_count: int
    selected_k: Optional[int]
    solver_status: Optional[str]
    mip_gap: Optional[float]
    offline_validation: str
    online_validation: str
    tuning_strategy: str
    eligible_for_comparison: bool
    baseline_inplace_algbw_gbps: Optional[float]
    relative_improvement: Optional[float]
```

CSV/JSON fields are:

```text
topology,collective,size_bytes,source,xml_name,inplace_algbw_gbps,
out_of_place_busbw_gbps,in_place_busbw_gbps,busbw_relative_difference,
msccl_activation,wrong_count,selected_k,solver_status,mip_gap,
offline_validation,online_validation,tuning_strategy,
eligible_for_comparison,baseline_inplace_algbw_gbps,relative_improvement
```

Sort by topology, collective, size, source and XML name. Preserve negative improvements. Filter baseline range output to the six requested sizes `{4 MiB, 16 MiB, 64 MiB, 256 MiB, 1 GiB, 2 GiB}` before comparison; retain all extra factor-two rows only in raw logs and `measurements.json`.

- [ ] **Step 5: 生成 Markdown 限制说明**

The report explicitly states:

```text
Primary metric: in-place algorithm bandwidth (GB/s).
Network path: TCP/Ethernet because node2 IB was unavailable and unchanged.
Optimality: best accepted K<=16 candidate; not globally proven unless proven_optimal=true.
Activation: NCCL INFO evidence plus at least 5% in/out busbw difference.
```

It also lists failed cases, unconfirmed XMLs, clock uncertainty, `tuning_eligible`, calibration cache hits and every final XML's tuning strategy.

Add this exact exception after the global `*.log` rule so required raw experiment logs are trackable:

```gitignore
!exp/results/**/*.log
```

- [ ] **Step 6: 运行报告测试**

```bash
.venv/bin/python -m pytest -q tests/unit/experiments/test_report.py
```

- [ ] **Step 7: 提交报告生成器**

```bash
git add .gitignore vericcl/experiments/report.py \
  tests/unit/experiments/test_report.py
git commit -m "feat: report in-place V100 performance"
```

---

### Task 13: 补充可复现运行文档

**Files:**
- Create: `docs/experiments/v100-k16.md`
- Modify: `README.md`
- Modify: `README.zh-CN.md`
- Modify: `docs/runtime-configuration.md`
- Modify: `Vericcl-work-document.md`
- Test: `tests/unit/cli/test_solve.py`

**Interfaces:**
- Consumes: Task 10 的 CLI 和固定实验路径。
- Produces: node2 控制端可直接复制执行的命令；中英文 README 保持命令一致。

- [ ] **Step 1: 写文档命令合同失败测试**

```python
def test_v100_experiment_document_uses_safe_fixed_contract():
    text = Path("docs/experiments/v100-k16.md").read_text(encoding="utf-8")
    assert "NCCL_IB_DISABLE=1" in text
    assert "--config" in text
    assert "--resume" in text
    assert "/home/zdl/VeriCCL-experiments/2026-09-01-v100-k16" in text
    assert "/home/cc" not in text
```

- [ ] **Step 2: 运行失败测试**

```bash
.venv/bin/python -m pytest -q tests/unit/cli/test_solve.py -k v100
```

- [ ] **Step 3: 写 node2 控制端命令**

Document these concrete commands:

```bash
export VERICCL_EXPERIMENT_ROOT=/home/zdl/VeriCCL-experiments/2026-09-01-v100-k16
cd "$VERICCL_EXPERIMENT_ROOT/repo"
.venv/bin/python -m vericcl.experiments.v100 preflight \
  --config "$VERICCL_EXPERIMENT_ROOT/config.json"
.venv/bin/python -m vericcl.experiments.v100 all \
  --config "$VERICCL_EXPERIMENT_ROOT/config.json" --resume
```

Explain every path and the fact that MPI is remotely launched on node4.

- [ ] **Step 4: 写 exact nccl-tests 配置**

Document VeriCCL exact-size commands and baseline `-b 4M -e 2G -f 2 -g 1 -n 15`. Explain that final tables retain both placements but conclusions use in-place `algbw` only.

- [ ] **Step 5: 更新 K=16 与非共享 trace 文档**

Ensure README, runtime configuration and work document all state:

```text
K_effective=min(16,max_calibration_channels,128MiB/S,link_max_channels)
```

and explain node4 Rank trace collection without introducing system-modification instructions. Do not name external synthesizer projects in README.

- [ ] **Step 6: 运行文档测试并提交**

```bash
.venv/bin/python -m pytest -q \
  tests/unit/cli/test_solve.py \
  tests/unit/cli/test_online.py
git diff --check
git add docs/experiments/v100-k16.md README.md README.zh-CN.md \
  docs/runtime-configuration.md Vericcl-work-document.md \
  tests/unit/cli/test_solve.py
git commit -m "docs: document V100 experiment workflow"
```

---

### Task 14: 完成本地全量验证并推送实验分支

**Files:**
- Verify: all code and test files from Tasks 3–13

**Interfaces:**
- Consumes: 完整本地实现。
- Produces: 可在 node2 clone 的远程功能分支。

- [ ] **Step 1: 运行完整测试**

```bash
.venv/bin/python -m pytest -q
```

Expected: all non-hardware tests pass; hardware tests skip unless explicitly enabled.

- [ ] **Step 2: 运行静态与输入检查**

```bash
git diff --check
.venv/bin/python -m compileall -q vericcl tests
if rg -n '[^\x00-\x7F]' vericcl tests \
  --glob '*.py' --glob '*.cu' --glob '*.cc' --glob '*.h' --glob '*.sh'; \
then exit 1; fi
.venv/bin/python -m pytest -q tests/integration/test_experiment_inputs.py
```

Expected: checks pass and ASCII scan has no output.

- [ ] **Step 3: 检查提交边界与分支差异**

```bash
git status --short
git log --oneline feature/vericcl-implementation..HEAD
git diff --stat feature/vericcl-implementation...HEAD
```

Expected: working tree clean; branch diff is limited to the approved design, plan, implementation, tests, docs and experiment inputs.

- [ ] **Step 4: 推送功能分支**

```bash
git push -u origin feature/vericcl-scalable-hierarchical-solving
```

Expected: remote branch points at verified HEAD.

---

### Task 15: 在 node2/node4 准备隔离运行环境

**Files:**
- Create remote: `/home/zdl/VeriCCL-experiments/2026-09-01-v100-k16/config.json`
- Create remote: `/home/zdl/VeriCCL-experiments/2026-09-01-v100-k16/hostfile-2x4`
- Create remote: `/home/zdl/VeriCCL-experiments/2026-09-01-v100-k16/hostfile-2x8`
- Create remote: isolated VeriCCL checkout and MSCCL/nccl-tests runtime copies under the experiment root
- Preserve remote: `/home/zdl/MSCCL/test`

**Interfaces:**
- Consumes: Task 14 remote branch and existing `/home/zdl/MSCCL` sources/builds.
- Produces: 两台机器相同绝对路径的 patched runtime、tests、hostfiles 和实验目录。

- [ ] **Step 1: 只读记录当前状态**

Run from the local controller for both hosts:

```bash
ssh node2 'hostname; nvidia-smi -L; ip -br addr; git --version'
ssh node4 'hostname; nvidia-smi -L; ip -br addr; git --version'
```

Expected: 只读取状态；node2 `eno0=10.0.0.102/24`，node4 `enp4s0=10.0.0.104/24`；不执行任何 link/config command.

- [ ] **Step 2: 创建相同实验根并 clone 功能分支**

Run on node2:

```bash
mkdir -p /home/zdl/VeriCCL-experiments/2026-09-01-v100-k16
cd /home/zdl/VeriCCL-experiments/2026-09-01-v100-k16
git clone --branch feature/vericcl-scalable-hierarchical-solving \
  https://github.com/SlienceZDL/VeriCCL.git repo
```

Expected: both `repo` checkouts have the same HEAD. If the directory already exists, verify its remote and use `git fetch` plus fast-forward only; never reset it.

- [ ] **Step 3: 建立 node2 Python 环境**

Run on node2:

```bash
cd /home/zdl/VeriCCL-experiments/2026-09-01-v100-k16/repo
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -e . -r requirements-dev.txt
.venv/bin/python -c 'import gurobipy as gp; m=gp.Model(); m.Params.OutputFlag=0; x=m.addVar(vtype=gp.GRB.BINARY); m.setObjective(x, gp.GRB.MAXIMIZE); m.optimize(); assert m.Status == gp.GRB.OPTIMAL'
```

Expected: license ID 2802355 initializes and the one-variable model is optimal. Do not run Gurobi on node4.

- [ ] **Step 4: 在两台机器创建并 patch 隔离的 MSCCL 源码**

Run separately on node2 and node4:

```bash
export EXP=/home/zdl/VeriCCL-experiments/2026-09-01-v100-k16
mkdir -p "$EXP/build-logs" "$EXP/bin"
cp -a /home/zdl/MSCCL/MSCCL_TIME "$EXP/msccl"
patch -d "$EXP/msccl" -p1 < \
  "$EXP/repo/runtime/msccl-trace/patches/0001-vericcl-fixed-step-trace.patch"
patch -d "$EXP/msccl" -p1 < \
  "$EXP/repo/runtime/msccl-trace/patches/0002-vericcl-host-step-signature.patch"
ssh -o BatchMode=yes 10.0.0.104 "mkdir -p '$EXP'"
scp -rq "$EXP/msccl" 10.0.0.104:"$EXP/msccl"
```

On node2 verify with the venv:

```bash
"$EXP/repo/.venv/bin/python" \
  "$EXP/repo/runtime/msccl-trace/tools/verify_patch.py" \
  --source-dir "$EXP/msccl"
```

On node4 verify with the system Python because Gurobi is not needed:

```bash
/usr/bin/python3 "$EXP/repo/runtime/msccl-trace/tools/verify_patch.py" \
  --source-dir "$EXP/msccl"
```

Expected: both patch applications and both verifiers succeed. If an isolated directory already exists, do not overwrite it; compare its source hashes with the manifest and reuse only an exact verified match.

- [ ] **Step 5: 在两台机器构建 V100 runtime、clock helper 与 nccl-tests**

Run separately on node2 and node4:

```bash
set -o pipefail
export EXP=/home/zdl/VeriCCL-experiments/2026-09-01-v100-k16
export CUDA_HOME=/usr/local/cuda-12.8
export NVCC_GENCODE='-gencode=arch=compute_70,code=sm_70'
make -C "$EXP/msccl" -j "$(nproc)" \
  CUDA_HOME="$CUDA_HOME" NVCC_GENCODE="$NVCC_GENCODE" \
  2>&1 | tee "$EXP/build-logs/msccl.log"
cp -a /home/zdl/MSCCL/nccl-tests "$EXP/nccl-tests"
make -C "$EXP/nccl-tests" -j "$(nproc)" MPI=1 \
  MPI_HOME=/opt/openmpi-4.1.8 CUDA_HOME="$CUDA_HOME" \
  NCCL_HOME="$EXP/msccl/build" NVCC_GENCODE="$NVCC_GENCODE" \
  2>&1 | tee "$EXP/build-logs/nccl-tests.log"
PATH=/opt/openmpi-4.1.8/bin:"$CUDA_HOME/bin":/usr/bin:/bin \
  "$CUDA_HOME/bin/nvcc" -ccbin /opt/openmpi-4.1.8/bin/mpicxx \
  -O2 -std=c++11 -gencode=arch=compute_70,code=sm_70 \
  "$EXP/repo/runtime/msccl-trace/tools/vericcl_clock_sync.cu" \
  -o "$EXP/bin/vericcl_clock_sync" \
  2>&1 | tee "$EXP/build-logs/clock-sync.log"
test -f "$EXP/msccl/build/lib/libnccl.so"
test -x "$EXP/nccl-tests/build/all_gather_perf"
test -x "$EXP/nccl-tests/build/all_reduce_perf"
test -x "$EXP/bin/vericcl_clock_sync"
```

Do not modify `/home/zdl/MSCCL/test`, system libraries or CUDA/OpenMPI installations.

- [ ] **Step 6: 写相同 hostfile 与 config**

`hostfile-2x4`:

```text
10.0.0.104 slots=4
10.0.0.102 slots=4
```

`hostfile-2x8`:

```text
10.0.0.104 slots=8
10.0.0.102 slots=8
```

`config.json` contains absolute paths, `remote_host=node4`, `max_clock_uncertainty_us=50`, `NCCL_IB_DISABLE=1`, the manifest path and baseline source. Copy the three files to the same paths on node4.

- [ ] **Step 7: 运行 runtime patch 与远程路径预检**

Run on node2:

```bash
cd /home/zdl/VeriCCL-experiments/2026-09-01-v100-k16/repo
.venv/bin/python -m vericcl.experiments.v100 preflight \
  --config /home/zdl/VeriCCL-experiments/2026-09-01-v100-k16/config.json
```

Expected: preflight passes without changing hardware or system configuration.

---

### Task 16: 执行硬件 smoke、24 案例合成与在线验证

**Files:**
- Create remote results under: `/home/zdl/VeriCCL-experiments/2026-09-01-v100-k16/vericcl/`
- Create remote logs under: `/home/zdl/VeriCCL-experiments/2026-09-01-v100-k16/logs/`
- Create remote cache under: `/home/zdl/VeriCCL-experiments/2026-09-01-v100-k16/calibration/`

**Interfaces:**
- Consumes: prepared runtime, 24-case manifest, K<=16 solver and 144-point calibration cache。
- Produces: 24 个最终 XML 或明确失败记录、离线/在线验证、BDD/在线调优证据。

- [ ] **Step 1: 运行 2 Rank 跨节点 smoke**

Run on node2:

```bash
cd /home/zdl/VeriCCL-experiments/2026-09-01-v100-k16/repo
.venv/bin/python -m vericcl.experiments.v100 smoke \
  --config /home/zdl/VeriCCL-experiments/2026-09-01-v100-k16/config.json
```

Expected: the K=1 calibration XML runs at 128 MiB with one Rank per node; return code 0, `#wrong=0`, and MSCCL load evidence are recorded.

- [ ] **Step 2: 运行 2×4 与 2×8 单案例 smoke**

Run on node2:

```bash
cd /home/zdl/VeriCCL-experiments/2026-09-01-v100-k16/repo
.venv/bin/python -m vericcl.experiments.v100 solve \
  --config /home/zdl/VeriCCL-experiments/2026-09-01-v100-k16/config.json \
  --case v100-n2g4-ag-4m --case v100-n2g8-ag-4m --resume
```

Expected: both final XMLs are nonempty, all offline dimensions are valid, online status is valid, and trace rank counts 8/16 are complete.

- [ ] **Step 3: 执行完整 solve 阶段**

Run on node2 in a resumable PTY session:

```bash
cd /home/zdl/VeriCCL-experiments/2026-09-01-v100-k16/repo
.venv/bin/python -m vericcl.experiments.v100 solve \
  --config /home/zdl/VeriCCL-experiments/2026-09-01-v100-k16/config.json \
  --resume
```

Expected: process continues through independent failures, periodically flushes state, and never treats an empty directory as success.

- [ ] **Step 4: 核对校准覆盖与缓存复用**

Run:

```bash
.venv/bin/python -m vericcl.experiments.v100 summarize \
  --config /home/zdl/VeriCCL-experiments/2026-09-01-v100-k16/config.json
```

Expected: at most 144 distinct inter-node points; stable AG/AR-equivalent signatures are cache hits; no `k>16`; intra-node curves remain topo inputs.

- [ ] **Step 5: 核对每个最终 XML 的强制验证**

Run:

```bash
cd /home/zdl/VeriCCL-experiments/2026-09-01-v100-k16/repo
.venv/bin/python -m vericcl.experiments.v100 summarize \
  --config /home/zdl/VeriCCL-experiments/2026-09-01-v100-k16/config.json
```

Require exit code 0. The audit must assert for every PASSED case that validation JSON reports valid semantic, state, topology, resource, timing, buffer, endpoint, deadlock and XML dimensions; `#wrong=0`; tuning strategy recorded. Cases with uncertain trace may retain XML but must report `tuning_eligible=false`.

---

### Task 17: 执行 47 次主性能运行与结果汇总

**Files:**
- Create remote: copied baseline XMLs under experiment root
- Create remote: all raw performance logs
- Create remote: `summary/results.csv`, `summary/results.json`, `summary/report.md`

**Interfaces:**
- Consumes: Task 16 final XMLs and 23 baseline XML copies。
- Produces: 24 VeriCCL exact-size runs、23 baseline range runs、MSCCL activation evidence and final in-place report。

- [ ] **Step 1: 复制并哈希 23 个基线 XML**

Copy from `/home/zdl/MSCCL/test` to the experiment root without modifying the source. Parse every root contract and assert the inventory is:

```text
2x4 allgather: 5
2x4 allreduce: 1
2x8 allgather: 9
2x8 allreduce: 8
total: 23
```

Save source path and SHA-256 for every XML.

- [ ] **Step 2: 执行 benchmark 阶段**

Run on node2:

```bash
cd /home/zdl/VeriCCL-experiments/2026-09-01-v100-k16/repo
.venv/bin/python -m vericcl.experiments.v100 benchmark \
  --config /home/zdl/VeriCCL-experiments/2026-09-01-v100-k16/config.json \
  --resume
```

Expected: 24 VeriCCL exact-size tasks and 23 baseline range tasks have raw logs or explicit FAILED records.

- [ ] **Step 3: 验证性能日志完整性**

For every successful run, verify:

```text
returncode=0
#wrong=0 for both placements
in-place algbw is present
out-of-place busbw is present
in-place busbw is present
NCCL INFO load evidence is recorded
```

Do not delete unconfirmed activation results; mark them ineligible for comparative conclusions.

- [ ] **Step 4: 生成最终报告**

```bash
.venv/bin/python -m vericcl.experiments.v100 summarize \
  --config /home/zdl/VeriCCL-experiments/2026-09-01-v100-k16/config.json
```

Expected: report ranks only confirmed in-place `algbw`, includes negative improvements, records Ethernet limitation, optimality scope, selected K and tuning strategy.

---

### Task 18: 回收结果、最终验证、合并并推送

**Files:**
- Add: `exp/results/2026-09-01-v100-k16/manifest.json`
- Add: `exp/results/2026-09-01-v100-k16/calibration/`
- Add: `exp/results/2026-09-01-v100-k16/vericcl/`
- Add: `exp/results/2026-09-01-v100-k16/baselines/`
- Add: `exp/results/2026-09-01-v100-k16/logs/`
- Add: `exp/results/2026-09-01-v100-k16/summary/`
- Create: `tests/integration/test_v100_results.py`

**Interfaces:**
- Consumes: node2 完整实验目录和所有本地验证。
- Produces: 默认 clone 可获得的最终代码、XML、日志、结构化数据和报告。

- [ ] **Step 1: 从 node2 回收可提交结果**

Copy only the approved result tree to the feature worktree. Keep large binary step traces remote and record their path, size and SHA-256 in `manifest.json`. Do not copy caches containing secrets or full process environments.

- [ ] **Step 2: 验证回收清单与无敏感信息**

Run:

```bash
git status --short
git diff --check
if rg -n 'WLSSecret|WLSAccessID|PRIVATE KEY|SSH_AUTH_SOCK|GRB_LICENSE_FILE' \
  exp/results/2026-09-01-v100-k16; then exit 1; fi
```

Expected: secret scan has no output; result paths match the design.

- [ ] **Step 3: 重新运行完整代码验证**

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q vericcl tests
if rg -n '[^\x00-\x7F]' vericcl tests \
  --glob '*.py' --glob '*.cu' --glob '*.cc' --glob '*.h' --glob '*.sh'; \
then exit 1; fi
```

Expected: all tests pass and ASCII scan has no output.

- [ ] **Step 4: 验证每个结果与哈希绑定**

Create:

```python
import hashlib
import json
from pathlib import Path

from vericcl.verification.online.nccl_tests import parse_nccl_tests_table


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v100_result_manifest_binds_every_case():
    repo_root = Path(__file__).parents[2]
    root = repo_root / "exp/results/2026-09-01-v100-k16"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    cases = manifest["cases"]
    assert len(cases) == 24
    for case in cases:
        if case["status"] == "passed":
            xml_path = root / case["xml_path"]
            validation_path = root / case["validation_path"]
            assert _sha256_file(xml_path) == case["xml_sha256"]
            validation = json.loads(validation_path.read_text(encoding="utf-8"))
            assert validation["accepted"] is True
            assert case["performance_log_path"]
            parse_nccl_tests_table(
                (root / case["performance_log_path"]).read_text(encoding="utf-8")
            )
        else:
            assert case["failure_code"]
            assert case["failure_log_path"]
```

Run:

```bash
.venv/bin/python -m pytest -q tests/integration/test_v100_results.py
```

Expected: every present XML SHA-256, validation JSON and successful raw nccl-tests log parses; every failed case has a nonempty failure code and log path.

- [ ] **Step 5: 提交实验结果**

```bash
git add exp/results/2026-09-01-v100-k16 \
  tests/integration/test_v100_results.py
git commit -m "results: add V100 schedule evaluation"
```

- [ ] **Step 6: 合并到默认功能分支并推送**

In `/Users/zdl/work/code/VeriCCL`, preserve the main worktree before merging:

```bash
git status --short
git stash push -u -m "preserve-user-work-before-v100-results-20260902"
git status --short
```

Then run:

```bash
git switch feature/vericcl-implementation
git merge --ff-only feature/vericcl-scalable-hierarchical-solving
.venv/bin/python -m pytest -q
git push origin feature/vericcl-implementation
```

Restore without dropping the backup first:

```bash
git stash apply 'stash@{0}'
git status --short
```

If restoration conflicts, stop further Git operations, keep the stash, compare the three index stages, and use `apply_patch` to preserve both the verified branch content and the user's README/论文 changes. Drop the stash only after the restored files are verified.

Expected: remote `feature/vericcl-implementation` points to the fully verified result commit.

- [ ] **Step 7: 确认默认 clone 内容**

Run in a new temporary directory:

```bash
git clone https://github.com/SlienceZDL/VeriCCL.git vericcl-clone-check
cd vericcl-clone-check
git branch --show-current
test -f exp/results/2026-09-01-v100-k16/summary/report.md
```

Expected: clone directly checks out the complete branch and the report exists without branch switching. If GitHub default branch does not point to `feature/vericcl-implementation`, update the repository default branch through the already-authorized repository setting before repeating this check.
