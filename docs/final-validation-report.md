# VeriCCL Scalable Hierarchical Solving Validation

Validation date: 2026-09-01

Branch: `feature/vericcl-scalable-hierarchical-solving`

Validation host: macOS 26.6.2 arm64, Python 3.9.6
Gurobi: 12.0.3, restricted non-production license expiring 2026-11-23

## Result

The scalable template route pipeline passed the focused acceptance suite, all
non-hardware unit/property/integration/end-to-end tests, and the dedicated
Gurobi tests. The tested two-node gateway AllGather and AllReduce paths
completed planning, solving, route composition, offline verification, buffer
planning, XML lowering, XML validation, and XML readback.

The default template backend is a restricted composition strategy. Its
candidates report `template_route_composition`, never report global optimality,
and remain subject to the complete offline verification pipeline. Strict
optimality requests use only the legacy full-time MILP backend.

## Test Evidence

| Scope | Command | Measured result |
|---|---|---|
| Task 9 focused acceptance | `.venv/bin/python -m pytest tests/integration/test_scalable_template_solving.py tests/e2e/test_hierarchical_allgather.py tests/e2e/test_hierarchical_allreduce.py tests/e2e/test_reproducibility.py tests/e2e/test_candidate_xml.py tests/unit/verification/test_semantics.py tests/unit/verification/test_constraints.py tests/unit/verification/test_bdd_flow.py -q` | 64 passed in 30.57s |
| All non-hardware tests | `.venv/bin/python -m pytest tests/unit tests/property tests/integration tests/e2e -q` | 1341 passed, 1 skipped in 67.92s |
| Gurobi tests | `.venv/bin/python -m pytest tests/gurobi -q` | 37 passed in 0.35s |
| Documented commands | `.venv/bin/python -m pytest tests/integration/test_documented_commands.py -q` | 29 passed in 6.13s |
| Python compilation | `.venv/bin/python -m compileall -q vericcl tests` | Exit code 0 |
| CLI startup | `.venv/bin/python -m vericcl --help` | Exit code 0; `solve` and `verify` subcommands listed |
| Whitespace validation | `git diff --check` | Exit code 0; no errors |
| Source character scan | CJK `U+4E00..U+9FA5` scan over `vericcl` and `tests` | No matches |

The single non-hardware skip was
`tests/unit/online/test_runtime_patch.py::test_patch_dry_run_and_post_apply_source_scan`:
the MSCCL reference source tree was not available on the validation host.

## Structural Scaling Evidence

### Direct eight-rank AllGather

For 128 slices per rank, the planner produced 1024 solver problems and 1024
routing units. Exact deduplication produced eight templates, one per fixed
source/root, with 128 members per template. At fixed `K=1` and one objective,
the route work queue started eight representative models rather than 1024.

### Two-node gateway hierarchy

For the bundled two-node gateway topology:

- Hierarchical AllGather used five PlanNodes and nineteen templates for both one
  and eight slices per rank. The template count did not grow with the number of
  logical positions.
- Hierarchical AllReduce used six PlanNodes and seven templates for 8, 16, 64,
  and 128 slices per rank.
- At `K=1` with the latency objective, each gateway ReduceScatter or
  AllGather representative had 7 variables, 22 linear constraints, and 0
  general constraints. Each local Reduce or AllGather representative had 15
  variables, 71 linear constraints, and 0 general constraints. These counts
  were identical for all four tested slice counts.

### Real experiment input

The files `exp/topo/v100-n2g4.json` and
`exp/sketch/v100-n2g4/ag/ag-1g.json` were read from the main checkout without
modification. Only input resolution, topology loading, planning, solver-problem
construction, template construction, and one representative model build were
performed. No model optimization or full solve was started.

Measured structure:

| Field | Value |
|---|---:|
| Rank count | 8 |
| Slice count per rank | 128 |
| Planning mode | `direct` |
| Requested problem count | 1024 |
| Routing unit count | 1024 |
| Template count | 8 |
| Template member count | 1024 |
| Members in the inspected representative | 128 |
| Representative variables (`K=1`, latency) | 533 |
| Representative linear constraints | 865 |
| Representative general constraints | 0 |

The template count is therefore independent of the 128 logical positions in
this input.

## Semantic, Resource, BDD, and XML Evidence

The gateway AllGather and AllReduce end-to-end fixtures used eight ranks and
eight slices per rank with the constructive template route backend. The tests
verified:

- exact final contributor sets and output offsets;
- complete stage dependencies without a stage barrier;
- non-overlap on each directed-link/channel lane while allowing opposite
  directions and distinct channels to overlap;
- separate unidirectional send and receive/reduce threadblocks;
- valid `s`, `r`, and `rrc` lowering, including a send dependency on the final
  `rrc` step;
- valid `depid` and `deps` coordinates after XML readback;
- valid buffer offsets, `cnt=1`, channel identifiers, and paired endpoints;
- a deadlock-free endpoint program;
- BDD flow identities derived from instantiated slice, stage, Rank, lane, and
  timing data rather than template identifiers;
- deterministic sidecar, canonical report sections, and XML content for both
  direct AllReduce and gateway AllGather reruns with `solver_seed=0`.

## Optimality and Hardware Limitations

The default route-template pipeline is not a global optimality proof. It
reuses only route structure, then applies deterministic global scheduling and
full validation. A tested strict AllReduce request bypassed all template route
models and started four full-time MILP models. Because independently solved
PlanNodes do not prove globally optimal composition, the request correctly
returned `optimality proof was required but not obtained` instead of returning
a restricted candidate as proven optimal.

No fixed solver-time or throughput claim is made. The structural checks above
measure model and template counts only. The real experiment input was not
fully solved.

`tests/hardware` was not run. The validation host did not provide the target
GPU/NIC topology, MSCCL runtime, `nccl-tests`, MPI execution environment, or
online trace toolchain. The software and Gurobi results do not constitute GPU
execution, online calibration, or measured communication-performance evidence.
