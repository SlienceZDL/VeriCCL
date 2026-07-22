# VeriCCL Installation and Usage README Design

## 1. Goal

Expand `README.md` and add `README.zh-CN.md` as reproducible English and
Chinese installation, build, input preparation, solving, verification, and
MSCCL execution guides for Ubuntu 22.04 and Ubuntu 24.04 servers.

Both documents must carry the same technical content and commands. Each file
links to the other at the beginning; changes to version pins, environment
variables, examples, or expected results must update both files together.

The guide must support two deployment levels:

1. Offline solving and verification with Python and Gurobi.
2. Online calibration, per-step tracing, and XML execution with CUDA, the
   VeriCCL-compatible MSCCL runtime, NCCL Tests, and Open MPI.

## 2. Supported Server Baseline

The documented reference platform is an x86_64 Ubuntu 22.04 or Ubuntu 24.04
server with:

- one or more NVIDIA GPUs;
- an NVIDIA driver compatible with the selected CUDA Toolkit;
- Python 3.10 through 3.12;
- passwordless SSH between participating nodes for multi-node execution;
- synchronized node clocks and a shared or identically located installation;
- `sudo` access for system packages, or equivalent preinstalled modules.

The README will not claim that an arbitrary latest CUDA version is compatible
with the pinned MSCCL runtime. It will require users to verify the driver,
CUDA, host compiler, and target GPU architecture before compilation. Actual
version strings used by online validation must be supplied through VeriCCL
environment variables and become part of the calibration cache signature.

## 3. MSCCL Distribution Strategies

Both strategies must produce the same patched source state and runtime
behavior.

### 3.1 Strategy A: Official Source plus VeriCCL Patch

Users clone `https://github.com/microsoft/msccl.git`, check out the pinned
upstream commit
`b23e9cd5dd63f82ee1c5aae7e0a2042079be903a`, copy the bundled trace header,
apply the bundled patch, run `verify_patch.py`, and build `src.build`.

The current patch cannot be used as-is because it was generated from a local
instrumented tree rather than clean upstream MSCCL. Implementation therefore
includes rebasing the patch onto the pinned commit and updating the verifier
to reject the wrong source revision.

### 3.2 Strategy C: Pre-integrated VeriCCL-MSCCL Repository

Create a public repository named `SlienceZDL/VeriCCL-MSCCL`, preserving the
official MSCCL history and recording `microsoft/msccl` as the upstream source.
Its default VeriCCL branch contains the same source state produced by Strategy
A. A versioned tag identifies the exact runtime used by the README.

The fork records:

- the upstream repository and commit;
- the VeriCCL repository commit containing the equivalent patch;
- required `MSCCL_CHUNKSTEPS=4` and `MSCCL_SLICESTEPS=4` values;
- build and verification commands;
- a warning that release measurements must disable trace collection.

The implementation must compare the relevant source tree after Strategy A
against the fork tag and fail validation if they differ.

## 4. README Information Architecture

Both README files will use the following order so a new server user can
proceed without consulting source code:

1. Project scope and supported collectives.
2. Offline versus full online installation matrix.
3. Ubuntu 22.04/24.04 system prerequisites.
4. NVIDIA driver, CUDA, NCCL, Open MPI, compiler, and Python preflight checks.
5. VeriCCL acquisition by SSH or HTTPS.
6. Python virtual environment and editable installation.
7. Gurobi package and license validation.
8. Minimal offline smoke test.
9. Strategy A MSCCL installation.
10. Strategy C MSCCL installation.
11. NCCL Tests and clock synchronization helper compilation.
12. Topology, sketch, and atom input contracts.
13. Repository example locations and selection guidance.
14. Offline `solve` and `verify` commands.
15. Semantic overrides, constructive solving, MILP, hierarchical solving,
    tuning, and timeout usage.
16. Online calibration and validation environment.
17. Single-node and multi-node XML execution with NCCL Tests.
18. Output directory and validation report interpretation.
19. Exit codes, common failures, and reproducibility checklist.
20. Development tests and further documentation.

## 5. Dependency Policy

The README will distinguish four dependency groups:

- Required for all uses: Git, Python, `venv`, pip, and the Python packages
  installed by `setup.py`.
- Required for MILP: a usable Gurobi license; the size-limited bundled license
  is not presented as sufficient for production VeriCCL models.
- Required for runtime compilation: build tools, patch, CUDA Toolkit, and the
  pinned or pre-integrated MSCCL source.
- Required for online and multi-node execution: Open MPI, MPI-enabled NCCL
  Tests, the clock synchronization helper, reachable hosts, and one visible
  GPU per MPI rank.

Commands will use shell variables such as `VERICCL_ROOT`, `MSCCL_ROOT`,
`NCCL_TESTS_ROOT`, and `CUDA_HOME`. Credentials, license secrets, hostnames,
and site-specific network interface names will never be embedded.

## 6. Input and Command Examples

The primary offline example remains:

- topology: `vericcl/examples/topo/two_rank.json`;
- sketch: `vericcl/examples/sketch/allreduce_8m_1m.json`;
- constructive atom policy: `vericcl/examples/atom/constructive.json`;
- MILP-capable atom policy: `vericcl/examples/atom/default.json`.

The hierarchical topology example is
`vericcl/examples/topo/two_node_gateway.json`. Legacy files remain explicitly
marked as reference-only inputs rather than supported examples.

The README will explain the meaning and unit of every field shown in these
examples, including directed links, channels, shared resources,
`alpha/beta/invbw`, collective semantics, total and slice sizes, solver
budgets, strategy flags, manual hierarchy, and forbidden transfers.

Every documented solve sequence writes to an explicit output root and uses a
deterministic run ID. The execution sequence derives `NCCL_BUFFSIZE` as twice
the sketch `slice_size_bytes`, loads exactly one XML, fixes the protocol to
`Simple`, and matches NCCL Tests message size, datatype, reduction operation,
root, rank count, and in-place mode to the generated XML.

## 7. Validation and Evidence Boundaries

Repository-local validation includes:

- README command parsing and execution for hardware-independent commands;
- JSON input resolution;
- pinned MSCCL revision and patch dry-run checks;
- trace header and required step constant checks;
- equality checks between the Strategy A source state and Strategy C tag;
- `git diff --check` and scans for stale paths or contradictory commands.

Ubuntu GPU server validation includes:

- `nvidia-smi`, `nvcc`, MPI, Gurobi, and Python import checks;
- MSCCL compilation and library discovery;
- NCCL Tests compilation with MPI support;
- clock synchronization helper compilation;
- a two-GPU offline-generated XML smoke run;
- optional intra-node and inter-node online verification.

Local static checks must not be reported as successful CUDA compilation or
hardware execution. The README will label commands that require a GPU server
and state the expected success indicators.

## 8. Files and External State

Implementation changes are limited to:

- `README.md`;
- `README.zh-CN.md`;
- `runtime/msccl-trace/README.md`;
- `runtime/msccl-trace/patches/0001-vericcl-fixed-step-trace.patch`;
- `runtime/msccl-trace/tools/verify_patch.py`;
- a small pinned-upstream metadata file under `runtime/msccl-trace/`;
- documentation integration tests;
- the new public `SlienceZDL/VeriCCL-MSCCL` repository.

Core solver, semantics, XML, and verification behavior will not be changed.

## 9. Completion Criteria

The work is complete only when:

1. both MSCCL strategies are documented and source-equivalent;
2. both README files contain the same commands and all local commands marked
   executable pass in order;
3. the rebased patch applies cleanly to the pinned official commit;
4. the verifier rejects an unpinned or unpatched MSCCL tree;
5. the public fork and versioned tag are readable through both HTTPS and SSH;
6. the README contains offline, online, single-node, and multi-node examples;
7. unexecuted hardware steps are explicitly identified rather than reported
   as validated.
