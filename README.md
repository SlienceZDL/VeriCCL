# VeriCCL

VeriCCL synthesizes, verifies, and tunes AllGather and AllReduce schedules. It uses SCIP and produces XML for the bundled MSCCL runtime.

## Environment

Use Linux with Git, Make, and the following dependency ranges. Choose a CUDA-compatible host compiler and NVIDIA driver.

| Dependency | Minimum | Upper bound |
|---|---|---|
| Python | 3.12 | < 3.14 |
| CUDA Toolkit | 11.0 | < 13.0 for Volta/V100 |
| Open MPI | 4.1 | < 5.0 for this guide |
| CMake | 3.20 | No project-imposed upper bound |
| C++ compiler | C++17 support | Within the selected CUDA toolkit's compiler range |

Python bounds follow the package metadata. CUDA needs [C++17 support](https://docs.nvidia.com/cuda/archive/11.0/cuda-c-programming-guide/#cpp17-language-features); [Volta compilation support ends before CUDA 13](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/). Other GPUs require a toolkit that supports their architecture. The Open MPI range retains the ORTE launch options used by the runtime. These are build and interface bounds; choose matching versions on all participating hosts.

## Install

Clone the repository and install the Python package:

```bash
git clone https://github.com/SlienceZDL/VeriCCL.git
cd VeriCCL
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m pip check
```

For Python 3.12, `requirements-lock/python312.txt` provides a pinned dependency set. CPU synthesis is ready after installation.

To build the optional CPU accelerator, install GMP development headers and run:

```bash
.venv/bin/python tools/build_native.py
```

Use `--gmp-root /path/to/gmp` when GMP is installed separately.

## Build the GPU runtime

Install CUDA and Open MPI within the ranges above. Run the following commands from the repository root. Replace `/path/to/...`, `GPU_ARCH`, `JOBS`, and `SECONDS` with your installation paths, GPU architecture, build parallelism, and timeout.

Build the bundled MSCCL library:

```bash
.venv/bin/python runtime/build_runtime.py \
  --output-dir /path/to/runtime-build \
  --cuda-home /path/to/cuda --arch GPU_ARCH --jobs JOBS --timeout-s SECONDS
```

Build the measurement probe and schedule executor against that library:

```bash
.venv/bin/python runtime/build_probes.py \
  --output-dir /path/to/probe-build \
  --cuda-home /path/to/cuda --mpi-bin /path/to/mpi/bin \
  --msccl-build /path/to/runtime-build/build --arch GPU_ARCH --timeout-s SECONDS

.venv/bin/python runtime/build_native.py \
  --output-dir /path/to/native-build \
  --cuda-home /path/to/cuda --mpi-bin /path/to/mpi/bin \
  --msccl-build /path/to/runtime-build/build --arch GPU_ARCH --timeout-s SECONDS
```

The outputs are `runtime-build/build/lib/libnccl.so`, `probe-build/build/bin/vericcl-probe`, and `native-build/build/bin/vericcl-native`. Each build directory also contains `result.json`. Use a new output directory for each build.

For multi-host execution, install the same versions at the same paths and enable passwordless SSH between the participating hosts. Set executable paths and `LD_LIBRARY_PATH` in `runtime.json`.

## Prepare inputs

A task uses four JSON files:

| File | Contents |
|---|---|
| `topology.json` | Ranks, directed links, shared resources, and initial costs |
| `sketch.json` | Collective, message and slice sizes, solver limits, measurement, and tuning |
| `atom.json` | Synthesis method and optional manual constraints |
| `runtime.json` | Rank-to-host/GPU mapping, executable paths, and environment |

Create them from the [field reference](docs/input-schema.md). `M` is input bytes per rank, `S` is slice bytes, and `M/S` must be an integer. For P ranks, nccl-tests size is P*M for AllGather and M for AllReduce. Set `inplace` for the intended buffer layout.

Choose `hierarchy` or `constructive_trees` in `atom.synthesis_method`. In `sketch.json`, set budgets, `calibration.enabled`, `trace_and_measure.enabled`, and `symbolic.enabled` as needed. `budgets.max_tuning_rounds` controls the repeated verification-repair loop.

Set `sketch.symbolic.enabled` to `true` to enable the full symbolic execution and BDD features.

## Synthesize

`solve` synthesizes a schedule and runs the enabled calibration and tuning stages:

```bash
.venv/bin/python -m vericcl solve \
  --project-root /path/to/VeriCCL --case CASE --run-id RUN_ID \
  --topology /path/to/input/topology.json --sketch /path/to/input/sketch.json \
  --atom /path/to/input/atom.json --runtime /path/to/input/runtime.json
```

Results are saved in `/path/to/VeriCCL/examples/CASE/output/RUN_ID/`. Use a new run ID for each invocation. Add `--override-input --set FIELD=JSON_VALUE` to change an explicit input value.

## Verify and repair

`verify` checks an existing XML with its matching sidecar and original inputs. Read the accepted XML, sidecar, and model version from `best/accepted.json`; paths in that record are relative to the source run directory.

```bash
.venv/bin/python -m vericcl verify \
  --project-root /path/to/VeriCCL --case CASE --run-id RUN_ID \
  --topology /path/to/input/topology.json --sketch /path/to/input/sketch.json \
  --atom /path/to/input/atom.json --runtime /path/to/input/runtime.json \
  --xml /path/to/schedule.xml --sidecar /path/to/sidecar.json \
  --model /path/to/source-run/models/MODEL_VERSION.json
```

Add `--tune` to continue the verification-repair loop from this schedule.

## Calibrate

`calibrate` measures link, shared-resource, and local-operation costs and writes the model to the run's `models/` directory:

```bash
.venv/bin/python -m vericcl calibrate \
  --project-root /path/to/VeriCCL --case CASE --run-id RUN_ID \
  --topology /path/to/input/topology.json --sketch /path/to/input/sketch.json \
  --atom /path/to/input/atom.json --runtime /path/to/input/runtime.json
```

Configure the calibration budget, fit tolerance, job timeout, rank map, and runtime paths before starting.

## Inspect and resume

`inspect` displays the run summary and accepted schedule:

```bash
.venv/bin/python -m vericcl inspect --run /path/to/run
```

`resume` continues from the latest checkpoint with the original inputs and remaining budget:

```bash
.venv/bin/python -m vericcl resume --run /path/to/run
```

## Resident execution

`live` serves repeated collective requests. Configure `sketch.live` and `runtime.control`, then set the request count and session timeout:

```bash
.venv/bin/python -m vericcl live \
  --project-root /path/to/VeriCCL --case CASE --run-id RUN_ID \
  --topology /path/to/input/topology.json --sketch /path/to/input/sketch.json \
  --atom /path/to/input/atom.json --runtime /path/to/input/runtime.json \
  --requests REQUESTS --session-timeout-s SECONDS
```

## Measure with nccl-tests

Build the test binaries against the same MSCCL library:

```bash
.venv/bin/python examples/nccl-tests-build/input/build.py --run-id BUILD_ID \
  --cuda-home /path/to/cuda --mpi-home /path/to/mpi \
  --msccl-build /path/to/runtime-build/build --arch GPU_ARCH --release-receipts
```

Binaries are saved under `examples/nccl-tests-build/output/BUILD_ID/rebuildable-cache/source/build/`. Use `all_gather_perf` or `all_reduce_perf` as `COLLECTIVE_PERF`. Set host slots and rank placement to match `runtime.json`, and make the XML available at the same path on every host:

```bash
/path/to/mpi/bin/mpirun --bind-to none -np RANKS --host HOST_SLOTS \
  -x LD_LIBRARY_PATH=/path/to/runtime-build/build/lib:/path/to/mpi/lib:/path/to/cuda/lib64 \
  -x NCCL_ALGO=MSCCL,RING -x NCCL_PROTO=Simple \
  -x MSCCL_XML_FILES=/path/to/schedule.xml \
  /path/to/nccl-tests/build/COLLECTIVE_PERF \
  -g 1 -b SIZE_BYTES -e SIZE_BYTES -d float -c 1
```

Pass the network and buffer settings from `runtime.json` with additional `-x NAME=VALUE` arguments. Select the in-place or out-of-place result columns to match the XML.

## Results

| Path within a run | Contents |
|---|---|
| `run-summary.json`, `reports/timing.json` | Status, stop reason, and timing |
| `best/accepted.json` | Accepted schedule and model references |
| `candidates/` | XML, sidecar, and checks for each candidate |
| `models/` | Initial and calibrated cost models |
| `input_snapshot/`, `effective-request.json` | Original and effective inputs |
| `jobs/`, `run.log`, `events.jsonl` | Execution logs |
| `checkpoints/`, `live/` | Resume state and resident-session records |

Use `.venv/bin/python -m vericcl COMMAND --help` for all options. See [supported configurations](docs/supported-configuration.md), [architecture](docs/architecture.md), and the [runtime guide](runtime/msccl-trace/README.md) for details.
