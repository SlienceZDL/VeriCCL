# Supported configurations and result interpretation

| Item | Configuration and conditions |
|---|---|
| Environment | Linux and Python 3.12; GPU execution uses NVIDIA CUDA, Open MPI, and the pinned MSCCL source and patches |
| Collectives | Broadcast, Reduce, AllGather, AllReduce, AllToAll, ReduceScatter; defaults are FP32, sum, root=0, and out-of-place |
| Types and reductions | Ten input datatypes and five reduction operations are checked against runtime support; native avg uses floating-point summation and final scaling, whose predicted cost uses the copy model |
| Topology | Explicit ranks, directed edges, and shared resources; the runtime maps ranks to hosts and GPUs |
| Input size | M is input bytes per rank, S is slice bytes, and M/S must be integral; AllToAll and ReduceScatter also require the slice count to be divisible by rank count |
| Remote paths | Resolved writable paths stay within the execution account home directory; configured system CUDA and MPI installations are read as dependencies |
| Runtime | Simple protocol; slices, channels, TBs, dependencies, and buffers must satisfy compiled and device limits |
| Cost model | Measured components record supported sizes and concurrency patterns; other ranges retain estimate provenance or missing status; full T_pred is recomputed under the current model |
| Repair comparison | Compare complete schedules under the same model, external request, and hard constraints; decisions outside the repair region stay fixed while timing propagates |
| Termination | Time limits, round limits, or no-improvement evidence within a stated legal space; preserve the best accepted schedule and certificate scope |
| Trace | One complete call after warmup; cross-device timing includes alignment error; primitive traces and full-call CUDA time are stored separately |
| Measured gain | Paired sampling and stability criteria; distinguish numerical correctness, resource release, predicted gain, and measured gain |
| Resident mode | Fixed members switch versions between requests; recovery requires protected input, an available control plane, valid routes, and confirmed termination of the old execution |

Predicted optimality applies to the recorded model, objective function, hard constraints, and search space. Choose synthesis, calibration, tracing, and measurement switches for the target topology, message size, and runtime environment. GPU ownership records persist across runs, and release confirmation affects admission of subsequent jobs.
