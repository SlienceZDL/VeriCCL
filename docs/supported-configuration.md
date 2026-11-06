# Supported configurations and result interpretation

| Item | Configuration and conditions |
|---|---|
| Environment | Linux and Python 3.12; GPU execution uses NVIDIA CUDA, Open MPI, and the repository's pinned MSCCL source and patches |
| Collectives | Broadcast, Reduce, AllGather, AllReduce, AllToAll, and ReduceScatter; defaults are FP32, sum, root=0, and out-of-place execution |
| Data types and reductions | Ten input types and five reduction options are checked against runtime capabilities; native avg uses floating-point summation followed by scaling, with scaling predicted from copy costs |
| Topology | Users explicitly provide ranks, directed edges, and shared resources; runtime configuration maps ranks to hosts and GPUs |
| Input size | M is the input size in bytes per rank, S is the minimum slice size, and M/S is an integer; AllToAll and ReduceScatter require the slice count to be divisible by the rank count |
| Remote paths | Resolved writable paths must be under the executing account's home directory; system CUDA/MPI dependencies are read from configured paths |
| Runtime | The Simple protocol is fixed; slices, channels, TBs, dependencies, and buffers must satisfy both code and device limits |
| Cost model | Measured components record valid sizes and concurrency modes; other ranges retain estimated sources or missing status, and the full T_pred is recomputed under the current model |
| Repair comparison | Complete schedule times are compared under the same model, external request, and hard constraints; decisions outside the repair domain stay fixed, and related timing propagates through dependencies |
| Search termination | Time limits, round limits, or no-improvement evidence over the specified legal space determine termination; output retains the best accepted schedule and proof scope |
| Trace | One complete invocation is captured after warmup, with calibration errors for cross-device timing; detailed primitive traces and CUDA full-invocation times are saved separately |
| Measured gains | Comparisons use predefined paired-sampling and stability criteria; reports distinguish numerical correctness, resource release, model gains, and measured gains |
| Native execution | Each verification run uses a fixed schedule; ranks coordinate preparation and launch, then record numerical checks and resource release on completion |

Predicted optimality applies to the recorded model, objective function, hard constraints, and search space. Choose synthesis, calibration, tracing, and measurement switches for the target topology, message size, and runtime environment. GPU ownership records persist across runs, and release confirmation affects admission of subsequent jobs.
