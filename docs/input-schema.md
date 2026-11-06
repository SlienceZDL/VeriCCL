# Input field reference

This table is generated from `vericcl/input/schema.py`. All four JSON inputs reject duplicate keys, unknown fields, and non-finite numbers; original files are never overwritten. An omitted or null value is unspecified: a default is applied when available, and a required field without a default raises an error when its feature is enabled. A field present in both its old and new locations raises an error even if the values match.

M must be divisible by S, and S must contain complete datatype elements. For AllToAll/ReduceScatter, N=M/S must also be divisible by the rank count. Do not provide N or input_chunkup directly. Omit root/reduction_op or use null for operators to which they do not apply; concrete values are rejected.

The command-line option `--set path=JSON` can fill unspecified values. Replacing a different existing value requires `--override-input`. Original files are saved in `process/input_snapshot/`; the final effective configuration and value sources are stored as `effective_request` and `value_sources` in `run.json`. Configuration fields do not expand shell placeholders; write actual absolute paths into JSON.

solve requires an initial synthesis budget and maximum round count; a count above 0 also requires repair budgets. All processing entry points require per-model/CPU, per-round/total offline, and cleanup budgets. Enabling calibration or using calibrate requires online budgets, a fit-error limit, and a per-job deadline. Enabling tracing/measurement also requires trace capacity and a job deadline. Each history reuse feature requires its index and the corresponding tolerance or age limit. Runtime numerical checks use explicit absolute and relative tolerances.

JSON Schema describes the standard structure; load_request checks cross-field arithmetic, enabling conditions, duplicate keys, and explicitly supported legacy migrations. Operator names are case-insensitive; compatible aliases are all_gather, all_reduce, all_to_all, and reducescatter. Explicitly listed solver parameters in legacy hyperparameters can be migrated, as can symmetry, shortest_paths, batching, and objective_mode at the atom top level. Legacy total_solve_timeout_s migrates only to the initial synthesis budget. force_resolve, force_recalibrate, and legacy algorithm boolean switches have been removed.

A manual node's `atoms` specify only sets of slices, source ranks, and destination ranks; the stage comes from the node's `stage_id`. Each expanded constraint restricts one directed edge. Manual mode uses hierarchy to solve each stage and retains hierarchical subproblem reuse; whole-schedule history reuse is limited to automatic plans. When manual mode is disabled, its nodes and variable groups do not participate in automatic synthesis.

The object/array rows below describe configuration sections, and subsequent rows describe their members. Within arrays, [] denotes each element. See the [README](../README.md) for more workflows and [supported configurations](supported-configuration.md) for the scope of evidence.


## topology.json

[Machine-readable Schema](input-schemas/topology.json)

| Field | Type | Default | Requirements and values | Purpose |
|---|---|---|---|---|
| `topology.name` | string | `unspecified` | Checked when enabled; — | Topology name, used only for identification. |
| `topology.ranks` | integer | `unspecified` | Required; ≥1, ≤9223372036854775807 | Number of logical ranks in the task; within a node, the list of its ranks. |
| `topology.nodes` | array | `unspecified` | Required; — | Node groups covering every rank; node IDs do not replace actual host addresses. |
| `topology.nodes[].id` | integer | `unspecified` | Required; ≥0, ≤9223372036854775807 | Unique identifier within this configuration. |
| `topology.nodes[].ranks` | array | `unspecified` | Required; — | Number of logical ranks in the task; within a node, the list of its ranks. |
| `topology.nodes[].gateways` | array | `[]` | May be omitted or null; — | Ranks that can serve as external connection endpoints for this node. |
| `topology.directed_links` | array | `unspecified` | Required; — | Directed links, either direct or abstracted through switches; provide the reverse direction separately. |
| `topology.directed_links[].src` | integer | `unspecified` | Required; ≥0, ≤9223372036854775807 | Sending rank. |
| `topology.directed_links[].dst` | integer | `unspecified` | Required; ≥0, ≤9223372036854775807 | Receiving rank. |
| `topology.directed_links[].alpha_us` | number | `unspecified` | Required; ≥0 | Fixed service overhead per operation, in microseconds. |
| `topology.directed_links[].beta_us_per_byte` | number | `unspecified` | Required; >0 | Service cost per byte, in microseconds per byte; bandwidth is its reciprocal. |
| `topology.directed_links[].max_channels` | integer | `16` | May be omitted or null; ≥1, ≤16 | Maximum allowed channel count; the default search covers 1 through 7, with an explicit upper limit of up to 16. |
| `topology.directed_links[].resources` | array | `[]` | May be omitted or null; — | Identifiers of shared resources used by this link. |
| `topology.directed_links[].physical_path` | array | `[]` | May be omitted or null; — | Description of the physical path represented by the abstract link. |
| `topology.shared_resources` | array | `[]` | May be omitted or null; — | Shared bottlenecks across abstract links, preventing physical bandwidth from being counted more than once. |
| `topology.shared_resources[].id` | string | `unspecified` | Required; — | Unique identifier within this configuration. |
| `topology.shared_resources[].capacity_bytes_per_us` | number | `unspecified` | Required; >0 | Total effective resource bandwidth, in bytes per microsecond. |
| `topology.shared_resources[].links` | array | `[]` | May be omitted or null; — | Directed links [src,dst] belonging to this resource. |
| `topology.shared_resources[].kind` | string | `"other"` | May be omitted or null; nic / pcie / switch_port / link / other | Shared resource category. |
| `topology.groups` | array | `[]` | May be omitted or null; — | User-provided candidate groups for hierarchical planning. |
| `topology.local_costs` | array | `[]` | May be omitted or null; — | Initial local cost estimates for the current datatype and reduction operation. |
| `topology.local_costs[].rank` | integer | `unspecified` | Required; ≥0, ≤9223372036854775807 | Logical rank ID, starting at 0. |
| `topology.local_costs[].copy` | object | `unspecified` | Checked when enabled; — | Initial cost of local reads, writes, or copies. |
| `topology.local_costs[].copy.alpha_us` | number | `unspecified` | Required; ≥0 | Fixed service overhead per operation, in microseconds. |
| `topology.local_costs[].copy.beta_us_per_byte` | number | `unspecified` | Required; >0 | Service cost per byte, in microseconds per byte; bandwidth is its reciprocal. |
| `topology.local_costs[].reduce` | object | `unspecified` | Checked when enabled; — | Initial cost of local reduction. |
| `topology.local_costs[].reduce.alpha_us` | number | `unspecified` | Required; ≥0 | Fixed service overhead per operation, in microseconds. |
| `topology.local_costs[].reduce.beta_us_per_byte` | number | `unspecified` | Required; >0 | Service cost per byte, in microseconds per byte; bandwidth is its reciprocal. |

## sketch.json

[Machine-readable Schema](input-schemas/sketch.json)

| Field | Type | Default | Requirements and values | Purpose |
|---|---|---|---|---|
| `sketch.collective` | object | `unspecified` | Required; — | Application-visible collective semantics. |
| `sketch.collective.operator` | string | `unspecified` | Required; broadcast / reduce / allgather / allreduce / alltoall / reduce_scatter | Collective operator; internal manual nodes also support Scatter/Gather. |
| `sketch.collective.datatype` | string | `"float32"` | May be omitted or null; float16 / float32 / float64 / bfloat16 / int8 / uint8 / int32 / uint32 / int64 / uint64 | Datatype, defaulting to float32; actual support must pass runtime checks. |
| `sketch.collective.root` | integer | `0 when applicable` | Checked when enabled; ≥0, ≤9223372036854775807 | External root for Broadcast/Reduce, defaulting to 0; for a manual node, that node's root. |
| `sketch.collective.reduction_op` | string | `sum when applicable` | Checked when enabled; sum / prod / min / max / avg | Reduction operation for Reduce/AllReduce/ReduceScatter, defaulting to sum. Omit or use null for other external operators. |
| `sketch.collective.inplace` | boolean | `false` | May be omitted or null; — | Whether to use a valid in-place buffer layout; disabled by default. |
| `sketch.hyperparameters` | object | `unspecified` | Required; — | Application input size and minimum slice size. |
| `sketch.hyperparameters.total_size_bytes` | integer | `unspecified` | Required; ≥1, ≤9223372036854775807 | Input bytes M per rank; for Broadcast, the root message size. |
| `sketch.hyperparameters.slice_size_bytes` | integer | `null` | May be omitted or null; ≥1, ≤9223372036854775807 | Fixed minimum slice size S in bytes; if omitted, search for a valid partition first. N is derived as M/S. |
| `sketch.hyperparameters.reference_slice_size_bytes` | integer | `null` | May be omitted or null; ≥1, ≤9223372036854775807 | Reference partition for hard constraints on original slice_id values; repartitioning preserves their original byte ranges. |
| `sketch.solver` | object | `{}` | May be omitted or null; — | Solver, CPU quota, and solving strategies. |
| `sketch.solver.backend` | string | `"scip"` | May be omitted or null; scip | Only SCIP is supported; its absence causes an immediate error. |
| `sketch.solver.per_model_timeout_s` | number | `unspecified` | Checked when enabled; >0 | Time limit for one solver model, also bounded by the remaining parent-phase budget. |
| `sketch.solver.cpu_budget` | integer | `unspecified` | Checked when enabled; ≥1, ≤9223372036854775807 | CPU quota available to the entire task. |
| `sketch.solver.max_threads_per_model` | integer | `12` | May be omitted or null; ≥1, ≤9223372036854775807 | Thread limit per model, defaulting to 12; also bounded by the CPU quota. |
| `sketch.solver.max_parallel_models` | integer | `unspecified` | Checked when enabled; ≥1, ≤9223372036854775807 | Maximum number of concurrent models; managed by the budget manager if omitted. |
| `sketch.solver.max_channels` | integer | `7` | May be omitted or null; ≥1, ≤16 | Maximum allowed channel count; the default search covers 1 through 7, with an explicit upper limit of up to 16. |
| `sketch.solver.solver_seed` | integer | `0` | May be omitted or null; ≥0, ≤9223372036854775807 | SCIP random seed. |
| `sketch.solver.mip_gap` | number | `0` | May be omitted or null; ≥0, ≤1 | Allowed numerical MIP gap; it does not prove coverage of the full repair space. |
| `sketch.solver.require_proven_optimal` | boolean | `false` | May be omitted or null; — | Whether the current solver model must reach numerical optimality. |
| `sketch.solver.bdd_backend` | string | `"cudd"` | May be omitted or null; cudd / autoref | BDD backend for analysis; CUDD is the default, and autoref requires explicit selection. |
| `sketch.budgets` | object | `{}` | May be omitted or null; — | Phase time limits and maximum tuning rounds; all times are in seconds. |
| `sketch.budgets.max_initial_synthesis_s` | number | `unspecified` | Checked when enabled; >0 | Total time limit for initial synthesis in this run. |
| `sketch.budgets.max_online_validation_per_round_s` | number | `unspecified` | Checked when enabled; >0 | Cumulative limit for online calibration, tracing, and measurement in one round. |
| `sketch.budgets.max_online_validation_total_s` | number | `unspecified` | Checked when enabled; >0 | Cumulative online validation limit for the entire task. |
| `sketch.budgets.max_offline_validation_per_round_s` | number | `unspecified` | Checked when enabled; >0 | Cumulative limit for offline checks and analysis in one round. |
| `sketch.budgets.max_offline_validation_total_s` | number | `unspecified` | Checked when enabled; >0 | Cumulative limit for offline checks and analysis across the task. |
| `sketch.budgets.max_repair_per_round_s` | number | `unspecified` | Checked when enabled; >0 | Cumulative limit for all local, structural, and partition repairs in one round. |
| `sketch.budgets.max_repair_total_s` | number | `unspecified` | Checked when enabled; >0 | Cumulative repair limit for the entire task. |
| `sketch.budgets.max_run_wall_s` | number | `unspecified` | Checked when enabled; >0 | Optional wall-clock time limit for the entire task. |
| `sketch.budgets.cleanup_reserve_s` | number | `unspecified` | Checked when enabled; >0 | Cleanup time reserved from the remaining budget; does not extend the parent deadline. |
| `sketch.budgets.max_tuning_rounds` | integer | `unspecified` | Checked when enabled; ≥0, ≤9223372036854775807 | Maximum number of tuning rounds to start; 0 stops after establishing the initial result. |
| `sketch.tolerances` | object | `{}` | May be omitted or null; — | Floating-point comparison tolerances, which must be provided explicitly for execution. |
| `sketch.tolerances.absolute` | number | `unspecified` | Checked when enabled; ≥0 | Absolute error tolerance. |
| `sketch.tolerances.relative` | number | `unspecified` | Checked when enabled; ≥0 | Relative error tolerance. |
| `sketch.calibration` | object | `{}` | May be omitted or null; — | Independent primitive measurement and cost-model calibration. |
| `sketch.calibration.enabled` | boolean | `true` | May be omitted or null; — | Enable the feature in this configuration section. |
| `sketch.calibration.launches` | integer | `20` | May be omitted or null; ≥1, ≤9223372036854775807 | Independent process launches per round; iterations within one launch do not count as independent launches. |
| `sketch.calibration.warmup_iterations` | integer | `5` | May be omitted or null; ≥0, ≤9223372036854775807 | Warmup iterations per launch; detailed tracing resets after warmup and records one iteration. |
| `sketch.calibration.timed_iterations` | integer | `20` | May be omitted or null; ≥1, ≤9223372036854775807 | Timed iterations per launch without detailed tracing. |
| `sketch.calibration.max_rounds` | integer | `3` | May be omitted or null; ≥1, ≤9223372036854775807 | Maximum measurement rounds when stability criteria are unmet. |
| `sketch.calibration.max_cv` | number | `0.05` | May be omitted or null; >0 | Maximum coefficient of variation among samples in one round. |
| `sketch.calibration.max_relative_fit_error` | number | `unspecified` | Checked when enabled; ≥0 | Maximum allowed relative error in cost fitting. |
| `sketch.calibration.job_timeout_s` | number | `unspecified` | Checked when enabled; >0 | Time limit for one complete measurement job, including launch, retries, collection, and cleanup. |
| `sketch.calibration.payload_sizes` | array | `[]` | May be omitted or null; — | Additional probe sizes; do not change the application M/S. |
| `sketch.trace_and_measure` | object | `{}` | May be omitted or null; — | One switch controls detailed tracing and the formal measured-gain threshold for candidates; disabled by default. |
| `sketch.trace_and_measure.enabled` | boolean | `false` | May be omitted or null; — | Enable the feature in this configuration section. |
| `sketch.trace_and_measure.pairs` | integer | `20` | May be omitted or null; ≥1, ≤9223372036854775807 | Predetermined number of paired AB/BA launches; must be an even number of at least 2. |
| `sketch.trace_and_measure.warmup_iterations` | integer | `5` | May be omitted or null; ≥0, ≤9223372036854775807 | Warmup iterations per launch; detailed tracing resets after warmup and records one iteration. |
| `sketch.trace_and_measure.timed_iterations` | integer | `20` | May be omitted or null; ≥1, ≤9223372036854775807 | Timed iterations per launch without detailed tracing. |
| `sketch.trace_and_measure.max_rounds` | integer | `3` | May be omitted or null; ≥1, ≤9223372036854775807 | Maximum measurement rounds when stability criteria are unmet. |
| `sketch.trace_and_measure.max_cv` | number | `0.05` | May be omitted or null; >0 | Maximum coefficient of variation among samples in one round. |
| `sketch.trace_and_measure.min_relative_gain` | number | `0` | May be omitted or null; ≥0, ≤1 | Minimum relative improvement required in measurements. |
| `sketch.trace_and_measure.task_alpha` | number | `0.05` | May be omitted or null; >0, ≤0.05 | Statistical error probability budget for the entire task, at most 0.05. |
| `sketch.trace_and_measure.single_attempt_alpha_cap` | number | `0.05` | May be omitted or null; >0, ≤0.05 | Error probability limit for one comparison, also subject to allocation by the task ledger. |
| `sketch.trace_and_measure.seed` | integer | `0` | May be omitted or null; ≥0, ≤9223372036854775807 | Seed that fixes the sampling order in advance. |
| `sketch.trace_and_measure.job_timeout_s` | number | `unspecified` | Checked when enabled; >0 | Time limit for one complete measurement job, including launch, retries, collection, and cleanup. |
| `sketch.trace_and_measure.trace_capacity_bytes` | integer | `unspecified` | Checked when enabled; ≥1, ≤9223372036854775807 | Detailed trace buffer bytes per rank; must accommodate the file header and all trace records for one complete invocation. |
| `sketch.repair` | object | `{}` | May be omitted or null; — | Division of analysis and solving work during repair. |
| `sketch.repair.mode` | string | `"strategy_first"` | May be omitted or null; strategy_first / joint | strategy_first selects a strategy first; joint selects valid decisions jointly. |
| `sketch.repair.candidate_batch_size` | integer | `4` | May be omitted or null; ≥1, ≤9223372036854775807 | Candidates sent to local SCIP per batch; does not limit the total search. |
| `sketch.repair.analysis_batch_size` | integer | `16` | May be omitted or null; ≥1, ≤9223372036854775807 | Analysis suggestions per batch; expand further when exhausted. |
| `sketch.runtime_compatibility` | object | `{}` | May be omitted or null; — | Full static runtime compatibility checks, which can be disabled; basic launch safety checks always remain active. |
| `sketch.runtime_compatibility.enabled` | boolean | `true` | May be omitted or null; — | Enable the feature in this configuration section. |
| `sketch.history` | object | `{}` | May be omitted or null; — | Cross-task schedule and absolute calibration caches; the two switches are independent and disabled by default. |
| `sketch.history.schedule_reuse` | boolean | `false` | May be omitted or null; — | In automatic mode, retrieve isomorphic historical schedules with similar relative bandwidths, then adapt and revalidate them; use false for manual plans. |
| `sketch.history.calibration_reuse` | boolean | `false` | May be omitted or null; — | Reuse only valid measurement components with exactly matching device, runtime, datatype, and mode identities. |
| `sketch.history.relative_bandwidth_tolerance` | number | `unspecified` | Checked when enabled; ≥0 | Maximum allowed value of the largest mapped bandwidth scale divided by the smallest, minus one. |
| `sketch.history.max_schedule_candidates` | integer | `unspecified` | Checked when enabled; ≥1, ≤9223372036854775807 | Maximum candidates checked by fast history lookup. |
| `sketch.history.calibration_max_age_s` | number | `unspecified` | Checked when enabled; >0 | Maximum calibration cache age. |
| `sketch.history.schedule_index` | string | `unspecified` | Checked when enabled; — | Historical schedule index path; resolve relative paths from the case directory. |
| `sketch.history.calibration_index` | string | `unspecified` | Checked when enabled; — | Calibration index path; resolve relative paths from the case directory. |

## atom.json

[Machine-readable Schema](input-schemas/atom.json)

| Field | Type | Default | Requirements and values | Purpose |
|---|---|---|---|---|
| `atom.synthesis_method` | string | `"auto"` | May be omitted or null; auto / hierarchy / constructive_trees | For automatic plans, auto uses M_switch to select hierarchy for small messages and constructive_trees for large messages; manual plans use hierarchy, and auto also resolves to hierarchy. |
| `atom.M_switch` | integer | `unspecified` | Checked when enabled; ≥1, ≤9223372036854775807 | Explicit message-size threshold in bytes when automatic plans use auto. |
| `atom.stage_num` | integer | `null` | May be omitted or null; ≥1, ≤9223372036854775807 | Hard constraint on the number of logical stages; determined automatically if omitted. |
| `atom.forbidden_transfers` | array | `[]` | May be omitted or null; — | [original slice_id,src,dst,stage]; forbids only this direct edge in the specified stage, while valid alternate paths remain allowed. |
| `atom.strategies` | object | `{}` | May be omitted or null; — | Optional initial solving strategies, disabled by default. |
| `atom.strategies.symmetry` | boolean | `false` | May be omitted or null; — | Enable symmetry reduction. |
| `atom.strategies.shortest_paths` | boolean | `false` | May be omitted or null; — | Enable candidate path reduction. |
| `atom.strategies.batching` | boolean | `false` | May be omitted or null; — | Explicitly enable batching for hierarchical solving; the constructive tree method manages batches by its own rules. |
| `atom.strategies.objective_mode` | string | `null` | May be omitted or null; latency / throughput / auto | Explicitly select a latency or throughput objective, or choose automatically by message size. |
| `atom.manual` | object | `{}` | May be omitted or null; — | Explicit manual plan, used only when enabled=true. |
| `atom.manual.enabled` | boolean | `false` | May be omitted or null; — | Enable the feature in this configuration section. |
| `atom.manual.nodes` | array | `[]` | May be omitted or null; — | Manual logical nodes; each specifies a group, operator, dependencies, and optional state contracts. |
| `atom.manual.nodes[].id` | string | `unspecified` | Required; — | Unique identifier within this configuration. |
| `atom.manual.nodes[].stage_id` | integer | `unspecified` | Required; ≥0, ≤9223372036854775807 | Logical stage ID, starting at 0. |
| `atom.manual.nodes[].operator` | string | `unspecified` | Required; broadcast / reduce / scatter / gather / allgather / allreduce / alltoall / reduce_scatter | Collective operator; internal manual nodes also support Scatter/Gather. |
| `atom.manual.nodes[].group` | array | `unspecified` | Required; — | Ranks available to this manual node. |
| `atom.manual.nodes[].root` | integer | `unspecified` | Checked when enabled; ≥0, ≤9223372036854775807 | External root for Broadcast/Reduce, defaulting to 0; for a manual node, that node's root. |
| `atom.manual.nodes[].relay_ranks` | array | `[]` | May be omitted or null; — | Ranks allowed to relay within the group; ranks outside the group are not allowed. |
| `atom.manual.nodes[].atoms` | array | `[]` | May be omitted or null; — | Variable groups that forbid directed edges within a stage, expanded over all combinations of their sets. |
| `atom.manual.nodes[].atoms[].slice_ids` | array | `unspecified` | Required; — | Original slice IDs subject to the constraint. |
| `atom.manual.nodes[].atoms[].src_ranks` | array | `unspecified` | Required; — | Sending ranks of forbidden edges. |
| `atom.manual.nodes[].atoms[].dst_ranks` | array | `unspecified` | Required; — | Receiving ranks of forbidden edges. |
| `atom.manual.nodes[].depends_on` | array | `[]` | May be omitted or null; — | IDs of nodes on which this manual node depends. |
| `atom.manual.nodes[].input_states` | array | `unspecified` | Checked when enabled; — | Input state contract for the manual node. |
| `atom.manual.nodes[].input_states[].rank` | integer | `unspecified` | Required; ≥0, ≤9223372036854775807 | Logical rank ID, starting at 0. |
| `atom.manual.nodes[].input_states[].offset` | integer | `unspecified` | Required; ≥0, ≤9223372036854775807 | Logical data position ID. |
| `atom.manual.nodes[].input_states[].contributors` | array | `unspecified` | Required; — | Original slice_id values contained in this state. |
| `atom.manual.nodes[].output_states` | array | `unspecified` | Checked when enabled; — | Output state contract for the manual node. |
| `atom.manual.nodes[].output_states[].rank` | integer | `unspecified` | Required; ≥0, ≤9223372036854775807 | Logical rank ID, starting at 0. |
| `atom.manual.nodes[].output_states[].offset` | integer | `unspecified` | Required; ≥0, ≤9223372036854775807 | Logical data position ID. |
| `atom.manual.nodes[].output_states[].contributors` | array | `unspecified` | Required; — | Original slice_id values contained in this state. |

## runtime.json

[Machine-readable Schema](input-schemas/runtime.json)

| Field | Type | Default | Requirements and values | Purpose |
|---|---|---|---|---|
| `runtime.rank_map` | array | `unspecified` | Required; — | Actual host and device placement for every rank; must be complete with no duplicate devices. |
| `runtime.rank_map[].rank` | integer | `unspecified` | Required; ≥0, ≤9223372036854775807 | Logical rank ID, starting at 0. |
| `runtime.rank_map[].host` | string | `unspecified` | Required; — | Hostname or address reachable through both SSH and MPI. |
| `runtime.rank_map[].device` | integer | `unspecified` | Required; ≥0, ≤9223372036854775807 | CUDA device ID visible to the process on this host. |
| `runtime.environment` | object | `unspecified` | Required; — | Allowed runtime environment variables, all with string values. |
| `runtime.environment.CUDA_HOME` | string | `unspecified` | Checked when enabled; — | CUDA root directory. |
| `runtime.environment.CUDA_PATH` | string | `unspecified` | Checked when enabled; — | Compatibility variable for the CUDA root directory. |
| `runtime.environment.CUDA_VISIBLE_DEVICES` | string | `unspecified` | Checked when enabled; — | Visible device set, which changes the meaning of device IDs. |
| `runtime.environment.LD_LIBRARY_PATH` | string | `unspecified` | Checked when enabled; — | Dynamic library search path, prioritizing the target runtime and MPI. |
| `runtime.environment.MSCCL_ROOT` | string | `unspecified` | Checked when enabled; — | Runtime root directory. |
| `runtime.environment.MSCCL_XML_FILES` | string | `unspecified` | Checked when enabled; — | Existing MSCCL XML loading interface; the native versioned interface binds plans explicitly. |
| `runtime.environment.NCCL_ALGO` | string | `unspecified` | Checked when enabled; — | Retained NCCL algorithm interface. |
| `runtime.environment.NCCL_BUFFSIZE` | string | `unspecified` | Required; — | Communication buffer size per channel, as a decimal byte string from 1 to 2147483647. |
| `runtime.environment.NCCL_DEBUG` | string | `unspecified` | Checked when enabled; — | Runtime logging level. |
| `runtime.environment.NCCL_DEBUG_SUBSYS` | string | `unspecified` | Checked when enabled; — | Runtime logging subsystems. |
| `runtime.environment.NCCL_IB_GID_INDEX` | string | `unspecified` | Checked when enabled; — | Explicit GID index. |
| `runtime.environment.NCCL_IB_HCA` | string | `unspecified` | Checked when enabled; — | Explicitly selected IB/RoCE devices. |
| `runtime.environment.NCCL_P2P_DISABLE` | string | `unspecified` | Checked when enabled; — | Whether to disable GPU P2P paths. |
| `runtime.environment.NCCL_PROTO` | string | `unspecified` | Checked when enabled; — | Runtime protocol; tasks use Simple. |
| `runtime.environment.NCCL_SHM_DISABLE` | string | `unspecified` | Checked when enabled; — | Whether to disable host shared-memory paths. |
| `runtime.environment.NCCL_SOCKET_IFNAME` | string | `unspecified` | Checked when enabled; — | Communication network interface selection. |
| `runtime.environment.NCCL_TESTS_ROOT` | string | `unspecified` | Checked when enabled; — | Optional nccl-tests root directory. |
| `runtime.environment.OMPI_MCA_btl` | string | `unspecified` | Checked when enabled; — | Open MPI BTL components. |
| `runtime.environment.OMPI_MCA_btl_tcp_if_include` | string | `unspecified` | Checked when enabled; — | Interfaces or subnets for the Open MPI TCP data channel; nodes must be mutually reachable. |
| `runtime.environment.OMPI_MCA_oob_tcp_if_include` | string | `unspecified` | Checked when enabled; — | Interfaces or subnets for the Open MPI control channel; may explicitly use the same subnet as the TCP data channel. |
| `runtime.environment.OMPI_MCA_pml` | string | `unspecified` | Checked when enabled; — | Open MPI PML component. |
| `runtime.environment.OMP_NUM_THREADS` | string | `unspecified` | Checked when enabled; — | OpenMP thread count. |
| `runtime.environment.PATH` | string | `unspecified` | Checked when enabled; — | Executable search path for child processes. |
| `runtime.environment.PYTHONPATH` | string | `unspecified` | Checked when enabled; — | Python module search path. |
| `runtime.environment.UCX_NET_DEVICES` | string | `unspecified` | Checked when enabled; — | UCX network devices. |
| `runtime.environment.UCX_TLS` | string | `unspecified` | Checked when enabled; — | UCX transport selection. |
| `runtime.paths` | object | `{}` | May be omitted or null; — | Explicit toolchain and executable paths. |
| `runtime.paths.cuda_home` | string | `unspecified` | Checked when enabled; — | Root of an existing CUDA installation. |
| `runtime.paths.mpi_bin` | string | `unspecified` | Checked when enabled; — | Directory containing mpirun and mpicxx. |
| `runtime.paths.msccl_root` | string | `unspecified` | Checked when enabled; — | Built MSCCL root, containing include/lib. |
| `runtime.paths.nccl_tests_root` | string | `unspecified` | Checked when enabled; — | Optional nccl-tests root directory. |
| `runtime.paths.vericcl_driver` | string | `unspecified` | Checked when enabled; — | Absolute path to the independent primitive measurement program vericcl-probe. |
| `runtime.paths.vericcl_native` | string | `unspecified` | Checked when enabled; — | Absolute path to the full-schedule execution program vericcl-native. |
| `runtime.device_memory_budget_bytes` | integer | `0` | May be omitted or null; ≥0, ≤9223372036854775807 | Total device memory budget; 0 uses available memory at launch as the bound, while still checking actual allocations. |
| `runtime.protocol` | string | `"Simple"` | May be omitted or null; Simple | Currently fixed to Simple. |
| `runtime.scratch_bytes_per_rank` | integer | `unspecified` | Required; ≥0, ≤9223372036854775807 | Allowed scratch bytes per rank. |
| `runtime.limits` | object | `{}` | May be omitted or null; — | Runtime structural limits; may be reduced but cannot exceed compiled limits. |
| `runtime.limits.steps_per_tb` | integer | `256` | May be omitted or null; ≥1, ≤256 | Maximum instructions per TB. |
| `runtime.limits.tbs_per_rank` | integer | `216` | May be omitted or null; ≥1, ≤216 | Maximum TBs per rank; must also fit the actual device residency capacity. |
| `runtime.limits.send_tbs_per_channel` | integer | `32` | May be omitted or null; ≥1, ≤32 | Maximum send TBs per channel. |
| `runtime.limits.recv_tbs_per_channel` | integer | `32` | May be omitted or null; ≥1, ≤32 | Maximum receive TBs per channel. |
| `runtime.limits.dependency_id` | integer | `127` | May be omitted or null; ≥0, ≤127 | Maximum dependency ID. |
| `runtime.limits.buffer_offset` | integer | `32767` | May be omitted or null; ≥0, ≤32767 | Maximum buffer slice offset. |
| `runtime.tb_layout` | string | `"separate"` | May be omitted or null; separate / fused | Separate send/receive layout by default; fused requires explicit selection. |
| `runtime.control` | object | `{}` | May be omitted or null; — | Rank synchronization configuration for one execution. |
| `runtime.control.bind_address` | string | `unspecified` | Checked when enabled; — | Reachable listening address of rank 0. |
| `runtime.control.port` | integer | `unspecified` | Checked when enabled; ≥1, ≤65535 | Control TCP port. |
