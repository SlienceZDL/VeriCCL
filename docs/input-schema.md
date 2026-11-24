# Input field reference

This table is generated from `vericcl/input/schema.py`. All four JSON inputs reject duplicate keys, unknown fields, and non-finite numbers; original files are never overwritten. An omitted or null value is unspecified: a default is applied when available, and a required field without a default raises an error when its feature is enabled. A field present in both its old and new locations raises an error even if the values match.

M must be divisible by S, and S must contain complete datatype elements. For AllToAll/ReduceScatter, N=M/S must also be divisible by the rank count. Do not provide N or input_chunkup directly. Omit root/reduction_op or use null for operators to which they do not apply; concrete values are rejected.

Use `--set path=JSON` to fill unspecified values. Add `--override-input` to replace a different explicit value. Original files are stored in `input_snapshot`; the effective configuration and value origins are written to `effective-request.json` and `value-sources.json`. Write actual absolute paths in JSON; configuration fields do not expand shell variables.

solve/live require an initial synthesis budget and a round limit, plus repair budgets when the round limit exceeds zero. Processing commands require per-model and CPU limits, per-round and total offline budgets, and a cleanup reserve. Calibration requires online budgets, a fit-error tolerance, and a job timeout. Trace and measurement require trace capacity and a job timeout. History reuse requires an index and the relevant tolerance or age limit. live requires all seven phase deadlines and control endpoints. Runtime numerical checks use explicit absolute and relative tolerances.

JSON Schema describes the standard structure; load_request checks cross-field arithmetic, enabling conditions, duplicate keys, and explicitly supported legacy migrations. Operator names are case-insensitive; compatible aliases are all_gather, all_reduce, all_to_all, and reducescatter. Explicitly listed solver parameters in legacy hyperparameters can be migrated, as can symmetry, shortest_paths, batching, and objective_mode at the atom top level. Legacy total_solve_timeout_s migrates only to the initial synthesis budget. force_resolve, force_recalibrate, and legacy algorithm boolean switches have been removed.

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
| `topology.directed_links[].max_channels` | integer | `16` | May be omitted or null; ≥1, ≤16 | Maximum channel count: 8 by default, or up to 16 when explicitly configured. |
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
| `sketch.solver.max_channels` | integer | `8` | May be omitted or null; ≥1, ≤16 | Maximum channel count: 8 by default, or up to 16 when explicitly configured. |
| `sketch.solver.channel_candidates` | array | `[1, 2, 3, 4, 5, 6, 7, 8]` | May be omitted or null; — | Candidate channel counts, defaulting to 1 through 8. Any integer in the permitted range is supported, subject to max_channels and runtime limits. |
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
| `sketch.calibration.chain_steps` | integer | `8` | May be omitted or null; ≥1, ≤9223372036854775807 | Consecutive steps per probe flow on one TB. Fitting uses time per step to capture pipelining; set to 1 for single-step probes. |
| `sketch.calibration.chain_max_bytes` | integer | `4194304` | May be omitted or null; ≥1, ≤9223372036854775807 | Maximum total bytes in a probe chain. Large payloads shorten the chain to at least one step, limiting calibration time. |
| `sketch.calibration.equivalence_classes` | boolean | `true` | May be omitted or null; — | Group structurally equivalent links and concurrency patterns. Measure one representative per class and apply qualified results to its members. |
| `sketch.calibration.candidate_limit` | integer | `2` | May be omitted or null; ≥1, ≤9223372036854775807 | Candidates per slice size selected by the initial prediction for first-round calibration; measure the patterns these candidates use. |
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
| `sketch.symbolic` | object | `unspecified` | Checked when enabled; — | Settings for full symbolic execution and BDD analysis. Omitting this section preserves the standard workflow and its input hash. |
| `sketch.symbolic.enabled` | boolean | `false` | May be omitted or null; — | Enable the feature in this configuration section. |
| `sketch.symbolic.loop_iterations` | integer | `2` | May be omitted or null; ≥1, ≤9223372036854775807 | Number of MSCCL loop iterations in the progress model. Dependency flags and connection FIFOs persist between iterations; two iterations expose cross-iteration waits. |
| `sketch.symbolic.fifo_slots` | integer | `8` | May be omitted or null; ≥1, ≤9223372036854775807 | Buffer slots per connection, corresponding to runtime NCCL_STEPS. |
| `sketch.symbolic.chunk_steps` | integer | `4` | May be omitted or null; ≥1, ≤9223372036854775807 | FIFO slots consumed by a full MSCCL chunk, corresponding to MSCCL_CHUNKSTEPS. Message capacity is derived for the size being checked. |
| `sketch.symbolic.max_related_operations` | integer | `48` | May be omitted or null; ≥1, ≤9223372036854775807 | Maximum related operations per hotspot, including resource competitors, ordering predecessors, and dependencies leading to completion. |
| `sketch.symbolic.max_order_hints` | integer | `32` | May be omitted or null; ≥1, ≤9223372036854775807 | Maximum TB-order suggestions per analysis round. |
| `sketch.symbolic.max_plan_choices` | integer | `6` | May be omitted or null; ≥1, ≤9223372036854775807 | Candidate paths per transfer considered by the BDD decision filter during bottleneck repair. |
| `sketch.repair` | object | `{}` | May be omitted or null; — | Division of analysis and solving work during repair. |
| `sketch.repair.mode` | string | `"strategy_first"` | May be omitted or null; strategy_first / joint | strategy_first selects a strategy first; joint selects valid decisions jointly. |
| `sketch.repair.candidate_batch_size` | integer | `4` | May be omitted or null; ≥1, ≤9223372036854775807 | Candidates sent to local SCIP per batch; does not limit the total search. |
| `sketch.repair.analysis_batch_size` | integer | `16` | May be omitted or null; ≥1, ≤9223372036854775807 | Analysis suggestions per batch; expand further when exhausted. |
| `sketch.repair.refinement_factors` | array | `[2, 4, 8]` | May be omitted or null; — | Slice subdivision factors tried first when expanding the search. Preserve routes, channels, and TB order while splitting each slice into factor pieces to enable pipelining. Acceptance requires model and paired-measurement gains. An empty list disables refinement. |
| `sketch.repair.hotspot` | object | `{}` | May be omitted or null; — | Hotspot analysis settings for symbolic execution, BDDs, and counters. |
| `sketch.repair.hotspot.bucket_count` | integer | `64` | May be omitted or null; ≥1, ≤9223372036854775807 | Number of time buckets; each bucket spans T_pred / bucket_count. |
| `sketch.repair.hotspot.wait_threshold_ratio` | number | `0.05` | May be omitted or null; ≥0 | Contention slowdown or TB waiting above this fraction of service time is classified as congestion. |
| `sketch.repair.hotspot.idle_utilization_threshold` | number | `0.5` | May be omitted or null; ≥0, ≤1 | A link below this utilization within a time bucket is treated as idle or lightly used. |
| `sketch.repair.hotspot.max_alt_hops` | integer | `3` | May be omitted or null; ≥1, ≤9223372036854775807 | Maximum hops in an alternative path. |
| `sketch.repair.hotspot.max_hotspots` | integer | `16` | May be omitted or null; ≥1, ≤9223372036854775807 | Maximum hotspots reported per round. |
| `sketch.repair.hotspot.tail_only` | boolean | `true` | May be omitted or null; — | Generate repairs only for hotspots that can delay completion, either on the critical path or competing for a congested resource used by critical operations. |
| `sketch.repair.correctness` | object | `{}` | May be omitted or null; — | Type 1 repair: locally repair violations of semantics, progress, forbidden edges, or buffer access rules, then recheck the candidate. |
| `sketch.repair.correctness.max_steps` | integer | `8` | May be omitted or null; ≥1, ≤9223372036854775807 | Maximum correctness-repair attempts per candidate. |
| `sketch.repair.reallocation` | object | `{}` | May be omitted or null; — | Type 2 repair: redistribute existing slices between flows of a chunk when a hot flow and an idle alternative coexist. Slice geometry stays fixed. |
| `sketch.repair.reallocation.max_chunks` | integer | `8` | May be omitted or null; ≥1, ≤9223372036854775807 | Maximum chunks jointly adjusted by one allocation repair, including chunks competing for the hot resource. |
| `sketch.repair.reallocation.max_incremental_moves` | integer | `4` | May be omitted or null; ≥0, ≤9223372036854775807 | Maximum additional candidates that move slices one at a time from hot to idle flows, alongside the balanced allocation. |
| `sketch.repair.bottleneck` | object | `{}` | May be omitted or null; — | Type 3 repair: identify related operations symbolically, filter alternative forwarding plans using BDDs and cost estimates, and solve the bounded region with local MILP. |
| `sketch.repair.bottleneck.max_scope_operations` | integer | `48` | May be omitted or null; ≥1, ≤9223372036854775807 | Maximum operations in a bottleneck-repair region, including rerouted transfers and downstream consumers. |
| `sketch.repair.bottleneck.dependency_depth` | integer | `2` | May be omitted or null; ≥0, ≤9223372036854775807 | Number of downstream data-dependency levels included in the repair region. |
| `sketch.repair.bottleneck.plans_per_slice` | integer | `3` | May be omitted or null; ≥1, ≤9223372036854775807 | Plans retained per reroutable transfer after filtering. |
| `sketch.repair.bottleneck.max_plans` | integer | `24` | May be omitted or null; ≥1, ≤9223372036854775807 | Maximum filtered plans passed to local MILP. |
| `sketch.runtime_compatibility` | object | `{}` | May be omitted or null; — | Full static runtime compatibility checks, which can be disabled; basic launch safety checks always remain active. |
| `sketch.runtime_compatibility.enabled` | boolean | `true` | May be omitted or null; — | Enable the feature in this configuration section. |
| `sketch.size_bands` | array | `[]` | May be omitted or null; — | Up to four ordered, non-overlapping message-size bands, matching the MSCCL_XML_FILES limit. Synthesize, measure, and repair at each representative size; merge adjacent bands with identical programs. |
| `sketch.size_bands[].min_bytes` | integer | `unspecified` | Required; ≥0, ≤9223372036854775807 | Inclusive lower bound, defaulting to 0. |
| `sketch.size_bands[].max_bytes` | integer | `unspecified` | Required; ≥1, ≤9223372036854775807 | Exclusive upper bound, defaulting to 4294967296. |
| `sketch.size_bands[].representative_bytes` | integer | `unspecified` | Required; ≥1, ≤9223372036854775807 | Representative size within [min_bytes, max_bytes), in MSCCL selection units: the nccl-tests size column, or total output bytes for AllGather. Converted to per-rank input bytes using collective semantics. |
| `sketch.size_bands[].slice_size_bytes` | integer | `null` | May be omitted or null; ≥1, ≤9223372036854775807 | Fixed minimum slice size S in bytes; if omitted, search for a valid partition first. N is derived as M/S. |
| `sketch.history` | object | `{}` | May be omitted or null; — | Cross-task schedule and absolute calibration caches; the two switches are independent and disabled by default. |
| `sketch.history.schedule_reuse` | boolean | `false` | May be omitted or null; — | Find historical schedules with the same graph structure and similar relative bandwidths, then adapt and revalidate them. |
| `sketch.history.calibration_reuse` | boolean | `false` | May be omitted or null; — | Reuse only valid measurement components with exactly matching device, runtime, datatype, and mode identities. |
| `sketch.history.relative_bandwidth_tolerance` | number | `unspecified` | Checked when enabled; ≥0 | Maximum allowed value of the largest mapped bandwidth scale divided by the smallest, minus one. |
| `sketch.history.max_schedule_candidates` | integer | `unspecified` | Checked when enabled; ≥1, ≤9223372036854775807 | Maximum candidates checked by fast history lookup. |
| `sketch.history.calibration_max_age_s` | number | `unspecified` | Checked when enabled; >0 | Maximum calibration cache age. |
| `sketch.history.schedule_index` | string | `unspecified` | Checked when enabled; — | Historical schedule index path; resolve relative paths from the case directory. |
| `sketch.history.calibration_index` | string | `unspecified` | Checked when enabled; — | Calibration index path; resolve relative paths from the case directory. |
| `sketch.live` | object | `{}` | May be omitted or null; — | Optional resident execution mode; the main offline loop runs independently. |
| `sketch.live.enabled` | boolean | `false` | May be omitted or null; — | Enable the feature in this configuration section. |
| `sketch.live.request_prepare_timeout_s` | number | `unspecified` | Checked when enabled; >0 | Deadline for request preparation and input protection. |
| `sketch.live.execution_attempt_timeout_s` | number | `unspecified` | Checked when enabled; >0 | Deadline for each GPU execution attempt. |
| `sketch.live.background_tuning_task_timeout_s` | number | `unspecified` | Checked when enabled; >0 | Deadline for one background tuning task. |
| `sketch.live.version_prepare_timeout_s` | number | `unspecified` | Checked when enabled; >0 | Deadline for preparing a new version. |
| `sketch.live.control_phase_timeout_s` | number | `unspecified` | Checked when enabled; >0 | Deadline for one control phase. |
| `sketch.live.recovery_timeout_s` | number | `unspecified` | Checked when enabled; >0 | Fixed total deadline for failure recovery. |
| `sketch.live.close_cleanup_timeout_s` | number | `unspecified` | Checked when enabled; >0 | Shared deadline for shutdown, background-task completion, and resource cleanup. |
| `sketch.live.window_requests` | integer | `8` | May be omitted or null; ≥1, ≤9223372036854775807 | Requests per qualified telemetry window. |
| `sketch.live.consecutive_windows` | integer | `2` | May be omitted or null; ≥1, ≤9223372036854775807 | Consecutive windows with a consistent change required to trigger an update. |
| `sketch.live.relative_change_threshold` | number | `0.1` | May be omitted or null; >0, ≤0.99 | Change threshold relative to the baseline. |
| `sketch.live.baseline_max_cv` | number | `0.05` | May be omitted or null; ≥0, ≤0.05 | Maximum coefficient of variation for establishing a stable baseline. |
| `sketch.live.min_residence_requests` | integer | `32` | May be omitted or null; ≥0, ≤9223372036854775807 | Minimum successful requests after adopting a version before an ordinary update. |
| `sketch.live.max_retries` | integer | `2` | May be omitted or null; ≥0, ≤4294967295 | Additional full-request retries after failure; defaults to 2. |
| `sketch.live.external_telemetry_path` | string | `unspecified` | Checked when enabled; — | Path to the caller-maintained JSONL file containing in-band telemetry. |
| `sketch.live.external_max_age_s` | number | `60` | May be omitted or null; >0 | Maximum accepted age of external telemetry. |
| `sketch.live.fault_injection` | object | `{}` | May be omitted or null; — | Development fault injection, enabled explicitly. |
| `sketch.live.fault_injection.request_id` | integer | `0` | May be omitted or null; ≥0, ≤9223372036854775807 | Logical request to fail; 0 disables injection. |
| `sketch.live.fault_injection.attempts` | integer | `0` | May be omitted or null; ≥0, ≤4294967295 | Initial attempts of that request marked as failed after actual completion. |

## atom.json

[Machine-readable Schema](input-schemas/atom.json)

| Field | Type | Default | Requirements and values | Purpose |
|---|---|---|---|---|
| `atom.synthesis_method` | string | `"auto"` | May be omitted or null; auto / hierarchy / constructive_trees | auto selects hierarchy below M_switch and constructive_trees for larger messages. |
| `atom.M_switch` | integer | `unspecified` | Checked when enabled; ≥1, ≤9223372036854775807 | Explicit message-size threshold in bytes, required for auto selection. |
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
| `runtime.environment.NCCL_IB_DISABLE` | string | `unspecified` | Checked when enabled; — | Disable IB/RoCE and use sockets. |
| `runtime.environment.NCCL_IB_GID_INDEX` | string | `unspecified` | Checked when enabled; — | Explicit GID index. |
| `runtime.environment.NCCL_IB_HCA` | string | `unspecified` | Checked when enabled; — | Explicitly selected IB/RoCE devices. |
| `runtime.environment.NCCL_IGNORE_DISABLED_P2P` | string | `unspecified` | Checked when enabled; — | Suppress warnings about disabled P2P. |
| `runtime.environment.NCCL_NET_GDR_LEVEL` | string | `unspecified` | Checked when enabled; — | Maximum topology distance for GPUDirect RDMA. |
| `runtime.environment.NCCL_NET_GDR_READ` | string | `unspecified` | Checked when enabled; — | Use GPUDirect RDMA reads on the sending side. |
| `runtime.environment.NCCL_P2P_DISABLE` | string | `unspecified` | Checked when enabled; — | Whether to disable GPU P2P paths. |
| `runtime.environment.NCCL_P2P_LEVEL` | string | `unspecified` | Checked when enabled; — | Maximum topology distance for GPU P2P. |
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
| `runtime.paths.vericcl_native` | string | `unspecified` | Checked when enabled; — | Absolute path to vericcl-native for full-schedule and resident execution. |
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
| `runtime.xml_size_range` | object | `{}` | May be omitted or null; — | Message-size interval [min_bytes, max_bytes) in which MSCCL selects this XML. |
| `runtime.xml_size_range.min_bytes` | integer | `0` | May be omitted or null; ≥0, ≤9223372036854775807 | Inclusive lower bound, defaulting to 0. |
| `runtime.xml_size_range.max_bytes` | integer | `4294967296` | May be omitted or null; ≥1, ≤9223372036854775807 | Exclusive upper bound, defaulting to 4294967296. |
| `runtime.control` | object | `{}` | May be omitted or null; — | Resident control-plane settings. |
| `runtime.control.bind_address` | string | `unspecified` | Checked when enabled; — | Reachable listening address of rank 0. |
| `runtime.control.port` | integer | `unspecified` | Checked when enabled; ≥1, ≤65535 | Control TCP port. |
| `runtime.control.unix_socket` | string | `unspecified` | Checked when enabled; — | Absolute Unix socket path on the controller host. |
