"""Generate field references and JSON Schema from the implementation to keep documentation in sync."""
import argparse
import json
from pathlib import Path

from vericcl.input.schema import SCHEMAS, public_schemas

HELP = {
    'name': 'Topology name, used only for identification.', 'ranks': 'Number of logical ranks in the task; within a node, the list of its ranks.',
    'nodes': 'Node groups covering every rank; node IDs do not replace actual host addresses.', 'id': 'Unique identifier within this configuration.',
    'gateways': 'Ranks that can serve as external connection endpoints for this node.', 'directed_links': 'Directed links, either direct or abstracted through switches; provide the reverse direction separately.',
    'src': 'Sending rank.', 'dst': 'Receiving rank.', 'alpha_us': 'Fixed service overhead per operation, in microseconds.',
    'beta_us_per_byte': 'Service cost per byte, in microseconds per byte; bandwidth is its reciprocal.',
    'max_channels': 'Maximum channel count: 8 by default, or up to 16 when explicitly configured.',
    'channel_candidates': 'Candidate channel counts, defaulting to 1 through 8. Any integer in the permitted range is supported, subject to max_channels and runtime limits.',
    'xml_size_range': 'Message-size interval [min_bytes, max_bytes) in which MSCCL selects this XML.',
    'min_bytes': 'Inclusive lower bound, defaulting to 0.', 'max_bytes': 'Exclusive upper bound, defaulting to 4294967296.',
    'hotspot': 'Hotspot analysis settings for symbolic execution, BDDs, and counters.',
    'bucket_count': 'Number of time buckets; each bucket spans T_pred / bucket_count.',
    'wait_threshold_ratio': 'Contention slowdown or TB waiting above this fraction of service time is classified as congestion.',
    'symbolic': 'Settings for full symbolic execution and BDD analysis. Omitting this section preserves the standard workflow and its input hash.',
    'loop_iterations': 'Number of MSCCL loop iterations in the progress model. Dependency flags and connection FIFOs persist between iterations; two iterations expose cross-iteration waits.',
    'fifo_slots': 'Buffer slots per connection, corresponding to runtime NCCL_STEPS.',
    'chunk_steps': 'FIFO slots consumed by a full MSCCL chunk, corresponding to MSCCL_CHUNKSTEPS. Message capacity is derived for the size being checked.',
    'max_related_operations': 'Maximum related operations per hotspot, including resource competitors, ordering predecessors, and dependencies leading to completion.',
    'max_order_hints': 'Maximum TB-order suggestions per analysis round.',
    'max_plan_choices': 'Candidate paths per transfer considered by the BDD decision filter during bottleneck repair.',
    'chain_max_bytes': 'Maximum total bytes in a probe chain. Large payloads shorten the chain to at least one step, limiting calibration time.',
    'chain_steps': 'Consecutive steps per probe flow on one TB. Fitting uses time per step to capture pipelining; set to 1 for single-step probes.',
    'equivalence_classes': 'Group structurally equivalent links and concurrency patterns. Measure one representative per class and apply qualified results to its members.',
    'candidate_limit': 'Candidates per slice size selected by the initial prediction for first-round calibration; measure the patterns these candidates use.',
    'tail_only': 'Generate repairs only for hotspots that can delay completion, either on the critical path or competing for a congested resource used by critical operations.',
    'correctness': 'Type 1 repair: locally repair violations of semantics, progress, forbidden edges, or buffer access rules, then recheck the candidate.',
    'max_steps': 'Maximum correctness-repair attempts per candidate.',
    'reallocation': 'Type 2 repair: redistribute existing slices between flows of a chunk when a hot flow and an idle alternative coexist. Slice geometry stays fixed.',
    'max_chunks': 'Maximum chunks jointly adjusted by one allocation repair, including chunks competing for the hot resource.',
    'max_incremental_moves': 'Maximum additional candidates that move slices one at a time from hot to idle flows, alongside the balanced allocation.',
    'bottleneck': 'Type 3 repair: identify related operations symbolically, filter alternative forwarding plans using BDDs and cost estimates, and solve the bounded region with local MILP.',
    'max_scope_operations': 'Maximum operations in a bottleneck-repair region, including rerouted transfers and downstream consumers.',
    'dependency_depth': 'Number of downstream data-dependency levels included in the repair region.',
    'plans_per_slice': 'Plans retained per reroutable transfer after filtering.',
    'max_plans': 'Maximum filtered plans passed to local MILP.',
    'idle_utilization_threshold': 'A link below this utilization within a time bucket is treated as idle or lightly used.',
    'max_alt_hops': 'Maximum hops in an alternative path.', 'max_hotspots': 'Maximum hotspots reported per round.',
    'resources': 'Identifiers of shared resources used by this link.', 'physical_path': 'Description of the physical path represented by the abstract link.',
    'shared_resources': 'Shared bottlenecks across abstract links, preventing physical bandwidth from being counted more than once.',
    'capacity_bytes_per_us': 'Total effective resource bandwidth, in bytes per microsecond.', 'links': 'Directed links [src,dst] belonging to this resource.',
    'kind': 'Shared resource category.', 'groups': 'User-provided candidate groups for hierarchical planning.', 'local_costs': 'Initial local cost estimates for the current datatype and reduction operation.',
    'rank': 'Logical rank ID, starting at 0.', 'copy': 'Initial cost of local reads, writes, or copies.', 'reduce': 'Initial cost of local reduction.',
    'collective': 'Application-visible collective semantics.', 'operator': 'Collective operator; internal manual nodes also support Scatter/Gather.',
    'datatype': 'Datatype, defaulting to float32; actual support must pass runtime checks.',
    'root': "External root for Broadcast/Reduce, defaulting to 0; for a manual node, that node's root.",
    'reduction_op': 'Reduction operation for Reduce/AllReduce/ReduceScatter, defaulting to sum. Omit or use null for other external operators.',
    'inplace': 'Whether to use a valid in-place buffer layout; disabled by default.', 'hyperparameters': 'Application input size and minimum slice size.',
    'total_size_bytes': 'Input bytes M per rank; for Broadcast, the root message size.',
    'slice_size_bytes': 'Fixed minimum slice size S in bytes; if omitted, search for a valid partition first. N is derived as M/S.',
    'reference_slice_size_bytes': 'Reference partition for hard constraints on original slice_id values; repartitioning preserves their original byte ranges.',
    'solver': 'Solver, CPU quota, and solving strategies.', 'backend': 'Only SCIP is supported; its absence causes an immediate error.',
    'per_model_timeout_s': 'Time limit for one solver model, also bounded by the remaining parent-phase budget.', 'cpu_budget': 'CPU quota available to the entire task.',
    'max_threads_per_model': 'Thread limit per model, defaulting to 12; also bounded by the CPU quota.',
    'max_parallel_models': 'Maximum number of concurrent models; managed by the budget manager if omitted.',
    'solver_seed': 'SCIP random seed.', 'mip_gap': 'Allowed numerical MIP gap; it does not prove coverage of the full repair space.',
    'require_proven_optimal': 'Whether the current solver model must reach numerical optimality.', 'bdd_backend': 'BDD backend for analysis; CUDD is the default, and autoref requires explicit selection.',
    'budgets': 'Phase time limits and maximum tuning rounds; all times are in seconds.',
    'max_initial_synthesis_s': 'Total time limit for initial synthesis in this run.', 'max_online_validation_per_round_s': 'Cumulative limit for online calibration, tracing, and measurement in one round.',
    'max_online_validation_total_s': 'Cumulative online validation limit for the entire task.', 'max_offline_validation_per_round_s': 'Cumulative limit for offline checks and analysis in one round.',
    'max_offline_validation_total_s': 'Cumulative limit for offline checks and analysis across the task.', 'max_repair_per_round_s': 'Cumulative limit for all local, structural, and partition repairs in one round.',
    'max_repair_total_s': 'Cumulative repair limit for the entire task.', 'max_run_wall_s': 'Optional wall-clock time limit for the entire task.',
    'cleanup_reserve_s': 'Cleanup time reserved from the remaining budget; does not extend the parent deadline.', 'max_tuning_rounds': 'Maximum number of tuning rounds to start; 0 stops after establishing the initial result.',
    'tolerances': 'Floating-point comparison tolerances, which must be provided explicitly for execution.', 'absolute': 'Absolute error tolerance.', 'relative': 'Relative error tolerance.',
    'calibration': 'Independent primitive measurement and cost-model calibration.', 'enabled': 'Enable the feature in this configuration section.',
    'launches': 'Independent process launches per round; iterations within one launch do not count as independent launches.',
    'warmup_iterations': 'Warmup iterations per launch; detailed tracing resets after warmup and records one iteration.',
    'timed_iterations': 'Timed iterations per launch without detailed tracing.', 'max_rounds': 'Maximum measurement rounds when stability criteria are unmet.',
    'max_cv': 'Maximum coefficient of variation among samples in one round.', 'max_relative_fit_error': 'Maximum allowed relative error in cost fitting.',
    'job_timeout_s': 'Time limit for one complete measurement job, including launch, retries, collection, and cleanup.', 'payload_sizes': 'Additional probe sizes; do not change the application M/S.',
    'trace_and_measure': 'One switch controls detailed tracing and the formal measured-gain threshold for candidates; disabled by default.',
    'pairs': 'Predetermined number of paired AB/BA launches; must be an even number of at least 2.', 'min_relative_gain': 'Minimum relative improvement required in measurements.',
    'task_alpha': 'Statistical error probability budget for the entire task, at most 0.05.',
    'single_attempt_alpha_cap': 'Error probability limit for one comparison, also subject to allocation by the task ledger.',
    'seed': 'Seed that fixes the sampling order in advance.', 'trace_capacity_bytes': 'Detailed trace buffer bytes per rank; must accommodate the file header and all trace records for one complete invocation.',
    'repair': 'Division of analysis and solving work during repair.', 'mode': 'strategy_first selects a strategy first; joint selects valid decisions jointly.',
    'candidate_batch_size': 'Candidates sent to local SCIP per batch; does not limit the total search.', 'analysis_batch_size': 'Analysis suggestions per batch; expand further when exhausted.', 'refinement_factors': 'Slice subdivision factors tried first when expanding the search. Preserve routes, channels, and TB order while splitting each slice into factor pieces to enable pipelining. Acceptance requires model and paired-measurement gains. An empty list disables refinement.',
    'runtime_compatibility': 'Full static runtime compatibility checks, which can be disabled; basic launch safety checks always remain active.',
    'history': 'Cross-task schedule and absolute calibration caches; the two switches are independent and disabled by default.',
    'schedule_reuse': 'Find historical schedules with the same graph structure and similar relative bandwidths, then adapt and revalidate them.',
    'calibration_reuse': 'Reuse only valid measurement components with exactly matching device, runtime, datatype, and mode identities.',
    'relative_bandwidth_tolerance': 'Maximum allowed value of the largest mapped bandwidth scale divided by the smallest, minus one.',
    'max_schedule_candidates': 'Maximum candidates checked by fast history lookup.', 'calibration_max_age_s': 'Maximum calibration cache age.',
    'schedule_index': 'Historical schedule index path; resolve relative paths from the case directory.', 'calibration_index': 'Calibration index path; resolve relative paths from the case directory.',
    'live': 'Optional resident execution mode; the main offline loop runs independently.',
    'request_prepare_timeout_s': 'Deadline for request preparation and input protection.', 'execution_attempt_timeout_s': 'Deadline for each GPU execution attempt.',
    'background_tuning_task_timeout_s': 'Deadline for one background tuning task.', 'version_prepare_timeout_s': 'Deadline for preparing a new version.',
    'control_phase_timeout_s': 'Deadline for one control phase.', 'recovery_timeout_s': 'Fixed total deadline for failure recovery.',
    'close_cleanup_timeout_s': 'Shared deadline for shutdown, background-task completion, and resource cleanup.',
    'window_requests': 'Requests per qualified telemetry window.', 'consecutive_windows': 'Consecutive windows with a consistent change required to trigger an update.',
    'relative_change_threshold': 'Change threshold relative to the baseline.', 'baseline_max_cv': 'Maximum coefficient of variation for establishing a stable baseline.',
    'min_residence_requests': 'Minimum successful requests after adopting a version before an ordinary update.',
    'max_retries': 'Additional full-request retries after failure; defaults to 2.', 'external_telemetry_path': 'Path to the caller-maintained JSONL file containing in-band telemetry.',
    'external_max_age_s': 'Maximum accepted age of external telemetry.', 'fault_injection': 'Development fault injection, enabled explicitly.',
    'request_id': 'Logical request to fail; 0 disables injection.', 'attempts': 'Initial attempts of that request marked as failed after actual completion.',
    'synthesis_method': 'auto selects hierarchy below M_switch and constructive_trees for larger messages.',
    'M_switch': 'Explicit message-size threshold in bytes, required for auto selection.', 'stage_num': 'Hard constraint on the number of logical stages; determined automatically if omitted.',
    'forbidden_transfers': '[original slice_id,src,dst,stage]; forbids only this direct edge in the specified stage, while valid alternate paths remain allowed.',
    'strategies': 'Optional initial solving strategies, disabled by default.', 'symmetry': 'Enable symmetry reduction.', 'shortest_paths': 'Enable candidate path reduction.',
    'batching': 'Explicitly enable batching for hierarchical solving; the constructive tree method manages batches by its own rules.',
    'objective_mode': 'Explicitly select a latency or throughput objective, or choose automatically by message size.',
    'manual': 'Explicit manual plan, used only when enabled=true.', 'stage_id': 'Logical stage ID, starting at 0.',
    'group': 'Ranks available to this manual node.', 'relay_ranks': 'Ranks allowed to relay within the group; ranks outside the group are not allowed.',
    'depends_on': 'IDs of nodes on which this manual node depends.', 'input_states': 'Input state contract for the manual node.', 'output_states': 'Output state contract for the manual node.',
    'offset': 'Logical data position ID.', 'contributors': 'Original slice_id values contained in this state.',
    'rank_map': 'Actual host and device placement for every rank; must be complete with no duplicate devices.', 'host': 'Hostname or address reachable through both SSH and MPI.',
    'device': 'CUDA device ID visible to the process on this host.', 'environment': 'Allowed runtime environment variables, all with string values.',
    'paths': 'Explicit toolchain and executable paths.', 'cuda_home': 'Root of an existing CUDA installation.', 'mpi_bin': 'Directory containing mpirun and mpicxx.',
    'msccl_root': 'Built MSCCL root, containing include/lib.', 'nccl_tests_root': 'Optional nccl-tests root directory.',
    'vericcl_driver': 'Absolute path to the independent primitive measurement program vericcl-probe.', 'vericcl_native': 'Absolute path to vericcl-native for full-schedule and resident execution.',
    'device_memory_budget_bytes': 'Total device memory budget; 0 uses available memory at launch as the bound, while still checking actual allocations.',
    'protocol': 'Currently fixed to Simple.', 'scratch_bytes_per_rank': 'Allowed scratch bytes per rank.',
    'limits': 'Runtime structural limits; may be reduced but cannot exceed compiled limits.', 'steps_per_tb': 'Maximum instructions per TB.',
    'tbs_per_rank': 'Maximum TBs per rank; must also fit the actual device residency capacity.',
    'send_tbs_per_channel': 'Maximum send TBs per channel.', 'recv_tbs_per_channel': 'Maximum receive TBs per channel.',
    'dependency_id': 'Maximum dependency ID.', 'buffer_offset': 'Maximum buffer slice offset.', 'tb_layout': 'Separate send/receive layout by default; fused requires explicit selection.',
    'size_bands': 'Up to four ordered, non-overlapping message-size bands, matching the MSCCL_XML_FILES limit. Synthesize, measure, and repair at each representative size; merge adjacent bands with identical programs.',
    'representative_bytes': 'Representative size within [min_bytes, max_bytes), in MSCCL selection units: the nccl-tests size column, or total output bytes for AllGather. Converted to per-rank input bytes using collective semantics.',
    'control': 'Resident control-plane settings.', 'bind_address': 'Reachable listening address of rank 0.', 'port': 'Control TCP port.',
    'unix_socket': 'Absolute Unix socket path on the controller host.',
}
ENV = {
    'PATH': 'Executable search path for child processes.', 'LD_LIBRARY_PATH': 'Dynamic library search path, prioritizing the target runtime and MPI.', 'PYTHONPATH': 'Python module search path.',
    'CUDA_HOME': 'CUDA root directory.', 'CUDA_PATH': 'Compatibility variable for the CUDA root directory.', 'CUDA_VISIBLE_DEVICES': 'Visible device set, which changes the meaning of device IDs.',
    'NCCL_ALGO': 'Retained NCCL algorithm interface.', 'NCCL_PROTO': 'Runtime protocol; tasks use Simple.',
    'NCCL_BUFFSIZE': 'Communication buffer size per channel, as a decimal byte string from 1 to 2147483647.',
    'NCCL_DEBUG': 'Runtime logging level.', 'NCCL_DEBUG_SUBSYS': 'Runtime logging subsystems.', 'NCCL_IB_HCA': 'Explicitly selected IB/RoCE devices.',
    'NCCL_IB_GID_INDEX': 'Explicit GID index.', 'NCCL_SOCKET_IFNAME': 'Communication network interface selection.', 'NCCL_P2P_DISABLE': 'Whether to disable GPU P2P paths.',
    'NCCL_SHM_DISABLE': 'Whether to disable host shared-memory paths.', 'NCCL_IB_DISABLE': 'Disable IB/RoCE and use sockets.',
    'NCCL_NET_GDR_LEVEL': 'Maximum topology distance for GPUDirect RDMA.', 'NCCL_NET_GDR_READ': 'Use GPUDirect RDMA reads on the sending side.',
    'NCCL_P2P_LEVEL': 'Maximum topology distance for GPU P2P.', 'NCCL_IGNORE_DISABLED_P2P': 'Suppress warnings about disabled P2P.',
    'MSCCL_XML_FILES': 'Existing MSCCL XML loading interface; the native versioned interface binds plans explicitly.',
    'NCCL_TESTS_ROOT': 'Optional nccl-tests root directory.', 'MSCCL_ROOT': 'Runtime root directory.', 'OMP_NUM_THREADS': 'OpenMP thread count.',
    'OMPI_MCA_btl': 'Open MPI BTL components.', 'OMPI_MCA_pml': 'Open MPI PML component.', 'UCX_NET_DEVICES': 'UCX network devices.', 'UCX_TLS': 'UCX transport selection.',
    'OMPI_MCA_btl_tcp_if_include': 'Interfaces or subnets for the Open MPI TCP data channel; nodes must be mutually reachable.',
    'OMPI_MCA_oob_tcp_if_include': 'Interfaces or subnets for the Open MPI control channel; may explicitly use the same subnet as the TCP data channel.',
}


def rows(spec, prefix):
    for key, child in spec.get('properties', {}).items():
        path = prefix + '.' + key
        description = ENV[key] if prefix == 'runtime.environment' else HELP[key]
        if path == 'atom.manual.nodes':
            description = 'Manual logical nodes; each specifies a group, operator, dependencies, and optional state contracts.'
        default = json.dumps(child['default'], ensure_ascii=False) if 'default' in child else 'unspecified'
        if path == 'sketch.collective.root': default = '0 when applicable'
        if path == 'sketch.collective.reduction_op': default = 'sum when applicable'
        limits = []
        for name, symbol in [('minimum', '≥'), ('exclusiveMinimum', '>'), ('maximum', '≤')]:
            if name in child: limits.append(symbol + str(child[name]))
        if 'enum' in child: limits.append(' / '.join(map(str, child['enum'])))
        required = 'Required' if key in spec.get('required', []) else 'Checked when enabled' if 'default' not in child else 'May be omitted or null'
        yield f'| `{path}` | {child["type"]} | `{default}` | {required}; {", ".join(limits) or "—"} | {description} |'
        yield from rows(child, path)
        if child['type'] == 'array': yield from rows(child['items'], path + '[]')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', required=True, type=Path)
    args = parser.parse_args(argv)
    directory = args.project_root / 'docs/input-schemas'; directory.mkdir(parents=True, exist_ok=True)
    for name, schema in public_schemas().items():
        (directory / (name + '.json')).write_text(json.dumps(schema, ensure_ascii=False, indent=2) + '\n')
    intro = '''# Input field reference

This table is generated from `vericcl/input/schema.py`. All four JSON inputs reject duplicate keys, unknown fields, and non-finite numbers; original files are never overwritten. An omitted or null value is unspecified: a default is applied when available, and a required field without a default raises an error when its feature is enabled. A field present in both its old and new locations raises an error even if the values match.

M must be divisible by S, and S must contain complete datatype elements. For AllToAll/ReduceScatter, N=M/S must also be divisible by the rank count. Do not provide N or input_chunkup directly. Omit root/reduction_op or use null for operators to which they do not apply; concrete values are rejected.

Use `--set path=JSON` to fill unspecified values. Add `--override-input` to replace a different explicit value. Original files are stored in `input_snapshot`; the effective configuration and value origins are written to `effective-request.json` and `value-sources.json`. Write actual absolute paths in JSON; configuration fields do not expand shell variables.

solve/live require an initial synthesis budget and a round limit, plus repair budgets when the round limit exceeds zero. Processing commands require per-model and CPU limits, per-round and total offline budgets, and a cleanup reserve. Calibration requires online budgets, a fit-error tolerance, and a job timeout. Trace and measurement require trace capacity and a job timeout. History reuse requires an index and the relevant tolerance or age limit. live requires all seven phase deadlines and control endpoints. Runtime numerical checks use explicit absolute and relative tolerances.

JSON Schema describes the standard structure; load_request checks cross-field arithmetic, enabling conditions, duplicate keys, and explicitly supported legacy migrations. Operator names are case-insensitive; compatible aliases are all_gather, all_reduce, all_to_all, and reducescatter. Explicitly listed solver parameters in legacy hyperparameters can be migrated, as can symmetry, shortest_paths, batching, and objective_mode at the atom top level. Legacy total_solve_timeout_s migrates only to the initial synthesis budget. force_resolve, force_recalibrate, and legacy algorithm boolean switches have been removed.

The object/array rows below describe configuration sections, and subsequent rows describe their members. Within arrays, [] denotes each element. See the [README](../README.md) for more workflows and [supported configurations](supported-configuration.md) for the scope of evidence.
'''
    parts = [intro]
    for name, spec in SCHEMAS.items():
        parts += [f'\n## {name}.json\n\n[Machine-readable Schema](input-schemas/{name}.json)\n',
                  '| Field | Type | Default | Requirements and values | Purpose |\n|---|---|---|---|---|', *rows(spec, name)]
    (args.project_root / 'docs/input-schema.md').write_text('\n'.join(parts) + '\n')
    print('Generated four schemas and the complete field reference.')


if __name__ == '__main__':
    main()
