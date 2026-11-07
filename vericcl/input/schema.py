"""Declarative v5 schemas, normalization provenance and cross-field contracts."""
from __future__ import annotations

import copy
import math
from pathlib import Path
from types import MappingProxyType

from vericcl.errors import InputValidationError
from vericcl.execution.security import ENVIRONMENT_ALLOWLIST
from vericcl.input.json_codec import sha256_json
from vericcl.input.migration import migrate, normalize_operator
from vericcl.input.models import (AtomConstraints, ForbiddenTransfer, Hyperparameters,
                                 ObjectiveMode, ResolvedInput, SolverConfig, StrategyConfig)
from vericcl.input.validation import validate_collective
from vericcl.semantics.collective import CollectiveKind, CollectiveSpec


DATATYPE_BYTES = {"float16": 2, "float32": 4, "float64": 8, "bfloat16": 2,
                  "int8": 1, "uint8": 1, "int32": 4, "uint32": 4, "int64": 8, "uint64": 8}
BUDGET_SECONDS = ("max_initial_synthesis_s", "max_online_validation_per_round_s", "max_online_validation_total_s",
                  "max_offline_validation_per_round_s", "max_offline_validation_total_s", "max_repair_per_round_s",
                  "max_repair_total_s", "max_run_wall_s", "cleanup_reserve_s")
_ABSENT = object()


def scalar(kind, *, default=_ABSENT, enum=None, minimum=None, maximum=None, exclusive_minimum=None):
    value = {"type": kind}
    if kind == "integer" and maximum is None:
        maximum = 2**63 - 1
    if default is not _ABSENT:
        value["default"] = default
    if enum is not None:
        value["enum"] = list(enum)
    if minimum is not None:
        value["minimum"] = minimum
    if maximum is not None:
        value["maximum"] = maximum
    if exclusive_minimum is not None:
        value["exclusiveMinimum"] = exclusive_minimum
    if kind == "string":
        value["minLength"] = 1
    return value


def obj(properties, *, required=(), default=_ABSENT):
    value = {"type": "object", "properties": properties, "required": list(required), "additionalProperties": False}
    if default is not _ABSENT:
        value["default"] = default
    return value


def array(items, *, default=_ABSENT, min_items=0, max_items=None):
    value = {"type": "array", "items": items, "minItems": min_items}
    if max_items is not None:
        value["maxItems"] = max_items
    if default is not _ABSENT:
        value["default"] = default
    return value


def positive(**kw):
    return scalar("number", exclusive_minimum=0, **kw)


def integer(**kw):
    return scalar("integer", minimum=0, **kw)


def flag(default=False):
    return scalar("boolean", default=default)


def section(properties):
    return obj(properties, default={})


TOPOLOGY = obj({
    "name": scalar("string"), "ranks": scalar("integer", minimum=1),
    "nodes": array(obj({"id": integer(), "ranks": array(integer(), min_items=1),
                        "gateways": array(integer(), default=[])}, required=("id", "ranks")), min_items=1),
    "directed_links": array(obj({
        "src": integer(), "dst": integer(), "alpha_us": scalar("number", minimum=0),
        "beta_us_per_byte": positive(), "max_channels": scalar("integer", minimum=1, maximum=16, default=16),
        "resources": array(scalar("string"), default=[]), "physical_path": array(scalar("string"), default=[]),
    }, required=("src", "dst", "alpha_us", "beta_us_per_byte"))),
    "shared_resources": array(obj({"id": scalar("string"), "capacity_bytes_per_us": positive(),
        "links": array(array(integer(), min_items=2, max_items=2), default=[]),
        "kind": scalar("string", enum=("nic", "pcie", "switch_port", "link", "other"), default="other")},
        required=("id", "capacity_bytes_per_us")), default=[]),
    "groups": array(array(integer(), min_items=1), default=[]),
    "local_costs": array(obj({"rank": integer(),
        "copy": obj({"alpha_us": scalar("number", minimum=0), "beta_us_per_byte": positive()}, required=("alpha_us", "beta_us_per_byte")),
        "reduce": obj({"alpha_us": scalar("number", minimum=0), "beta_us_per_byte": positive()}, required=("alpha_us", "beta_us_per_byte")),
    }, required=("rank",)), default=[]),
}, required=("ranks", "nodes", "directed_links"))

SOLVER = section({
    "backend": scalar("string", enum=("scip",), default="scip"), "per_model_timeout_s": positive(),
    "cpu_budget": scalar("integer", minimum=1), "max_threads_per_model": scalar("integer", minimum=1, default=12),
    "max_parallel_models": scalar("integer", minimum=1), "max_channels": scalar("integer", minimum=1, maximum=16, default=7),
    "solver_seed": integer(default=0), "mip_gap": scalar("number", minimum=0, maximum=1, default=0),
    "require_proven_optimal": flag(), "bdd_backend": scalar("string", enum=("cudd", "autoref"), default="cudd"),
})

SKETCH = obj({
    "collective": obj({"operator": scalar("string", enum=("broadcast", "reduce", "allgather", "allreduce", "alltoall", "reduce_scatter")),
        "datatype": scalar("string", enum=DATATYPE_BYTES, default="float32"), "root": integer(),
        "reduction_op": scalar("string", enum=("sum", "prod", "min", "max", "avg")), "inplace": flag()}, required=("operator",)),
    "hyperparameters": obj({"total_size_bytes": scalar("integer", minimum=1),
        "slice_size_bytes": scalar("integer", minimum=1, default=None),
        "reference_slice_size_bytes": scalar("integer", minimum=1, default=None)}, required=("total_size_bytes",)),
    "solver": SOLVER,
    "budgets": section({**{key: positive() for key in BUDGET_SECONDS}, "max_tuning_rounds": integer()}),
    "tolerances": section({"absolute": scalar("number", minimum=0), "relative": scalar("number", minimum=0)}),
    "calibration": section({"enabled": flag(True), "launches": scalar("integer", minimum=1, default=20),
        "warmup_iterations": integer(default=5), "timed_iterations": scalar("integer", minimum=1, default=20),
        "max_rounds": scalar("integer", minimum=1, default=3), "max_cv": positive(default=0.05),
        "max_relative_fit_error": scalar("number", minimum=0), "job_timeout_s": positive(),
        "payload_sizes": array(scalar("integer", minimum=1), default=[])}),
    "trace_and_measure": section({"enabled": flag(), "pairs": scalar("integer", minimum=1, default=20),
        "warmup_iterations": integer(default=5), "timed_iterations": scalar("integer", minimum=1, default=20),
        "max_rounds": scalar("integer", minimum=1, default=3), "max_cv": positive(default=0.05),
        "min_relative_gain": scalar("number", minimum=0, maximum=1, default=0),
        "task_alpha": scalar("number", exclusive_minimum=0, maximum=0.05, default=0.05),
        "single_attempt_alpha_cap": scalar("number", exclusive_minimum=0, maximum=0.05, default=0.05),
        "seed": integer(default=0), "job_timeout_s": positive(), "trace_capacity_bytes": scalar("integer", minimum=1)}),
    "repair": section({"mode": scalar("string", enum=("strategy_first", "joint"), default="strategy_first"),
        "candidate_batch_size": scalar("integer", minimum=1, default=4),
        "analysis_batch_size": scalar("integer", minimum=1, default=16)}),
    "runtime_compatibility": section({"enabled": flag(True)}),
    "history": section({"schedule_reuse": flag(), "calibration_reuse": flag(),
        "relative_bandwidth_tolerance": scalar("number", minimum=0), "max_schedule_candidates": scalar("integer", minimum=1),
        "calibration_max_age_s": positive(), "schedule_index": scalar("string"), "calibration_index": scalar("string")}),
}, required=("collective", "hyperparameters"))

MANUAL_STATE = obj({"rank": integer(), "offset": integer(), "contributors": array(integer(), min_items=1)},
                   required=("rank", "offset", "contributors"))
MANUAL_ATOM = obj({key: array(integer(), min_items=1) for key in ("slice_ids", "src_ranks", "dst_ranks")},
                  required=("slice_ids", "src_ranks", "dst_ranks"))
MANUAL_NODE = obj({"id": scalar("string"), "stage_id": integer(),
    "operator": scalar("string", enum=tuple(k.value for k in CollectiveKind)),
    "group": array(integer(), min_items=1), "root": integer(),
    "relay_ranks": array(integer(), default=[]),
    "atoms": array(MANUAL_ATOM, default=[]),
    "depends_on": array(scalar("string"), default=[]),
    "input_states": array(MANUAL_STATE, min_items=1), "output_states": array(MANUAL_STATE, min_items=1)},
    required=("id", "stage_id", "operator", "group"))
ATOM = obj({
    "synthesis_method": scalar("string", enum=("auto", "hierarchy", "constructive_trees"), default="auto"),
    "M_switch": scalar("integer", minimum=1), "stage_num": scalar("integer", minimum=1, default=None),
    "forbidden_transfers": array(array(integer(), min_items=4, max_items=4), default=[]),
    "strategies": section({"symmetry": flag(), "shortest_paths": flag(), "batching": flag(),
        "objective_mode": scalar("string", enum=("latency", "throughput", "auto"), default=None)}),
    "manual": section({"enabled": flag(), "nodes": array(MANUAL_NODE, default=[])}),
})

RUNTIME = obj({
    "rank_map": array(obj({"rank": integer(), "host": scalar("string"), "device": integer()}, required=("rank", "host", "device")), min_items=1),
    "environment": obj({key: scalar("string") for key in sorted(ENVIRONMENT_ALLOWLIST)}, required=("NCCL_BUFFSIZE",)),
    "paths": section({key: scalar("string") for key in ("cuda_home", "mpi_bin", "msccl_root", "nccl_tests_root", "vericcl_driver", "vericcl_native")}),
    "device_memory_budget_bytes": integer(default=0),
    "protocol": scalar("string", enum=("Simple",), default="Simple"),
    "scratch_bytes_per_rank": scalar("integer", minimum=0),
    "limits": section({"steps_per_tb": scalar("integer", minimum=1, maximum=256, default=256),
        "tbs_per_rank": scalar("integer", minimum=1, maximum=216, default=216),
        "send_tbs_per_channel": scalar("integer", minimum=1, maximum=32, default=32),
        "recv_tbs_per_channel": scalar("integer", minimum=1, maximum=32, default=32),
        "dependency_id": integer(maximum=127, default=127), "buffer_offset": integer(maximum=32767, default=32767)}),
    "tb_layout": scalar("string", enum=("separate", "fused"), default="separate"),
    "control": section({"bind_address": scalar("string"), "port": scalar("integer", minimum=1, maximum=65535)}),
}, required=("rank_map", "environment", "scratch_bytes_per_rank"))

SCHEMAS = {"topology": TOPOLOGY, "sketch": SKETCH, "atom": ATOM, "runtime": RUNTIME}


LIVE_MIGRATION_ERROR = ("sketch.live has been removed; use solve or verify --tune for offline tuning, "
                        "calibrate for calibration, and trace_and_measure for fixed-schedule execution")


def migrate_disabled_live(documents) -> bool:
    """Remove inactive legacy settings; callers retain checkpoint identity separately."""
    sketch = documents.get("sketch")
    if not isinstance(sketch, dict) or "live" not in sketch:
        return False
    live = sketch["live"]
    if live is not None and (not isinstance(live, dict) or
                            (live.get("enabled") is not None and live.get("enabled") is not False)):
        raise InputValidationError(LIVE_MIGRATION_ERROR)
    del sketch["live"]
    runtime = documents.get("runtime")
    if isinstance(runtime, dict) and isinstance(runtime.get("control"), dict):
        runtime["control"].pop("unix_socket", None)
    return True


def public_schemas() -> dict[str, dict]:
    """Describe canonical files, including null-as-unspecified semantics.

    Arithmetic, operator-dependent fields, enabled-module requirements and
    duplicate JSON keys are checked by load_request. Listed legacy spellings
    first pass through migration; the published files describe their targets.
    """
    def convert(spec, required=True):
        result = copy.deepcopy(spec)
        if spec["type"] == "object":
            result["properties"] = {key: convert(child, key in spec["required"])
                                     for key, child in spec["properties"].items()}
        elif spec["type"] == "array":
            result["items"] = convert(spec["items"])
        if not required or "default" in spec:
            result["type"] = [spec["type"], "null"]
            if "enum" in result:
                result["enum"].append(None)
        return result
    return {name: {"$schema": "https://json-schema.org/draft/2020-12/schema",
                   "title": "VeriCCL " + name, **convert(spec)} for name, spec in SCHEMAS.items()}


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


def _normalize(spec, value, path, sources, filename, origins, inherited_default=False):
    defaulted = value is None or value is _ABSENT
    if defaulted:
        if "default" not in spec:
            raise InputValidationError(path + " is required")
        value = copy.deepcopy(spec["default"])
    sources.setdefault(path, {"kind": "default" if defaulted or inherited_default else "file", "file": filename,
                              "original_path": origins.get(path, path)})
    if value is None and spec.get("default", _ABSENT) is None:
        return None
    kind = spec["type"]
    if kind == "object":
        if not isinstance(value, dict):
            raise InputValidationError(path + " must be an object")
        unknown = set(value) - spec["properties"].keys()
        if unknown:
            raise InputValidationError("unknown field " + path + "." + sorted(unknown)[0])
        result = {}
        for key, child in spec["properties"].items():
            item = value.get(key)
            if item is None and key not in spec["required"] and "default" not in child:
                continue
            result[key] = _normalize(child, value.get(key, _ABSENT), path + "." + key,
                                     sources, filename, origins, inherited_default or defaulted)
        return result
    if kind == "array":
        if not isinstance(value, list) or len(value) < spec.get("minItems", 0) or len(value) > spec.get("maxItems", math.inf):
            raise InputValidationError(path + " has invalid array length or type")
        return [_normalize(spec["items"], item, path + "[" + str(i) + "]", sources,
                           filename, origins, inherited_default or defaulted) for i, item in enumerate(value)]
    expected = {"integer": int, "number": (int, float), "boolean": bool, "string": str}[kind]
    if not isinstance(value, expected) or (kind in {"integer", "number"} and isinstance(value, bool)):
        raise InputValidationError(path + " must be " + kind)
    if kind in {"integer", "number"}:
        if (value < spec.get("minimum", -math.inf) or value > spec.get("maximum", math.inf)
                or value <= spec.get("exclusiveMinimum", -math.inf) or not math.isfinite(value)):
            raise InputValidationError(path + " is outside its finite numeric range")
    if kind == "string" and (len(value) < spec.get("minLength", 0) or "\x00" in value):
        raise InputValidationError(path + " must be a nonempty string without NUL")
    if "enum" in spec and value not in spec["enum"]:
        raise InputValidationError(path + " has unsupported value " + repr(value))
    return value


def _apply_overrides(documents, overrides, enabled, sources):
    for path, value in (overrides or {}).items():
        if not isinstance(path, str) or not path:
            raise InputValidationError("CLI override paths must be strings")
        if path == "sketch.live" or path.startswith("sketch.live."):
            raise InputValidationError(LIVE_MIGRATION_ERROR)
        if path == "sketch.collective.operator":
            value = normalize_operator(value)
        parts = path.split(".")
        if parts[0] not in SCHEMAS or len(parts) < 2:
            raise InputValidationError("unknown CLI field " + path)
        spec, node = SCHEMAS[parts[0]], documents[parts[0]]
        for key in parts[1:-1]:
            if spec.get("type") != "object" or key not in spec["properties"]:
                raise InputValidationError("unknown CLI field " + path)
            spec = spec["properties"][key]
            if node.get(key) is None:
                node[key] = {}
            if not isinstance(node[key], dict):
                raise InputValidationError("CLI field parent is not an object: " + path)
            node = node[key]
        key = parts[-1]
        if spec.get("type") != "object" or key not in spec["properties"]:
            raise InputValidationError("unknown CLI field " + path)
        previous = copy.deepcopy(node.get(key))
        if previous is not None and previous != value and not enabled:
            raise InputValidationError(path + " conflicts with input; use --override-input")
        node[key] = copy.deepcopy(value)
        sources[path] = {"kind": "cli", "previous": previous, "override": previous is not None and previous != value}


def _required(mapping, key, prefix):
    if key not in mapping or mapping[key] is None:
        raise InputValidationError(prefix + "." + key + " is required for the enabled operation")
    return mapping[key]


def _validate_topology(topology, runtime):
    ranks = topology["ranks"]
    membership, node_ids = [], set()
    for node in topology["nodes"]:
        if node["id"] in node_ids or not set(node["gateways"]) <= set(node["ranks"]):
            raise InputValidationError("invalid or duplicate topology node")
        node_ids.add(node["id"])
        membership.extend(node["ranks"])
    if len(membership) != ranks or any(r != i for i, r in enumerate(sorted(membership))):
        raise InputValidationError("topology nodes must cover each rank exactly once")
    cost_ranks = set()
    for entry in topology["local_costs"]:
        if entry["rank"] >= ranks or entry["rank"] in cost_ranks or not ({"copy", "reduce"} & entry.keys()):
            raise InputValidationError("local_costs requires a unique valid rank and at least one local component")
        cost_ranks.add(entry["rank"])
    resource_ids = [r["id"] for r in topology["shared_resources"]]
    if len(set(resource_ids)) != len(resource_ids):
        raise InputValidationError("duplicate shared resource")
    links = set()
    for edge in topology["directed_links"]:
        pair = edge["src"], edge["dst"]
        if max(pair) >= ranks or pair[0] == pair[1] or pair in links:
            raise InputValidationError("invalid or duplicate directed link")
        if not set(edge["resources"]) <= set(resource_ids):
            raise InputValidationError("directed link references an unknown shared resource")
        links.add(pair)
    for resource in topology["shared_resources"]:
        if not {tuple(edge) for edge in resource["links"]} <= links:
            raise InputValidationError("shared resource references an unknown directed link")
    for group in topology["groups"]:
        if len(set(group)) != len(group) or max(group) >= ranks:
            raise InputValidationError("invalid topology group")
    mapping = runtime["rank_map"]
    if len(mapping) != ranks or any(r != i for i, r in enumerate(sorted(item["rank"] for item in mapping))):
        raise InputValidationError("runtime.rank_map must contain every rank exactly once")
    if len({(item["host"], item["device"]) for item in mapping}) != ranks:
        raise InputValidationError("runtime.rank_map must use a distinct device for each rank on a host")
    raw = runtime["environment"]["NCCL_BUFFSIZE"]
    # Plan section 7.1: the valid native range is 8 through 2147483647 bytes.
    # Smaller values leave no Simple step (NCCL_STEPS=8) and previously ended
    # as "no initial candidate" instead of an input error.
    if not raw.isascii() or not raw.isdigit() or not 8 <= int(raw) <= 2**31 - 1:
        raise InputValidationError("NCCL_BUFFSIZE must be an integer byte count from 8 to 2147483647")


def _validate_enabled(config, purpose):
    sketch = config["sketch"]
    budgets, solver = sketch["budgets"], sketch["solver"]
    for key in ("per_model_timeout_s", "cpu_budget"):
        _required(solver, key, "sketch.solver")
    for key in ("cleanup_reserve_s", "max_offline_validation_per_round_s", "max_offline_validation_total_s"):
        _required(budgets, key, "sketch.budgets")
    if purpose == "solve":
        for key in ("max_initial_synthesis_s", "max_tuning_rounds"):
            _required(budgets, key, "sketch.budgets")
        if budgets["max_tuning_rounds"]:
            for key in ("max_repair_per_round_s", "max_repair_total_s"):
                _required(budgets, key, "sketch.budgets")
    calibration, measured = sketch["calibration"], sketch["trace_and_measure"]
    if calibration["enabled"] or measured["enabled"] or purpose == "calibrate":
        for key in ("max_online_validation_per_round_s", "max_online_validation_total_s"):
            _required(budgets, key, "sketch.budgets")
    if calibration["enabled"] or purpose == "calibrate":
        for key in ("max_relative_fit_error", "job_timeout_s"):
            _required(calibration, key, "sketch.calibration")
    if measured["pairs"] % 2 or measured["pairs"] < 2:
        raise InputValidationError("sketch.trace_and_measure.pairs must be even and at least two")
    if measured["max_rounds"] > 3 or measured["max_cv"] > .05 or measured["min_relative_gain"] >= 1:
        raise InputValidationError("trace_and_measure requires max_rounds <= 3, max_cv <= .05 and min_relative_gain < 1")
    if measured["enabled"]:
        for key in ("job_timeout_s", "trace_capacity_bytes"):
            _required(measured, key, "sketch.trace_and_measure")
    history = sketch["history"]
    if history["schedule_reuse"]:
        for key in ("relative_bandwidth_tolerance", "max_schedule_candidates", "schedule_index"):
            _required(history, key, "sketch.history")
    if history["calibration_reuse"]:
        for key in ("calibration_max_age_s", "calibration_index"):
            _required(history, key, "sketch.history")


def resolve_request(files, overrides=None, *, override_input=False, purpose="solve"):
    from vericcl.input.loader import _load_json_object
    if purpose == "live":
        raise InputValidationError(LIVE_MIGRATION_ERROR)
    if purpose not in {"solve", "verify", "calibrate"}:
        raise InputValidationError("invalid request purpose")
    if not isinstance(files, dict) or set(files) != set(SCHEMAS):
        raise InputValidationError("request files must explicitly contain topology, sketch, atom and runtime")
    documents = {name: _load_json_object(Path(path), name) for name, path in files.items()}
    origins = migrate(documents)
    migrate_disabled_live(documents)
    sources = {}
    _apply_overrides(documents, overrides, override_input, sources)
    config = {name: _normalize(spec, documents[name], name, sources, str(Path(files[name]).resolve()), origins)
              for name, spec in SCHEMAS.items()}
    topology, sketch, atom, runtime = (config[k] for k in ("topology", "sketch", "atom", "runtime"))
    _validate_topology(topology, runtime)
    ranks = topology["ranks"]
    collective = sketch["collective"]
    kind = CollectiveKind(collective["operator"])
    rooted = kind in {CollectiveKind.BROADCAST, CollectiveKind.REDUCE}
    reduction = kind in {CollectiveKind.REDUCE, CollectiveKind.ALL_REDUCE, CollectiveKind.REDUCE_SCATTER}
    for key, applies, default in (("root", rooted, 0), ("reduction_op", reduction, "sum")):
        if applies and key not in collective:
            collective[key] = default
            sources["sketch.collective." + key] = {"kind": "default", "file": str(files["sketch"])}
        elif not applies and key in collective:
            raise InputValidationError("collective." + key + " is not applicable to " + kind.value)
        elif not applies:
            collective[key] = None
    spec = CollectiveSpec(kind, collective["datatype"], collective["reduction_op"], collective["root"], collective["inplace"])
    hyper = sketch["hyperparameters"]
    size, slice_size, reference = hyper["total_size_bytes"], hyper["slice_size_bytes"], hyper["reference_slice_size_bytes"]
    element_size = DATATYPE_BYTES[spec.datatype]
    if size % element_size:
        raise InputValidationError("total_size_bytes must contain whole datatype elements")
    for value in (slice_size, reference):
        if value is not None and (value % element_size or size % value):
            raise InputValidationError("slice size must divide total_size_bytes and contain whole datatype elements")
    slice_count = None if slice_size is None else size // slice_size
    validate_collective(spec, ranks, slice_count if slice_count is not None else size // element_size)
    if reduction and spec.datatype in {"float16", "float32", "float64", "bfloat16"}:
        for key in ("absolute", "relative"):
            _required(sketch["tolerances"], key, "sketch.tolerances")
    selected = atom["synthesis_method"]
    if atom["manual"]["enabled"]:
        if selected == "constructive_trees":
            raise InputValidationError("manual stage plans require synthesis_method=hierarchy or auto")
        if sketch["history"]["schedule_reuse"]:
            raise InputValidationError("manual stage plans require history.schedule_reuse=false; subproblem reuse remains active")
        selected = "hierarchy"
    elif selected == "auto":
        threshold = _required(atom, "M_switch", "atom")
        selected = "hierarchy" if size < threshold else "constructive_trees"
    if selected == "constructive_trees" and atom["strategies"]["shortest_paths"]:
        raise InputValidationError("shortest_paths is only an initial hierarchy option")
    from vericcl.input.manual_atoms import expanded_forbidden_transfers
    forbidden_values = expanded_forbidden_transfers(atom)
    forbidden = []
    if forbidden_values:
        if reference is None and slice_size is not None:
            reference = slice_size
            sources["sketch.hyperparameters.reference_slice_size_bytes"] = {
                "kind": "derived", "expression": "original explicit slice_size_bytes", "value": reference}
        if reference is None:
            raise InputValidationError("reference_slice_size_bytes is required to bind slice_id constraints")
        hyper["reference_slice_size_bytes"] = reference
        for slice_id, src, dst, stage in forbidden_values:
            if (slice_id >= ranks * (size // reference) or max(src, dst) >= ranks or src == dst
                    or (atom["stage_num"] is not None and stage >= atom["stage_num"])):
                raise InputValidationError("forbidden transfer is outside its reference partition/rank/stage domain")
            forbidden.append(ForbiddenTransfer(slice_id, src, dst, stage))
    _validate_enabled(config, purpose)
    solver = sketch["solver"]
    budgets = sketch["budgets"]
    output_size = size * ranks if kind is CollectiveKind.ALL_GATHER else size // ranks if kind is CollectiveKind.REDUCE_SCATTER else size
    accounting = {"input_bytes_per_rank": size, "slice_size_bytes": slice_size, "slices_per_rank": slice_count,
                  "valid_output_bytes": output_size, "output_ranks": [spec.root] if kind is CollectiveKind.REDUCE else list(range(ranks)),
                  "nccl_tests_size_bytes": size * ranks if kind is CollectiveKind.ALL_GATHER else size,
                  "element_bytes": element_size}
    sources["derived.slices_per_rank"] = {"kind": "derived", "expression": "total_size_bytes / slice_size_bytes", "value": slice_count}
    return ResolvedInput(
        spec, Hyperparameters(size, slice_size, ObjectiveMode(atom["strategies"]["objective_mode"] or "auto"),
            max_calibration_channels=solver["max_channels"], min_expected_improvement=0, min_tuning_improvement=0,
            max_tuning_iterations=budgets.get("max_tuning_rounds", 0),
            total_verification_timeout_s=budgets["max_offline_validation_total_s"], reference_slice_size_bytes=reference),
        SolverConfig(total_solve_timeout_s=budgets.get("max_initial_synthesis_s", solver["per_model_timeout_s"]),
            per_model_timeout_s=solver["per_model_timeout_s"], mip_gap=solver["mip_gap"], require_proven_optimal=solver["require_proven_optimal"],
            solver_seed=solver["solver_seed"], max_channels=solver["max_channels"], max_threads_per_model=solver["max_threads_per_model"],
            max_parallel_models=solver.get("max_parallel_models", solver["cpu_budget"])),
        StrategyConfig(selected == "hierarchy", atom["strategies"]["symmetry"], atom["strategies"]["shortest_paths"],
            atom["strategies"]["batching"], selected == "constructive_trees", True,
            tuple(_freeze(n) for n in atom["manual"]["nodes"]) if atom["manual"]["enabled"] else ()),
        AtomConstraints(atom["stage_num"], tuple(forbidden)), ranks, _freeze(topology), _freeze(sketch), _freeze(atom), sha256_json(config),
        request_config=_freeze(config), value_sources=_freeze(sources), size_accounting=_freeze(accounting),
        required_capabilities=_freeze({**collective, "protocol": runtime["protocol"]}), selected_synthesis_method=selected)
