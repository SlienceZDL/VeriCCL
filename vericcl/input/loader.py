import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional, Tuple

from vericcl.constants import SOFTWARE_MAX_CONCURRENCY
from vericcl.errors import InputValidationError
from vericcl.input.json_codec import sha256_json
from vericcl.input.models import (
    AtomConstraints,
    ForbiddenTransfer,
    Hyperparameters,
    ObjectiveMode,
    ResolvedInput,
    SolverConfig,
    StrategyConfig,
)
from vericcl.input.validation import validate_collective
from vericcl.semantics.collective import CollectiveKind, CollectiveSpec


_MISSING = object()
_OPERATOR_ALIASES = {
    "all_gather": "allgather",
    "all_reduce": "allreduce",
    "all_to_all": "alltoall",
    "reducescatter": "reduce_scatter",
}
_COLLECTIVE_KEYS = frozenset(
    {"operator", "root", "datatype", "reduction_op", "inplace"}
)
_HYPERPARAMETER_KEYS = frozenset(
    {
        "total_size_bytes",
        "slice_size_bytes",
        "input_chunkup",
        "objective_mode",
        "max_calibration_channels",
        "min_expected_improvement",
        "min_tuning_improvement",
        "max_tuning_iterations",
        "total_verification_timeout_s",
        "force_recalibrate",
    }
)
_SOLVER_KEYS = frozenset(
    {
        "total_solve_timeout_s",
        "per_model_timeout_s",
        "mip_gap",
        "require_proven_optimal",
        "solver_seed",
        "max_channels",
        "max_threads_per_model",
        "max_parallel_models",
        "force_resolve",
    }
)
_STRATEGY_DEFAULTS = {
    "hierarchy": False,
    "symmetry": False,
    "shortest_paths": False,
    "batching": False,
    "constructive_trees": True,
    "milp": True,
}
_ATOM_KEYS = frozenset(
    {
        "stage_num",
        "forbidden_transfers",
        "strategies",
        "manual_hierarchy",
    }
) | frozenset(_STRATEGY_DEFAULTS)


def _reject_nonfinite_json(value: str) -> None:
    raise InputValidationError("non-finite JSON number: {}".format(value))


def _reject_duplicate_keys(pairs: object) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise InputValidationError("duplicate JSON key: {}".format(key))
        result[key] = value
    return result


def _load_json_object(path: Path, label: str) -> dict:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise InputValidationError(
            "cannot read {} input {}: {}".format(label, path, error)
        ) from error
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except InputValidationError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise InputValidationError(
            "invalid JSON in {} input {}: {}".format(label, path, error)
        ) from error
    if not isinstance(value, dict):
        raise InputValidationError("{} input must be a JSON object".format(label))
    return value


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise InputValidationError("{} must be a JSON object".format(field))
    return value


def _reject_unknown_keys(
    value: Mapping[str, object],
    allowed: frozenset,
    field: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise InputValidationError(
            "unknown {} field: {}".format(field, unknown[0])
        )


def _integer(
    value: object,
    field: str,
    *,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InputValidationError("{} must be an integer".format(field))
    if minimum is not None and value < minimum:
        raise InputValidationError(
            "{} must be at least {}".format(field, minimum)
        )
    if maximum is not None and value > maximum:
        raise InputValidationError(
            "{} must be at most {}".format(field, maximum)
        )
    return value


def _number(
    value: object,
    field: str,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputValidationError("{} must be a number".format(field))
    result = float(value)
    if not math.isfinite(result):
        raise InputValidationError("{} must be finite".format(field))
    if minimum is not None and result < minimum:
        raise InputValidationError(
            "{} must be at least {}".format(field, minimum)
        )
    if maximum is not None and result > maximum:
        raise InputValidationError(
            "{} must be at most {}".format(field, maximum)
        )
    return result


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise InputValidationError("{} must be a boolean".format(field))
    return value


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise InputValidationError("{} must be a non-empty string".format(field))
    return value


def _rank_count(topology: Mapping[str, object]) -> int:
    if "ranks" in topology:
        return _integer(topology["ranks"], "topology.ranks", minimum=1)
    if "nnodes" not in topology or "gpus_per_node" not in topology:
        raise InputValidationError(
            "topology must define ranks or both nnodes and gpus_per_node"
        )
    node_count = _integer(topology["nnodes"], "topology.nnodes", minimum=1)
    ranks_per_node = _integer(
        topology["gpus_per_node"],
        "topology.gpus_per_node",
        minimum=1,
    )
    return node_count * ranks_per_node


def _collective(sketch: Mapping[str, object]) -> CollectiveSpec:
    if "collective" not in sketch:
        raise InputValidationError("sketch.collective is required")
    value = _mapping(sketch["collective"], "sketch.collective")
    _reject_unknown_keys(value, _COLLECTIVE_KEYS, "collective")
    if "operator" not in value:
        raise InputValidationError("sketch.collective.operator is required")
    operator = _string(value["operator"], "sketch.collective.operator").lower()
    operator = _OPERATOR_ALIASES.get(operator, operator)
    try:
        kind = CollectiveKind(operator)
    except ValueError as error:
        raise InputValidationError(
            "unsupported collective operator: {}".format(operator)
        ) from error
    if kind in {CollectiveKind.SCATTER, CollectiveKind.GATHER}:
        raise InputValidationError(
            "{} is only available as an internal plan operator".format(
                kind.value
            )
        )
    if "datatype" not in value:
        raise InputValidationError("sketch.collective.datatype is required")
    datatype = _string(value["datatype"], "sketch.collective.datatype").lower()
    reduction_op_value = value.get("reduction_op")
    reduction_op = None
    if reduction_op_value is not None:
        reduction_op = _string(
            reduction_op_value,
            "sketch.collective.reduction_op",
        ).lower()
    root_value = value.get("root")
    root = None
    if root_value is not None:
        root = _integer(root_value, "sketch.collective.root")
    inplace = _boolean(
        value.get("inplace", False),
        "sketch.collective.inplace",
    )
    return CollectiveSpec(
        kind=kind,
        datatype=datatype,
        reduction_op=reduction_op,
        root=root,
        inplace=inplace,
    )


def _section_value(
    primary: Mapping[str, object],
    legacy: Mapping[str, object],
    key: str,
    default: object = _MISSING,
) -> object:
    if key in primary and key in legacy:
        raise InputValidationError("duplicate sketch setting: {}".format(key))
    if key in primary:
        return primary[key]
    if key in legacy:
        return legacy[key]
    if default is _MISSING:
        raise InputValidationError("missing sketch setting: {}".format(key))
    return default


def _configs(
    sketch: Mapping[str, object],
) -> Tuple[Hyperparameters, SolverConfig, dict, dict]:
    hyperparameters_value = sketch.get("hyperparameters", _MISSING)
    if hyperparameters_value is _MISSING:
        raise InputValidationError("sketch.hyperparameters is required")
    hyperparameters = _mapping(
        hyperparameters_value,
        "sketch.hyperparameters",
    )
    allowed_hyperparameters = _HYPERPARAMETER_KEYS | _SOLVER_KEYS
    _reject_unknown_keys(
        hyperparameters,
        allowed_hyperparameters,
        "hyperparameter",
    )
    solver = _mapping(sketch.get("solver", {}), "sketch.solver")
    _reject_unknown_keys(solver, _SOLVER_KEYS, "solver")

    total_size = _integer(
        _section_value(hyperparameters, {}, "total_size_bytes"),
        "hyperparameters.total_size_bytes",
        minimum=1,
    )
    slice_size = _integer(
        _section_value(hyperparameters, {}, "slice_size_bytes"),
        "hyperparameters.slice_size_bytes",
        minimum=1,
    )
    if total_size % slice_size != 0:
        raise InputValidationError(
            "total_size_bytes must be divisible by slice_size_bytes"
        )
    slice_count = total_size // slice_size
    if "input_chunkup" in hyperparameters:
        input_chunkup = _integer(
            hyperparameters["input_chunkup"],
            "hyperparameters.input_chunkup",
            minimum=1,
        )
        if input_chunkup != slice_count:
            raise InputValidationError(
                "input_chunkup must equal total_size_bytes / slice_size_bytes"
            )

    objective_raw = _string(
        hyperparameters.get("objective_mode", ObjectiveMode.AUTO.value),
        "hyperparameters.objective_mode",
    ).lower()
    try:
        objective_mode = ObjectiveMode(objective_raw)
    except ValueError as error:
        raise InputValidationError(
            "unsupported objective_mode: {}".format(objective_raw)
        ) from error
    hyperparameter_model = Hyperparameters(
        total_size_bytes=total_size,
        slice_size_bytes=slice_size,
        objective_mode=objective_mode,
        max_calibration_channels=_integer(
            hyperparameters.get(
                "max_calibration_channels",
                SOFTWARE_MAX_CONCURRENCY,
            ),
            "hyperparameters.max_calibration_channels",
            minimum=1,
            maximum=SOFTWARE_MAX_CONCURRENCY,
        ),
        min_expected_improvement=_number(
            hyperparameters.get("min_expected_improvement", 0.01),
            "hyperparameters.min_expected_improvement",
            minimum=0.0,
            maximum=1.0,
        ),
        min_tuning_improvement=_number(
            hyperparameters.get("min_tuning_improvement", 0.01),
            "hyperparameters.min_tuning_improvement",
            minimum=0.0,
            maximum=1.0,
        ),
        max_tuning_iterations=_integer(
            hyperparameters.get("max_tuning_iterations", 20),
            "hyperparameters.max_tuning_iterations",
            minimum=0,
        ),
        total_verification_timeout_s=_integer(
            hyperparameters.get("total_verification_timeout_s", 10800),
            "hyperparameters.total_verification_timeout_s",
            minimum=1,
        ),
        force_recalibrate=_boolean(
            hyperparameters.get("force_recalibrate", False),
            "hyperparameters.force_recalibrate",
        ),
    )

    solver_model = SolverConfig(
        total_solve_timeout_s=_integer(
            _section_value(solver, hyperparameters, "total_solve_timeout_s", 10800),
            "solver.total_solve_timeout_s",
            minimum=1,
        ),
        per_model_timeout_s=_integer(
            _section_value(solver, hyperparameters, "per_model_timeout_s", 1800),
            "solver.per_model_timeout_s",
            minimum=1,
        ),
        mip_gap=_number(
            _section_value(solver, hyperparameters, "mip_gap", 1e-4),
            "solver.mip_gap",
            minimum=0.0,
            maximum=1.0,
        ),
        require_proven_optimal=_boolean(
            _section_value(
                solver,
                hyperparameters,
                "require_proven_optimal",
                False,
            ),
            "solver.require_proven_optimal",
        ),
        solver_seed=_integer(
            _section_value(solver, hyperparameters, "solver_seed", 0),
            "solver.solver_seed",
            minimum=0,
        ),
        max_channels=_integer(
            _section_value(
                solver,
                hyperparameters,
                "max_channels",
                SOFTWARE_MAX_CONCURRENCY,
            ),
            "solver.max_channels",
            minimum=1,
            maximum=SOFTWARE_MAX_CONCURRENCY,
        ),
        max_threads_per_model=_integer(
            _section_value(solver, hyperparameters, "max_threads_per_model", 12),
            "solver.max_threads_per_model",
            minimum=1,
        ),
        max_parallel_models=_integer(
            _section_value(solver, hyperparameters, "max_parallel_models", 4),
            "solver.max_parallel_models",
            minimum=1,
        ),
        force_resolve=_boolean(
            _section_value(solver, hyperparameters, "force_resolve", False),
            "solver.force_resolve",
        ),
    )
    normalized_hyperparameters = {
        "total_size_bytes": hyperparameter_model.total_size_bytes,
        "slice_size_bytes": hyperparameter_model.slice_size_bytes,
        "input_chunkup": hyperparameter_model.slice_count,
        "objective_mode": hyperparameter_model.objective_mode.value,
        "max_calibration_channels": hyperparameter_model.max_calibration_channels,
        "min_expected_improvement": hyperparameter_model.min_expected_improvement,
        "min_tuning_improvement": hyperparameter_model.min_tuning_improvement,
        "max_tuning_iterations": hyperparameter_model.max_tuning_iterations,
        "total_verification_timeout_s": (
            hyperparameter_model.total_verification_timeout_s
        ),
        "force_recalibrate": hyperparameter_model.force_recalibrate,
    }
    normalized_solver = {
        "total_solve_timeout_s": solver_model.total_solve_timeout_s,
        "per_model_timeout_s": solver_model.per_model_timeout_s,
        "mip_gap": solver_model.mip_gap,
        "require_proven_optimal": solver_model.require_proven_optimal,
        "solver_seed": solver_model.solver_seed,
        "max_channels": solver_model.max_channels,
        "max_threads_per_model": solver_model.max_threads_per_model,
        "max_parallel_models": solver_model.max_parallel_models,
        "force_resolve": solver_model.force_resolve,
    }
    return (
        hyperparameter_model,
        solver_model,
        normalized_hyperparameters,
        normalized_solver,
    )


def _strategy_value(
    atom: Mapping[str, object],
    strategies: Mapping[str, object],
    key: str,
) -> bool:
    if key in atom and key in strategies:
        raise InputValidationError("duplicate atom strategy: {}".format(key))
    value = strategies.get(key, atom.get(key, _STRATEGY_DEFAULTS[key]))
    return _boolean(value, "atom.strategies.{}".format(key))


def _atom_config(
    atom: Mapping[str, object],
    rank_count: int,
    slice_count: int,
) -> Tuple[AtomConstraints, StrategyConfig, dict]:
    _reject_unknown_keys(atom, _ATOM_KEYS, "atom")
    stage_num_value = atom.get("stage_num")
    stage_num = None
    if stage_num_value is not None:
        stage_num = _integer(stage_num_value, "atom.stage_num", minimum=1)

    forbidden_value = atom.get("forbidden_transfers", [])
    if not isinstance(forbidden_value, list):
        raise InputValidationError("atom.forbidden_transfers must be a list")
    forbidden = []
    max_slice_id = rank_count * slice_count
    for index, item in enumerate(forbidden_value):
        field = "atom.forbidden_transfers[{}]".format(index)
        if not isinstance(item, (list, tuple)) or len(item) != 4:
            raise InputValidationError(
                "{} must contain slice_id, src_rank, dst_rank, and stage_id".format(
                    field
                )
            )
        slice_id = _integer(item[0], field + ".slice_id", minimum=0)
        src_rank = _integer(item[1], field + ".src_rank", minimum=0)
        dst_rank = _integer(item[2], field + ".dst_rank", minimum=0)
        stage_id = _integer(item[3], field + ".stage_id", minimum=0)
        if slice_id >= max_slice_id:
            raise InputValidationError("{} slice_id is out of range".format(field))
        if src_rank >= rank_count or dst_rank >= rank_count:
            raise InputValidationError("{} rank is out of range".format(field))
        if src_rank == dst_rank:
            raise InputValidationError("{} must use distinct ranks".format(field))
        if stage_num is not None and stage_id >= stage_num:
            raise InputValidationError("{} stage_id is out of range".format(field))
        forbidden.append(
            ForbiddenTransfer(slice_id, src_rank, dst_rank, stage_id)
        )

    strategies_value = _mapping(atom.get("strategies", {}), "atom.strategies")
    _reject_unknown_keys(
        strategies_value,
        frozenset(_STRATEGY_DEFAULTS),
        "strategy",
    )
    strategy_values = {
        key: _strategy_value(atom, strategies_value, key)
        for key in _STRATEGY_DEFAULTS
    }
    manual_value = atom.get("manual_hierarchy", [])
    if not isinstance(manual_value, list):
        raise InputValidationError("atom.manual_hierarchy must be a list")
    manual_hierarchy = []
    for index, item in enumerate(manual_value):
        if not isinstance(item, dict):
            raise InputValidationError(
                "atom.manual_hierarchy[{}] must be a JSON object".format(index)
            )
        manual_hierarchy.append(_freeze(item))

    constraints = AtomConstraints(stage_num, tuple(forbidden))
    strategy_config = StrategyConfig(
        hierarchy=strategy_values["hierarchy"],
        symmetry=strategy_values["symmetry"],
        shortest_paths=strategy_values["shortest_paths"],
        batching=strategy_values["batching"],
        constructive_trees=strategy_values["constructive_trees"],
        milp=strategy_values["milp"],
        manual_hierarchy=tuple(manual_hierarchy),
    )
    normalized = {
        "stage_num": stage_num,
        "forbidden_transfers": [
            [
                item.slice_id,
                item.src_rank,
                item.dst_rank,
                item.stage_id,
            ]
            for item in constraints.forbidden_transfers
        ],
        "strategies": strategy_values,
        "manual_hierarchy": manual_value,
    }
    return constraints, strategy_config, normalized


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def resolve_inputs(
    topology_path: Path,
    sketch_path: Path,
    atom_path: Path,
) -> ResolvedInput:
    topology_source = _load_json_object(Path(topology_path), "topology")
    sketch_source = _load_json_object(Path(sketch_path), "sketch")
    atom_source = _load_json_object(Path(atom_path), "atom")
    duplicate_rank_keys = sorted(
        {"ranks", "rank_count", "nnodes", "gpus_per_node"} & set(sketch_source)
    )
    if duplicate_rank_keys:
        raise InputValidationError(
            "rank geometry belongs only in topology: {}".format(
                duplicate_rank_keys[0]
            )
        )

    rank_count = _rank_count(topology_source)
    collective = _collective(sketch_source)
    hyperparameters, solver, normalized_hyperparameters, normalized_solver = (
        _configs(sketch_source)
    )
    validate_collective(
        collective,
        rank_count=rank_count,
        slice_count=hyperparameters.slice_count,
    )
    atom_constraints, strategies, normalized_atom = _atom_config(
        atom_source,
        rank_count,
        hyperparameters.slice_count,
    )

    normalized_topology = dict(topology_source)
    normalized_topology["ranks"] = rank_count
    normalized_sketch = {
        key: value
        for key, value in sketch_source.items()
        if key not in {"collective", "hyperparameters", "solver"}
    }
    normalized_sketch.update(
        {
            "collective": {
                "operator": collective.kind.value,
                "root": collective.root,
                "datatype": collective.datatype,
                "reduction_op": collective.reduction_op,
                "inplace": collective.inplace,
            },
            "hyperparameters": normalized_hyperparameters,
            "solver": normalized_solver,
        }
    )
    signature = {
        "topology": normalized_topology,
        "sketch": normalized_sketch,
        "atom": normalized_atom,
    }
    input_sha256 = sha256_json(signature)
    return ResolvedInput(
        collective=collective,
        hyperparameters=hyperparameters,
        solver=solver,
        strategies=strategies,
        atom_constraints=atom_constraints,
        rank_count=rank_count,
        resolved_topology=_freeze(normalized_topology),
        resolved_sketch=_freeze(normalized_sketch),
        resolved_atom=_freeze(normalized_atom),
        input_sha256=input_sha256,
    )
