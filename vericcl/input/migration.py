"""Only explicitly supported input spellings and setting locations migrate."""
from __future__ import annotations

from vericcl.errors import InputValidationError


OPERATOR_ALIASES = {"all_gather": "allgather", "all_reduce": "allreduce",
                    "all_to_all": "alltoall", "reducescatter": "reduce_scatter"}
SOLVER_LEGACY_KEYS = {"per_model_timeout_s", "mip_gap", "require_proven_optimal",
                      "solver_seed", "max_channels", "max_threads_per_model", "max_parallel_models"}
STRATEGY_LEGACY_KEYS = {"symmetry", "shortest_paths", "batching", "objective_mode"}
REMOVED = {"input_chunkup", "N", "force_resolve", "force_recalibrate"}


def normalize_operator(value):
    if not isinstance(value, str):
        return value
    lowered = value.lower()
    return OPERATOR_ALIASES.get(lowered, lowered)


def migrate(documents: dict) -> dict[str, str]:
    origins = {}
    def reject_removed(value, path):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in REMOVED:
                    raise InputValidationError(path + "." + key + " is removed; slice count is derived and reuse is explicitly configured")
                reject_removed(item, path + "." + key)
        elif isinstance(value, list):
            for item in value:
                reject_removed(item, path)
    for name in ("sketch", "atom"):
        reject_removed(documents[name], name)
    sketch, atom = documents["sketch"], documents["atom"]
    if isinstance(sketch.get("collective"), dict):
        operator = sketch["collective"].get("operator")
        if isinstance(operator, str):
            sketch["collective"]["operator"] = normalize_operator(operator)
    hyper = sketch.get("hyperparameters")
    solver = sketch.get("solver")
    old_total = []
    for section_name, section_value in (("hyperparameters", hyper), ("solver", solver)):
        if isinstance(section_value, dict) and "total_solve_timeout_s" in section_value:
            old_total.append((section_name, section_value.pop("total_solve_timeout_s")))
    if old_total:
        budgets = sketch.get("budgets")
        if budgets is None:
            sketch["budgets"] = budgets = {}
        if not isinstance(budgets, dict):
            raise InputValidationError("sketch.budgets must be an object")
        if len(old_total) != 1 or "max_initial_synthesis_s" in budgets:
            raise InputValidationError("duplicate initial synthesis budget")
        budgets["max_initial_synthesis_s"] = old_total[0][1]
        origins["sketch.budgets.max_initial_synthesis_s"] = "sketch." + old_total[0][0] + ".total_solve_timeout_s"
    if isinstance(hyper, dict):
        for key in SOLVER_LEGACY_KEYS:
            if key in hyper:
                if solver is None:
                    sketch["solver"] = solver = {}
                if not isinstance(solver, dict):
                    raise InputValidationError("sketch.solver must be an object")
                if key in solver:
                    raise InputValidationError("duplicate sketch setting: " + key)
                solver[key] = hyper.pop(key)
                origins["sketch.solver." + key] = "sketch.hyperparameters." + key
    strategies = atom.get("strategies")
    for key in ("hierarchy", "constructive_trees", "milp"):
        if key in atom or (isinstance(strategies, dict) and key in strategies):
            raise InputValidationError("removed algorithm boolean " + key + "; use atom.synthesis_method")
    for key in STRATEGY_LEGACY_KEYS:
        if key in atom:
            if strategies is None:
                atom["strategies"] = strategies = {}
            if not isinstance(strategies, dict):
                raise InputValidationError("atom.strategies must be an object")
            if key in strategies:
                raise InputValidationError("duplicate atom strategy: " + key)
            strategies[key] = atom.pop(key)
            origins["atom.strategies." + key] = "atom." + key
    return origins
