"""Solver package exports.

The public names below are resolved lazily (PEP 562). Owned SCIP workers start
with ``python -m vericcl.solver.worker_scope``; importing every solver,
template, routing and verification module eagerly made each worker spend most
of its life in imports. Attribute access and ``from vericcl.solver import X``
behave exactly as before.
"""
from importlib import import_module

_EXPORTS = {
    "ModelBudget": "budget",
    "SolveBudget": "budget",
    "candidate_cache_key": "cache",
    "construct_candidate": "constructive",
    "construct_route_pattern": "constructive",
    "build_solver_problem": "demands",
    "ScipAdapter": "scip_api",
    "GLOBAL_SCHEDULER_VERSION": "global_scheduler",
    "GlobalSchedulingError": "global_scheduler",
    "assign_global_resources": "global_scheduler",
    "InstantiationFailure": "instantiate",
    "InstantiationResult": "instantiate",
    "instantiate_route_patterns": "instantiate",
    "LowerBound": "lower_bounds",
    "global_throughput_time_lower_bound": "lower_bounds",
    "throughput_time_lower_bound": "lower_bounds",
    "solve_milp": "milp",
    "SearchDiagnostics": "model",
    "SolveCandidate": "model",
    "SolveRequest": "model",
    "SolveResult": "model",
    "SolveStatus": "model",
    "SolverMetrics": "model",
    "rank_candidates": "objectives",
    "solve": "orchestrator",
    "RoutePattern": "routing",
    "RoutingModelStats": "routing",
    "solve_route_milp": "routing_milp",
    "ModelSearchResult": "search",
    "allocate_model_threads": "search",
    "search_models": "search",
    "search_models_with_diagnostics": "search",
    "TemplateSearchResult": "template_search",
    "search_route_models": "template_search",
    "RoutingUnit": "templates",
    "SolverTemplate": "templates",
    "TemplateMember": "templates",
    "build_solver_templates": "templates",
    "split_routing_units": "templates",
}

__all__ = [
    "ModelBudget",
    "ScipAdapter",
    "GLOBAL_SCHEDULER_VERSION",
    "GlobalSchedulingError",
    "InstantiationFailure",
    "InstantiationResult",
    "LowerBound",
    "ModelSearchResult",
    "SearchDiagnostics",
    "SolveBudget",
    "SolveCandidate",
    "SolveRequest",
    "SolveResult",
    "SolveStatus",
    "SolverTemplate",
    "SolverMetrics",
    "RoutingUnit",
    "RoutePattern",
    "RoutingModelStats",
    "TemplateMember",
    "TemplateSearchResult",
    "allocate_model_threads",
    "assign_global_resources",
    "build_solver_problem",
    "build_solver_templates",
    "candidate_cache_key",
    "construct_candidate",
    "construct_route_pattern",
    "global_throughput_time_lower_bound",
    "instantiate_route_patterns",
    "rank_candidates",
    "search_models",
    "search_models_with_diagnostics",
    "search_route_models",
    "solve",
    "solve_milp",
    "solve_route_milp",
    "split_routing_units",
    "throughput_time_lower_bound",
]


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is not None:
        value = getattr(import_module(__name__ + "." + module), name)
        globals()[name] = value
        return value
    if not name.startswith("__"):
        try:
            return import_module(__name__ + "." + name)
        except ModuleNotFoundError as error:
            if error.name != __name__ + "." + name:
                raise
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))


def __dir__():
    return sorted(set(globals()) | set(__all__))
