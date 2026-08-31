from vericcl.solver.budget import ModelBudget, SolveBudget
from vericcl.solver.cache import candidate_cache_key
from vericcl.solver.constructive import (
    construct_candidate,
    construct_route_pattern,
)
from vericcl.solver.demands import build_solver_problem
from vericcl.solver.gurobi_api import GurobiAdapter
from vericcl.solver.global_scheduler import (
    GLOBAL_SCHEDULER_VERSION,
    GlobalSchedulingError,
    assign_global_resources,
)
from vericcl.solver.instantiate import (
    InstantiationFailure,
    InstantiationResult,
    instantiate_route_patterns,
)
from vericcl.solver.lower_bounds import (
    LowerBound,
    global_throughput_time_lower_bound,
    throughput_time_lower_bound,
)
from vericcl.solver.milp import solve_milp
from vericcl.solver.model import (
    SearchDiagnostics,
    SolveCandidate,
    SolveRequest,
    SolveResult,
    SolveStatus,
    SolverMetrics,
)
from vericcl.solver.objectives import rank_candidates
from vericcl.solver.orchestrator import solve
from vericcl.solver.routing import RoutePattern, RoutingModelStats
from vericcl.solver.routing_milp import solve_route_milp
from vericcl.solver.search import (
    ModelSearchResult,
    allocate_model_threads,
    search_models,
    search_models_with_diagnostics,
)
from vericcl.solver.template_search import (
    TemplateSearchResult,
    search_route_models,
)
from vericcl.solver.templates import (
    RoutingUnit,
    SolverTemplate,
    TemplateMember,
    build_solver_templates,
    split_routing_units,
)

__all__ = [
    "ModelBudget",
    "GurobiAdapter",
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
