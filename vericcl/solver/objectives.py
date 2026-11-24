from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

from vericcl.errors import SemanticError
from vericcl.input.models import ObjectiveMode
from vericcl.solver.model import SolveCandidate


@dataclass(frozen=True)
class ObjectiveExpressions:
    makespan: object
    operation_count: object
    hop_count: object
    maximum_resource_load: Optional[object] = None


# Weight of the secondary terms in a single-pass (screening) objective: far
# below any makespan difference the 1% screening gap can resolve, so they only
# break ties toward fewer operations/hops.
SCREENING_TIE_WEIGHT = 1e-4


def configure_lexicographic_objective(
    model,
    gp,
    objective: ObjectiveMode,
    expressions: ObjectiveExpressions,
    *, single_pass: bool = False,
) -> None:
    """Lexicographic objectives, or with ``single_pass`` one weighted objective.

    A lexicographic order costs one full SCIP solve (re-presolve included)
    per level; screening solves that only rank plans use ``single_pass``.
    """
    if not isinstance(objective, ObjectiveMode):
        raise SemanticError("objective must be an ObjectiveMode")
    if objective is ObjectiveMode.AUTO:
        raise SemanticError("AUTO must be resolved before building a MILP model")
    if objective is ObjectiveMode.LATENCY:
        ordered = (
            (expressions.makespan, 3, "makespan"),
            (expressions.operation_count, 2, "operation-count"),
            (expressions.hop_count, 1, "hop-count"),
        )
    else:
        if expressions.maximum_resource_load is None:
            raise SemanticError(
                "throughput objective requires a maximum resource load"
            )
        ordered = (
            (expressions.maximum_resource_load, 2, "maximum-resource-load"),
            (expressions.makespan, 1, "makespan"),
        )
    if single_pass:
        primary = ordered[0][0]
        secondary = gp.quicksum(expression for expression, _, _ in ordered[1:])
        model.set_lexicographic_objectives(((primary + SCREENING_TIE_WEIGHT * secondary, ordered[0][2]),))
        return
    model.set_lexicographic_objectives(tuple((expression, name) for expression, _, name in ordered))


def _ranking_key(candidate: SolveCandidate) -> Tuple[object, ...]:
    metrics = candidate.metrics
    if candidate.objective_mode is ObjectiveMode.LATENCY:
        return (
            metrics.makespan_us,
            metrics.operation_count,
            metrics.hop_count,
            candidate.candidate_id,
        )
    return (
        metrics.maximum_normalized_resource_load,
        metrics.makespan_us,
        candidate.candidate_id,
    )


def rank_candidates(
    candidates: Iterable[SolveCandidate],
) -> Tuple[SolveCandidate, ...]:
    values = tuple(candidates)
    if not all(isinstance(item, SolveCandidate) for item in values):
        raise SemanticError("candidates must contain SolveCandidate values")
    if not values:
        return ()
    objectives = {item.objective_mode for item in values}
    if ObjectiveMode.AUTO in objectives:
        raise SemanticError("AUTO candidates require orchestrator comparison")
    if len(objectives) != 1:
        raise SemanticError("candidate objective modes must agree")
    return tuple(sorted(values, key=_ranking_key))
