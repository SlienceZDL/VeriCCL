from vericcl.tuning.model import (
    RepairResult,
    RepairStatus,
    TuningOverlay,
)

__all__ = [
    "ImpactClosure",
    "RepairResult",
    "RepairStatus",
    "TuningOverlay",
    "compute_impact_closure",
    "RepairDomain", "RepairContext", "V5RepairResult", "build_domain", "solve_repair",
    "ResynthesisRequest", "resynthesize_branch",
]


def __getattr__(name):
    if name in {"RepairDomain", "RepairContext", "V5RepairResult", "build_domain", "solve_repair"}:
        from vericcl.tuning import strategies
        return getattr(strategies, "RepairResult" if name == "V5RepairResult" else name)
    if name in {"ResynthesisRequest", "resynthesize_branch"}:
        from vericcl.tuning import resynthesis
        return getattr(resynthesis, name)
    if name in {"ImpactClosure", "compute_impact_closure"}:
        from vericcl.tuning import impact

        return getattr(impact, name)
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
