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
    "repair_flow_suffix",
    "solve_local_repair",
]


def __getattr__(name):
    if name in {"ImpactClosure", "compute_impact_closure"}:
        from vericcl.tuning import impact

        return getattr(impact, name)
    if name == "repair_flow_suffix":
        from vericcl.tuning.repair import repair_flow_suffix

        return repair_flow_suffix
    if name == "solve_local_repair":
        from vericcl.tuning.local_milp import solve_local_repair

        return solve_local_repair
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
