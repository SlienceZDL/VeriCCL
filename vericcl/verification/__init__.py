"""Verification exports, resolved lazily (PEP 562).

Measurement helpers run as ``python -m vericcl.verification.online.remote``;
importing this package must not load BDD, flow and pipeline analysis for every
helper process. Attribute access and ``from vericcl.verification import X`` are
unchanged.
"""
from importlib import import_module

_EXPORTS = {
    "BDDAnalysisResult": "bdd_backend",
    "FlowReplacementHint": "bdd_flow",
    "analyze_flow_congestion": "bdd_flow",
    "TBOrderHint": "bdd_order",
    "analyze_tb_order": "bdd_order",
    "verify_schedule_constraints": "constraints",
    "verify_schedule_pre_lowering": "constraints",
    "FlowRecord": "flow_index",
    "LaneState": "flow_index",
    "build_flow_index": "flow_index",
    "CheckResult": "model",
    "ValidationReport": "model",
    "ValidationStatus": "model",
    "VerificationOutcome": "pipeline",
    "validate_and_lower_candidate": "pipeline",
    "verify_candidate": "pipeline",
    "verify_candidate_outcome": "pipeline",
    "ResourceInterval": "resource_events",
    "ResourceTimeline": "resource_events",
    "verify_schedule_semantics": "semantics",
    "SimulationEvent": "simulator",
    "SimulationResult": "simulator",
    "simulate_schedule": "simulator",
}

__all__ = [
    "BDDAnalysisResult",
    "CheckResult",
    "FlowRecord",
    "FlowReplacementHint",
    "LaneState",
    "ResourceInterval",
    "ResourceTimeline",
    "SimulationEvent",
    "SimulationResult",
    "TBOrderHint",
    "ValidationReport",
    "ValidationStatus",
    "VerificationOutcome",
    "analyze_flow_congestion",
    "analyze_tb_order",
    "build_flow_index",
    "simulate_schedule",
    "verify_schedule_constraints",
    "verify_schedule_pre_lowering",
    "verify_schedule_semantics",
    "validate_and_lower_candidate",
    "verify_candidate",
    "verify_candidate_outcome",
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
