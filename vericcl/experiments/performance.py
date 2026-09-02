from __future__ import annotations

from dataclasses import dataclass
import math

from vericcl.errors import SemanticError
from vericcl.verification.online.model import NcclTestRun


_MSCCL_INFO_MARKER = "Connected 1 MSCCL algorithms"


def _ratio(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SemanticError("{} must be a number".format(field))
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or result > 1.0:
        raise SemanticError("{} must be between zero and one".format(field))
    return result


@dataclass(frozen=True)
class ActivationEvidence:
    info_loaded: bool
    relative_busbw_difference: float
    threshold: float
    confirmed: bool

    def __post_init__(self) -> None:
        if not isinstance(self.info_loaded, bool):
            raise SemanticError("activation info flag must be boolean")
        object.__setattr__(
            self,
            "relative_busbw_difference",
            _ratio(
                self.relative_busbw_difference,
                "relative bus bandwidth difference",
            ),
        )
        object.__setattr__(
            self,
            "threshold",
            _ratio(self.threshold, "activation threshold"),
        )
        if not isinstance(self.confirmed, bool):
            raise SemanticError("activation confirmation must be boolean")
        if self.confirmed and (
            not self.info_loaded
            or self.relative_busbw_difference < self.threshold
        ):
            raise SemanticError("activation confirmation is inconsistent")


def evaluate_msccl_activation(
    text: str,
    run: NcclTestRun,
    threshold: float = 0.05,
) -> ActivationEvidence:
    if not isinstance(text, str):
        raise SemanticError("activation log must be text")
    if not isinstance(run, NcclTestRun):
        raise SemanticError("activation run must be an NcclTestRun")
    normalized_threshold = _ratio(threshold, "activation threshold")
    if run.in_place is None:
        return ActivationEvidence(False, 0.0, normalized_threshold, False)
    denominator = max(
        run.out_of_place.bus_bandwidth_gbps,
        run.in_place.bus_bandwidth_gbps,
    )
    difference = (
        abs(
            run.in_place.bus_bandwidth_gbps
            - run.out_of_place.bus_bandwidth_gbps
        )
        / denominator
        if denominator > 0.0
        else 0.0
    )
    loaded = _MSCCL_INFO_MARKER in text
    return ActivationEvidence(
        info_loaded=loaded,
        relative_busbw_difference=difference,
        threshold=normalized_threshold,
        confirmed=loaded and difference >= normalized_threshold,
    )
