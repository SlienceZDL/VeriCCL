from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Tuple

from lxml import etree

from vericcl.errors import SemanticError
from vericcl.input.models import ResolvedInput
from vericcl.semantics.collective import CollectiveKind
from vericcl.xml.compatibility import (
    MAX_CHANNELS,
    MAX_DEPENDENT_TB_ID,
    MAX_DIRECTION_TBS_PER_CHANNEL,
    MAX_STEPS_PER_TB,
    check_msccl_compatibility,
)


@dataclass(frozen=True)
class Recommendation:
    kind: str
    priority: int
    parameters: Mapping[str, int]
    reason_codes: Tuple[str, ...]
    requires_resolve: bool

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind:
            raise SemanticError("recommendation kind must be non-empty")
        if (
            isinstance(self.priority, bool)
            or not isinstance(self.priority, int)
            or self.priority < 0
        ):
            raise SemanticError("recommendation priority must be an integer")
        parameters = dict(self.parameters)
        if not all(
            isinstance(key, str)
            and key
            and isinstance(value, int)
            and not isinstance(value, bool)
            and value > 0
            for key, value in parameters.items()
        ):
            raise SemanticError("recommendation parameters are invalid")
        reasons = tuple(self.reason_codes)
        if not reasons or not all(
            isinstance(reason, str) and reason for reason in reasons
        ):
            raise SemanticError("recommendation reason_codes must be non-empty")
        if not isinstance(self.requires_resolve, bool):
            raise SemanticError("recommendation requires_resolve must be boolean")
        object.__setattr__(self, "parameters", MappingProxyType(parameters))
        object.__setattr__(self, "reason_codes", tuple(sorted(set(reasons))))


def artifact_xml_filename(schedule_name: str, artifact) -> str:
    if not isinstance(schedule_name, str) or not schedule_name:
        raise SemanticError("schedule_name must be a non-empty string")
    suffix = ".xml" if artifact.runtime_compatible else ".candidate.xml"
    return schedule_name + suffix


def _divisors(value: int) -> Tuple[int, ...]:
    values = set()
    for divisor in range(1, math.isqrt(value) + 1):
        if value % divisor == 0:
            values.add(divisor)
            values.add(value // divisor)
    return tuple(sorted(values))


def _larger_slice(inputs: ResolvedInput) -> int:
    total = inputs.hyperparameters.total_size_bytes
    current = inputs.hyperparameters.slice_size_bytes
    partitioned = inputs.collective.kind in {
        CollectiveKind.ALL_TO_ALL,
        CollectiveKind.REDUCE_SCATTER,
    }
    for candidate in _divisors(total):
        if candidate <= current:
            continue
        slice_count = total // candidate
        if not partitioned or slice_count % inputs.rank_count == 0:
            return candidate
    return 0


def recommend_runtime_compatible_inputs(
    inputs: ResolvedInput,
    artifact,
) -> Tuple[Recommendation, ...]:
    if not isinstance(inputs, ResolvedInput):
        raise SemanticError("inputs must be a ResolvedInput")
    report = check_msccl_compatibility(artifact)
    if report.runtime_compatible:
        return ()
    root = etree.fromstring(artifact.xml_text.encode("utf-8"))
    codes = tuple(sorted({issue.code for issue in report.issues}))
    recommendations = []
    dependent_ids = {
        int(step.attrib["depid"])
        for step in root.xpath(".//step")
        if int(step.attrib.get("depid", "-1")) >= 0
    }
    if (
        "dependent_tb_id" in codes
        and len(dependent_ids) <= MAX_DEPENDENT_TB_ID + 1
    ):
        recommendations.append(
            Recommendation(
                kind="renumber_threadblocks",
                priority=0,
                parameters={"dependent_tb_count": len(dependent_ids)},
                reason_codes=("dependent_tb_id",),
                requires_resolve=False,
            )
        )

    current_channels = int(root.attrib["nchannels"])
    maximum_steps = max(
        (len(tb.xpath("./step")) for tb in root.xpath("./gpu/tb")),
        default=0,
    )
    required_channels = max(
        current_channels,
        math.ceil(maximum_steps / MAX_STEPS_PER_TB),
    )
    for gpu in root.xpath("./gpu"):
        for direction in ("send", "recv"):
            total = sum(
                int(tb.attrib[direction]) >= 0 for tb in gpu.xpath("./tb")
            )
            required_channels = max(
                required_channels,
                math.ceil(total / MAX_DIRECTION_TBS_PER_CHANNEL),
            )
    if current_channels < required_channels <= MAX_CHANNELS:
        recommendations.append(
            Recommendation(
                kind="increase_channels",
                priority=1,
                parameters={"max_channels": required_channels},
                reason_codes=codes,
                requires_resolve=True,
            )
        )

    larger_slice = _larger_slice(inputs)
    if larger_slice:
        recommendations.append(
            Recommendation(
                kind="increase_slice_size",
                priority=2,
                parameters={"slice_size_bytes": larger_slice},
                reason_codes=codes,
                requires_resolve=True,
            )
        )
    return tuple(sorted(recommendations, key=lambda item: item.priority))
