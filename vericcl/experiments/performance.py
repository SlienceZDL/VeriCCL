from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from pathlib import Path
import re
from typing import Iterable, Tuple

from lxml import etree

from vericcl.errors import SemanticError
from vericcl.verification.online.model import NcclTestRun


_MSCCL_INFO_MARKER = "Connected 1 MSCCL algorithms"
_SIZE_PATTERN = re.compile(r"^[1-9][0-9]*[KMG]?$")


class XmlSource(str, Enum):
    VERICCL = "vericcl"
    BASELINE = "baseline"


@dataclass(frozen=True)
class PerformanceResult:
    task_id: str
    topology_name: str
    collective_label: str
    source: XmlSource
    xml_name: str
    runs: Tuple[NcclTestRun, ...]
    activation: Tuple["ActivationEvidence", ...]
    stdout_path: Path
    stderr_path: Path

    def __post_init__(self) -> None:
        for field in (
            "task_id",
            "topology_name",
            "collective_label",
            "xml_name",
        ):
            value = getattr(self, field)
            if not isinstance(value, str) or not value or "\x00" in value:
                raise SemanticError(
                    "performance {} is invalid".format(field)
                )
        if not isinstance(self.source, XmlSource):
            raise SemanticError("performance XML source is invalid")
        runs = tuple(self.runs)
        if not runs or not all(isinstance(run, NcclTestRun) for run in runs):
            raise SemanticError("performance runs are invalid")
        sizes = tuple(run.message_size_bytes for run in runs)
        if sizes != tuple(sorted(set(sizes))):
            raise SemanticError("performance run size order is invalid")
        object.__setattr__(self, "runs", runs)
        activation = tuple(self.activation)
        if len(activation) != len(runs) or not all(
            isinstance(value, ActivationEvidence) for value in activation
        ):
            raise SemanticError("performance activation evidence is invalid")
        object.__setattr__(self, "activation", activation)
        for field in ("stdout_path", "stderr_path"):
            try:
                value = Path(getattr(self, field))
            except TypeError as error:
                raise SemanticError(
                    "performance {} is invalid".format(field)
                ) from error
            object.__setattr__(self, field, value)


def build_performance_command(
    *,
    binary: str,
    begin: str,
    end: str,
    factor: int,
    iterations: int,
) -> Tuple[str, ...]:
    if not isinstance(binary, str) or not binary or "\x00" in binary:
        raise SemanticError("performance binary is invalid")
    for value, field in ((begin, "begin"), (end, "end")):
        if not isinstance(value, str) or not _SIZE_PATTERN.fullmatch(value):
            raise SemanticError("performance {} size is invalid".format(field))
    for value, field, minimum in (
        (factor, "factor", 2),
        (iterations, "iterations", 1),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise SemanticError("performance {} is invalid".format(field))
    return (
        binary,
        "-b",
        begin,
        "-e",
        end,
        "-f",
        str(factor),
        "-g",
        "1",
        "-n",
        str(iterations),
    )


def select_baselines(
    paths: Iterable[Path],
    *,
    collective: str,
    rank_count: int,
) -> Tuple[Path, ...]:
    if collective not in {"allgather", "allreduce"}:
        raise SemanticError("baseline collective is unsupported")
    if (
        isinstance(rank_count, bool)
        or not isinstance(rank_count, int)
        or rank_count < 1
    ):
        raise SemanticError("baseline rank count is invalid")
    try:
        candidates = tuple(Path(path) for path in paths)
    except TypeError as error:
        raise SemanticError("baseline paths must be iterable") from error
    selected = []
    parser = etree.XMLParser(
        resolve_entities=False,
        no_network=True,
        remove_comments=False,
    )
    for path in candidates:
        try:
            root = etree.parse(str(path), parser).getroot()
        except (OSError, etree.XMLSyntaxError) as error:
            raise SemanticError(
                "baseline XML is unreadable: {}".format(path)
            ) from error
        if root.tag != "algo":
            raise SemanticError("baseline XML root must be algo")
        if (
            root.attrib.get("coll") == collective
            and root.attrib.get("ngpus") == str(rank_count)
            and root.attrib.get("proto") == "Simple"
        ):
            selected.append(path)
    return tuple(sorted(selected, key=lambda path: (path.name, str(path))))


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
