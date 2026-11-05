from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Tuple

from lxml import etree

from vericcl.errors import SemanticError
from vericcl.xml.endpoints import EndpointType
from vericcl.xml.threadblocks import ThreadblockProgram


MAX_STEPS_PER_TB = 256
MAX_DIRECTION_TBS_PER_CHANNEL = 32
MAX_TBS_PER_RANK = 216
MAX_CHANNELS = 16
MAX_BUFFER_OFFSET = 32767
MAX_DEPENDENT_TB_ID = 127


@dataclass(frozen=True)
class CompatibilityIssue:
    code: str
    message: str
    rank: int
    tb_id: Optional[int]
    channel: Optional[int]
    current_value: int
    limit: int
    transfer_ids: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code:
            raise SemanticError("compatibility issue code must be non-empty")
        if not isinstance(self.message, str) or not self.message:
            raise SemanticError("compatibility issue message must be non-empty")
        if (
            isinstance(self.rank, bool)
            or not isinstance(self.rank, int)
            or self.rank < 0
        ):
            raise SemanticError("compatibility issue rank is invalid")
        for value, field in ((self.tb_id, "tb_id"), (self.channel, "channel")):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise SemanticError(
                    "compatibility issue {} is invalid".format(field)
                )
        if (
            isinstance(self.current_value, bool)
            or not isinstance(self.current_value, int)
            or self.current_value < 0
        ):
            raise SemanticError("compatibility issue current_value is invalid")
        if (
            isinstance(self.limit, bool)
            or not isinstance(self.limit, int)
            or self.limit < 0
        ):
            raise SemanticError("compatibility issue limit is invalid")
        transfer_ids = tuple(self.transfer_ids)
        if not transfer_ids or not all(
            isinstance(value, str) and value for value in transfer_ids
        ):
            raise SemanticError("compatibility issue transfer_ids are invalid")
        object.__setattr__(self, "transfer_ids", tuple(sorted(set(transfer_ids))))


@dataclass(frozen=True)
class CompatibilityReport:
    issues: Tuple[CompatibilityIssue, ...]

    def __post_init__(self) -> None:
        issues = tuple(self.issues)
        if not all(isinstance(issue, CompatibilityIssue) for issue in issues):
            raise SemanticError("compatibility report contains an invalid issue")
        object.__setattr__(self, "issues", issues)

    @property
    def runtime_compatible(self) -> bool:
        return not self.issues

    def apply(self, artifact):
        return replace(
            artifact,
            runtime_compatible=self.runtime_compatible,
        )


def renumber_dependent_threadblocks(
    program: ThreadblockProgram,
) -> ThreadblockProgram:
    if not isinstance(program, ThreadblockProgram):
        raise SemanticError("program must be a ThreadblockProgram")
    ordered = []
    ranks = sorted({tb.key.rank for tb in program.threadblocks})
    for rank in ranks:
        blocks = sorted(
            (tb for tb in program.threadblocks if tb.key.rank == rank),
            key=lambda tb: tb.tb_id,
        )
        dependent = [
            tb
            for tb in blocks
            if any(
                step.step_id in program.referenced_step_ids
                for step in tb.steps
            )
        ]
        independent = [tb for tb in blocks if tb not in dependent]
        ordered.extend(
            replace(threadblock, tb_id=tb_id)
            for tb_id, threadblock in enumerate(dependent + independent)
        )
    return replace(program, threadblocks=tuple(ordered))


def _rank_transfer_ids(artifact, rank: int) -> Tuple[str, ...]:
    values = {
        step.transfer_id
        for threadblock in artifact.tb_program.threadblocks
        if threadblock.key.rank == rank
        for step in threadblock.steps
        if step.xml_type is not EndpointType.NOP
    }
    if not values:
        values = {"rank-{}-unknown-transfer".format(rank)}
    return tuple(sorted(values))


def _tb_transfer_ids(artifact, rank: int, tb_id: int) -> Tuple[str, ...]:
    for threadblock in artifact.tb_program.threadblocks:
        if threadblock.key.rank == rank and threadblock.tb_id == tb_id:
            values = {
                step.transfer_id
                for step in threadblock.steps
                if step.xml_type is not EndpointType.NOP
            }
            if values:
                return tuple(sorted(values))
    return _rank_transfer_ids(artifact, rank)


def _issue(
    artifact,
    code: str,
    rank: int,
    current_value: int,
    limit: int,
    *,
    tb_id: Optional[int] = None,
    channel: Optional[int] = None,
    transfer_ids: Optional[Tuple[str, ...]] = None,
) -> CompatibilityIssue:
    return CompatibilityIssue(
        code=code,
        message="{} exceeds the MSCCL execution limit".format(code),
        rank=rank,
        tb_id=tb_id,
        channel=channel,
        current_value=current_value,
        limit=limit,
        transfer_ids=(
            transfer_ids
            if transfer_ids is not None
            else _rank_transfer_ids(artifact, rank)
        ),
    )


def execution_shape_from_xml(xml_text):
    """Read typed program fields; semantic and deadlock checks are separate."""
    from vericcl.xml.capacity import BlockCapacity, ExecutionShape, StepCapacity
    try:
        root = etree.fromstring(xml_text.encode("utf-8"))
    except etree.XMLSyntaxError as error:
        raise SemanticError("artifact XML is not well formed") from error
    try:
        channels = int(root.attrib["nchannels"])
    except (KeyError, ValueError) as error:
        raise SemanticError("artifact XML nchannels is invalid") from error
    try:
        ranks = int(root.attrib["ngpus"])
        blocks, scratch, gpu_ids = [], {}, []
        for gpu in root.xpath("./gpu"):
            rank = int(gpu.attrib["id"])
            gpu_ids.append(rank)
            scratch[rank] = int(gpu.attrib["s_chunks"])
            for tb in gpu.xpath("./tb"):
                steps = tuple(StepCapacity(step.attrib["type"], int(step.attrib["srcoff"]),
                    int(step.attrib["dstoff"]), int(step.attrib.get("depid", "-1")),
                    int(step.attrib.get("deps", "-1")), int(step.attrib["cnt"])) for step in tb.xpath("./step"))
                if [int(step.attrib["s"]) for step in tb.xpath("./step")] != list(range(len(steps))):
                    raise SemanticError("artifact XML step indices must be dense and ordered")
                blocks.append(BlockCapacity(rank, int(tb.attrib["id"]), int(tb.attrib["chan"]),
                                            int(tb.attrib["send"]), int(tb.attrib["recv"]), steps))
        if sorted(gpu_ids) != list(range(ranks)):
            raise SemanticError("artifact XML must contain each GPU exactly once")
        return ExecutionShape(ranks, channels, tuple(blocks), scratch, int(root.attrib["nchunksperloop"]))
    except (KeyError, ValueError) as error:
        raise SemanticError("artifact XML capacity fields are invalid") from error


def check_msccl_compatibility(artifact, *, limits=None, slice_bytes=None, scratch_bytes_per_rank=None) -> CompatibilityReport:
    from vericcl.xml.lower import XmlArtifact
    from vericcl.xml.capacity import RuntimeLimits, check_capacity
    if not isinstance(artifact, XmlArtifact):
        raise SemanticError("artifact must be an XmlArtifact")
    shape = execution_shape_from_xml(artifact.xml_text)
    checked = check_capacity(shape, RuntimeLimits() if limits is None else limits,
                             slice_bytes=slice_bytes, scratch_bytes_per_rank=scratch_bytes_per_rank)
    issues = []
    blocks = {(block.rank, block.tb_id): block for block in shape.blocks}
    for violation in checked.violations:
        # Invalid negative identifiers cannot be represented by the legacy
        # issue format. Reject them without silently repairing the XML.
        if violation.current < 0 or (violation.rank is not None and violation.rank < 0):
            raise SemanticError("artifact XML contains invalid negative " + violation.code)
        rank = violation.rank if violation.rank is not None else 0
        block = blocks.get((rank, violation.tb_id))
        channel = block.channel if block is not None and block.channel >= 0 else None
        if violation.code == 'channels':
            channel = max(0, shape.channels - 1)
        elif violation.detail.startswith('channel='):
            channel = int(violation.detail.split('=', 1)[1])
        code = 'dependent_tb_id' if violation.code == 'dependency_id' else violation.code
        issues.append(_issue(artifact, code, rank, violation.current, violation.limit,
            tb_id=violation.tb_id, channel=channel,
            transfer_ids=_tb_transfer_ids(artifact, rank, violation.tb_id) if violation.tb_id is not None else None))
    issues.sort(key=lambda issue: (issue.rank, issue.code,
        -1 if issue.tb_id is None else issue.tb_id, -1 if issue.channel is None else issue.channel))
    return CompatibilityReport(tuple(issues))
