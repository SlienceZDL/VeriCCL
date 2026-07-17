from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from vericcl.errors import SemanticError
from vericcl.xml.endpoints import EndpointType
from vericcl.xml.threadblocks import ThreadblockProgram


@dataclass(frozen=True)
class DeadlockResult:
    deadlocked: bool
    completed_step_ids: frozenset[str]
    blocked_transfer_ids: frozenset[str]
    tb_heads: Mapping[tuple[int, int], str]

    def __post_init__(self) -> None:
        if not isinstance(self.deadlocked, bool):
            raise SemanticError("deadlocked must be a boolean")
        object.__setattr__(
            self,
            "completed_step_ids",
            frozenset(self.completed_step_ids),
        )
        object.__setattr__(
            self,
            "blocked_transfer_ids",
            frozenset(self.blocked_transfer_ids),
        )
        object.__setattr__(self, "tb_heads", MappingProxyType(dict(self.tb_heads)))


def simulate_endpoint_execution(program: ThreadblockProgram) -> DeadlockResult:
    if not isinstance(program, ThreadblockProgram):
        raise SemanticError("program must be a ThreadblockProgram")
    positions = {
        (threadblock.key.rank, threadblock.tb_id): 0
        for threadblock in program.threadblocks
    }
    blocks = {
        (threadblock.key.rank, threadblock.tb_id): threadblock
        for threadblock in program.threadblocks
    }
    completed = set()

    def heads():
        return {
            key: block.steps[positions[key]]
            for key, block in blocks.items()
            if positions[key] < len(block.steps)
        }

    def dependency_ready(step) -> bool:
        return (
            step.dependency_step_id is None
            or step.dependency_step_id in completed
        )

    while True:
        current = heads()
        if not current:
            return DeadlockResult(False, completed, frozenset(), {})
        progressed = False
        for key, step in sorted(current.items()):
            is_local = step.xml_type in {
                EndpointType.COPY,
                EndpointType.NOP,
            }
            if is_local and dependency_ready(step):
                completed.add(step.step_id)
                positions[key] += 1
                progressed = True
        if progressed:
            continue
        current = heads()
        head_ids = {step.step_id for step in current.values()}
        for transfer_id, step_ids in sorted(program.transfer_steps.items()):
            if set(step_ids) <= head_ids and all(
                dependency_ready(program.steps_by_id[step_id])
                for step_id in step_ids
            ):
                for key, step in current.items():
                    if step.step_id in step_ids:
                        completed.add(step.step_id)
                        positions[key] += 1
                progressed = True
        if progressed:
            continue
        blocked = frozenset(
            step.transfer_id
            for step in current.values()
            if step.xml_type
            in {
                EndpointType.SEND,
                EndpointType.RECV,
                EndpointType.RECV_REDUCE_COPY,
            }
        )
        return DeadlockResult(
            True,
            completed,
            blocked,
            {key: step.step_id for key, step in current.items()},
        )
