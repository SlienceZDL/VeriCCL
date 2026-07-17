from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

from vericcl.errors import SemanticError
from vericcl.semantics.atom import Schedule
from vericcl.verification.bdd_backend import BDDAnalysisResult, CompactBDD
from vericcl.verification.model import ValidationStatus
from vericcl.xml.endpoints import EndpointType
from vericcl.xml.threadblocks import ThreadblockProgram, XmlStep


@dataclass(frozen=True)
class TBOrderHint:
    tb_id: str
    rank: int
    earlier_step_id: str
    later_step_id: str
    earlier_step_index: int
    later_step_index: int
    earlier_ready_time_us: float
    later_ready_time_us: float

    @property
    def ready_time_gain_us(self) -> float:
        return self.earlier_ready_time_us - self.later_ready_time_us


@dataclass(frozen=True)
class _OrderComparison:
    comparison_id: int
    tb_id: str
    tb_index: int
    rank: int
    earlier: XmlStep
    later: XmlStep
    earlier_index: int
    later_index: int
    earlier_ready: float
    later_ready: float
    necessary: bool


def _schedule_predecessors(schedule: Schedule) -> Mapping[str, frozenset[str]]:
    raw = schedule.metadata.get("semantic_predecessors", {})
    if not isinstance(raw, Mapping):
        raise SemanticError("semantic_predecessors must be a mapping")
    transfer_ids = {transfer.transfer_id for transfer in schedule.transfers}
    result = {}
    for transfer in schedule.transfers:
        try:
            semantic = frozenset(raw.get(transfer.transfer_id, ()))
        except TypeError as error:
            raise SemanticError("semantic predecessor IDs must be iterable") from error
        predecessors = transfer.predecessor_ids | semantic
        if not predecessors <= transfer_ids:
            raise SemanticError("semantic predecessor is missing")
        result[transfer.transfer_id] = predecessors
    return result


def _ancestors(predecessors: Mapping[str, frozenset[str]]) -> Mapping[str, frozenset[str]]:
    cache: Dict[str, frozenset[str]] = {}
    active = set()

    def visit(node_id: str) -> frozenset[str]:
        if node_id in cache:
            return cache[node_id]
        if node_id in active:
            raise SemanticError("dependency graph contains a cycle")
        active.add(node_id)
        values = set(predecessors.get(node_id, ()))
        for predecessor in tuple(values):
            values.update(visit(predecessor))
        active.remove(node_id)
        cache[node_id] = frozenset(values)
        return cache[node_id]

    for node_id in predecessors:
        visit(node_id)
    return cache


def _step_dependency_ancestors(program: ThreadblockProgram) -> Mapping[str, frozenset[str]]:
    direct = {step_id: set() for step_id in program.steps_by_id}
    for step in program.steps_by_id.values():
        if step.dependency_step_id is not None:
            direct[step.step_id].add(step.dependency_step_id)
    return _ancestors(
        {step_id: frozenset(values) for step_id, values in direct.items()}
    )


def _node_predecessors(
    program: ThreadblockProgram,
    schedule: Schedule,
) -> Mapping[str, frozenset[str]]:
    result = {
        node_id: set(values)
        for node_id, values in _schedule_predecessors(schedule).items()
    }
    for step in program.steps_by_id.values():
        result.setdefault(step.node_id, set()).update(
            step.semantic_predecessor_node_ids
        )
    return {
        node_id: frozenset(values) for node_id, values in result.items()
    }


def _ready_times(
    program: ThreadblockProgram,
    schedule: Schedule,
) -> Mapping[str, float]:
    transfers = {
        transfer.transfer_id: transfer for transfer in schedule.transfers
    }
    predecessors = _node_predecessors(program, schedule)
    node_end_times = {
        transfer_id: transfer.ed_time
        for transfer_id, transfer in transfers.items()
    }
    for step in program.steps_by_id.values():
        node_end_times[step.node_id] = max(
            node_end_times.get(step.node_id, 0.0),
            step.effective_ed_time,
        )
    atom_ready_times = {
        transfer_id: max(
            atom.current_symbol.ready_time for atom in transfer.atoms
        )
        for transfer_id, transfer in transfers.items()
    }
    return {
        step.step_id: max(
            [atom_ready_times.get(step.transfer_id, 0.0)]
            + [
                node_end_times[item]
                for item in predecessors.get(step.node_id, ())
                if item in node_end_times
            ]
            + (
                [program.steps_by_id[step.dependency_step_id].effective_ed_time]
                if step.dependency_step_id is not None
                else []
            ),
            default=0.0,
        )
        for step in program.steps_by_id.values()
    }


def _comparisons(
    program: ThreadblockProgram,
    schedule: Schedule,
) -> Tuple[_OrderComparison, ...]:
    ready = _ready_times(program, schedule)
    node_ancestors = _ancestors(_node_predecessors(program, schedule))
    step_ancestors = _step_dependency_ancestors(program)
    comparisons = []
    tb_keys = tuple(
        sorted(
            (
                "r{:08d}-tb{:08d}".format(tb.key.rank, tb.tb_id),
                tb,
            )
            for tb in program.threadblocks
        )
    )
    comparison_id = 0
    for tb_index, (tb_id, threadblock) in enumerate(tb_keys):
        steps = tuple(
            (index, step)
            for index, step in enumerate(threadblock.steps)
            if step.xml_type is not EndpointType.NOP
        )
        for position, (earlier_index, earlier) in enumerate(steps):
            for later_index, later in steps[position + 1 :]:
                earlier_ready = ready[earlier.step_id]
                later_ready = ready[later.step_id]
                if later_ready >= earlier_ready:
                    continue
                necessary = (
                    earlier.node_id
                    in node_ancestors.get(later.node_id, frozenset())
                    or earlier.step_id
                    in step_ancestors.get(later.step_id, frozenset())
                )
                comparisons.append(
                    _OrderComparison(
                        comparison_id=comparison_id,
                        tb_id=tb_id,
                        tb_index=tb_index,
                        rank=threadblock.key.rank,
                        earlier=earlier,
                        later=later,
                        earlier_index=earlier_index,
                        later_index=later_index,
                        earlier_ready=earlier_ready,
                        later_ready=later_ready,
                        necessary=necessary,
                    )
                )
                comparison_id += 1
    return tuple(comparisons)


def _success(hints: Tuple[TBOrderHint, ...], evidence: Mapping[str, object]):
    return BDDAnalysisResult(
        status=ValidationStatus.VALID,
        code="tb_order_bdd_analysis_complete",
        message="threadblock order BDD analysis completed",
        hints=hints,
        evidence=evidence,
    )


def analyze_tb_order(
    tb_program: ThreadblockProgram,
    schedule: Schedule,
) -> BDDAnalysisResult:
    try:
        if not isinstance(tb_program, ThreadblockProgram):
            raise SemanticError("tb_program must be a ThreadblockProgram")
        if not isinstance(schedule, Schedule):
            raise SemanticError("schedule must be a Schedule")
        comparisons = _comparisons(tb_program, schedule)
        if not comparisons:
            return _success((), {"comparison_count": 0, "relation_count": 0})
        backend = CompactBDD(
            {
                "tb_id": len({item.tb_id for item in comparisons}),
                "op_id": len(comparisons),
                "step_index": max(item.later_index for item in comparisons) + 1,
            }
        )
        tb_ids = {
            value: index
            for index, value in enumerate(
                sorted({item.tb_id for item in comparisons})
            )
        }
        rows = tuple(
            (tb_ids[item.tb_id], item.comparison_id, item.later_index)
            for item in comparisons
        )
        necessary_rows = tuple(
            (tb_ids[item.tb_id], item.comparison_id, item.later_index)
            for item in comparisons
            if item.necessary
        )
        inverted = backend.relation(rows)
        necessary = backend.relation(necessary_rows)
        selected_ids = {
            row[1] for row in inverted.difference(necessary).tuples()
        }
        selected = tuple(
            item for item in comparisons if item.comparison_id in selected_ids
        )
        hints = tuple(
            TBOrderHint(
                tb_id=item.tb_id,
                rank=item.rank,
                earlier_step_id=item.earlier.step_id,
                later_step_id=item.later.step_id,
                earlier_step_index=item.earlier_index,
                later_step_index=item.later_index,
                earlier_ready_time_us=item.earlier_ready,
                later_ready_time_us=item.later_ready,
            )
            for item in selected
        )
        return _success(
            hints,
            {
                "comparison_count": len(comparisons),
                "relation_count": len(selected_ids),
                "bdd_variable_count": backend.variable_count,
            },
        )
    except Exception as error:
        return BDDAnalysisResult(
            status=ValidationStatus.ANALYSIS_ERROR,
            code="tb_order_bdd_analysis_error",
            message="threadblock order BDD analysis failed",
            hints=(),
            evidence={
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
