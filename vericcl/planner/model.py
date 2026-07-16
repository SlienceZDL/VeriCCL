from dataclasses import dataclass
from types import MappingProxyType
from typing import FrozenSet, Mapping, Optional, Tuple

from vericcl.errors import SemanticError
from vericcl.semantics.collective import (
    CollectiveSpec,
    OutputSlot,
    required_outputs,
)
from vericcl.topology.model import LinkKey


def _integer(value: object, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SemanticError("{} must be an integer".format(field))
    if value < minimum:
        raise SemanticError("{} must be at least {}".format(field, minimum))
    return value


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise SemanticError("{} must be a non-empty string".format(field))
    return value


def _contributors(value: object, field: str) -> FrozenSet[int]:
    try:
        contributors = frozenset(value)
    except TypeError as error:
        raise SemanticError("{} must be an iterable".format(field)) from error
    if not contributors:
        raise SemanticError("{} must not be empty".format(field))
    for contributor in contributors:
        _integer(contributor, field)
    return contributors


@dataclass(frozen=True)
class LogicalValue:
    slot: OutputSlot
    contributors: FrozenSet[int]

    def __post_init__(self) -> None:
        if not isinstance(self.slot, OutputSlot):
            raise SemanticError("logical_value.slot must be an OutputSlot")
        object.__setattr__(
            self,
            "contributors",
            _contributors(
                self.contributors,
                "logical_value.contributors",
            ),
        )


@dataclass(frozen=True)
class StageInterface:
    values: Mapping[OutputSlot, FrozenSet[int]]

    def __post_init__(self) -> None:
        try:
            raw_values = dict(self.values)
        except (TypeError, ValueError) as error:
            raise SemanticError("stage interface values must be a mapping") from error
        if not raw_values:
            raise SemanticError("stage interface values must not be empty")
        normalized = {}
        for slot, contributors in raw_values.items():
            value = LogicalValue(slot, contributors)
            normalized[value.slot] = value.contributors
        object.__setattr__(
            self,
            "values",
            MappingProxyType(dict(sorted(normalized.items()))),
        )

    @property
    def logical_values(self) -> Tuple[LogicalValue, ...]:
        return tuple(
            LogicalValue(slot, contributors)
            for slot, contributors in self.values.items()
        )


def _rank_group(value: object) -> Tuple[int, ...]:
    try:
        group = tuple(value)
    except TypeError as error:
        raise SemanticError("communication_group must be iterable") from error
    if not group:
        raise SemanticError("communication_group must not be empty")
    for rank in group:
        _integer(rank, "communication_group rank")
    if group != tuple(sorted(group)) or len(group) != len(set(group)):
        raise SemanticError("communication_group must be sorted and unique")
    return group


@dataclass(frozen=True)
class PlanNode:
    node_id: str
    stage_id: int
    local_collective: CollectiveSpec
    communication_group: Tuple[int, ...]
    logical_input: StageInterface
    logical_output: StageInterface
    allowed_links: FrozenSet[LinkKey]
    shared_resource_ids: FrozenSet[str]
    dual_of_node_id: Optional[str] = None

    def __post_init__(self) -> None:
        _identifier(self.node_id, "plan_node.node_id")
        _integer(self.stage_id, "plan_node.stage_id")
        if not isinstance(self.local_collective, CollectiveSpec):
            raise SemanticError(
                "plan_node.local_collective must be a CollectiveSpec"
            )
        group = _rank_group(self.communication_group)
        if not isinstance(self.logical_input, StageInterface):
            raise SemanticError("plan_node.logical_input must be a StageInterface")
        if not isinstance(self.logical_output, StageInterface):
            raise SemanticError("plan_node.logical_output must be a StageInterface")
        try:
            allowed_links = frozenset(self.allowed_links)
        except TypeError as error:
            raise SemanticError("plan_node.allowed_links must be iterable") from error
        if not all(isinstance(link, LinkKey) for link in allowed_links):
            raise SemanticError("plan_node.allowed_links must contain LinkKey values")
        if any(
            link.src_rank not in group or link.dst_rank not in group
            for link in allowed_links
        ):
            raise SemanticError("plan_node allowed link leaves its communication group")
        try:
            resource_ids = frozenset(self.shared_resource_ids)
        except TypeError as error:
            raise SemanticError(
                "plan_node.shared_resource_ids must be iterable"
            ) from error
        for resource_id in resource_ids:
            _identifier(resource_id, "plan_node.shared_resource_ids")
        if self.dual_of_node_id is not None:
            _identifier(self.dual_of_node_id, "plan_node.dual_of_node_id")
            if self.dual_of_node_id == self.node_id:
                raise SemanticError("plan node cannot be its own dual")
        object.__setattr__(self, "communication_group", group)
        object.__setattr__(self, "allowed_links", allowed_links)
        object.__setattr__(self, "shared_resource_ids", resource_ids)


@dataclass(frozen=True)
class PlanEdge:
    producer_id: str
    consumer_id: str
    interface: StageInterface

    def __post_init__(self) -> None:
        _identifier(self.producer_id, "plan_edge.producer_id")
        _identifier(self.consumer_id, "plan_edge.consumer_id")
        if not isinstance(self.interface, StageInterface):
            raise SemanticError("plan_edge.interface must be a StageInterface")


def _contains(interface: StageInterface, slot: OutputSlot, contributors: frozenset) -> bool:
    return interface.values.get(slot) == contributors


@dataclass(frozen=True)
class PlanDAG:
    collective: CollectiveSpec
    rank_count: int
    slice_count: int
    initial_inputs: StageInterface
    nodes: Tuple[PlanNode, ...]
    edges: Tuple[PlanEdge, ...]
    final_outputs: StageInterface

    def __post_init__(self) -> None:
        if not isinstance(self.collective, CollectiveSpec):
            raise SemanticError("plan collective must be a CollectiveSpec")
        rank_count = _integer(self.rank_count, "plan.rank_count", minimum=1)
        slice_count = _integer(self.slice_count, "plan.slice_count", minimum=1)
        if not isinstance(self.initial_inputs, StageInterface):
            raise SemanticError("plan.initial_inputs must be a StageInterface")
        if not isinstance(self.final_outputs, StageInterface):
            raise SemanticError("plan.final_outputs must be a StageInterface")
        try:
            nodes = tuple(self.nodes)
            edges = tuple(self.edges)
        except TypeError as error:
            raise SemanticError("plan nodes and edges must be iterable") from error
        if not nodes or not all(isinstance(node, PlanNode) for node in nodes):
            raise SemanticError("plan must contain PlanNode values")
        if not all(isinstance(edge, PlanEdge) for edge in edges):
            raise SemanticError("plan edges must contain PlanEdge values")
        node_ids = [node.node_id for node in nodes]
        if len(node_ids) != len(set(node_ids)):
            raise SemanticError("plan node IDs must be unique")
        by_id = {node.node_id: node for node in nodes}
        global_slice_count = rank_count * slice_count
        self._validate_interfaces(nodes, global_slice_count, rank_count)
        expected_initial = {
            OutputSlot(rank, logical_address): frozenset(
                {rank * slice_count + logical_address}
            )
            for rank in range(rank_count)
            for logical_address in range(slice_count)
        }
        if dict(self.initial_inputs.values) != expected_initial:
            raise SemanticError(
                "plan initial inputs do not match the global slice layout"
            )
        expected = required_outputs(self.collective, rank_count, slice_count)
        if dict(self.final_outputs.values) != dict(expected):
            raise SemanticError("plan final outputs do not match collective semantics")
        incoming = {node_id: [] for node_id in node_ids}
        outgoing = {node_id: [] for node_id in node_ids}
        for edge in edges:
            if edge.producer_id not in by_id or edge.consumer_id not in by_id:
                raise SemanticError("plan edge references an unknown node")
            producer = by_id[edge.producer_id]
            consumer = by_id[edge.consumer_id]
            for slot, contributors in edge.interface.values.items():
                if not _contains(producer.logical_output, slot, contributors):
                    raise SemanticError(
                        "plan edge interface is not produced by its producer"
                    )
                if not _contains(consumer.logical_input, slot, contributors):
                    raise SemanticError(
                        "plan edge interface is not required by its consumer"
                    )
            incoming[edge.consumer_id].append(edge)
            outgoing[edge.producer_id].append(edge)
        self._validate_acyclic(node_ids, edges)
        self._validate_inputs(nodes, incoming)
        self._validate_outputs(nodes, outgoing)
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "edges", edges)

    def _validate_interfaces(
        self,
        nodes: Tuple[PlanNode, ...],
        global_slice_count: int,
        rank_count: int,
    ) -> None:
        interfaces = [self.initial_inputs, self.final_outputs]
        for node in nodes:
            if any(rank >= rank_count for rank in node.communication_group):
                raise SemanticError("plan node rank is outside the global range")
            interfaces.extend((node.logical_input, node.logical_output))
        for interface in interfaces:
            for slot, contributors in interface.values.items():
                if slot.rank >= rank_count:
                    raise SemanticError("plan interface rank is outside the global range")
                if any(value >= global_slice_count for value in contributors):
                    raise SemanticError(
                        "plan interface contributor is outside the global range"
                    )

    def _validate_acyclic(
        self,
        node_ids: list,
        edges: Tuple[PlanEdge, ...],
    ) -> None:
        successors = {node_id: set() for node_id in node_ids}
        indegree = {node_id: 0 for node_id in node_ids}
        for edge in edges:
            if edge.consumer_id not in successors[edge.producer_id]:
                successors[edge.producer_id].add(edge.consumer_id)
                indegree[edge.consumer_id] += 1
        ready = sorted(node_id for node_id, degree in indegree.items() if degree == 0)
        visited = []
        while ready:
            node_id = ready.pop(0)
            visited.append(node_id)
            for successor in sorted(successors[node_id]):
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    ready.append(successor)
                    ready.sort()
        if len(visited) != len(node_ids):
            raise SemanticError("plan graph must be acyclic")

    def _validate_inputs(self, nodes: tuple, incoming: dict) -> None:
        for node in nodes:
            supplied = {}
            for edge in incoming[node.node_id]:
                for slot, contributors in edge.interface.values.items():
                    if slot in supplied:
                        raise SemanticError("plan node input is supplied more than once")
                    supplied[slot] = contributors
            for slot, contributors in node.logical_input.values.items():
                if slot in supplied:
                    continue
                elif not _contains(self.initial_inputs, slot, contributors):
                    raise SemanticError("plan node input is not available")

    def _validate_outputs(self, nodes: tuple, outgoing: dict) -> None:
        for slot, contributors in self.final_outputs.values.items():
            if not any(
                _contains(node.logical_output, slot, contributors)
                for node in nodes
            ):
                raise SemanticError("plan final output is not produced")
        for node in nodes:
            used = {}
            for edge in outgoing[node.node_id]:
                used.update(edge.interface.values)
            for slot, contributors in node.logical_output.values.items():
                if used.get(slot) == contributors:
                    continue
                if _contains(self.final_outputs, slot, contributors):
                    continue
                raise SemanticError("plan node produces an unused logical value")
