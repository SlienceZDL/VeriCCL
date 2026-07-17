from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, Mapping, Tuple

from vericcl.errors import SemanticError
from vericcl.semantics.atom import Schedule
from vericcl.semantics.slice import logical_slice_index
from vericcl.xml.endpoints import EndpointProgram
from vericcl.xml.model import AggregateValue, BufferPlan, RawValue, ValueKey


@dataclass(frozen=True)
class TransferNode:
    node_id: str
    kind: str
    endpoint_ids: Tuple[str, ...]
    member_slice_ids: frozenset[int]
    stage_id: int
    logical_slice_index: int
    src_rank: int
    dst_rank: int
    channel: int
    st_time: float
    ed_time: float
    effective_st_time: float
    effective_ed_time: float


@dataclass(frozen=True)
class TransferDAG:
    nodes: Mapping[str, TransferNode]
    predecessors: Mapping[str, frozenset[str]]
    edge_reasons: Mapping[Tuple[str, str], frozenset[str]]
    topological_order: Tuple[str, ...]

    def __post_init__(self) -> None:
        nodes = dict(self.nodes)
        predecessors = {
            node_id: frozenset(values)
            for node_id, values in self.predecessors.items()
        }
        if set(nodes) != set(predecessors):
            raise SemanticError("dependency graph must cover every node")
        for node_id, values in predecessors.items():
            if node_id in values or not values <= set(nodes):
                raise SemanticError("dependency graph contains an invalid edge")
        reasons = {
            edge: frozenset(values) for edge, values in self.edge_reasons.items()
        }
        expected_edges = {
            (predecessor, node_id)
            for node_id, values in predecessors.items()
            for predecessor in values
        }
        if any(
            edge[0] not in nodes
            or edge[1] not in nodes
            or edge[0] not in predecessors[edge[1]]
            or not values
            for edge, values in reasons.items()
        ) or set(reasons) != expected_edges:
            raise SemanticError("dependency reason does not match an edge")
        order = tuple(self.topological_order)
        if set(order) != set(nodes) or len(order) != len(nodes):
            raise SemanticError("dependency topological order is incomplete")
        position = {node_id: index for index, node_id in enumerate(order)}
        if any(
            position[predecessor] >= position[node_id]
            for node_id, values in predecessors.items()
            for predecessor in values
        ):
            raise SemanticError("dependency graph contains a cycle")
        object.__setattr__(self, "nodes", MappingProxyType(nodes))
        object.__setattr__(
            self,
            "predecessors",
            MappingProxyType(predecessors),
        )
        object.__setattr__(self, "edge_reasons", MappingProxyType(reasons))
        object.__setattr__(self, "topological_order", order)


def _contributors(value: ValueKey) -> frozenset[int]:
    if isinstance(value, RawValue):
        return frozenset({value.slice_id})
    return value.contributors


def _logical(value: ValueKey, slice_count: int) -> int:
    if isinstance(value, RawValue):
        return logical_slice_index(value.slice_id, slice_count)
    return value.logical_slice_index


def _topological_order(
    nodes: Mapping[str, TransferNode],
    predecessors: Mapping[str, set[str]],
) -> Tuple[str, ...]:
    completed = set()
    remaining = set(nodes)
    ordered = []
    while remaining:
        ready = [
            nodes[node_id]
            for node_id in remaining
            if predecessors[node_id] <= completed
        ]
        if not ready:
            raise SemanticError("transfer dependency graph contains a cycle")
        selected = min(
            ready,
            key=lambda node: (
                node.st_time,
                node.ed_time,
                node.stage_id,
                node.node_id,
            ),
        )
        ordered.append(selected.node_id)
        completed.add(selected.node_id)
        remaining.remove(selected.node_id)
    return tuple(ordered)


def build_transfer_dag(
    program: EndpointProgram,
    schedule: Schedule,
    buffers: BufferPlan,
) -> TransferDAG:
    if not isinstance(program, EndpointProgram):
        raise SemanticError("program must be an EndpointProgram")
    if not isinstance(schedule, Schedule):
        raise SemanticError("schedule must be a Schedule")
    if not isinstance(buffers, BufferPlan):
        raise SemanticError("buffers must be a BufferPlan")
    transfer_ids = {transfer.transfer_id for transfer in schedule.transfers}
    if set(program.by_transfer_id) != transfer_ids:
        raise SemanticError("endpoint program does not match schedule transfers")
    required_mappings = (
        buffers.transfer_src_refs,
        buffers.transfer_dst_refs,
        buffers.transfer_input_values,
        buffers.transfer_output_values,
        buffers.transfer_effective_times,
    )
    if any(set(values) != transfer_ids for values in required_mappings):
        raise SemanticError("buffer plan does not cover schedule transfers")
    reduce_ids = {
        transfer.transfer_id
        for transfer in schedule.transfers
        if transfer.kind == "REDUCE"
    }
    if (
        set(buffers.transfer_accumulator_refs) != reduce_ids
        or set(buffers.transfer_accumulator_values) != reduce_ids
    ):
        raise SemanticError("buffer plan reduction state is incomplete")

    nodes: Dict[str, TransferNode] = {}
    for transfer in schedule.transfers:
        if transfer.transfer_id not in buffers.transfer_effective_times:
            raise SemanticError("transfer effective timing is missing")
        effective_start, effective_end = buffers.transfer_effective_times[
            transfer.transfer_id
        ]
        nodes[transfer.transfer_id] = TransferNode(
            node_id=transfer.transfer_id,
            kind=transfer.kind,
            endpoint_ids=tuple(
                endpoint.endpoint_id
                for endpoint in program.by_transfer_id[transfer.transfer_id]
            ),
            member_slice_ids=transfer.member_slice_ids,
            stage_id=transfer.stage_id,
            logical_slice_index=min(
                logical_slice_index(member, schedule.slice_count)
                for member in transfer.member_slice_ids
            ),
            src_rank=transfer.src_rank,
            dst_rank=transfer.dst_rank,
            channel=transfer.channel,
            st_time=transfer.st_time,
            ed_time=transfer.ed_time,
            effective_st_time=effective_start,
            effective_ed_time=effective_end,
        )
    copies = {copy.copy_id: copy for copy in buffers.local_copies}
    if transfer_ids.intersection(copies):
        raise SemanticError("transfer and local copy IDs must be disjoint")
    if set(program.local_endpoints) != set(copies):
        raise SemanticError("endpoint program does not match local copies")
    for copy_id, copy in copies.items():
        nodes[copy_id] = TransferNode(
            node_id=copy_id,
            kind="COPY",
            endpoint_ids=(program.local_endpoints[copy_id].endpoint_id,),
            member_slice_ids=frozenset(),
            stage_id=-1,
            logical_slice_index=-1,
            src_rank=copy.rank,
            dst_rank=copy.rank,
            channel=-1,
            st_time=copy.st_time,
            ed_time=copy.ed_time,
            effective_st_time=copy.st_time,
            effective_ed_time=copy.ed_time,
        )

    predecessors = {node_id: set() for node_id in nodes}
    reasons = defaultdict(set)

    def add_edge(predecessor: str, consumer: str, reason: str) -> None:
        if predecessor not in nodes or consumer not in nodes:
            raise SemanticError("dependency references a missing node")
        if predecessor == consumer:
            raise SemanticError("dependency node must not depend on itself")
        predecessors[consumer].add(predecessor)
        reasons[(predecessor, consumer)].add(reason)

    raw_semantic = schedule.metadata.get("semantic_predecessors", {})
    if not isinstance(raw_semantic, Mapping):
        raise SemanticError("semantic_predecessors metadata must be a mapping")
    for transfer in schedule.transfers:
        for predecessor in transfer.predecessor_ids:
            add_edge(predecessor, transfer.transfer_id, "schedule")
        try:
            semantic = tuple(raw_semantic.get(transfer.transfer_id, ()))
        except TypeError as error:
            raise SemanticError(
                "semantic predecessor IDs must be iterable"
            ) from error
        for predecessor in semantic:
            add_edge(predecessor, transfer.transfer_id, "semantic")

    path_index = {}
    for transfer in schedule.transfers:
        for atom in transfer.atoms:
            symbol = atom.current_symbol
            key = (
                atom.slice_id,
                transfer.stage_id,
                transfer.kind,
                symbol.src_rank,
                symbol.dst_rank,
                symbol.ready_time,
            )
            if key in path_index:
                raise SemanticError("path operation maps to multiple transfers")
            path_index[key] = transfer.transfer_id
    for transfer in schedule.transfers:
        for atom in transfer.atoms:
            flattened = [
                (stage.stage_id, stage.operator, symbol)
                for stage in atom.path
                for symbol in stage.symbols
            ]
            if len(flattened) == 1:
                continue
            immediate_predecessor = None
            for stage_id, operator, symbol in flattened[:-1]:
                key = (
                    atom.slice_id,
                    stage_id,
                    operator,
                    symbol.src_rank,
                    symbol.dst_rank,
                    symbol.ready_time,
                )
                predecessor = path_index.get(key)
                if predecessor is None:
                    raise SemanticError("atom path predecessor is missing")
                immediate_predecessor = predecessor
            add_edge(
                immediate_predecessor,
                transfer.transfer_id,
                "path",
            )

    output_producers = defaultdict(set)
    exact_producers = defaultdict(set)
    for transfer_id, value in buffers.transfer_output_values.items():
        output_producers[value].add(transfer_id)
        exact_producers[
            (value, buffers.transfer_dst_refs[transfer_id])
        ].add(transfer_id)
    for copy in buffers.local_copies:
        if copy.predecessor_state_id in nodes:
            add_edge(copy.predecessor_state_id, copy.copy_id, "copy_source")
    for transfer_id, value in buffers.transfer_input_values.items():
        source_ref = buffers.transfer_src_refs[transfer_id]
        for producer in exact_producers.get((value, source_ref), ()):
            add_edge(producer, transfer_id, "buffer_state")
        for copy in buffers.local_copies:
            if copy.dst_ref == source_ref:
                add_edge(copy.copy_id, transfer_id, "buffer_init")

    for transfer in schedule.transfers:
        transfer_id = transfer.transfer_id
        input_value = buffers.transfer_input_values[transfer_id]
        output_value = buffers.transfer_output_values[transfer_id]
        if transfer.kind == "SEND":
            if input_value != output_value:
                raise SemanticError("SEND crosses value state versions")
            continue
        accumulator = buffers.transfer_accumulator_values.get(transfer_id)
        if accumulator is None:
            raise SemanticError("REDUCE accumulator state is missing")
        if (
            _logical(input_value, schedule.slice_count)
            != _logical(accumulator, schedule.slice_count)
            or _contributors(input_value).intersection(_contributors(accumulator))
            or not isinstance(output_value, AggregateValue)
            or output_value.contributors
            != _contributors(input_value) | _contributors(accumulator)
            or output_value.logical_slice_index
            != _logical(input_value, schedule.slice_count)
            or (
                isinstance(accumulator, AggregateValue)
                and output_value.state_version <= accumulator.state_version
            )
        ):
            raise SemanticError("REDUCE crosses an accumulator state version")
        for producer in output_producers.get(accumulator, ()):
            add_edge(producer, transfer_id, "buffer_state")
        accumulator_ref = buffers.transfer_accumulator_refs.get(transfer_id)
        if accumulator_ref is None:
            raise SemanticError("REDUCE accumulator reference is missing")
        for copy in buffers.local_copies:
            if copy.dst_ref == accumulator_ref:
                add_edge(copy.copy_id, transfer_id, "buffer_init")

    alias_parent = {}

    def address(ref):
        return ref.rank, ref.buffer, ref.offset

    def find(value):
        alias_parent.setdefault(value, value)
        if alias_parent[value] != value:
            alias_parent[value] = find(alias_parent[value])
        return alias_parent[value]

    def union(left, right):
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            root, child = sorted((left_root, right_root))
            alias_parent[child] = root

    for left, right in buffers.aliases:
        union(address(left), address(right))
    for copy in buffers.local_copies:
        if copy.reason != "preserve live in-place input":
            continue
        source_address = find(address(copy.src_ref))
        for transfer in schedule.transfers:
            if (
                transfer.dst_rank == copy.rank
                and find(address(buffers.transfer_dst_refs[transfer.transfer_id]))
                == source_address
                and buffers.transfer_effective_times[transfer.transfer_id][0]
                >= copy.ed_time
            ):
                add_edge(copy.copy_id, transfer.transfer_id, "buffer_antidependency")

    order = _topological_order(nodes, predecessors)
    return TransferDAG(
        nodes=nodes,
        predecessors={
            node_id: frozenset(values)
            for node_id, values in predecessors.items()
        },
        edge_reasons={
            edge: frozenset(values) for edge, values in reasons.items()
        },
        topological_order=order,
    )
