from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Dict, List, Mapping, Tuple

from vericcl.errors import SemanticError
from vericcl.xml.dependencies import TransferDAG, TransferNode
from vericcl.xml.endpoints import EndpointAtom, EndpointProgram, EndpointType
from vericcl.xml.threadblocks import (
    Threadblock,
    ThreadblockKey,
    ThreadblockProgram,
    XmlStep,
)


def _threadblock_key(endpoint: EndpointAtom) -> ThreadblockKey:
    if endpoint.xml_type is EndpointType.COPY:
        return ThreadblockKey(endpoint.rank, "copy", -1, -1)
    direction = "send" if endpoint.xml_type is EndpointType.SEND else "recv"
    return ThreadblockKey(
        endpoint.rank,
        direction,
        endpoint.peer,
        endpoint.channel,
    )


def _critical_paths(dag: TransferDAG) -> Mapping[str, float]:
    successors = defaultdict(set)
    for node_id, predecessors in dag.predecessors.items():
        for predecessor in predecessors:
            successors[predecessor].add(node_id)
    lengths = {}
    for node_id in reversed(dag.topological_order):
        node = dag.nodes[node_id]
        duration = max(0.0, node.effective_ed_time - node.effective_st_time)
        lengths[node_id] = duration + max(
            (lengths[successor] for successor in successors[node_id]),
            default=0.0,
        )
    return lengths


def _node_priority(node: TransferNode, critical_path: float) -> tuple:
    return (
        node.st_time,
        -critical_path,
        node.ed_time,
        node.stage_id,
        node.logical_slice_index,
        node.src_rank,
        node.dst_rank,
        node.channel,
        node.node_id,
    )


def _inversion_count(
    threadblocks: Mapping[ThreadblockKey, List[XmlStep]],
    dag: TransferDAG,
) -> int:
    count = 0
    for steps in threadblocks.values():
        node_ids = [
            step.node_id for step in steps if step.xml_type is not EndpointType.NOP
        ]
        expected = sorted(
            node_ids,
            key=lambda node_id: _node_priority(dag.nodes[node_id], 0.0),
        )
        position = {node_id: index for index, node_id in enumerate(expected)}
        count += sum(
            position[left] > position[right]
            for index, left in enumerate(node_ids)
            for right in node_ids[index + 1 :]
        )
    return count


def _verify_step_graph(program: ThreadblockProgram) -> None:
    predecessors = {step_id: set() for step_id in program.steps_by_id}
    for threadblock in program.threadblocks:
        for left, right in zip(threadblock.steps, threadblock.steps[1:]):
            predecessors[right.step_id].add(left.step_id)
    for step in program.steps_by_id.values():
        if step.dependency_step_id is not None:
            predecessors[step.step_id].add(step.dependency_step_id)
    completed = set()
    remaining = set(predecessors)
    while remaining:
        ready = sorted(
            step_id
            for step_id in remaining
            if predecessors[step_id] <= completed
        )
        if not ready:
            raise SemanticError("threadblock lowering introduces a dependency cycle")
        for step_id in ready:
            completed.add(step_id)
            remaining.remove(step_id)


def schedule_threadblocks(
    program: EndpointProgram,
    dag: TransferDAG,
) -> ThreadblockProgram:
    if not isinstance(program, EndpointProgram):
        raise SemanticError("program must be an EndpointProgram")
    if not isinstance(dag, TransferDAG):
        raise SemanticError("dag must be a TransferDAG")
    endpoint_by_id = {endpoint.endpoint_id: endpoint for endpoint in program.endpoints}
    if any(
        endpoint_id not in endpoint_by_id
        for node in dag.nodes.values()
        for endpoint_id in node.endpoint_ids
    ):
        raise SemanticError("DAG endpoint is missing from the endpoint program")

    critical_paths = _critical_paths(dag)
    remaining_predecessors = {
        node_id: set(predecessors)
        for node_id, predecessors in dag.predecessors.items()
    }
    completed_nodes = set()
    remaining_nodes = set(dag.nodes)
    threadblock_steps: Dict[ThreadblockKey, List[XmlStep]] = defaultdict(list)
    node_rank_steps: Dict[str, Dict[int, str]] = defaultdict(dict)
    node_steps = {}
    nop_index = 0

    def predecessor_assignments(
        node_id: str,
        endpoints: Tuple[EndpointAtom, ...],
    ) -> Mapping[str, List[Tuple[str, str]]]:
        assignments = {endpoint.endpoint_id: [] for endpoint in endpoints}
        if len(endpoints) == 1:
            endpoint = endpoints[0]
            for predecessor in dag.predecessors[node_id]:
                step_id = node_rank_steps[predecessor].get(endpoint.rank)
                if step_id is None:
                    raise SemanticError("local dependency has no same-rank step")
                assignments[endpoint.endpoint_id].append((predecessor, step_id))
            return assignments
        send = next(
            endpoint
            for endpoint in endpoints
            if endpoint.xml_type is EndpointType.SEND
        )
        recv = next(endpoint for endpoint in endpoints if endpoint is not send)
        for predecessor in dag.predecessors[node_id]:
            reasons = dag.edge_reasons[(predecessor, node_id)]
            send_step = node_rank_steps[predecessor].get(send.rank)
            recv_step = node_rank_steps[predecessor].get(recv.rank)
            if send_step is not None:
                assignments[send.endpoint_id].append((predecessor, send_step))
            elif recv_step is not None and reasons.intersection(
                {"buffer_state", "buffer_init", "buffer_antidependency"}
            ):
                assignments[recv.endpoint_id].append((predecessor, recv_step))
            elif reasons - {"schedule"}:
                raise SemanticError(
                    "semantic dependency cannot be represented on one rank"
                )
        return assignments

    def append_endpoint(
        endpoint: EndpointAtom,
        node: TransferNode,
        dependencies: List[Tuple[str, str]],
    ) -> str:
        nonlocal nop_index
        key = _threadblock_key(endpoint)
        existing = {step.step_id for step in threadblock_steps[key]}
        unique = {}
        for predecessor, step_id in dependencies:
            if step_id not in existing:
                unique[step_id] = predecessor
        ordered = sorted(
            unique.items(),
            key=lambda item: (
                -dag.nodes[item[1]].effective_ed_time,
                -critical_paths[item[1]],
                item[1],
                item[0],
            ),
        )
        direct_dependency = ordered[0][0] if ordered else None
        for step_id, predecessor in ordered[1:]:
            nop = XmlStep(
                step_id="nop:{}:{:08d}".format(node.node_id, nop_index),
                node_id=node.node_id,
                transfer_id=node.node_id,
                endpoint_id=None,
                xml_type=EndpointType.NOP,
                rank=endpoint.rank,
                peer=endpoint.peer,
                channel=endpoint.channel,
                src_ref=None,
                dst_ref=None,
                dependency_step_id=step_id,
                has_dependence=False,
                semantic_predecessor_node_ids=(predecessor,),
                member_slice_ids=node.member_slice_ids,
                solver_st_time=node.st_time,
                solver_ed_time=node.ed_time,
                effective_st_time=node.effective_st_time,
                effective_ed_time=node.effective_ed_time,
            )
            nop_index += 1
            threadblock_steps[key].append(nop)
        step = XmlStep(
            step_id=endpoint.endpoint_id,
            node_id=node.node_id,
            transfer_id=endpoint.transfer_id,
            endpoint_id=endpoint.endpoint_id,
            xml_type=endpoint.xml_type,
            rank=endpoint.rank,
            peer=endpoint.peer,
            channel=endpoint.channel,
            src_ref=endpoint.src_ref,
            dst_ref=endpoint.dst_ref,
            dependency_step_id=direct_dependency,
            has_dependence=False,
            semantic_predecessor_node_ids=tuple(
                sorted(dag.predecessors[node.node_id])
            ),
            member_slice_ids=node.member_slice_ids,
            solver_st_time=node.st_time,
            solver_ed_time=node.ed_time,
            effective_st_time=node.effective_st_time,
            effective_ed_time=node.effective_ed_time,
        )
        threadblock_steps[key].append(step)
        return step.step_id

    while remaining_nodes:
        ready = [
            dag.nodes[node_id]
            for node_id in remaining_nodes
            if remaining_predecessors[node_id] <= completed_nodes
        ]
        if not ready:
            raise SemanticError("no dependency-ready TransferNode is available")
        node = min(
            ready,
            key=lambda value: _node_priority(
                value,
                critical_paths[value.node_id],
            ),
        )
        endpoints = tuple(
            endpoint_by_id[endpoint_id] for endpoint_id in node.endpoint_ids
        )
        assignments = predecessor_assignments(node.node_id, endpoints)
        actual_step_ids = []
        for endpoint in endpoints:
            step_id = append_endpoint(
                endpoint,
                node,
                assignments[endpoint.endpoint_id],
            )
            actual_step_ids.append(step_id)
            node_rank_steps[node.node_id][endpoint.rank] = step_id
        node_steps[node.node_id] = tuple(actual_step_ids)
        completed_nodes.add(node.node_id)
        remaining_nodes.remove(node.node_id)

    referenced = frozenset(
        step.dependency_step_id
        for steps in threadblock_steps.values()
        for step in steps
        if step.dependency_step_id is not None
    )
    for key, steps in tuple(threadblock_steps.items()):
        threadblock_steps[key] = [
            replace(step, has_dependence=step.step_id in referenced)
            for step in steps
        ]

    direction_order = {"copy": 0, "recv": 1, "send": 2}
    ordered_keys = sorted(
        threadblock_steps,
        key=lambda key: (
            key.rank,
            direction_order[key.direction],
            key.peer,
            key.channel,
        ),
    )
    next_id = defaultdict(int)
    threadblocks = []
    for key in ordered_keys:
        threadblocks.append(
            Threadblock(
                key=key,
                tb_id=next_id[key.rank],
                steps=tuple(threadblock_steps[key]),
            )
        )
        next_id[key.rank] += 1
    steps_by_id = {
        step.step_id: step for tb in threadblocks for step in tb.steps
    }
    transfer_steps = {
        transfer_id: tuple(endpoint.endpoint_id for endpoint in endpoints)
        for transfer_id, endpoints in program.by_transfer_id.items()
    }
    lowered = ThreadblockProgram(
        threadblocks=tuple(threadblocks),
        steps_by_id=steps_by_id,
        transfer_steps=transfer_steps,
        node_steps=node_steps,
        referenced_step_ids=referenced,
        inversion_count=_inversion_count(threadblock_steps, dag),
    )
    _verify_step_graph(lowered)
    return lowered
