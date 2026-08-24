from typing import Dict, Mapping, Tuple

from vericcl.composer.dual import reverse_allgather_schedule
from vericcl.composer.timing import _retime
from vericcl.errors import SemanticError
from vericcl.planner.model import PlanDAG, PlanNode
from vericcl.semantics.atom import Atom, PathStage, Schedule, Symbol, Transfer
from vericcl.semantics.collective import OutputSlot
from vericcl.solver.global_scheduler import assign_global_resources
from vericcl.solver.model import SolveCandidate
from vericcl.topology.model import Topology


def _topological_nodes(plan: PlanDAG) -> Tuple[PlanNode, ...]:
    by_id = {node.node_id: node for node in plan.nodes}
    successors = {node.node_id: set() for node in plan.nodes}
    indegree = {node.node_id: 0 for node in plan.nodes}
    for edge in plan.edges:
        if edge.consumer_id not in successors[edge.producer_id]:
            successors[edge.producer_id].add(edge.consumer_id)
            indegree[edge.consumer_id] += 1
    ready = sorted(node_id for node_id, value in indegree.items() if value == 0)
    ordered = []
    while ready:
        node_id = ready.pop(0)
        ordered.append(by_id[node_id])
        for successor in sorted(successors[node_id]):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
                ready.sort()
    return tuple(ordered)


def _node_schedules(
    plan: PlanDAG,
    candidates: Mapping[str, SolveCandidate],
) -> Mapping[str, Schedule]:
    if not isinstance(candidates, Mapping):
        raise SemanticError("candidates must be a mapping")
    expected = {node.node_id for node in plan.nodes}
    if set(candidates) != expected:
        raise SemanticError("one candidate is required for every plan node")
    schedules = {}
    by_id = {node.node_id: node for node in plan.nodes}
    for node_id, candidate in candidates.items():
        if not isinstance(candidate, SolveCandidate):
            raise SemanticError("candidates must contain SolveCandidate values")
        if node_id not in candidate.node_schedules:
            raise SemanticError("candidate does not contain the plan node schedule")
        schedule = candidate.node_schedules[node_id]
        if (
            schedule.rank_count != plan.rank_count
            or schedule.slice_count != plan.slice_count
        ):
            raise SemanticError("node schedule dimensions do not match the plan")
        if schedule.metadata.get("reduction_dual") is True:
            node = by_id[node_id]
            schedule = reverse_allgather_schedule(
                schedule,
                node.local_collective,
                node.logical_output,
            )
        schedules[node_id] = schedule
    return schedules


def _route_node_schedules(
    plan: PlanDAG,
    node_schedules: Mapping[str, Schedule],
) -> Mapping[str, Schedule]:
    if not isinstance(node_schedules, Mapping):
        raise SemanticError("node_schedules must be a mapping")
    expected = {node.node_id for node in plan.nodes}
    if set(node_schedules) != expected:
        raise SemanticError("one routing schedule is required for every plan node")
    schedules = {}
    by_id = {node.node_id: node for node in plan.nodes}
    for node_id, schedule in node_schedules.items():
        if not isinstance(schedule, Schedule):
            raise SemanticError("node_schedules must contain Schedule values")
        if (
            schedule.rank_count != plan.rank_count
            or schedule.slice_count != plan.slice_count
        ):
            raise SemanticError("node schedule dimensions do not match the plan")
        if schedule.metadata.get("routing_only") is not True:
            raise SemanticError("compose_routes requires routing-only schedules")
        if schedule.metadata.get("reduction_dual") is True:
            node = by_id[node_id]
            schedule = reverse_allgather_schedule(
                schedule,
                node.local_collective,
                node.logical_output,
            )
        schedules[node_id] = schedule
    return schedules


def _path_root(schedule: Schedule, transfer: Transfer, slice_id: int) -> int:
    roots = schedule.metadata.get("path_roots")
    if not isinstance(roots, Mapping) or transfer.transfer_id not in roots:
        raise SemanticError("stage schedule requires path_roots metadata")
    value = roots[transfer.transfer_id]
    if isinstance(value, Mapping):
        return value[slice_id]
    return value


def _input_slot(node: PlanNode, root: int, slice_id: int) -> OutputSlot:
    matches = [
        slot
        for slot, contributors in node.logical_input.values.items()
        if slot.rank == root and slice_id in contributors
    ]
    if len(matches) != 1:
        raise SemanticError("atom does not map to one logical stage input")
    return matches[0]


def _tree_contributors(schedule: Schedule, transfer: Transfer) -> frozenset:
    values = schedule.metadata.get(
        "tree_contributors",
        schedule.metadata.get("semantic_contributors", {}),
    )
    if not isinstance(values, Mapping) or transfer.transfer_id not in values:
        raise SemanticError("stage schedule requires contributor metadata")
    return frozenset(values[transfer.transfer_id])


def _semantic_predecessors(
    schedule: Schedule,
    transfer: Transfer,
) -> frozenset:
    values = schedule.metadata.get("semantic_predecessors")
    if values is None:
        return transfer.predecessor_ids
    if not isinstance(values, Mapping) or transfer.transfer_id not in values:
        raise SemanticError("semantic_predecessors metadata is incomplete")
    return frozenset(values[transfer.transfer_id])


def _resource_slots(
    schedule: Schedule,
    transfer: Transfer,
) -> Mapping[str, int]:
    values = schedule.metadata.get("resource_slots", {})
    if not isinstance(values, Mapping):
        raise SemanticError("resource_slots metadata must be a mapping")
    slots = values.get(transfer.transfer_id, {})
    if not isinstance(slots, Mapping):
        raise SemanticError("transfer resource slots must be a mapping")
    return dict(slots)


def _incoming_producers(plan: PlanDAG) -> Mapping[Tuple[str, OutputSlot], str]:
    result = {}
    for edge in plan.edges:
        for slot in edge.interface.values:
            result[(edge.consumer_id, slot)] = edge.producer_id
    return result


def _placeholder_atom(
    atom: Atom,
    prior_path: Tuple[PathStage, ...],
    duration: float,
) -> Atom:
    prior_ready = 0.0
    if prior_path:
        prior_ready = prior_path[-1].symbols[-1].ready_time
    suffix = []
    for stage in atom.path:
        symbols = tuple(
            Symbol(
                symbol.src_rank,
                symbol.dst_rank,
                max(prior_ready, symbol.ready_time),
            )
            for symbol in stage.symbols
        )
        suffix.append(PathStage(stage.stage_id, stage.operator, symbols))
        prior_ready = symbols[-1].ready_time
    start = max(atom.st_time, prior_ready)
    return Atom(
        slice_id=atom.slice_id,
        slice_size_bytes=atom.slice_size_bytes,
        path=prior_path + tuple(suffix),
        st_time=start,
        ed_time=start + duration,
    )


def _output_matches(
    schedule: Schedule,
    slot: OutputSlot,
    contributors: frozenset,
) -> Tuple[Transfer, ...]:
    matches = []
    for transfer in schedule.transfers:
        if (
            transfer.dst_rank == slot.rank
            and _tree_contributors(schedule, transfer) == contributors
        ):
            matches.append(transfer)
    return tuple(sorted(matches, key=lambda item: item.transfer_id))


def _output_path_transfer(
    schedule: Schedule,
    matches: Tuple[Transfer, ...],
    slot: OutputSlot,
    member: int,
) -> Transfer | None:
    transfers = {
        transfer.transfer_id: transfer for transfer in schedule.transfers
    }
    pending = [transfer.transfer_id for transfer in matches]
    reachable = set()
    while pending:
        transfer_id = pending.pop()
        if transfer_id in reachable:
            continue
        reachable.add(transfer_id)
        pending.extend(
            predecessor_id
            for predecessor_id in _semantic_predecessors(
                schedule,
                transfers[transfer_id],
            )
            if predecessor_id in transfers
        )
    candidates = tuple(
        transfer
        for transfer in schedule.transfers
        if transfer.transfer_id in reachable
        and transfer.dst_rank == slot.rank
        and member in transfer.member_slice_ids
    )
    if len(candidates) > 1:
        raise SemanticError("node output member has ambiguous accumulator paths")
    return candidates[0] if candidates else None


def _passthrough_input(
    node: PlanNode,
    slot: OutputSlot,
    contributors: frozenset,
) -> OutputSlot:
    matches = [
        input_slot
        for input_slot, input_contributors in node.logical_input.values.items()
        if input_slot.rank == slot.rank
        and input_contributors == contributors
    ]
    if len(matches) != 1:
        raise SemanticError("node output has no unique pass-through input")
    return matches[0]


def _compose_schedules(
    plan: PlanDAG,
    schedules: Mapping[str, Schedule],
    *,
    routing_only: bool,
) -> Schedule:
    slice_sizes = {
        schedule.slice_size_bytes for schedule in schedules.values()
    }
    if len(slice_sizes) != 1:
        raise SemanticError("node schedules must use one slice size")
    slice_size_bytes = next(iter(slice_sizes))
    producers = _incoming_producers(plan)
    value_dependencies = {}
    value_paths = {}
    transfers_by_id: Dict[str, Transfer] = {}
    semantic = {}
    resource_slots = {}
    for node in _topological_nodes(plan):
        schedule = schedules[node.node_id]
        input_dependencies = {}
        input_paths = {}
        for slot, contributors in node.logical_input.values.items():
            producer_id = producers.get((node.node_id, slot))
            if producer_id is None:
                if plan.initial_inputs.values.get(slot) != contributors:
                    raise SemanticError("plan node input has no producer")
                input_dependencies[slot] = frozenset()
                input_paths[slot] = {
                    member: tuple() for member in contributors
                }
            else:
                key = (producer_id, slot)
                input_dependencies[slot] = value_dependencies[key]
                input_paths[slot] = value_paths[key]
        global_atoms = {}
        for transfer in schedule.transfers:
            duration = (
                0.0
                if routing_only
                else transfer.ed_time - transfer.st_time
            )
            atoms = []
            cross_dependencies = set()
            for atom in transfer.atoms:
                root = _path_root(
                    schedule,
                    transfer,
                    atom.slice_id,
                )
                slot = _input_slot(node, root, atom.slice_id)
                prior_path = input_paths[slot][atom.slice_id]
                suffix_atom = atom
                if routing_only:
                    suffix_atom = Atom(
                        slice_id=atom.slice_id,
                        slice_size_bytes=atom.slice_size_bytes,
                        path=tuple(
                            PathStage(
                                stage.stage_id,
                                stage.operator,
                                tuple(
                                    Symbol(
                                        symbol.src_rank,
                                        symbol.dst_rank,
                                        0.0,
                                    )
                                    for symbol in stage.symbols
                                ),
                            )
                            for stage in atom.path
                        ),
                        st_time=0.0,
                        ed_time=0.0,
                    )
                placeholder = _placeholder_atom(
                    suffix_atom,
                    prior_path,
                    duration,
                )
                atoms.append(placeholder)
                cross_dependencies.update(input_dependencies[slot])
                global_atoms[(transfer.transfer_id, atom.slice_id)] = placeholder
            start = max(atom.st_time for atom in atoms)
            rebuilt_atoms = tuple(
                Atom(
                    slice_id=atom.slice_id,
                    slice_size_bytes=atom.slice_size_bytes,
                    path=atom.path,
                    st_time=start,
                    ed_time=start + duration,
                )
                for atom in atoms
            )
            predecessors = _semantic_predecessors(schedule, transfer)
            predecessors = frozenset(set(predecessors) | cross_dependencies)
            rebuilt = Transfer(
                transfer_id=transfer.transfer_id,
                kind=transfer.kind,
                src_rank=transfer.src_rank,
                dst_rank=transfer.dst_rank,
                channel=0 if routing_only else transfer.channel,
                stage_id=transfer.stage_id,
                member_slice_ids=transfer.member_slice_ids,
                atoms=rebuilt_atoms,
                st_time=start,
                ed_time=start + duration,
                predecessor_ids=predecessors,
            )
            slots = {} if routing_only else _resource_slots(schedule, transfer)
            if rebuilt.transfer_id in transfers_by_id:
                if (
                    transfers_by_id[rebuilt.transfer_id] != rebuilt
                    or resource_slots[rebuilt.transfer_id] != slots
                ):
                    raise SemanticError("reused transfer ID has conflicting content")
            else:
                transfers_by_id[rebuilt.transfer_id] = rebuilt
                resource_slots[rebuilt.transfer_id] = slots
            semantic[rebuilt.transfer_id] = predecessors
            for atom in rebuilt_atoms:
                global_atoms[(transfer.transfer_id, atom.slice_id)] = atom
        for slot, contributors in node.logical_output.values.items():
            matches = _output_matches(schedule, slot, contributors)
            if matches:
                dependencies = frozenset(
                    transfer.transfer_id for transfer in matches
                )
            else:
                passthrough = _passthrough_input(node, slot, contributors)
                dependencies = input_dependencies[passthrough]
            paths = {}
            for member in contributors:
                path = None
                transfer = _output_path_transfer(
                    schedule,
                    matches,
                    slot,
                    member,
                )
                if transfer is not None:
                    path = global_atoms[(transfer.transfer_id, member)].path
                if path is None:
                    input_matches = [
                        input_slot
                        for input_slot, values in node.logical_input.values.items()
                        if input_slot.rank == slot.rank and member in values
                    ]
                    if len(input_matches) != 1:
                        raise SemanticError(
                            "node output member has no input path"
                        )
                    path = input_paths[input_matches[0]][member]
                paths[member] = path
            value_dependencies[(node.node_id, slot)] = dependencies
            value_paths[(node.node_id, slot)] = paths
    final_dependencies = {}
    final_outputs = {}
    for slot, contributors in plan.final_outputs.values.items():
        producers_for_value = [
            node
            for node in plan.nodes
            if node.logical_output.values.get(slot) == contributors
        ]
        if not producers_for_value:
            raise SemanticError("final output has no producing node")
        producer = max(
            producers_for_value,
            key=lambda node: (node.stage_id, node.node_id),
        )
        key = "r{:08d}-o{:08d}".format(slot.rank, slot.offset)
        final_dependencies[key] = tuple(
            sorted(value_dependencies[(producer.node_id, slot)])
        )
        final_outputs[key] = tuple(sorted(contributors))
    provisional = Schedule(
        schedule_id="vericcl-composed",
        transfers=tuple(
            transfers_by_id[key] for key in sorted(transfers_by_id)
        ),
        final_state_ids=tuple(
            "final-{}".format(key) for key in sorted(final_outputs)
        ),
        rank_count=plan.rank_count,
        slice_count=plan.slice_count,
        slice_size_bytes=slice_size_bytes,
        metadata={
            "path_scope": "global",
            "semantic_predecessors": {
                key: tuple(sorted(values))
                for key, values in semantic.items()
            },
            "resource_slots": resource_slots,
            "final_outputs": final_outputs,
            "final_dependencies": final_dependencies,
            "plan_nodes": tuple(node.node_id for node in plan.nodes),
        },
    )
    return provisional


def compose(
    plan: PlanDAG,
    candidates: Mapping[str, SolveCandidate],
) -> Schedule:
    if not isinstance(plan, PlanDAG):
        raise SemanticError("plan must be a PlanDAG")
    provisional = _compose_schedules(
        plan,
        _node_schedules(plan, candidates),
        routing_only=False,
    )
    return _retime(provisional, topology=None)


def compose_routes(
    plan: PlanDAG,
    node_schedules: Mapping[str, Schedule],
    topology: Topology,
    channel_count: int,
) -> Schedule:
    if not isinstance(plan, PlanDAG):
        raise SemanticError("plan must be a PlanDAG")
    provisional = _compose_schedules(
        plan,
        _route_node_schedules(plan, node_schedules),
        routing_only=True,
    )
    return assign_global_resources(
        provisional,
        topology,
        channel_count,
    )
