from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import PurePath
from typing import Any, Dict, Mapping, Tuple

from vericcl.composer.dual import reverse_allgather_schedule
from vericcl.composer.timing import _retime
from vericcl.errors import SemanticError
from vericcl.input.json_codec import canonical_json, sha256_json
from vericcl.planner.model import PlanDAG, PlanNode
from vericcl.semantics.atom import Atom, PathStage, Schedule, Symbol, Transfer
from vericcl.semantics.collective import OutputSlot
from vericcl.solver.model import SolveCandidate
from vericcl.topology.model import Topology


def _identity_json(value: object) -> Any:
    if isinstance(value, Enum):
        return {
            "enum": type(value).__qualname__,
            "value": _identity_json(value.value),
        }
    if value is None or isinstance(value, (bool, int, str, float)):
        return value
    if isinstance(value, PurePath):
        return {"path": str(value)}
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "dataclass": type(value).__qualname__,
            "fields": [
                [field.name, _identity_json(getattr(value, field.name))]
                for field in fields(value)
            ],
        }
    if isinstance(value, Mapping):
        entries = [
            [_identity_json(key), _identity_json(item)]
            for key, item in value.items()
        ]
        entries.sort(key=lambda entry: canonical_json(entry[0]))
        return {"mapping": entries}
    if isinstance(value, (frozenset, set)):
        items = [_identity_json(item) for item in value]
        items.sort(key=canonical_json)
        return {
            "set": type(value).__qualname__,
            "items": items,
        }
    if isinstance(value, (tuple, list)):
        return {
            "sequence": type(value).__qualname__,
            "items": [_identity_json(item) for item in value],
        }
    raise SemanticError(
        "node schedule identity contains unsupported type: {}".format(
            type(value).__qualname__
        )
    )


def route_node_schedule_identity(
    node_schedules: Mapping[str, Schedule],
) -> str:
    if not isinstance(node_schedules, Mapping):
        raise SemanticError("node_schedules must be a mapping")
    values = dict(node_schedules)
    if any(
        not isinstance(node_id, str)
        or not node_id
        or not isinstance(schedule, Schedule)
        for node_id, schedule in values.items()
    ):
        raise SemanticError("node_schedules contains invalid entries")
    return sha256_json(
        {
            "node_schedules": [
                [node_id, _identity_json(values[node_id])]
                for node_id in sorted(values)
            ]
        }
    )


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
        raise SemanticError("one route schedule is required for every plan node")
    by_id = {node.node_id: node for node in plan.nodes}
    schedules = {}
    for node_id in sorted(node_schedules):
        schedule = node_schedules[node_id]
        if not isinstance(schedule, Schedule):
            raise SemanticError(
                "node_schedules must contain Schedule values"
            )
        if (
            schedule.rank_count != plan.rank_count
            or schedule.slice_count != plan.slice_count
        ):
            raise SemanticError("node schedule dimensions do not match the plan")
        if schedule.metadata.get("routing_only") is not True:
            raise SemanticError("route schedule must be routing_only")
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
    if schedule.metadata.get("routing_only") is True:
        key = "r{:08d}-o{:08d}".format(slot.rank, slot.offset)
        final_outputs = schedule.metadata.get("final_outputs")
        final_dependencies = schedule.metadata.get("final_dependencies")
        if not isinstance(final_outputs, Mapping) or not isinstance(
            final_dependencies,
            Mapping,
        ):
            raise SemanticError(
                "routing-only schedule requires final dependency metadata"
            )
        if tuple(final_outputs.get(key, ())) != tuple(sorted(contributors)):
            raise SemanticError("routing-only final output metadata is inconsistent")
        by_id = {
            transfer.transfer_id: transfer
            for transfer in schedule.transfers
        }
        dependency_ids = tuple(final_dependencies.get(key, ()))
        if any(transfer_id not in by_id for transfer_id in dependency_ids):
            raise SemanticError("final dependency references a missing transfer")
        return tuple(by_id[transfer_id] for transfer_id in dependency_ids)
    matches = []
    for transfer in schedule.transfers:
        if (
            transfer.dst_rank == slot.rank
            and _tree_contributors(schedule, transfer) == contributors
        ):
            matches.append(transfer)
    return tuple(sorted(matches, key=lambda item: item.transfer_id))


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


def _output_path_transfers(
    schedule: Schedule,
    matches: Tuple[Transfer, ...],
) -> Tuple[Transfer, ...]:
    if schedule.metadata.get("routing_only") is not True or not matches:
        return matches
    by_id = {
        transfer.transfer_id: transfer for transfer in schedule.transfers
    }
    known_ids = set(by_id)
    pending = [transfer.transfer_id for transfer in matches]
    closure = set()
    while pending:
        transfer_id = pending.pop()
        if transfer_id in closure:
            continue
        closure.add(transfer_id)
        predecessors = _semantic_predecessors(
            schedule,
            by_id[transfer_id],
        )
        if not predecessors <= known_ids:
            raise SemanticError(
                "semantic predecessor references a missing transfer"
            )
        pending.extend(sorted(predecessors))
    match_ids = tuple(transfer.transfer_id for transfer in matches)
    remaining = sorted(closure - set(match_ids))
    ordered_ids = match_ids + tuple(remaining)
    return tuple(by_id[transfer_id] for transfer_id in ordered_ids)


def _compose_schedules(
    plan: PlanDAG,
    schedules: Mapping[str, Schedule],
) -> Schedule:
    slice_sizes = {
        schedule.slice_size_bytes for schedule in schedules.values()
    }
    if len(slice_sizes) != 1:
        raise SemanticError("node schedules must use one slice size")
    slice_size_bytes = next(iter(slice_sizes))
    path_hop_counts = tuple(
        schedule.metadata.get("instantiated_path_hop_count")
        for schedule in schedules.values()
    )
    if any(
        value is not None
        and (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        )
        for value in path_hop_counts
    ):
        raise SemanticError(
            "instantiated_path_hop_count metadata must be a non-negative integer"
        )
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
            duration = transfer.ed_time - transfer.st_time
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
                placeholder = _placeholder_atom(atom, prior_path, duration)
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
                channel=transfer.channel,
                stage_id=transfer.stage_id,
                member_slice_ids=transfer.member_slice_ids,
                atoms=rebuilt_atoms,
                st_time=start,
                ed_time=start + duration,
                predecessor_ids=predecessors,
            )
            slots = _resource_slots(schedule, transfer)
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
            path_matches = _output_path_transfers(schedule, matches)
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
                for transfer in path_matches:
                    atom = next(
                        (
                            value
                            for value in transfer.atoms
                            if value.slice_id == member
                        ),
                        None,
                    )
                    if atom is not None:
                        path = global_atoms[(transfer.transfer_id, member)].path
                        break
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
    metadata = {
        "path_scope": "global",
        "semantic_predecessors": {
            key: tuple(sorted(values))
            for key, values in semantic.items()
        },
        "resource_slots": resource_slots,
        "final_outputs": final_outputs,
        "final_dependencies": final_dependencies,
        "plan_nodes": tuple(node.node_id for node in plan.nodes),
    }
    if all(value is not None for value in path_hop_counts):
        metadata["instantiated_path_hop_count"] = sum(path_hop_counts)
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
        metadata=metadata,
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
    )
    return _retime(provisional, topology=None)


def compose_routes(
    plan: PlanDAG,
    node_schedules: Mapping[str, Schedule],
    topology: Topology,
    channel_count: int,
) -> Schedule:
    from vericcl.solver.global_scheduler import (
        _validate_channel_count,
        assign_global_resources,
    )

    if not isinstance(plan, PlanDAG):
        raise SemanticError("plan must be a PlanDAG")
    if not isinstance(topology, Topology):
        raise SemanticError("topology must be a Topology")
    if topology.rank_count != plan.rank_count:
        raise SemanticError("plan and topology rank counts must agree")
    channels = _validate_channel_count(channel_count)
    node_schedule_identity = route_node_schedule_identity(node_schedules)
    provisional = _compose_schedules(
        plan,
        _route_node_schedules(plan, node_schedules),
    )
    metadata = dict(provisional.metadata)
    metadata["route_node_schedule_identity"] = node_schedule_identity
    provisional = Schedule(
        schedule_id=provisional.schedule_id,
        transfers=provisional.transfers,
        final_state_ids=provisional.final_state_ids,
        rank_count=provisional.rank_count,
        slice_count=provisional.slice_count,
        slice_size_bytes=provisional.slice_size_bytes,
        metadata=metadata,
    )
    return assign_global_resources(
        provisional,
        topology,
        channels,
    )
