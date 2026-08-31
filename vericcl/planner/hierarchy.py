from collections import deque
from typing import Dict, Mapping, Sequence, Tuple

from vericcl.errors import InputValidationError, SemanticError
from vericcl.input.models import ResolvedInput
from vericcl.planner.groups import (
    CommunicationGroups,
    eligible_gateway_group,
    eligible_gateway_groups,
)
from vericcl.planner.model import (
    PlanDAG,
    PlanEdge,
    PlanNode,
    PlanningMode,
    StageInterface,
)
from vericcl.semantics.collective import (
    CollectiveKind,
    CollectiveSpec,
    OutputSlot,
    required_outputs,
)
from vericcl.topology.model import Topology


_MANUAL_NODE_KEYS = frozenset(
    {
        "node_id",
        "stage_id",
        "operator",
        "communication_group",
        "root",
        "logical_input",
        "logical_output",
        "depends_on",
    }
)
_ROOTED_KINDS = frozenset(
    {
        CollectiveKind.BROADCAST,
        CollectiveKind.REDUCE,
        CollectiveKind.SCATTER,
        CollectiveKind.GATHER,
    }
)
_REDUCTION_KINDS = frozenset(
    {
        CollectiveKind.REDUCE,
        CollectiveKind.ALL_REDUCE,
        CollectiveKind.REDUCE_SCATTER,
    }
)
_OPERATOR_ALIASES = {
    "all_gather": "allgather",
    "all_reduce": "allreduce",
    "all_to_all": "alltoall",
    "reducescatter": "reduce_scatter",
}


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise InputValidationError("{} must be a mapping".format(field))
    return value


def _sequence(value: object, field: str) -> tuple:
    if not isinstance(value, (tuple, list)):
        raise InputValidationError("{} must be a sequence".format(field))
    return tuple(value)


def _integer(value: object, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InputValidationError("{} must be an integer".format(field))
    if value < minimum:
        raise InputValidationError("{} must be at least {}".format(field, minimum))
    return value


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise InputValidationError("{} must be a non-empty string".format(field))
    return value


def _initial_inputs(rank_count: int, slice_count: int) -> StageInterface:
    return StageInterface(
        {
            OutputSlot(rank, address): frozenset(
                {rank * slice_count + address}
            )
            for rank in range(rank_count)
            for address in range(slice_count)
        }
    )


def _group_resources(
    topology: Topology,
    group: Tuple[int, ...],
) -> Tuple[frozenset, frozenset]:
    ranks = set(group)
    links = frozenset(
        key
        for key in topology.links
        if key.src_rank in ranks and key.dst_rank in ranks
    )
    resources = frozenset(
        resource_id
        for key in links
        for resource_id in topology.resources_for(key)
    )
    return links, resources


def _contributors(
    ranks: Sequence[int],
    logical_address: int,
    slice_count: int,
) -> frozenset:
    return frozenset(
        rank * slice_count + logical_address for rank in ranks
    )


def _local_spec(
    kind: CollectiveKind,
    collective: CollectiveSpec,
    root: int = None,
) -> CollectiveSpec:
    return CollectiveSpec(
        kind=kind,
        datatype=collective.datatype,
        reduction_op=(
            collective.reduction_op if kind in _REDUCTION_KINDS else None
        ),
        root=root,
        inplace=collective.inplace,
    )


def _plan_node(
    *,
    node_id: str,
    stage_id: int,
    local_collective: CollectiveSpec,
    group: Tuple[int, ...],
    logical_input: StageInterface,
    logical_output: StageInterface,
    topology: Topology,
    dual_of_node_id: str = None,
) -> PlanNode:
    links, resources = _group_resources(topology, group)
    return PlanNode(
        node_id=node_id,
        stage_id=stage_id,
        local_collective=local_collective,
        communication_group=group,
        logical_input=logical_input,
        logical_output=logical_output,
        allowed_links=links,
        shared_resource_ids=resources,
        dual_of_node_id=dual_of_node_id,
    )


def _reachable(
    topology: Topology,
    group: Tuple[int, ...],
    source: int,
) -> frozenset:
    group_set = set(group)
    reached = {source}
    pending = deque([source])
    while pending:
        rank = pending.popleft()
        for destination in topology.destinations(rank):
            if destination in group_set and destination not in reached:
                reached.add(destination)
                pending.append(destination)
    return frozenset(reached)


def _validate_connected_domain(
    topology: Topology,
    group: Tuple[int, ...],
    kind: CollectiveKind,
    root: int,
) -> None:
    group_set = frozenset(group)
    if len(group) == 1:
        return
    if kind in {CollectiveKind.BROADCAST, CollectiveKind.SCATTER}:
        sources = (root,)
        destinations = group_set
    elif kind in {CollectiveKind.REDUCE, CollectiveKind.GATHER}:
        sources = group
        destinations = frozenset({root})
    else:
        sources = group
        destinations = group_set
    for source in sources:
        reached = _reachable(topology, group, source)
        if not destinations.issubset(reached):
            raise InputValidationError(
                "manual communication group is not connected by allowed links"
            )


def _manual_group(value: object, topology: Topology, field: str) -> Tuple[int, ...]:
    group = tuple(
        _integer(rank, field + ".rank")
        for rank in _sequence(value, field)
    )
    if not group:
        raise InputValidationError("{} must not be empty".format(field))
    if group != tuple(sorted(group)) or len(group) != len(set(group)):
        raise InputValidationError("{} must be sorted and unique".format(field))
    if any(rank >= topology.rank_count for rank in group):
        raise InputValidationError("{} rank is outside the topology".format(field))
    return group


def _manual_interface(
    value: object,
    group: Tuple[int, ...],
    field: str,
) -> StageInterface:
    entries = _sequence(value, field)
    values = {}
    for index, raw_entry in enumerate(entries):
        entry_field = "{}[{}]".format(field, index)
        entry = _sequence(raw_entry, entry_field)
        if len(entry) != 3:
            raise InputValidationError(
                "{} must contain rank, offset, and contributors".format(
                    entry_field
                )
            )
        rank = _integer(entry[0], entry_field + ".rank")
        offset = _integer(entry[1], entry_field + ".offset")
        if rank not in group:
            raise InputValidationError(
                "{} rank is outside the communication group".format(entry_field)
            )
        contributors = frozenset(
            _integer(item, entry_field + ".contributors")
            for item in _sequence(entry[2], entry_field + ".contributors")
        )
        if not contributors:
            raise InputValidationError(
                "{} contributors must not be empty".format(entry_field)
            )
        slot = OutputSlot(rank, offset)
        if slot in values:
            raise InputValidationError("{} contains a duplicate slot".format(field))
        values[slot] = contributors
    try:
        return StageInterface(values)
    except SemanticError as error:
        raise InputValidationError(str(error)) from error


def _manual_spec(
    raw: Mapping[str, object],
    group: Tuple[int, ...],
    collective: CollectiveSpec,
    field: str,
) -> CollectiveSpec:
    operator = _identifier(raw.get("operator"), field + ".operator").lower()
    operator = _OPERATOR_ALIASES.get(operator, operator)
    try:
        kind = CollectiveKind(operator)
    except ValueError as error:
        raise InputValidationError(
            "unsupported manual operator: {}".format(operator)
        ) from error
    root = raw.get("root")
    if kind in _ROOTED_KINDS:
        root = _integer(root, field + ".root")
        if root not in group:
            raise InputValidationError("manual root must belong to its group")
    elif root is not None:
        raise InputValidationError(
            "{} must not define root".format(kind.value)
        )
    if kind in _REDUCTION_KINDS and collective.reduction_op is None:
        raise InputValidationError(
            "manual reduction requires a global reduction operation"
        )
    return _local_spec(kind, collective, root=root)


def _parse_manual_nodes(
    plan_spec: object,
    topology: Topology,
    collective: CollectiveSpec,
) -> Tuple[Tuple[PlanNode, ...], Tuple[Tuple[str, Tuple[str, ...]], ...]]:
    raw_nodes = _sequence(plan_spec, "manual hierarchy")
    if not raw_nodes:
        raise InputValidationError("manual hierarchy must not be empty")
    nodes = []
    dependencies = []
    seen_ids = set()
    for index, raw_node in enumerate(raw_nodes):
        field = "manual hierarchy[{}]".format(index)
        raw = _mapping(raw_node, field)
        unknown = sorted(set(raw) - _MANUAL_NODE_KEYS)
        if unknown:
            raise InputValidationError(
                "unknown manual hierarchy field: {}".format(unknown[0])
            )
        node_id = _identifier(raw.get("node_id"), field + ".node_id")
        if node_id in seen_ids:
            raise InputValidationError("manual node IDs must be unique")
        seen_ids.add(node_id)
        stage_id = _integer(raw.get("stage_id"), field + ".stage_id")
        group = _manual_group(
            raw.get("communication_group"),
            topology,
            field + ".communication_group",
        )
        local_collective = _manual_spec(raw, group, collective, field)
        _validate_connected_domain(
            topology,
            group,
            local_collective.kind,
            local_collective.root,
        )
        logical_input = _manual_interface(
            raw.get("logical_input"),
            group,
            field + ".logical_input",
        )
        logical_output = _manual_interface(
            raw.get("logical_output"),
            group,
            field + ".logical_output",
        )
        depends_on = tuple(
            _identifier(item, field + ".depends_on")
            for item in _sequence(raw.get("depends_on", ()), field + ".depends_on")
        )
        if len(depends_on) != len(set(depends_on)):
            raise InputValidationError("manual dependencies must be unique")
        nodes.append(
            _plan_node(
                node_id=node_id,
                stage_id=stage_id,
                local_collective=local_collective,
                group=group,
                logical_input=logical_input,
                logical_output=logical_output,
                topology=topology,
            )
        )
        dependencies.append((node_id, depends_on))
    return tuple(nodes), tuple(dependencies)


def _manual_edges(
    nodes: Tuple[PlanNode, ...],
    dependencies: Tuple[Tuple[str, Tuple[str, ...]], ...],
) -> Tuple[PlanEdge, ...]:
    by_id = {node.node_id: node for node in nodes}
    edges = []
    for consumer_id, producer_ids in dependencies:
        consumer = by_id[consumer_id]
        for producer_id in producer_ids:
            if producer_id not in by_id:
                raise InputValidationError(
                    "manual dependency references an unknown node"
                )
            if producer_id == consumer_id:
                raise InputValidationError("manual node cannot depend on itself")
            producer = by_id[producer_id]
            overlapping = set(producer.logical_output.values) & set(
                consumer.logical_input.values
            )
            for slot in overlapping:
                if (
                    producer.logical_output.values[slot]
                    != consumer.logical_input.values[slot]
                ):
                    raise InputValidationError(
                        "adjacent manual stages have mismatched contributors"
                    )
            matched = {
                slot: contributors
                for slot, contributors in producer.logical_output.values.items()
                if consumer.logical_input.values.get(slot) == contributors
            }
            if not matched:
                raise InputValidationError(
                    "manual dependency interfaces do not match"
                )
            edges.append(
                PlanEdge(
                    producer_id,
                    consumer_id,
                    StageInterface(matched),
                )
            )
    return tuple(edges)


def _validate_dependency_acyclic(
    nodes: Tuple[PlanNode, ...],
    edges: Tuple[PlanEdge, ...],
) -> None:
    successors: Dict[str, set] = {node.node_id: set() for node in nodes}
    indegree = {node.node_id: 0 for node in nodes}
    for edge in edges:
        if edge.consumer_id not in successors[edge.producer_id]:
            successors[edge.producer_id].add(edge.consumer_id)
            indegree[edge.consumer_id] += 1
    pending = deque(sorted(node_id for node_id, value in indegree.items() if value == 0))
    count = 0
    while pending:
        node_id = pending.popleft()
        count += 1
        for successor in sorted(successors[node_id]):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                pending.append(successor)
    if count != len(nodes):
        raise InputValidationError("manual hierarchy must be acyclic")


def validate_manual_hierarchy(
    plan_spec: object,
    topology: Topology,
    collective: CollectiveSpec,
) -> None:
    if not isinstance(topology, Topology):
        raise InputValidationError("topology must be a Topology")
    if not isinstance(collective, CollectiveSpec):
        raise InputValidationError("collective must be a CollectiveSpec")
    if isinstance(plan_spec, PlanDAG):
        if plan_spec.collective != collective:
            raise InputValidationError(
                "manual PlanDAG collective does not match the request"
            )
        if plan_spec.rank_count != topology.rank_count:
            raise InputValidationError(
                "manual PlanDAG rank count does not match the topology"
            )
        for node in plan_spec.nodes:
            if any(key not in topology.links for key in node.allowed_links):
                raise InputValidationError(
                    "manual PlanDAG contains a nonexistent communication link"
                )
        return
    nodes, dependencies = _parse_manual_nodes(
        plan_spec,
        topology,
        collective,
    )
    edges = _manual_edges(nodes, dependencies)
    _validate_dependency_acyclic(nodes, edges)


def build_manual_plan(inputs: ResolvedInput, topology: Topology) -> PlanDAG:
    validate_manual_hierarchy(
        inputs.strategies.manual_hierarchy,
        topology,
        inputs.collective,
    )
    nodes, dependencies = _parse_manual_nodes(
        inputs.strategies.manual_hierarchy,
        topology,
        inputs.collective,
    )
    edges = _manual_edges(nodes, dependencies)
    slice_count = inputs.hyperparameters.slice_count
    return PlanDAG(
        collective=inputs.collective,
        rank_count=inputs.rank_count,
        slice_count=slice_count,
        initial_inputs=_initial_inputs(inputs.rank_count, slice_count),
        nodes=nodes,
        edges=edges,
        final_outputs=StageInterface(
            required_outputs(
                inputs.collective,
                inputs.rank_count,
                slice_count,
            )
        ),
        planning_mode=PlanningMode.MANUAL,
        planning_reason="manual_hierarchy",
    )


def build_gateway_allreduce_plan(
    inputs: ResolvedInput,
    topology: Topology,
    groups: CommunicationGroups,
) -> PlanDAG:
    if inputs.collective.kind is not CollectiveKind.ALL_REDUCE:
        raise InputValidationError("gateway template requires AllReduce")
    gateway_group = eligible_gateway_group(topology, groups)
    if gateway_group is None:
        raise InputValidationError(
            "no real gateway communication group covers every node"
        )
    slice_count = inputs.hyperparameters.slice_count
    if slice_count % len(gateway_group) != 0:
        raise InputValidationError(
            "slice count must be divisible by the gateway group size"
        )
    by_node = {
        topology.node_membership[group[0]]: group
        for group in groups.intra_node
    }
    gateways_by_node = {
        topology.node_membership[gateway]: gateway
        for gateway in gateway_group
    }
    if set(by_node) != set(gateways_by_node):
        raise InputValidationError("gateway group does not cover every local group")

    local_reduce_nodes = []
    local_outputs = {}
    for node_id, group in sorted(by_node.items(), key=lambda item: item[1]):
        gateway = gateways_by_node[node_id]
        logical_input = StageInterface(
            {
                OutputSlot(rank, address): frozenset(
                    {rank * slice_count + address}
                )
                for rank in group
                for address in range(slice_count)
            }
        )
        logical_output = StageInterface(
            {
                OutputSlot(gateway, address): _contributors(
                    group,
                    address,
                    slice_count,
                )
                for address in range(slice_count)
            }
        )
        local_outputs[node_id] = logical_output
        node_name = "local-reduce-node-{}".format(node_id)
        local_reduce_nodes.append(
            _plan_node(
                node_id=node_name,
                stage_id=0,
                local_collective=_local_spec(
                    CollectiveKind.REDUCE,
                    inputs.collective,
                    root=gateway,
                ),
                group=group,
                logical_input=logical_input,
                logical_output=logical_output,
                topology=topology,
                dual_of_node_id="dual-ag-{}".format(node_name),
            )
        )

    inter_input_values = {}
    for interface in local_outputs.values():
        inter_input_values.update(interface.values)
    inter_input = StageInterface(inter_input_values)
    quotient = slice_count // len(gateway_group)
    inter_output = StageInterface(
        {
            OutputSlot(
                gateway_group[address // quotient],
                address % quotient,
            ): _contributors(
                range(inputs.rank_count),
                address,
                slice_count,
            )
            for address in range(slice_count)
        }
    )
    inter_reduce_scatter = _plan_node(
        node_id="gateway-reduce-scatter",
        stage_id=1,
        local_collective=_local_spec(
            CollectiveKind.REDUCE_SCATTER,
            inputs.collective,
        ),
        group=gateway_group,
        logical_input=inter_input,
        logical_output=inter_output,
        topology=topology,
        dual_of_node_id="gateway-allgather",
    )
    gateway_output = StageInterface(
        {
            OutputSlot(gateway, address): _contributors(
                range(inputs.rank_count),
                address,
                slice_count,
            )
            for gateway in gateway_group
            for address in range(slice_count)
        }
    )
    gateway_allgather = _plan_node(
        node_id="gateway-allgather",
        stage_id=2,
        local_collective=_local_spec(
            CollectiveKind.ALL_GATHER,
            inputs.collective,
        ),
        group=gateway_group,
        logical_input=inter_output,
        logical_output=gateway_output,
        topology=topology,
    )

    local_allgather_nodes = []
    local_inputs = {}
    for node_id, group in sorted(by_node.items(), key=lambda item: item[1]):
        gateway = gateways_by_node[node_id]
        logical_input = StageInterface(
            {
                OutputSlot(gateway, address): _contributors(
                    range(inputs.rank_count),
                    address,
                    slice_count,
                )
                for address in range(slice_count)
            }
        )
        logical_output = StageInterface(
            {
                OutputSlot(rank, address): _contributors(
                    range(inputs.rank_count),
                    address,
                    slice_count,
                )
                for rank in group
                for address in range(slice_count)
            }
        )
        local_inputs[node_id] = logical_input
        local_allgather_nodes.append(
            _plan_node(
                node_id="local-allgather-node-{}".format(node_id),
                stage_id=3,
                local_collective=_local_spec(
                    CollectiveKind.ALL_GATHER,
                    inputs.collective,
                ),
                group=group,
                logical_input=logical_input,
                logical_output=logical_output,
                topology=topology,
            )
        )

    edges = []
    for node in local_reduce_nodes:
        node_id = topology.node_membership[node.communication_group[0]]
        edges.append(
            PlanEdge(
                node.node_id,
                inter_reduce_scatter.node_id,
                local_outputs[node_id],
            )
        )
    edges.append(
        PlanEdge(
            inter_reduce_scatter.node_id,
            gateway_allgather.node_id,
            inter_output,
        )
    )
    for node in local_allgather_nodes:
        node_id = topology.node_membership[node.communication_group[0]]
        edges.append(
            PlanEdge(
                gateway_allgather.node_id,
                node.node_id,
                local_inputs[node_id],
            )
        )
    nodes = (
        tuple(local_reduce_nodes)
        + (inter_reduce_scatter, gateway_allgather)
        + tuple(local_allgather_nodes)
    )
    return PlanDAG(
        collective=inputs.collective,
        rank_count=inputs.rank_count,
        slice_count=slice_count,
        initial_inputs=_initial_inputs(inputs.rank_count, slice_count),
        nodes=nodes,
        edges=tuple(edges),
        final_outputs=StageInterface(
            required_outputs(
                inputs.collective,
                inputs.rank_count,
                slice_count,
            )
        ),
        planning_mode=PlanningMode.GATEWAY_ALLREDUCE,
        planning_reason="eligible_gateway_domain",
    )


def build_gateway_allgather_plan(
    inputs: ResolvedInput,
    topology: Topology,
    groups: CommunicationGroups,
) -> PlanDAG:
    if inputs.collective.kind is not CollectiveKind.ALL_GATHER:
        raise InputValidationError("gateway template requires AllGather")
    gateway_groups = eligible_gateway_groups(topology, groups)
    if not gateway_groups:
        raise InputValidationError(
            "no real gateway communication group covers every node"
        )

    slice_count = inputs.hyperparameters.slice_count
    rail_count = len(gateway_groups)
    by_node = {
        topology.node_membership[group[0]]: group
        for group in groups.intra_node
    }
    gateways_by_rail = tuple(
        {
            topology.node_membership[gateway]: gateway
            for gateway in gateway_group
        }
        for gateway_group in gateway_groups
    )
    if any(set(by_node) != set(gateways) for gateways in gateways_by_rail):
        raise InputValidationError("gateway groups do not cover every local group")

    local_gather_nodes = []
    gateway_nodes = []
    local_allgather_nodes = []
    rail_inputs = [dict() for _ in gateway_groups]
    edges = []
    global_slice_ids = tuple(range(inputs.rank_count * slice_count))

    for rail_index, gateways in enumerate(gateways_by_rail):
        for node_id, group in sorted(by_node.items(), key=lambda item: item[1]):
            gateway = gateways[node_id]
            input_values = {
                OutputSlot(rank, address): frozenset({slice_id})
                for rank in group
                for address in range(slice_count)
                for slice_id in (rank * slice_count + address,)
                if slice_id % rail_count == rail_index
            }
            if not input_values:
                continue
            logical_input = StageInterface(input_values)
            logical_output = StageInterface(
                {
                    OutputSlot(gateway, slice_id): contributors
                    for contributors in logical_input.values.values()
                    for slice_id in contributors
                }
            )
            node_name = "local-gather-node-{}-rail-{}".format(
                node_id,
                rail_index,
            )
            rail_inputs[rail_index].update(logical_output.values)
            local_gather_nodes.append(
                _plan_node(
                    node_id=node_name,
                    stage_id=0,
                    local_collective=_local_spec(
                        CollectiveKind.GATHER,
                        inputs.collective,
                        root=gateway,
                    ),
                    group=group,
                    logical_input=logical_input,
                    logical_output=logical_output,
                    topology=topology,
                )
            )
            edges.append(
                PlanEdge(
                    node_name,
                    "gateway-allgather-rail-{}".format(rail_index),
                    logical_output,
                )
            )

    for rail_index, gateway_group in enumerate(gateway_groups):
        inter_input = StageInterface(rail_inputs[rail_index])
        rail_slices = tuple(
            slice_id
            for slice_id in global_slice_ids
            if slice_id % rail_count == rail_index
        )
        inter_output = StageInterface(
            {
                OutputSlot(gateway, slice_id): frozenset({slice_id})
                for gateway in gateway_group
                for slice_id in rail_slices
            }
        )
        gateway_nodes.append(
            _plan_node(
                node_id="gateway-allgather-rail-{}".format(rail_index),
                stage_id=1,
                local_collective=_local_spec(
                    CollectiveKind.ALL_GATHER,
                    inputs.collective,
                ),
                group=gateway_group,
                logical_input=inter_input,
                logical_output=inter_output,
                topology=topology,
            )
        )

    for rail_index, gateways in enumerate(gateways_by_rail):
        rail_slices = tuple(
            slice_id
            for slice_id in global_slice_ids
            if slice_id % rail_count == rail_index
        )
        for node_id, group in sorted(by_node.items(), key=lambda item: item[1]):
            gateway = gateways[node_id]
            logical_input = StageInterface(
                {
                    OutputSlot(gateway, slice_id): frozenset({slice_id})
                    for slice_id in rail_slices
                }
            )
            logical_output = StageInterface(
                {
                    OutputSlot(rank, slice_id): frozenset({slice_id})
                    for rank in group
                    for slice_id in rail_slices
                }
            )
            local_allgather_nodes.append(
                _plan_node(
                    node_id="local-allgather-node-{}-rail-{}".format(
                        node_id,
                        rail_index,
                    ),
                    stage_id=2,
                    local_collective=_local_spec(
                        CollectiveKind.ALL_GATHER,
                        inputs.collective,
                    ),
                    group=group,
                    logical_input=logical_input,
                    logical_output=logical_output,
                    topology=topology,
                )
            )
            edges.append(
                PlanEdge(
                    "gateway-allgather-rail-{}".format(rail_index),
                    "local-allgather-node-{}-rail-{}".format(
                        node_id,
                        rail_index,
                    ),
                    logical_input,
                )
            )

    return PlanDAG(
        collective=inputs.collective,
        rank_count=inputs.rank_count,
        slice_count=slice_count,
        initial_inputs=_initial_inputs(inputs.rank_count, slice_count),
        nodes=(
            tuple(local_gather_nodes)
            + tuple(gateway_nodes)
            + tuple(local_allgather_nodes)
        ),
        edges=tuple(edges),
        final_outputs=StageInterface(
            required_outputs(
                inputs.collective,
                inputs.rank_count,
                slice_count,
            )
        ),
        planning_mode=PlanningMode.GATEWAY_ALLGATHER,
        planning_reason="eligible_gateway_domain",
    )
