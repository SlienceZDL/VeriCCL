from typing import Dict, FrozenSet, Tuple

from vericcl.errors import InputValidationError
from vericcl.input.json_codec import sha256_json
from vericcl.input.models import ResolvedInput
from vericcl.input.validation import validate_collective
from vericcl.planner.groups import discover_communication_groups
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
from vericcl.topology.model import LinkKey, Topology


_DIRECT_KINDS = frozenset(
    {
        CollectiveKind.BROADCAST,
        CollectiveKind.REDUCE,
        CollectiveKind.ALL_GATHER,
        CollectiveKind.ALL_REDUCE,
        CollectiveKind.ALL_TO_ALL,
        CollectiveKind.REDUCE_SCATTER,
    }
)


def _initial_inputs(rank_count: int, slice_count: int) -> StageInterface:
    return StageInterface(
        {
            OutputSlot(rank, logical_address): frozenset(
                {rank * slice_count + logical_address}
            )
            for rank in range(rank_count)
            for logical_address in range(slice_count)
        }
    )


def _group_resources(
    topology: Topology,
    group: Tuple[int, ...],
) -> Tuple[FrozenSet[LinkKey], FrozenSet[str]]:
    group_set = set(group)
    links = frozenset(
        key
        for key in topology.links
        if key.src_rank in group_set and key.dst_rank in group_set
    )
    resources = frozenset(
        resource_id
        for key in links
        for resource_id in topology.resources_for(key)
    )
    return links, resources


def _spec(
    kind: CollectiveKind,
    source: CollectiveSpec,
    root: int = None,
) -> CollectiveSpec:
    reduced = kind in {
        CollectiveKind.REDUCE,
        CollectiveKind.ALL_REDUCE,
        CollectiveKind.REDUCE_SCATTER,
    }
    return CollectiveSpec(
        kind=kind,
        datatype=source.datatype,
        reduction_op=source.reduction_op if reduced else None,
        root=root,
        inplace=source.inplace,
    )


def _node(
    *,
    node_id: str,
    stage_id: int,
    collective: CollectiveSpec,
    group: Tuple[int, ...],
    logical_input: StageInterface,
    logical_output: StageInterface,
    links: FrozenSet[LinkKey],
    resources: FrozenSet[str],
    dual_of_node_id: str = None,
) -> PlanNode:
    return PlanNode(
        node_id=node_id,
        stage_id=stage_id,
        local_collective=collective,
        communication_group=group,
        logical_input=logical_input,
        logical_output=logical_output,
        allowed_links=links,
        shared_resource_ids=resources,
        dual_of_node_id=dual_of_node_id,
    )


def _broadcast_nodes(
    spec: CollectiveSpec,
    rank_count: int,
    slice_count: int,
    group: Tuple[int, ...],
    links: FrozenSet[LinkKey],
    resources: FrozenSet[str],
) -> Tuple[PlanNode, ...]:
    nodes = []
    for logical_address in range(slice_count):
        contributor = spec.root * slice_count + logical_address
        nodes.append(
            _node(
                node_id="broadcast-a{:08d}".format(logical_address),
                stage_id=0,
                collective=spec,
                group=group,
                logical_input=StageInterface(
                    {
                        OutputSlot(spec.root, logical_address): frozenset(
                            {contributor}
                        )
                    }
                ),
                logical_output=StageInterface(
                    {
                        OutputSlot(rank, logical_address): frozenset(
                            {contributor}
                        )
                        for rank in range(rank_count)
                    }
                ),
                links=links,
                resources=resources,
            )
        )
    return tuple(nodes)


def _allgather_nodes(
    spec: CollectiveSpec,
    rank_count: int,
    slice_count: int,
    group: Tuple[int, ...],
    links: FrozenSet[LinkKey],
    resources: FrozenSet[str],
) -> Tuple[PlanNode, ...]:
    nodes = []
    for source_rank in range(rank_count):
        broadcast = _spec(CollectiveKind.BROADCAST, spec, root=source_rank)
        for logical_address in range(slice_count):
            contributor = source_rank * slice_count + logical_address
            nodes.append(
                _node(
                    node_id="allgather-r{:08d}-a{:08d}".format(
                        source_rank,
                        logical_address,
                    ),
                    stage_id=0,
                    collective=broadcast,
                    group=group,
                    logical_input=StageInterface(
                        {
                            OutputSlot(
                                source_rank,
                                logical_address,
                            ): frozenset({contributor})
                        }
                    ),
                    logical_output=StageInterface(
                        {
                            OutputSlot(rank, contributor): frozenset(
                                {contributor}
                            )
                            for rank in range(rank_count)
                        }
                    ),
                    links=links,
                    resources=resources,
                )
            )
    return tuple(nodes)


def _reduce_nodes(
    spec: CollectiveSpec,
    rank_count: int,
    slice_count: int,
    group: Tuple[int, ...],
    links: FrozenSet[LinkKey],
    resources: FrozenSet[str],
) -> Tuple[PlanNode, ...]:
    nodes = []
    for logical_address in range(slice_count):
        node_id = "reduce-a{:08d}".format(logical_address)
        contributors = frozenset(
            rank * slice_count + logical_address
            for rank in range(rank_count)
        )
        nodes.append(
            _node(
                node_id=node_id,
                stage_id=0,
                collective=spec,
                group=group,
                logical_input=StageInterface(
                    {
                        OutputSlot(rank, logical_address): frozenset(
                            {rank * slice_count + logical_address}
                        )
                        for rank in range(rank_count)
                    }
                ),
                logical_output=StageInterface(
                    {OutputSlot(spec.root, logical_address): contributors}
                ),
                links=links,
                resources=resources,
                dual_of_node_id="dual-ag-{}".format(node_id),
            )
        )
    return tuple(nodes)


def _reduce_scatter_nodes(
    spec: CollectiveSpec,
    rank_count: int,
    slice_count: int,
    group: Tuple[int, ...],
    links: FrozenSet[LinkKey],
    resources: FrozenSet[str],
) -> Tuple[PlanNode, ...]:
    quotient = slice_count // rank_count
    nodes = []
    for logical_address in range(slice_count):
        owner = logical_address // quotient
        offset = logical_address % quotient
        node_id = "reduce-scatter-a{:08d}".format(logical_address)
        contributors = frozenset(
            rank * slice_count + logical_address
            for rank in range(rank_count)
        )
        nodes.append(
            _node(
                node_id=node_id,
                stage_id=0,
                collective=spec,
                group=group,
                logical_input=StageInterface(
                    {
                        OutputSlot(rank, logical_address): frozenset(
                            {rank * slice_count + logical_address}
                        )
                        for rank in range(rank_count)
                    }
                ),
                logical_output=StageInterface(
                    {OutputSlot(owner, offset): contributors}
                ),
                links=links,
                resources=resources,
                dual_of_node_id="dual-ag-{}".format(node_id),
            )
        )
    return tuple(nodes)


def _allreduce_plan(
    inputs: ResolvedInput,
    topology: Topology,
    initial_inputs: StageInterface,
) -> PlanDAG:
    rank_count = inputs.rank_count
    slice_count = inputs.hyperparameters.slice_count
    group = tuple(range(rank_count))
    links, resources = _group_resources(topology, group)
    quotient = slice_count // rank_count
    reduce_scatter = _spec(
        CollectiveKind.REDUCE_SCATTER,
        inputs.collective,
    )
    nodes = []
    edges = []
    gather_nodes = []
    for logical_address in range(slice_count):
        owner = logical_address // quotient
        offset = logical_address % quotient
        contributors = frozenset(
            rank * slice_count + logical_address
            for rank in range(rank_count)
        )
        intermediate = StageInterface(
            {OutputSlot(owner, offset): contributors}
        )
        reduce_id = "allreduce-rs-a{:08d}".format(logical_address)
        gather_id = "allreduce-ag-a{:08d}".format(logical_address)
        nodes.append(
            _node(
                node_id=reduce_id,
                stage_id=0,
                collective=reduce_scatter,
                group=group,
                logical_input=StageInterface(
                    {
                        OutputSlot(rank, logical_address): frozenset(
                            {rank * slice_count + logical_address}
                        )
                        for rank in range(rank_count)
                    }
                ),
                logical_output=intermediate,
                links=links,
                resources=resources,
                dual_of_node_id=gather_id,
            )
        )
        gather_nodes.append(
            _node(
                node_id=gather_id,
                stage_id=1,
                collective=_spec(
                    CollectiveKind.BROADCAST,
                    inputs.collective,
                    root=owner,
                ),
                group=group,
                logical_input=intermediate,
                logical_output=StageInterface(
                    {
                        OutputSlot(rank, logical_address): contributors
                        for rank in range(rank_count)
                    }
                ),
                links=links,
                resources=resources,
            )
        )
        edges.append(PlanEdge(reduce_id, gather_id, intermediate))
    nodes.extend(gather_nodes)
    return PlanDAG(
        collective=inputs.collective,
        rank_count=rank_count,
        slice_count=slice_count,
        initial_inputs=initial_inputs,
        nodes=tuple(nodes),
        edges=tuple(edges),
        final_outputs=StageInterface(
            required_outputs(inputs.collective, rank_count, slice_count)
        ),
        planning_mode=PlanningMode.DIRECT,
        planning_reason="direct_request",
    )


def _alltoall_nodes(
    spec: CollectiveSpec,
    rank_count: int,
    slice_count: int,
    group: Tuple[int, ...],
    links: FrozenSet[LinkKey],
    resources: FrozenSet[str],
) -> Tuple[PlanNode, ...]:
    quotient = slice_count // rank_count
    nodes = []
    for source_rank in range(rank_count):
        for logical_address in range(slice_count):
            destination = logical_address // quotient
            offset = source_rank * quotient + logical_address % quotient
            contributor = source_rank * slice_count + logical_address
            nodes.append(
                _node(
                    node_id="alltoall-r{:08d}-a{:08d}".format(
                        source_rank,
                        logical_address,
                    ),
                    stage_id=0,
                    collective=spec,
                    group=group,
                    logical_input=StageInterface(
                        {
                            OutputSlot(
                                source_rank,
                                logical_address,
                            ): frozenset({contributor})
                        }
                    ),
                    logical_output=StageInterface(
                        {
                            OutputSlot(destination, offset): frozenset(
                                {contributor}
                            )
                        }
                    ),
                    links=links,
                    resources=resources,
                )
            )
    return tuple(nodes)


def build_direct_plan(inputs: ResolvedInput, topology: Topology) -> PlanDAG:
    if not isinstance(inputs, ResolvedInput):
        raise InputValidationError("inputs must be a ResolvedInput")
    if not isinstance(topology, Topology):
        raise InputValidationError("topology must be a Topology")
    if inputs.rank_count != topology.rank_count:
        raise InputValidationError("input and topology rank counts do not match")
    if inputs.collective.kind not in _DIRECT_KINDS:
        raise InputValidationError(
            "{} is only available as an internal plan operator".format(
                inputs.collective.kind.value
            )
        )
    validate_collective(
        inputs.collective,
        inputs.rank_count,
        inputs.hyperparameters.slice_count,
    )
    if (
        inputs.collective.kind is CollectiveKind.ALL_REDUCE
        and inputs.strategies.hierarchy
        and len(set(topology.node_membership.values())) > 1
        and discover_communication_groups(topology).inter_node
    ):
        raise InputValidationError(
            "hierarchical AllReduce must use the gateway plan builder"
        )
    rank_count = inputs.rank_count
    slice_count = inputs.hyperparameters.slice_count
    initial_inputs = _initial_inputs(rank_count, slice_count)
    if inputs.collective.kind is CollectiveKind.ALL_REDUCE:
        return _allreduce_plan(inputs, topology, initial_inputs)
    group = tuple(range(rank_count))
    links, resources = _group_resources(topology, group)
    builders = {
        CollectiveKind.BROADCAST: _broadcast_nodes,
        CollectiveKind.REDUCE: _reduce_nodes,
        CollectiveKind.ALL_GATHER: _allgather_nodes,
        CollectiveKind.ALL_TO_ALL: _alltoall_nodes,
        CollectiveKind.REDUCE_SCATTER: _reduce_scatter_nodes,
    }
    nodes = builders[inputs.collective.kind](
        inputs.collective,
        rank_count,
        slice_count,
        group,
        links,
        resources,
    )
    return PlanDAG(
        collective=inputs.collective,
        rank_count=rank_count,
        slice_count=slice_count,
        initial_inputs=initial_inputs,
        nodes=nodes,
        edges=(),
        final_outputs=StageInterface(
            required_outputs(inputs.collective, rank_count, slice_count)
        ),
        planning_mode=PlanningMode.DIRECT,
        planning_reason="direct_request",
    )


def _internal_values_token(values: StageInterface) -> str:
    return sha256_json(
        [
            {
                "rank": value.slot.rank,
                "offset": value.slot.offset,
                "contributors": sorted(value.contributors),
            }
            for value in values.logical_values
        ]
    )[:12]


def _internal_builder_inputs(
    root: int,
    group: Tuple[int, ...],
    values: StageInterface,
    topology: Topology,
    kind: CollectiveKind,
) -> Tuple[Tuple[int, ...], FrozenSet[LinkKey], FrozenSet[str], StageInterface]:
    if not isinstance(topology, Topology):
        raise InputValidationError("topology must be a Topology")
    if not isinstance(values, StageInterface):
        raise InputValidationError("values must be a StageInterface")
    try:
        group = tuple(group)
    except TypeError as error:
        raise InputValidationError("group must be iterable") from error
    if any(
        isinstance(rank, bool) or not isinstance(rank, int)
        for rank in group
    ):
        raise InputValidationError("group ranks must be integers")
    if not group or group != tuple(sorted(group)) or len(group) != len(set(group)):
        raise InputValidationError("group must be non-empty, sorted, and unique")
    if isinstance(root, bool) or not isinstance(root, int):
        raise InputValidationError("root must be an integer")
    if root not in group:
        raise InputValidationError("root must belong to the communication group")
    if any(rank < 0 or rank >= topology.rank_count for rank in group):
        raise InputValidationError("group rank is outside the topology")
    if kind is CollectiveKind.SCATTER:
        if any(slot.rank not in group for slot in values.values):
            raise InputValidationError("scatter output rank is outside the group")
        logical_input = StageInterface(
            {
                OutputSlot(root, index): value.contributors
                for index, value in enumerate(values.logical_values)
            }
        )
    else:
        if any(slot.rank != root for slot in values.values):
            raise InputValidationError("gather output values must be at the root")
        logical_input = StageInterface(
            {
                OutputSlot(
                    group[index % len(group)],
                    index // len(group),
                ): value.contributors
                for index, value in enumerate(values.logical_values)
            }
        )
    links, resources = _group_resources(topology, group)
    return group, links, resources, logical_input


def build_internal_scatter(
    root: int,
    group: Tuple[int, ...],
    values: StageInterface,
    topology: Topology,
) -> Tuple[PlanNode, ...]:
    group, links, resources, logical_input = _internal_builder_inputs(
        root,
        group,
        values,
        topology,
        CollectiveKind.SCATTER,
    )
    return (
        _node(
            node_id="internal-scatter-r{}-{}".format(
                root,
                _internal_values_token(values),
            ),
            stage_id=0,
            collective=CollectiveSpec(
                kind=CollectiveKind.SCATTER,
                datatype="opaque",
                root=root,
            ),
            group=group,
            logical_input=logical_input,
            logical_output=values,
            links=links,
            resources=resources,
        ),
    )


def build_internal_gather(
    root: int,
    group: Tuple[int, ...],
    values: StageInterface,
    topology: Topology,
) -> Tuple[PlanNode, ...]:
    group, links, resources, logical_input = _internal_builder_inputs(
        root,
        group,
        values,
        topology,
        CollectiveKind.GATHER,
    )
    return (
        _node(
            node_id="internal-gather-r{}-{}".format(
                root,
                _internal_values_token(values),
            ),
            stage_id=0,
            collective=CollectiveSpec(
                kind=CollectiveKind.GATHER,
                datatype="opaque",
                root=root,
            ),
            group=group,
            logical_input=logical_input,
            logical_output=values,
            links=links,
            resources=resources,
        ),
    )
