from dataclasses import dataclass
from typing import FrozenSet, Tuple

from vericcl.errors import SemanticError
from vericcl.input.models import ForbiddenTransfer, ResolvedInput
from vericcl.planner.model import PlanNode
from vericcl.semantics.collective import CollectiveKind
from vericcl.semantics.slice import logical_slice_index
from vericcl.solver.pruning import (
    ranked_simple_paths,
    retain_shortest_paths,
    viable_path_links,
)
from vericcl.topology.model import LinkKey, Topology


_REDUCTION_KINDS = frozenset(
    {
        CollectiveKind.REDUCE,
        CollectiveKind.REDUCE_SCATTER,
    }
)


RoutingUnitKey = Tuple[int, int, Tuple[int, ...], bool]


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise SemanticError("{} must be a non-empty string".format(field))
    return value


def _integer(value: object, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SemanticError("{} must be an integer".format(field))
    if value < minimum:
        raise SemanticError("{} must be at least {}".format(field, minimum))
    return value


@dataclass(frozen=True, order=True)
class CandidateEdge:
    src_rank: int
    dst_rank: int
    channel: int

    def __post_init__(self) -> None:
        _integer(self.src_rank, "candidate_edge.src_rank")
        _integer(self.dst_rank, "candidate_edge.dst_rank")
        if self.src_rank == self.dst_rank:
            raise SemanticError("candidate edge ranks must be distinct")
        _integer(self.channel, "candidate_edge.channel")


@dataclass(frozen=True)
class TransferDemand:
    demand_id: str
    node_id: str
    stage_id: int
    root_rank: int
    required_leaf_rank: int
    logical_position: int
    contributors: FrozenSet[int]
    member_slice_ids: FrozenSet[int]
    allowed_links: FrozenSet[LinkKey]
    legal_links: FrozenSet[LinkKey]
    forbidden_members: Tuple[ForbiddenTransfer, ...]
    candidate_paths: Tuple[Tuple[int, ...], ...]
    reduction_dual: bool = False

    def __post_init__(self) -> None:
        _identifier(self.demand_id, "transfer_demand.demand_id")
        _identifier(self.node_id, "transfer_demand.node_id")
        _integer(self.stage_id, "transfer_demand.stage_id")
        _integer(self.root_rank, "transfer_demand.root_rank")
        _integer(
            self.required_leaf_rank,
            "transfer_demand.required_leaf_rank",
        )
        if self.root_rank == self.required_leaf_rank:
            raise SemanticError("transfer demand must not be a self transfer")
        _integer(self.logical_position, "transfer_demand.logical_position")
        contributors = frozenset(self.contributors)
        members = frozenset(self.member_slice_ids)
        if not contributors or not members or not members <= contributors:
            raise SemanticError(
                "transfer demand members must be non-empty contributors"
            )
        object.__setattr__(self, "contributors", contributors)
        object.__setattr__(self, "member_slice_ids", members)
        links = frozenset(self.allowed_links)
        if not all(isinstance(link, LinkKey) for link in links):
            raise SemanticError(
                "transfer_demand.allowed_links must contain LinkKey values"
            )
        object.__setattr__(self, "allowed_links", links)
        legal_links = frozenset(self.legal_links)
        if not legal_links <= links:
            raise SemanticError(
                "transfer_demand.legal_links must be allowed links"
            )
        object.__setattr__(self, "legal_links", legal_links)
        forbidden = tuple(self.forbidden_members)
        if not all(isinstance(item, ForbiddenTransfer) for item in forbidden):
            raise SemanticError(
                "transfer_demand.forbidden_members must contain ForbiddenTransfer values"
            )
        object.__setattr__(self, "forbidden_members", forbidden)
        paths = tuple(sorted(set(tuple(path) for path in self.candidate_paths)))
        for path in paths:
            if (
                len(path) < 2
                or path[0] != self.root_rank
                or path[-1] != self.required_leaf_rank
                or len(path) != len(set(path))
            ):
                raise SemanticError("transfer demand contains an invalid path")
            if any(
                LinkKey(src, dst) not in legal_links
                for src, dst in zip(path, path[1:])
            ):
                raise SemanticError("transfer demand path uses a disallowed link")
        object.__setattr__(self, "candidate_paths", paths)
        if not isinstance(self.reduction_dual, bool):
            raise SemanticError("transfer_demand.reduction_dual must be a boolean")

    def physical_link(self, src_rank: int, dst_rank: int) -> Tuple[int, int]:
        if self.reduction_dual:
            return dst_rank, src_rank
        return src_rank, dst_rank


def routing_unit_key(demand: TransferDemand) -> RoutingUnitKey:
    if not isinstance(demand, TransferDemand):
        raise SemanticError("demand must be a TransferDemand")
    return (
        demand.root_rank,
        demand.logical_position,
        tuple(sorted(demand.contributors)),
        demand.reduction_dual,
    )


@dataclass(frozen=True)
class SolverProblem:
    node: PlanNode
    inputs: ResolvedInput
    topology: Topology
    demands: Tuple[TransferDemand, ...]
    candidate_edges: FrozenSet[CandidateEdge]
    infeasible_demand_ids: Tuple[str, ...]
    restrictions: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.node, PlanNode):
            raise SemanticError("solver_problem.node must be a PlanNode")
        if not isinstance(self.inputs, ResolvedInput):
            raise SemanticError("solver_problem.inputs must be a ResolvedInput")
        if not isinstance(self.topology, Topology):
            raise SemanticError("solver_problem.topology must be a Topology")
        demands = tuple(self.demands)
        if not all(isinstance(item, TransferDemand) for item in demands):
            raise SemanticError(
                "solver_problem.demands must contain TransferDemand values"
            )
        demand_ids = tuple(item.demand_id for item in demands)
        if len(demand_ids) != len(set(demand_ids)):
            raise SemanticError("solver problem demand IDs must be unique")
        object.__setattr__(self, "demands", demands)
        edges = frozenset(self.candidate_edges)
        if not all(isinstance(item, CandidateEdge) for item in edges):
            raise SemanticError(
                "solver_problem.candidate_edges must contain CandidateEdge values"
            )
        object.__setattr__(self, "candidate_edges", edges)
        infeasible = tuple(sorted(self.infeasible_demand_ids))
        if not set(infeasible) <= set(demand_ids):
            raise SemanticError("infeasible demand ID is unknown")
        object.__setattr__(self, "infeasible_demand_ids", infeasible)
        restrictions = tuple(sorted(set(self.restrictions)))
        object.__setattr__(self, "restrictions", restrictions)

    @property
    def reduction_dual(self) -> bool:
        return bool(self.demands) and all(
            demand.reduction_dual for demand in self.demands
        )

    @property
    def search_space_restricted(self) -> bool:
        return bool(self.restrictions)

    @property
    def slice_count(self) -> int:
        return self.inputs.hyperparameters.slice_count

    @property
    def slice_size_bytes(self) -> int:
        return self.inputs.hyperparameters.slice_size_bytes

    def physical_key(self, demand: TransferDemand, link: LinkKey) -> LinkKey:
        src, dst = demand.physical_link(link.src_rank, link.dst_rank)
        return LinkKey(src, dst)


def _logical_position(contributors: FrozenSet[int], slice_count: int) -> int:
    positions = {
        logical_slice_index(slice_id, slice_count)
        for slice_id in contributors
    }
    if len(positions) != 1:
        raise SemanticError(
            "logical transfer contributors must share one logical position"
        )
    return next(iter(positions))


def _value_identity(values: FrozenSet[int]) -> str:
    return ".".join("{:08d}".format(value) for value in sorted(values))


def _virtual_links(node: PlanNode, reduction_dual: bool) -> FrozenSet[LinkKey]:
    if reduction_dual:
        return frozenset(
            LinkKey(link.dst_rank, link.src_rank)
            for link in node.allowed_links
        )
    return node.allowed_links


def _matching_forbidden(
    inputs: ResolvedInput,
    members: FrozenSet[int],
    stage_id: int,
) -> Tuple[ForbiddenTransfer, ...]:
    return tuple(
        item
        for item in inputs.atom_constraints.forbidden_transfers
        if item.stage_id == stage_id and item.slice_id in members
    )


def _forbidden_virtual_edges(
    members: FrozenSet[int],
    forbidden: Tuple[ForbiddenTransfer, ...],
    reduction_dual: bool,
) -> FrozenSet[LinkKey]:
    edges = set()
    for item in forbidden:
        if item.slice_id not in members:
            continue
        if reduction_dual:
            edges.add(LinkKey(item.dst_rank, item.src_rank))
        else:
            edges.add(LinkKey(item.src_rank, item.dst_rank))
    return frozenset(edges)


def _candidate_paths(
    *,
    node: PlanNode,
    inputs: ResolvedInput,
    topology: Topology,
    root: int,
    leaf: int,
    members: FrozenSet[int],
    reduction_dual: bool,
) -> Tuple[FrozenSet[LinkKey], Tuple[Tuple[int, ...], ...]]:
    links = _virtual_links(node, reduction_dual)
    forbidden = _matching_forbidden(inputs, members, node.stage_id)
    links = links - _forbidden_virtual_edges(
        members,
        forbidden,
        reduction_dual,
    )
    links = viable_path_links(links, root, leaf)

    def edge_cost(src: int, dst: int) -> float:
        physical = LinkKey(dst, src) if reduction_dual else LinkKey(src, dst)
        return topology.link(physical).performance.invbw_us

    paths = ranked_simple_paths(links, root, leaf, edge_cost, limit=32)
    if inputs.strategies.shortest_paths:
        paths = retain_shortest_paths(paths, edge_cost)
        links = frozenset(
            LinkKey(src, dst)
            for path in paths
            for src, dst in zip(path, path[1:])
        )
    return links, paths


def _demand(
    *,
    node: PlanNode,
    inputs: ResolvedInput,
    topology: Topology,
    root: int,
    leaf: int,
    contributors: FrozenSet[int],
    members: FrozenSet[int],
    reduction_dual: bool,
) -> TransferDemand:
    logical_position = _logical_position(
        contributors,
        inputs.hyperparameters.slice_count,
    )
    demand_id = "{}-a{:08d}-r{:08d}-l{:08d}-c{}-m{}".format(
        node.node_id,
        logical_position,
        root,
        leaf,
        _value_identity(contributors),
        _value_identity(members),
    )
    forbidden = _matching_forbidden(inputs, members, node.stage_id)
    legal_links, candidate_paths = _candidate_paths(
        node=node,
        inputs=inputs,
        topology=topology,
        root=root,
        leaf=leaf,
        members=members,
        reduction_dual=reduction_dual,
    )
    return TransferDemand(
        demand_id=demand_id,
        node_id=node.node_id,
        stage_id=node.stage_id,
        root_rank=root,
        required_leaf_rank=leaf,
        logical_position=logical_position,
        contributors=contributors,
        member_slice_ids=members,
        allowed_links=_virtual_links(node, reduction_dual),
        legal_links=legal_links,
        forbidden_members=forbidden,
        candidate_paths=candidate_paths,
        reduction_dual=reduction_dual,
    )


def _broadcast_demands(
    node: PlanNode,
    inputs: ResolvedInput,
    topology: Topology,
) -> Tuple[TransferDemand, ...]:
    demands = []
    kind = node.local_collective.kind
    if kind is CollectiveKind.ALL_GATHER:
        roots = {
            contributors: slot.rank
            for slot, contributors in node.logical_input.values.items()
        }
    else:
        root = node.local_collective.root
        roots = {
            contributors: root
            for contributors in node.logical_input.values.values()
        }
    for output_slot, contributors in node.logical_output.values.items():
        root = roots.get(contributors)
        if root is None:
            raise SemanticError("broadcast output has no matching logical input")
        if output_slot.rank == root:
            continue
        demands.append(
            _demand(
                node=node,
                inputs=inputs,
                topology=topology,
                root=root,
                leaf=output_slot.rank,
                contributors=contributors,
                members=contributors,
                reduction_dual=False,
            )
        )
    return tuple(demands)


def _chain_demands(
    node: PlanNode,
    inputs: ResolvedInput,
    topology: Topology,
) -> Tuple[TransferDemand, ...]:
    input_ranks = {
        contributors: slot.rank
        for slot, contributors in node.logical_input.values.items()
    }
    demands = []
    for output_slot, contributors in node.logical_output.values.items():
        root = input_ranks.get(contributors)
        if root is None:
            raise SemanticError("chain output has no matching logical input")
        if output_slot.rank == root:
            continue
        demands.append(
            _demand(
                node=node,
                inputs=inputs,
                topology=topology,
                root=root,
                leaf=output_slot.rank,
                contributors=contributors,
                members=contributors,
                reduction_dual=False,
            )
        )
    return tuple(demands)


def _reduction_demands(
    node: PlanNode,
    inputs: ResolvedInput,
    topology: Topology,
) -> Tuple[TransferDemand, ...]:
    demands = []
    slice_count = inputs.hyperparameters.slice_count
    for output_slot, contributors in node.logical_output.values.items():
        logical_position = _logical_position(contributors, slice_count)
        for input_slot, members in node.logical_input.values.items():
            if input_slot.rank == output_slot.rank:
                continue
            if not members <= contributors:
                continue
            if _logical_position(members, slice_count) != logical_position:
                continue
            demands.append(
                _demand(
                    node=node,
                    inputs=inputs,
                    topology=topology,
                    root=output_slot.rank,
                    leaf=input_slot.rank,
                    contributors=contributors,
                    members=members,
                    reduction_dual=True,
                )
            )
    return tuple(demands)


def _edge_channel_limit(
    topology: Topology,
    physical: LinkKey,
    requested: int,
) -> int:
    edge = topology.link(physical)
    limits = [requested, edge.max_channels]
    limits.extend(
        topology.shared_resources[resource_id].max_channels
        for resource_id in edge.resource_ids
    )
    return min(limits)


def build_solver_problem(
    node: PlanNode,
    inputs: ResolvedInput,
    topology: Topology,
) -> SolverProblem:
    if not isinstance(node, PlanNode):
        raise SemanticError("node must be a PlanNode")
    if not isinstance(inputs, ResolvedInput):
        raise SemanticError("inputs must be a ResolvedInput")
    if not isinstance(topology, Topology):
        raise SemanticError("topology must be a Topology")
    if inputs.rank_count != topology.rank_count:
        raise SemanticError("input and topology rank counts must agree")
    if any(link not in topology.links for link in node.allowed_links):
        raise SemanticError("plan node contains a link outside the topology")
    kind = node.local_collective.kind
    if kind in _REDUCTION_KINDS:
        demands = _reduction_demands(node, inputs, topology)
    elif kind in {
        CollectiveKind.BROADCAST,
        CollectiveKind.ALL_GATHER,
        CollectiveKind.SCATTER,
    }:
        demands = _broadcast_demands(node, inputs, topology)
    elif kind in {
        CollectiveKind.GATHER,
        CollectiveKind.ALL_TO_ALL,
    }:
        demands = _chain_demands(node, inputs, topology)
    else:
        raise SemanticError(
            "{} must be decomposed before solving".format(kind.value)
        )
    candidate_edges = set()
    for demand in demands:
        for virtual in demand.legal_links:
            physical = LinkKey(
                *demand.physical_link(
                    virtual.src_rank,
                    virtual.dst_rank,
                )
            )
            for channel in range(
                _edge_channel_limit(
                    topology,
                    physical,
                    inputs.solver.max_channels,
                )
            ):
                candidate_edges.add(
                    CandidateEdge(
                        virtual.src_rank,
                        virtual.dst_rank,
                        channel,
                    )
                )
    restrictions = []
    if inputs.strategies.shortest_paths:
        restrictions.append("shortest_paths")
    if inputs.strategies.symmetry:
        restrictions.append("symmetry")
    if inputs.strategies.batching:
        restrictions.append("batching")
    return SolverProblem(
        node=node,
        inputs=inputs,
        topology=topology,
        demands=tuple(sorted(demands, key=lambda item: item.demand_id)),
        candidate_edges=frozenset(candidate_edges),
        infeasible_demand_ids=tuple(
            demand.demand_id
            for demand in demands
            if not demand.candidate_paths
        ),
        restrictions=tuple(restrictions),
    )
