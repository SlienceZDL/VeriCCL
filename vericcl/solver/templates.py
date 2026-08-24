from dataclasses import dataclass
from itertools import permutations
from typing import Dict, Iterable, List, Mapping, Tuple

from vericcl.errors import SemanticError
from vericcl.input.json_codec import sha256_json
from vericcl.planner.model import PlanNode, PlanningMode, StageInterface
from vericcl.semantics.collective import CollectiveKind
from vericcl.solver.demands import SolverProblem, TransferDemand
from vericcl.topology.isomorphism import (
    exact_domain_mapping_is_valid,
    exact_domain_signature,
)
from vericcl.topology.model import LinkKey


_TREE_KINDS = frozenset(
    {
        CollectiveKind.BROADCAST,
        CollectiveKind.ALL_GATHER,
    }
)
_CHAIN_KINDS = frozenset(
    {
        CollectiveKind.GATHER,
        CollectiveKind.SCATTER,
        CollectiveKind.ALL_TO_ALL,
    }
)


# Exact semantic-node labeling is factorial. Seven external nodes need at most
# 5,040 labelings; larger units retain raw identity and only lose reuse.
_EXACT_SEMANTIC_EXTERNAL_NODE_LIMIT = 7


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise SemanticError("{} must be a non-empty string".format(field))
    return value


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SemanticError("{} must be a non-negative integer".format(field))
    return value


def _invertible_mapping(value: object, field: str) -> Tuple[Tuple[int, int], ...]:
    try:
        mapping = tuple(tuple(pair) for pair in value)
    except TypeError as error:
        raise SemanticError("{} must be an iterable of pairs".format(field)) from error
    if not mapping or any(len(pair) != 2 for pair in mapping):
        raise SemanticError("{} must contain mapping pairs".format(field))
    for source, target in mapping:
        _integer(source, field)
        _integer(target, field)
    sources = tuple(source for source, _ in mapping)
    targets = tuple(target for _, target in mapping)
    if len(sources) != len(set(sources)) or len(targets) != len(set(targets)):
        raise SemanticError("{} must be invertible".format(field))
    return tuple(sorted(mapping))


@dataclass(frozen=True)
class RoutingUnit:
    unit_id: str
    node: PlanNode
    demands: tuple[TransferDemand, ...]

    def __post_init__(self) -> None:
        _identifier(self.unit_id, "routing_unit.unit_id")
        if not isinstance(self.node, PlanNode):
            raise SemanticError("routing_unit.node must be a PlanNode")
        demands = tuple(self.demands)
        if not demands or not all(
            isinstance(demand, TransferDemand) for demand in demands
        ):
            raise SemanticError(
                "routing_unit.demands must contain TransferDemand values"
            )
        if any(demand.node_id != self.node.node_id for demand in demands):
            raise SemanticError("routing unit demand belongs to another node")
        demand_ids = tuple(demand.demand_id for demand in demands)
        if len(demand_ids) != len(set(demand_ids)):
            raise SemanticError("routing unit demand IDs must be unique")
        object.__setattr__(
            self,
            "demands",
            tuple(sorted(demands, key=lambda demand: demand.demand_id)),
        )


def _unit_slice_ids(unit: RoutingUnit) -> Tuple[int, ...]:
    return tuple(
        sorted(
            {
                slice_id
                for demand in unit.demands
                for values in (demand.contributors, demand.member_slice_ids)
                for slice_id in values
            }
        )
    )


def _unit_logical_positions(unit: RoutingUnit) -> Tuple[int, ...]:
    return tuple(
        sorted({demand.logical_position for demand in unit.demands})
    )


@dataclass(frozen=True)
class TemplateMember:
    unit_id: str
    node_id: str
    rank_map: tuple[tuple[int, int], ...]
    contributor_map: tuple[tuple[int, int], ...]
    logical_position_map: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        _identifier(self.unit_id, "template_member.unit_id")
        _identifier(self.node_id, "template_member.node_id")
        object.__setattr__(
            self,
            "rank_map",
            _invertible_mapping(self.rank_map, "template_member.rank_map"),
        )
        object.__setattr__(
            self,
            "contributor_map",
            _invertible_mapping(
                self.contributor_map,
                "template_member.contributor_map",
            ),
        )
        object.__setattr__(
            self,
            "logical_position_map",
            _invertible_mapping(
                self.logical_position_map,
                "template_member.logical_position_map",
            ),
        )


@dataclass(frozen=True)
class SolverTemplate:
    template_id: str
    representative: RoutingUnit
    members: tuple[TemplateMember, ...]
    exact_signature: str

    def __post_init__(self) -> None:
        _identifier(self.template_id, "solver_template.template_id")
        if not isinstance(self.representative, RoutingUnit):
            raise SemanticError(
                "solver_template.representative must be a RoutingUnit"
            )
        members = tuple(self.members)
        if not members or not all(
            isinstance(member, TemplateMember) for member in members
        ):
            raise SemanticError(
                "solver_template.members must contain TemplateMember values"
            )
        unit_ids = tuple(member.unit_id for member in members)
        if len(unit_ids) != len(set(unit_ids)):
            raise SemanticError("solver template member unit IDs must be unique")
        if self.representative.unit_id not in unit_ids:
            raise SemanticError("solver template must include its representative")
        expected_sources = {
            "rank_map": set(self.representative.node.communication_group),
            "contributor_map": set(_unit_slice_ids(self.representative)),
            "logical_position_map": set(
                _unit_logical_positions(self.representative)
            ),
        }
        for member in members:
            for field, expected in expected_sources.items():
                actual = {source for source, _ in getattr(member, field)}
                if actual != expected:
                    raise SemanticError(
                        "solver template member mapping source coverage is invalid"
                    )
        representative_member = next(
            member
            for member in members
            if member.unit_id == self.representative.unit_id
        )
        if representative_member.node_id != self.representative.node.node_id:
            raise SemanticError(
                "solver template representative member node does not match"
            )
        if any(
            source != target
            for field in expected_sources
            for source, target in getattr(representative_member, field)
        ):
            raise SemanticError(
                "solver template representative member must use identity maps"
            )
        _identifier(self.exact_signature, "solver_template.exact_signature")
        object.__setattr__(
            self,
            "members",
            tuple(sorted(members, key=lambda member: member.unit_id)),
        )


def _unit_id(node: PlanNode, demands: Iterable[TransferDemand]) -> str:
    demand_ids = sorted(demand.demand_id for demand in demands)
    return "{}-unit-{}".format(
        node.node_id,
        sha256_json(demand_ids)[:16],
    )


def _routing_unit(
    problem: SolverProblem,
    demands: Tuple[TransferDemand, ...],
) -> RoutingUnit:
    return RoutingUnit(
        unit_id=_unit_id(problem.node, demands),
        node=problem.node,
        demands=demands,
    )


def split_routing_units(problem: SolverProblem) -> tuple[RoutingUnit, ...]:
    if not isinstance(problem, SolverProblem):
        raise SemanticError("problem must be a SolverProblem")
    if not problem.demands:
        return ()
    reduction_flags = {demand.reduction_dual for demand in problem.demands}
    if len(reduction_flags) != 1:
        raise SemanticError("solver problem mixes reduction-dual demand roles")
    reduction_dual = next(iter(reduction_flags))
    kind = problem.node.local_collective.kind
    groups: List[Tuple[TransferDemand, ...]] = []
    if reduction_dual or kind in _TREE_KINDS:
        grouped: Dict[tuple, List[TransferDemand]] = {}
        for demand in problem.demands:
            key = (
                demand.root_rank,
                demand.logical_position,
                tuple(sorted(demand.contributors)),
                demand.reduction_dual,
            )
            grouped.setdefault(key, []).append(demand)
        groups.extend(
            tuple(sorted(values, key=lambda demand: demand.demand_id))
            for _, values in sorted(grouped.items())
        )
    elif kind in _CHAIN_KINDS:
        groups.extend((demand,) for demand in problem.demands)
    else:
        raise SemanticError(
            "{} does not define routing-unit semantics".format(kind.value)
        )
    return tuple(
        sorted(
            (
                _routing_unit(problem, demands)
                for demands in groups
            ),
            key=lambda unit: unit.unit_id,
        )
    )


def _rank_indices(problem: SolverProblem) -> Dict[int, int]:
    return {
        rank: index
        for index, rank in enumerate(problem.node.communication_group)
    }


def _logical_indices(unit: RoutingUnit) -> Dict[int, int]:
    positions = sorted({demand.logical_position for demand in unit.demands})
    return {position: index for index, position in enumerate(positions)}


def _rank_token(
    rank: int,
    problem: SolverProblem,
    rank_indices: Mapping[int, int],
) -> tuple:
    if rank in rank_indices:
        token = rank_indices[rank]
        if isinstance(token, tuple):
            return ("semantic", token)
        return ("domain", token)
    return (
        "external",
        rank,
        problem.topology.node_membership[rank],
        rank in problem.topology.gateways,
    )


def _logical_token(
    logical_position: int,
    logical_indices: Mapping[int, int],
) -> tuple:
    if logical_position in logical_indices:
        return ("position", logical_indices[logical_position])
    return ("external_position", logical_position)


def _slice_token(
    slice_id: int,
    problem: SolverProblem,
    rank_indices: Mapping[int, int],
    logical_indices: Mapping[int, int],
) -> tuple:
    slice_count = problem.slice_count
    source_rank = slice_id // slice_count
    logical_position = slice_id % slice_count
    return (
        _rank_token(source_rank, problem, rank_indices),
        _logical_token(logical_position, logical_indices),
    )


def _offset_token(
    offset: int,
    contributors: frozenset[int],
    problem: SolverProblem,
    rank_indices: Mapping[int, int],
    logical_indices: Mapping[int, int],
) -> tuple:
    slice_count = problem.slice_count
    logical_positions = {
        contributor % slice_count for contributor in contributors
    }
    if len(logical_positions) == 1:
        logical_position = next(iter(logical_positions))
        logical_token = _logical_token(logical_position, logical_indices)
        if offset == logical_position:
            return ("logical", logical_token)
    if len(contributors) == 1:
        contributor = next(iter(contributors))
        if offset == contributor:
            return (
                "contributor",
                _slice_token(
                    contributor,
                    problem,
                    rank_indices,
                    logical_indices,
                ),
            )
        quotient, remainder = divmod(
            slice_count,
            problem.inputs.rank_count,
        )
        if quotient > 0 and remainder == 0:
            source_rank, logical_position = divmod(
                contributor,
                slice_count,
            )
            if offset == source_rank * quotient + logical_position % quotient:
                return (
                    "source_compact",
                    _rank_token(source_rank, problem, rank_indices),
                    _logical_token(logical_position, logical_indices),
                )
    if len(logical_positions) == 1:
        logical_position = next(iter(logical_positions))
        quotient, remainder = divmod(
            slice_count,
            problem.inputs.rank_count,
        )
        if (
            quotient > 0
            and remainder == 0
            and offset == logical_position % quotient
        ):
            return (
                "compact",
                _logical_token(logical_position, logical_indices),
            )
    return ("absolute", offset)


def _link_token(key: LinkKey, rank_indices: Mapping[int, int]) -> tuple:
    return (rank_indices[key.src_rank], rank_indices[key.dst_rank])


def _interface_descriptor(
    interface: StageInterface,
    problem: SolverProblem,
    rank_indices: Mapping[int, int],
    logical_indices: Mapping[int, int],
    relevant_slice_ids: frozenset[int],
) -> tuple:
    values = []
    for slot, contributors in interface.values.items():
        if contributors.isdisjoint(relevant_slice_ids):
            continue
        values.append(
            (
                _rank_token(slot.rank, problem, rank_indices),
                _offset_token(
                    slot.offset,
                    contributors,
                    problem,
                    rank_indices,
                    logical_indices,
                ),
                tuple(
                    sorted(
                        _slice_token(
                            contributor,
                            problem,
                            rank_indices,
                            logical_indices,
                        )
                        for contributor in contributors
                    )
                ),
            )
        )
    return tuple(sorted(values))


def _forbidden_descriptor(
    demand: TransferDemand,
    problem: SolverProblem,
    rank_indices: Mapping[int, int],
    logical_indices: Mapping[int, int],
) -> tuple:
    return tuple(
        sorted(
            (
                (
                    _slice_token(
                        item.slice_id,
                        problem,
                        rank_indices,
                        logical_indices,
                    ),
                    _rank_token(
                        item.src_rank,
                        problem,
                        rank_indices,
                    ),
                    _rank_token(
                        item.dst_rank,
                        problem,
                        rank_indices,
                    ),
                    item.stage_id,
                )
                for item in demand.forbidden_members
            ),
        )
    )


def _demand_descriptor(
    demand: TransferDemand,
    problem: SolverProblem,
    rank_indices: Mapping[int, int],
    logical_indices: Mapping[int, int],
) -> tuple:
    return (
        demand.stage_id,
        _rank_token(demand.root_rank, problem, rank_indices),
        _rank_token(
            demand.required_leaf_rank,
            problem,
            rank_indices,
        ),
        _logical_token(demand.logical_position, logical_indices),
        tuple(
            sorted(
                _slice_token(
                    contributor,
                    problem,
                    rank_indices,
                    logical_indices,
                )
                for contributor in demand.contributors
            )
        ),
        tuple(
            sorted(
                _slice_token(
                    member,
                    problem,
                    rank_indices,
                    logical_indices,
                )
                for member in demand.member_slice_ids
            )
        ),
        tuple(
            sorted(_link_token(key, rank_indices) for key in demand.allowed_links)
        ),
        tuple(
            sorted(_link_token(key, rank_indices) for key in demand.legal_links)
        ),
        _forbidden_descriptor(
            demand,
            problem,
            rank_indices,
            logical_indices,
        ),
        tuple(
            sorted(
                tuple(rank_indices[rank] for rank in path)
                for path in demand.candidate_paths
            )
        ),
        demand.reduction_dual,
    )


def _exact_signature(
    unit: RoutingUnit,
    problem: SolverProblem,
    planning_mode: PlanningMode,
    structural: dict,
) -> str:
    rank_indices = structural["rank_indices"]
    logical_indices = _logical_indices(unit)
    collective = problem.node.local_collective
    relevant_slice_ids = frozenset(_all_slice_ids(unit))
    value = {
        "planning_mode": planning_mode.value,
        "routing_role": "tree"
        if unit.demands[0].reduction_dual
        or collective.kind in _TREE_KINDS
        else "chain",
        "stage_id": problem.node.stage_id,
        "collective": {
            "kind": collective.kind.value,
            "datatype": collective.datatype,
            "reduction_op": collective.reduction_op,
            "root": None
            if collective.root is None
            else rank_indices[collective.root],
            "inplace": collective.inplace,
        },
        "dual": problem.node.dual_of_node_id is not None,
        "slice_size_bytes": problem.slice_size_bytes,
        "rank_roles": structural["rank_roles"],
        "domain_signature": structural["domain_signature"],
        "allowed_links": structural["allowed_links"],
        "logical_input": _interface_descriptor(
            problem.node.logical_input,
            problem,
            rank_indices,
            logical_indices,
            relevant_slice_ids,
        ),
        "logical_output": _interface_descriptor(
            problem.node.logical_output,
            problem,
            rank_indices,
            logical_indices,
            relevant_slice_ids,
        ),
        "demands": tuple(
            sorted(
                _demand_descriptor(
                    demand,
                    problem,
                    rank_indices,
                    logical_indices,
                )
                for demand in unit.demands
            )
        ),
        "restrictions": problem.restrictions,
    }
    return sha256_json(value)


def _structural_descriptor(problem: SolverProblem) -> dict:
    rank_indices = _rank_indices(problem)
    node_classes: Dict[int, int] = {}
    roles = []
    for rank in problem.node.communication_group:
        node_id = problem.topology.node_membership[rank]
        if node_id not in node_classes:
            node_classes[node_id] = len(node_classes)
        roles.append(
            (node_classes[node_id], rank in problem.topology.gateways)
        )
    return {
        "rank_indices": rank_indices,
        "rank_roles": tuple(roles),
        "domain_signature": exact_domain_signature(
            problem.topology,
            problem.node.communication_group,
            problem.node.shared_resource_ids,
        ),
        "allowed_links": tuple(
            sorted(
                _link_token(key, rank_indices)
                for key in problem.node.allowed_links
            )
        ),
    }


def _structural_cache_key(problem: SolverProblem) -> tuple:
    return (
        id(problem.topology),
        problem.node.communication_group,
        tuple(sorted(problem.node.allowed_links)),
        tuple(sorted(problem.node.shared_resource_ids)),
    )


def _all_slice_ids(unit: RoutingUnit) -> Tuple[int, ...]:
    return _unit_slice_ids(unit)


def _external_semantic_ranks(
    unit: RoutingUnit,
    problem: SolverProblem,
) -> Tuple[int, ...]:
    group = set(unit.node.communication_group)
    return tuple(
        sorted(
            {
                slice_id // problem.slice_count
                for slice_id in _unit_slice_ids(unit)
                if slice_id // problem.slice_count not in group
            }
        )
    )


def _external_role_shape(
    unit: RoutingUnit,
    problem: SolverProblem,
) -> tuple:
    slice_count = problem.slice_count
    return tuple(
        sorted(
            (
                demand.root_rank,
                demand.required_leaf_rank,
                tuple(
                    sorted(
                        {
                            slice_id // slice_count
                            for slice_id in demand.contributors
                        }
                    )
                ),
                tuple(
                    sorted(
                        {
                            slice_id // slice_count
                            for slice_id in demand.member_slice_ids
                        }
                    )
                ),
                tuple(
                    sorted(
                        (
                            item.slice_id // slice_count,
                            item.src_rank,
                            item.dst_rank,
                            item.stage_id,
                        )
                        for item in demand.forbidden_members
                    )
                ),
                demand.reduction_dual,
            )
            for demand in unit.demands
        )
    )


def _external_label_cache_key(
    unit: RoutingUnit,
    problem: SolverProblem,
) -> tuple:
    return (
        id(problem.topology),
        unit.node.communication_group,
        _external_semantic_ranks(unit, problem),
        _external_role_shape(unit, problem),
    )


def _external_labels(
    unit: RoutingUnit,
    problem: SolverProblem,
    external_node_labels: Mapping[int, int],
) -> dict[int, object]:
    external_ranks = _external_semantic_ranks(unit, problem)
    node_ranks: Dict[int, List[int]] = {}
    for rank, node_id in problem.topology.node_membership.items():
        node_ranks.setdefault(node_id, []).append(rank)
    rank_positions = {
        rank: position
        for ranks in node_ranks.values()
        for position, rank in enumerate(sorted(ranks))
    }
    domain_node_labels = {}
    for rank in unit.node.communication_group:
        node_id = problem.topology.node_membership[rank]
        if node_id not in domain_node_labels:
            domain_node_labels[node_id] = len(domain_node_labels)
    labels = {}
    for rank in external_ranks:
        node_id = problem.topology.node_membership[rank]
        if node_id in domain_node_labels:
            node_token = (
                "semantic_domain_node",
                domain_node_labels[node_id],
            )
        else:
            node_token = (
                "semantic_external_node",
                external_node_labels[node_id],
            )
        labels[rank] = (
            node_token,
            rank_positions[rank],
            rank in problem.topology.gateways,
        )
    return labels


def _signature_with_external_labels(
    unit: RoutingUnit,
    problem: SolverProblem,
    planning_mode: PlanningMode,
    structural: dict,
    external_labels: Mapping[int, object],
) -> str:
    descriptor = dict(structural)
    descriptor["rank_indices"] = dict(structural["rank_indices"])
    descriptor["rank_indices"].update(external_labels)
    return _exact_signature(unit, problem, planning_mode, descriptor)


def _canonical_unit_signature(
    unit: RoutingUnit,
    problem: SolverProblem,
    planning_mode: PlanningMode,
    structural: dict,
    external_label_cache: dict,
) -> tuple[str, dict[int, object]]:
    cache_key = _external_label_cache_key(unit, problem)
    cached = external_label_cache.get(cache_key)
    if cached is not None:
        return (
            _signature_with_external_labels(
                unit,
                problem,
                planning_mode,
                structural,
                cached,
            ),
            cached,
        )
    external_ranks = _external_semantic_ranks(unit, problem)
    domain_node_labels = {}
    for rank in unit.node.communication_group:
        node_id = problem.topology.node_membership[rank]
        if node_id not in domain_node_labels:
            domain_node_labels[node_id] = len(domain_node_labels)
    external_nodes = tuple(
        sorted(
            {
                problem.topology.node_membership[rank]
                for rank in external_ranks
                if problem.topology.node_membership[rank]
                not in domain_node_labels
            }
        )
    )
    if len(external_nodes) > _EXACT_SEMANTIC_EXTERNAL_NODE_LIMIT:
        labels = {
            rank: ("semantic_identity", rank) for rank in external_ranks
        }
        external_label_cache[cache_key] = labels
        return (
            _signature_with_external_labels(
                unit,
                problem,
                planning_mode,
                structural,
                labels,
            ),
            labels,
        )
    if not external_nodes:
        labels = _external_labels(unit, problem, {})
        external_label_cache[cache_key] = labels
        return (
            _signature_with_external_labels(
                unit,
                problem,
                planning_mode,
                structural,
                labels,
            ),
            labels,
        )
    best = None
    for node_order in permutations(external_nodes):
        external_node_labels = {
            node_id: label for label, node_id in enumerate(node_order)
        }
        labels = _external_labels(unit, problem, external_node_labels)
        signature = _signature_with_external_labels(
            unit,
            problem,
            planning_mode,
            structural,
            labels,
        )
        candidate = (signature, node_order, labels)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    signature, _, labels = best
    external_label_cache[cache_key] = labels
    return signature, labels


def _mapping_dictionary(mapping: Tuple[Tuple[int, int], ...]) -> Dict[int, int]:
    return dict(mapping)


def _mapped_rank(rank: int, rank_map: Mapping[int, int]) -> int:
    return rank_map.get(rank, rank)


def _mapped_slice(
    slice_id: int,
    representative_problem: SolverProblem,
    member_problem: SolverProblem,
    rank_map: Mapping[int, int],
    logical_map: Mapping[int, int],
) -> int:
    source = slice_id // representative_problem.slice_count
    position = slice_id % representative_problem.slice_count
    return (
        _mapped_rank(source, rank_map) * member_problem.slice_count
        + logical_map[position]
    )


def _mapped_demand_value(
    demand: TransferDemand,
    representative_problem: SolverProblem,
    member_problem: SolverProblem,
    rank_map: Mapping[int, int],
    logical_map: Mapping[int, int],
) -> tuple:
    def mapped_link(key: LinkKey) -> tuple:
        return (
            _mapped_rank(key.src_rank, rank_map),
            _mapped_rank(key.dst_rank, rank_map),
        )

    return (
        demand.stage_id,
        _mapped_rank(demand.root_rank, rank_map),
        _mapped_rank(demand.required_leaf_rank, rank_map),
        logical_map[demand.logical_position],
        tuple(
            sorted(
                _mapped_slice(
                    slice_id,
                    representative_problem,
                    member_problem,
                    rank_map,
                    logical_map,
                )
                for slice_id in demand.contributors
            )
        ),
        tuple(
            sorted(
                _mapped_slice(
                    slice_id,
                    representative_problem,
                    member_problem,
                    rank_map,
                    logical_map,
                )
                for slice_id in demand.member_slice_ids
            )
        ),
        tuple(sorted(mapped_link(key) for key in demand.allowed_links)),
        tuple(sorted(mapped_link(key) for key in demand.legal_links)),
        tuple(
            sorted(
                (
                    _mapped_slice(
                        item.slice_id,
                        representative_problem,
                        member_problem,
                        rank_map,
                        logical_map,
                    ),
                    _mapped_rank(item.src_rank, rank_map),
                    _mapped_rank(item.dst_rank, rank_map),
                    item.stage_id,
                )
                for item in demand.forbidden_members
            )
        ),
        tuple(
            sorted(
                tuple(_mapped_rank(rank, rank_map) for rank in path)
                for path in demand.candidate_paths
            )
        ),
        demand.reduction_dual,
    )


def _target_demand_value(demand: TransferDemand) -> tuple:
    return (
        demand.stage_id,
        demand.root_rank,
        demand.required_leaf_rank,
        demand.logical_position,
        tuple(sorted(demand.contributors)),
        tuple(sorted(demand.member_slice_ids)),
        tuple(
            sorted((key.src_rank, key.dst_rank) for key in demand.allowed_links)
        ),
        tuple(
            sorted((key.src_rank, key.dst_rank) for key in demand.legal_links)
        ),
        tuple(
            sorted(
                (
                    item.slice_id,
                    item.src_rank,
                    item.dst_rank,
                    item.stage_id,
                )
                for item in demand.forbidden_members
            )
        ),
        demand.candidate_paths,
        demand.reduction_dual,
    )


def _template_member(
    representative: RoutingUnit,
    representative_problem: SolverProblem,
    representative_external_labels: Mapping[int, object],
    unit: RoutingUnit,
    problem: SolverProblem,
    external_labels: Mapping[int, object],
    resource_mapping_cache: dict,
) -> TemplateMember | None:
    if len(representative.node.communication_group) != len(
        unit.node.communication_group
    ):
        return None
    route_rank_pairs = list(
        zip(
            representative.node.communication_group,
            unit.node.communication_group,
        )
    )
    representative_by_label = {
        label: rank for rank, label in representative_external_labels.items()
    }
    member_by_label = {
        label: rank for rank, label in external_labels.items()
    }
    if set(representative_by_label) != set(member_by_label):
        return None
    extended_rank_pairs = list(route_rank_pairs)
    extended_rank_pairs.extend(
        (representative_by_label[label], member_by_label[label])
        for label in sorted(representative_by_label, key=repr)
    )
    if len({source for source, _ in extended_rank_pairs}) != len(
        extended_rank_pairs
    ) or len({target for _, target in extended_rank_pairs}) != len(
        extended_rank_pairs
    ):
        return None
    rank_map = tuple(sorted(route_rank_pairs))
    extended_rank_mapping = _mapping_dictionary(
        tuple(sorted(extended_rank_pairs))
    )
    if len(extended_rank_pairs) != len(route_rank_pairs):
        resource_mapping_key = (
            id(representative_problem.topology),
            representative.node.communication_group,
            tuple(sorted(representative.node.shared_resource_ids)),
            id(problem.topology),
            unit.node.communication_group,
            tuple(sorted(unit.node.shared_resource_ids)),
            tuple(sorted(extended_rank_pairs)),
        )
        if resource_mapping_key not in resource_mapping_cache:
            resource_mapping_cache[resource_mapping_key] = (
                exact_domain_mapping_is_valid(
                    representative_problem.topology,
                    representative.node.communication_group,
                    representative.node.shared_resource_ids,
                    problem.topology,
                    unit.node.communication_group,
                    unit.node.shared_resource_ids,
                    extended_rank_pairs,
                )
            )
        if not resource_mapping_cache[resource_mapping_key]:
            return None
    representative_positions = sorted(
        {demand.logical_position for demand in representative.demands}
    )
    member_positions = sorted(
        {demand.logical_position for demand in unit.demands}
    )
    if len(representative_positions) != len(member_positions):
        return None
    logical_map = tuple(zip(representative_positions, member_positions))
    logical_mapping = _mapping_dictionary(logical_map)
    mapped_demands = tuple(
        sorted(
            _mapped_demand_value(
                demand,
                representative_problem,
                problem,
                extended_rank_mapping,
                logical_mapping,
            )
            for demand in representative.demands
        )
    )
    target_demands = tuple(
        sorted(_target_demand_value(demand) for demand in unit.demands)
    )
    if mapped_demands != target_demands:
        return None
    representative_slices = _all_slice_ids(representative)
    target_slices = set(_all_slice_ids(unit))
    contributor_map = tuple(
        (
            slice_id,
            _mapped_slice(
                slice_id,
                representative_problem,
                problem,
                extended_rank_mapping,
                logical_mapping,
            ),
        )
        for slice_id in representative_slices
    )
    if {target for _, target in contributor_map} != target_slices:
        return None
    return TemplateMember(
        unit_id=unit.unit_id,
        node_id=unit.node.node_id,
        rank_map=rank_map,
        contributor_map=contributor_map,
        logical_position_map=logical_map,
    )


def build_solver_templates(
    problems: tuple[SolverProblem, ...],
    planning_mode: PlanningMode,
) -> tuple[SolverTemplate, ...]:
    try:
        problems = tuple(problems)
    except TypeError as error:
        raise SemanticError("problems must be an iterable") from error
    if not all(isinstance(problem, SolverProblem) for problem in problems):
        raise SemanticError("problems must contain SolverProblem values")
    if not isinstance(planning_mode, PlanningMode):
        raise SemanticError("planning_mode must be a PlanningMode")
    contexts = []
    for problem in problems:
        contexts.extend(
            (unit, problem) for unit in split_routing_units(problem)
        )
    contexts.sort(key=lambda item: item[0].unit_id)
    classes = []
    class_indices_by_signature: Dict[str, List[int]] = {}
    structural_cache = {}
    external_label_cache = {}
    resource_mapping_cache = {}
    for unit, problem in contexts:
        structural_key = _structural_cache_key(problem)
        structural = structural_cache.get(structural_key)
        if structural is None:
            structural = _structural_descriptor(problem)
            structural_cache[structural_key] = structural
        signature, external_labels = _canonical_unit_signature(
            unit,
            problem,
            planning_mode,
            structural,
            external_label_cache,
        )
        selected = None
        member = None
        for index in class_indices_by_signature.get(signature, ()):
            item = classes[index]
            candidate = _template_member(
                item["representative"],
                item["problem"],
                item["external_labels"],
                unit,
                problem,
                external_labels,
                resource_mapping_cache,
            )
            if candidate is not None:
                selected = index
                member = candidate
                break
        if selected is None:
            member = _template_member(
                unit,
                problem,
                external_labels,
                unit,
                problem,
                external_labels,
                resource_mapping_cache,
            )
            if member is None:
                raise SemanticError("routing unit cannot map to itself")
            classes.append(
                {
                    "representative": unit,
                    "problem": problem,
                    "external_labels": external_labels,
                    "signature": signature,
                    "members": [member],
                }
            )
            class_indices_by_signature.setdefault(signature, []).append(
                len(classes) - 1
            )
        else:
            classes[selected]["members"].append(member)
    templates = []
    for item in classes:
        representative = item["representative"]
        signature = item["signature"]
        template_id = "template-{}-{}".format(
            planning_mode.value,
            sha256_json(
                {
                    "signature": signature,
                    "representative": representative.unit_id,
                }
            )[:20],
        )
        templates.append(
            SolverTemplate(
                template_id=template_id,
                representative=representative,
                members=tuple(item["members"]),
                exact_signature=signature,
            )
        )
    return tuple(sorted(templates, key=lambda template: template.template_id))
