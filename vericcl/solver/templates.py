from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Tuple

from vericcl.errors import SemanticError
from vericcl.input.json_codec import sha256_json
from vericcl.planner.model import PlanNode, PlanningMode, StageInterface
from vericcl.semantics.collective import CollectiveKind
from vericcl.solver.demands import SolverProblem, TransferDemand
from vericcl.topology.model import LinkKey, PerformanceCurve, Topology


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
                RoutingUnit(
                    unit_id=_unit_id(problem.node, demands),
                    node=problem.node,
                    demands=demands,
                )
                for demands in groups
            ),
            key=lambda unit: unit.unit_id,
        )
    )


def _performance(curve: PerformanceCurve) -> tuple:
    return (
        curve.alpha_us,
        curve.invbw_us,
        tuple(curve.bandwidth_bytes_per_us.items()),
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
        return ("domain", rank_indices[rank])
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


def _link_token(key: LinkKey, rank_indices: Mapping[int, int]) -> tuple:
    return (rank_indices[key.src_rank], rank_indices[key.dst_rank])


def _resource_descriptor(
    resource_id: str,
    problem: SolverProblem,
    rank_indices: Mapping[int, int],
) -> tuple:
    topology = problem.topology
    resource = topology.shared_resources[resource_id]
    members = []
    for key in resource.member_links:
        edge = topology.link(key)
        members.append(
            (
                _rank_token(key.src_rank, problem, rank_indices),
                _rank_token(key.dst_rank, problem, rank_indices),
                edge.max_channels,
                _performance(edge.performance),
            )
        )
    return (
        tuple(sorted(members)),
        resource.max_channels,
        _performance(resource.performance),
    )


def _link_descriptor(
    key: LinkKey,
    problem: SolverProblem,
    rank_indices: Mapping[int, int],
) -> tuple:
    edge = problem.topology.link(key)
    resources = tuple(
        sorted(
            _resource_descriptor(resource_id, problem, rank_indices)
            for resource_id in edge.resource_ids
        )
    )
    return (
        _rank_token(key.src_rank, problem, rank_indices),
        _rank_token(key.dst_rank, problem, rank_indices),
        edge.max_channels,
        _performance(edge.performance),
        resources,
    )


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
                _logical_token(slot.offset, logical_indices),
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
        "domain_links": structural["domain_links"],
        "allowed_links": structural["allowed_links"],
        "declared_resources": structural["declared_resources"],
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
    group_set = set(problem.node.communication_group)
    node_classes: Dict[int, int] = {}
    roles = []
    for rank in problem.node.communication_group:
        node_id = problem.topology.node_membership[rank]
        if node_id not in node_classes:
            node_classes[node_id] = len(node_classes)
        roles.append(
            (node_classes[node_id], rank in problem.topology.gateways)
        )
    domain_links = tuple(
        sorted(
            _link_descriptor(key, problem, rank_indices)
            for key in problem.topology.links
            if key.src_rank in group_set and key.dst_rank in group_set
        )
    )
    declared_resources = tuple(
        sorted(
            _resource_descriptor(resource_id, problem, rank_indices)
            for resource_id in problem.node.shared_resource_ids
        )
    )
    return {
        "rank_indices": rank_indices,
        "rank_roles": tuple(roles),
        "domain_links": domain_links,
        "allowed_links": tuple(
            sorted(
                _link_token(key, rank_indices)
                for key in problem.node.allowed_links
            )
        ),
        "declared_resources": declared_resources,
    }


def _structural_cache_key(problem: SolverProblem) -> tuple:
    return (
        problem.topology.isomorphism_signature,
        problem.node.communication_group,
        tuple(sorted(problem.node.allowed_links)),
        tuple(sorted(problem.node.shared_resource_ids)),
    )


def _all_slice_ids(unit: RoutingUnit) -> Tuple[int, ...]:
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
    unit: RoutingUnit,
    problem: SolverProblem,
) -> TemplateMember | None:
    if len(representative.node.communication_group) != len(
        unit.node.communication_group
    ):
        return None
    rank_map = tuple(
        zip(
            representative.node.communication_group,
            unit.node.communication_group,
        )
    )
    rank_mapping = _mapping_dictionary(rank_map)
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
                rank_mapping,
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
                rank_mapping,
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
    for unit, problem in contexts:
        structural_key = _structural_cache_key(problem)
        structural = structural_cache.get(structural_key)
        if structural is None:
            structural = _structural_descriptor(problem)
            structural_cache[structural_key] = structural
        signature = _exact_signature(
            unit,
            problem,
            planning_mode,
            structural,
        )
        selected = None
        member = None
        for index in class_indices_by_signature.get(signature, ()):
            item = classes[index]
            candidate = _template_member(
                item["representative"],
                item["problem"],
                unit,
                problem,
            )
            if candidate is not None:
                selected = index
                member = candidate
                break
        if selected is None:
            member = _template_member(
                unit,
                problem,
                unit,
                problem,
            )
            if member is None:
                raise SemanticError("routing unit cannot map to itself")
            classes.append(
                {
                    "representative": unit,
                    "problem": problem,
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
