import hashlib
import math
import time
from dataclasses import dataclass, field, replace
from threading import RLock
from typing import Dict, Iterable, Optional, Tuple

from vericcl.errors import SemanticError
from vericcl.input.json_codec import canonical_json, sha256_json
from vericcl.planner.model import StageInterface
from vericcl.solver.demands import SolverProblem
from vericcl.solver.global_scheduler import GLOBAL_SCHEDULER_VERSION
from vericcl.solver.model import SearchDiagnostics, SolveCandidate, SolveRequest
from vericcl.solver.templates import SolverTemplate


_ROUTE_MODEL_VERSION = "1"
_TEMPLATE_ROUTE_BACKEND = "template_route"
_LEGACY_FULL_TIME_BACKEND = "legacy_full_time_milp"
_CACHE_BACKENDS = frozenset(
    {_TEMPLATE_ROUTE_BACKEND, _LEGACY_FULL_TIME_BACKEND}
)
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def _non_negative_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SemanticError("{} must be a non-negative integer".format(field))
    return value


def _sha256_digest(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SemanticError("{} must be a lowercase SHA-256 digest".format(field))
    return value


@dataclass(frozen=True)
class CacheSignature:
    backend_type: str = _TEMPLATE_ROUTE_BACKEND
    planning_mode: str = "unknown"
    route_model_version: str = _ROUTE_MODEL_VERSION
    global_scheduler_version: str = GLOBAL_SCHEDULER_VERSION
    plan_node_count: int = 0
    problem_count: int = 0
    demand_count: int = 0
    template_count: int = 0
    template_member_count: int = 0
    structure_digest_sha256: str = _EMPTY_SHA256
    problem_digest_sha256: str = _EMPTY_SHA256
    template_digest_sha256: str = _EMPTY_SHA256

    def __post_init__(self) -> None:
        if self.backend_type not in _CACHE_BACKENDS:
            raise SemanticError("cache backend type is unsupported")
        if not isinstance(self.planning_mode, str) or not self.planning_mode:
            raise SemanticError("cache planning_mode must be a non-empty string")
        for field in ("route_model_version", "global_scheduler_version"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value:
                raise SemanticError(
                    "cache {} must be a non-empty string".format(field)
                )
        for field in (
            "plan_node_count",
            "problem_count",
            "demand_count",
            "template_count",
            "template_member_count",
        ):
            _non_negative_integer(getattr(self, field), "cache {}".format(field))
        for field in (
            "structure_digest_sha256",
            "problem_digest_sha256",
            "template_digest_sha256",
        ):
            _sha256_digest(getattr(self, field), "cache {}".format(field))


def _digest_records(records: Iterable[object]) -> str:
    digest = hashlib.sha256()
    for record in records:
        encoded = canonical_json(record).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _collective_record(spec) -> tuple:
    return (
        spec.kind.value,
        spec.datatype,
        spec.reduction_op,
        spec.root,
        spec.inplace,
    )


def _interface_records(prefix: tuple, interface: StageInterface):
    for slot, contributors in sorted(interface.values.items()):
        yield prefix + ("slot", slot.rank, slot.offset)
        for contributor in sorted(contributors):
            yield prefix + (
                "slot_contributor",
                slot.rank,
                slot.offset,
                contributor,
            )


def _structure_records(request: SolveRequest):
    plan = request.plan
    topology = request.topology
    yield (
        "plan",
        _collective_record(plan.collective),
        plan.rank_count,
        plan.slice_count,
        plan.planning_mode.value,
        plan.planning_reason,
    )
    yield from _interface_records(("plan", "initial"), plan.initial_inputs)
    for node in sorted(plan.nodes, key=lambda value: value.node_id):
        yield (
            "plan_node",
            node.node_id,
            node.stage_id,
            _collective_record(node.local_collective),
            node.dual_of_node_id,
        )
        for rank in node.communication_group:
            yield ("plan_node_rank", node.node_id, rank)
        for link in sorted(node.allowed_links):
            yield (
                "plan_node_allowed_link",
                node.node_id,
                link.src_rank,
                link.dst_rank,
            )
        for resource_id in sorted(node.shared_resource_ids):
            yield ("plan_node_resource", node.node_id, resource_id)
        yield from _interface_records(
            ("plan_node", node.node_id, "input"),
            node.logical_input,
        )
        yield from _interface_records(
            ("plan_node", node.node_id, "output"),
            node.logical_output,
        )
    for edge in sorted(
        plan.edges,
        key=lambda value: (value.producer_id, value.consumer_id),
    ):
        prefix = ("plan_edge", edge.producer_id, edge.consumer_id)
        yield prefix
        yield from _interface_records(prefix, edge.interface)
    yield from _interface_records(("plan", "final"), plan.final_outputs)
    yield (
        "topology",
        topology.rank_count,
        topology.isomorphism_signature,
    )
    for link, edge in sorted(topology.links.items()):
        yield (
            "topology_link",
            link.src_rank,
            link.dst_rank,
            edge.max_channels,
        )
        for resource_id in edge.resource_ids:
            yield (
                "topology_link_resource",
                link.src_rank,
                link.dst_rank,
                resource_id,
            )
    for resource_id, resource in sorted(topology.shared_resources.items()):
        yield (
            "topology_resource",
            resource_id,
            resource.max_channels,
        )
        for link in sorted(resource.member_links):
            yield (
                "topology_resource_link",
                resource_id,
                link.src_rank,
                link.dst_rank,
            )
    for rank, node_id in sorted(topology.node_membership.items()):
        yield ("topology_node_membership", rank, node_id)
    for rank in sorted(topology.gateways):
        yield ("topology_gateway", rank)


def _problem_records(problems: Tuple[SolverProblem, ...]):
    for problem in sorted(problems, key=lambda value: value.node.node_id):
        node_id = problem.node.node_id
        yield (
            "problem",
            node_id,
            problem.slice_count,
            problem.slice_size_bytes,
            problem.reduction_dual,
        )
        for demand in sorted(problem.demands, key=lambda value: value.demand_id):
            demand_id = demand.demand_id
            yield (
                "demand",
                node_id,
                demand_id,
                demand.node_id,
                demand.stage_id,
                demand.root_rank,
                demand.required_leaf_rank,
                demand.logical_position,
                demand.reduction_dual,
            )
            for contributor in sorted(demand.contributors):
                yield ("demand_contributor", node_id, demand_id, contributor)
            for member_slice_id in sorted(demand.member_slice_ids):
                yield (
                    "demand_member_slice",
                    node_id,
                    demand_id,
                    member_slice_id,
                )
            for link in sorted(demand.allowed_links):
                yield (
                    "demand_allowed_link",
                    node_id,
                    demand_id,
                    link.src_rank,
                    link.dst_rank,
                )
            for link in sorted(demand.legal_links):
                yield (
                    "demand_legal_link",
                    node_id,
                    demand_id,
                    link.src_rank,
                    link.dst_rank,
                )
            for item in sorted(
                demand.forbidden_members,
                key=lambda value: (
                    value.slice_id,
                    value.src_rank,
                    value.dst_rank,
                    value.stage_id,
                ),
            ):
                yield (
                    "demand_forbidden_member",
                    node_id,
                    demand_id,
                    item.slice_id,
                    item.src_rank,
                    item.dst_rank,
                    item.stage_id,
                )
            for path in sorted(demand.candidate_paths):
                yield ("demand_candidate_path", node_id, demand_id, path)
        for edge in sorted(problem.candidate_edges):
            yield (
                "problem_candidate_edge",
                node_id,
                edge.src_rank,
                edge.dst_rank,
                edge.channel,
            )
        for demand_id in problem.infeasible_demand_ids:
            yield ("problem_infeasible_demand", node_id, demand_id)
        for restriction in problem.restrictions:
            yield ("problem_restriction", node_id, restriction)


def _template_records(templates: Tuple[SolverTemplate, ...]):
    for template in sorted(templates, key=lambda value: value.template_id):
        yield (
            "template",
            template.template_id,
            template.exact_signature,
            template.representative.node.node_id,
            template.representative.unit_id,
        )
        for demand in template.representative.demands:
            yield (
                "template_representative_demand",
                template.template_id,
                demand.demand_id,
            )
        for member in sorted(
            template.members,
            key=lambda value: (value.node_id, value.unit_id),
        ):
            yield (
                "template_member",
                template.template_id,
                member.node_id,
                member.unit_id,
            )
            for source, target in member.rank_map:
                yield (
                    "template_member_rank",
                    template.template_id,
                    member.node_id,
                    member.unit_id,
                    source,
                    target,
                )
            for source, target in member.contributor_map:
                yield (
                    "template_member_contributor",
                    template.template_id,
                    member.node_id,
                    member.unit_id,
                    source,
                    target,
                )
            for source, target in member.logical_position_map:
                yield (
                    "template_member_logical_position",
                    template.template_id,
                    member.node_id,
                    member.unit_id,
                    source,
                    target,
                )


def build_cache_signature(
    request: SolveRequest,
    problems: Tuple[SolverProblem, ...],
    templates: Tuple[SolverTemplate, ...],
) -> CacheSignature:
    if not isinstance(request, SolveRequest):
        raise SemanticError("cache signature requires a SolveRequest")
    try:
        problems = tuple(problems)
        templates = tuple(templates)
    except TypeError as error:
        raise SemanticError(
            "cache signature problems and templates must be iterable"
        ) from error
    if not all(isinstance(value, SolverProblem) for value in problems):
        raise SemanticError(
            "cache signature problems must contain SolverProblem values"
        )
    if not all(isinstance(value, SolverTemplate) for value in templates):
        raise SemanticError(
            "cache signature templates must contain SolverTemplate values"
        )
    backend_type = (
        _LEGACY_FULL_TIME_BACKEND
        if request.inputs.solver.require_proven_optimal
        else _TEMPLATE_ROUTE_BACKEND
    )
    if backend_type == _TEMPLATE_ROUTE_BACKEND and problems and not templates:
        raise SemanticError("template route cache signature requires templates")
    if backend_type == _LEGACY_FULL_TIME_BACKEND and templates:
        raise SemanticError("legacy cache signature must not contain templates")
    return CacheSignature(
        backend_type=backend_type,
        planning_mode=request.plan.planning_mode.value,
        plan_node_count=len(request.plan.nodes),
        problem_count=len(problems),
        demand_count=sum(len(problem.demands) for problem in problems),
        template_count=len(templates),
        template_member_count=sum(
            len(template.members) for template in templates
        ),
        structure_digest_sha256=_digest_records(
            _structure_records(request)
        ),
        problem_digest_sha256=_digest_records(_problem_records(problems)),
        template_digest_sha256=_digest_records(_template_records(templates)),
    )


def _collective(spec):
    return {
        "kind": spec.kind.value,
        "datatype": spec.datatype,
        "reduction_op": spec.reduction_op,
        "root": spec.root,
        "inplace": spec.inplace,
    }


def _interface(interface: StageInterface):
    return [
        {
            "rank": slot.rank,
            "offset": slot.offset,
            "contributors": sorted(contributors),
        }
        for slot, contributors in interface.values.items()
    ]


def _plan(plan):
    return {
        "collective": _collective(plan.collective),
        "rank_count": plan.rank_count,
        "slice_count": plan.slice_count,
        "planning_mode": plan.planning_mode.value,
        "planning_reason": plan.planning_reason,
        "initial_inputs": _interface(plan.initial_inputs),
        "nodes": [
            {
                "node_id": node.node_id,
                "stage_id": node.stage_id,
                "local_collective": _collective(node.local_collective),
                "communication_group": node.communication_group,
                "logical_input": _interface(node.logical_input),
                "logical_output": _interface(node.logical_output),
                "allowed_links": [
                    (key.src_rank, key.dst_rank)
                    for key in sorted(node.allowed_links)
                ],
                "shared_resource_ids": sorted(node.shared_resource_ids),
                "dual_of_node_id": node.dual_of_node_id,
            }
            for node in sorted(plan.nodes, key=lambda value: value.node_id)
        ],
        "edges": [
            {
                "producer_id": edge.producer_id,
                "consumer_id": edge.consumer_id,
                "interface": _interface(edge.interface),
            }
            for edge in sorted(
                plan.edges,
                key=lambda value: (value.producer_id, value.consumer_id),
            )
        ],
        "final_outputs": _interface(plan.final_outputs),
    }


def _forbidden(items):
    return [
        (
            item.slice_id,
            item.src_rank,
            item.dst_rank,
            item.stage_id,
        )
        for item in sorted(
            items,
            key=lambda value: (
                value.slice_id,
                value.src_rank,
                value.dst_rank,
                value.stage_id,
            ),
        )
    ]


def _overlay(overlay):
    if overlay is None:
        return None
    return {
        "overlay_id": overlay.overlay_id,
        "parent_candidate_id": overlay.parent_candidate_id,
        "channel_count": overlay.channel_count,
        "path_weights": overlay.path_weights,
        "temporary_forbidden": _forbidden(overlay.temporary_forbidden),
        "batch_size": overlay.batch_size,
        "tree_roots": overlay.tree_roots,
        "tree_edges": overlay.tree_edges,
        "lane_order": overlay.lane_order,
        "milp_parameters": overlay.milp_parameters,
        "warm_start_candidate_id": overlay.warm_start_candidate_id,
        "resolve_scope": overlay.resolve_scope,
        "hierarchy_template": overlay.hierarchy_template,
    }


def _topology_structure(topology):
    return {
        "rank_count": topology.rank_count,
        "links": [
            {
                "src_rank": key.src_rank,
                "dst_rank": key.dst_rank,
                "max_channels": edge.max_channels,
                "resource_ids": edge.resource_ids,
            }
            for key, edge in topology.links.items()
        ],
        "shared_resources": [
            {
                "resource_id": resource_id,
                "member_links": [
                    (key.src_rank, key.dst_rank)
                    for key in resource.member_links
                ],
                "max_channels": resource.max_channels,
            }
            for resource_id, resource in topology.shared_resources.items()
        ],
        "node_membership": [
            (rank, node) for rank, node in topology.node_membership.items()
        ],
        "gateways": sorted(topology.gateways),
    }


def _curve(curve):
    return {
        "alpha_us": curve.alpha_us,
        "beta_effective_us": curve.beta_effective_us,
        "invbw_us": curve.invbw_us,
        "bandwidth_bytes_per_us": [
            (concurrency, bandwidth)
            for concurrency, bandwidth in curve.bandwidth_bytes_per_us.items()
        ],
    }


def _topology_performance(topology):
    return {
        "links": [
            {
                "src_rank": key.src_rank,
                "dst_rank": key.dst_rank,
                "performance": _curve(edge.performance),
            }
            for key, edge in topology.links.items()
        ],
        "shared_resources": [
            {
                "resource_id": resource_id,
                "performance": _curve(resource.performance),
            }
            for resource_id, resource in topology.shared_resources.items()
        ],
    }


def _structural_payload(request: SolveRequest):
    inputs = request.inputs
    hyper = inputs.hyperparameters
    solver = inputs.solver
    strategies = inputs.strategies
    return {
        "collective": _collective(inputs.collective),
        "rank_count": inputs.rank_count,
        "total_size_bytes": hyper.total_size_bytes,
        "slice_count": hyper.slice_count,
        "objective_mode": hyper.objective_mode.value,
        "solver": {
            "total_solve_timeout_s": solver.total_solve_timeout_s,
            "per_model_timeout_s": solver.per_model_timeout_s,
            "mip_gap": solver.mip_gap,
            "require_proven_optimal": solver.require_proven_optimal,
            "solver_seed": solver.solver_seed,
            "max_channels": solver.max_channels,
            "max_threads_per_model": solver.max_threads_per_model,
            "max_parallel_models": solver.max_parallel_models,
        },
        "strategies": {
            "hierarchy": strategies.hierarchy,
            "symmetry": strategies.symmetry,
            "shortest_paths": strategies.shortest_paths,
            "batching": strategies.batching,
            "constructive_trees": strategies.constructive_trees,
            "milp": strategies.milp,
        },
        "stage_num": inputs.atom_constraints.stage_num,
        "forbidden_transfers": _forbidden(
            inputs.atom_constraints.forbidden_transfers
        ),
        "topology": _topology_structure(request.topology),
        "plan": _plan(request.plan),
        "overlay": _overlay(request.overlay),
        "solver_version": request.solver_version,
        "model_version": request.model_version,
    }


def _default_cache_signature(request: SolveRequest) -> CacheSignature:
    return CacheSignature(
        backend_type=(
            _LEGACY_FULL_TIME_BACKEND
            if request.inputs.solver.require_proven_optimal
            else _TEMPLATE_ROUTE_BACKEND
        ),
        planning_mode=request.plan.planning_mode.value,
    )


def structural_cache_key(
    request: SolveRequest,
    cache_signature: Optional[CacheSignature] = None,
) -> str:
    if not isinstance(request, SolveRequest):
        raise SemanticError("cache key requires a SolveRequest")
    signature = (
        _default_cache_signature(request)
        if cache_signature is None
        else cache_signature
    )
    if not isinstance(signature, CacheSignature):
        raise SemanticError("cache key signature must be a CacheSignature")
    payload = _structural_payload(request)
    payload["cache_signature"] = signature
    return sha256_json(payload)


def performance_cache_key(
    request: SolveRequest,
    cache_signature: Optional[CacheSignature] = None,
) -> str:
    if not isinstance(request, SolveRequest):
        raise SemanticError("cache key requires a SolveRequest")
    return sha256_json(
        {
            "structural_cache_key": structural_cache_key(
                request,
                cache_signature,
            ),
            "topology_performance": _topology_performance(request.topology),
            "slice_size_bytes": request.inputs.hyperparameters.slice_size_bytes,
            "environment_signature": request.environment_signature,
        }
    )


def candidate_cache_key(
    request: SolveRequest,
    cache_signature: Optional[CacheSignature] = None,
) -> str:
    return performance_cache_key(request, cache_signature)


@dataclass(frozen=True)
class CachedCandidate:
    candidate: SolveCandidate
    diagnostics: SearchDiagnostics = field(default_factory=SearchDiagnostics)

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, SolveCandidate):
            raise SemanticError("cached candidate is invalid")
        if not isinstance(self.diagnostics, SearchDiagnostics):
            raise SemanticError("cached diagnostics are invalid")


@dataclass(frozen=True)
class _CacheEntry:
    value: object
    expires_at: float
    complete: bool


class CandidateCache:
    def __init__(self) -> None:
        self._entries: Dict[str, _CacheEntry] = {}
        self._lock = RLock()

    def put(
        self,
        key: str,
        candidate: SolveCandidate,
        ttl_seconds: float,
        complete: bool,
        now: Optional[float] = None,
        diagnostics: Optional[SearchDiagnostics] = None,
    ) -> None:
        if not isinstance(key, str) or not key:
            raise SemanticError("cache key must be a non-empty string")
        if not isinstance(candidate, SolveCandidate):
            raise SemanticError("cache candidate must be a SolveCandidate")
        if (
            isinstance(ttl_seconds, bool)
            or not isinstance(ttl_seconds, (int, float))
            or not math.isfinite(float(ttl_seconds))
            or ttl_seconds <= 0
        ):
            raise SemanticError("cache ttl_seconds must be finite and positive")
        if not isinstance(complete, bool):
            raise SemanticError("cache complete must be a boolean")
        diagnostics_value = (
            SearchDiagnostics() if diagnostics is None else diagnostics
        )
        if not isinstance(diagnostics_value, SearchDiagnostics):
            raise SemanticError("cache diagnostics are invalid")
        current = time.monotonic() if now is None else float(now)
        if not math.isfinite(current):
            raise SemanticError("cache current time must be finite")
        entry = _CacheEntry(
            value=CachedCandidate(candidate, diagnostics_value),
            expires_at=current + float(ttl_seconds),
            complete=complete,
        )
        with self._lock:
            self._entries[key] = entry

    def get(
        self,
        key: str,
        now: Optional[float] = None,
    ) -> Optional[SolveCandidate]:
        entry = self.get_entry(key, now)
        return None if entry is None else entry.candidate

    def get_entry(
        self,
        key: str,
        now: Optional[float] = None,
    ) -> Optional[CachedCandidate]:
        current = time.monotonic() if now is None else float(now)
        if not math.isfinite(current):
            raise SemanticError("cache current time must be finite")
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if current >= entry.expires_at:
                del self._entries[key]
                return None
            value = entry.value
            if isinstance(value, SolveCandidate):
                value = CachedCandidate(value)
            if not isinstance(value, CachedCandidate):
                raise SemanticError("cache entry value is invalid")
            if entry.complete:
                return value
            return replace(
                value,
                candidate=replace(
                    value.candidate,
                    proven_optimal=False,
                ),
            )
