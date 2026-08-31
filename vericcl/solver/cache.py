import math
import time
from dataclasses import dataclass, replace
from threading import RLock
from typing import Dict, Optional

from vericcl.errors import SemanticError
from vericcl.input.json_codec import sha256_json
from vericcl.planner.model import StageInterface
from vericcl.solver.model import SolveCandidate, SolveRequest


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


def structural_cache_key(request: SolveRequest) -> str:
    if not isinstance(request, SolveRequest):
        raise SemanticError("cache key requires a SolveRequest")
    return sha256_json(_structural_payload(request))


def performance_cache_key(request: SolveRequest) -> str:
    if not isinstance(request, SolveRequest):
        raise SemanticError("cache key requires a SolveRequest")
    return sha256_json(
        {
            "structural_cache_key": structural_cache_key(request),
            "topology_performance": _topology_performance(request.topology),
            "slice_size_bytes": request.inputs.hyperparameters.slice_size_bytes,
            "environment_signature": request.environment_signature,
        }
    )


def candidate_cache_key(request: SolveRequest) -> str:
    return performance_cache_key(request)


@dataclass(frozen=True)
class _CacheEntry:
    candidate: SolveCandidate
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
        current = time.monotonic() if now is None else float(now)
        if not math.isfinite(current):
            raise SemanticError("cache current time must be finite")
        entry = _CacheEntry(
            candidate=candidate,
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
            if entry.complete:
                return entry.candidate
            return replace(entry.candidate, proven_optimal=False)
