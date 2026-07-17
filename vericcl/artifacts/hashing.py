from __future__ import annotations

import hashlib
from typing import Optional

from vericcl.errors import SemanticError
from vericcl.input.json_codec import sha256_json
from vericcl.input.models import ResolvedInput
from vericcl.semantics.atom import Schedule
from vericcl.topology.model import Topology
from vericcl.tuning.model import TuningOverlay


def _digest(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise SemanticError("{} must be a SHA-256 digest".format(field))
    try:
        int(value, 16)
    except ValueError as error:
        raise SemanticError(
            "{} must be a SHA-256 digest".format(field)
        ) from error
    return value.lower()


def _performance(topology: Topology) -> tuple:
    links = tuple(
        {
            "src_rank": key.src_rank,
            "dst_rank": key.dst_rank,
            "alpha_us": edge.performance.alpha_us,
            "invbw_us": edge.performance.invbw_us,
            "bandwidth_bytes_per_us": tuple(
                edge.performance.bandwidth_bytes_per_us.items()
            ),
        }
        for key, edge in topology.links.items()
    )
    resources = tuple(
        {
            "resource_id": resource_id,
            "alpha_us": resource.performance.alpha_us,
            "invbw_us": resource.performance.invbw_us,
            "bandwidth_bytes_per_us": tuple(
                resource.performance.bandwidth_bytes_per_us.items()
            ),
        }
        for resource_id, resource in topology.shared_resources.items()
    )
    return links, resources


def _schedule_payload(schedule: Schedule) -> dict:
    return {
        "transfers": schedule.transfers,
        "final_state_ids": schedule.final_state_ids,
        "rank_count": schedule.rank_count,
        "slice_count": schedule.slice_count,
        "slice_size_bytes": schedule.slice_size_bytes,
        "metadata": schedule.metadata,
    }


def _overlay_payload(overlay: Optional[TuningOverlay]):
    if overlay is None:
        return None
    value = {
        "channel_count": overlay.channel_count,
        "path_weights": overlay.path_weights,
        "temporary_forbidden": overlay.temporary_forbidden,
        "batch_size": overlay.batch_size,
        "tree_roots": overlay.tree_roots,
        "tree_edges": overlay.tree_edges,
        "lane_order": overlay.lane_order,
        "milp_parameters": overlay.milp_parameters,
        "warm_start_candidate_id": overlay.warm_start_candidate_id,
        "resolve_scope": overlay.resolve_scope,
        "hierarchy_template": overlay.hierarchy_template,
    }
    changed = any(
        item not in (None, (), frozenset())
        for item in value.values()
    )
    return value if changed else None


def candidate_signature(
    schedule: Schedule,
    inputs: ResolvedInput,
    topology: Topology,
    overlay: Optional[TuningOverlay],
) -> str:
    if not isinstance(schedule, Schedule):
        raise SemanticError("schedule must be a Schedule")
    if not isinstance(inputs, ResolvedInput):
        raise SemanticError("inputs must be a ResolvedInput")
    if not isinstance(topology, Topology):
        raise SemanticError("topology must be a Topology")
    if overlay is not None and not isinstance(overlay, TuningOverlay):
        raise SemanticError("overlay must be a TuningOverlay or None")
    return sha256_json(
        {
            "normalized_input_sha256": inputs.input_sha256,
            "topology_isomorphism_signature": (
                topology.isomorphism_signature
            ),
            "topology_performance": _performance(topology),
            "schedule": _schedule_payload(schedule),
            "overlay": _overlay_payload(overlay),
        }
    )


def artifact_binding_sha256(
    normalized_input_sha256: str,
    schedule_signature: str,
    xml_sha256: str,
) -> str:
    input_digest = _digest(
        normalized_input_sha256,
        "normalized_input_sha256",
    )
    schedule_digest = _digest(schedule_signature, "schedule_signature")
    xml_digest = _digest(xml_sha256, "xml_sha256")
    encoded = "{}:{}:{}".format(
        input_digest,
        schedule_digest,
        xml_digest,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def verify_artifact_binding(
    binding_sha256: str,
    normalized_input_sha256: str,
    schedule_signature: str,
    xml_sha256: str,
) -> bool:
    try:
        expected = artifact_binding_sha256(
            normalized_input_sha256,
            schedule_signature,
            xml_sha256,
        )
        actual = _digest(binding_sha256, "binding_sha256")
    except SemanticError:
        return False
    return actual == expected
