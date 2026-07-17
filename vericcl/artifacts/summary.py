from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence

from vericcl.artifacts.layout import RunLayout
from vericcl.artifacts.writer import CandidateArtifact
from vericcl.errors import SemanticError
from vericcl.input.models import ResolvedInput


def _relative(layout: RunLayout, path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    try:
        return str(path.relative_to(layout.root))
    except ValueError as error:
        raise SemanticError("artifact path is outside the run directory") from error


def candidate_summary(
    layout: RunLayout,
    artifact: CandidateArtifact,
) -> Mapping[str, object]:
    if not isinstance(layout, RunLayout):
        raise SemanticError("layout must be a RunLayout")
    if not isinstance(artifact, CandidateArtifact):
        raise SemanticError("artifact must be a CandidateArtifact")
    return {
        "candidate_id": artifact.candidate_id,
        "parent_candidate_id": artifact.parent_candidate_id,
        "iteration": artifact.iteration,
        "xml_path": _relative(layout, artifact.xml_path),
        "report_path": _relative(layout, artifact.report_path),
        "schedule_path": _relative(layout, artifact.schedule_path),
        "xml_sha256": artifact.xml_sha256,
        "report_sha256": artifact.report_sha256,
        "candidate_signature": artifact.candidate_signature,
        "artifact_binding_sha256": artifact.binding_sha256,
        "validation": dict(artifact.validation),
        "runtime_compatible": artifact.runtime_compatible,
        "accepted": artifact.accepted,
        "rejection_reason": artifact.rejection_reason,
        "selected_best": artifact.selected_best,
        "proven_optimal": artifact.proven_optimal,
        "restrictions": artifact.restrictions,
    }


def build_run_summary(
    *,
    mode: str,
    layout: RunLayout,
    inputs: ResolvedInput,
    candidates: Sequence[CandidateArtifact],
    final_candidate_id: Optional[str],
    final_xml: Optional[Path],
    final_report: Optional[Path],
    status: str,
    message: str,
    elapsed_s: float,
) -> Mapping[str, object]:
    if mode not in {"solve", "verify"}:
        raise SemanticError("run summary mode must be solve or verify")
    if not isinstance(inputs, ResolvedInput):
        raise SemanticError("inputs must be a ResolvedInput")
    values = tuple(candidates)
    if not all(isinstance(item, CandidateArtifact) for item in values):
        raise SemanticError("run summary candidates are invalid")
    if isinstance(elapsed_s, bool) or not isinstance(elapsed_s, (int, float)):
        raise SemanticError("run summary elapsed_s must be numeric")
    if elapsed_s < 0.0:
        raise SemanticError("run summary elapsed_s must be non-negative")
    final_selection = None
    if final_candidate_id is not None:
        selected = tuple(
            item for item in values if item.candidate_id == final_candidate_id
        )
        if len(selected) != 1:
            raise SemanticError("final candidate must identify one artifact")
        final_selection = {
            "candidate_id": final_candidate_id,
            "xml_path": _relative(layout, final_xml),
            "report_path": _relative(layout, final_report),
            "xml_sha256": selected[0].xml_sha256,
            "report_sha256": selected[0].report_sha256,
            "selected_best": selected[0].selected_best,
            "proven_optimal": selected[0].proven_optimal,
        }
    return {
        "schema_version": "1",
        "mode": mode,
        "normalized_input_sha256": inputs.input_sha256,
        "status": status,
        "message": message,
        "elapsed_s": float(elapsed_s),
        "candidates": tuple(candidate_summary(layout, item) for item in values),
        "final_selection": final_selection,
    }
