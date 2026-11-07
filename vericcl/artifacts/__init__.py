"""Artifact exports, resolved lazily (PEP 562).

Run contexts and remote/solver helper processes import submodules such as
``vericcl.artifacts.layout``; eager package imports made every helper load the
complete report/verification stack. Public names are unchanged.
"""
from importlib import import_module

_EXPORTS = {
    "artifact_binding_sha256": "hashing",
    "candidate_signature": "hashing",
    "verify_artifact_binding": "hashing",
    "CandidateReport": "reports",
    "build_candidate_report": "reports",
    "build_validation_json": "reports",
    "RunLayout": "layout",
    "create_run_layout": "layout",
    "format_scale": "layout",
    "build_run_summary": "summary",
    "candidate_summary": "summary",
    "CandidateArtifact": "writer",
    "ScheduleSidecar": "writer",
    "atomic_write_bytes": "writer",
    "atomic_write_text": "writer",
    "load_schedule_sidecar": "writer",
    "read_schedule_sidecar": "writer",
    "write_candidate_artifact": "writer",
    "write_final_alias": "writer",
    "write_resolved_input": "writer",
    "write_run_summary": "writer",
}

__all__ = [
    "CandidateReport",
    "CandidateArtifact",
    "RunLayout",
    "ScheduleSidecar",
    "artifact_binding_sha256",
    "atomic_write_bytes",
    "atomic_write_text",
    "build_candidate_report",
    "build_run_summary",
    "build_validation_json",
    "candidate_summary",
    "candidate_signature",
    "create_run_layout",
    "format_scale",
    "load_schedule_sidecar",
    "read_schedule_sidecar",
    "verify_artifact_binding",
    "write_candidate_artifact",
    "write_final_alias",
    "write_resolved_input",
    "write_run_summary",
]


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is not None:
        value = getattr(import_module(__name__ + "." + module), name)
        globals()[name] = value
        return value
    if not name.startswith("__"):
        try:
            return import_module(__name__ + "." + name)
        except ModuleNotFoundError as error:
            if error.name != __name__ + "." + name:
                raise
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))


def __dir__():
    return sorted(set(globals()) | set(__all__))
