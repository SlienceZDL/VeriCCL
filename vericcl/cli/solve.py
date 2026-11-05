from __future__ import annotations

from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from vericcl.cli.overrides import (
    SemanticOverrides,
    require_input_files,
    resolve_semantic_overrides,
)
from vericcl.cli.online import build_online_context_factory
from vericcl.workflow import RunContext, execute_solve


def default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def _overrides(args: Namespace) -> SemanticOverrides:
    return SemanticOverrides(
        operator=args.operator,
        total_size_bytes=args.total_size_bytes,
        slice_size_bytes=args.slice_size_bytes,
        root=args.root,
        inplace=args.inplace,
    )


def _selected(result):
    return next(
        (
            item
            for item in result.candidates
            if item.candidate_id == result.final_candidate_id
        ),
        None,
    )


def emit_result_summary(result, *, command: str, online: bool) -> int:
    selected = _selected(result)
    if result.final_candidate_id is None:
        print(
            "VeriCCL {} failed: no semantic-valid candidate; run={}".format(
                command,
                result.layout.root,
            ),
            file=sys.stderr,
        )
        return 3
    if online and (
        selected is None or selected.validation.get("online") != "valid"
    ):
        print(
            "VeriCCL {} failed: online validation failed; run={}".format(
                command,
                result.layout.root,
            ),
            file=sys.stderr,
        )
        return 4
    runtime_compatible = (
        True if selected is None else selected.runtime_compatible
    )
    print(
        "VeriCCL {} complete: status={} runtime_compatible={} run={} final={}".format(
            command,
            result.status,
            str(runtime_compatible).lower(),
            result.layout.root,
            result.final_xml,
        )
    )
    return 0


def run(args: Namespace) -> int:
    require_input_files(args.topology, args.sketch, args.atoms)
    with TemporaryDirectory(prefix="vericcl-cli-") as temporary:
        effective_sketch = resolve_semantic_overrides(
            args.sketch,
            _overrides(args),
            allow_override=args.override_input,
            output_dir=Path(temporary),
        )
        online_factory = (
            build_online_context_factory() if args.online else None
        )
        result = execute_solve(
            RunContext(
                topology_path=args.topology,
                sketch_path=effective_sketch,
                atom_path=args.atoms,
                output_base=args.output_dir,
                run_id=args.run_id or default_run_id(),
                online=args.online,
                tune=args.tune,
                timeout_s=args.timeout_s,
                online_context_factory=online_factory,
            )
        )
    return emit_result_summary(result, command="solve", online=args.online)
