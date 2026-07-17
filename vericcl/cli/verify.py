from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory

from vericcl.cli.overrides import (
    require_input_files,
    resolve_semantic_overrides,
)
from vericcl.cli.online import build_online_context_factory
from vericcl.cli.solve import (
    _overrides,
    default_run_id,
    emit_result_summary,
)
from vericcl.workflow import RunContext, execute_verify


def infer_schedule_sidecar(xml_path: Path) -> Path:
    path = Path(xml_path)
    name = path.name
    if name.endswith(".candidate.xml"):
        base = name[: -len(".candidate.xml")]
    elif name.endswith(".xml"):
        base = name[: -len(".xml")]
    else:
        base = name
    return path.with_name("{}.schedule.json".format(base))


def run(args: Namespace) -> int:
    sidecar = args.sidecar or infer_schedule_sidecar(args.xml)
    require_input_files(
        args.topology,
        args.sketch,
        args.atoms,
        args.xml,
        sidecar,
    )
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
        result = execute_verify(
            RunContext(
                topology_path=args.topology,
                sketch_path=effective_sketch,
                atom_path=args.atoms,
                output_base=args.output_dir,
                run_id=args.run_id or default_run_id(),
                xml_path=args.xml,
                sidecar_path=sidecar,
                online=args.online,
                tune=args.tune,
                timeout_s=args.timeout_s,
                online_context_factory=online_factory,
            )
        )
    return emit_result_summary(result, command="verify", online=args.online)
