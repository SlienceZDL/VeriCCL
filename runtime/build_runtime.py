"""Prepare and build the pinned runtime without changing the vendored source."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

from vericcl.artifacts.distribution import verify_source_manifest
from vericcl.execution.context import RunContext, finalize_run
from vericcl.execution.events import emit_event
from vericcl.execution.processes import ProcessLease, run_owned_process
from vericcl.execution.security import Redactor
from vericcl.execution.status import RunResult


def prepare_runtime(project_root: Path, destination: Path) -> Path:
    root = Path(project_root).resolve()
    source = root / "third_party/msccl"
    target = Path(destination).resolve()
    if target == source or target.is_relative_to(source):
        raise ValueError("runtime destination must not modify the vendored source")
    if target.exists():
        raise FileExistsError(target)
    findings = verify_source_manifest(root)
    if findings:
        raise ValueError("source distribution integrity check failed: " + repr(findings))
    metadata = json.loads((root / "runtime/msccl-trace/upstream.json").read_text())
    shutil.copytree(source, target)
    shutil.copy2(root / "runtime/msccl-trace/include/vericcl_trace_format.h",
                 target / "src/include/vericcl_trace_format.h")
    shutil.copy2(root / "runtime/include/vericcl_probe.h",
                 target / "src/include/vericcl_probe.h")
    shutil.copy2(root / "runtime/include/vericcl_plan.h", target / "src/include/vericcl_plan.h")
    with (target.parent / "patch.log").open("w", encoding="utf-8") as log:
        for patch in sorted((root / "runtime/msccl-trace/patches").glob("*.patch")):
            argv = ["patch", "--batch", "--forward", "-p1", "-i", str(patch)]
            log.write(json.dumps({"argv": argv, "cwd": str(target)}) + "\n")
            log.flush()
            result = subprocess.run(argv, cwd=target, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError("runtime patch failed; see " + str(target.parent / "patch.log"))
    for relative, expected in metadata["patched_files"].items():
        actual = hashlib.sha256((target / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError("prepared runtime file mismatch: " + relative)
    return target


def _compile(source: Path, output: Path, cuda: Path, arch: int,
             jobs: int, timeout: float, context: RunContext) -> dict:
    command = ["make", "-C", str(source), "-j", str(jobs), "src.build",
               "BUILDDIR=" + str(output / "build"), "CUDA_HOME=" + str(cuda),
               "NVCC_GENCODE=-gencode=arch=compute_{0},code=sm_{0}".format(arch)]
    result = run_owned_process(context, command,
                               ProcessLease.after(timeout, cleanup_reserve_s=min(5.0, timeout / 5)),
                               cwd=source, job_id="compile")
    report = result.as_json()
    report.update(compiled=result.status == "completed" and result.cleanup_complete,
                  argv=command, source_dir=str(source), build_dir=str(output / "build"),
                  budget_overrun_s=max(0.0, result.elapsed_s - timeout))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--cuda-home", type=Path)
    parser.add_argument("--arch", type=int)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--timeout-s", type=float)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    if output.exists():
        parser.error("output directory already exists; use a new run directory")
    if args.jobs < 1:
        parser.error("jobs must be positive")
    if not args.prepare_only:
        if args.cuda_home is None or not (args.cuda_home / "bin/nvcc").is_file():
            parser.error("a CUDA directory containing bin/nvcc is required")
        if args.arch is None or args.arch < 10:
            parser.error("an explicit CUDA compute capability is required")
        if args.timeout_s is None or not math.isfinite(args.timeout_s) or args.timeout_s <= 0:
            parser.error("a finite positive build timeout is required")
    output.mkdir(parents=True)
    context = RunContext(output, task_id=uuid.uuid4().hex, run_id=output.name, redactor=Redactor(), version=2)
    context.initialize_directories()
    context.write_json('run.json', {'kind': 'runtime_build'})
    context.write_json('invocation.json', {'command': 'build_runtime', 'argv': [sys.executable, __file__, *(argv or sys.argv[1:])]})
    emit_event(context, 'run_started')
    try:
        source = prepare_runtime(root, context.artifact('rebuildable-cache/source'))
        if args.prepare_only:
            report = {"status": "prepared", "compiled": False, "source_dir": str(source)}
        else:
            report = _compile(source, output, args.cuda_home.resolve(),
                              args.arch, args.jobs, args.timeout_s, context)
    except (OSError, ValueError, RuntimeError) as error:
        report = {"status": "failed", "compiled": False, "error": str(error)}
    context.write_json('result.json', report)
    success = report['status'] == 'prepared' or bool(report.get('compiled'))
    finalized = finalize_run(context, RunResult(success, 'completed' if success else 'failed',
        stop_reason=report['status'], failure_category=None if success else 'backend'))
    print(json.dumps(report | {'evidence_complete': finalized.complete}))
    return finalized.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
