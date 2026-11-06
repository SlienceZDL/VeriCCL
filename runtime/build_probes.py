"""Build the native probe against an explicitly selected patched runtime."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import uuid

from vericcl.execution.context import RunContext, finalize_run
from vericcl.execution.events import emit_event
from vericcl.execution.processes import ProcessLease, run_owned_process
from vericcl.execution.security import Redactor
from vericcl.execution.status import RunResult


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--cuda-home', required=True, type=Path)
    parser.add_argument('--msccl-build', required=True, type=Path)
    parser.add_argument('--mpi-bin', required=True, type=Path)
    parser.add_argument('--arch', required=True, type=int)
    parser.add_argument('--timeout-s', required=True, type=float)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    if output.exists():
        parser.error('output already exists; use a new run directory')
    nvcc = args.cuda_home.resolve() / 'bin/nvcc'
    mpi = args.mpi_bin.resolve() / 'mpicxx'
    msccl = args.msccl_build.resolve()
    if not nvcc.is_file() or not mpi.is_file() or not (msccl / 'lib/libnccl.so').is_file():
        parser.error('explicit CUDA, MPI or patched runtime path is unavailable')
    output.mkdir(parents=True)
    context = RunContext(output, task_id=uuid.uuid4().hex, run_id=output.name, redactor=Redactor(), version=2)
    context.initialize_directories()
    (output / 'build/bin').mkdir(parents=True)
    context.write_json('run.json', {'kind': 'native_probe_build'})
    context.write_json('invocation.json', {'command': 'build_probes', 'argv': [sys.executable, __file__, *(argv or sys.argv[1:])]})
    emit_event(context, 'run_started')
    # Open MPI's wrapper publishes the actual compiler/link flags. Record the
    # command and result; avoid shell interpolation and vendor-specific guesses.
    flags = {}
    for category in ('compile', 'link'):
        command = [str(mpi), '--showme:' + category]
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        flags[category] = {'argv': command, 'returncode': result.returncode,
                           'stdout': result.stdout, 'stderr': result.stderr}
        if result.returncode:
            context.write_json('mpi-flags.json', flags)
            raise RuntimeError('the configured MPI compiler must support --showme')
    context.write_json('mpi-flags.json', flags)
    link_flags = []
    for flag in shlex.split(flags['link']['stdout']):
        if flag.startswith('-Wl,'):
            link_flags.extend('-Xlinker=' + part for part in flag[4:].split(','))
        else:
            link_flags.append(flag)
    binary = output / 'build/bin/vericcl-probe'
    command = [str(nvcc), '-std=c++17', '-O2', '-gencode=arch=compute_%d,code=sm_%d' % (args.arch, args.arch),
               '-I' + str(msccl / 'include'), '-I' + str(root / 'runtime/include'),
               *shlex.split(flags['compile']['stdout']), str(root / 'runtime/probes/main.cu'),
               str(root / 'runtime/probes/operations.cu'), '-L' + str(msccl / 'lib'), '-lnccl',
               '-Xlinker=-rpath=' + str(msccl / 'lib'), *link_flags, '-o', str(binary)]
    result = run_owned_process(context, command, ProcessLease.after(args.timeout_s, cleanup_reserve_s=5),
                               cwd=root, job_id='compile')
    report = result.as_json() | {'binary': str(binary), 'argv': command,
                                 'compiled': result.status == 'completed' and result.cleanup_complete}
    context.write_json('result.json', report)
    finalized = finalize_run(context, RunResult(report['compiled'], 'completed' if report['compiled'] else 'failed',
        stop_reason=result.status, failure_category=None if report['compiled'] else 'backend'))
    print(json.dumps(report))
    return finalized.exit_code


if __name__ == '__main__':
    raise SystemExit(main())
