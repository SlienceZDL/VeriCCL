"""Build the native probe against an explicitly selected patched runtime."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import uuid

from vericcl.execution.context import RunContext, StorageFailure, finalize_run
from vericcl.execution.events import emit_event
from vericcl.execution.processes import ProcessLease, run_owned_process
from vericcl.execution.security import Redactor
from vericcl.execution.status import RunResult


def _mpi_flags(context, mpi):
    """Persist each wrapper response, including partial output on a timeout."""
    flags = {}
    for category in ('compile', 'link'):
        command = [str(mpi), '--showme:' + category]
        record = dict(argv=command, returncode=None, stdout='', stderr='', status='failed')
        try:
            response = subprocess.run(command, capture_output=True, text=True, timeout=10)
            record.update(returncode=response.returncode, stdout=response.stdout, stderr=response.stderr,
                          status='completed' if response.returncode == 0 else 'failed')
        except subprocess.TimeoutExpired as error:
            record.update(status='timeout', stdout=error.stdout or '', stderr=error.stderr or '',
                          error=str(error))
            raise
        except Exception as error:
            record['error'] = str(error)
            raise
        finally:
            for key in ('stdout', 'stderr'):
                if isinstance(record[key], bytes):
                    record[key] = record[key].decode('utf-8', errors='replace')
            flags[category] = record
            context.write_json('mpi-flags.json', flags, replace=category != 'compile')
        if response.returncode:
            raise RuntimeError('configured MPI wrapper must support --showme')
    return flags


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
    report = {'compiled': False}
    outcome = RunResult(False, 'failed', stop_reason='native_probe_build', failure_category='backend')
    try:
        context.initialize_directories()
        (output / 'build/bin').mkdir(parents=True)
        context.write_json('run.json', {'kind': 'native_probe_build'})
        context.write_json('invocation.json', {'command': 'build_probes', 'argv': [sys.executable, __file__, *(sys.argv[1:] if argv is None else argv)]})
        emit_event(context, 'run_started')
        # Open MPI's wrapper publishes the actual compiler/link flags. Record the
        # command and result; avoid shell interpolation and vendor-specific guesses.
        flags = _mpi_flags(context, mpi)
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
        outcome = RunResult(report['compiled'], 'completed' if report['compiled'] else 'failed',
            stop_reason='native_probe_build', failure_category=None if report['compiled'] else 'backend')
    except KeyboardInterrupt as error:
        report.update(compiled=False, error={'type': type(error).__name__, 'message': 'build cancelled'})
        outcome = RunResult(False, 'cancelled', stop_reason='build_cancelled', failure_category='cancelled')
    except Exception as error:
        report.update(compiled=False, error={'type': type(error).__name__, 'message': str(error)})
        outcome = RunResult(False, 'failed', stop_reason=type(error).__name__ + ': ' + str(error),
                            failure_category='storage' if isinstance(error, StorageFailure) else 'backend')
    try:
        context.write_json('result.json', report)
    except (StorageFailure, OSError) as error:
        outcome = RunResult(False, 'failed', stop_reason=str(error), failure_category='storage')
    finally:
        finalized = finalize_run(context, outcome)
    print(json.dumps(report))
    return finalized.exit_code


if __name__ == '__main__':
    raise SystemExit(main())
