"""Build the shared C API and full-schedule driver on an explicit toolchain."""
import argparse
import hashlib
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
    p = argparse.ArgumentParser()
    for name in ('output-dir', 'cuda-home', 'msccl-build', 'mpi-bin'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--arch', type=int, required=True)
    p.add_argument('--timeout-s', type=float, required=True)
    a = p.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    output = a.output_dir.resolve()
    if output.exists():
        p.error('output already exists; select a new evidence directory')
    nvcc, mpi = a.cuda_home.resolve() / 'bin/nvcc', a.mpi_bin.resolve() / 'mpicxx'
    runtime = a.msccl_build.resolve()
    if not nvcc.is_file() or not mpi.is_file() or not (runtime / 'lib/libnccl.so').is_file():
        p.error('explicit CUDA, MPI or prepared runtime is unavailable')
    output.mkdir(parents=True)
    context = RunContext(output, task_id=uuid.uuid4().hex, run_id=output.name, redactor=Redactor(), version=2)
    report = {'compiled': False}
    outcome = RunResult(False, 'failed', stop_reason='native_api_build', failure_category='backend')
    try:
        context.initialize_directories()
        for name in ('bin', 'lib', 'include'):
            (output / 'build' / name).mkdir(parents=True)
        context.write_json('run.json', {'kind': 'native_api_build', 'root': str(output), 'source_root': str(root),
            'task_id': context.task_id, 'run_id': context.run_id,
            'invocation': {'command': 'build_native', 'argv': [sys.executable, str(Path(__file__).resolve()),
                            *(sys.argv[1:] if argv is None else argv)], 'cwd': str(Path.cwd())}})
        emit_event(context, 'run_started')
        flags = _mpi_flags(context, mpi)
        link = []
        for flag in shlex.split(flags['link']['stdout']):
            link.extend(['-Xlinker=' + part for part in flag[4:].split(',')] if flag.startswith('-Wl,') else [flag])
        common = [str(nvcc), '-std=c++17', '-O2', '-DOMPI_SKIP_MPICXX=1', '-Xcompiler=-pthread', '-Xcompiler=-fPIC',
                  f'-gencode=arch=compute_{a.arch},code=sm_{a.arch}',
                  '-I' + str(runtime / 'include'), '-I' + str(root / 'runtime/include'),
                  '-I' + str(root / 'runtime/execution'), '-I' + str(root / 'third_party/nlohmann/include')]
        sources = [str(root / 'runtime/execution' / name) for name in ('api.cc', 'schedule_store.cc', 'control_transport.cc', 'buffer_guard.cc', 'gpu_operations.cu')]
        lib = output / 'build/lib/libvericcl.so'
        library_command = [*common, '-shared', '-Xlinker=--no-undefined', *sources, '-L' + str(runtime / 'lib'), '-lnccl',
                           '-Xlinker=-rpath=' + str(runtime / 'lib'), '-o', str(lib)]
        lease = ProcessLease.after(a.timeout_s, cleanup_reserve_s=5)
        library = run_owned_process(context, library_command, lease, cwd=root, job_id='library')
        reports = {'library': library.as_json()}
        binary = output / 'build/bin/vericcl-native'
        if library.status == 'completed' and library.cleanup_complete:
            driver_command = [*common, *shlex.split(flags['compile']['stdout']), str(root / 'runtime/execution/driver.cc'),
                              '-L' + str(output / 'build/lib'), '-lvericcl', '-L' + str(runtime / 'lib'), '-lnccl',
                              '-Xlinker=-rpath=' + str(output / 'build/lib'), '-Xlinker=-rpath=' + str(runtime / 'lib'),
                              *link, '-o', str(binary)]
            driver = run_owned_process(context, driver_command, lease, cwd=root, job_id='driver')
            reports['driver'] = driver.as_json()
        compiled = len(reports) == 2 and all(item['status'] == 'completed' and item['cleanup_complete'] for item in reports.values())
        import shutil
        shutil.copy2(root / 'runtime/include/vericcl.h', output / 'build/include/vericcl.h')
        report = dict(compiled=compiled, jobs=reports, library=str(lib), binary=str(binary),
                      sha256={str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in (lib, binary) if path.is_file()})
        outcome = RunResult(report['compiled'], 'completed' if report['compiled'] else 'failed',
            stop_reason='native_api_build', failure_category=None if report['compiled'] else 'backend')
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
