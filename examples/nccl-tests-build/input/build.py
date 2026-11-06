"""Build the pinned optional nccl-tests version in the case directory without launching GPU tests."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shlex
import sys

from vericcl.execution.context import RunContext, finalize_run
from vericcl.execution.processes import ProcessLease, run_owned_process
from vericcl.execution.status import RunResult

REVISION = '0b4c4cb99fb1381edec1f78c37230688ea1ceb26'


def add_release_receipts(source, context):
    """Only replace post-measurement cleanup; the benchmark body stays pinned."""
    path = source / 'src/common.cu'
    original = path.read_text()
    old = "  // 'cuda-memcheck --leak-check full' requires this\n  cudaDeviceReset();"
    if original.count(old) != 1:
        raise RuntimeError('pinned nccl-tests cleanup site differs')
    new = r'''  // VeriCCL: checked, process-local cleanup after all measurements.
  for (int i=0; i<nGpus*nThreads; i++) {
    CUDACHECK(cudaSetDevice(gpus[i]));
    CUDACHECK(cudaDeviceSynchronize());
    if (streams[i] != NULL) CUDACHECK(cudaStreamDestroy(streams[i]));
  }
  const char* receiptPrefix = getenv("VERICCL_NCCL_TESTS_RECEIPT_PREFIX");
  if (receiptPrefix) {
    char receiptPath[4096], temporaryPath[4096];
    int count = snprintf(receiptPath, sizeof(receiptPath), "%s%d.json", receiptPrefix, proc);
    if (count < 0 || count >= (int)sizeof(receiptPath)) exit(EXIT_FAILURE);
    count = snprintf(temporaryPath, sizeof(temporaryPath), "%s.partial", receiptPath);
    if (count < 0 || count >= (int)sizeof(temporaryPath)) exit(EXIT_FAILURE);
    FILE* receipt = fopen(temporaryPath, "wx");
    if (!receipt) exit(EXIT_FAILURE);
    int written = fprintf(receipt, "{\"rank\":%d,\"world_size\":%d,\"collective_ranks\":%d,\"gpu_released\":true,\"numerical_errors\":%d,\"cleanup\":\"checked_free_comm_stream_sync\"}\n", proc, totalProcs, ncclProcs*nGpus*nThreads, errors[0]);
    int flushed = fflush(receipt);
    int synced = fsync(fileno(receipt));
    int closed = fclose(receipt);
    if (written < 0 || flushed != 0 || synced != 0 || closed != 0) exit(EXIT_FAILURE);
    if (rename(temporaryPath, receiptPath) != 0) exit(EXIT_FAILURE);
  }'''
    updated = '#include <unistd.h>\n' + original.replace(old, new)
    path.write_text(updated)
    context.write_bytes('input_snapshot/release-receipt.patch', ''.join(difflib.unified_diff(
        original.splitlines(keepends=True), updated.splitlines(keepends=True),
        fromfile='a/src/common.cu', tofile='b/src/common.cu')).encode())
    context.write_json('reports/release-receipt-patch.json', {
        'source_commit': REVISION, 'original_sha256': hashlib.sha256(original.encode()).hexdigest(),
        'patched_sha256': hashlib.sha256(updated.encode()).hexdigest(),
        'measurement_body_changed': False,
        'scope': 'checked process-local cleanup and optional per-rank receipt after MPI_Finalize'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--cuda-home', type=Path, required=True)
    parser.add_argument('--mpi-home', type=Path, required=True)
    parser.add_argument('--msccl-build', type=Path, required=True)
    parser.add_argument('--arch', type=int, required=True)
    parser.add_argument('--jobs', type=int, default=8)
    parser.add_argument('--source-repository', type=Path, help='optional local source repository; the pinned commit is still checked')
    parser.add_argument('--release-receipts', action='store_true', help='add per-rank cleanup receipts after timing and preserve the source patch')
    args = parser.parse_args()
    if args.arch <= 0 or args.jobs <= 0:
        parser.error('arch and jobs must be positive')
    root = Path(__file__).resolve().parents[3]
    context = RunContext.create(root, 'nccl-tests-build', args.run_id)
    context.write_json('invocation.json', {'argv': [sys.executable, *sys.argv]})
    source = context.artifact('rebuildable-cache/source')
    results = []
    error = None
    environment = {'CUDA_HOME': str(args.cuda_home.resolve()),
                   'LD_LIBRARY_PATH': str(args.msccl_build.resolve() / 'lib') + ':' +
                       str(args.cuda_home.resolve() / 'lib64') + ':' + str(args.mpi_home.resolve() / 'lib'),
                   'TMPDIR': str(context.artifact('tmp'))}

    def run(name, argv, *, cwd=root, seconds=180):
        result = run_owned_process(context, list(map(str, argv)), ProcessLease.after(seconds, cleanup_reserve_s=5),
                                   cwd=cwd, environment=environment, job_id=name)
        results.append({'name': name, **result.as_json()})
        if result.status != 'completed' or not result.cleanup_complete:
            raise RuntimeError(name + ' failed; see jobs/' + name)
        return context.artifact('jobs/' + name + '/stdout.log').read_text()

    try:
        source_repository = str(args.source_repository.resolve()) if args.source_repository else 'https://github.com/NVIDIA/nccl-tests.git'
        run('source', ['git', 'clone', '--depth', '1', '--branch', 'v2.13.6',
                      source_repository, source], seconds=120)
        actual = run('revision', ['git', 'rev-parse', 'HEAD'], cwd=source).strip()
        if actual != REVISION:
            raise RuntimeError('source tag no longer matches pinned commit')
        if args.release_receipts:
            add_release_receipts(source, context)
        mpi = args.mpi_home.resolve() / 'bin/mpicxx'
        compile_flags = shlex.split(run('mpi-compile', [mpi, '--showme:compile']))
        link_flags = []
        for flag in shlex.split(run('mpi-link', [mpi, '--showme:link'])):
            link_flags.extend('-Xlinker=' + part for part in flag[4:].split(',')) if flag.startswith('-Wl,') else link_flags.append(flag)
        # Distribution MPI packages need not put headers in <prefix>/include.
        # Use the wrapper's actual flags, preserving the pinned Makefile's
        # CUDA/NCCL flags when overriding its generic MPI_HOME assumption.
        cuda_flags = ['-std=c++11', '-O3', '-g', '-DMPI_SUPPORT', '-DOMPI_SKIP_MPICXX=1',
            '-gencode=arch=compute_%d,code=sm_%d' % (args.arch, args.arch),
            '-I' + str(args.msccl_build.resolve() / 'include'), *compile_flags]
        libraries = ['-L' + str(args.cuda_home.resolve() / 'lib64'), '-lcudart', '-lrt',
            '-L' + str(args.msccl_build.resolve() / 'lib'), '-lnccl', *link_flags]
        run('build', ['make', '-j' + str(args.jobs), 'MPI=1', 'MPI_HOME=' + str(args.mpi_home.resolve()),
                     'CUDA_HOME=' + str(args.cuda_home.resolve()), 'NCCL_HOME=' + str(args.msccl_build.resolve()),
                     'NVCC_GENCODE=-gencode=arch=compute_%d,code=sm_%d' % (args.arch, args.arch),
                     'NVCUFLAGS=' + shlex.join(cuda_flags), 'NVLDFLAGS=' + shlex.join(libraries)],
            cwd=source, seconds=300)
        binaries = sorted((source / 'build').glob('*_perf'))
        if not binaries:
            raise RuntimeError('build did not produce test executables')
        for binary in binaries:
            if 'not found' in run('link-' + binary.stem, ['ldd', binary]):
                raise RuntimeError(binary.name + ' has an unresolved dynamic library')
        context.write_json('reports/binaries.json', {path.name: {
            'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for path in binaries})
    except Exception as failure:
        error = str(failure)
    context.write_json('reports/build.json', {'success': error is None, 'source_commit': REVISION,
        'source': str(source), 'gpu_execution': False, 'error': error, 'jobs': results})
    result = finalize_run(context, RunResult(error is None, 'completed' if error is None else 'failed',
        stop_reason='nccl_tests_build', failure_category=None if error is None else 'internal',
        optional_failures=() if error is None else (error,)))
    print(json.dumps({'success': error is None, 'source': str(source), 'error': error}))
    return result.exit_code


if __name__ == '__main__':
    raise SystemExit(main())
