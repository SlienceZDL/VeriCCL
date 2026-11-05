"""Build and execute CPU-only native contracts with owned process evidence."""
import argparse
from pathlib import Path
import socket

from vericcl.execution.context import RunContext, finalize_run
from vericcl.execution.processes import ProcessLease, run_owned_process
from vericcl.execution.status import RunResult


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--timeout-s', type=float, default=60)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    context = RunContext.create(root, 'native-tests', args.run_id)
    cache = context.artifact('rebuildable-cache'); cache.mkdir(exist_ok=True)
    common = ['/usr/bin/g++', '-std=c++17', '-O1', '-pthread',
        '-I' + str(root / 'runtime/include'), '-I' + str(root / 'runtime/live'),
        '-I' + str(root / 'third_party/nlohmann/include')]
    lease = ProcessLease.after(args.timeout_s, cleanup_reserve_s=3)
    tests = {'live_api': [], 'recovery': [], 'operation_wait': [], 'control_transport': ['control_transport.cc'],
             'control_recovery': ['control_transport.cc'],
             'local_control': ['control_transport.cc', 'local_control.cc']}
    success = True
    for name, sources in tests.items():
        command = common + [str(root / 'tests/native' / ('test_' + name + '.cc'))]
        command += [str(root / 'runtime/live' / path) for path in sources]
        command += ['-o', str(cache / name)]
        built = run_owned_process(context, command, lease, cwd=root, job_id='build-' + name)
        if built.status != 'completed' or not built.cleanup_complete:
            success = False; break
        if name in {'control_transport', 'control_recovery'}:
            with socket.socket() as descriptor:
                descriptor.bind(('127.0.0.1', 0)); arguments = [str(descriptor.getsockname()[1])]
        elif name == 'local_control':
            arguments = [str(context.root / 'control.sock')]
        else:
            arguments = []
        result = run_owned_process(context, [str(cache / name), *arguments], lease, cwd=root, job_id='test-' + name)
        print(name, result.status, 'CPU-only')
        if result.status != 'completed' or not result.cleanup_complete:
            success = False; break
    finalized = finalize_run(context, RunResult(success, 'completed' if success else 'failed',
        stop_reason='native_cpu_contracts', failure_category=None if success else 'internal'))
    return finalized.exit_code


if __name__ == '__main__':
    raise SystemExit(main())
