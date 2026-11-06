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
        '-I' + str(root / 'runtime/include'), '-I' + str(root / 'runtime/execution'),
        '-I' + str(root / 'third_party/nlohmann/include')]
    lease = ProcessLease.after(args.timeout_s, cleanup_reserve_s=3)
    tests = {'contract': [], 'static_schedule': ['schedule_store.cc'], 'operation_wait': [],
             'control_transport': ['control_transport.cc'],
             'execute_ownership': ['schedule_store.cc', 'control_transport.cc', 'buffer_guard.cc']}
    success = True
    for name, sources in tests.items():
        command = common + [str(root / 'tests/native' / ('test_' + name + '.cc'))]
        if name == 'execute_ownership':
            command += ['-I' + str(root / 'tests/native/cpu_device_stubs')]
        command += [str(root / 'runtime/execution' / path) for path in sources]
        command += ['-o', str(cache / name)]
        built = run_owned_process(context, command, lease, cwd=root, job_id='build-' + name)
        if built.status != 'completed' or not built.cleanup_complete:
            success = False; break
        if name == 'control_transport':
            with socket.socket() as descriptor:
                descriptor.bind(('127.0.0.1', 0)); arguments = [str(descriptor.getsockname()[1])]
        elif name == 'static_schedule':
            arguments = [str(cache / 'schedule-fixture.xml')]
        elif name == 'execute_ownership':
            arguments = [str(cache / 'execute-ownership')]
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
