"""在独立虚拟环境安装交付包，并从安装后的源码工具包构建与运行。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from vericcl.execution.context import RunContext, finalize_run
from vericcl.execution.processes import ProcessLease, run_owned_process
from vericcl.execution.status import RunResult


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--cuda-home', type=Path, required=True)
    parser.add_argument('--mpi-bin', type=Path, required=True)
    parser.add_argument('--arch', type=int, required=True)
    parser.add_argument('--jobs', type=int, default=8)
    parser.add_argument('--compile-runtime', action='store_true')
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    context = RunContext.create(root, 'clean-install', args.run_id)
    cache = context.artifact('rebuildable-cache'); environment = cache / 'venv'; wheels = cache / 'wheels'
    wheels.mkdir(parents=True); python = environment / 'bin/python'
    env = {'PYTHONPATH': '', 'TMPDIR': str(context.artifact('tmp')), 'PIP_CACHE_DIR': str(cache / 'pip-cache'),
           'PIP_DISABLE_PIP_VERSION_CHECK': '1'}
    results = []; success = False; error = None
    def run(name, command, *, cwd=cache, seconds=180):
        value = run_owned_process(context, [str(item) for item in command],
            ProcessLease.after(seconds, cleanup_reserve_s=min(5, seconds / 5)), cwd=cwd, environment=env, job_id=name)
        results.append({'name': name, **value.as_json()})
        if value.status != 'completed' or not value.cleanup_complete:
            raise RuntimeError(name + ' failed: ' + str(value.returncode))
        return value
    try:
        run('venv', [sys.executable, '-m', 'venv', environment])
        run('wheel', [sys.executable, '-m', 'pip', 'wheel', '--no-deps', '--no-build-isolation', '--wheel-dir', wheels, root], cwd=root)
        wheel, = wheels.glob('vericcl-*.whl')
        run('dependencies', [python, '-m', 'pip', 'install', '-r', root / 'requirements-lock/python312.txt'], seconds=300)
        run('install', [python, '-m', 'pip', 'install', '--no-deps', wheel])
        kit = environment / 'share/vericcl'
        required = ['README.md', 'requirements-lock/python312.txt', 'tools/generate_examples.py',
                    'docs/input-schema.md', 'docs/input-schemas/sketch.json', 'tests/native/test_control_recovery.cc',
                    'tools/native_live_smoke.py',
                    'examples/nccl-tests-build/input/build.py',
                    'runtime/build_runtime.py', 'runtime/build_native.py', 'runtime/live/buffer_guard.cc',
                    'runtime/include/vericcl.h', 'runtime/msccl-trace/patches/0009-vericcl-live-trace-cycle.patch',
                    'third_party/msccl/src/init.cc', 'third_party/nlohmann/include/nlohmann/json.hpp',
                    'examples/quickstart/input/allreduce-hierarchy-outofplace/sketch.json']
        missing = [name for name in required if not (kit / name).is_file()]
        context.write_json('reports/installed-files.json', {'kit': str(kit), 'missing': missing,
            'sha256': {name: hashlib.sha256((kit / name).read_bytes()).hexdigest() for name in required if (kit / name).is_file()}})
        if missing:
            raise RuntimeError('installed source kit is incomplete: ' + ', '.join(missing))
        run('inspect-installed', [python, '-c',
            'from pathlib import Path; import vericcl,sys; p=Path(vericcl.__file__).resolve(); '
            'assert p.is_relative_to(Path(sys.prefix)); print(p)'])
        run('help', [environment / 'bin/vericcl', '--help'])
        build = context.artifact('runtime-build')
        command = [python, kit / 'runtime/build_runtime.py', '--output-dir', build]
        if args.compile_runtime:
            command += ['--cuda-home', args.cuda_home, '--arch', str(args.arch), '--jobs', str(args.jobs), '--timeout-s', '600']
        else:
            command += ['--prepare-only']
        run('runtime', command, seconds=630 if args.compile_runtime else 30)
        if args.compile_runtime:
            native = context.artifact('native-build')
            run('native', [python, kit / 'runtime/build_native.py', '--output-dir', native,
                '--cuda-home', args.cuda_home, '--msccl-build', build / 'build', '--mpi-bin', args.mpi_bin,
                '--arch', str(args.arch), '--timeout-s', '180'], seconds=200)
        example = kit / 'examples/quickstart/input/allreduce-hierarchy-outofplace'
        common = ['--project-root', cache / 'project', '--case', 'installed-example']
        for name in ('topology', 'sketch', 'atom', 'runtime'):
            common += ['--' + name, example / (name + '.json')]
        run('solve', [environment / 'bin/vericcl', 'solve', *common, '--run-id', 'solve'], seconds=45)
        output = cache / 'project/examples/installed-example/output/solve'
        record = json.loads((output / 'best/accepted.json').read_text())['candidate']
        run('verify', [environment / 'bin/vericcl', 'verify', *common, '--run-id', 'verify',
                       '--xml', output / record['xml'], '--sidecar', output / record['sidecar']], seconds=30)
        context.write_json('reports/installed-solve.json', {'summary': json.loads((output / 'run-summary.json').read_text()),
            'candidate': record, 'wheel_sha256': hashlib.sha256(wheel.read_bytes()).hexdigest()})
        success = True
    except Exception as failure:
        error = str(failure)
    context.write_json('reports/clean-install.json', {'success': success, 'error': error,
        'compiled_runtime': args.compile_runtime and success, 'gpu_execution': False, 'jobs': results})
    finalized = finalize_run(context, RunResult(success, 'completed' if success else 'failed',
        stop_reason='clean_install', failure_category=None if success else 'internal', optional_failures=() if success else (error,)))
    print(json.dumps({'success': success, 'error': error, 'root': str(context.root)}))
    return finalized.exit_code


if __name__ == '__main__':
    raise SystemExit(main())
