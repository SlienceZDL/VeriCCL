"""通过公开 CLI 执行案例，保存每个案例的输入、命令、输出与检查结果。"""
import argparse
import json
from pathlib import Path
import sys

from tools.generate_examples import example_cases, write_documents
from vericcl.cli.main import main as cli
from vericcl.execution.context import RunContext, finalize_run
from vericcl.execution.status import RunResult


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--case', default='v5-smoke')
    parser.add_argument('--run-prefix', required=True)
    parser.add_argument('--names', nargs='+')
    args = parser.parse_args(argv)
    cases = dict(example_cases())
    selected = args.names or list(cases)
    if set(selected) - cases.keys() or len(selected) != len(set(selected)):
        parser.error('unknown or duplicate example name')
    context = RunContext.create(args.project_root, args.case, args.run_prefix + '-index')
    context.write_json('invocation.json', {'argv': [sys.executable, '-m', 'tools.run_examples', *(argv or sys.argv[1:])]})
    records = []
    for name in selected:
        inputs = context.root.parent.parent / 'input' / name
        paths = {key: inputs / (key + '.json') for key in cases[name]}
        if not any(path.exists() for path in paths.values()):
            paths = write_documents(inputs, cases[name])
        else:
            for key, path in paths.items():
                if not path.is_file() or json.loads(path.read_text()) != cases[name][key]:
                    raise ValueError('existing example input differs: ' + str(path))
        common = ['--project-root', str(args.project_root), '--case', args.case]
        for key, path in paths.items(): common += ['--' + key, str(path)]
        run_id = args.run_prefix + '-' + name
        command = ['solve', *common, '--run-id', run_id]
        solved = cli(command)
        record = {'name': name, 'solve_argv': command, 'solve_exit': solved, 'verify_exit': None,
                  'expected': 'both commands return 0 and required checks pass', 'gpu_execution': False}
        if solved == 0:
            run = context.root.parent / run_id
            candidate = json.loads((run / 'best/accepted.json').read_text())['candidate']
            verify = ['verify', *common, '--run-id', run_id + '-check', '--xml', str(run / candidate['xml']),
                      '--sidecar', str(run / candidate['sidecar'])]
            record.update(verify_argv=verify, verify_exit=cli(verify), candidate=candidate,
                          summary=json.loads((run / 'run-summary.json').read_text()))
        records.append(record)
        context.write_json('reports/cases.json', records, replace=True)
    success = all(record['solve_exit'] == record['verify_exit'] == 0 for record in records)
    result = finalize_run(context, RunResult(success, 'completed' if success else 'failed',
        stop_reason='example_matrix', failure_category=None if success else 'invalid'))
    print(json.dumps({'cases': len(records), 'success': success, 'evidence': str(context.root)}))
    return result.exit_code


if __name__ == '__main__':
    raise SystemExit(main())
