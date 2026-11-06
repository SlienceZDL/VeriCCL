"""Public v5 commands use explicit inputs and one owned case run."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

from vericcl import __version__
from vericcl.errors import InputValidationError, SolverUnavailableError, VeriCCLError
from vericcl.execution.context import StorageFailure


def _inputs(parser):
    parser.add_argument('--topology', required=True, type=Path)
    parser.add_argument('--sketch', required=True, type=Path)
    parser.add_argument('--atom', '--atoms', dest='atom', required=True, type=Path)
    parser.add_argument('--runtime', required=True, type=Path)
    parser.add_argument('--project-root', type=Path, default=Path.cwd())
    parser.add_argument('--case', required=True)
    parser.add_argument('--run-id')
    parser.add_argument('--set', dest='overrides', action='append', default=[], metavar='FIELD=JSON')
    parser.add_argument('--override-input', action='store_true')


def build_parser():
    parser = argparse.ArgumentParser(prog='vericcl', description='VeriCCL collective communication synthesis, calibration, and repair')
    parser.add_argument('--version', action='version', version=__version__)
    commands = parser.add_subparsers(dest='command', required=True)
    from vericcl.cli.lifecycle import new_run, inspect_run, resume_run
    solve = commands.add_parser('solve', help='run the full synthesis and tuning loop with explicit configuration')
    _inputs(solve); solve.set_defaults(handler=new_run)
    verify = commands.add_parser('verify', help='check existing XML; explicitly enable repair with --tune')
    _inputs(verify)
    verify.add_argument('--xml', required=True, type=Path)
    verify.add_argument('--sidecar', required=True, type=Path)
    verify.add_argument('--model', type=Path)
    verify.add_argument('--tune', action='store_true')
    verify.set_defaults(handler=new_run)
    calibrate = commands.add_parser('calibrate', help='run independent measurements and publish a cost model')
    _inputs(calibrate); calibrate.set_defaults(handler=new_run)
    inspect = commands.add_parser('inspect', help='view the run summary and current result without modifying them')
    inspect.add_argument('--run', required=True, type=Path); inspect.set_defaults(handler=inspect_run)
    resume = commands.add_parser('resume', help='resume a checkpoint with the original task identity and budget')
    resume.add_argument('--run', required=True, type=Path); resume.set_defaults(handler=resume_run)
    return parser


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    obsolete = {'--online', '--timeout-s', '--output-dir'} & {item.split('=')[0] for item in argv}
    if obsolete:
        print('VeriCCL v5: explicitly configure calibration, trace_and_measure, and budgets in sketch; '
              'use --project-root, --case, and --run-id for output. Obsolete options: ' + ', '.join(sorted(obsolete)), file=sys.stderr)
        return 2
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as error:
        return int(error.code)
    try:
        args.invocation_argv = [sys.executable, '-m', 'vericcl', *argv]
        return int(args.handler(args))
    except KeyboardInterrupt:
        print('VeriCCL: task canceled.', file=sys.stderr); return 130
    except SolverUnavailableError as error:
        print('VeriCCL solver error: ' + str(error), file=sys.stderr); return 8
    except StorageFailure as error:
        print('VeriCCL storage error: ' + str(error), file=sys.stderr); return 7
    except (InputValidationError, VeriCCLError, ValueError, OSError) as error:
        print('VeriCCL input or path error: ' + str(error), file=sys.stderr); return 2
    except Exception as error:
        print('VeriCCL internal error: ' + str(error), file=sys.stderr); return 5


def console_main():
    raise SystemExit(main())
