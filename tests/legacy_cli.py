"""Test-only adapter for the retained pre-v5 workflow; not a public CLI.

These tests keep the old workflow's coverage without reintroducing removed
options into VeriCCL v5. Public v5 behavior is tested through vericcl.cli.main.
"""
import argparse
from pathlib import Path
import sys

from vericcl.errors import VeriCCLError


def build_parser():
    parser = argparse.ArgumentParser(prog='retained-workflow-test')
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('solve', 'verify'):
        command = commands.add_parser(name)
        for option in ('topology', 'sketch', 'atoms'):
            command.add_argument('--' + option, type=Path, required=True)
        command.add_argument('--output-dir', type=Path, default=Path('runs'))
        command.add_argument('--run-id')
        command.add_argument('--online', action='store_true')
        command.add_argument('--tune', action='store_true')
        command.add_argument('--timeout-s', type=float)
        command.add_argument('--override-input', action='store_true')
        command.add_argument('--operator')
        command.add_argument('--total-size-bytes', type=int)
        command.add_argument('--slice-size-bytes', type=int)
        command.add_argument('--root', type=int)
        command.add_argument('--inplace', action='store_true', default=None)
        if name == 'verify':
            command.add_argument('--xml', type=Path, required=True)
            command.add_argument('--sidecar', type=Path)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == 'solve':
            from vericcl.cli.solve import run
        else:
            from vericcl.cli.verify import run
        return run(args)
    except TimeoutError as error:
        print(str(error), file=sys.stderr)
        return 4 if args.online else 3
    except (VeriCCLError, OSError, ValueError) as error:
        print(str(error), file=sys.stderr); return 2
    except Exception as error:
        print(str(error), file=sys.stderr); return 5


if __name__ == '__main__':
    raise SystemExit(main())
