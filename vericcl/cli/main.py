import argparse
from typing import Optional, Sequence

from vericcl import __version__


def _add_common_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--topology", required=True)
    parser.add_argument("--sketch", required=True)
    parser.add_argument("--atoms", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vericcl")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    solve_parser = subparsers.add_parser("solve")
    _add_common_inputs(solve_parser)

    verify_parser = subparsers.add_parser("verify")
    _add_common_inputs(verify_parser)
    verify_parser.add_argument("--xml", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    try:
        parser.parse_args(argv)
    except SystemExit as error:
        if error.code == 0:
            return 0
        raise
    return 0


def console_main() -> None:
    raise SystemExit(main())
