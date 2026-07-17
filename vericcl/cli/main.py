from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional, Sequence

from vericcl import __version__
from vericcl.errors import VeriCCLError
from vericcl.semantics.collective import CollectiveKind


def _add_common_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--topology", required=True, type=Path)
    parser.add_argument("--sketch", required=True, type=Path)
    parser.add_argument("--atoms", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path.cwd())
    parser.add_argument("--run-id")


def _add_semantic_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--operator",
        choices=tuple(value.value for value in CollectiveKind),
    )
    parser.add_argument("--total-size-bytes", type=int)
    parser.add_argument("--slice-size-bytes", type=int)
    parser.add_argument("--root", type=int)
    inplace = parser.add_mutually_exclusive_group()
    inplace.add_argument(
        "--inplace",
        action="store_const",
        const=True,
        default=None,
    )
    inplace.add_argument(
        "--out-of-place",
        dest="inplace",
        action="store_const",
        const=False,
    )
    parser.add_argument("--override-input", action="store_true")


def _add_workflow_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--timeout-s", type=float)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vericcl")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    solve_parser = subparsers.add_parser("solve")
    _add_common_inputs(solve_parser)
    _add_semantic_overrides(solve_parser)
    _add_workflow_options(solve_parser)
    from vericcl.cli.solve import run as run_solve

    solve_parser.set_defaults(handler=run_solve)

    verify_parser = subparsers.add_parser("verify")
    _add_common_inputs(verify_parser)
    _add_semantic_overrides(verify_parser)
    _add_workflow_options(verify_parser)
    verify_parser.add_argument("--xml", required=True, type=Path)
    verify_parser.add_argument("--sidecar", type=Path)
    from vericcl.cli.verify import run as run_verify

    verify_parser.set_defaults(handler=run_verify)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as error:
        return int(error.code)
    try:
        return int(args.handler(args))
    except TimeoutError as error:
        code = 4 if getattr(args, "online", False) else 3
        print("VeriCCL timeout: {}".format(error), file=sys.stderr)
        return code
    except (VeriCCLError, FileExistsError, OSError) as error:
        print("VeriCCL input error: {}".format(error), file=sys.stderr)
        return 2
    except Exception as error:
        print("VeriCCL internal error: {}".format(error), file=sys.stderr)
        return 5


def console_main() -> None:
    raise SystemExit(main())
