import pytest

from vericcl.cli.main import build_parser, main


pytestmark = pytest.mark.phase01


def test_parser_exposes_solve_and_verify():
    parser = build_parser()
    solve_args = parser.parse_args(
        [
            "solve",
            "--topology",
            "t.json",
            "--sketch",
            "s.json",
            "--atoms",
            "a.json",
        ]
    )
    verify_args = parser.parse_args(
        [
            "verify",
            "--topology",
            "t.json",
            "--sketch",
            "s.json",
            "--atoms",
            "a.json",
            "--xml",
            "x.xml",
        ]
    )

    assert solve_args.command == "solve"
    assert verify_args.command == "verify"


def test_main_returns_zero_for_version(capsys):
    assert main(["--version"]) == 0
    assert capsys.readouterr().out.strip()
