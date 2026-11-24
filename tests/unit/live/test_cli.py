from types import SimpleNamespace

import pytest


def test_live_cli_has_explicit_request_count_and_session_deadline():
    from vericcl.cli.main import build_parser
    args = build_parser().parse_args(['live', '--topology', 't.json', '--sketch', 's.json',
        '--atom', 'a.json', '--runtime', 'r.json', '--case', 'live-demo',
        '--requests', '3', '--session-timeout-s', '90'])
    assert args.command == 'live' and args.requests == 3 and args.session_timeout_s == 90


def test_live_loop_never_counts_enqueue_or_partial_group_as_success():
    from vericcl.cli.live import run_requests
    class Session:
        inputs = SimpleNamespace(rank_count=2)
        def execute(self):
            return dict(logical_success=True, request_id=1, ranks=[dict(rank=0, gpu_completed=True, correct=True)])
    with pytest.raises(RuntimeError, match='complete group'):
        run_requests(Session(), 1)


def test_live_loop_rejects_duplicate_logical_request_receipts():
    from vericcl.cli.live import run_requests
    class Session:
        inputs = SimpleNamespace(rank_count=2)
        def execute(self):
            return dict(logical_success=True, request_id=1, version_id='v', ranks=[
                dict(rank=r, request_id=1, version_id='v', gpu_completed=True, correct=True) for r in range(2)])
    with pytest.raises(RuntimeError, match='request identity'):
        run_requests(Session(), 2)
