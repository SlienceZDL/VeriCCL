import pytest


def test_lost_result_keeps_command_identity_and_original_deadline(monkeypatch):
    from vericcl.live.controller import LiveClient
    client = LiveClient('/owned/control.sock', 'session')
    attempts = []
    def exchange(request, deadline):
        attempts.append((dict(request), deadline))
        if len(attempts) == 1:
            raise TimeoutError('injected lost result')
        return {'command_id': 1, 'result': {'logical_success': True}}
    monkeypatch.setattr(client, '_exchange', exchange)
    with pytest.raises(TimeoutError):
        client.command('execute', 1)
    with pytest.raises(RuntimeError, match='unresolved'):
        client.command('close', 1)
    assert client.command('execute', 100)['logical_success']
    assert attempts[0] == attempts[1]
    assert client.next_id == 2 and client.pending is None


def test_status_query_reconciles_a_completed_command_without_reexecution(monkeypatch):
    from vericcl.live.controller import LiveClient
    client = LiveClient('/owned/control.sock', 'session')
    def exchange(request, deadline):
        if request['action'] == 'execute':
            raise TimeoutError('injected lost acknowledgement')
        assert request['action'] == 'status' and request['command_id'] == 0
        return {'last_command_id': 1, 'last_command_reply': {'command_id': 1, 'result': {'request_id': 8}}}
    monkeypatch.setattr(client, '_exchange', exchange)
    with pytest.raises(TimeoutError):
        client.command('execute', 1)
    assert client.reconcile(1) == {'request_id': 8}
    assert client.next_id == 2


@pytest.mark.parametrize('launch_error,session_error,expected', [
    ('native rank failed', None, 'failed'),
    (None, RuntimeError('collector failed'), 'failed'),
    (None, None, 'completed'),
])
def test_live_close_preserves_execution_failure_after_resource_release(tmp_path, launch_error, session_error, expected):
    import threading
    from types import SimpleNamespace
    from vericcl.execution.context import RunContext
    from vericcl.live.controller import NativeLiveSession

    session = NativeLiveSession.__new__(NativeLiveSession)
    session.settings = {'close_cleanup_timeout_s': 1}
    session.context = RunContext.create(tmp_path, 'live-close', expected)
    session.finished = threading.Event()
    session.finished.set()
    session.cancel = threading.Event()
    session.client = None
    session.error = session_error
    session.driver = SimpleNamespace(last_evidence=dict(cleanup_confirmed=True, gpu_released=True, error=launch_error))
    session.thread = threading.Thread(target=lambda: None)
    session.thread.start()
    result = session.close()
    assert result['closed'] is True and result['gpu_released'] is True
    assert result['status'] == expected
    assert bool(result['error']) is (expected == 'failed')
