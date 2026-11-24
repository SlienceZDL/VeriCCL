"""Fixtures shared by the online calibration tests."""
import pytest


@pytest.fixture
def home_is_tmp(tmp_path, monkeypatch):
    """Treat the test's temporary directory as the executing user's home.

    The native probe driver only writes inside the executing user's home
    (``remote.owned_home_path``). pytest's temporary directories normally live
    under ``/tmp``, which that rule rejects before the behaviour under test is
    reached. Moving the boundary to ``tmp_path`` keeps the rule in force: paths
    outside ``tmp_path`` are still rejected.
    """
    from vericcl.verification.online import remote
    monkeypatch.setattr(remote, 'executing_user_home', lambda: tmp_path.resolve())
    return tmp_path
