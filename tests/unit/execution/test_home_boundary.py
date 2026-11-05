from pathlib import Path
from types import SimpleNamespace

import pytest


def test_owned_path_uses_current_user_home(monkeypatch):
    from vericcl.verification.online.remote import owned_home_path
    monkeypatch.setattr('pwd.getpwuid', lambda uid: SimpleNamespace(pw_dir='/home/example-account'))
    assert owned_home_path('/home/example-account/project/run') == Path('/home/example-account/project/run')
    with pytest.raises(ValueError):
        owned_home_path('/home/other-account/project/run')


def test_owned_path_checks_resolved_symlinks(tmp_path, monkeypatch):
    from vericcl.verification.online.remote import owned_home_path
    home = tmp_path / 'home'; home.mkdir()
    outside = tmp_path / 'outside'; outside.mkdir()
    (home / 'escape').symlink_to(outside, target_is_directory=True)
    (home / 'inside').mkdir()
    (home / 'alias').symlink_to(home / 'inside', target_is_directory=True)
    monkeypatch.setattr('pwd.getpwuid', lambda uid: SimpleNamespace(pw_dir=str(home)))
    assert owned_home_path(home / 'alias/data') == home / 'inside/data'
    with pytest.raises(ValueError):
        owned_home_path(home / 'escape/data')
    with pytest.raises(ValueError):
        owned_home_path('relative/path')


def test_dependency_paths_keep_system_and_own_home_but_reject_foreign_homes(monkeypatch):
    from vericcl.execution.security import is_foreign_home_path
    monkeypatch.setattr('pwd.getpwuid', lambda uid: SimpleNamespace(pw_dir='/home/example-account'))
    assert not is_foreign_home_path('/usr/local/cuda/lib64/libcudart.so')
    assert not is_foreign_home_path('/home/example-account/project/bin/native')
    assert is_foreign_home_path('/home/other-account/project/bin/native')


def test_shell_home_cannot_expand_the_account_boundary(monkeypatch):
    from vericcl.verification.online.remote import owned_home_path
    from vericcl.execution.security import is_foreign_home_path
    monkeypatch.setattr('pwd.getpwuid', lambda uid: SimpleNamespace(pw_dir='/home/example-account'))
    monkeypatch.setenv('HOME', '/')
    with pytest.raises(ValueError):
        owned_home_path('/tmp/outside')
    assert is_foreign_home_path('/home/other-account/library.so')


def test_root_directory_is_not_a_valid_account_home(monkeypatch):
    from vericcl.verification.online.remote import owned_home_path
    monkeypatch.setattr('pwd.getpwuid', lambda uid: SimpleNamespace(pw_dir='/'))
    with pytest.raises(ValueError):
        owned_home_path('/tmp/outside')


def test_account_home_symlink_remains_a_valid_dependency_root(tmp_path, monkeypatch):
    from vericcl.execution.security import is_foreign_home_path
    accounts = tmp_path / 'accounts'; accounts.mkdir()
    actual = tmp_path / 'storage/account'; actual.mkdir(parents=True)
    home = accounts / 'example-account'; home.symlink_to(actual, target_is_directory=True)
    monkeypatch.setattr('pwd.getpwuid', lambda uid: SimpleNamespace(pw_dir=str(home)))
    assert not is_foreign_home_path(home / 'VeriCCL/native')
    assert not is_foreign_home_path(actual / 'VeriCCL/native')
    assert is_foreign_home_path(accounts / 'other-account/native')
