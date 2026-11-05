"""A resumed/new native driver shares a run directory with older helper logs."""
import sys
from types import SimpleNamespace

import pytest

from vericcl.execution.context import RunContext
from vericcl.verification.online.remote import NativeProbeDriver


@pytest.mark.parametrize('label', ['stage', 'schedule-stage', 'collect'])
def test_new_driver_preserves_existing_helper_jobs_in_the_same_run(tmp_path, label):
    context = SimpleNamespace(run_context=RunContext.create(tmp_path, 'driver-reentry', 'run'))
    first_driver = NativeProbeDriver(project_root=tmp_path)
    first = first_driver._call(context, [sys.executable, '-c', "print('first')"], 5, label)
    first_bytes = first.stdout_path.read_bytes()
    assert first.status == 'completed' and first.cleanup_complete

    # Reopening a checkpoint or constructing a second component resets the
    # driver's local sequence, but retained process records must stay distinct.
    second_driver = NativeProbeDriver(project_root=tmp_path)
    second = second_driver._call(context, [sys.executable, '-c', "print('second')"], 5, label)
    assert second.status == 'completed' and second.cleanup_complete
    assert first.stdout_path != second.stdout_path
    assert first.stdout_path.read_bytes() == first_bytes == b'first\n'
    assert second.stdout_path.read_bytes() == b'second\n'
    assert first.identity.pid != second.identity.pid
