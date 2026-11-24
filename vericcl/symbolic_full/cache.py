"""Symbolic executions reused across the correctness gate and the analysis.

An execution depends only on the XML and the request (never on the cost
model), so after a model update the analysis rebinds it instead of executing
again.
"""
from collections import OrderedDict
import threading

from .executor import execute_xml

_EXECUTIONS: 'OrderedDict[tuple, object]' = OrderedDict()
_LIMIT = 8
_LOCK = threading.Lock()


def remember(key, execution):
    with _LOCK:
        _EXECUTIONS[key] = execution
        _EXECUTIONS.move_to_end(key)
        while len(_EXECUTIONS) > _LIMIT:
            _EXECUTIONS.popitem(last=False)


def execution_for(bundle, inputs, *, backend='cudd', lease=None):
    key = (bundle.xml_sha256, inputs.input_sha256, backend)
    with _LOCK:
        hit = _EXECUTIONS.get(key)
    if hit is not None:
        return hit
    execution = execute_xml(bundle.xml_text, inputs, program=bundle.program, storage=bundle.storage_plan,
                            executable=bundle.executable, backend=backend, lease=lease)
    remember(key, execution)
    return execution


def clear():
    with _LOCK:
        _EXECUTIONS.clear()
