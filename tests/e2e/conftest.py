import json

import pytest


@pytest.fixture(scope='session')
def solved(tmp_path_factory):
    """Solve each distinct request once per session through the public CLI.

    Several e2e modules check different properties of the same accepted XML;
    sharing the run keeps the suite fast. Tests must not modify the result.
    """
    from tests.e2e._support import solve
    cache = {}

    def get(documents):
        key = json.dumps(documents, sort_keys=True)
        if key not in cache:
            directory = tmp_path_factory.mktemp('solved')
            cache[key] = solve(directory, json.loads(key))
        return cache[key]

    return get
