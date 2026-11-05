from pathlib import Path

import pytest

from vericcl.input.json_codec import canonical_json, sha256_json
from vericcl.input.models import Hyperparameters, ObjectiveMode


pytestmark = pytest.mark.phase01


def test_canonical_json_is_order_independent():
    left = {"b": 2, "a": [1, 3]}
    right = {"a": [1, 3], "b": 2}

    assert canonical_json(left) == canonical_json(right)
    assert sha256_json(left) == sha256_json(right)


def test_canonical_json_converts_supported_python_types():
    value = {
        "dataclass": Hyperparameters(total_size_bytes=4, slice_size_bytes=1),
        "enum": ObjectiveMode.THROUGHPUT,
        "path": Path("inputs/topology.json"),
        "set": frozenset({3, 1, 2}),
        "tuple": ("a", "b"),
    }

    encoded = canonical_json(value)

    assert '"objective_mode":"auto"' in encoded
    assert '"enum":"throughput"' in encoded
    assert '"path":"inputs/topology.json"' in encoded
    assert '"set":[1,2,3]' in encoded
    assert '"tuple":["a","b"]' in encoded


def test_canonical_json_uses_ascii_escapes():
    assert canonical_json({"value": "caf\u00e9"}) == '{"value":"caf\\u00e9"}'


def test_canonical_json_rejects_unsupported_values():
    with pytest.raises(TypeError, match="unsupported canonical JSON type"):
        canonical_json(object())


def test_sha256_json_returns_lowercase_hex_digest():
    digest = sha256_json({"a": 1})

    assert len(digest) == 64
    assert digest == digest.lower()
    assert set(digest) <= set("0123456789abcdef")


@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_canonical_json_rejects_non_finite_floats(value):
    with pytest.raises(ValueError, match="finite"):
        canonical_json(value)


def test_canonical_json_rejects_non_string_mapping_keys():
    with pytest.raises(TypeError, match="mapping keys"):
        canonical_json({1: "value"})
