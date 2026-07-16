import hashlib
import json
import math
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import PurePath
from typing import Any, Mapping


def _json_sort_key(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _to_json_native(value: object) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical JSON requires finite floating-point values")
        return value
    if isinstance(value, Enum):
        return _to_json_native(value.value)
    if isinstance(value, PurePath):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _to_json_native(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("canonical JSON mapping keys must be strings")
            result[key] = _to_json_native(item)
        return result
    if isinstance(value, (frozenset, set)):
        items = [_to_json_native(item) for item in value]
        return sorted(items, key=_json_sort_key)
    if isinstance(value, (tuple, list)):
        return [_to_json_native(item) for item in value]
    raise TypeError(
        "unsupported canonical JSON type: {}".format(type(value).__qualname__)
    )


def canonical_json(value: object) -> str:
    return json.dumps(
        _to_json_native(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def sha256_json(value: object) -> str:
    encoded = canonical_json(value).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
