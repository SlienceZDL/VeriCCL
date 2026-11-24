"""Exact-mode ``predict`` on the native core (format VCP1, see native/src/simulate.cpp).

The caller (vericcl.performance.simulation.predict) has already resolved
service costs and the unavailable set exactly as the Python reference does;
this module encodes the remaining replay, runs it natively and rebuilds the
same Prediction record.
"""
from __future__ import annotations

from types import MappingProxyType

import numpy as np

from vericcl.errors import SemanticError
from vericcl import native

_KINDS = {"send": 0, "rrc": 1, "copy": 2, "reduce": 3, "nop": 4}
_SOURCES = ("initial", "estimated", "measured", "history", "initial_sharing_estimate")
_SOURCE_CODES = {name: index for index, name in enumerate(_SOURCES)}
_EVENT_KINDS = ("start", "startup_complete", "complete")
_CAPACITY_ERROR = "measured rate vector conflicts with resource capacity "


def _decimal(value) -> str:
    # The Python reference reads mpq(value) for ints and mpq(str(value)) for
    # floats; str() of either is the exact decimal the native parser expects.
    return str(value)


def _csr(rows):
    pointer = np.zeros(len(rows) + 1, dtype=np.int64)
    flat = []
    for index, row in enumerate(rows):
        flat.extend(row)
        pointer[index + 1] = len(flat)
    return pointer, np.asarray(flat, dtype=np.int32)


def native_predict(executable, model, operations, costs, unavailable, missing, lease=None, *, detail=True):
    """Prediction from the native replay, or None when the core is not in use."""
    if native.library() is None:
        return None
    from vericcl.performance.simulation import Prediction
    from vericcl.performance.events import ProgressEvent, ProgressSegment
    if lease is not None:
        lease.check()
    keys = sorted(operations)
    index = {key: position for position, key in enumerate(keys)}
    resource_names = sorted({resource for operation in operations.values() for resource in operation.resources}
                            | {resource for profile in model.profiles for resource in profile.resources})
    resource_id = {name: position for position, name in enumerate(resource_names)}
    flow_names = sorted({operation.flow_id for operation in operations.values() if operation.flow_id}
                        | {flow for profile in model.profiles for flow in profile.mode})
    flow_id = {name: position for position, name in enumerate(flow_names)}
    cost_names = sorted({operation.cost_key for operation in operations.values() if operation.kind != "nop"})
    cost_id = {name: position for position, name in enumerate(cost_names)}

    n = len(keys)
    kind = np.empty(n, dtype=np.int8)
    available = np.empty(n, dtype=np.int8)
    payload = np.empty(n, dtype=np.int64)
    flow = np.empty(n, dtype=np.int32)
    source = np.empty(n, dtype=np.int32)
    cost_key = np.empty(n, dtype=np.int32)
    dependencies, resources = [], []
    not_before, alpha, beta = [], [], []
    for position, key in enumerate(keys):
        operation = operations[key]
        kind[position] = _KINDS[operation.kind]
        available[position] = key not in unavailable
        payload[position] = operation.payload_bytes
        flow[position] = flow_id[operation.flow_id] if operation.flow_id else -1
        cost = costs.get(key)
        source[position] = _SOURCE_CODES[cost.source] if cost is not None else -1
        cost_key[position] = cost_id[operation.cost_key] if operation.kind != "nop" else -1
        dependencies.append([index[dependency] for dependency in operation.dependencies])
        resources.append(sorted(resource_id[resource] for resource in operation.resources))
        not_before.append(_decimal(operation.not_before_us))
        alpha.append(_decimal(cost.alpha_us) if cost is not None else "0")
        beta.append(_decimal(cost.beta_us_per_byte) if cost is not None else "1")
    capacity = [_decimal(model.resources[name]) if name in model.resources else "1" for name in resource_names]
    outputs = np.asarray([index[key] for key in executable.output_operations], dtype=np.int32)

    profiles = model.profiles
    profile_source = np.asarray([_SOURCE_CODES[profile.source] for profile in profiles], dtype=np.int32)
    profile_min = np.asarray([profile.min_bytes for profile in profiles], dtype=np.int64)
    profile_max = np.asarray([-1 if profile.max_bytes is None else profile.max_bytes for profile in profiles],
                             dtype=np.int64)
    profile_membership = np.asarray([bool(profile.flow_resources) for profile in profiles], dtype=np.int8)
    modes, profile_resources, has_startup, membership, rates, startup = [], [], [], [], [], []
    for profile in profiles:
        mode = sorted(profile.mode, key=lambda name: flow_id[name])
        modes.append([flow_id[name] for name in mode])
        profile_resources.append(sorted(resource_id[name] for name in profile.resources))
        for name in mode:
            rates.append(_decimal(profile.rates[name]))
            has_startup.append(name in profile.startup_us)
            startup.append(_decimal(profile.startup_us[name]) if name in profile.startup_us else "")
            membership.append(sorted(resource_id[resource] for resource in profile.flow_resources.get(name, ())))
    dependency_ptr, dependency_index = _csr(dependencies)
    resource_ptr, resource_index = _csr(resources)
    mode_ptr, mode_flow = _csr(modes)
    profile_resource_ptr, profile_resource = _csr(profile_resources)
    membership_ptr, membership_index = _csr(membership)
    text = "".join(field + "\n" for field in not_before + alpha + beta + capacity + rates + startup).encode("ascii")
    header = np.asarray([n, len(resource_names), len(profiles), len(outputs), len(cost_names), int(detail)],
                        dtype=np.int64)
    payload_bytes = b"".join((
        b"VCP1", header.tobytes(), kind.tobytes(), available.tobytes(), payload.tobytes(), flow.tobytes(),
        source.tobytes(), cost_key.tobytes(), dependency_ptr.tobytes(), dependency_index.tobytes(),
        resource_ptr.tobytes(), resource_index.tobytes(), outputs.tobytes(), profile_source.tobytes(),
        profile_min.tobytes(), profile_max.tobytes(), profile_membership.tobytes(), mode_ptr.tobytes(),
        mode_flow.tobytes(), profile_resource_ptr.tobytes(), profile_resource.tobytes(),
        np.asarray(has_startup, dtype=np.int8).tobytes(), membership_ptr.tobytes(), membership_index.tobytes(),
        np.asarray([len(text)], dtype=np.int64).tobytes(), text))

    status, data = native.call("vc_predict", payload_bytes)
    reader = _Reader(data)
    reader.scalar(np.int32)
    message = reader.text()
    if status == 1:
        if message.startswith(_CAPACITY_ERROR):
            message = _CAPACITY_ERROR + resource_names[int(message[len(_CAPACITY_ERROR):])]
        raise SemanticError(message)
    if status != 0:
        raise RuntimeError("native predict failed: " + message)
    count = int(reader.scalar(np.int64))
    times = reader.array(np.float64, count)
    event_kinds = reader.array(np.int8, count)
    event_ops = reader.array(np.int32, count)
    segment_count = int(reader.scalar(np.int64))
    segment_op = reader.array(np.int32, segment_count)
    segment_start = reader.array(np.float64, segment_count)
    segment_end = reader.array(np.float64, segment_count)
    segment_rate = reader.array(np.float64, segment_count)
    segment_source = reader.array(np.int32, segment_count)
    used_service = reader.array(np.int8, len(cost_names))
    used_resource = reader.array(np.int8, len(resource_names))
    used_profile = reader.array(np.int8, len(profiles))
    sources_used = reader.array(np.int8, len(_SOURCES))
    unmeasured_count = int(reader.scalar(np.int64))
    unmeasured_ptr = reader.array(np.int64, unmeasured_count + 1)
    unmeasured_flow = reader.array(np.int32, int(unmeasured_ptr[-1]))
    t_pred_float = float(reader.scalar(np.float64))
    output_ready_float = float(reader.scalar(np.float64))
    numerator, denominator = reader.text().split("/")
    if lease is not None:
        lease.check()

    events, starts, ends = [], {}, {}
    for time, code, position in zip(times.tolist(), event_kinds.tolist(), event_ops.tolist()):
        key = keys[position]
        kind_name = _EVENT_KINDS[code]
        events.append(ProgressEvent(time, kind_name, key))
        if code == 0:
            starts[key] = time
        elif code == 2:
            ends[key] = time
    segments = tuple(ProgressSegment(keys[position], start, end, rate, _SOURCES[code]) for position, start, end, rate, code
                     in zip(segment_op.tolist(), segment_start.tolist(), segment_end.tolist(),
                            segment_rate.tolist(), segment_source.tolist()))
    used = {"service:" + cost_names[i] for i in np.flatnonzero(used_service)}
    used |= {"resource:" + resource_names[i] for i in np.flatnonzero(used_resource)}
    used |= {"profile:" + profiles[i].profile_id for i in np.flatnonzero(used_profile)}
    sources = {_SOURCES[i] for i in np.flatnonzero(sources_used)}
    unmeasured = tuple(sorted(tuple(sorted(flow_names[flow] for flow in unmeasured_flow[start:end].tolist()))
                              for start, end in zip(unmeasured_ptr[:-1].tolist(), unmeasured_ptr[1:].tolist())))
    t_pred = None if unavailable else t_pred_float
    output_ready = None if set(executable.output_operations) & unavailable else output_ready_float
    coverage = ("unavailable" if unavailable else "not_required" if not sources
                else "measured" if sources <= {"measured", "history"}
                else "partial" if sources & {"measured", "history"} else "initial")
    ratio = (int(numerator), int(denominator)) if t_pred is not None else None
    return Prediction(model.version, t_pred, output_ready, MappingProxyType(starts), MappingProxyType(ends),
                      tuple(events), segments, frozenset(used), frozenset(missing), tuple(sorted(unavailable)), coverage,
                      ratio, unmeasured)


class _Reader:
    def __init__(self, data: bytes):
        self.data, self.offset = data, 0

    def scalar(self, dtype):
        return self.array(dtype, 1)[0]

    def array(self, dtype, count):
        dtype = np.dtype(dtype)
        size = dtype.itemsize * count
        if self.offset + size > len(self.data):
            raise RuntimeError("native result is truncated")
        values = np.frombuffer(self.data, dtype=dtype, count=count, offset=self.offset)
        self.offset += size
        return values

    def text(self) -> str:
        length = int(self.scalar(np.int64))
        value = self.data[self.offset:self.offset + length].decode("utf-8")
        self.offset += length
        return value
