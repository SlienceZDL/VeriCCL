"""Observed output tails and conservative causal regions, separate from T_pred.

Clock bounds are intervals. A resource component is a connected set of possible
overlaps, not a claim that every member overlaps or slows every other member.
The report prioritizes legal repair search and never certifies absence of gain.
"""
from collections import deque
import hashlib

from vericcl.errors import SemanticError
from vericcl.verification.search_coverage import digest
from vericcl.xml.endpoints import EndpointType
from vericcl.xml.trace_sidecar import build_v5_trace_sidecar


def _identity(bundle):
    sidecar = build_v5_trace_sidecar(bundle)
    executable = {'outputs': bundle.executable.output_operations,
        'operations': [dict(operation_id=op.operation_id, kind=op.kind, payload_bytes=op.payload_bytes,
            dependencies=op.dependencies, resources=op.resources, flow_id=op.flow_id,
            not_before_us=op.not_before_us) for op in bundle.executable.operations]}
    return sidecar, hashlib.sha256(sidecar.to_json_text().encode()).hexdigest(), digest(executable)


def validate_trace_causality(report, bundle):
    _, mapping_hash, executable_hash = _identity(bundle)
    if (not isinstance(report, dict) or report.get('schema_version') != 1
            or report.get('xml_sha256') != bundle.xml_sha256
            or report.get('sidecar_sha256') != mapping_hash):
        raise SemanticError('trace causal report XML or endpoint mapping differs from candidate')
    if report.get('executable_sha256') != executable_hash:
        raise SemanticError('trace causal report executable or necessary outputs differ from candidate')
    keys = {op.operation_id for op in bundle.executable.operations}
    if (set(report['operations']) != keys or set(report['priority_operations']) - keys
            or set(report['completion_tails']) - keys
            or report.get('global_no_improvement_proven') is not False):
        raise SemanticError('trace causal report operation scope is invalid')


def _bounds(trace, record, field):
    if trace.alignment is None or record.rank not in trace.alignment.transforms:
        return None
    stamp = trace.alignment.timestamp(record.rank, getattr(record.raw, field))
    return [stamp.value_us - stamp.uncertainty_us, stamp.value_us + stamp.uncertainty_us]


def _maximum(intervals):
    return [max(row[0] for row in intervals), max(row[1] for row in intervals)] if intervals else None


def _minimum(intervals):
    return [min(row[0] for row in intervals), min(row[1] for row in intervals)] if intervals else None


def _tails(bounds):
    if not bounds:
        return [], []
    latest_lower = max(row[0] for row in bounds.values())
    possible = sorted(key for key, row in bounds.items() if row[1] >= latest_lower)
    # Compare against the largest *other* upper bound without a quadratic scan
    # when clock uncertainty makes thousands of outputs possible tails.
    latest_upper = sorted(((row[1], key) for key, row in bounds.items()), reverse=True)
    definite = []
    for key in possible:
        other = latest_upper[0] if latest_upper[0][1] != key else latest_upper[1] if len(latest_upper) > 1 else None
        if other is None or bounds[key][0] >= other[0]:
            definite.append(key)
    return possible, definite


def _overlap_groups(interval_sets, operations, check):
    groups = {}
    for (resource, clock_domain), intervals in sorted(interval_sets.items()):
        check()
        ordered = sorted(intervals, key=lambda row: (row[1][0], row[0]))
        members, witnesses, bridge = [], [], None
        def finish():
            if len(members) > 1:
                identity = 'overlap-%06d' % len(groups)
                groups[identity] = dict(resource=resource, clock_domain=clock_domain,
                    operation_ids=list(members), witnesses=list(witnesses),
                    interpretation='connected endpoint call overlaps; not all pairs overlap; actual resource contention is inferred')
                for key in members:
                    operations[key]['overlap_groups'].append(identity)
        for item in ordered:
            check()
            key, start, end = item
            if bridge is None or bridge[2][1] <= start[0]:
                finish()
                members, witnesses, bridge = [key], [], item
                continue
            members.append(key)
            certain = min(bridge[2][0], end[0]) > max(bridge[1][1], start[1])
            witnesses.append(dict(left=bridge[0], right=key,
                overlap='definite' if certain else 'possible_within_clock_bounds'))
            if end[1] > bridge[2][1]:
                bridge = item
        finish()
    return groups


def _frontier(work, observed, operations, output_rows, outputs, trace, single_invocation, check):
    """Separate data-output readiness from a native postprocess's local fence."""
    unknown = sorted(key for key in outputs if output_rows[key] is None)
    rows, missing, visited = {}, set(), set()
    pending = [(key, 'data_output', None) for key in outputs]
    while pending:
        check()
        key, kind, rank = pending.pop()
        if (key, kind, rank) in visited:
            continue
        visited.add((key, kind, rank))
        if kind == 'data_output' and key.startswith('native-postscale:') and not observed[key]:
            local_rank = int(key.split(':')[1])
            pending.extend((dep, 'local_kernel_frontier', local_rank) for dep in work[key].dependencies)
            continue
        selected = ([output_rows[key]] if output_rows[key] is not None else []) if kind == 'data_output' else [
            row for row in observed[key] if rank is None or row.rank == rank]
        if selected:
            for record in selected:
                identity = (key, record.rank, record.tb_id, record.step_index)
                row = rows.setdefault(identity, dict(operation_id=key, rank=record.rank,
                    tb_id=record.tb_id, step_index=record.step_index, kinds=[],
                    end_ns=record.raw.transfer_end,
                    end_us_bounds=_bounds(trace, record, 'transfer_end') if single_invocation else None))
                if kind not in row['kinds']:
                    row['kinds'].append(kind)
        else:
            missing.add(key)
            pending.extend((dep, 'dependency_completion_frontier', rank) for dep in work[key].dependencies)
    frontier, global_rows, local_ends = set(), {}, {}
    for row in rows.values():
        key, rank = row['operation_id'], str(row['rank'])
        frontier.add(key)
        global_rows.setdefault(key, []).append(row['end_us_bounds'])
        previous = local_ends.setdefault(rank, {}).get(key, [0, 0])[0]
        local_ends[rank][key] = [max(previous, row['end_ns'])] * 2
    bounds = {}
    for key, values in global_rows.items():
        interval = _maximum(values) if all(value is not None for value in values) else None
        operations[key]['frontier_us_bounds'] = interval
        if interval is not None:
            bounds[key] = interval
    return unknown, frontier, bounds, local_ends, list(rows.values()), missing


def analyze_trace_causality(bundle, trace, *, lease=None):
    check = lease.check if lease is not None else lambda: None
    check()
    sidecar, mapping_hash, executable_hash = _identity(bundle)
    if trace.xml_sha256 != bundle.xml_sha256:
        raise SemanticError('trace XML identity differs from candidate')
    work = {op.operation_id: op for op in bundle.executable.operations}
    expected, observed = {key: set() for key in work}, {key: [] for key in work}
    for entry in sidecar.entries.values():
        expected[entry.transfer_id].add(entry.key)
    seen = set()
    for rank, records in trace.records_by_rank.items():
        check()
        for record in records:
            if (record.rank != rank or record.metadata != sidecar.entries.get(record.metadata.key)
                    or record.metadata.key in seen):
                raise SemanticError('trace endpoint mapping or record identity differs from candidate')
            seen.add(record.metadata.key)
            observed[record.transfer_id].append(record)
    invocation_ids = {record.iteration for records in observed.values() for record in records}
    single_invocation = len(invocation_ids) == 1
    from vericcl.verification.bottlenecks import mandatory_dependencies
    semantic = mandatory_dependencies(bundle.storage_plan)
    operations, interval_sets, output_rows, by_resource, unknown_resources = {}, {}, {}, {}, {}
    for key, op in work.items():
        check()
        records = observed[key]
        complete = (bool(expected[key]) and {r.metadata.key for r in records} == expected[key]
                    and len({r.iteration for r in records}) == 1)
        aligned = complete and single_invocation and all(_bounds(trace, r, 'transfer_end') is not None for r in records)
        start = _minimum([_bounds(trace, r, 'transfer_start') for r in records]) if aligned else None
        end = _maximum([_bounds(trace, r, 'transfer_end') for r in records]) if aligned else None
        output = next((r for r in records if r.endpoint_type is not EndpointType.SEND), None)
        output_rows[key] = output
        output_end = _bounds(trace, output, 'transfer_end') if output is not None and single_invocation else None
        local = [dict(rank=r.rank, tb_id=r.tb_id, step_index=r.step_index, channel=r.raw.channel,
            iteration=r.iteration, endpoint=r.endpoint_type.value, payload_bytes=r.metadata.payload_bytes,
            transfer_start_ns=r.raw.transfer_start, transfer_end_ns=r.raw.transfer_end,
            start_us_bounds=_bounds(trace, r, 'transfer_start') if single_invocation else None,
            end_us_bounds=_bounds(trace, r, 'transfer_end') if single_invocation else None,
            dependency_wait_us=(r.raw.dependency_done - r.raw.tb_reach) / 1000,
            transfer_us=(r.raw.transfer_end - r.raw.transfer_start) / 1000,
            flow_ids=list(r.flow_ids)) for r in records]
        operations[key] = dict(endpoint_complete=complete, start_us_bounds=start, end_us_bounds=end,
            output_ready_us_bounds=output_end, frontier_us_bounds=None, local_endpoints=local, flow_id=op.flow_id,
            interval_interpretation='endpoint call envelope; actual resource occupancy is not directly measured',
            payload_bytes=op.payload_bytes, resources=list(op.resources), overlap_groups=[],
            predecessors=[dict(operation_id=dep,
                kind='data_or_storage_dependency' if dep in semantic.get(key, ()) else 'execution_order',
                evidence='executable_dependency; timing alone does not establish reducible delay')
                for dep in sorted(op.dependencies)])
        for resource in op.resources:
            by_resource.setdefault(resource, []).append(key)
            if not aligned:
                unknown_resources.setdefault(resource, []).append(key)
            if aligned:
                interval_sets.setdefault((resource, 'aligned_global'), []).append((key, start, end))
            for r in records:
                domain = 'rank-%d-invocation-%d' % (r.rank, r.iteration)
                interval_sets.setdefault((resource, domain), []).append((key,
                    [r.raw.transfer_start / 1000] * 2, [r.raw.transfer_end / 1000] * 2))
    groups = _overlap_groups(interval_sets, operations, check)
    # A missing endpoint/clock gives no finite global occupancy bound. Keep this
    # uncertainty on that resource instead of assuming disjointness or deleting
    # valid observations on independent resources.
    for resource, unknown in sorted(unknown_resources.items()):
        check()
        members = by_resource[resource]
        if len(members) > 1:
            identity = 'overlap-%06d' % len(groups)
            groups[identity] = dict(resource=resource, clock_domain='unresolved_global',
                operation_ids=members, witnesses=[], uncertain_operation_ids=unknown,
                uncertainty='missing_endpoint_or_clock',
                interpretation='unknown occupancy may overlap this resource; no overlap or slowdown is proven')
            for key in members:
                operations[key]['overlap_groups'].append(identity)
    outputs = tuple(work) if bundle.executable.output_operations is None else bundle.executable.output_operations
    unknown_outputs, frontier, output_bounds, local_ends, frontier_rows, missing_frontier = _frontier(
        work, observed, operations, output_rows, outputs, trace, single_invocation, check)
    tails, definite = _tails(output_bounds)
    local_tails = {rank: _tails(ends)[0] for rank, ends in local_ends.items()}
    # An unaligned/missing region remains a possible cause. It cannot disappear
    # merely because another rank supplies a globally aligned completion time.
    seeds = set(tails) | set(unknown_outputs) | missing_frontier
    for values in local_tails.values():
        seeds.update(key for key in values if key not in output_bounds)
    distances, queue, expanded_groups = {key: 0 for key in seeds}, deque(sorted(seeds)), set()
    while queue:
        check()
        key = queue.popleft()
        neighbours = set(work[key].dependencies)
        for group in operations[key]['overlap_groups']:
            if group not in expanded_groups:
                expanded_groups.add(group)
                neighbours.update(groups[group]['operation_ids'])
        for other in sorted(neighbours):
            if other not in distances:
                distances[other] = distances[key] + 1
                queue.append(other)
    complete_outputs = (trace.global_complete and not unknown_outputs and set(output_bounds) == set(outputs))
    report = dict(schema_version=1, xml_sha256=bundle.xml_sha256, sidecar_sha256=mapping_hash,
        executable_sha256=executable_hash,
        capture_id=trace.capture_id, evidence_kind=trace.evidence_kind, status=trace.status,
        global_complete=trace.global_complete, single_invocation=single_invocation,
        global_span_us=trace.global_span_us,
        device_durations_us={str(rank): value for rank, value in trace.device_durations_us.items()},
        operations=operations, resource_overlap_groups=groups,
        necessary_output_operations=sorted(outputs), unobserved_output_operations=unknown_outputs,
        output_frontier=sorted(frontier), completion_tails=tails, definite_completion_tails=definite,
        output_frontier_observations=frontier_rows, missing_frontier_operations=sorted(missing_frontier),
        local_completion_tails=local_tails,
        completion_scope='all_necessary_outputs' if complete_outputs else
            'observed_output_frontier' if unknown_outputs else 'observed_necessary_outputs',
        priority_operations=sorted(distances, key=lambda key: (distances[key], key)),
        issues=list(trace.issues), uncertain_relations=list(trace.uncertain_relations),
        global_no_improvement_proven=False,
        interpretation='timestamps are observations; dependency/resource causes are conservative inferences; '
                       'legal complete candidate replay and configured measurement decide improvement')
    validate_trace_causality(report, bundle)
    return report
