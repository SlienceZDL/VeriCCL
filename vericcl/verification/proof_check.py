"""Independent, exact certificates for a declared finite native decision space.

The built-in grammar fixes data, paths, buffers and the listed hard dependencies.
It enumerates every topological instruction order and every admitted per-operation
channel choice. Other repair families remain separate coverage obligations.
"""
from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import product

from vericcl.errors import SemanticError
from vericcl.execution.budget import BudgetExhausted
from vericcl.performance.ratios import ratio_from_json, ratio_to_json
from vericcl.solver.proof_bounds import rational, verify_lp_dual
from vericcl.topology.model import LinkKey
from vericcl.xml.v5_lower import (check_xml_bundle, evaluate_program, _serialize_storage, _request_contract)
from vericcl.xml.v5_program import lower_program, emit_native_xml
from .search_coverage import digest, model_digest


@dataclass(frozen=True)
class NativeDecisionSpace:
    bundle: object
    context: object
    vary_order: bool = True
    vary_channels: bool = False

    @property
    def families(self):
        return tuple(name for name, enabled in (('order', self.vary_order), ('channel', self.vary_channels)) if enabled)

    @property
    def scope_id(self):
        return digest({'grammar': 'fixed_state_native_order_channels_v1',
            'storage': _serialize_storage(self.bundle.storage_plan), 'vary_order': self.vary_order,
            'vary_channels': self.vary_channels, 'contract': _request_contract(self.context.inputs),
            'channel_domains': channel_domains(self), 'tb_layout': self.context.inputs.request_config['runtime']['tb_layout'],
            'limits': vars(self.context.limits), 'input': self.context.inputs.input_sha256,
            'topology': self.context.topology.isomorphism_signature})


def channel_domains(space):
    result = []
    for op in space.bundle.storage_plan.instructions:
        if op.kind not in {'send', 'rrc'}:
            continue
        source = space.bundle.storage_plan.values[op.inputs[0]].rank
        edge = space.context.topology.links[LinkKey(source, op.rank)]
        maximum = min(space.context.inputs.solver.max_channels, edge.max_channels,
                      *(space.context.topology.shared_resources[key].max_channels for key in edge.resource_ids))
        result.append((op.operation_id, tuple(range(maximum)) if space.vary_channels else (op.channel,)))
    return tuple(result)


def _orders(operations, lease):
    dependencies = {op.operation_id: op.dependencies for op in operations}
    if any(set(deps) - dependencies.keys() for deps in dependencies.values()):
        raise SemanticError('finite proof grammar has an unknown dependency')
    def visit(prefix, remaining):
        lease.check()
        if not remaining:
            yield prefix
        else:
            ready = sorted(key for key in remaining if not (dependencies[key] & remaining))
            if not ready:
                raise SemanticError('finite proof grammar dependencies contain a cycle')
            for key in ready:
                yield from visit(prefix + (key,), remaining - {key})
    yield from visit((), frozenset(dependencies))


def decisions(space):
    storage, lease = space.bundle.storage_plan, space.context.lease
    orders = _orders(storage.instructions, lease) if space.vary_order else (tuple(op.operation_id for op in storage.instructions),)
    domains = channel_domains(space)
    for order in orders:
        for channels in product(*(domain for _, domain in domains)):
            lease.check()
            assignment = tuple(zip((key for key, _ in domains), channels))
            yield {'key': digest({'order': order, 'channels': assignment}), 'order': order, 'channels': assignment}


def replay_decision(space, decision):
    operations = {op.operation_id: op for op in space.bundle.storage_plan.instructions}
    channels = dict(decision['channels'])
    storage = replace(space.bundle.storage_plan, instructions=tuple(
        replace(operations[key], channel=channels.get(key, operations[key].channel)) for key in decision['order']))
    program = lower_program(storage, space.context)
    xml = emit_native_xml(program, space.context.inputs)
    return evaluate_program(space.bundle.source, storage, program, xml, space.context)


def _record(space, decision):
    try:
        bundle = replay_decision(space, decision)
    except BudgetExhausted:
        raise
    except SemanticError as error:
        return {'key': decision['key'], 'status': 'invalid', 'reason': str(error)}
    if bundle.can_execute:
        return {'key': decision['key'], 'status': 'valid', 'xml_sha256': bundle.xml_sha256,
                't_pred_ratio': ratio_to_json(bundle.prediction.exact_completion_ratio)}
    invalid = {key: result.message for key, result in bundle.checks.items() if result.status.value == 'invalid'}
    # Missing or inconsistent costs are unresolved modelling, not infeasibility.
    if set(invalid) <= {'prediction'}:
        return {'key': decision['key'], 'status': 'unresolved', 'checks': invalid}
    if not invalid:
        return {'key': decision['key'], 'status': 'unresolved', 'checks': {}}
    return {'key': decision['key'], 'status': 'invalid', 'checks': invalid, 'xml_sha256': bundle.xml_sha256}


def _baseline(space):
    result = check_xml_bundle(space.bundle.xml_text, space.bundle.sidecar, space.context)
    if not result.can_execute or result.prediction.exact_completion_ratio is None:
        raise SemanticError('proof needs a fully checked current-model incumbent')
    if _serialize_storage(result.storage_plan) != _serialize_storage(space.bundle.storage_plan):
        raise SemanticError('proof grammar storage differs from the independently reloaded incumbent')
    return result


def _optimistic_lp(space):
    """Build, rather than trust, a relaxation of this exact fixed-work grammar.

    Each operation pays at least its minimum possible startup and bytes divided
    by its maximum possible service rate across every size-applicable profile
    and permitted channel. Competition can only make these lower bounds slower.
    Summing them along fixed dependency paths lower-bounds complete execution.
    """
    model, storage = space.context.model, space.bundle.storage_plan
    size = space.context.inputs.hyperparameters.slice_size_bytes
    domains = dict(channel_domains(space))
    durations = {}
    for op in storage.instructions:
        space.context.lease.check()
        if op.kind in {'send', 'rrc'}:
            source = storage.values[op.inputs[0]].rank
            key = ('link:' if op.kind == 'send' else 'rrc:') + '%d:%d' % (source, op.rank)
            flows = {('%s:%d:%d:c%d' % (op.kind, source, op.rank, channel)) for channel in domains[op.operation_id]}
        else:
            key, flows = '%s:%d' % (op.kind, op.rank), set()
        cost = model.service_for(key, size)
        if cost is None:
            raise SemanticError('optimistic bound needs every fixed-work service cost')
        startup, rate = rational(cost.alpha_us), 1 / rational(cost.beta_us_per_byte)
        for profile in model.profiles:
            if profile.min_bytes <= size and (profile.max_bytes is None or size <= profile.max_bytes):
                for flow in flows.intersection(profile.mode):
                    rate = max(rate, rational(profile.rates[flow]))
                    startup = min(startup, rational(profile.startup_us.get(flow, cost.alpha_us)))
        durations[op.operation_id] = startup + Fraction(size) / rate
    longest, pending = {}, {op.operation_id: op for op in storage.instructions}
    while pending:
        ready = sorted(key for key, op in pending.items() if op.dependencies <= longest.keys())
        if not ready:
            raise SemanticError('relaxation dependencies contain a cycle')
        for key in ready:
            op = pending.pop(key)
            longest[key] = durations[key] + max((longest[dep] for dep in op.dependencies), default=Fraction(0))
    bounds = [longest[key] for key in sorted(longest)] or [Fraction(0)]
    return [[1] for _ in bounds], list(map(str, bounds)), [1]


def optimistic_bound(space):
    matrix, rhs, objective = _optimistic_lp(space)
    maximum = max(range(len(rhs)), key=lambda index: rational(rhs[index]))
    dual = [int(index == maximum) for index in range(len(rhs))]
    return verify_lp_dual(matrix=matrix, rhs=rhs, objective=objective, dual=dual,
                          model_version=space.context.model.version, scope_id=space.scope_id)


def certify_space(space, *, use_bound=True):
    baseline = _baseline(space)
    certificate = {'schema': 1, 'grammar': 'fixed_state_native_order_channels_v1', 'scope_id': space.scope_id,
        'model_version': space.context.model.version, 'model_digest': model_digest(space.context.model),
        'incumbent_xml_sha256': space.bundle.xml_sha256,
        'incumbent_ratio': ratio_to_json(baseline.prediction.exact_completion_ratio), 'leaves': [], 'bound': None,
        'generation_status': 'in_progress'}
    try:
        if use_bound:
            bound = optimistic_bound(space)
            if bound.can_prune(baseline.prediction.exact_completion_ratio,
                               model_version=space.context.model.version, scope_id=space.scope_id):
                certificate['bound'] = bound.certificate
                certificate['generation_status'] = 'complete'
                return certificate
        for decision in decisions(space):
            certificate['leaves'].append(_record(space, decision))
        certificate['generation_status'] = 'complete'
    except BudgetExhausted:
        certificate['generation_status'] = 'budget_exhausted'
    return certificate


@dataclass(frozen=True)
class ProofResult:
    status: str
    message: str
    scope_id: str
    model_version: str
    model_digest: str
    families: tuple
    checked_leaves: int = 0
    certificate_digest: str | None = None
    counterexample: dict | None = None
    full_repair_space: bool = False


def check_no_improvement(certificate, *, space):
    checked = 0
    scope_id = space.scope_id
    def result(status, message, counterexample=None):
        return ProofResult(status, message, scope_id, space.context.model.version, model_digest(space.context.model),
            space.families, checked, digest(certificate) if status == 'proven_no_improvement' else None, counterexample)
    try:
        space.context.lease.check()
        if (certificate.get('model_version') != space.context.model.version
                or certificate.get('model_digest') != model_digest(space.context.model)
                or certificate.get('scope_id') != scope_id
                or certificate.get('incumbent_xml_sha256') != space.bundle.xml_sha256):
            return result('stale_certificate', 'model, incumbent or exact decision space changed')
        if certificate.get('schema') != 1 or certificate.get('grammar') != 'fixed_state_native_order_channels_v1':
            return result('invalid_certificate', 'unknown certificate grammar')
        baseline = _baseline(space)
        incumbent = Fraction(*baseline.prediction.exact_completion_ratio)
        if ratio_from_json(certificate.get('incumbent_ratio')) != baseline.prediction.exact_completion_ratio:
            return result('invalid_certificate', 'incumbent completion differs from exact replay')
        if certificate.get('bound') is not None:
            if certificate.get('leaves'):
                return result('invalid_certificate', 'overlapping root bound and leaf claims')
            raw = certificate['bound']
            matrix, rhs, objective = _optimistic_lp(space)
            # Reconstruct the proven relaxation; checking an arbitrary supplied
            # LP dual does not establish a bound on this scheduling problem.
            for key, expected in (('matrix', matrix), ('rhs', rhs), ('objective', objective)):
                supplied = raw[key]
                canon = (lambda rows: [[str(rational(x)) for x in row] for row in rows]) if key == 'matrix' else (lambda row: [str(rational(x)) for x in row])
                if canon(supplied) != canon(expected):
                    return result('invalid_certificate', 'submitted LP is not the reconstructed fixed-work relaxation')
            if raw.get('model_version') != space.context.model.version or raw.get('scope_id') != scope_id:
                return result('stale_certificate', 'dual evidence has another model or scope')
            bound = verify_lp_dual(matrix=matrix, rhs=rhs, objective=objective, dual=raw['dual'],
                                  model_version=space.context.model.version, scope_id=scope_id)
            if bound.value < incumbent:
                return result('incomplete', 'verified lower bound does not reach incumbent')
            return result('proven_no_improvement', 'exact reconstructed relaxation closes the declared finite scope')
        records = certificate.get('leaves', [])
        by_key = {item['key']: item for item in records}
        if len(by_key) != len(records):
            return result('invalid_certificate', 'duplicate leaf claims')
        for decision in decisions(space):
            space.context.lease.check()
            submitted = by_key.pop(decision['key'], None)
            if submitted is None:
                return result('incomplete', 'a legal decision branch has no evidence')
            actual = _record(space, decision)
            checked += 1
            normalized = dict(submitted)
            if actual['status'] == 'valid':
                normalized['t_pred_ratio'] = ratio_to_json(ratio_from_json(submitted.get('t_pred_ratio')))
            if actual != normalized:
                return result('invalid_certificate', 'leaf evidence differs from independent native replay')
            if actual['status'] == 'unresolved':
                return result('incomplete', 'a leaf has missing model or validation evidence')
            if actual['status'] == 'valid' and Fraction(*ratio_from_json(actual['t_pred_ratio'])) < incumbent:
                return result('improvement_found', 'an independently replayed decision strictly improves complete time', decision)
        if by_key:
            return result('invalid_certificate', 'certificate contains leaves outside the declared grammar')
        return result('proven_no_improvement', 'all decisions in the declared finite scope were independently replayed')
    except BudgetExhausted as error:
        return result('budget_exhausted', str(error))
    except (SemanticError, KeyError, ValueError, TypeError, ZeroDivisionError) as error:
        return result('invalid_certificate', str(error))
