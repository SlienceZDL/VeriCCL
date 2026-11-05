"""Recheckable directed-cut lower bounds across every legal routing/order/partition."""
from fractions import Fraction
from itertools import combinations

from vericcl.verification.search_coverage import model_digest


def _traffic(inputs, subset):
    ranks, size = inputs.rank_count, inputs.hyperparameters.total_size_bytes
    operator, root = inputs.collective.kind.value, inputs.collective.root
    count = len(subset)
    if not count or count == ranks:
        return Fraction(0)
    if operator == 'broadcast':
        return Fraction(size if root in subset else 0)
    if operator == 'reduce':
        return Fraction(size if root not in subset else 0)
    if operator == 'allgather':
        return Fraction(size * count)
    if operator == 'allreduce':
        return Fraction(size)
    if operator == 'alltoall':
        return Fraction(size * count * (ranks - count), ranks)
    if operator == 'reduce_scatter':
        return Fraction(size * (ranks - count), ranks)
    raise ValueError('unknown external collective')


def _capacities(inputs, model):
    capacities = {}
    for edge in inputs.request_config['topology']['directed_links']:
        key = 'link:%d:%d' % (edge['src'], edge['dst'])
        if key not in model.resources:
            return None
        capacities[key] = Fraction(str(model.resources[key]))
    # Every measured vector must explicitly declare its link membership. The
    # largest vector is an upper envelope even for unmeasured shared modes.
    for profile in model.profiles:
        if not profile.flow_resources or set(profile.flow_resources) != set(profile.mode):
            return None
        for key in capacities:
            rate = sum((Fraction(str(profile.rates[flow])) for flow in profile.mode
                        if key in profile.flow_resources[flow]), Fraction(0))
            capacities[key] = max(capacities[key], rate)
    return capacities


def _bound(inputs, model, subset):
    capacities = _capacities(inputs, model)
    if capacities is None:
        return None
    capacity = sum((capacities['link:%d:%d' % (edge['src'], edge['dst'])]
        for edge in inputs.request_config['topology']['directed_links']
        if edge['src'] in subset and edge['dst'] not in subset), Fraction(0))
    traffic = _traffic(inputs, subset)
    return traffic / capacity if capacity > 0 and traffic > 0 else None


def certify_global_cut(bundle, context):
    if not bundle.can_execute or bundle.prediction.exact_completion_ratio is None:
        return None
    target = Fraction(*bundle.prediction.exact_completion_ratio)
    for count in range(1, context.inputs.rank_count):
        for ranks in combinations(range(context.inputs.rank_count), count):
            context.lease.check()
            bound = _bound(context.inputs, context.model, frozenset(ranks))
            if bound is not None and bound == target:
                return {'schema_version': 1, 'kind': 'directed_information_cut_v1',
                    'status': 'proven_no_improvement', 'objective': 'minimize_complete_T_pred',
                    'scope': 'all legal schedules for this external request and current cost model, including S and channels',
                    'input_sha256': context.inputs.input_sha256, 'model_digest': model_digest(context.model),
                    'xml_sha256': bundle.xml_sha256, 'cut_ranks': list(ranks),
                    'lower_bound_ratio': [bound.numerator, bound.denominator]}
    return None


def check_global_cut(certificate, bundle, context):
    if not certificate or certificate.get('kind') != 'directed_information_cut_v1' or not bundle.can_execute:
        return False
    if (certificate.get('input_sha256') != context.inputs.input_sha256 or
            certificate.get('model_digest') != model_digest(context.model) or
            certificate.get('xml_sha256') != bundle.xml_sha256):
        return False
    subset = frozenset(certificate.get('cut_ranks', ()))
    if not subset or not subset < frozenset(range(context.inputs.rank_count)):
        return False
    value = _bound(context.inputs, context.model, subset)
    return (value is not None and value == Fraction(*certificate['lower_bound_ratio']) and
            value == Fraction(*bundle.prediction.exact_completion_ratio))
