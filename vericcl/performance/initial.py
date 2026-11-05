"""Initial alpha/beta model from explicit request components; no zero filling."""
from dataclasses import asdict

from vericcl.performance.model import CostModel, ServiceCost
from vericcl.verification.search_coverage import digest


def initial_cost_model(inputs):
    topology = inputs.request_config['topology']
    services, resources, estimates = {}, {}, {}
    for item in topology['local_costs']:
        for kind in ('copy', 'reduce'):
            if kind in item:
                services[f'{kind}:{item["rank"]}'] = (ServiceCost(**dict(item[kind])),)
    for edge in topology['directed_links']:
        suffix = '%d:%d' % (edge['src'], edge['dst'])
        link = ServiceCost(edge['alpha_us'], edge['beta_us_per_byte'])
        services['link:' + suffix] = (link,)
        resources['link:' + suffix] = 1 / link.beta_us_per_byte
        local = services.get('reduce:' + str(edge['dst']))
        if local:
            reduction = local[0]
            services['rrc:' + suffix] = (ServiceCost(link.alpha_us + reduction.alpha_us,
                link.beta_us_per_byte + reduction.beta_us_per_byte, source='estimated'),)
            estimates['rrc:' + suffix] = {'formula': 'link + receiver_local_reduce',
                'components': ['link:' + suffix, 'reduce:' + str(edge['dst'])],
                'scope': 'initial_serial_cost_estimate; replaced by measured receive-reduce-copy'}
    resources.update({item['id']: item['capacity_bytes_per_us'] for item in topology['shared_resources']})
    required = {f'copy:{rank}' for rank in range(inputs.rank_count)}
    if inputs.collective.reduction_op is not None:
        required |= {f'reduce:{rank}' for rank in range(inputs.rank_count)}
        required |= {'rrc:%d:%d' % (edge['src'], edge['dst']) for edge in topology['directed_links']}
    context = {'datatype': inputs.collective.datatype, 'reduction_op': inputs.collective.reduction_op,
               'protocol': inputs.request_config['runtime']['protocol']}
    metadata = {'origin': 'explicit_request', 'initial_estimates': estimates,
                'missing_initial_components': sorted(required - services.keys()),
                'local_cost_scope': 'costs apply to this request datatype and reduction operation'}
    version = 'initial-' + digest({'services': {key: [asdict(value) for value in values] for key, values in services.items()},
                                  'resources': resources, 'context': context, 'metadata': metadata})[:24]
    return CostModel(version, services, resources, context=context, metadata=metadata)
