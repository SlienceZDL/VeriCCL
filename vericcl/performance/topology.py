"""Scalar cost projection for initial subproblem search at a fixed slice size."""
from dataclasses import replace
from vericcl.errors import SemanticError
from vericcl.topology.model import PerformanceCurve


def link_cost_key(src, dst, *, reduction=False):
    return ('rrc:' if reduction else 'link:') + str(src) + ':' + str(dst)


def project_topology(topology, model, slice_bytes):
    links = {}
    for key, edge in topology.links.items():
        component = link_cost_key(key.src_rank, key.dst_rank)
        service = model.service_for(component, slice_bytes)
        if service is None:
            raise SemanticError('current model lacks the requested link/size component: ' + component)
        curve = PerformanceCurve(service.alpha_us, service.alpha_us + service.beta_us_per_byte, {}, service.beta_us_per_byte)
        links[key] = replace(edge, performance=curve)
    resources = {}
    for resource_id, resource in topology.shared_resources.items():
        capacity = model.resources.get(resource_id)
        if capacity is None:
            raise SemanticError('current model lacks the shared resource capacity: ' + resource_id)
        resources[resource_id] = replace(resource, capacity_bytes_per_us=capacity,
                                         performance=PerformanceCurve(0., 1 / capacity, {}, 1 / capacity))
    return replace(topology, links=links, shared_resources=resources, isomorphism_signature='')
