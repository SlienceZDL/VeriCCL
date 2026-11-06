"""Expand manual stage variable groups into directed-edge bans."""
from itertools import product

from vericcl.errors import InputValidationError


def expanded_forbidden_transfers(atom):
    """Return unique (slice, src, dst, stage) tuples before solver construction."""
    result = dict.fromkeys(tuple(item) for item in atom['forbidden_transfers'])
    manual = atom['manual']
    if manual['enabled']:
        for node in manual['nodes']:
            group = set(node['group'])
            for variables in node['atoms']:
                if not (set(variables['src_ranks']) | set(variables['dst_ranks'])) <= group:
                    raise InputValidationError('manual atom ranks must stay inside the stage group')
                for slice_id, src, dst in product(variables['slice_ids'], variables['src_ranks'], variables['dst_ranks']):
                    result[(slice_id, src, dst, node['stage_id'])] = None
    return tuple(result)
