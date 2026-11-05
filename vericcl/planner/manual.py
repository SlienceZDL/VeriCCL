"""Explicit manual entry; automatic synthesis never reads disabled residue."""
from vericcl.errors import InputValidationError
from vericcl.planner.hierarchy import build_manual_plan
from dataclasses import replace


def manual_enabled(inputs):
    return bool(inputs.request_config and inputs.request_config['atom']['manual']['enabled'])


def plan_manual(inputs, topology):
    if not manual_enabled(inputs):
        raise InputValidationError('manual plan entry requires manual.enabled=true')
    nodes = []
    for node in inputs.request_config['atom']['manual']['nodes']:
        def interface(key):
            values = node.get(key)
            if not values:
                raise InputValidationError('manual nodes need nonempty input_states and output_states')
            if any(len(set(value['contributors'])) != len(value['contributors']) for value in values):
                raise InputValidationError('manual state contributors must be unique')
            return tuple((value['rank'], value['offset'], tuple(value['contributors'])) for value in values)
        nodes.append({'node_id': node['id'], 'stage_id': node['stage_id'], 'operator': node['operator'],
                      'communication_group': node['group'], 'root': node.get('root'),
                      'depends_on': node['depends_on'], 'logical_input': interface('input_states'),
                      'logical_output': interface('output_states')})
    translated = replace(inputs, strategies=replace(inputs.strategies, manual_hierarchy=tuple(nodes)))
    plan = build_manual_plan(translated, topology)
    for edge in plan.edges:
        by_id = {node.node_id: node for node in plan.nodes}
        if by_id[edge.producer_id].stage_id > by_id[edge.consumer_id].stage_id:
            raise InputValidationError('manual dependency cannot move backward across logical stages')
    # Stated input/output ranks are the default per-chunk roles. A manually
    # chosen group does not silently authorize every GPU as a relay.
    checked_nodes = []
    for node in plan.nodes:
        participants = {}
        raw = next(value for value in inputs.request_config['atom']['manual']['nodes'] if value['id'] == node.node_id)
        relays = set(raw['relay_ranks'])
        if not relays <= set(node.communication_group):
            raise InputValidationError('manual relay ranks must stay inside the group')
        for contributors in node.logical_output.values.values():
            roles = relays | {slot.rank for interface in (node.logical_input, node.logical_output)
                             for slot, value in interface.values.items() if contributors & value}
            for contributor in contributors:
                participants.setdefault(contributor, set()).update(roles)
        checked_nodes.append(replace(node, participants_by_slice={key: tuple(sorted(value)) for key, value in participants.items()}))
    plan = replace(plan, nodes=tuple(checked_nodes))
    from vericcl.planner.build import _validate_stage_count
    _validate_stage_count(plan, inputs.atom_constraints.stage_num)
    return plan
