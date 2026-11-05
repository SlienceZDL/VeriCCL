"""A reduction consumes both the incoming value and the local accumulator."""

from vericcl.composer.compose import compose
from vericcl.planner.model import PlanDAG, PlanEdge, PlanNode
from vericcl.semantics.atom import Atom, PathStage, Schedule, Symbol, Transfer
from vericcl.topology.model import LinkKey

from tests.unit.composer.helpers import reduce_spec
from tests.unit.composer.test_compose import _candidate, _interface


def test_reduction_waits_for_receiver_input_without_other_slice_barrier():
    initial = _interface((rank, offset, {rank * 2 + offset})
                         for rank in range(3) for offset in range(2))
    partial = _interface((0, offset, {offset, 2 + offset}) for offset in range(2))
    final = _interface((0, offset, {offset, 2 + offset, 4 + offset}) for offset in range(2))
    first = PlanNode(
        node_id='local', stage_id=0, local_collective=reduce_spec(),
        communication_group=(0, 1),
        logical_input=_interface((rank, offset, {rank * 2 + offset})
                                 for rank in (0, 1) for offset in range(2)),
        logical_output=partial, allowed_links=frozenset({LinkKey(1, 0)}),
        shared_resource_ids=frozenset(),
    )
    second = PlanNode(
        node_id='inter', stage_id=1, local_collective=reduce_spec(),
        communication_group=(0, 2),
        logical_input=_interface([(slot.rank, slot.offset, members)
                                 for slot, members in partial.values.items()]
                                + [(2, offset, {4 + offset}) for offset in range(2)]),
        logical_output=final, allowed_links=frozenset({LinkKey(2, 0)}),
        shared_resource_ids=frozenset(),
    )
    plan = PlanDAG(
        collective=reduce_spec(), rank_count=3, slice_count=2, initial_inputs=initial,
        nodes=(first, second),
        edges=(PlanEdge(producer_id='local', consumer_id='inter', interface=partial),),
        final_outputs=final,
    )
    candidates = {}
    for node, source, duration in ((first, 1, 10.0), (second, 2, 1.0)):
        transfers, roots, contributors = [], {}, {}
        for offset in range(2):
            member = source * 2 + offset
            transfer_id = node.node_id + str(offset)
            atom = Atom(member, 1024, (PathStage(node.stage_id, 'REDUCE',
                        (Symbol(source, 0, 0.0),)),), 0.0, duration)
            transfers.append(Transfer(
                transfer_id, 'REDUCE', source, 0, 0, node.stage_id,
                frozenset({member}), (atom,), 0.0, duration, frozenset(),
            ))
            roots[transfer_id] = source
            contributors[transfer_id] = tuple(sorted(node.logical_output.values[
                next(slot for slot in node.logical_output.values if slot.offset == offset)]))
        schedule = Schedule(
            node.node_id, tuple(transfers), (), 3, 2, 1024,
            metadata={'path_scope': 'stage_suffix', 'path_roots': roots,
                      'tree_contributors': contributors, 'resource_slots': {}},
        )
        candidates[node.node_id] = _candidate(node.node_id, schedule)

    schedule = compose(plan, candidates)
    transfers = {item.transfer_id: item for item in schedule.transfers}
    semantic = schedule.metadata['semantic_predecessors']
    assert set(semantic['inter0']) == {'local0'}
    assert set(semantic['inter1']) == {'local1'}
    assert transfers['inter0'].st_time >= transfers['local0'].ed_time
    assert transfers['inter1'].st_time >= transfers['local1'].ed_time
    assert transfers['inter0'].ed_time < transfers['local1'].ed_time
