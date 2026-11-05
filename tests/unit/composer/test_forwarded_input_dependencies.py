"""The first stage hop reads its input; later hops wait for the forwarding hop."""
from vericcl.composer.compose import _compose_schedules
from vericcl.planner.model import PlanDAG, PlanEdge, PlanNode
from vericcl.semantics.atom import Atom, PathStage, Schedule, Symbol, Transfer
from vericcl.semantics.collective import CollectiveKind, CollectiveSpec
from vericcl.topology.model import LinkKey
from tests.unit.composer.helpers import reduce_spec
from tests.unit.composer.test_compose import _interface


def test_forwarded_aggregate_preserves_data_chain_without_nonlocal_wait():
    initial = _interface((rank, 0, {rank}) for rank in range(4))
    local_input = _interface([(2, 0, {2}), (3, 0, {3})])
    local_output = _interface([(2, 0, {2, 3})])
    inter_input = _interface([(0, 0, {0}), (1, 0, {1}), (2, 0, {2, 3})])
    final = _interface([(0, 0, {0, 1, 2, 3})])
    local = PlanNode('local', 0, CollectiveSpec(CollectiveKind.REDUCE, 'float32', reduction_op='sum', root=2),
        (2, 3), local_input, local_output, frozenset({LinkKey(3, 2)}), frozenset())
    inter = PlanNode('inter', 1, reduce_spec(), (0, 1, 2), inter_input, final,
        frozenset({LinkKey(2, 1), LinkKey(1, 0)}), frozenset())
    plan = PlanDAG(reduce_spec(), 4, 1, initial, (local, inter),
        (PlanEdge('local', 'inter', local_output),), final)

    def transfer(name, stage, source, target, paths, start, predecessors=()):
        atoms = tuple(Atom(member, 1024, (PathStage(stage, 'REDUCE', tuple(Symbol(*hop) for hop in path)),),
                           start, start + 1) for member, path in sorted(paths.items()))
        return Transfer(name, 'REDUCE', source, target, 0, stage, frozenset(paths), atoms,
                        start, start + 1, frozenset(predecessors))

    first = transfer('local-input', 0, 3, 2, {3: [(3, 2, 0)]}, 0)
    forward = transfer('forward', 1, 2, 1, {2: [(2, 1, 0)], 3: [(2, 1, 0)]}, 0)
    terminal = transfer('terminal', 1, 1, 0,
        {1: [(1, 0, 1)], 2: [(2, 1, 0), (1, 0, 1)], 3: [(2, 1, 0), (1, 0, 1)]}, 1, ['forward'])
    schedules = {
        'local': Schedule('local', (first,), (), 4, 1, 1024, metadata={
            'path_scope': 'stage_suffix', 'path_roots': {'local-input': 3},
            'tree_contributors': {'local-input': (2, 3)}}),
        'inter': Schedule('inter', (forward, terminal), (), 4, 1, 1024, metadata={
            'path_scope': 'stage_suffix', 'path_roots': {'forward': 2, 'terminal': {1: 1, 2: 2, 3: 2}},
            'tree_contributors': {'forward': (0, 1, 2, 3), 'terminal': (0, 1, 2, 3)}}),
    }
    schedule = _compose_schedules(plan, schedules)
    transfers = {item.transfer_id: item for item in schedule.transfers}
    assert transfers['forward'].predecessor_ids == frozenset({'local-input'})
    assert transfers['terminal'].predecessor_ids == frozenset({'forward'})
    atom = next(item for item in transfers['terminal'].atoms if item.slice_id == 3)
    assert [(s.src_rank, s.dst_rank) for stage in atom.path for s in stage.symbols] == [(3, 2), (2, 1), (1, 0)]


def test_forwarded_send_waits_for_local_arrival_and_keeps_prior_path():
    initial = _interface((rank, 0, {rank}) for rank in range(4))
    first_input = _interface([(0, 0, {0})])
    first_output = _interface([(0, 0, {0}), (1, 0, {0})])
    second_input = _interface([(1, 0, {0})])
    second_output = _interface((rank, 0, {0}) for rank in (1, 2, 3))
    final = _interface((rank, 0, {0}) for rank in range(4))
    spec = CollectiveSpec(CollectiveKind.BROADCAST, 'float32', root=0)
    first = PlanNode('first', 0, spec, (0, 1), first_input, first_output,
                     frozenset({LinkKey(0, 1)}), frozenset())
    second = PlanNode('second', 1, CollectiveSpec(CollectiveKind.BROADCAST, 'float32', root=1),
                     (1, 2, 3), second_input, second_output,
                     frozenset({LinkKey(1, 2), LinkKey(2, 3)}), frozenset())
    plan = PlanDAG(spec, 4, 1, initial, (first, second), (PlanEdge('first', 'second', second_input),), final)

    def send(name, stage, hops, start, predecessors=()):
        atom = Atom(0, 1024, (PathStage(stage, 'SEND', tuple(Symbol(*hop) for hop in hops)),), start, start + 1)
        source, target, _ = hops[-1]
        return Transfer(name, 'SEND', source, target, 0, stage, frozenset({0}), (atom,),
                        start, start + 1, frozenset(predecessors))

    original = send('original', 0, [(0, 1, 0)], 0)
    read = send('read', 1, [(1, 2, 0)], 0)
    forwarded = send('forwarded', 1, [(1, 2, 0), (2, 3, 1)], 1, ['read'])
    schedules = {
        'first': Schedule('first', (original,), (), 4, 1, 1024, metadata={
            'path_scope': 'stage_suffix', 'path_roots': {'original': 0}, 'tree_contributors': {'original': (0,)}}),
        'second': Schedule('second', (read, forwarded), (), 4, 1, 1024, metadata={
            'path_scope': 'stage_suffix', 'path_roots': {'read': 1, 'forwarded': 1},
            'tree_contributors': {'read': (0,), 'forwarded': (0,)}}),
    }
    composed = _compose_schedules(plan, schedules)
    transfers = {item.transfer_id: item for item in composed.transfers}
    assert transfers['read'].predecessor_ids == frozenset({'original'})
    assert transfers['forwarded'].predecessor_ids == frozenset({'read'})
    assert [(s.src_rank, s.dst_rank) for stage in transfers['forwarded'].atoms[0].path
            for s in stage.symbols] == [(0, 1), (1, 2), (2, 3)]
