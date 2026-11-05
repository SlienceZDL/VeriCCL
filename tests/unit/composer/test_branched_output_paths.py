"""A completed reduction retains each contributor's path to the output rank."""
import pytest

from vericcl.composer.compose import _compose_schedules
from vericcl.planner.model import PlanDAG, PlanEdge, PlanNode
from vericcl.semantics.atom import Atom, PathStage, Schedule, Symbol, Transfer
from vericcl.semantics.collective import CollectiveKind, CollectiveSpec
from vericcl.topology.model import LinkKey
from tests.unit.composer.helpers import reduce_spec
from tests.unit.composer.test_compose import _interface


@pytest.mark.parametrize('child_id, arrival_id', [('a-child', 'z-arrival'), ('z-child', 'a-arrival')])
def test_branched_reduction_passes_complete_paths_to_distribution(child_id, arrival_id):
    initial = _interface((rank, 0, {rank}) for rank in range(4))
    reduced = _interface([(0, 0, set(range(4)))])
    final = _interface((rank, 0, set(range(4))) for rank in range(4))
    reduction = PlanNode('reduce', 0, reduce_spec(), (0, 1, 2, 3), initial, reduced,
                         frozenset({LinkKey(3, 1), LinkKey(1, 0), LinkKey(2, 0)}), frozenset())
    distribution = PlanNode('distribute', 1,
        CollectiveSpec(CollectiveKind.BROADCAST, 'float32', root=0), (0, 1, 2, 3), reduced, final,
        frozenset(LinkKey(0, rank) for rank in (1, 2, 3)), frozenset())
    plan = PlanDAG(CollectiveSpec(CollectiveKind.ALL_REDUCE, 'float32', reduction_op='sum'),
        4, 1, initial, (reduction, distribution), (PlanEdge('reduce', 'distribute', reduced),), final)

    def transfer(name, stage, source, target, paths, start, predecessors=()):
        kind = 'REDUCE' if stage == 0 else 'SEND'
        atoms = tuple(Atom(member, 1024, (PathStage(stage, kind, tuple(Symbol(*edge) for edge in path)),),
                           start, start + 1) for member, path in sorted(paths.items()))
        return Transfer(name, kind, source, target, 0, stage, frozenset(paths), atoms,
                        start, start + 1, frozenset(predecessors))

    transfers = (
        transfer(child_id, 0, 3, 1, {3: [(3, 1, 0)]}, 0),
        transfer(arrival_id, 0, 1, 0, {1: [(1, 0, 1)], 3: [(3, 1, 0), (1, 0, 1)]}, 1, [child_id]),
        transfer('final', 0, 2, 0, {2: [(2, 0, 2)]}, 2, [arrival_id]),
    )
    reduced_schedule = Schedule('reduced', transfers, (), 4, 1, 1024, metadata={
        'routing_only': True, 'path_scope': 'stage_suffix',
        'path_roots': {t.transfer_id: {a.slice_id: a.slice_id for a in t.atoms} for t in transfers},
        'tree_contributors': {t.transfer_id: tuple(range(4)) for t in transfers},
        'semantic_predecessors': {t.transfer_id: t.predecessor_ids for t in transfers},
        'final_outputs': {'r00000000-o00000000': tuple(range(4))},
        'final_dependencies': {'r00000000-o00000000': ('final',)},
    })
    sends = tuple(transfer('send' + str(rank), 1, 0, rank,
                  {member: [(0, rank, 0)] for member in range(4)}, 0) for rank in (1, 2, 3))
    distributed_schedule = Schedule('distributed', sends, (), 4, 1, 1024, metadata={
        'path_scope': 'stage_suffix', 'path_roots': {t.transfer_id: 0 for t in sends},
        'tree_contributors': {t.transfer_id: tuple(range(4)) for t in sends},
    })
    composed = _compose_schedules(plan, {'reduce': reduced_schedule, 'distribute': distributed_schedule})
    for send in (t for t in composed.transfers if t.stage_id == 1):
        paths = {a.slice_id: [(s.src_rank, s.dst_rank) for stage in a.path for s in stage.symbols]
                 for a in send.atoms}
        assert paths[3] == [(3, 1), (1, 0), (0, send.dst_rank)]
        assert paths[1] == [(1, 0), (0, send.dst_rank)]
        assert paths[2] == [(2, 0), (0, send.dst_rank)]
        assert paths[0] == [(0, send.dst_rank)]
        assert send.predecessor_ids == frozenset({'final'})
