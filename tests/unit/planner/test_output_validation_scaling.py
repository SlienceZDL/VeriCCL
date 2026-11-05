"""Output validation must scale with stored logical states, not their product."""
from vericcl.planner.direct import build_direct_plan
from vericcl.semantics.collective import CollectiveKind, OutputSlot
from tests.unit.planner.test_direct import resolved_input, topology_for


def test_many_allgather_slices_use_linear_output_lookup_work(monkeypatch):
    inputs = resolved_input(CollectiveKind.ALL_GATHER, slice_count=64)
    plan = build_direct_plan(inputs, topology_for(inputs))
    outgoing = {node.node_id: [] for node in plan.nodes}
    for edge in plan.edges:
        outgoing[edge.producer_id].append(edge)
    probes = 0
    original_hash = OutputSlot.__hash__

    def counted_hash(self):
        nonlocal probes
        probes += 1
        return original_hash(self)

    monkeypatch.setattr(OutputSlot, '__hash__', counted_hash)
    plan._validate_outputs(plan.nodes, outgoing)
    states = len(plan.final_outputs.values) + sum(len(node.logical_output.values) for node in plan.nodes)
    assert probes <= 10 * states

