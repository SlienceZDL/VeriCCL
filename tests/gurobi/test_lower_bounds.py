from dataclasses import replace
from itertools import product
import math

import pytest

from vericcl.planner.model import PlanNode, PlanningMode, StageInterface
from vericcl.semantics.collective import (
    CollectiveKind,
    CollectiveSpec,
    OutputSlot,
)
from vericcl.solver.demands import build_solver_problem
from vericcl.solver.gurobi_api import GurobiAdapter
import vericcl.solver.lower_bounds as lower_bounds_module
from vericcl.solver.lower_bounds import (
    global_throughput_time_lower_bound,
    throughput_time_lower_bound,
)
from vericcl.solver.templates import build_solver_templates, split_routing_units
from vericcl.topology.model import (
    DirectedLink,
    LinkKey,
    PerformanceCurve,
    SharedResource,
    Topology,
)

from tests.gurobi.helpers import (
    broadcast_problem,
    multihop_problem,
    reduction_dual_problem,
    require_gurobi_license,
)


pytestmark = [pytest.mark.phase03, pytest.mark.gurobi]


def _problem_with_logical_positions(count):
    original = broadcast_problem(logical_positions=tuple(range(count)))
    inputs = replace(
        original.inputs,
        hyperparameters=replace(
            original.inputs.hyperparameters,
            total_size_bytes=(
                count * original.inputs.hyperparameters.slice_size_bytes
            ),
        ),
    )
    return build_solver_problem(
        original.node,
        inputs,
        original.topology,
    )


def _record_resource_model_shapes(monkeypatch):
    gp = GurobiAdapter.require()
    shapes = []

    class RecordingModel:
        def __init__(self, name):
            self.model = gp.Model(name)

        def __getattr__(self, name):
            return getattr(self.model, name)

        def dispose(self):
            shapes.append(
                (
                    int(self.model.NumVars),
                    int(self.model.NumConstrs),
                    int(self.model.NumGenConstrs),
                )
            )
            self.model.dispose()

    class RecordingGp:
        GRB = gp.GRB
        GurobiError = gp.GurobiError
        quicksum = staticmethod(gp.quicksum)
        Model = RecordingModel

    monkeypatch.setattr(
        GurobiAdapter,
        "require",
        classmethod(lambda cls: RecordingGp),
    )
    return shapes


def _two_path_problem():
    original = broadcast_problem(logical_positions=(0, 1))
    inputs = replace(
        original.inputs,
        rank_count=4,
        hyperparameters=replace(
            original.inputs.hyperparameters,
            total_size_bytes=(
                2 * original.inputs.hyperparameters.slice_size_bytes
            ),
        ),
        strategies=replace(
            original.inputs.strategies,
            shortest_paths=False,
        ),
    )
    curve = PerformanceCurve(0.0, 1.0, {})
    keys = tuple(
        LinkKey(src_rank, dst_rank)
        for src_rank, dst_rank in (
            (0, 1),
            (1, 3),
            (0, 2),
            (2, 3),
        )
    )
    topology = Topology(
        rank_count=4,
        links={
            key: DirectedLink(key, 1, curve, ()) for key in keys
        },
        shared_resources={},
        node_membership={rank: 0 for rank in range(4)},
        gateways=frozenset(),
        warnings=(),
    )
    logical_input = {
        OutputSlot(0, position): frozenset({position})
        for position in (0, 1)
    }
    logical_output = dict(logical_input)
    logical_output.update(
        {
            OutputSlot(3, position): frozenset({position})
            for position in (0, 1)
        }
    )
    node = PlanNode(
        node_id="two-path-broadcast",
        stage_id=0,
        local_collective=CollectiveSpec(
            kind=CollectiveKind.BROADCAST,
            datatype="float32",
            root=0,
        ),
        communication_group=(0, 1, 2, 3),
        logical_input=StageInterface(logical_input),
        logical_output=StageInterface(logical_output),
        allowed_links=frozenset(keys),
        shared_resource_ids=frozenset(),
    )
    return build_solver_problem(node, inputs, topology)


def _brute_resource_bound(problems, max_channels):
    choices = []
    for problem in problems:
        for unit in split_routing_units(problem):
            demands = tuple(unit.demands)
            unit_choices = []
            for paths in product(
                *(demand.candidate_paths for demand in demands)
            ):
                physical_edges = {}
                parents = {}
                valid = True
                for demand, path in zip(demands, paths):
                    for src_rank, dst_rank in zip(path, path[1:]):
                        logical = LinkKey(src_rank, dst_rank)
                        previous = parents.setdefault(dst_rank, src_rank)
                        if previous != src_rank:
                            valid = False
                        physical = LinkKey(
                            *demand.physical_link(src_rank, dst_rank)
                        )
                        physical_edges.setdefault(physical, demand)
                if valid:
                    unit_choices.append((problem, tuple(physical_edges)))
            choices.append(tuple(unit_choices))
    best = math.inf
    reference = problems[0]
    for selected in product(*choices):
        link_bytes = {}
        resource_bytes = {}
        for problem, physical_edges in selected:
            for physical in physical_edges:
                link_bytes[physical] = (
                    link_bytes.get(physical, 0)
                    + problem.slice_size_bytes
                )
                for resource_id in problem.topology.link(
                    physical
                ).resource_ids:
                    resource_bytes[resource_id] = (
                        resource_bytes.get(resource_id, 0)
                        + problem.slice_size_bytes
                    )
        loads = []
        for physical, byte_count in link_bytes.items():
            edge = reference.topology.link(physical)
            channel_limit = min(max_channels, edge.max_channels)
            channel_limit = min(
                [channel_limit]
                + [
                    reference.topology.shared_resources[
                        resource_id
                    ].max_channels
                    for resource_id in edge.resource_ids
                ]
            )
            capacity = lower_bounds_module._maximum_capacity(
                edge.performance,
                reference.slice_size_bytes,
                channel_limit,
            )
            if math.isfinite(capacity):
                loads.append(byte_count / capacity)
        for resource_id, byte_count in resource_bytes.items():
            resource = reference.topology.shared_resources[resource_id]
            capacity = lower_bounds_module._maximum_capacity(
                resource.performance,
                reference.slice_size_bytes,
                min(max_channels, resource.max_channels),
            )
            if math.isfinite(capacity):
                loads.append(byte_count / capacity)
        best = min(best, max(loads, default=0.0))
    return best


def _mapped_domain_problems():
    original = broadcast_problem(logical_positions=(0,))
    inputs = replace(
        original.inputs,
        rank_count=4,
        hyperparameters=replace(
            original.inputs.hyperparameters,
            total_size_bytes=(
                original.inputs.hyperparameters.slice_size_bytes
            ),
        ),
        strategies=replace(
            original.inputs.strategies,
            shortest_paths=False,
        ),
    )
    curve = PerformanceCurve(0.0, 1.0, {})
    links = (LinkKey(0, 1), LinkKey(2, 3))
    resource = SharedResource("fabric", links, 1, curve)
    topology = Topology(
        rank_count=4,
        links={
            link: DirectedLink(link, 1, curve, ("fabric",))
            for link in links
        },
        shared_resources={"fabric": resource},
        node_membership={0: 0, 1: 0, 2: 1, 3: 1},
        gateways=frozenset(),
        warnings=(),
    )
    problems = []
    slice_count = inputs.hyperparameters.slice_count
    for index, (root, leaf) in enumerate(((0, 1), (2, 3))):
        contributor = root * slice_count
        values = frozenset({contributor})
        node = PlanNode(
            node_id="mapped-domain-{}".format(index),
            stage_id=0,
            local_collective=CollectiveSpec(
                kind=CollectiveKind.BROADCAST,
                datatype="float32",
                root=root,
            ),
            communication_group=(root, leaf),
            logical_input=StageInterface(
                {OutputSlot(root, 0): values}
            ),
            logical_output=StageInterface(
                {
                    OutputSlot(root, 0): values,
                    OutputSlot(leaf, 0): values,
                }
            ),
            allowed_links=frozenset({LinkKey(root, leaf)}),
            shared_resource_ids=frozenset({"fabric"}),
        )
        problems.append(build_solver_problem(node, inputs, topology))
    return tuple(problems)


def test_continuous_resource_relaxation_counts_each_payload_tree_once():
    require_gurobi_license()
    problem = broadcast_problem(logical_positions=(0, 1))

    bound = throughput_time_lower_bound(problem, max_channels=1)

    assert bound.resource_us == 4.0
    assert bound.dependency_us == 2.0


def test_shared_resource_capacity_is_included_in_the_relaxation():
    require_gurobi_license()
    problem = multihop_problem(shared_resource=True)

    bound = throughput_time_lower_bound(problem, max_channels=2)

    assert bound.resource_us == 4.0
    assert bound.dependency_us == 4.0


def test_global_resource_bound_accumulates_shared_load_across_plan_nodes():
    require_gurobi_license()
    problem = multihop_problem(shared_resource=True)

    bound = global_throughput_time_lower_bound(
        (problem, problem),
        max_channels=2,
    )

    assert bound.resource_us == 8.0
    assert bound.dependency_us == 4.0


def test_template_multiplicity_counts_every_real_payload_tree():
    require_gurobi_license()
    problem = _problem_with_logical_positions(128)
    templates = build_solver_templates(
        (problem,),
        PlanningMode.DIRECT,
    )

    bound = global_throughput_time_lower_bound(
        (problem,),
        max_channels=1,
    )

    assert len(templates) == 1
    assert len(templates[0].members) == 128
    assert bound.resource_us == 256.0
    assert bound.dependency_us == 2.0


def test_equivalent_template_multiplicity_does_not_expand_lp_shape(
    monkeypatch,
):
    require_gurobi_license()
    shapes = _record_resource_model_shapes(monkeypatch)
    single = _problem_with_logical_positions(1)
    repeated = _problem_with_logical_positions(128)

    single_bound = global_throughput_time_lower_bound(
        (single,),
        max_channels=1,
    )
    repeated_bound = global_throughput_time_lower_bound(
        (repeated,),
        max_channels=1,
    )

    assert shapes[0] == (3, 4, 0)
    assert shapes[1] == shapes[0]
    assert single_bound.resource_us == 2.0
    assert repeated_bound.resource_us == 256.0


def test_compressed_relaxation_matches_small_bruteforce_reference():
    require_gurobi_license()
    problem = _two_path_problem()
    templates = build_solver_templates((problem,), PlanningMode.DIRECT)

    bound = global_throughput_time_lower_bound(
        (problem,),
        max_channels=1,
    )
    reference = _brute_resource_bound((problem,), max_channels=1)

    assert len(templates) == 1
    assert len(templates[0].members) == 2
    assert reference == 1.0
    assert bound.resource_us == pytest.approx(reference)


def test_physical_rank_mapping_splits_resource_equivalence_classes(
    monkeypatch,
):
    require_gurobi_license()
    shapes = _record_resource_model_shapes(monkeypatch)
    problems = _mapped_domain_problems()
    templates = build_solver_templates(problems, PlanningMode.DIRECT)

    bound = global_throughput_time_lower_bound(
        problems,
        max_channels=1,
    )

    assert len(templates) == 1
    assert len(templates[0].members) == 2
    assert shapes == [(5, 9, 0)]
    assert bound.resource_us == 2.0


def test_reduction_dual_resource_bound_uses_reversed_physical_direction():
    require_gurobi_license()
    original = reduction_dual_problem()
    fast = PerformanceCurve(0.0, 1.0, {})
    slow = PerformanceCurve(0.0, 5.0, {})
    topology = replace(
        original.topology,
        links={
            key: replace(
                edge,
                max_channels=1,
                performance=(
                    slow if key == LinkKey(1, 0) else fast
                ),
            )
            for key, edge in original.topology.links.items()
        },
    )
    problem = replace(original, topology=topology)

    bound = global_throughput_time_lower_bound(
        (problem,),
        max_channels=1,
    )

    assert bound.resource_us == 5.0
    assert bound.dependency_us == 5.0
