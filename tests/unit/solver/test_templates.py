from dataclasses import replace
from pathlib import Path

import pytest

from vericcl.errors import SemanticError
from vericcl.input.loader import resolve_inputs
from vericcl.input.models import AtomConstraints, ForbiddenTransfer
from vericcl.planner.model import PlanningMode, PlanNode, StageInterface
from vericcl.semantics.collective import (
    CollectiveKind,
    CollectiveSpec,
    OutputSlot,
)
from vericcl.solver.demands import (
    SolverProblem,
    TransferDemand,
    build_solver_problem,
)
from vericcl.solver.templates import (
    RoutingUnit,
    SolverTemplate,
    TemplateMember,
    build_solver_templates,
    split_routing_units,
)
from vericcl.topology.model import (
    DirectedLink,
    LinkKey,
    PerformanceCurve,
    SharedResource,
    Topology,
)


pytestmark = pytest.mark.phase03


EXAMPLES = Path(__file__).parents[3] / "vericcl" / "examples"


def _inputs(
    rank_count,
    slice_count,
    *,
    slice_size=1,
    forbidden=(),
):
    base = resolve_inputs(
        EXAMPLES / "topo" / "two_rank.json",
        EXAMPLES / "sketch" / "allreduce_8m_1m.json",
        EXAMPLES / "atom" / "default.json",
    )
    return replace(
        base,
        rank_count=rank_count,
        collective=CollectiveSpec(
            kind=CollectiveKind.ALL_GATHER,
            datatype="float32",
        ),
        hyperparameters=replace(
            base.hyperparameters,
            total_size_bytes=slice_count * slice_size,
            slice_size_bytes=slice_size,
        ),
        atom_constraints=AtomConstraints(
            stage_num=None,
            forbidden_transfers=tuple(forbidden),
        ),
        strategies=replace(base.strategies, symmetry=False),
    )


def _curve(invbw=2.0, bandwidth=None):
    return PerformanceCurve(
        alpha_us=1.0,
        invbw_us=invbw,
        bandwidth_bytes_per_us={} if bandwidth is None else {1: bandwidth},
    )


def _topology(
    groups,
    *,
    max_channels=4,
    invbw=2.0,
    bandwidth=None,
    resources=False,
    resource_channels=2,
    resource_invbw=2.0,
    missing_links=(),
):
    rank_count = max(rank for group in groups for rank in group) + 1
    missing = set(missing_links)
    links = {}
    resource_specs = {}
    for group_index, group in enumerate(groups):
        keys = tuple(
            LinkKey(src, dst)
            for src in group
            for dst in group
            if src != dst and LinkKey(src, dst) not in missing
        )
        resource_id = "domain-resource-{}".format(group_index)
        for key in keys:
            links[key] = DirectedLink(
                key=key,
                max_channels=max_channels,
                performance=_curve(invbw, bandwidth),
                resource_ids=(resource_id,) if resources else (),
            )
        if resources:
            resource_specs[resource_id] = SharedResource(
                resource_id=resource_id,
                member_links=keys,
                max_channels=resource_channels,
                performance=_curve(resource_invbw),
            )
    return Topology(
        rank_count=rank_count,
        links=links,
        shared_resources=resource_specs,
        node_membership={
            rank: group_index
            for group_index, group in enumerate(groups)
            for rank in group
        },
        gateways=frozenset(group[0] for group in groups),
        warnings=(),
    )


def _tree_problem(
    inputs,
    topology,
    group,
    root,
    logical_position,
    *,
    node_id=None,
    contributors=None,
    reduction_dual=False,
    stage_id=0,
):
    group = tuple(group)
    slice_count = inputs.hyperparameters.slice_count
    if contributors is None:
        contributors = frozenset({root * slice_count + logical_position})
    else:
        contributors = frozenset(contributors)
    if reduction_dual:
        kind = CollectiveKind.REDUCE_SCATTER
        logical_input = StageInterface(
            {
                OutputSlot(rank, logical_position): frozenset(
                    {rank * slice_count + logical_position}
                )
                for rank in group
            }
        )
        logical_output = StageInterface(
            {OutputSlot(root, logical_position): contributors}
        )
        spec = CollectiveSpec(
            kind=kind,
            datatype="float32",
            reduction_op="sum",
        )
    else:
        logical_input = StageInterface(
            {OutputSlot(root, logical_position): contributors}
        )
        logical_output = StageInterface(
            {
                OutputSlot(rank, logical_position): contributors
                for rank in group
            }
        )
        spec = CollectiveSpec(
            kind=CollectiveKind.BROADCAST,
            datatype="float32",
            root=root,
        )
    allowed_physical = frozenset(
        key
        for key in topology.links
        if key.src_rank in group and key.dst_rank in group
    )
    allowed_virtual = frozenset(
        LinkKey(key.dst_rank, key.src_rank)
        for key in allowed_physical
    ) if reduction_dual else allowed_physical
    demands = []
    for leaf in group:
        if leaf == root:
            continue
        members = (
            frozenset({leaf * slice_count + logical_position})
            if reduction_dual
            else contributors
        )
        forbidden = tuple(
            item
            for item in inputs.atom_constraints.forbidden_transfers
            if item.stage_id == stage_id and item.slice_id in members
        )
        forbidden_virtual = {
            LinkKey(
                item.dst_rank if reduction_dual else item.src_rank,
                item.src_rank if reduction_dual else item.dst_rank,
            )
            for item in forbidden
        }
        legal = allowed_virtual - forbidden_virtual
        direct = LinkKey(root, leaf)
        demands.append(
            TransferDemand(
                demand_id="{}-leaf-{}".format(
                    node_id or "tree-{}-{}".format(root, logical_position),
                    leaf,
                ),
                node_id=node_id or "tree-{}-{}".format(root, logical_position),
                stage_id=stage_id,
                root_rank=root,
                required_leaf_rank=leaf,
                logical_position=logical_position,
                contributors=contributors,
                member_slice_ids=members,
                allowed_links=allowed_virtual,
                legal_links=legal,
                forbidden_members=forbidden,
                candidate_paths=((root, leaf),) if direct in legal else (),
                reduction_dual=reduction_dual,
            )
        )
    actual_node_id = node_id or "tree-{}-{}".format(root, logical_position)
    node = PlanNode(
        node_id=actual_node_id,
        stage_id=stage_id,
        local_collective=spec,
        communication_group=group,
        logical_input=logical_input,
        logical_output=logical_output,
        allowed_links=allowed_physical,
        shared_resource_ids=frozenset(
            resource_id
            for key in allowed_physical
            for resource_id in topology.link(key).resource_ids
        ),
        dual_of_node_id="dual-{}".format(actual_node_id)
        if reduction_dual
        else None,
    )
    return SolverProblem(
        node=node,
        inputs=inputs,
        topology=topology,
        demands=tuple(demands),
        candidate_edges=frozenset(),
        infeasible_demand_ids=tuple(
            demand.demand_id for demand in demands if not demand.candidate_paths
        ),
        restrictions=(),
    )


def _chain_problem(kind):
    inputs = _inputs(3, 4)
    topology = _topology(((0, 1, 2),))
    if kind is CollectiveKind.GATHER:
        routes = ((1, 0, 4), (2, 0, 8))
        root = 0
    elif kind is CollectiveKind.SCATTER:
        routes = ((0, 1, 0), (0, 2, 1))
        root = 0
    else:
        routes = ((0, 1, 0), (1, 2, 4))
        root = None
    input_values = {}
    output_values = {}
    demands = []
    allowed = frozenset(topology.links)
    for index, (source, destination, contributor) in enumerate(routes):
        logical_position = contributor % inputs.hyperparameters.slice_count
        members = frozenset({contributor})
        input_values[OutputSlot(source, index)] = members
        output_values[OutputSlot(destination, index)] = members
        demands.append(
            TransferDemand(
                demand_id="{}-{}".format(kind.value, index),
                node_id="{}-node".format(kind.value),
                stage_id=0,
                root_rank=source,
                required_leaf_rank=destination,
                logical_position=logical_position,
                contributors=members,
                member_slice_ids=members,
                allowed_links=allowed,
                legal_links=allowed,
                forbidden_members=(),
                candidate_paths=((source, destination),),
            )
        )
    node = PlanNode(
        node_id="{}-node".format(kind.value),
        stage_id=0,
        local_collective=CollectiveSpec(
            kind=kind,
            datatype="float32",
            root=root,
        ),
        communication_group=(0, 1, 2),
        logical_input=StageInterface(input_values),
        logical_output=StageInterface(output_values),
        allowed_links=allowed,
        shared_resource_ids=frozenset(),
    )
    return SolverProblem(
        node=node,
        inputs=inputs,
        topology=topology,
        demands=tuple(demands),
        candidate_edges=frozenset(),
        infeasible_demand_ids=(),
        restrictions=(),
    )


def test_direct_allgather_uses_one_template_per_source_root():
    rank_count = 8
    slice_count = 128
    inputs = _inputs(rank_count, slice_count)
    topology = _topology((tuple(range(rank_count)),))
    problems = tuple(
        _tree_problem(
            inputs,
            topology,
            tuple(range(rank_count)),
            root,
            logical_position,
            node_id="allgather-r{:08d}-a{:08d}".format(
                root,
                logical_position,
            ),
        )
        for root in range(rank_count)
        for logical_position in range(slice_count)
    )

    units = tuple(
        unit
        for problem in problems
        for unit in split_routing_units(problem)
    )
    templates = build_solver_templates(problems, PlanningMode.DIRECT)

    assert len(units) == 1024
    assert len(templates) == 8
    assert sorted(len(template.members) for template in templates) == [128] * 8
    assert {
        template.representative.demands[0].root_rank
        for template in templates
    } == set(range(8))


def test_same_root_logical_positions_use_explicit_identity_rank_mapping():
    inputs = _inputs(3, 4)
    topology = _topology(((0, 1, 2),))
    problems = tuple(
        _tree_problem(
            inputs,
            topology,
            (0, 1, 2),
            0,
            logical_position,
            node_id="logical-{}".format(logical_position),
        )
        for logical_position in (0, 1, 2)
    )

    templates = build_solver_templates(problems, PlanningMode.DIRECT)

    assert len(templates) == 1
    translated = next(
        member for member in templates[0].members if member.node_id == "logical-1"
    )
    assert translated.rank_map == ((0, 0), (1, 1), (2, 2))
    assert translated.contributor_map == ((0, 1),)
    assert translated.logical_position_map == ((0, 1),)


def test_batched_tree_interfaces_do_not_block_logical_translation_reuse():
    inputs = _inputs(3, 4)
    topology = _topology(((0, 1, 2),))
    first = frozenset({0})
    second = frozenset({1})
    node = PlanNode(
        node_id="batched-tree",
        stage_id=0,
        local_collective=CollectiveSpec(
            kind=CollectiveKind.BROADCAST,
            datatype="float32",
            root=0,
        ),
        communication_group=(0, 1, 2),
        logical_input=StageInterface(
            {
                OutputSlot(0, 0): first,
                OutputSlot(0, 1): second,
            }
        ),
        logical_output=StageInterface(
            {
                OutputSlot(rank, logical_position): contributors
                for logical_position, contributors in ((0, first), (1, second))
                for rank in range(3)
            }
        ),
        allowed_links=frozenset(topology.links),
        shared_resource_ids=frozenset(),
    )
    problem = build_solver_problem(node, inputs, topology)

    units = split_routing_units(problem)
    templates = build_solver_templates((problem,), PlanningMode.DIRECT)

    assert len(units) == 2
    assert len(templates) == 1
    assert len(templates[0].members) == 2


def test_overlapping_interface_contributors_prevent_unsafe_reuse():
    inputs = _inputs(3, 4)
    topology = _topology(((0, 1, 2),))
    baseline = _tree_problem(
        inputs,
        topology,
        (0, 1, 2),
        0,
        0,
        node_id="overlap-a",
    )
    changed = _tree_problem(
        inputs,
        topology,
        (0, 1, 2),
        0,
        0,
        node_id="overlap-b",
    )
    baseline = replace(
        baseline,
        node=replace(
            baseline.node,
            logical_output=StageInterface(
                {OutputSlot(1, 0): frozenset({0, 4})}
            ),
        ),
    )
    changed = replace(
        changed,
        node=replace(
            changed.node,
            logical_output=StageInterface(
                {OutputSlot(1, 0): frozenset({0, 8})}
            ),
        ),
    )

    templates = build_solver_templates(
        (baseline, changed),
        PlanningMode.DIRECT,
    )

    assert len(templates) == 2


def test_slice_specific_forbidden_transfer_splits_only_impacted_unit():
    forbidden = ForbiddenTransfer(
        slice_id=1,
        src_rank=0,
        dst_rank=1,
        stage_id=0,
    )
    inputs = _inputs(3, 4, forbidden=(forbidden,))
    topology = _topology(((0, 1, 2),))
    problems = tuple(
        _tree_problem(
            inputs,
            topology,
            (0, 1, 2),
            0,
            logical_position,
            node_id="forbidden-{}".format(logical_position),
        )
        for logical_position in (0, 1, 2)
    )

    templates = build_solver_templates(problems, PlanningMode.DIRECT)

    assert sorted(len(template.members) for template in templates) == [1, 2]
    impacted = next(
        template
        for template in templates
        if any(member.node_id == "forbidden-1" for member in template.members)
    )
    assert [member.node_id for member in impacted.members] == ["forbidden-1"]


@pytest.mark.parametrize(
    "kind",
    (
        CollectiveKind.GATHER,
        CollectiveKind.SCATTER,
        CollectiveKind.ALL_TO_ALL,
    ),
)
def test_chain_collectives_keep_each_route_in_a_separate_unit(kind):
    units = split_routing_units(_chain_problem(kind))

    assert len(units) == 2
    assert all(len(unit.demands) == 1 for unit in units)


@pytest.mark.parametrize("reduction_dual", (False, True))
def test_tree_and_reduction_dual_demands_remain_one_coherent_unit(
    reduction_dual,
):
    inputs = _inputs(3, 4)
    topology = _topology(((0, 1, 2),))
    contributors = frozenset({0, 4, 8}) if reduction_dual else None
    problem = _tree_problem(
        inputs,
        topology,
        (0, 1, 2),
        0,
        0,
        contributors=contributors,
        reduction_dual=reduction_dual,
    )

    units = split_routing_units(problem)

    assert len(units) == 1
    assert len(units[0].demands) == 2


def test_batched_allgather_preserves_one_tree_per_source_payload():
    inputs = _inputs(3, 4)
    topology = _topology(((0, 1, 2),))
    first = frozenset({0})
    second = frozenset({4})
    node = PlanNode(
        node_id="batched-allgather",
        stage_id=0,
        local_collective=CollectiveSpec(
            kind=CollectiveKind.ALL_GATHER,
            datatype="float32",
        ),
        communication_group=(0, 1, 2),
        logical_input=StageInterface(
            {
                OutputSlot(0, 0): first,
                OutputSlot(1, 0): second,
            }
        ),
        logical_output=StageInterface(
            {
                OutputSlot(rank, contributor): values
                for contributor, values in ((0, first), (4, second))
                for rank in range(3)
            }
        ),
        allowed_links=frozenset(topology.links),
        shared_resource_ids=frozenset(),
    )

    units = split_routing_units(build_solver_problem(node, inputs, topology))

    assert len(units) == 2
    assert all(len(unit.demands) == 2 for unit in units)
    assert {unit.demands[0].root_rank for unit in units} == {0, 1}


@pytest.mark.parametrize("difference", ("root", "contributors", "reduction_dual"))
def test_semantic_differences_prevent_exact_template_merging(difference):
    inputs = _inputs(3, 4)
    topology = _topology(((0, 1, 2),))
    baseline = _tree_problem(
        inputs,
        topology,
        (0, 1, 2),
        0,
        0,
        node_id="baseline",
    )
    if difference == "root":
        changed = _tree_problem(
            inputs,
            topology,
            (0, 1, 2),
            1,
            0,
            node_id="changed-root",
        )
    elif difference == "contributors":
        changed = _tree_problem(
            inputs,
            topology,
            (0, 1, 2),
            0,
            0,
            node_id="changed-contributors",
            contributors=frozenset({4}),
        )
    else:
        changed = _tree_problem(
            inputs,
            topology,
            (0, 1, 2),
            0,
            0,
            node_id="changed-reduction",
            contributors=frozenset({0, 4, 8}),
            reduction_dual=True,
        )

    templates = build_solver_templates(
        (baseline, changed),
        PlanningMode.DIRECT,
    )

    assert len(templates) == 2


def test_planning_mode_is_part_of_the_exact_template_signature():
    inputs = _inputs(2, 2)
    topology = _topology(((0, 1),))
    problem = _tree_problem(inputs, topology, (0, 1), 0, 0)

    direct = build_solver_templates((problem,), PlanningMode.DIRECT)
    hierarchical = build_solver_templates(
        (problem,),
        PlanningMode.GATEWAY_ALLGATHER,
    )

    assert direct[0].exact_signature != hierarchical[0].exact_signature


def test_public_template_models_reject_noninvertible_mappings():
    inputs = _inputs(2, 2)
    topology = _topology(((0, 1),))
    unit = split_routing_units(
        _tree_problem(inputs, topology, (0, 1), 0, 0)
    )[0]
    member = TemplateMember(
        unit_id=unit.unit_id,
        node_id=unit.node.node_id,
        rank_map=((0, 0), (1, 1)),
        contributor_map=((0, 0),),
        logical_position_map=((0, 0),),
    )

    template = SolverTemplate(
        template_id="template",
        representative=unit,
        members=(member,),
        exact_signature="signature",
    )

    assert isinstance(unit, RoutingUnit)
    assert template.members == (member,)
    with pytest.raises(SemanticError, match="invertible"):
        replace(member, rank_map=((0, 0), (1, 0)))
