from dataclasses import replace
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from vericcl.input.loader import resolve_inputs
from vericcl.input.models import AtomConstraints, ForbiddenTransfer
from vericcl.planner.model import PlanNode, PlanningMode, StageInterface
from vericcl.semantics.collective import (
    CollectiveKind,
    CollectiveSpec,
    OutputSlot,
)
from vericcl.solver.demands import build_solver_problem
from vericcl.solver.templates import build_solver_templates
from vericcl.topology.loader import topology_from_mapping
from vericcl.topology.model import LinkKey


EXAMPLES = Path(__file__).parents[2] / "vericcl" / "examples"


def _case(max_channels, invbw, forbidden, mutated_resource=False):
    raw_links = []
    raw_resources = []
    for group_index, (root, leaf) in enumerate(((0, 1), (2, 3))):
        resource_id = "fabric-{}".format(group_index)
        raw_resources.append(
            {
                "id": resource_id,
                "member_links": [[root, leaf], [leaf, root]],
                "alpha": 1,
                "invbw": invbw,
                "max_channels": (
                    max_channels + 1
                    if mutated_resource and group_index == 1
                    else max_channels
                ),
            }
        )
        for src, dst in ((root, leaf), (leaf, root)):
            raw_links.append(
                {
                    "src": src,
                    "dst": dst,
                    "alpha": 1,
                    "invbw": invbw,
                    "max_channels": max_channels,
                    "resources": [resource_id],
                }
            )
    topology = topology_from_mapping(
        {
            "ranks": 4,
            "nodes": [
                {"id": 11, "ranks": [0, 1], "gateways": [0]},
                {"id": 29, "ranks": [2, 3], "gateways": [2]},
            ],
            "directed_links": raw_links,
            "shared_resources": raw_resources,
        }
    )
    base = resolve_inputs(
        EXAMPLES / "topo" / "two_rank.json",
        EXAMPLES / "sketch" / "allreduce_8m_1m.json",
        EXAMPLES / "atom" / "default.json",
    )
    forbidden_items = ()
    if forbidden:
        forbidden_items = (
            ForbiddenTransfer(0, 0, 1, 0),
            ForbiddenTransfer(2, 2, 3, 0),
        )
    inputs = replace(
        base,
        collective=CollectiveSpec(
            kind=CollectiveKind.BROADCAST,
            datatype="float32",
            root=0,
        ),
        hyperparameters=replace(
            base.hyperparameters,
            total_size_bytes=1024,
            slice_size_bytes=1024,
        ),
        rank_count=4,
        atom_constraints=AtomConstraints(None, forbidden_items),
    )
    problems = []
    for group in ((0, 1), (2, 3)):
        root, leaf = group
        contributor = frozenset({root})
        node = PlanNode(
            node_id="property-domain-{}".format(root),
            stage_id=0,
            local_collective=CollectiveSpec(
                kind=CollectiveKind.BROADCAST,
                datatype="float32",
                root=root,
            ),
            communication_group=group,
            logical_input=StageInterface(
                {OutputSlot(root, 0): contributor}
            ),
            logical_output=StageInterface(
                {
                    OutputSlot(root, 0): contributor,
                    OutputSlot(leaf, 0): contributor,
                }
            ),
            allowed_links=frozenset(
                {LinkKey(root, leaf), LinkKey(leaf, root)}
            ),
            shared_resource_ids=frozenset(
                {"fabric-{}".format(root // 2)}
            ),
        )
        problems.append(build_solver_problem(node, inputs, topology))
    return tuple(problems)


@given(
    max_channels=st.integers(min_value=1, max_value=8),
    invbw=st.integers(min_value=2, max_value=20),
    forbidden=st.booleans(),
)
@settings(max_examples=20, deadline=None)
def test_exact_rank_renumbering_maps_every_demand_candidate_and_forbidden(
    max_channels,
    invbw,
    forbidden,
):
    problems = _case(max_channels, invbw, forbidden)

    templates = build_solver_templates(problems, PlanningMode.DIRECT)

    assert len(templates) == 1
    template = templates[0]
    assert len(template.members) == 2
    member = next(
        item
        for item in template.members
        if item.unit_id.endswith("domain-2-u00000000")
    )
    rank_map = dict(member.rank_map)
    contributor_map = dict(member.contributor_map)
    representative = template.representative.demands[0]
    mapped = problems[1].demands[0]
    assert rank_map[representative.root_rank] == mapped.root_rank
    assert rank_map[representative.required_leaf_rank] == mapped.required_leaf_rank
    assert {contributor_map[value] for value in representative.contributors} == set(
        mapped.contributors
    )
    assert {
        (rank_map[link.src_rank], rank_map[link.dst_rank])
        for link in representative.allowed_links
    } == {(link.src_rank, link.dst_rank) for link in mapped.allowed_links}
    assert {
        tuple(rank_map[rank] for rank in path)
        for path in representative.candidate_paths
    } == set(mapped.candidate_paths)
    assert {
        (
            contributor_map[item.slice_id],
            rank_map[item.src_rank],
            rank_map[item.dst_rank],
            item.stage_id,
        )
        for item in representative.forbidden_members
    } == {
        (item.slice_id, item.src_rank, item.dst_rank, item.stage_id)
        for item in mapped.forbidden_members
    }


@given(
    max_channels=st.integers(min_value=1, max_value=8),
    invbw=st.integers(min_value=2, max_value=20),
)
@settings(max_examples=20, deadline=None)
def test_any_shared_resource_mutation_produces_independent_templates(
    max_channels,
    invbw,
):
    problems = _case(
        max_channels,
        invbw,
        forbidden=False,
        mutated_resource=True,
    )

    assert len(build_solver_templates(problems, PlanningMode.DIRECT)) == 2
