import json
from copy import deepcopy
from pathlib import Path

import pytest

from vericcl.errors import InputValidationError
from vericcl.input.loader import resolve_inputs
from vericcl.provenance import LEGACY_TACCL_TOPOLOGY_FORMAT
from vericcl.topology.legacy import convert_legacy_topology
from vericcl.topology.loader import load_topology, topology_from_mapping
from vericcl.topology.model import LinkKey


pytestmark = pytest.mark.phase02


REPOSITORY_ROOT = Path(__file__).parents[3]
EXAMPLES = REPOSITORY_ROOT / "vericcl" / "examples"


def load_example(name):
    resolved = resolve_inputs(
        EXAMPLES / "topo" / name,
        EXAMPLES / "sketch" / "allreduce_8m_1m.json",
        EXAMPLES / "atom" / "default.json",
    )
    return load_topology(resolved)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def explicit_topology():
    return {
        "ranks": 2,
        "nodes": [{"id": 0, "ranks": [0, 1], "gateways": []}],
        "directed_links": [
            {
                "src": 0,
                "dst": 1,
                "max_channels": 2,
                "alpha_us": 1,
                "invbw_us": 2,
                "bandwidth_bytes_per_us": {"1": 100, "2": 180},
                "resources": [],
            }
        ],
        "shared_resources": [],
        "warnings": ["source warning"],
    }


def legacy_pair(
    *,
    name="legacy",
    gpus_per_node=2,
    node_count=2,
    connection="direct-map",
    intranode=None,
):
    topology = {
        "name": name,
        "gpus_per_node": gpus_per_node,
        "nics_per_node": 1,
        "alpha": 1,
        "node_betas_list": [1, 2] if gpus_per_node == 8 else [1],
        "node_invbws_list": [2, 3] if gpus_per_node == 8 else [2],
        "remote_alpha": 2,
        "remote_beta": 3,
        "remote_invbw": 5,
    }
    sketch = {
        "nnodes": node_count,
        "intranode_sketch": intranode or {"strategy": "none"},
        "internode_sketch": {
            "strategy": "relay",
            "internode_conn": connection,
        },
    }
    return topology, sketch


def test_gateway_topology_contains_only_real_cross_node_links():
    topology = load_example("two_node_gateway.json")

    assert topology.has_link(0, 4)
    assert topology.has_link(4, 0)
    assert not topology.has_link(1, 5)
    assert topology.gateways == frozenset({0, 4})


def test_explicit_topology_has_deterministic_node_membership():
    topology = load_example("two_node_gateway.json")

    assert tuple(topology.node_membership.items()) == tuple(
        [(rank, 0) for rank in range(4)]
        + [(rank, 1) for rank in range(4, 8)]
    )


def test_legacy_ndv2_example_resolves_rank_count_without_mutation():
    raw_topology = read_json(
        EXAMPLES / "legacy" / "topo" / "topo-ndv2-1MB.json"
    )
    raw_sketch = read_json(
        EXAMPLES / "legacy" / "sketch" / "sk2-ndv2-n2.json"
    )
    topology_before = deepcopy(raw_topology)
    sketch_before = deepcopy(raw_sketch)

    converted = convert_legacy_topology(raw_topology, raw_sketch)
    topology = topology_from_mapping(converted)

    assert topology.rank_count == 16
    assert topology.has_link(0, 4)
    assert not topology.has_link(0, 5)
    assert topology.has_link(0, 8)
    assert topology.has_link(7, 15)
    assert raw_topology == topology_before
    assert raw_sketch == sketch_before
    assert (
        converted["provenance"]["legacy_format"]
        == LEGACY_TACCL_TOPOLOGY_FORMAT
    )


def test_legacy_conversion_preserves_source_snapshots():
    raw_topology = {
        "name": "legacy",
        "gpus_per_node": 2,
        "nics_per_node": 1,
        "alpha": 1,
        "node_betas_list": [1],
        "node_invbws_list": [2],
        "remote_alpha": 2,
        "remote_beta": 3,
        "remote_invbw": 5,
    }
    raw_sketch = {
        "nnodes": 2,
        "intranode_sketch": {"strategy": "none"},
        "internode_sketch": {
            "strategy": "relay",
            "internode_conn": "direct-map",
        },
    }

    converted = convert_legacy_topology(raw_topology, raw_sketch)

    assert converted["provenance"]["source_topology"] == raw_topology
    assert converted["provenance"]["source_sketch"] == raw_sketch
    assert [
        (item["src"], item["dst"])
        for item in converted["directed_links"]
        if item["src"] // 2 != item["dst"] // 2
    ] == [(0, 2), (1, 3), (2, 0), (3, 1)]
    topology = topology_from_mapping(converted)
    assert topology.has_link(0, 2)
    assert topology.has_link(1, 3)
    assert not topology.has_link(0, 3)
    assert topology.has_link(2, 0)


def test_loader_accepts_us_fields_and_canonical_calibration_keys():
    topology = topology_from_mapping(explicit_topology())

    edge = topology.link(LinkKey(0, 1))
    assert edge.performance.alpha_us == 1.0
    assert dict(edge.performance.bandwidth_bytes_per_us) == {1: 100.0, 2: 180.0}
    assert topology.warnings == ("source warning",)


def test_loader_rejects_expected_rank_count_mismatch():
    with pytest.raises(InputValidationError, match="rank count"):
        topology_from_mapping(explicit_topology(), expected_rank_count=3)


def test_load_topology_requires_resolved_input():
    with pytest.raises(InputValidationError, match="ResolvedInput"):
        load_topology(explicit_topology())


@pytest.mark.parametrize(
    "nodes, message",
    [
        ([], "must not be empty"),
        (
            [
                {"id": 0, "ranks": [0], "gateways": []},
                {"id": 0, "ranks": [1], "gateways": []},
            ],
            "IDs must be unique",
        ),
        ([{"id": 0, "ranks": [], "gateways": []}], "must not be empty"),
        ([{"id": 0, "ranks": [0, 2], "gateways": []}], "out of range"),
        (
            [
                {"id": 0, "ranks": [0, 1], "gateways": []},
                {"id": 1, "ranks": [1], "gateways": []},
            ],
            "multiple nodes",
        ),
        ([{"id": 0, "ranks": [0, 1], "gateways": [2]}], "must belong"),
        ([{"id": 0, "ranks": [0], "gateways": []}], "cover every rank"),
    ],
)
def test_loader_rejects_invalid_node_partitions(nodes, message):
    raw = explicit_topology()
    raw["nodes"] = nodes

    with pytest.raises(InputValidationError, match=message):
        topology_from_mapping(raw)


@pytest.mark.parametrize(
    "link_update, message",
    [
        ({"dst": 0}, "distinct"),
        ({"dst": 2}, "out of range"),
        ({"alpha_us": None}, "must define alpha"),
        ({"max_channels": 0}, "at least 1"),
        ({"resources": [""]}, "non-empty string"),
        ({"bandwidth_bytes_per_us": {"x": 1}}, "integer concurrency"),
        ({"bandwidth_bytes_per_us": {"01": 1}}, "non-canonical"),
        ({"invbw_us": "slow"}, "must be a number"),
    ],
)
def test_loader_rejects_invalid_directed_links(link_update, message):
    raw = explicit_topology()
    raw["directed_links"][0].update(link_update)

    with pytest.raises(InputValidationError, match=message):
        topology_from_mapping(raw)


def test_loader_rejects_duplicate_directed_links():
    raw = explicit_topology()
    raw["directed_links"].append(deepcopy(raw["directed_links"][0]))

    with pytest.raises(InputValidationError, match="must be unique"):
        topology_from_mapping(raw)


@pytest.mark.parametrize(
    "resource_update, message",
    [
        ({"id": ""}, "non-empty string"),
        ({"member_links": [[0]]}, "contain src and dst"),
        ({"member_links": [[0, 0]]}, "distinct"),
        ({"member_links": []}, "must contain LinkKey"),
        ({"max_channels": 0}, "at least 1"),
    ],
)
def test_loader_rejects_invalid_shared_resources(resource_update, message):
    raw = explicit_topology()
    raw["shared_resources"] = [
        {
            "id": "fabric",
            "member_links": [[0, 1]],
            "alpha": 1,
            "invbw": 2,
            "max_channels": 2,
            **resource_update,
        }
    ]

    with pytest.raises(InputValidationError, match=message):
        topology_from_mapping(raw)


def test_loader_rejects_inconsistent_resource_membership():
    raw = explicit_topology()
    raw["shared_resources"] = [
        {
            "id": "fabric",
            "member_links": [[0, 1]],
            "alpha": 1,
            "invbw": 2,
            "max_channels": 2,
        }
    ]

    with pytest.raises(InputValidationError, match="not declared by its link"):
        topology_from_mapping(raw)


def test_legacy_explicit_matrices_preserve_directed_edges():
    raw_topology = {
        "name": "matrix",
        "gpus_per_node": 2,
        "nics_per_node": 0,
        "alpha": 1,
        "links": [[0, 0], [1, 0]],
        "betas": [[0, 0], [1, 0]],
        "invbws": [[0, 0], [2, 0]],
    }
    raw_sketch = {
        "nnodes": 1,
        "intranode_sketch": {"strategy": "none"},
    }

    topology = topology_from_mapping(
        convert_legacy_topology(raw_topology, raw_sketch)
    )

    assert topology.has_link(0, 1)
    assert not topology.has_link(1, 0)
    assert topology.gateways == frozenset()


def test_legacy_fit8_uses_two_link_classes():
    raw_topology, raw_sketch = legacy_pair(
        name="FIT8",
        gpus_per_node=8,
        node_count=1,
    )

    topology = topology_from_mapping(
        convert_legacy_topology(raw_topology, raw_sketch)
    )

    assert topology.link(LinkKey(0, 1)).performance.invbw_us == 2.0
    assert topology.link(LinkKey(0, 4)).performance.invbw_us == 3.0


def test_legacy_fully_connected_mapping_creates_all_cross_node_links():
    raw_topology, raw_sketch = legacy_pair(connection="fully-connected")

    topology = topology_from_mapping(
        convert_legacy_topology(raw_topology, raw_sketch)
    )

    cross_node = [
        key
        for key in topology.links
        if topology.node_membership[key.src_rank]
        != topology.node_membership[key.dst_rank]
    ]
    assert len(cross_node) == 8
    assert topology.gateways == frozenset({0, 1, 2, 3})


def test_legacy_switch_hyperedges_become_directional_resources():
    raw_topology, raw_sketch = legacy_pair(
        node_count=1,
        intranode={"strategy": "switch", "switches": [[0, 1]]},
    )

    topology = topology_from_mapping(
        convert_legacy_topology(raw_topology, raw_sketch)
    )

    assert topology.resources_for(LinkKey(0, 1)) == (
        "node-0-switch-0-rank-0-egress",
        "node-0-switch-0-rank-1-ingress",
    )
    assert topology.resources_for(LinkKey(1, 0)) == (
        "node-0-switch-0-rank-0-ingress",
        "node-0-switch-0-rank-1-egress",
    )


def test_legacy_empty_connection_map_creates_no_remote_resources():
    raw_topology, raw_sketch = legacy_pair(connection={})

    converted = convert_legacy_topology(raw_topology, raw_sketch)
    topology = topology_from_mapping(converted)

    assert topology.gateways == frozenset()
    assert topology.shared_resources == {}


@pytest.mark.parametrize(
    "connection, message",
    [
        ({"x": [0]}, "source must be an integer"),
        ({"2": [0]}, "source is out of range"),
        ({"0": [2]}, "destination is out of range"),
        ({"0": [1, 1]}, "must be unique"),
        ({"0": "1"}, "must be a JSON array"),
    ],
)
def test_legacy_rejects_invalid_connection_maps(connection, message):
    raw_topology, raw_sketch = legacy_pair(connection=connection)

    with pytest.raises(InputValidationError, match=message):
        convert_legacy_topology(raw_topology, raw_sketch)


def test_legacy_rejects_non_relay_internode_strategy():
    raw_topology, raw_sketch = legacy_pair()
    raw_sketch["internode_sketch"]["strategy"] = "direct"

    with pytest.raises(InputValidationError, match="must be relay"):
        convert_legacy_topology(raw_topology, raw_sketch)


@pytest.mark.parametrize(
    "missing",
    [("remote_alpha",), ("remote_beta", "remote_invbw")],
)
def test_legacy_requires_remote_performance_for_multiple_nodes(missing):
    raw_topology, raw_sketch = legacy_pair()
    for field in missing:
        del raw_topology[field]

    with pytest.raises(InputValidationError, match="remote performance"):
        convert_legacy_topology(raw_topology, raw_sketch)


@pytest.mark.parametrize("missing", ["remote_beta", "remote_invbw"])
def test_legacy_accepts_either_remote_beta_or_invbw(missing):
    raw_topology, raw_sketch = legacy_pair()
    del raw_topology[missing]

    topology = topology_from_mapping(
        convert_legacy_topology(raw_topology, raw_sketch)
    )

    assert topology.has_link(0, 2)


@pytest.mark.parametrize(
    "matrix_name, matrix, message",
    [
        ("links", [[0, 1]], "wrong row count"),
        ("betas", [[0], [1]], "wrong column count"),
    ],
)
def test_legacy_rejects_malformed_explicit_matrices(
    matrix_name,
    matrix,
    message,
):
    raw_topology = {
        "name": "matrix",
        "gpus_per_node": 2,
        "alpha": 1,
        "links": [[0, 1], [1, 0]],
        "betas": [[0, 1], [1, 0]],
        "invbws": [[0, 2], [2, 0]],
    }
    raw_topology[matrix_name] = matrix
    raw_sketch = {"nnodes": 1, "intranode_sketch": {"strategy": "none"}}

    with pytest.raises(InputValidationError, match=message):
        convert_legacy_topology(raw_topology, raw_sketch)


def test_legacy_rejects_invalid_switch_rank():
    raw_topology, raw_sketch = legacy_pair(
        node_count=1,
        intranode={"strategy": "switch", "switches": [[0, 2]]},
    )

    with pytest.raises(InputValidationError, match="switch rank is out of range"):
        convert_legacy_topology(raw_topology, raw_sketch)
