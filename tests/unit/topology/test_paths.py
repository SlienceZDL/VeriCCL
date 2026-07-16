import pytest

from vericcl.errors import SemanticError
from vericcl.topology.loader import topology_from_mapping
from vericcl.topology.paths import shortest_path_set


pytestmark = pytest.mark.phase02


def topology_with_edges(rank_count, edges):
    return topology_from_mapping(
        {
            "ranks": rank_count,
            "nodes": [
                {
                    "id": 0,
                    "ranks": list(range(rank_count)),
                    "gateways": [],
                }
            ],
            "directed_links": [
                {
                    "src": src,
                    "dst": dst,
                    "alpha": 0,
                    "invbw": cost,
                    "max_channels": 4,
                    "resources": [],
                }
                for src, dst, cost in edges
            ],
            "shared_resources": [],
        }
    )


def test_shortest_path_set_keeps_equal_cost_routes():
    topology = topology_with_edges(
        4,
        [(0, 1, 2), (0, 2, 2), (1, 3, 2), (2, 3, 2)],
    )

    assert shortest_path_set(topology, 0, 3) == (
        (0, 1, 3),
        (0, 2, 3),
    )


def test_shortest_path_uses_invbw_cost_instead_of_hop_count():
    topology = topology_with_edges(
        4,
        [(0, 3, 10), (0, 1, 2), (1, 2, 2), (2, 3, 2)],
    )

    assert shortest_path_set(topology, 0, 3) == ((0, 1, 2, 3),)


def test_shortest_paths_respect_link_direction():
    topology = topology_with_edges(3, [(0, 1, 1), (2, 1, 1)])

    assert shortest_path_set(topology, 0, 2) == ()
    assert shortest_path_set(topology, 0, 1) == ((0, 1),)


def test_shortest_path_from_rank_to_itself_is_trivial():
    topology = topology_with_edges(2, [(0, 1, 1)])

    assert shortest_path_set(topology, 1, 1) == ((1,),)


def test_zero_cost_cycles_do_not_create_repeated_rank_paths():
    topology = topology_with_edges(
        4,
        [(0, 1, 0), (1, 2, 0), (2, 1, 0), (1, 3, 1), (2, 3, 1)],
    )

    paths = shortest_path_set(topology, 0, 3)

    assert paths == ((0, 1, 2, 3), (0, 1, 3))
    assert all(len(path) == len(set(path)) for path in paths)


@pytest.mark.parametrize("src,dst", [(-1, 0), (0, 2), (True, 0)])
def test_shortest_path_rejects_invalid_ranks(src, dst):
    topology = topology_with_edges(2, [(0, 1, 1)])

    with pytest.raises(SemanticError, match="rank"):
        shortest_path_set(topology, src, dst)


def test_shortest_path_requires_topology_model():
    with pytest.raises(SemanticError, match="Topology"):
        shortest_path_set({}, 0, 1)
