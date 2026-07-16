from copy import deepcopy
from typing import Mapping, Tuple

from vericcl.errors import InputValidationError


_NDV2_CLASSES = (
    (0, 1, 2, 2, 1, 0, 0, 0),
    (1, 0, 2, 1, 0, 2, 0, 0),
    (2, 2, 0, 1, 0, 0, 1, 0),
    (2, 1, 1, 0, 0, 0, 0, 2),
    (1, 0, 0, 0, 0, 1, 2, 2),
    (0, 2, 0, 0, 1, 0, 2, 1),
    (0, 0, 1, 0, 2, 2, 0, 1),
    (0, 0, 0, 2, 2, 1, 1, 0),
)


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise InputValidationError("{} must be a JSON object".format(field))
    return value


def _sequence(value: object, field: str) -> tuple:
    if not isinstance(value, (list, tuple)):
        raise InputValidationError("{} must be a JSON array".format(field))
    return tuple(value)


def _integer(value: object, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InputValidationError("{} must be an integer".format(field))
    if value < minimum:
        raise InputValidationError("{} must be at least {}".format(field, minimum))
    return value


def _square_matrix(value: object, size: int, field: str) -> Tuple[tuple, ...]:
    rows = _sequence(value, field)
    if len(rows) != size:
        raise InputValidationError("{} has the wrong row count".format(field))
    normalized = []
    for row in rows:
        values = _sequence(row, field)
        if len(values) != size:
            raise InputValidationError("{} has the wrong column count".format(field))
        normalized.append(values)
    return tuple(normalized)


def _derived_node_matrices(
    raw_topology: Mapping[str, object],
    gpus_per_node: int,
) -> Tuple[Tuple[tuple, ...], Tuple[tuple, ...], Tuple[tuple, ...]]:
    if all(key in raw_topology for key in ("links", "betas", "invbws")):
        return (
            _square_matrix(raw_topology["links"], gpus_per_node, "links"),
            _square_matrix(raw_topology["betas"], gpus_per_node, "betas"),
            _square_matrix(raw_topology["invbws"], gpus_per_node, "invbws"),
        )
    beta_classes = _sequence(
        raw_topology.get("node_betas_list"),
        "node_betas_list",
    )
    invbw_classes = _sequence(
        raw_topology.get("node_invbws_list"),
        "node_invbws_list",
    )
    if len(beta_classes) != len(invbw_classes) or not beta_classes:
        raise InputValidationError(
            "node beta and invbw classes must be non-empty and aligned"
        )
    name = str(raw_topology.get("name", "")).lower()
    if "ndv2" in name:
        if gpus_per_node != 8 or len(beta_classes) < 2:
            raise InputValidationError("NDv2 legacy topology requires two 8-rank classes")
        classes = _NDV2_CLASSES
    elif "fit8" in name and gpus_per_node == 8 and len(beta_classes) >= 2:
        classes = tuple(
            tuple(
                0
                if src == dst
                else 1
                if src // 4 == dst // 4
                else 2
                for src in range(gpus_per_node)
            )
            for dst in range(gpus_per_node)
        )
    else:
        classes = tuple(
            tuple(0 if src == dst else 1 for src in range(gpus_per_node))
            for dst in range(gpus_per_node)
        )
    links = []
    betas = []
    invbws = []
    for dst in range(gpus_per_node):
        link_row = []
        beta_row = []
        invbw_row = []
        for src in range(gpus_per_node):
            class_id = classes[dst][src]
            link_row.append(1 if class_id else 0)
            beta_row.append(0 if class_id == 0 else beta_classes[class_id - 1])
            invbw_row.append(0 if class_id == 0 else invbw_classes[class_id - 1])
        links.append(tuple(link_row))
        betas.append(tuple(beta_row))
        invbws.append(tuple(invbw_row))
    return tuple(links), tuple(betas), tuple(invbws)


def _internode_connections(value: object, gpus_per_node: int) -> dict:
    if value == "fully-connected":
        return {rank: tuple(range(gpus_per_node)) for rank in range(gpus_per_node)}
    if value == "direct-map":
        return {rank: (rank,) for rank in range(gpus_per_node)}
    mapping = _mapping(value, "internode_conn")
    result = {}
    for raw_src, raw_destinations in mapping.items():
        try:
            src = int(raw_src)
        except (TypeError, ValueError) as error:
            raise InputValidationError("internode source must be an integer") from error
        if src < 0 or src >= gpus_per_node:
            raise InputValidationError("internode source is out of range")
        destinations = tuple(
            _integer(item, "internode destination")
            for item in _sequence(raw_destinations, "internode destinations")
        )
        if any(item >= gpus_per_node for item in destinations):
            raise InputValidationError("internode destination is out of range")
        if len(destinations) != len(set(destinations)):
            raise InputValidationError("internode destinations must be unique")
        result[src] = tuple(sorted(destinations))
    return dict(sorted(result.items()))


def _performance_fields(
    alpha: object,
    beta: object,
    invbw: object,
) -> dict:
    return {"alpha": alpha, "beta": beta, "invbw": invbw}


def _add_resource(
    resources: list,
    link_records: dict,
    resource_id: str,
    members: tuple,
    performance: Mapping[str, object],
) -> None:
    if not members:
        return
    resources.append(
        {
            "id": resource_id,
            "member_links": [list(member) for member in sorted(members)],
            "max_channels": 32,
            **dict(performance),
        }
    )
    for member in members:
        link_records[member]["resources"].append(resource_id)


def _add_switch_resources(
    resources: list,
    link_records: dict,
    raw_sketch: Mapping[str, object],
    node_count: int,
    gpus_per_node: int,
) -> None:
    intranode = _mapping(
        raw_sketch.get("intranode_sketch", {"strategy": "none"}),
        "intranode_sketch",
    )
    if intranode.get("strategy") != "switch":
        return
    switches = _sequence(intranode.get("switches"), "intranode switches")
    for node in range(node_count):
        base = node * gpus_per_node
        for switch_index, raw_group in enumerate(switches):
            group = tuple(
                _integer(item, "switch rank")
                for item in _sequence(raw_group, "switch group")
            )
            if any(rank >= gpus_per_node for rank in group):
                raise InputValidationError("switch rank is out of range")
            if len(group) != len(set(group)):
                raise InputValidationError("switch ranks must be unique")
            for local_rank in group:
                rank = base + local_rank
                outgoing = tuple(
                    (rank, base + peer)
                    for peer in group
                    if peer != local_rank and (rank, base + peer) in link_records
                )
                incoming = tuple(
                    (base + peer, rank)
                    for peer in group
                    if peer != local_rank and (base + peer, rank) in link_records
                )
                for direction, members in (
                    ("egress", outgoing),
                    ("ingress", incoming),
                ):
                    if not members:
                        continue
                    sample = link_records[members[0]]
                    performance = {
                        key: sample[key] for key in ("alpha", "beta", "invbw")
                    }
                    _add_resource(
                        resources,
                        link_records,
                        "node-{}-switch-{}-rank-{}-{}".format(
                            node,
                            switch_index,
                            rank,
                            direction,
                        ),
                        members,
                        performance,
                    )


def convert_legacy_topology(
    raw_topology: Mapping[str, object],
    raw_sketch: Mapping[str, object],
) -> Mapping[str, object]:
    topology_source = deepcopy(dict(_mapping(raw_topology, "legacy topology")))
    sketch_source = deepcopy(dict(_mapping(raw_sketch, "legacy sketch")))
    gpus_per_node = _integer(
        topology_source.get("gpus_per_node"),
        "gpus_per_node",
        minimum=1,
    )
    node_count = _integer(sketch_source.get("nnodes"), "nnodes", minimum=1)
    rank_count = gpus_per_node * node_count
    links, betas, invbws = _derived_node_matrices(
        topology_source,
        gpus_per_node,
    )
    alpha = topology_source.get("alpha")
    if alpha is None:
        raise InputValidationError("legacy topology must define alpha")
    link_records = {}
    for node in range(node_count):
        base = node * gpus_per_node
        for src_local in range(gpus_per_node):
            for dst_local in range(gpus_per_node):
                if not links[dst_local][src_local]:
                    continue
                src = base + src_local
                dst = base + dst_local
                link_records[(src, dst)] = {
                    "src": src,
                    "dst": dst,
                    "max_channels": 32,
                    **_performance_fields(
                        alpha,
                        betas[dst_local][src_local],
                        invbws[dst_local][src_local],
                    ),
                    "resources": [],
                }

    resources = []
    _add_switch_resources(
        resources,
        link_records,
        sketch_source,
        node_count,
        gpus_per_node,
    )
    gateway_ranks = set()
    if node_count > 1:
        internode_sketch = _mapping(
            sketch_source.get("internode_sketch"),
            "internode_sketch",
        )
        if internode_sketch.get("strategy") != "relay":
            raise InputValidationError("legacy internode strategy must be relay")
        connections = _internode_connections(
            internode_sketch.get("internode_conn"),
            gpus_per_node,
        )
        remote = _performance_fields(
            topology_source.get("remote_alpha"),
            topology_source.get("remote_beta"),
            topology_source.get("remote_invbw"),
        )
        if remote["alpha"] is None or (
            remote["beta"] is None and remote["invbw"] is None
        ):
            raise InputValidationError(
                "legacy remote performance requires alpha and beta or invbw"
            )
        node_pair_members = {}
        egress_members = {node: [] for node in range(node_count)}
        ingress_members = {node: [] for node in range(node_count)}
        for src_node in range(node_count):
            for dst_node in range(node_count):
                if src_node == dst_node:
                    continue
                members = []
                for src_local, destinations in connections.items():
                    for dst_local in destinations:
                        src = src_node * gpus_per_node + src_local
                        dst = dst_node * gpus_per_node + dst_local
                        key = (src, dst)
                        if key in link_records:
                            raise InputValidationError("legacy conversion produced duplicate link")
                        link_records[key] = {
                            "src": src,
                            "dst": dst,
                            "max_channels": 32,
                            **remote,
                            "resources": [],
                        }
                        members.append(key)
                        egress_members[src_node].append(key)
                        ingress_members[dst_node].append(key)
                        gateway_ranks.update((src, dst))
                node_pair_members[(src_node, dst_node)] = tuple(members)
        for (src_node, dst_node), members in sorted(node_pair_members.items()):
            _add_resource(
                resources,
                link_records,
                "inter-node-{}-to-{}".format(src_node, dst_node),
                members,
                remote,
            )
        for node in range(node_count):
            _add_resource(
                resources,
                link_records,
                "nic-node-{}-egress".format(node),
                tuple(egress_members[node]),
                remote,
            )
            _add_resource(
                resources,
                link_records,
                "nic-node-{}-ingress".format(node),
                tuple(ingress_members[node]),
                remote,
            )

    nodes = []
    for node in range(node_count):
        ranks = list(
            range(node * gpus_per_node, (node + 1) * gpus_per_node)
        )
        nodes.append(
            {
                "id": node,
                "ranks": ranks,
                "gateways": [rank for rank in ranks if rank in gateway_ranks],
            }
        )
    for record in link_records.values():
        record["resources"] = sorted(record["resources"])
    return {
        "name": str(topology_source.get("name", "legacy-topology")),
        "ranks": rank_count,
        "nodes": nodes,
        "directed_links": [
            link_records[key] for key in sorted(link_records)
        ],
        "shared_resources": sorted(resources, key=lambda item: item["id"]),
        "provenance": {
            "legacy_format": "taccl_topology_v2",
            "source_topology": topology_source,
            "source_sketch": sketch_source,
        },
    }
