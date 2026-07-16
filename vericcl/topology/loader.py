from typing import Mapping, Optional, Tuple

from vericcl.errors import InputValidationError, SemanticError
from vericcl.input.models import ResolvedInput
from vericcl.topology.model import (
    DirectedLink,
    LinkKey,
    SharedResource,
    Topology,
)
from vericcl.topology.performance import normalize_performance_curve


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


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise InputValidationError("{} must be a non-empty string".format(field))
    return value


def _field(
    value: Mapping[str, object],
    primary: str,
    legacy: Optional[str] = None,
    default: object = None,
) -> object:
    if primary in value:
        return value[primary]
    if legacy is not None and legacy in value:
        return value[legacy]
    return default


def _calibration_points(value: object, field: str) -> dict:
    if value is None:
        return {}
    mapping = _mapping(value, field)
    points = {}
    for raw_concurrency, bandwidth in mapping.items():
        try:
            concurrency = int(raw_concurrency)
        except (TypeError, ValueError) as error:
            raise InputValidationError(
                "{} keys must be integer concurrency values".format(field)
            ) from error
        if str(concurrency) != str(raw_concurrency):
            raise InputValidationError(
                "{} contains a non-canonical concurrency key".format(field)
            )
        points[concurrency] = bandwidth
    return points


def _performance(
    value: Mapping[str, object],
    context: str,
) -> Tuple[object, Tuple[str, ...]]:
    alpha = _field(value, "alpha_us", "alpha")
    if alpha is None:
        raise InputValidationError("{} must define alpha".format(context))
    beta = _field(value, "beta_us", "beta")
    invbw = _field(value, "invbw_us", "invbw")
    bandwidth = _calibration_points(
        value.get("bandwidth_bytes_per_us", {}),
        context + ".bandwidth_bytes_per_us",
    )
    curve, warnings = normalize_performance_curve(
        alpha_us=alpha,
        beta_us=beta,
        invbw_us=invbw,
        bandwidth_bytes_per_us=bandwidth,
    )
    return curve, tuple("{}: {}".format(context, item) for item in warnings)


def _nodes(
    raw_nodes: object,
    rank_count: int,
) -> Tuple[dict, frozenset]:
    membership = {}
    gateways = set()
    nodes = _sequence(raw_nodes, "topology.nodes")
    if not nodes:
        raise InputValidationError("topology.nodes must not be empty")
    node_ids = set()
    for index, raw_node in enumerate(nodes):
        node = _mapping(raw_node, "topology.nodes[{}]".format(index))
        node_id = _integer(node.get("id"), "topology.nodes.id")
        if node_id in node_ids:
            raise InputValidationError("topology node IDs must be unique")
        node_ids.add(node_id)
        ranks = _sequence(node.get("ranks"), "topology.nodes.ranks")
        if not ranks:
            raise InputValidationError("topology node ranks must not be empty")
        for raw_rank in ranks:
            rank = _integer(raw_rank, "topology.nodes.rank")
            if rank >= rank_count:
                raise InputValidationError("topology node rank is out of range")
            if rank in membership:
                raise InputValidationError("topology rank belongs to multiple nodes")
            membership[rank] = node_id
        for raw_gateway in _sequence(
            node.get("gateways", ()),
            "topology.nodes.gateways",
        ):
            gateway = _integer(raw_gateway, "topology.nodes.gateway")
            if gateway not in ranks:
                raise InputValidationError("gateway must belong to its node")
            gateways.add(gateway)
    if set(membership) != set(range(rank_count)):
        raise InputValidationError("topology nodes must cover every rank")
    return membership, frozenset(gateways)


def _links(
    raw_links: object,
    rank_count: int,
) -> Tuple[dict, Tuple[str, ...]]:
    links = {}
    warnings = []
    for index, raw_link in enumerate(
        _sequence(raw_links, "topology.directed_links")
    ):
        value = _mapping(
            raw_link,
            "topology.directed_links[{}]".format(index),
        )
        src = _integer(value.get("src"), "directed_link.src")
        dst = _integer(value.get("dst"), "directed_link.dst")
        if src >= rank_count or dst >= rank_count:
            raise InputValidationError("directed link rank is out of range")
        try:
            key = LinkKey(src, dst)
        except SemanticError as error:
            raise InputValidationError(str(error)) from error
        if key in links:
            raise InputValidationError("directed links must be unique")
        curve, curve_warnings = _performance(
            value,
            "link {}->{}".format(src, dst),
        )
        warnings.extend(curve_warnings)
        resource_ids = tuple(
            _identifier(item, "directed_link.resources")
            for item in _sequence(
                value.get("resources", ()),
                "directed_link.resources",
            )
        )
        try:
            links[key] = DirectedLink(
                key=key,
                max_channels=_integer(
                    value.get("max_channels", 32),
                    "directed_link.max_channels",
                    minimum=1,
                ),
                performance=curve,
                resource_ids=resource_ids,
            )
        except SemanticError as error:
            raise InputValidationError(str(error)) from error
    return links, tuple(warnings)


def _resources(
    raw_resources: object,
) -> Tuple[dict, Tuple[str, ...]]:
    resources = {}
    warnings = []
    for index, raw_resource in enumerate(
        _sequence(raw_resources, "topology.shared_resources")
    ):
        value = _mapping(
            raw_resource,
            "topology.shared_resources[{}]".format(index),
        )
        resource_id = _identifier(value.get("id"), "shared_resource.id")
        if resource_id in resources:
            raise InputValidationError("shared resource IDs must be unique")
        member_links = []
        for raw_member in _sequence(
            value.get("member_links"),
            "shared_resource.member_links",
        ):
            member = _sequence(raw_member, "shared_resource.member_link")
            if len(member) != 2:
                raise InputValidationError(
                    "shared resource member link must contain src and dst"
                )
            try:
                member_links.append(
                    LinkKey(
                        _integer(member[0], "shared_resource.member_link.src"),
                        _integer(member[1], "shared_resource.member_link.dst"),
                    )
                )
            except SemanticError as error:
                raise InputValidationError(str(error)) from error
        curve, curve_warnings = _performance(
            value,
            "resource {}".format(resource_id),
        )
        warnings.extend(curve_warnings)
        try:
            resources[resource_id] = SharedResource(
                resource_id=resource_id,
                member_links=tuple(member_links),
                max_channels=_integer(
                    value.get("max_channels", 32),
                    "shared_resource.max_channels",
                    minimum=1,
                ),
                performance=curve,
            )
        except SemanticError as error:
            raise InputValidationError(str(error)) from error
    return resources, tuple(warnings)


def topology_from_mapping(
    raw_topology: Mapping[str, object],
    expected_rank_count: Optional[int] = None,
) -> Topology:
    raw = _mapping(raw_topology, "topology")
    rank_count = _integer(raw.get("ranks"), "topology.ranks", minimum=1)
    if expected_rank_count is not None and rank_count != expected_rank_count:
        raise InputValidationError(
            "topology rank count does not match resolved input"
        )
    membership, gateways = _nodes(raw.get("nodes"), rank_count)
    links, link_warnings = _links(raw.get("directed_links"), rank_count)
    resources, resource_warnings = _resources(raw.get("shared_resources"))
    source_warnings = tuple(
        _identifier(item, "topology.warnings")
        for item in _sequence(raw.get("warnings", ()), "topology.warnings")
    )
    try:
        return Topology(
            rank_count=rank_count,
            links=links,
            shared_resources=resources,
            node_membership=membership,
            gateways=gateways,
            warnings=source_warnings + link_warnings + resource_warnings,
        )
    except SemanticError as error:
        raise InputValidationError(str(error)) from error


def load_topology(inputs: ResolvedInput) -> Topology:
    if not isinstance(inputs, ResolvedInput):
        raise InputValidationError("inputs must be a ResolvedInput")
    return topology_from_mapping(
        inputs.resolved_topology,
        expected_rank_count=inputs.rank_count,
    )
