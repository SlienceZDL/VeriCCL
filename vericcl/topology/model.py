import hashlib
import json
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import FrozenSet, Mapping, Tuple

from vericcl.errors import SemanticError


def _integer(value: object, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SemanticError("{} must be an integer".format(field))
    if value < minimum:
        raise SemanticError("{} must be at least {}".format(field, minimum))
    return value


def _number(value: object, field: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SemanticError("{} must be a number".format(field))
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < minimum:
        raise SemanticError(
            "{} must be finite and at least {}".format(field, minimum)
        )
    return normalized


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise SemanticError("{} must be a non-empty string".format(field))
    return value


@dataclass(frozen=True, order=True)
class LinkKey:
    src_rank: int
    dst_rank: int

    def __post_init__(self) -> None:
        _integer(self.src_rank, "link.src_rank")
        _integer(self.dst_rank, "link.dst_rank")
        if self.src_rank == self.dst_rank:
            raise SemanticError("link ranks must be distinct")


@dataclass(frozen=True, order=True)
class LaneKey:
    src_rank: int
    dst_rank: int
    channel: int

    def __post_init__(self) -> None:
        LinkKey(self.src_rank, self.dst_rank)
        _integer(self.channel, "lane.channel")


@dataclass(frozen=True)
class PerformanceCurve:
    alpha_us: float
    invbw_us: float
    bandwidth_bytes_per_us: Mapping[int, float]

    def __post_init__(self) -> None:
        alpha_us = _number(self.alpha_us, "performance.alpha_us")
        invbw_us = _number(self.invbw_us, "performance.invbw_us")
        if invbw_us < alpha_us:
            raise SemanticError("performance.invbw_us must not be below alpha_us")
        try:
            raw_points = dict(self.bandwidth_bytes_per_us)
        except (TypeError, ValueError) as error:
            raise SemanticError("calibration points must be a mapping") from error
        points = {}
        for concurrency, bandwidth in raw_points.items():
            normalized_concurrency = _integer(
                concurrency,
                "calibration concurrency",
                minimum=1,
            )
            points[normalized_concurrency] = _number(
                bandwidth,
                "calibration bandwidth",
                minimum=1e-300,
            )
        object.__setattr__(self, "alpha_us", alpha_us)
        object.__setattr__(self, "invbw_us", invbw_us)
        object.__setattr__(
            self,
            "bandwidth_bytes_per_us",
            MappingProxyType(dict(sorted(points.items()))),
        )

    @property
    def beta_effective_us(self) -> float:
        return self.invbw_us - self.alpha_us

    @property
    def is_calibrated(self) -> bool:
        return bool(self.bandwidth_bytes_per_us)


@dataclass(frozen=True)
class DirectedLink:
    key: LinkKey
    max_channels: int
    performance: PerformanceCurve
    resource_ids: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.key, LinkKey):
            raise SemanticError("directed_link.key must be a LinkKey")
        _integer(self.max_channels, "directed_link.max_channels", minimum=1)
        if not isinstance(self.performance, PerformanceCurve):
            raise SemanticError(
                "directed_link.performance must be a PerformanceCurve"
            )
        resource_ids = tuple(self.resource_ids)
        for resource_id in resource_ids:
            _identifier(resource_id, "directed_link.resource_ids")
        if len(resource_ids) != len(set(resource_ids)):
            raise SemanticError("directed_link.resource_ids must be unique")
        if self.performance.bandwidth_bytes_per_us:
            maximum_point = max(self.performance.bandwidth_bytes_per_us)
            if maximum_point > self.max_channels:
                raise SemanticError(
                    "calibration concurrency exceeds link max_channels"
                )
        object.__setattr__(self, "resource_ids", tuple(sorted(resource_ids)))


@dataclass(frozen=True)
class SharedResource:
    resource_id: str
    member_links: Tuple[LinkKey, ...]
    max_channels: int
    performance: PerformanceCurve

    def __post_init__(self) -> None:
        _identifier(self.resource_id, "shared_resource.resource_id")
        links = tuple(self.member_links)
        if not links or not all(isinstance(key, LinkKey) for key in links):
            raise SemanticError(
                "shared_resource.member_links must contain LinkKey values"
            )
        if len(links) != len(set(links)):
            raise SemanticError("shared_resource.member_links must be unique")
        _integer(self.max_channels, "shared_resource.max_channels", minimum=1)
        if not isinstance(self.performance, PerformanceCurve):
            raise SemanticError(
                "shared_resource.performance must be a PerformanceCurve"
            )
        if self.performance.bandwidth_bytes_per_us:
            maximum_point = max(self.performance.bandwidth_bytes_per_us)
            if maximum_point > self.max_channels:
                raise SemanticError(
                    "calibration concurrency exceeds resource max_channels"
                )
        object.__setattr__(self, "member_links", tuple(sorted(links)))


@dataclass(frozen=True)
class Topology:
    rank_count: int
    links: Mapping[LinkKey, DirectedLink]
    shared_resources: Mapping[str, SharedResource]
    node_membership: Mapping[int, int]
    gateways: FrozenSet[int]
    warnings: Tuple[str, ...]
    isomorphism_signature: str = ""

    def __post_init__(self) -> None:
        _integer(self.rank_count, "topology.rank_count", minimum=1)
        links = self._normalize_links()
        resources = self._normalize_resources(links)
        membership = self._normalize_membership()
        gateways = frozenset(self.gateways)
        for rank in gateways:
            _integer(rank, "topology.gateways")
            if rank >= self.rank_count:
                raise SemanticError("topology gateway is outside the rank range")
        warnings = tuple(self.warnings)
        for warning in warnings:
            _identifier(warning, "topology.warnings")
        object.__setattr__(self, "links", MappingProxyType(links))
        object.__setattr__(
            self,
            "shared_resources",
            MappingProxyType(resources),
        )
        object.__setattr__(
            self,
            "node_membership",
            MappingProxyType(membership),
        )
        object.__setattr__(self, "gateways", gateways)
        object.__setattr__(self, "warnings", warnings)
        if self.isomorphism_signature:
            _identifier(
                self.isomorphism_signature,
                "topology.isomorphism_signature",
            )
        else:
            object.__setattr__(
                self,
                "isomorphism_signature",
                self._compute_signature(),
            )

    def _normalize_links(self) -> dict:
        try:
            raw_links = dict(self.links)
        except (TypeError, ValueError) as error:
            raise SemanticError("topology.links must be a mapping") from error
        links = {}
        for key, edge in raw_links.items():
            if not isinstance(key, LinkKey) or not isinstance(edge, DirectedLink):
                raise SemanticError(
                    "topology.links must map LinkKey to DirectedLink"
                )
            if key != edge.key:
                raise SemanticError("topology link key does not match its value")
            if key.src_rank >= self.rank_count or key.dst_rank >= self.rank_count:
                raise SemanticError("topology link rank is outside the rank range")
            links[key] = edge
        return dict(sorted(links.items()))

    def _normalize_resources(
        self,
        links: Mapping[LinkKey, DirectedLink],
    ) -> dict:
        try:
            raw_resources = dict(self.shared_resources)
        except (TypeError, ValueError) as error:
            raise SemanticError("topology.shared_resources must be a mapping") from error
        resources = {}
        for resource_id, resource in raw_resources.items():
            if (
                not isinstance(resource_id, str)
                or not isinstance(resource, SharedResource)
            ):
                raise SemanticError("invalid shared resource mapping")
            if resource_id != resource.resource_id:
                raise SemanticError(
                    "shared resource key does not match its resource_id"
                )
            for key in resource.member_links:
                if key not in links:
                    raise SemanticError("shared resource references an unknown link")
                if resource_id not in links[key].resource_ids:
                    raise SemanticError(
                        "shared resource membership is not declared by its link"
                    )
            resources[resource_id] = resource
        for key, edge in links.items():
            for resource_id in edge.resource_ids:
                if resource_id not in resources:
                    raise SemanticError(
                        "link references an unknown shared resource"
                    )
                if key not in resources[resource_id].member_links:
                    raise SemanticError(
                        "link resource is missing direction-specific membership"
                    )
        return dict(sorted(resources.items()))

    def _normalize_membership(self) -> dict:
        try:
            membership = dict(self.node_membership)
        except (TypeError, ValueError) as error:
            raise SemanticError("topology.node_membership must be a mapping") from error
        if set(membership) != set(range(self.rank_count)):
            raise SemanticError("topology.node_membership must cover every rank")
        for rank, node in membership.items():
            _integer(rank, "topology.node_membership rank")
            _integer(node, "topology.node_membership node")
        return dict(sorted(membership.items()))

    def _compute_signature(self) -> str:
        value = {
            "rank_count": self.rank_count,
            "links": [
                {
                    "src": key.src_rank,
                    "dst": key.dst_rank,
                    "max_channels": edge.max_channels,
                    "alpha_us": edge.performance.alpha_us,
                    "invbw_us": edge.performance.invbw_us,
                    "bandwidth": dict(edge.performance.bandwidth_bytes_per_us),
                    "resources": edge.resource_ids,
                }
                for key, edge in self.links.items()
            ],
            "resources": [
                {
                    "id": resource_id,
                    "members": [
                        (key.src_rank, key.dst_rank)
                        for key in resource.member_links
                    ],
                    "max_channels": resource.max_channels,
                    "alpha_us": resource.performance.alpha_us,
                    "invbw_us": resource.performance.invbw_us,
                    "bandwidth": dict(
                        resource.performance.bandwidth_bytes_per_us
                    ),
                }
                for resource_id, resource in self.shared_resources.items()
            ],
            "node_membership": dict(self.node_membership),
            "gateways": sorted(self.gateways),
        }
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def has_link(self, src_rank: int, dst_rank: int) -> bool:
        try:
            key = LinkKey(src_rank, dst_rank)
        except SemanticError:
            return False
        return key in self.links

    def link(self, key: LinkKey) -> DirectedLink:
        try:
            return self.links[key]
        except KeyError as error:
            raise SemanticError("topology does not contain the requested link") from error

    def destinations(self, src_rank: int) -> Tuple[int, ...]:
        _integer(src_rank, "src_rank")
        return tuple(
            key.dst_rank for key in self.links if key.src_rank == src_rank
        )

    def sources(self, dst_rank: int) -> Tuple[int, ...]:
        _integer(dst_rank, "dst_rank")
        return tuple(
            key.src_rank for key in self.links if key.dst_rank == dst_rank
        )

    def lanes(
        self,
        key: LinkKey,
        channel_count: int,
    ) -> Tuple[LaneKey, ...]:
        edge = self.link(key)
        normalized_count = _integer(channel_count, "channel_count", minimum=1)
        if normalized_count > edge.max_channels:
            raise SemanticError("channel_count exceeds link max_channels")
        return tuple(
            LaneKey(key.src_rank, key.dst_rank, channel)
            for channel in range(normalized_count)
        )

    def resources_for(self, key: LinkKey) -> Tuple[str, ...]:
        return self.link(key).resource_ids
