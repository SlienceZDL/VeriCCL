"""Restore explicitly supplied physical paths behind rank-to-rank links.

    Switch vertices are network facilities, not extra ranks or GPU relays.
    Capacity is charged to each declared resource once per physical transfer;
    the number of path labels does not multiply the payload.
"""
from dataclasses import dataclass
from vericcl.topology.model import LinkKey


@dataclass(frozen=True)
class RestoredPath:
    edge: tuple[int, int]
    physical_path: tuple[str, ...]
    resources: tuple[str, ...]


def restore_physical_paths(topology, edges):
    paths = []
    for src, dst in edges:
        edge = topology.link(LinkKey(src, dst))
        paths.append(RestoredPath((src, dst), edge.physical_path or ('rank:%d' % src, 'rank:%d' % dst), edge.resource_ids))
    return tuple(paths)
