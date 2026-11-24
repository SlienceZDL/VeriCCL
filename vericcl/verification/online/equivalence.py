"""Probe equivalence classes: measure one representative, publish to every member.

Two directed links are equivalent when their declared costs, channel limits,
physical path kind, scope (intra/inter host) and shared-resource signatures
(kind, capacity, startup, channel limit, member count) are identical. Two
concurrent modes are equivalent when, under one relabeling of ranks, hosts
and shared resources, they have the same flows (kind, link class, channel),
the same rank and host coincidences and the same resource-sharing pattern.
Local copy/reduce work is equivalent across ranks with identical declared
local costs.

A qualified measurement of a representative is published to the members of
its class with the representative's measurement identity, so the model keeps
the provenance of the actual native evidence. Only declared structure is
used: links that differ in any declared property are measured separately.
"""
import json

from vericcl.performance.profiles import RateProfile
from vericcl.topology.model import LinkKey
from vericcl.verification.search_coverage import digest

from .model_publication import ComponentUpdates, profile_key


def _plain(value):
    if hasattr(value, 'items'):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def parse_flow(flow_id):
    from .probe_jobs import ProbeFlow
    parts = flow_id.split(':')
    if parts[0] in {'send', 'rrc'}:
        return ProbeFlow(parts[0], int(parts[1]), int(parts[2]), int(parts[3][1:]))
    return ProbeFlow(parts[0], int(parts[1]), int(parts[1]))


def flow_resources(flow, topology):
    if flow.kind not in {'send', 'rrc'}:
        return ()
    edge = topology.links[LinkKey(flow.src_rank, flow.dst_rank)]
    return tuple(sorted({'link:%d:%d' % (flow.src_rank, flow.dst_rank), *edge.resource_ids}))


def _service_key(flow):
    return ('link:%d:%d' % (flow.src_rank, flow.dst_rank) if flow.kind == 'send' else
            'rrc:%d:%d' % (flow.src_rank, flow.dst_rank) if flow.kind == 'rrc' else
            '%s:%d' % (flow.kind, flow.src_rank))


class ProbeClasses:
    def __init__(self, inputs, topology):
        config = inputs.request_config
        declared = {(item['src'], item['dst']): item for item in config['topology']['directed_links']}
        declared_resources = {item['id']: item for item in config['topology'].get('shared_resources', ())}
        local = {item['rank']: item for item in config['topology'].get('local_costs', ())}
        self.topology = topology
        self.hosts = {item['rank']: item['host'] for item in config['runtime']['rank_map']}
        self._resource = {}
        for resource_id, resource in topology.shared_resources.items():
            item = declared_resources.get(resource_id, {})
            self._resource[resource_id] = (item.get('kind'), item.get('capacity_bytes_per_us'), item.get('alpha_us'),
                                           resource.max_channels, len(resource.member_links))
        self._link = {}
        for key, edge in topology.links.items():
            item = declared.get((key.src_rank, key.dst_rank), {})
            scope = 'intra' if self.hosts.get(key.src_rank) == self.hosts.get(key.dst_rank) else 'inter'
            self._link[(key.src_rank, key.dst_rank)] = (
                scope, tuple(edge.physical_path[1:-1]), item.get('alpha_us'), item.get('beta_us_per_byte'),
                edge.max_channels, tuple(sorted(self._resource[r] for r in edge.resource_ids)))
        self._local = {}
        for rank in range(inputs.rank_count):
            entry = {k: v for k, v in local.get(rank, {}).items() if k != 'rank'}
            self._local[rank] = json.dumps(_plain(entry), sort_keys=True)
        self._link_members, self._local_members = {}, {}
        for pair in sorted(self._link):
            self._link_members.setdefault(self._link[pair], []).append(pair)
        for rank in sorted(self._local):
            self._local_members.setdefault(self._local[rank], []).append(rank)
        self._representatives = {}  # mode key -> representative flows (key order)
        self.members = {}  # representative mode ids -> {member mode ids}

    # -- classes -----------------------------------------------------------
    def link_classes(self):
        return {cls: tuple(pairs) for cls, pairs in self._link_members.items()}

    def _flow_class(self, flow):
        if flow.kind in {'copy', 'reduce'}:
            return (flow.kind, self._local[flow.src_rank])
        return (flow.kind, self._link[(flow.src_rank, flow.dst_rank)], flow.channel)

    def _ordered(self, flows):
        """Flows in canonical order plus the relabeled structure key."""
        ordered = sorted(flows, key=lambda flow: (repr(self._flow_class(flow)), flow))
        ranks, hosts, resources, items = {}, {}, {}, []
        for flow in ordered:
            src = ranks.setdefault(flow.src_rank, len(ranks))
            dst = ranks.setdefault(flow.dst_rank, len(ranks))
            src_host = hosts.setdefault(self.hosts.get(flow.src_rank), len(hosts))
            dst_host = hosts.setdefault(self.hosts.get(flow.dst_rank), len(hosts))
            used = []
            if flow.kind in {'send', 'rrc'}:
                edge = self.topology.links[LinkKey(flow.src_rank, flow.dst_rank)]
                for resource in sorted(edge.resource_ids, key=lambda r: (repr(self._resource[r]), r)):
                    used.append((resources.setdefault(resource, len(resources)), self._resource[resource]))
            items.append((self._flow_class(flow), src, dst, src_host, dst_host, tuple(sorted(used))))
        return tuple(ordered), repr(tuple(items))

    def representative(self, flows):
        """(representative flows, {representative flow_id: requested flow_id})."""
        ordered, key = self._ordered(tuple(flows))
        representative = self._representatives.setdefault(key, ordered)
        rep_ids = tuple(sorted(flow.flow_id for flow in representative))
        self.members.setdefault(rep_ids, set()).add(tuple(sorted(flow.flow_id for flow in ordered)))
        mapping = {rep.flow_id: flow.flow_id for rep, flow in zip(representative, ordered)}
        return tuple(sorted(representative)), mapping

    def _mapping(self, rep_ids, member_ids):
        rep, rep_key = self._ordered(tuple(parse_flow(f) for f in rep_ids))
        member, member_key = self._ordered(tuple(parse_flow(f) for f in member_ids))
        if rep_key != member_key:
            return None
        return {a.flow_id: b.flow_id for a, b in zip(rep, member)}

    def service_members(self, key):
        flow = parse_flow(key.replace('link:', 'send:', 1) + (':c0' if key.startswith(('link:', 'rrc:')) else ''))
        if flow.kind in {'send', 'rrc'}:
            pairs = self._link_members.get(self._link.get((flow.src_rank, flow.dst_rank)), [(flow.src_rank, flow.dst_rank)])
            prefix = 'link' if flow.kind == 'send' else 'rrc'
            return ['%s:%d:%d' % (prefix, src, dst) for src, dst in pairs]
        return ['%s:%d' % (flow.kind, rank) for rank in self._local_members.get(self._local[flow.src_rank], [flow.src_rank])]

    # -- publication -------------------------------------------------------
    def expand(self, updates):
        """Publish representative components to their class members."""
        services, signatures = dict(updates.services), dict(updates.signatures)
        for key, values in updates.services.items():
            for member in self.service_members(key):
                if member not in services:
                    services[member] = values
                    signatures['service:' + member] = updates.signatures['service:' + key]
        direct = {profile_key(profile) for profile in updates.profiles}
        profiles = list(updates.profiles)
        for profile in updates.profiles:
            source_key = profile_key(profile)
            for member_ids in sorted(self.members.get(tuple(sorted(profile.mode)), ())):
                mapping = self._mapping(tuple(profile.mode), member_ids)
                if mapping is None or tuple(sorted(mapping.values())) == tuple(sorted(profile.mode)):
                    continue
                membership = {mapping[f]: flow_resources(parse_flow(mapping[f]), self.topology) for f in profile.mode}
                resources = tuple(sorted({r for values in membership.values() for r in values}))
                body = dict(mode=tuple(mapping[f] for f in profile.mode),
                            rates={mapping[f]: rate for f, rate in profile.rates.items()}, resources=resources,
                            min_bytes=profile.min_bytes, max_bytes=profile.max_bytes, source=profile.source,
                            measured_at=profile.measured_at,
                            startup_us={mapping[f]: value for f, value in profile.startup_us.items()},
                            flow_resources=membership)
                member = RateProfile('fit-' + digest({'class_of': profile.profile_id, 'mode': sorted(body['mode']),
                                                      'begin': profile.min_bytes})[:24], **body)
                key = profile_key(member)
                if key in direct:
                    continue
                profiles.append(member)
                signatures[key] = updates.signatures.get(source_key)
        signatures = {key: value for key, value in signatures.items() if value is not None}
        return ComponentUpdates(services, tuple(profiles), signatures, updates.issues, updates.records)

    def known_updates(self, model, flows):
        """Measured components of a representative already in ``model``, or None."""
        signatures = model.metadata.get('calibration_signatures', {})
        flows = tuple(sorted(flows))
        services, profiles, known = {}, [], {}
        if len(flows) == 1 and flows[0].channel == 0:
            key = _service_key(flows[0])
            values = tuple(v for v in model.services.get(key, ()) if v.source == 'measured' and v.qualified)
            if not values or 'service:' + key not in signatures:
                return None
            services[key] = values
            known['service:' + key] = signatures['service:' + key]
        if all(flow.kind in {'send', 'rrc'} for flow in flows):
            mode = tuple(sorted(flow.flow_id for flow in flows))
            matches = tuple(p for p in model.profiles if p.mode == mode and p.source == 'measured')
            if not matches:
                return None
            key = profile_key(matches[0])
            if key not in signatures:
                return None
            profiles.extend(p for p in matches if profile_key(p) == key)
            known[key] = signatures[key]
        if not services and not profiles:
            return None
        return ComponentUpdates(services, tuple(profiles), known)
