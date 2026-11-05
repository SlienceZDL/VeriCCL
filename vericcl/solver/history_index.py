"""Small topology-indexed schedule reuse; every adapted result is lowered afresh."""
from dataclasses import dataclass, replace
import fcntl
from functools import lru_cache
import json
import os
from pathlib import Path
import uuid
import warnings

import networkx as nx

from vericcl.artifacts.writer import atomic_write_bytes
from vericcl.errors import SemanticError
from vericcl.verification.search_coverage import digest
from vericcl.workflow_state import plain
from vericcl.xml.v5_model import NetworkCandidate, NetworkOperation


def _graph(topology, collective):
    graph = nx.DiGraph()
    for rank in range(topology['ranks']):
        graph.add_node('r:' + str(rank), kind='root' if rank == collective.get('root') else 'rank')
    for edge in topology['directed_links']:
        node = 'e:%d:%d' % (edge['src'], edge['dst'])
        graph.add_node(node, kind='link')
        graph.add_edge('r:' + str(edge['src']), node)
        graph.add_edge(node, 'r:' + str(edge['dst']))
    for resource in topology.get('shared_resources', ()):
        node = 's:' + resource['id']
        graph.add_node(node, kind='resource:' + resource['kind'])
        for src, dst in resource['links']:
            graph.add_edge('e:%d:%d' % (src, dst), node)
    return graph


def _bucket(topology, collective):
    graph = _graph(topology, collective)
    semantic = {key: collective.get(key) for key in ('operator', 'datatype', 'reduction_op', 'inplace')}
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='The hashes produced for directed graphs changed in version v3.5.*', category=UserWarning)
        fingerprint = nx.weisfeiler_lehman_graph_hash(graph, node_attr='kind', iterations=3)
    return digest({'graph': fingerprint, 'semantic': semantic})


@lru_cache(maxsize=2)
def _read(path, device, inode, modified, changed, size):
    value = json.loads(Path(path).read_text())
    if value.get('schema_version') != 5 or not isinstance(value.get('buckets'), dict):
        raise SemanticError('invalid schedule history index')
    return value


def _load(path):
    try:
        stat = path.stat()
    except FileNotFoundError:
        return {'schema_version': 5, 'buckets': {}}
    return _read(str(path.resolve()), stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)


def save_schedule(path, bundle, inputs):
    if not bundle.can_execute:
        raise SemanticError('only fully checked schedules enter history')
    from vericcl.tuning.strategies import _network
    path = Path(path)
    config = inputs.request_config
    payload = {'xml_sha256': bundle.xml_sha256, 'topology': plain(config['topology']),
        'collective': plain(config['sketch']['collective']), 'slice_count': inputs.hyperparameters.slice_count,
        'slice_bytes': inputs.hyperparameters.slice_size_bytes, 'channels': bundle.program.channels,
        't_pred_us': bundle.prediction.t_pred_us, 'model_version': bundle.prediction.model_version,
        'operations': plain(_network(bundle))}
    bucket = _bucket(payload['topology'], payload['collective'])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        cache = _load(path)
        buckets = dict(cache['buckets'])
        records = [record for record in buckets.get(bucket, []) if record['payload']['xml_sha256'] != bundle.xml_sha256]
        records.append({'payload': payload, 'sha256': digest(payload)})
        # A bounded, replaceable search accelerator, never proof coverage.
        records.sort(key=lambda item: item['payload']['t_pred_us'] /
                     (item['payload']['slice_count'] * item['payload']['slice_bytes']))
        buckets[bucket] = records[:32]
        temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
        atomic_write_bytes(temporary, (json.dumps({'schema_version': 5, 'buckets': buckets}, allow_nan=False) + '\n').encode())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


@dataclass(frozen=True)
class ScheduleHit:
    payload: dict
    rank_mapping: dict
    bandwidth_ratio_spread: float


def find_schedules(path, inputs, *, tolerance, maximum, lease):
    if tolerance < 0 or isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
        raise SemanticError('schedule lookup needs a nonnegative ratio tolerance and positive candidate limit')
    topology = plain(inputs.request_config['topology'])
    collective = plain(inputs.request_config['sketch']['collective'])
    graph = _graph(topology, collective)
    bucket = _bucket(topology, collective)
    hits = []
    # At most `maximum` records are inspected; misses have no persistent log.
    for record in _load(Path(path))['buckets'].get(bucket, ())[:maximum]:
        lease.check()
        payload = record['payload']
        if digest(payload) != record['sha256']:
            continue
        old_graph = _graph(payload['topology'], payload['collective'])
        labels = {key: key for key in old_graph}
        if set(old_graph) == set(graph) and set(old_graph.edges) == set(graph.edges) and all(
                old_graph.nodes[key]['kind'] == graph.nodes[key]['kind'] for key in old_graph):
            mapping = labels
        else:
            class BoundedMatcher(nx.algorithms.isomorphism.DiGraphMatcher):
                def semantic_feasibility(self, left, right):
                    lease.check()
                    return super().semantic_feasibility(left, right)
            matcher = BoundedMatcher(old_graph, graph,
                node_match=lambda left, right: left['kind'] == right['kind'])
            mapping = next(matcher.isomorphisms_iter(), None)
        lease.check()
        if mapping is None:
            continue
        rank_map = {rank: int(mapping['r:' + str(rank)].split(':')[1]) for rank in range(inputs.rank_count)}
        new_links = {(edge['src'], edge['dst']): edge for edge in topology['directed_links']}
        ratios = [edge['beta_us_per_byte'] / new_links[(rank_map[edge['src']], rank_map[edge['dst']])]['beta_us_per_byte']
                  for edge in payload['topology']['directed_links']]
        new_resources = {resource['id']: resource for resource in topology.get('shared_resources', ())}
        for resource in payload['topology'].get('shared_resources', ()):
            new_id = mapping['s:' + resource['id']][2:]
            ratios.append(new_resources[new_id]['capacity_bytes_per_us'] / resource['capacity_bytes_per_us'])
        spread = max(ratios) / min(ratios) - 1 if ratios else 0.
        if spread <= tolerance:
            hits.append(ScheduleHit(payload, rank_map, spread))
    return tuple(hits)


def adapt_schedule(hit, inputs):
    """Map complete logical slices; no fractional data or copied timing survives."""
    old_n, new_n = hit.payload['slice_count'], inputs.hyperparameters.slice_count
    if new_n is None:
        raise SemanticError('history adaptation requires resolved integer S')
    ranks = inputs.rank_count
    operator = inputs.collective.kind.value
    def old_position(new_position):
        if operator in {'alltoall', 'reduce_scatter'}:
            # Destination blocks remain owned by their mapped rank.
            per_new, per_old = new_n // ranks, old_n // ranks
            new_owner, local = divmod(new_position, per_new)
            old_owner = next(old for old, new in hit.rank_mapping.items() if new == new_owner)
            return old_owner * per_old + local * per_old // per_new
        return new_position * old_n // new_n
    targets = {}
    for new_position in range(new_n):
        targets.setdefault(old_position(new_position), []).append(new_position)
    raw = hit.payload['operations']
    cloned = {}
    for operation in raw:
        positions = {value % old_n for value in operation['contributors']}
        if len(positions) != 1:
            raise SemanticError('history operation mixes logical slice positions')
        position = next(iter(positions))
        cloned[operation['operation_id']] = tuple(targets.get(position, ()))
    def identifier(key, position):
        return 'history-' + key + '-p' + str(position)
    operations = []
    for operation in raw:
        for position in cloned[operation['operation_id']]:
            def contributors(values):
                return frozenset(hit.rank_mapping[value // old_n] * new_n + position for value in values)
            dependencies = set()
            for dependency in operation['dependencies']:
                choices = cloned.get(dependency, ())
                dependencies.update(identifier(dependency, value) for value in
                                    ((position,) if position in choices else choices))
            operations.append(NetworkOperation(identifier(operation['operation_id'], position), operation['kind'],
                hit.rank_mapping[operation['src_rank']], hit.rank_mapping[operation['dst_rank']],
                operation['channel'], operation['stage_id'], contributors(operation['contributors']),
                contributors(operation['target_contributors']),
                tuple(hit.rank_mapping[rank] for rank in operation['participants']), frozenset(dependencies),
                frozenset(), 0.))
    return NetworkCandidate(ranks, new_n, inputs.hyperparameters.slice_size_bytes, tuple(operations), 'history')
