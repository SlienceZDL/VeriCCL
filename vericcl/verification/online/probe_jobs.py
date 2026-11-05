"""Independent native-calibration jobs; planning never changes application M/S."""
from dataclasses import asdict, dataclass, field, replace
from itertools import combinations
import uuid

from vericcl.errors import SemanticError
from vericcl.execution.budget import BudgetExhausted
from vericcl.execution.context import StorageFailure
from vericcl.input.schema import DATATYPE_BYTES
from vericcl.topology.model import LinkKey
from vericcl.verification.search_coverage import digest
from vericcl.xml.capacity import simple_payload_capacity


@dataclass(frozen=True, order=True)
class ProbeFlow:
    kind: str
    src_rank: int
    dst_rank: int
    channel: int = 0

    def __post_init__(self):
        if self.kind not in {'send', 'rrc', 'copy', 'reduce'}:
            raise SemanticError('unknown probe primitive')
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (self.src_rank, self.dst_rank, self.channel)):
            raise SemanticError('invalid probe rank or channel')
        if (self.src_rank == self.dst_rank) != (self.kind in {'copy', 'reduce'}):
            raise SemanticError('communication and local probe endpoints disagree')
        if self.channel >= 16 or (self.kind in {'copy', 'reduce'} and self.channel != 0):
            raise SemanticError('invalid probe channel')

    @property
    def flow_id(self):
        return ('%s:%d:%d:c%d' % (self.kind, self.src_rank, self.dst_rank, self.channel)
                if self.kind in {'send', 'rrc'} else '%s:%d' % (self.kind, self.src_rank))


@dataclass(frozen=True)
class ProbeJob:
    job_id: str
    category: str
    host_scope: str
    flows: tuple[ProbeFlow, ...]
    payload_bytes: int
    datatype: str
    reduction_op: str
    rank_ids: tuple[int, ...]
    rank_map: tuple[dict, ...]
    resources: tuple[str, ...]
    launches: int
    warmup_iterations: int
    timed_iterations: int
    max_rounds: int
    max_cv: float
    timeout_s: float
    flow_resources: tuple = ()
    execution_id: str | None = None
    topology_facts: dict = field(default_factory=dict)


def probe_artifact_key(job):
    if job.execution_id is None:
        return job.job_id
    if not job.execution_id or any(not (c.isalnum() or c in '_-') for c in job.execution_id):
        raise SemanticError('invalid probe execution identity')
    return job.job_id + '/attempt-' + job.execution_id


@dataclass(frozen=True)
class ProbeResult:
    job_id: str
    status: str
    driver_kind: str
    samples: tuple = ()
    message: str | None = None
    artifact_paths: tuple = ()
    cleanup_confirmed: bool = False
    gpu_released: bool = False
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.driver_kind not in {'native_msccl', 'test_simulation'}:
            raise SemanticError('probe results must explicitly name their evidence source')


@dataclass
class ProbeContext:
    inputs: object
    topology: object
    lease: object
    observed_modes: tuple = ()
    run_context: object = None
    records: list = field(default_factory=list)
    native_driver: object = None


def payload_sizes(context):
    inputs = context.inputs
    element = DATATYPE_BYTES[inputs.collective.datatype]
    maximum = simple_payload_capacity(int(inputs.request_config['runtime']['environment']['NCCL_BUFFSIZE']), element)
    size = inputs.hyperparameters.slice_size_bytes
    if size is None or size > maximum:
        raise SemanticError('native calibration requires a resolved Simple-compatible application S')
    supplied = tuple(inputs.request_config['sketch']['calibration']['payload_sizes'])
    if any(isinstance(v, bool) or not isinstance(v, int) or v < element or v % element or v > maximum for v in supplied):
        raise SemanticError('probe payload violates datatype or Simple capacity')
    values = {size, *supplied}
    if not supplied:
        values.update((max(element, (size // 2 // element) * element), min(maximum // element * element, size * 2)))
    for candidate in (element, element * 2, element * 3, maximum // element * element):
        if len(values) >= 3:
            break
        if element <= candidate <= maximum:
            values.add(candidate)
    return tuple(sorted(values))


def _resources(flows, topology):
    resources = set()
    for flow in flows:
        if flow.kind in {'send', 'rrc'}:
            edge = topology.links[LinkKey(flow.src_rank, flow.dst_rank)]
            resources.add('link:%d:%d' % (flow.src_rank, flow.dst_rank))
            resources.update(edge.resource_ids)
    return tuple(sorted(resources))


def _mode(context, category, flows, size):
    inputs, topology = context.inputs, context.topology
    flows = tuple(sorted(flows))
    ranks = tuple(sorted({rank for flow in flows for rank in (flow.src_rank, flow.dst_rank)}))
    placements = {item['rank']: item for item in inputs.request_config['runtime']['rank_map']}
    if not flows or len({flow.flow_id for flow in flows}) != len(flows):
        raise SemanticError('probe mode needs distinct physical flow identities')
    for flow in flows:
        if flow.src_rank not in placements or flow.dst_rank not in placements:
            raise SemanticError('probe flow is outside the explicit rank map')
        if flow.kind in {'send', 'rrc'}:
            edge = topology.links.get(LinkKey(flow.src_rank, flow.dst_rank))
            if edge is None or flow.channel >= min(inputs.solver.max_channels, edge.max_channels,
                    *(topology.shared_resources[key].max_channels for key in edge.resource_ids)):
                raise SemanticError('probe mode exceeds the declared directed topology')
    mapping = tuple({'rank': local, 'original_rank': rank, 'host': placements[rank]['host'],
                     'device': placements[rank]['device']} for local, rank in enumerate(ranks))
    scope = 'local' if len(ranks) == 1 else 'intra' if len({item['host'] for item in mapping}) == 1 else 'inter'
    # send/copy execute no reduction. Keep their kernel selection and cache
    # identity independent of an unsupported application reduction operation.
    reduction = ((inputs.collective.reduction_op or 'sum')
                 if any(flow.kind in {'rrc', 'reduce'} for flow in flows) else 'sum')
    identity = {'flows': [flow.flow_id for flow in flows], 'payload_bytes': size, 'datatype': inputs.collective.datatype,
                'reduction_op': reduction, 'rank_map': mapping, 'resources': _resources(flows, topology),
                'flow_resources': tuple((flow.flow_id, _resources((flow,), topology)) for flow in flows),
                'protocol': inputs.request_config['runtime']['protocol']}
    facts = {}
    original_links = {(item['src'], item['dst']): item for item in inputs.request_config['topology']['directed_links']}
    original_resources = {item['id']: item for item in inputs.request_config['topology']['shared_resources']}
    for flow in flows:
        if flow.kind not in {'send', 'rrc'}:
            continue
        edge = topology.links[LinkKey(flow.src_rank, flow.dst_rank)]
        declared = original_links[(flow.src_rank, flow.dst_rank)]
        facts[flow.flow_id] = {'physical_path': edge.physical_path,
            'initial_alpha_us': declared['alpha_us'], 'initial_beta_us_per_byte': declared['beta_us_per_byte'],
            'shared_resources': {key: {'capacity_bytes_per_us': original_resources[key]['capacity_bytes_per_us'],
                'members': [(link.src_rank, link.dst_rank) for link in topology.shared_resources[key].member_links]}
                for key in edge.resource_ids}}
    identity['topology_facts'] = facts
    policy = inputs.request_config['sketch']['calibration']
    if 'job_timeout_s' not in policy:
        raise SemanticError('probe execution needs an explicit calibration job timeout')
    return ProbeJob('probe-' + digest(identity)[:24], category, scope, flows, size, inputs.collective.datatype,
        reduction, ranks, mapping, identity['resources'], policy['launches'], policy['warmup_iterations'],
        policy['timed_iterations'], policy['max_rounds'], policy['max_cv'], policy['job_timeout_s'],
        flow_resources=identity['flow_resources'], topology_facts=facts)


def plan_probes(context):
    sizes, seen = payload_sizes(context), set()
    topology, inputs = context.topology, context.inputs
    kinds = ('send', 'rrc') if inputs.collective.reduction_op is not None else ('send',)
    def emit(category, flows):
        for size in sizes:
            context.lease.check()
            job = _mode(context, category, flows, size)
            if job.job_id not in seen:
                seen.add(job.job_id)
                yield job
    # First cover independent directions. A later shared/inter job failure cannot
    # erase a completed link component. Scheduling may interleave disjoint GPUs.
    for count in range(1, inputs.solver.max_channels + 1):
        for key, edge in sorted(topology.links.items()):
            maximum = min(edge.max_channels, *(topology.shared_resources[r].max_channels for r in edge.resource_ids), inputs.solver.max_channels)
            if count <= maximum:
                for kind in kinds:
                    yield from emit('link', tuple(ProbeFlow(kind, key.src_rank, key.dst_rank, channel) for channel in range(count)))
    for rank in range(inputs.rank_count):
        yield from emit('copy', (ProbeFlow('copy', rank, rank),))
        if 'rrc' in kinds:
            yield from emit('reduce', (ProbeFlow('reduce', rank, rank),))
    pairs = set()
    for resource in topology.shared_resources.values():
        pairs.update(tuple(sorted(pair)) for pair in combinations(resource.member_links, 2))
    for first, second in sorted(pairs):
        maximum = min(inputs.solver.max_channels, topology.links[first].max_channels, topology.links[second].max_channels,
            *(topology.shared_resources[r].max_channels for edge in (first, second) for r in topology.links[edge].resource_ids))
        for count in range(1, maximum + 1):
            for kind in kinds:
                yield from emit('shared', tuple(ProbeFlow(kind, edge.src_rank, edge.dst_rank, channel)
                                               for edge in (first, second) for channel in range(count)))
    for mode in context.observed_modes:
        if any(flow.kind not in {'send', 'rrc'} for flow in mode):
            raise SemanticError('independent mixed local-memory/communication modes are not probed')
        yield from emit('shared', mode)


def execute_probe(job, context, driver=None):
    job = replace(job, execution_id=job.execution_id or uuid.uuid4().hex)
    if driver is None:
        from .remote import NativeProbeDriver
        if context.native_driver is None:
            context.native_driver = NativeProbeDriver()
        driver = context.native_driver
    context.lease.check()
    if driver.driver_kind not in {'native_msccl', 'test_simulation'}:
        raise SemanticError('probe executor must declare native or explicit test evidence')
    if context.run_context is not None:
        context.run_context.write_json('models/probes/' + probe_artifact_key(job) + '/job.json', asdict(job))
    try:
        result = driver.execute(job, context)
        if not isinstance(result, ProbeResult) or result.job_id != job.job_id or result.driver_kind != driver.driver_kind:
            raise SemanticError('probe executor returned another job or evidence source')
    except (BudgetExhausted, StorageFailure):
        raise
    except Exception as error:
        result = ProbeResult(job.job_id, 'failed', driver.driver_kind, message=str(error))
    if context.run_context is not None:
        result = replace(result, provenance=result.provenance | {'execution_id': job.execution_id})
        context.run_context.write_json('models/probes/' + probe_artifact_key(job) + '/result.json', asdict(result))
    context.records.append({'job_id': job.job_id, 'category': job.category, 'host_scope': job.host_scope,
                            'status': result.status, 'driver_kind': result.driver_kind})
    return result


def execute_campaign(jobs, context, driver=None):
    for job in jobs:
        context.lease.check()
        yield execute_probe(job, context, driver)
