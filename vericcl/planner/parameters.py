"""Necessary parameter bounds before synthesis; concrete programs are checked later."""
from dataclasses import dataclass, replace
from fractions import Fraction
from collections.abc import Mapping

from vericcl.errors import SemanticError
from vericcl.input.schema import DATATYPE_BYTES, _freeze
from vericcl.xml.capacity import RuntimeLimits, XML_MAX_RETAINED_NODES, simple_payload_capacity


@dataclass(frozen=True)
class ParameterCandidate:
    channels: int
    slice_bytes: int
    slice_count: int
    lower_bound_bytes: int
    upper_bound_bytes: int
    bounds: tuple[tuple[str, int], ...]
    adjustment_reason: str | None = None
    requires_structure_check: bool = True


@dataclass(frozen=True)
class ParameterExclusion:
    channels: int
    code: str
    detail: str
    proven: bool = True


@dataclass(frozen=True)
class ParameterDomain:
    candidates: tuple[ParameterCandidate, ...]
    exclusions: tuple[ParameterExclusion, ...]
    complete_runtime_validation: bool = False


def _ceil_div(numerator, denominator):
    return (numerator + denominator - 1) // denominator


def _minimum_rank_work(operator, rank, ranks, root):
    if ranks == 1:
        return Fraction(0), Fraction(0)
    if operator == 'broadcast':
        return (Fraction(1), Fraction(0)) if rank == root else (Fraction(0), Fraction(1))
    if operator == 'reduce':
        return (Fraction(0), Fraction(1)) if rank == root else (Fraction(1), Fraction(0))
    if operator == 'allgather':
        return Fraction(1), Fraction(ranks - 1)
    if operator == 'allreduce':
        return Fraction(1), Fraction(1)
    if operator == 'alltoall':
        return Fraction(ranks - 1, ranks), Fraction(ranks - 1, ranks)
    if operator == 'reduce_scatter':
        return Fraction(ranks - 1, ranks), Fraction(1, ranks)
    raise SemanticError('unsupported external collective for parameter selection')


def initial_parameter_domain(inputs, runtime=None, *, fixed_slice_bytes=None, lease=None):
    config = inputs.request_config
    if not config:
        raise SemanticError('v5 parameter selection requires normalized explicit inputs')
    runtime = config['runtime'] if runtime is None else runtime
    limits = RuntimeLimits.from_runtime(runtime)
    size = inputs.hyperparameters.total_size_bytes
    requested = inputs.hyperparameters.slice_size_bytes
    element = DATATYPE_BYTES[inputs.collective.datatype]
    ranks, operator = inputs.rank_count, inputs.collective.kind.value
    root = inputs.collective.root
    available = min(config['sketch']['solver']['max_channels'], limits.max_channels)
    if fixed_slice_bytes is not None:
        requested = fixed_slice_bytes
    if requested is not None and (isinstance(requested, bool) or not isinstance(requested, int) or requested < 1
                                  or size % requested or requested % element
                                  or (operator in {'alltoall', 'reduce_scatter'} and (size // requested) % ranks)):
        raise SemanticError('requested slice partition violates element or collective divisibility')
    bound_size = simple_payload_capacity(int(runtime['environment']['NCCL_BUFFSIZE']), element)
    links = config['topology']['directed_links']
    candidates, exclusions = [], []
    for channels in range(1, available + 1):
        if lease is not None:
            lease.check()
        if 1 + ranks > XML_MAX_RETAINED_NODES:
            exclusions.append(ParameterExclusion(channels, 'xml_header_capacity',
                'algo and all GPU headers exceed the fixed XML parser capacity'))
            continue
        if ranks == 1 and inputs.collective.inplace:
            # This semantic identity requires no payload operation or wire buffer.
            chosen = requested if requested is not None else size
            if size // chosen > 2**31 - 1:
                if fixed_slice_bytes is not None:
                    exclusions.append(ParameterExclusion(channels, 'chunks_per_loop', 'fixed partition exceeds the integer field'))
                    continue
                chosen = size
            candidates.append(ParameterCandidate(channels, chosen, size // chosen, 1, size,
                                                  (('chunks_per_loop', 2**31 - 1),),
                                                  'initial_slice_capacity' if requested is not None and chosen != requested else None))
            continue
        output_factor = ranks if operator == 'allgather' else 1
        n_bounds = [('buffer_offsets', (limits.buffer_offset + 1) // output_factor),
                    ('whole_elements', size // element), ('chunks_per_loop', (2**31 - 1) // output_factor)]
        impossible = None
        for rank in range(ranks):
            sending, receiving = _minimum_rank_work(operator, rank, ranks, root)
            required_directions = int(sending > 0) + int(receiving > 0)
            minimum_blocks = required_directions if runtime['tb_layout'] == 'separate' else min(1, required_directions)
            if minimum_blocks > limits.tbs_per_rank:
                impossible = ('structural_tb_minimum', 'required send/receive owners exceed the rank TB limit')
                break
            capacities = {}
            for direction, coefficient in (('send', sending), ('recv', receiving)):
                key = 'src' if direction == 'send' else 'dst'
                relevant = [edge for edge in links if edge[key] == rank]
                # Each peer/channel/direction can own at most one TB. Count
                # only declared channels, then apply the direction-wide cap.
                owners = sum(min(channels, edge['max_channels']) for edge in relevant)
                owners = min(owners, channels * getattr(limits, direction + '_tbs_per_channel'), limits.tbs_per_rank)
                capacities[direction] = owners
                if coefficient:
                    if not owners:
                        impossible = ('unreachable_required_rank', direction + ' has no declared link at rank ' + str(rank))
                        break
                    n_bounds.append((direction + '_steps_rank_' + str(rank), int(Fraction(owners * limits.steps_per_tb) / coefficient)))
            if impossible:
                break
            coefficient = sending + receiving if runtime['tb_layout'] == 'separate' else max(sending, receiving)
            if coefficient:
                owners = min(sum(capacities.values()), limits.tbs_per_rank)
                n_bounds.append(('total_steps_rank_' + str(rank), int(Fraction(owners * limits.steps_per_tb) / coefficient)))
            # At least these payload steps and owner headers must be retained
            # by the per-rank parser. Routing, local copies and dependency NOPs
            # can add more; the concrete program is checked after lowering.
            parser_coefficient = coefficient if ranks > 1 else Fraction(1)
            parser_blocks = minimum_blocks if ranks > 1 else 1
            if parser_coefficient:
                available_nodes = XML_MAX_RETAINED_NODES - 1 - ranks - parser_blocks
                n_bounds.append(('xml_nodes_rank_' + str(rank),
                                 int(Fraction(available_nodes) / parser_coefficient)))
        if impossible:
            exclusions.append(ParameterExclusion(channels, *impossible))
            continue
        maximum_n = min(bound for _, bound in n_bounds)
        if maximum_n < 1 or bound_size < element:
            exclusions.append(ParameterExclusion(channels, 'no_integer_capacity', 'no complete element fits the necessary limits'))
            continue
        lower = max(element, _ceil_div(size, maximum_n))
        upper = min(size, bound_size)
        def allowed(partition):
            n = size // partition
            return (size % partition == 0 and partition % element == 0 and n <= maximum_n and partition <= upper
                    and (operator not in {'alltoall', 'reduce_scatter'} or n % ranks == 0))
        if requested is not None and allowed(requested):
            chosen = requested
        elif fixed_slice_bytes is not None:
            exclusions.append(ParameterExclusion(channels, 'fixed_slice_capacity', 'fixed S violates a necessary typed capacity bound'))
            continue
        else:
            # Buffer index bounds limit this scan to at most 32768 integers,
            # independent of message bytes. No power-of-two restriction.
            chosen = next((size // n for n in range(maximum_n, 0, -1)
                           if size % n == 0 and allowed(size // n)), None)
        if chosen is None:
            exclusions.append(ParameterExclusion(channels, 'no_divisible_slice', 'no S satisfies the necessary bounds and exact partition arithmetic'))
            continue
        candidates.append(ParameterCandidate(channels, chosen, size // chosen, lower, upper, tuple(n_bounds),
                                              'initial_slice_capacity' if requested is not None and requested != chosen else None))
    return ParameterDomain(tuple(candidates), tuple(exclusions))


def resolve_parameters(inputs, candidate):
    """Bind one synthesis attempt without changing original identity references."""
    if not isinstance(candidate, ParameterCandidate):
        raise SemanticError('parameter choice must be a ParameterCandidate')
    if inputs.hyperparameters.total_size_bytes != candidate.slice_bytes * candidate.slice_count:
        raise SemanticError('parameter choice does not partition this request')
    def plain(value):
        if isinstance(value, Mapping):
            return {key: plain(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [plain(item) for item in value]
        return value
    config, sources, accounting = (plain(value) for value in (inputs.request_config, inputs.value_sources, inputs.size_accounting))
    if config:
        config['sketch']['hyperparameters']['slice_size_bytes'] = candidate.slice_bytes
        config['sketch']['solver']['max_channels'] = candidate.channels
        accounting.update(slice_size_bytes=candidate.slice_bytes, slices_per_rank=candidate.slice_count)
        for path, value in (('sketch.hyperparameters.slice_size_bytes', candidate.slice_bytes),
                            ('sketch.solver.max_channels', candidate.channels)):
            sources[path] = {'kind': 'parameter_selection', 'value': value, 'prior': sources.get(path),
                             'reason': candidate.adjustment_reason or 'initial_parameter_domain'}
        sources['derived.slices_per_rank'] = {'kind': 'derived', 'value': candidate.slice_count,
                                             'expression': 'total_size_bytes / slice_size_bytes'}
    return replace(inputs, hyperparameters=replace(inputs.hyperparameters, slice_size_bytes=candidate.slice_bytes),
                   solver=replace(inputs.solver, max_channels=candidate.channels), request_config=_freeze(config),
                   value_sources=_freeze(sources), size_accounting=_freeze(accounting),
                   resolved_sketch=_freeze(config['sketch']) if config else inputs.resolved_sketch)


def bind_artifact_partition(inputs, sidecar):
    """Recover an artifact's S while retaining the original request and limits."""
    if not isinstance(sidecar, Mapping) or sidecar.get('input_sha256') != inputs.input_sha256:
        raise SemanticError('sidecar input identity differs from this request')
    contract = sidecar.get('request_contract')
    size = inputs.hyperparameters.total_size_bytes
    slice_bytes = contract.get('slice_bytes') if isinstance(contract, Mapping) else None
    element = DATATYPE_BYTES[inputs.collective.datatype]
    if (isinstance(slice_bytes, bool) or not isinstance(slice_bytes, int) or slice_bytes < 1
            or size % slice_bytes or slice_bytes % element
            or (inputs.collective.kind.value in {'alltoall', 'reduce_scatter'}
                and (size // slice_bytes) % inputs.rank_count)):
        raise SemanticError('artifact slice partition violates element or collective divisibility')
    if slice_bytes == inputs.hyperparameters.slice_size_bytes:
        return inputs
    # Verify must still inspect capacity-invalid XML so --tune can recover it.
    # Concrete capacity checks follow binding; this is not synthesis admission.
    choice = ParameterCandidate(inputs.solver.max_channels, slice_bytes, size // slice_bytes,
                                1, size, (), 'artifact_slice_binding')
    return resolve_parameters(inputs, choice)
