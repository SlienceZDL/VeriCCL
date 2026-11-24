"""Size bands: one verified schedule per message-size window (plan.md 2.1, 3.6).

MSCCL loads at most four XMLs from ``MSCCL_XML_FILES`` and, for each call,
selects the first whose ``[minBytes, maxBytes)`` contains the message size.
The best schedule depends on the size (latency-bound small messages favour
few hops, bandwidth-bound large ones favour many concurrent threadblocks), so
a request may declare up to four ``sketch.size_bands``. Each band is a
complete standard run - synthesis, calibration and measurement, verification
and repair - at the band's representative size, with the band as its XML
window; the calibration cache shares measured components between bands.
Adjacent bands whose verified programs are identical are published as one
XML that covers both windows (each window was verified with that program).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re

from vericcl.errors import InputValidationError

MSCCL_MAX_XML_FILES = 4
_WINDOW = re.compile(r'\s(minBytes|maxBytes)="[0-9]+"')


@dataclass(frozen=True)
class SizeBand:
    index: int
    min_bytes: int
    max_bytes: int
    representative_bytes: int
    slice_size_bytes: int | None = None


def validate_bands(bands):
    if len(bands) > MSCCL_MAX_XML_FILES:
        raise InputValidationError('sketch.size_bands allows at most %d bands (MSCCL_XML_FILES limit)'
                                   % MSCCL_MAX_XML_FILES)
    previous = None
    for band in bands:
        if not 0 <= band.min_bytes < band.max_bytes:
            raise InputValidationError('size band %d needs 0 <= min_bytes < max_bytes' % band.index)
        if not band.min_bytes <= band.representative_bytes < band.max_bytes:
            raise InputValidationError('size band %d representative_bytes must lie in [min_bytes, max_bytes)'
                                       % band.index)
        if previous is not None and band.min_bytes < previous.max_bytes:
            raise InputValidationError('sketch.size_bands must be sorted and disjoint')
        previous = band
    return tuple(bands)


def size_bands(inputs):
    raw = inputs.request_config['sketch'].get('size_bands') or ()
    return validate_bands(tuple(
        SizeBand(index, int(band['min_bytes']), int(band['max_bytes']), int(band['representative_bytes']),
                 band.get('slice_size_bytes'))
        for index, band in enumerate(raw)))


def per_rank_bytes(inputs, message_bytes):
    """Per-rank input M (``total_size_bytes``) for an MSCCL message size.

    Band windows use the size MSCCL selects on (``info->nBytes``, the
    nccl-tests size column): for AllGather that is the whole output, per-rank
    input times the rank count; for the other collectives it equals M.
    """
    if inputs.collective.kind.value == 'allgather':
        if message_bytes % inputs.rank_count:
            raise InputValidationError('an AllGather band size must be divisible by the rank count')
        return message_bytes // inputs.rank_count
    return message_bytes


def band_overrides(band, inputs):
    """Field overrides that turn the request into one band's standard run."""
    return {'sketch.hyperparameters.total_size_bytes': per_rank_bytes(inputs, band.representative_bytes),
            'sketch.hyperparameters.slice_size_bytes': band.slice_size_bytes,
            'runtime.xml_size_range': {'min_bytes': band.min_bytes, 'max_bytes': band.max_bytes},
            'sketch.size_bands': []}


def program_identity(xml_text):
    """Digest of an XML program without its size window."""
    return hashlib.sha256(_WINDOW.sub('', xml_text).encode()).hexdigest()


def with_window(xml_text, low, high):
    """The same program with the root window set to [low, high)."""
    replaced = {'minBytes': False, 'maxBytes': False}

    def rewrite(match):
        name = match.group(1)
        if replaced[name]:
            return match.group(0)
        replaced[name] = True
        return ' %s="%d"' % (name, low if name == 'minBytes' else high)
    text = _WINDOW.sub(rewrite, xml_text)
    if not all(replaced.values()):
        raise ValueError('XML has no minBytes/maxBytes window to rewrite')
    return text


def merge_bands(results):
    """[(band, xml_text)] -> [(low, high, xml_text, [band indices])], adjacent equal programs merged."""
    merged = []
    for band, xml_text in results:
        if (merged and merged[-1][1] == band.min_bytes
                and program_identity(merged[-1][2]) == program_identity(xml_text)):
            low, _, text, members = merged[-1]
            merged[-1] = (low, band.max_bytes, text, members + [band.index])
        else:
            merged.append((band.min_bytes, band.max_bytes, xml_text, [band.index]))
    return [(low, high, with_window(text, low, high), members) for low, high, text, members in merged]


def accepted_xml(run_root):
    """The verified accepted XML of a finished band run, or None."""
    accepted = run_root / 'best' / 'accepted.json'
    if not accepted.is_file():
        return None
    record = json.loads(accepted.read_text())
    candidate = record.get('candidate') or {}
    relative = candidate.get('xml')
    if not relative:
        return None
    return (run_root / relative).read_text(), record


def final_model(run_root):
    """The cost model a finished band run ended with (latest checkpoint), or None."""
    from vericcl.performance.serialization import load_model
    latest = run_root / 'checkpoints' / 'latest.json'
    if not latest.is_file():
        return None
    try:
        state = json.loads((run_root / json.loads(latest.read_text())['state']).read_text())
        return load_model(run_root / state['model'])
    except (OSError, ValueError, KeyError):
        return None


def publish_bundle(parent, results):
    """Write band XMLs, a manifest and the MSCCL_XML_FILES value into ``parent``."""
    available = []
    missing = []
    for band, run_root, exit_code in results:
        found = accepted_xml(run_root)
        if found is None:
            missing.append({'band': asdict(band), 'run': str(run_root), 'exit_code': exit_code})
            continue
        available.append((band, found[0], found[1], run_root))
    merged = merge_bands([(band, text) for band, text, _, _ in available])
    files = []
    for position, (low, high, text, members) in enumerate(merged):
        relative = 'bands/band-%d.xml' % position
        path = parent.write_bytes(relative, text.encode())
        files.append({'path': str(path), 'min_bytes': low, 'max_bytes': high, 'bands': members,
                      'program_sha256': program_identity(text)})
    manifest = {
        'schema_version': 1,
        'bands': [{'band': asdict(band), 'run': str(root), 'accepted': record}
                  for band, _, record, root in available],
        'missing': missing,
        'xml_files': files,
        'msccl_xml_files': ':'.join(item['path'] for item in files),
        'merged': [item['bands'] for item in files if len(item['bands']) > 1],
    }
    parent.write_json('bands/manifest.json', manifest)
    parent.write_bytes('bands/MSCCL_XML_FILES', (manifest['msccl_xml_files'] + '\n').encode())
    return manifest


def run_bands(args, files, overrides, inputs, *, execute, create_context, load):
    """Run every band as a standard solve and publish the bundle.

    ``execute(request, context) -> exit code``, ``create_context(run_id)``
    and ``load(files, overrides)`` are the CLI's own entry points, so each band
    is exactly the run the CLI would perform for that request.
    """
    from vericcl.execution.events import emit_event
    from vericcl.execution.context import finalize_run
    from vericcl.execution.status import RunResult
    from vericcl.workflow_v5 import PrimaryRequest
    bands = size_bands(inputs)
    parent = create_context(args.run_id_resolved)
    parent.write_json('bands/plan.json', {'bands': [asdict(band) for band in bands]})
    results = []
    # Each band starts from the previous band's calibrated model, so a
    # component class measured once is published to later bands without
    # another launch (calibration equivalence classes, plan.md step 4.1).
    model = getattr(args, 'initial_model', None)
    for band in bands:
        band_inputs = load(files, {**overrides, **band_overrides(band, inputs)})
        context = create_context('%s-band%d' % (args.run_id_resolved, band.index))
        context.write_json('band.json', asdict(band) | {'parent_run': str(parent.root)})
        code = execute(PrimaryRequest(band_inputs, purpose='solve', cost_model=model), context)
        results.append((band, context.root, code))
        model = final_model(context.root) or model
        emit_event(parent, 'band_finished', {'band': asdict(band), 'run': str(context.root), 'exit_code': code,
                                             'model_version': None if model is None else model.version})
    manifest = publish_bundle(parent, results)
    complete = not manifest['missing'] and all(code == 0 for _, _, code in results)
    result = RunResult(complete, 'completed' if complete else 'incomplete',
                       stop_reason='size_bands_finished' if complete else 'size_band_incomplete',
                       failure_category=None if complete else 'online',
                       optional_failures=tuple('band %d exit %d' % (band.index, code)
                                               for band, _, code in results if code))
    finalized = finalize_run(parent, result)
    print('VeriCCL: size bands=%d, XML files=%d, MSCCL_XML_FILES=%s, exit code=%d' % (
        len(bands), len(manifest['xml_files']), parent.root / 'bands/MSCCL_XML_FILES', finalized.exit_code))
    return finalized.exit_code
