"""Size bands: validation, per-band overrides and bundle publication."""
import json
from types import SimpleNamespace

import pytest

from vericcl.bands import (SizeBand, band_overrides, merge_bands, program_identity, publish_bundle,
                           validate_bands, with_window)
from vericcl.errors import InputValidationError
from vericcl.execution.context import RunContext

XML = '<algo name="vericcl" nchannels="4" minBytes="{low}" maxBytes="{high}" ngpus="2"><gpu id="0"/></algo>'
OTHER = '<algo name="vericcl" nchannels="2" minBytes="{low}" maxBytes="{high}" ngpus="2"><gpu id="0"/></algo>'


def band(index, low, high, representative=None):
    return SizeBand(index, low, high, representative if representative is not None else low)


def test_bands_must_be_sorted_disjoint_and_contain_their_representative():
    validate_bands((band(0, 0, 64), band(1, 64, 512, 128)))
    with pytest.raises(InputValidationError, match='disjoint'):
        validate_bands((band(0, 0, 64), band(1, 32, 512, 128)))
    with pytest.raises(InputValidationError, match='representative'):
        validate_bands((band(0, 0, 64, 64),))
    with pytest.raises(InputValidationError, match='at most 4'):
        validate_bands(tuple(band(i, 10 * i, 10 * i + 5) for i in range(5)))


def test_band_overrides_bind_size_window_and_disable_nested_bands():
    from vericcl.semantics.collective import CollectiveKind
    allgather = SimpleNamespace(collective=SimpleNamespace(kind=CollectiveKind.ALL_GATHER), rank_count=16)
    allreduce = SimpleNamespace(collective=SimpleNamespace(kind=CollectiveKind.ALL_REDUCE), rank_count=16)
    # Windows use MSCCL's message size: the whole AllGather output.
    assert band_overrides(SizeBand(1, 64, 512, 256, 8), allgather)['sketch.hyperparameters.total_size_bytes'] == 16
    with pytest.raises(InputValidationError, match='divisible'):
        band_overrides(SizeBand(1, 64, 512, 100), allgather)
    overrides = band_overrides(SizeBand(1, 64, 512, 128, 8), allreduce)
    assert overrides['sketch.hyperparameters.total_size_bytes'] == 128
    assert overrides['sketch.hyperparameters.slice_size_bytes'] == 8
    assert overrides['runtime.xml_size_range'] == {'min_bytes': 64, 'max_bytes': 512}
    assert overrides['sketch.size_bands'] == []


def test_window_rewrite_keeps_the_program():
    text = XML.format(low=0, high=64)
    rewritten = with_window(text, 64, 4096)
    assert 'minBytes="64"' in rewritten and 'maxBytes="4096"' in rewritten
    assert program_identity(text) == program_identity(rewritten)


def test_adjacent_identical_programs_merge_into_one_window():
    results = [(band(0, 0, 64), XML.format(low=0, high=64)),
               (band(1, 64, 512, 128), XML.format(low=64, high=512)),
               (band(2, 512, 4096), OTHER.format(low=512, high=4096))]
    merged = merge_bands(results)
    assert [(low, high, members) for low, high, _, members in merged] == [(0, 512, [0, 1]), (512, 4096, [2])]
    assert 'maxBytes="512"' in merged[0][2]


def test_publish_bundle_lists_xml_files_in_band_order(tmp_path):
    parent = RunContext.create(tmp_path, 'bands', 'parent')
    results = []
    for index, (low, high, template) in enumerate(((0, 64, XML), (64, 4096, OTHER))):
        run = RunContext.create(tmp_path, 'bands', 'band%d' % index)
        xml = template.format(low=low, high=high)
        run.write_bytes('candidates/c/schedule.xml', xml.encode())
        run.write_json('best/accepted.json', {'candidate': {'xml': 'candidates/c/schedule.xml'}, 't_pred_us': 1.0})
        results.append((band(index, low, high), run.root, 0))
    manifest = publish_bundle(parent, results)
    paths = manifest['msccl_xml_files'].split(':')
    assert len(paths) == 2 and all(path.endswith('.xml') for path in paths)
    assert manifest['xml_files'][0]['max_bytes'] == 64 and manifest['xml_files'][1]['min_bytes'] == 64
    assert json.loads((parent.root / 'bands/manifest.json').read_text())['missing'] == []


def test_size_bands_field_is_validated_by_the_input_schema(tmp_path):
    from tests.unit.input.v5_support import request_documents, write_request
    from vericcl.input.loader import load_request
    from vericcl.bands import size_bands
    documents = request_documents('broadcast', 2)
    documents['sketch']['size_bands'] = [
        {'min_bytes': 0, 'max_bytes': 2048, 'representative_bytes': 1024},
        {'min_bytes': 2048, 'max_bytes': 1 << 20, 'representative_bytes': 4096}]
    inputs = load_request(write_request(tmp_path, documents))
    assert [b.representative_bytes for b in size_bands(inputs)] == [1024, 4096]
