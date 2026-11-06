"""Stable logical artifact names across legacy and compact run layouts."""
import json
from pathlib import Path


PROCESS_DIRECTORIES = ('input_snapshot', 'models', 'candidates', 'jobs', 'live',
                       'statistics', 'checkpoints', 'reports', 'best', 'tmp', 'rebuildable-cache')
METADATA_SECTIONS = {
    'invocation.json': 'invocation', 'input-snapshot.json': 'input_snapshot',
    'effective-request.json': 'effective_request', 'value-sources.json': 'value_sources',
    'size-accounting.json': 'size_accounting',
}


def layout_version(root):
    metadata = Path(root) / 'run.json'
    return json.loads(metadata.read_text()).get('schema_version', 1) if metadata.is_file() else 1


def artifact_path(root, relative, *, version=None):
    root, path = Path(root).resolve(), Path(relative)
    if path.is_absolute() or not path.parts or '..' in path.parts or '\\' in str(relative):
        raise ValueError('artifact path violates run boundary')
    version = layout_version(root) if version is None else version
    if version >= 2 and (path.parts[0] in PROCESS_DIRECTORIES
            or path.name == 'run-summary.json'
            or path.name.endswith('events.jsonl')
            or (path.name.startswith('rank-') and path.name.endswith('.run.log'))):
        if path.parts[0] != 'process':
            path = Path('process') / path
    target = root / path
    if not target.resolve().is_relative_to(root):
        raise ValueError('artifact symlink violates run boundary')
    return target


def read_run_section(root, filename):
    """Read a formerly separate metadata document from either layout."""
    root = Path(root)
    metadata = json.loads(artifact_path(root, 'run.json').read_text())
    if metadata.get('schema_version', 1) >= 2 and filename in METADATA_SECTIONS:
        return metadata[METADATA_SECTIONS[filename]]
    return json.loads(artifact_path(root, filename).read_text())
