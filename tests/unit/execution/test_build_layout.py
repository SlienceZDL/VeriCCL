import json


def test_runtime_prepare_uses_process_tree_and_consolidated_build_record(tmp_path):
    from runtime.build_runtime import main
    output = tmp_path / 'prepared'
    assert main(['--output-dir', str(output), '--prepare-only']) == 0
    assert {p.name for p in output.iterdir() if p.is_dir()} == {'process'}
    result = json.loads((output / 'process/result.json').read_text())
    assert result['status'] == 'prepared'
    assert result['source_dir'] == str(output / 'process/rebuildable-cache/source')
    assert (output / 'process/rebuildable-cache/source/src/init.cc').is_file()
    metadata = json.loads((output / 'run.json').read_text())
    assert metadata['invocation']['command'] == 'build_runtime'
    assert metadata['result']['contract_met']
    assert (output / 'report.md').is_file()
    assert (output / 'details.zip').is_file()
