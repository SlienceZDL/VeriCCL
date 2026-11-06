"""Generate small v5 cases with initial cost estimates and hardware measurement disabled."""
import argparse
import json
from pathlib import Path

OPERATORS = ('broadcast', 'reduce', 'allgather', 'allreduce', 'alltoall', 'reduce_scatter')
METHODS = ('hierarchy', 'constructive_trees')


def example_documents(operator='allreduce', method='hierarchy', *, inplace=False, ranks=2):
    if operator not in OPERATORS or method not in METHODS:
        raise ValueError('unsupported example operator or synthesis method')
    return {
        'topology': {'name': 'Tutorial example: initial cost estimates', 'ranks': ranks,
            'nodes': [{'id': 0, 'ranks': list(range(ranks))}],
            'directed_links': [{'src': a, 'dst': b, 'alpha_us': 1, 'beta_us_per_byte': .01}
                              for a in range(ranks) for b in range(ranks) if a != b],
            'local_costs': [{'rank': rank, 'copy': {'alpha_us': .1, 'beta_us_per_byte': .001},
                            'reduce': {'alpha_us': .2, 'beta_us_per_byte': .002}} for rank in range(ranks)]},
        'sketch': {
            'collective': {'operator': operator, 'inplace': inplace},
            'hyperparameters': {'total_size_bytes': ranks * 1024, 'slice_size_bytes': 1024},
            'solver': {'per_model_timeout_s': 5, 'cpu_budget': 2, 'max_channels': 1,
                       'max_threads_per_model': 1, 'max_parallel_models': 1, 'solver_seed': 0},
            'budgets': {'max_initial_synthesis_s': 20, 'max_offline_validation_per_round_s': 10,
                'max_offline_validation_total_s': 30, 'max_repair_per_round_s': 10,
                'max_repair_total_s': 30, 'max_tuning_rounds': 0, 'cleanup_reserve_s': .5},
            'tolerances': {'absolute': 1e-5, 'relative': 1e-5},
            'calibration': {'enabled': False},
        },
        'atom': {'synthesis_method': method},
        'runtime': {'rank_map': [{'rank': rank, 'host': 'localhost', 'device': rank} for rank in range(ranks)],
                    'environment': {'NCCL_BUFFSIZE': '16777216'}, 'scratch_bytes_per_rank': 16777216},
    }


def write_documents(directory, documents):
    directory = Path(directory)
    paths = {name: directory / (name + '.json') for name in ('topology', 'sketch', 'atom', 'runtime')}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError('example input already exists: ' + str(directory))
    directory.mkdir(parents=True, exist_ok=True)
    for name, path in paths.items():
        with path.open('x') as stream:
            stream.write(json.dumps(documents[name], ensure_ascii=False, indent=2) + '\n')
    return paths


def example_cases():
    for operator in OPERATORS:
        for method in METHODS:
            for inplace in (False, True):
                name = '%s-%s-%s' % (operator, method, 'inplace' if inplace else 'outofplace')
                yield name, example_documents(operator, method, inplace=inplace)
    for option in ('symmetry', 'shortest_paths', 'batching'):
        documents = example_documents('broadcast')
        documents['atom']['strategies'] = {option: True}
        yield 'explicit-' + option, documents
    documents = example_documents('allgather')
    documents['runtime']['tb_layout'] = 'fused'
    yield 'explicit-fused-tb', documents
    documents = example_documents('broadcast')
    documents['sketch']['collective']['root'] = 1
    yield 'explicit-root-1', documents
    documents = example_documents('broadcast')
    documents['sketch']['hyperparameters']['total_size_bytes'] = 1024
    documents['atom']['manual'] = {'enabled': True, 'nodes': [{'id': 'broadcast', 'stage_id': 0,
        'operator': 'broadcast', 'group': [0, 1], 'root': 0,
        'input_states': [{'rank': 0, 'offset': 0, 'contributors': [0]}],
        'output_states': [{'rank': rank, 'offset': 0, 'contributors': [0]} for rank in range(2)]}]}
    yield 'explicit-manual', documents


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--case', default='v5-smoke')
    args = parser.parse_args(argv)
    if args.case in {'', '.', '..'} or '/' in args.case or '\\' in args.case:
        parser.error('--case must be one directory name')
    root = args.project_root.resolve() / 'examples' / args.case / 'input'
    manifest = []
    for name, documents in example_cases():
        paths = write_documents(root / name, documents)
        manifest.append({'name': name, 'files': {key: str(path.relative_to(root)) for key, path in paths.items()},
                         'scope': 'Initial estimated model, CPU synthesis and verification; no GPU measurements performed'})
    with (root / 'cases.json').open('x') as stream:
        stream.write(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(str(root / 'cases.json'))


if __name__ == '__main__':
    main()
