"""Map actual JUnit results to requirements; passing mapped tests is not a complete proof."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def read_junit(path):
    tests = []
    for case in ET.parse(path).iter('testcase'):
        status = next((kind for tag, kind in (('failure', 'failed'), ('error', 'error'), ('skipped', 'skipped'))
                       if case.find(tag) is not None), 'passed')
        detail = next((case.find(tag).get('message', '') for tag in ('failure', 'error', 'skipped')
                       if case.find(tag) is not None), '')
        tests.append({'id': case.get('classname', '') + '::' + case.get('name', ''),
                      'file': case.get('classname', '').replace('.', '/') + '.py',
                      'status': status, 'detail': detail, 'time_s': float(case.get('time', 0))})
    return tests


def summarize_tests(tests, paths):
    selected = [case for case in tests if any(case['file'] == path or
                (path.endswith('/') and case['file'].startswith(path)) for path in paths)]
    counts = {key: sum(case['status'] == key for case in selected) for key in ('passed', 'failed', 'error', 'skipped')}
    status = ('not_executed' if not selected else 'failed' if counts['failed'] or counts['error']
              else 'passed_with_skips' if counts['skipped'] and counts['passed'] else 'skipped' if counts['skipped'] else 'passed')
    return {'status': status, 'counts': counts, 'test_ids': [case['id'] for case in selected]}


def build_report(mapping, junit, *, implementation=None):
    tests = read_junit(junit)
    tasks = {name: {key: item for key, item in value.items() if key != 'development_evidence_status'} |
             {'actual_tests': summarize_tests(tests, value['test_paths'])}
             for name, value in mapping['task_evidence'].items()}
    requirements = []
    for value in mapping['requirements']:
        names = sorted(set(value['implementation_tasks'] + value['verification_tasks']))
        paths = sorted({path for name in names for path in tasks.get(name, {}).get('test_paths', [])})
        evidence = {name: (implementation or {}).get(name, {}) for name in value['implementation_tasks']}
        requirements.append({key: item for key, item in value.items() if key not in
                             {'implementation_status', 'development_evidence_status', 'actual_test_status'}} |
            {'actual_tests': summarize_tests(tests, paths), 'implementation_evidence': evidence,
            'implementation_status': 'reported' if all(evidence.values()) else 'missing_evidence',
            'phase3_status': 'not_executed',
            'interpretation': 'Mapped test coverage; review the implementation report for specific features, hardware scope, and unexecuted items'})
    return {'schema_version': 1, 'scope': 'Development test mapping report; phase-three four-node acceptance has not been run',
        'junit': str(junit), 'junit_sha256': hashlib.sha256(Path(junit).read_bytes()).hexdigest(),
        'counts': dict(Counter(case['status'] for case in tests)), 'tasks': tasks,
        'requirements': requirements, 'tests': tests}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mapping', type=Path, required=True)
    parser.add_argument('--junit', type=Path, required=True)
    parser.add_argument('--implementation', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    implementation = json.loads(args.implementation.read_text()) if args.implementation else None
    report = build_report(json.loads(args.mapping.read_text()), args.junit, implementation=implementation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report['counts']))


if __name__ == '__main__':
    main()
