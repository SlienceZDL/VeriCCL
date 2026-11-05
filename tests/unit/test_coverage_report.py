"""A skip, an absent test and a failure are different evidence states."""
from pathlib import Path


def test_junit_evidence_keeps_skips_failures_errors_and_missing_separate(tmp_path):
    from tools.coverage_report import read_junit, summarize_tests
    path = tmp_path / 'results.xml'
    path.write_text('''<testsuites><testsuite>
      <testcase classname="tests.unit.example" name="passed" time=".1" />
      <testcase classname="tests.unit.example" name="skip"><skipped message="no device" /></testcase>
      <testcase classname="tests.unit.example" name="failed"><failure message="wrong value" /></testcase>
      <testcase classname="tests.unit.other" name="broken"><error message="fixture failed" /></testcase>
    </testsuite></testsuites>''')
    tests = read_junit(path)
    result = summarize_tests(tests, ['tests/unit/example.py'])
    assert result['counts'] == {'passed': 1, 'failed': 1, 'error': 0, 'skipped': 1}
    assert result['status'] == 'failed'
    assert summarize_tests(tests, ['tests/absent.py'])['status'] == 'not_executed'
    assert summarize_tests(tests, ['tests/unit/other.py'])['counts']['error'] == 1
