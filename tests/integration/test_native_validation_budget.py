"""An interrupted native graph check is incomplete, not a semantic counterexample."""
import pytest

from tests.integration.test_lowering_v5 import candidate_fixture
from vericcl.execution.budget import BudgetExhausted
from vericcl.xml import v5_lower
from vericcl.xml.lower import lower_and_check


@pytest.mark.parametrize('boundary', ['execution_dependencies', '_validate_bindings'])
def test_native_graph_deadline_propagates_before_later_checks(tmp_path, monkeypatch, boundary):
    candidate, context = candidate_fixture(tmp_path, 'allreduce')
    bundle = lower_and_check(candidate, context)
    assert bundle.can_execute

    def expired(*args, **kwargs):
        raise BudgetExhausted('native graph parent deadline')

    def should_not_continue(*args, **kwargs):
        raise AssertionError('capacity check ran after the shared parent deadline')

    monkeypatch.setattr(v5_lower, boundary, expired)
    monkeypatch.setattr(v5_lower, '_capacity', should_not_continue)
    with pytest.raises(BudgetExhausted, match='native graph parent deadline'):
        v5_lower.evaluate_program(candidate, bundle.storage_plan, bundle.program,
                                  bundle.xml_text, context)
