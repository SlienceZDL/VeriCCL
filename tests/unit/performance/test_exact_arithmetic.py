"""GMP rational replay retains decimal input and Python output rounding contracts."""
from fractions import Fraction
import random

import pytest

from vericcl.performance import simulation


@pytest.mark.parametrize('value', [0, 1, -1, .1, 1e-25, 1e25, 1.2345678901234567, 2**53+1])
def test_exact_decimal_input_matches_fraction(value):
    actual = simulation._exact_number(value)
    expected = Fraction(str(value))
    assert (int(actual.numerator), int(actual.denominator)) == expected.as_integer_ratio()


def test_exact_output_rounding_matches_fraction_for_large_values():
    rng = random.Random(765135)
    for _ in range(600):
        numerator = rng.getrandbits(rng.randrange(100, 3000))
        denominator = rng.getrandbits(numerator.bit_length() + rng.randrange(-80, 80)) or 1
        expected = Fraction(numerator, denominator)
        actual = simulation._exact_number(numerator) / simulation._exact_number(denominator)
        assert simulation._exact_float(actual) == float(expected)


def test_rational_prediction_is_independent_of_gmp_float_context():
    import gmpy2
    from vericcl.performance.events import Executable, WorkOperation
    from tests.unit.performance.test_event_progress import model
    schedule = Executable((WorkOperation('a', 'send', 'wire', 10, resources=('wire',), flow_id='a'),
                           WorkOperation('b', 'send', 'wire', 2, resources=('wire',), flow_id='b', not_before_us=.3)))
    baseline = simulation.predict(schedule, model(alpha=.1, beta=.03), exact=True)
    with gmpy2.context(precision=8, round=gmpy2.RoundDown):
        actual = simulation.predict(schedule, model(alpha=.1, beta=.03), exact=True)
    assert actual == baseline
    assert all(type(value) is int for value in actual.exact_completion_ratio)


def test_concurrent_replay_matches_fraction_arithmetic(monkeypatch):
    from vericcl.performance.events import Executable, WorkOperation
    from vericcl.performance.model import CostModel, ServiceCost
    from vericcl.performance.profiles import RateProfile
    original = simulation.mpq
    rng = random.Random(765136)
    for trial in range(100):
        operations = []
        for index in range(24):
            kind = rng.choice(('send', 'rrc', 'copy', 'reduce', 'nop'))
            prior = [operation.operation_id for operation in operations]
            dependencies = tuple(rng.sample(prior, min(len(prior), rng.randrange(3))))
            resources = tuple(rng.sample(('nic', 'pcie'), rng.randrange(1, 3))) if kind in {'send', 'rrc'} else ()
            operations.append(WorkOperation('op-' + str(index), kind, None if kind == 'nop' else kind,
                0 if kind == 'nop' else rng.randrange(1, 150), dependencies=dependencies,
                resources=resources, flow_id='flow-' + str(index), not_before_us=rng.randrange(10) / 10))
        profiles = (RateProfile('profile', ('flow-0', 'flow-1'), {'flow-0': 2, 'flow-1': 3},
            resources=('nic',), startup_us={'flow-0': .1, 'flow-1': .2}, measured_at='2026-11-05T00:00:00Z'),)
        model = CostModel('fixture', services={kind: (ServiceCost(.1, beta),) for kind, beta in
            [('send', .07), ('rrc', .11), ('copy', .03), ('reduce', .13)] if trial % 3 or kind != 'reduce'},
            resources={'nic': 7, 'pcie': 11}, profiles=profiles if trial % 2 else ())
        schedule = Executable(tuple(operations), output_operations=(operations[-1].operation_id,))
        monkeypatch.setattr(simulation, 'mpq', original)
        actual = simulation.predict(schedule, model, exact=True)
        monkeypatch.setattr(simulation, 'mpq', Fraction)
        expected = simulation.predict(schedule, model, exact=True)
        assert actual == expected
