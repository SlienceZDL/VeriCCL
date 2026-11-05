import math

import pytest


def test_relative_fit_recovers_positive_cost_and_requires_identifiable_bandwidth():
    from vericcl.verification.online.cost_fitting import fit_curve
    result = fit_curve([(size, 2 + .01 * size) for size in (100, 200, 400, 800)], .01)
    assert result.status == 'qualified'
    assert len(result.segments) == 1
    assert result.segments[0].alpha_us == pytest.approx(2)
    assert result.segments[0].beta_us_per_byte == pytest.approx(.01)
    assert fit_curve([(1, 3), (2, 3), (3, 3)], .1).status == 'unidentifiable'
    assert fit_curve([(1, 4), (2, 3), (3, 2)], .1).status != 'qualified'


def test_relative_fit_enforces_alpha_boundary_without_clipping_another_fit():
    from vericcl.verification.online.cost_fitting import fit_curve
    points = [(100, .9), (200, 1.9), (400, 3.9)]
    result = fit_curve(points, .1)
    assert result.status == 'qualified'
    piece = result.segments[0]
    assert piece.alpha_us == 0 and piece.beta_us_per_byte > 0
    # The constrained optimum recomputes beta after alpha reaches zero.
    expected = sum(x / y for x, y in points) / sum((x / y) ** 2 for x, y in points)
    assert piece.beta_us_per_byte == pytest.approx(expected)


@pytest.mark.parametrize('bad', [0, -1, math.nan, math.inf, True])
def test_invalid_duration_never_produces_a_cost_component(bad):
    from vericcl.verification.online.cost_fitting import fit_curve
    with pytest.raises(ValueError):
        fit_curve([(100, 2), (200, bad), (400, 4)], .1)


def test_piecewise_change_requires_independent_confirmation_and_preserves_gaps():
    from vericcl.verification.online.cost_fitting import fit_curve
    points = [(x, 1 + .01 * x) for x in (100, 200, 300)] + [(x, 8 + .04 * x) for x in (1000, 2000, 3000)]
    proposed = fit_curve(points, .01)
    assert proposed.status == 'needs_confirmation' and not proposed.segments
    confirmed = fit_curve(points, .01, confirmation=points)
    assert confirmed.status == 'qualified' and len(confirmed.segments) == 2
    assert [(s.min_bytes, s.max_bytes) for s in confirmed.segments] == [(100, 300), (1000, 3000)]
    assert not any(piece.covers(500) for piece in confirmed.segments)
    assert fit_curve(points, .01, confirmation=[(x, y * 1.2) for x, y in points]).status == 'needs_confirmation'


def test_too_few_distinct_payloads_are_not_joint_alpha_beta_measurement():
    from vericcl.verification.online.cost_fitting import fit_curve
    assert fit_curve([(100, 2), (200, 3)], .1).status == 'insufficient_sizes'
    with pytest.raises(ValueError):
        fit_curve([(100, 2), (100, 2), (200, 3)], .1)
