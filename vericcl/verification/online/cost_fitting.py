"""Relative-residual alpha/beta fits; qualification is separate from provenance."""
from dataclasses import dataclass
import math

import numpy as np

from vericcl.performance.model import ServiceCost


@dataclass(frozen=True)
class CurveFit:
    status: str
    segments: tuple = ()
    proposed_segments: tuple = ()
    diagnostics: tuple = ()


def _points(values):
    result = []
    for size, duration in values:
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError('fit payloads must be positive integer bytes')
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
            raise ValueError('fit durations must be finite and positive')
        result.append((size, float(duration)))
    result.sort()
    if len({size for size, _ in result}) != len(result):
        raise ValueError('fit needs distinct payload medians')
    return tuple(result)


def _linear(points):
    largest = points[-1][0]
    minimum = min(y for _, y in points)
    weights = np.array([minimum / y for _, y in points])
    x = np.array([size / largest for size, _ in points])
    design = np.column_stack((weights, weights * x))
    values = np.full(len(points), minimum)
    solution, _, rank, _ = np.linalg.lstsq(design, values, rcond=None)
    if rank != 2:
        return None, {'reason': 'rank_deficient_payloads'}
    alpha, scaled_beta = map(float, solution)
    if alpha < 0:
        alpha = 0.
        z = design[:, 1]
        scaled_beta = float(np.dot(z, values) / np.dot(z, z))
    beta = scaled_beta / largest
    signal = beta * (points[-1][0] - points[0][0])
    if not math.isfinite(alpha + beta) or beta <= 0 or signal <= max(y for _, y in points) * 1e-10:
        return None, {'reason': 'nonpositive_or_unidentifiable_slope'}
    residuals = tuple((alpha + beta * x - y) / y for x, y in points)
    diagnostic = {'sizes': [x for x, _ in points], 'relative_residuals': residuals,
                  'relative_sse': math.fsum(r * r for r in residuals),
                  'max_relative_error': max(abs(r) for r in residuals)}
    return ServiceCost(alpha, beta, 'estimated', points[0][0], points[-1][0]), diagnostic


def fit_curve(points, max_relative_error, *, confirmation=(), minimum_sizes=3, lease=None):
    """Fit medians without extending the measured size range.

    Piecewise suggestions become usable only after separate confirmation data
    agrees in every segment. The caller must verify independent run identities.
    """
    if isinstance(max_relative_error, bool) or not isinstance(max_relative_error, (int, float)) or not math.isfinite(max_relative_error) or max_relative_error < 0:
        raise ValueError('relative fit tolerance must be finite and nonnegative')
    if isinstance(minimum_sizes, bool) or not isinstance(minimum_sizes, int) or minimum_sizes < 3:
        raise ValueError('joint alpha/beta fitting needs at least three sizes')
    points = _points(points)
    confirmation = _points(confirmation)
    if lease is not None:
        lease.check()
    if len(points) < minimum_sizes:
        return CurveFit('insufficient_sizes')
    full, diagnostic = _linear(points)
    if full is not None and diagnostic['max_relative_error'] <= max_relative_error:
        return CurveFit('qualified', (full,), diagnostics=(diagnostic,))
    # Minimize the number of independently supported contiguous segments.
    # Gaps between sampled ranges stay uncovered instead of being extrapolated.
    best = {0: ((), ())}
    for end in range(minimum_sizes, len(points) + 1):
        for start in range(0, end - minimum_sizes + 1):
            if lease is not None:
                lease.check()
            if start not in best:
                continue
            piece, report = _linear(points[start:end])
            if piece is None or report['max_relative_error'] > max_relative_error:
                continue
            previous, reports = best[start]
            candidate = previous + (piece,)
            if end not in best or len(candidate) < len(best[end][0]):
                best[end] = candidate, reports + (report,)
    if len(points) not in best:
        return CurveFit('unidentifiable' if full is None else 'poor_fit', diagnostics=(diagnostic,))
    pieces, reports = best[len(points)]
    confirmed = True
    confirmation_reports = []
    for piece in pieces:
        observed = [(x, y) for x, y in confirmation if piece.covers(x)]
        residuals = [(piece.alpha_us + piece.beta_us_per_byte * x - y) / y for x, y in observed]
        passed = len(observed) >= minimum_sizes and all(abs(r) <= max_relative_error for r in residuals)
        confirmed = confirmed and passed
        confirmation_reports.append({'confirmation_sizes': [x for x, _ in observed],
                                      'confirmation_relative_residuals': residuals, 'confirmed': passed})
    return CurveFit('qualified' if confirmed else 'needs_confirmation', pieces if confirmed else (),
                    () if confirmed else pieces, reports + tuple(confirmation_reports))
