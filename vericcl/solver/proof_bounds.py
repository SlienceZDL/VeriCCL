"""Exact, scoped lower bounds; floating solver incumbents are not certificates."""
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json

from vericcl.errors import SemanticError


def rational(value):
    if isinstance(value, bool):
        raise SemanticError('rational evidence cannot contain booleans')
    try:
        return value if isinstance(value, Fraction) else Fraction(str(value))
    except (ValueError, TypeError, ZeroDivisionError) as error:
        raise SemanticError('evidence must contain finite rational numbers') from error


@dataclass(frozen=True)
class VerifiedBound:
    value: Fraction
    model_version: str
    scope_id: str
    certificate_digest: str
    certificate: dict

    @property
    def value_ratio(self):
        return self.value.numerator, self.value.denominator

    def can_prune(self, incumbent_ratio, *, model_version, scope_id):
        if model_version != self.model_version or scope_id != self.scope_id:
            return False
        incumbent = Fraction(*incumbent_ratio)
        return self.value >= incumbent


def verify_lp_dual(*, matrix, rhs, objective, dual, model_version, scope_id):
    """Check y>=0 and A^T y<=c for min c*x, A*x>=b, x>=0.

    A caller must separately establish that this LP is a relaxation of the
    stated complete objective and scope. The certificate binds that scope and
    the immutable model version, so it cannot be reused after either changes.
    """
    if not model_version or not scope_id:
        raise SemanticError('lower-bound evidence needs model and scope identities')
    a = tuple(tuple(rational(value) for value in row) for row in matrix)
    b, c, y = (tuple(rational(value) for value in sequence) for sequence in (rhs, objective, dual))
    if len(a) != len(b) or len(a) != len(y) or any(len(row) != len(c) for row in a):
        raise SemanticError('dual certificate dimensions disagree')
    if any(value < 0 for value in y):
        raise SemanticError('dual multiplier must be nonnegative')
    if any(sum(row[column] * weight for row, weight in zip(a, y)) > c[column] for column in range(len(c))):
        raise SemanticError('dual certificate violates A-transpose*y <= c')
    value = sum((entry * weight for entry, weight in zip(b, y)), Fraction(0))
    certificate = {'matrix': [[str(value) for value in row] for row in a], 'rhs': list(map(str, b)),
                   'objective': list(map(str, c)), 'dual': list(map(str, y)),
                   'model_version': model_version, 'scope_id': scope_id, 'form': 'min_cx_Ax_ge_b_x_ge_0'}
    digest = hashlib.sha256(json.dumps(certificate, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return VerifiedBound(value, model_version, scope_id, digest, certificate)
