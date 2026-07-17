class VeriCCLError(Exception):
    """Base error for expected VeriCCL failures."""


class InputValidationError(VeriCCLError):
    """Raised when an input file violates the public schema."""


class SemanticError(VeriCCLError):
    """Raised when a collective state transition is invalid."""


class SolverUnavailableError(VeriCCLError):
    """Raised when a requested solver backend is unavailable."""


class ConstructionInfeasibleError(VeriCCLError):
    """Raised when a constructive solver cannot produce a legal schedule."""


class RuntimeCompatibilityError(VeriCCLError):
    """Raised when an output cannot run on the selected runtime."""
