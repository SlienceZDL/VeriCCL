import importlib
import importlib.util

from vericcl.errors import SolverUnavailableError


class GurobiAdapter:
    @staticmethod
    def available() -> bool:
        return importlib.util.find_spec("gurobipy") is not None

    @classmethod
    def require(cls):
        if not cls.available():
            raise SolverUnavailableError(
                "gurobipy is unavailable; install Gurobi to enable MILP solving"
            )
        try:
            return importlib.import_module("gurobipy")
        except ImportError as error:
            raise SolverUnavailableError(
                "gurobipy could not be imported"
            ) from error
