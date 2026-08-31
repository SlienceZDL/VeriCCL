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

    @classmethod
    def create_environment(cls):
        gp = cls.require()
        environment = None
        try:
            environment = gp.Env(empty=True)
            environment.setParam("OutputFlag", 0)
            environment.start()
            return environment
        except gp.GurobiError as error:
            if environment is not None:
                environment.dispose()
            raise SolverUnavailableError(
                "Gurobi environment creation failed: {}".format(error)
            ) from error

    @staticmethod
    def dispose_environment(environment) -> None:
        environment.dispose()

    @classmethod
    def create_model(cls, name, environment=None):
        gp = cls.require()
        try:
            if environment is None:
                return gp, gp.Model(name)
            return gp, gp.Model(name, env=environment)
        except gp.GurobiError as error:
            raise SolverUnavailableError(
                "Gurobi model creation failed: {}".format(error)
            ) from error

    @staticmethod
    def version(gp) -> str:
        return ".".join(str(value) for value in gp.gurobi.version())
