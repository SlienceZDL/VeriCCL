from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional, Tuple

from vericcl.constants import SOFTWARE_MAX_CONCURRENCY
from vericcl.semantics.collective import CollectiveSpec


class ObjectiveMode(str, Enum):
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    AUTO = "auto"


@dataclass(frozen=True)
class Hyperparameters:
    total_size_bytes: int
    slice_size_bytes: int
    objective_mode: ObjectiveMode = ObjectiveMode.AUTO
    max_calibration_channels: int = SOFTWARE_MAX_CONCURRENCY
    min_expected_improvement: float = 0.01
    min_tuning_improvement: float = 0.01
    max_tuning_iterations: int = 20
    total_verification_timeout_s: int = 10800
    force_recalibrate: bool = False

    @property
    def slice_count(self) -> int:
        return self.total_size_bytes // self.slice_size_bytes


@dataclass(frozen=True)
class SolverConfig:
    total_solve_timeout_s: int = 10800
    per_model_timeout_s: int = 1800
    mip_gap: float = 1e-4
    require_proven_optimal: bool = False
    solver_seed: int = 0
    max_channels: int = SOFTWARE_MAX_CONCURRENCY
    max_threads_per_model: int = 12
    max_parallel_models: int = 4
    force_resolve: bool = False


@dataclass(frozen=True)
class ForbiddenTransfer:
    slice_id: int
    src_rank: int
    dst_rank: int
    stage_id: int


@dataclass(frozen=True)
class AtomConstraints:
    stage_num: Optional[int]
    forbidden_transfers: Tuple[ForbiddenTransfer, ...]


@dataclass(frozen=True)
class StrategyConfig:
    hierarchy: bool
    symmetry: bool
    shortest_paths: bool
    batching: bool
    constructive_trees: bool
    milp: bool
    manual_hierarchy: Tuple[Mapping[str, object], ...]


@dataclass(frozen=True)
class ResolvedInput:
    collective: CollectiveSpec
    hyperparameters: Hyperparameters
    solver: SolverConfig
    strategies: StrategyConfig
    atom_constraints: AtomConstraints
    rank_count: int
    resolved_topology: Mapping[str, object]
    resolved_sketch: Mapping[str, object]
    resolved_atom: Mapping[str, object]
    input_sha256: str
