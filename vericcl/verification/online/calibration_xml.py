from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from vericcl.errors import SemanticError
from vericcl.input.json_codec import sha256_json
from vericcl.input.models import (
    AtomConstraints,
    Hyperparameters,
    ResolvedInput,
    SolverConfig,
    StrategyConfig,
)
from vericcl.semantics.atom import Atom, PathStage, Schedule, Symbol, Transfer
from vericcl.semantics.collective import CollectiveKind, CollectiveSpec
from vericcl.topology.model import LinkKey, Topology
from vericcl.verification.online.calibration import (
    MAX_CALIBRATION_CONCURRENCY,
    CalibrationRequest,
)
from vericcl.verification.constraints import verify_schedule_pre_lowering
from vericcl.verification.model import ValidationStatus
from vericcl.xml.lower import XmlArtifact, lower_to_xml


@dataclass(frozen=True)
class CalibrationBenchmark:
    artifact: XmlArtifact
    schedule: Schedule
    inputs: ResolvedInput


def _validate_problem(
    request: CalibrationRequest,
    topology: Topology,
) -> None:
    if not isinstance(request, CalibrationRequest):
        raise SemanticError("request must be a CalibrationRequest")
    if not isinstance(topology, Topology):
        raise SemanticError("topology must be a Topology")
    if topology.rank_count != 2 or LinkKey(0, 1) not in topology.links:
        raise SemanticError(
            "calibration topology requires the directed link from rank 0 to 1"
        )
    same_node = topology.node_membership[0] == topology.node_membership[1]
    expected_same_node = request.link_class == "intra_node"
    if same_node != expected_same_node:
        raise SemanticError(
            "calibration topology does not match the requested link class"
        )


def _concurrency_limit(
    request: CalibrationRequest,
    topology: Topology,
) -> int:
    count = request.benchmark_slice_count
    if count is None:
        return 0
    return min(
        request.max_calibration_channels,
        MAX_CALIBRATION_CONCURRENCY,
        count,
        topology.links[LinkKey(0, 1)].max_channels,
    )


def _inputs(request: CalibrationRequest) -> ResolvedInput:
    collective = CollectiveSpec(
        kind=CollectiveKind.BROADCAST,
        datatype=request.datatype,
        reduction_op=None,
        root=0,
        inplace=True,
    )
    hyperparameters = Hyperparameters(
        total_size_bytes=request.benchmark_size_bytes,
        slice_size_bytes=request.slice_size_bytes,
        max_calibration_channels=request.max_calibration_channels,
    )
    signature_payload = {
        "collective": collective,
        "total_size_bytes": request.benchmark_size_bytes,
        "slice_size_bytes": request.slice_size_bytes,
        "max_calibration_channels": request.max_calibration_channels,
    }
    return ResolvedInput(
        collective=collective,
        hyperparameters=hyperparameters,
        solver=SolverConfig(),
        strategies=StrategyConfig(
            hierarchy=False,
            symmetry=False,
            shortest_paths=False,
            batching=False,
            constructive_trees=True,
            milp=False,
            manual_hierarchy=(),
        ),
        atom_constraints=AtomConstraints(None, ()),
        rank_count=2,
        resolved_topology={},
        resolved_sketch={},
        resolved_atom={},
        input_sha256=sha256_json(signature_payload),
    )


def _schedule(
    request: CalibrationRequest,
    concurrency: int,
) -> Schedule:
    count = request.benchmark_slice_count
    if count is None:
        raise SemanticError(
            "calibration benchmark size is not divisible by slice size"
        )
    transfers = []
    semantic = {}
    final_outputs = {}
    final_dependencies = {}
    for logical in range(count):
        transfer_id = "calibration-send-{:08d}".format(logical)
        wave = logical // concurrency
        start = float(wave)
        end = start + 1.0
        stage = PathStage(
            stage_id=0,
            operator="SEND",
            symbols=(Symbol(0, 1, 0.0),),
        )
        atom = Atom(
            slice_id=logical,
            slice_size_bytes=request.slice_size_bytes,
            path=(stage,),
            st_time=start,
            ed_time=end,
        )
        transfers.append(
            Transfer(
                transfer_id=transfer_id,
                kind="SEND",
                src_rank=0,
                dst_rank=1,
                channel=logical % concurrency,
                stage_id=0,
                member_slice_ids=frozenset({logical}),
                atoms=(atom,),
                st_time=start,
                ed_time=end,
                predecessor_ids=frozenset(),
            )
        )
        semantic[transfer_id] = ()
        for rank in (0, 1):
            output_id = "r{:08d}-o{:08d}".format(rank, logical)
            final_outputs[output_id] = (logical,)
            final_dependencies[output_id] = (
                () if rank == 0 else (transfer_id,)
            )
    return Schedule(
        schedule_id="calibration-k{:02d}".format(concurrency),
        transfers=tuple(transfers),
        final_state_ids=tuple(sorted(final_outputs)),
        rank_count=2,
        slice_count=count,
        slice_size_bytes=request.slice_size_bytes,
        metadata={
            "path_scope": "global",
            "semantic_predecessors": semantic,
            "final_outputs": final_outputs,
            "final_dependencies": final_dependencies,
            "calibration_concurrency": concurrency,
            "calibration_full_wave_count": count // concurrency,
            "calibration_tail_transfer_count": count % concurrency,
        },
    )


def build_calibration_artifact(
    request: CalibrationRequest,
    topology: Topology,
    *,
    concurrency: int,
) -> XmlArtifact:
    _validate_problem(request, topology)
    if (
        isinstance(concurrency, bool)
        or not isinstance(concurrency, int)
        or concurrency < 1
        or concurrency > _concurrency_limit(request, topology)
    ):
        raise SemanticError("calibration concurrency is outside the limit")
    return build_calibration_benchmark(
        request,
        topology,
        concurrency=concurrency,
    ).artifact


def build_calibration_benchmark(
    request: CalibrationRequest,
    topology: Topology,
    *,
    concurrency: int,
) -> CalibrationBenchmark:
    _validate_problem(request, topology)
    if (
        isinstance(concurrency, bool)
        or not isinstance(concurrency, int)
        or concurrency < 1
        or concurrency > _concurrency_limit(request, topology)
    ):
        raise SemanticError("calibration concurrency is outside the limit")
    schedule = _schedule(request, concurrency)
    inputs = _inputs(request)
    checks = verify_schedule_pre_lowering(schedule, inputs, topology)
    failed = tuple(
        check
        for check in checks
        if check.status is not ValidationStatus.VALID
    )
    if failed:
        raise SemanticError(
            "calibration schedule pre-validation failed: {}".format(
                ", ".join(check.code for check in failed)
            )
        )
    return CalibrationBenchmark(
        artifact=lower_to_xml(schedule, inputs, topology),
        schedule=schedule,
        inputs=inputs,
    )


def build_calibration_artifacts(
    request: CalibrationRequest,
    topology: Topology,
) -> Tuple[XmlArtifact, ...]:
    _validate_problem(request, topology)
    limit = _concurrency_limit(request, topology)
    if limit == 0:
        return ()
    return tuple(
        build_calibration_artifact(
            request,
            topology,
            concurrency=concurrency,
        )
        for concurrency in range(1, limit + 1)
    )
