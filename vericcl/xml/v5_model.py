"""Shared v5 lowering records; native work and logical state stay distinct."""
from dataclasses import dataclass, field
from types import MappingProxyType

from vericcl.xml.capacity import RuntimeLimits


@dataclass(frozen=True)
class LoweringContext:
    inputs: object
    topology: object
    model: object
    lease: object
    limits: RuntimeLimits | None = None
    run_context: object = None

    def __post_init__(self):
        if self.limits is None:
            object.__setattr__(self, 'limits', RuntimeLimits.from_runtime(self.inputs.request_config['runtime']))


@dataclass(frozen=True, order=True)
class Address:
    rank: int
    buffer: str
    offset: int


@dataclass(frozen=True)
class NetworkOperation:
    operation_id: str
    kind: str
    src_rank: int
    dst_rank: int
    channel: int
    stage_id: int
    contributors: frozenset[int]
    target_contributors: frozenset[int]
    participants: tuple[int, ...]
    dependencies: frozenset[str]
    reference_dependencies: frozenset[str]
    priority_time: float


@dataclass(frozen=True)
class Value:
    state_id: str
    rank: int
    logical_position: int
    contributors: frozenset[int]
    producer: str | None


@dataclass(frozen=True)
class Instruction:
    operation_id: str
    kind: str
    inputs: tuple[str, ...]
    output: str
    rank: int
    peer: int
    channel: int
    stage_id: int
    participants: tuple[int, ...]
    dependencies: frozenset[str]
    priority_time: float


@dataclass(frozen=True)
class StoragePlan:
    values: object
    initial_ids: tuple
    instructions: tuple[Instruction, ...]
    addresses: object
    outputs: object
    input_slots: object
    output_slots: object
    scratch_slots: object
    reuse: tuple
    stage_count: int
    reference_orders_omitted: tuple = ()


@dataclass(frozen=True)
class NativeStep:
    step_id: str
    operation_id: str
    kind: str
    src: Address | None
    dst: Address | None
    dependency_tb: int = -1
    dependency_step: int = -1
    has_dependence: bool = False
    count: int = 1


@dataclass(frozen=True)
class NativeBlock:
    rank: int
    tb_id: int
    channel: int
    send_peer: int
    recv_peer: int
    steps: tuple[NativeStep, ...]


@dataclass(frozen=True)
class NativeProgram:
    rank_count: int
    channels: int
    chunks_per_loop: int
    blocks: tuple[NativeBlock, ...]
    input_slots: object
    output_slots: object
    scratch_slots: object


@dataclass(frozen=True)
class CandidateBundle:
    source: object
    xml_text: str | None
    xml_sha256: str | None
    sidecar: dict
    program: NativeProgram | None
    version_schedule: object
    executable: object
    prediction: object
    checks: object
    capacity_report: object = None
    storage_plan: StoragePlan | None = None

    @property
    def can_execute(self):
        required = {'semantic', 'storage', 'endpoint', 'deadlock', 'xml', 'launch', 'prediction'}
        return (required <= self.checks.keys() and all(self.checks[key].status.value == 'valid' for key in required)
                and self.checks['compatibility'].status.value in {'valid', 'not_run'})

    @property
    def can_repair(self):
        return self.source is not None


@dataclass(frozen=True)
class NetworkCandidate:
    rank_count: int
    slice_count: int
    slice_size_bytes: int
    operations: tuple[NetworkOperation, ...]
    origin: str
    replaced_operations: tuple = ()


@dataclass(frozen=True)
class StorageCandidate:
    storage_plan: StoragePlan
    origin: object
    replaced_operations: tuple = ()
