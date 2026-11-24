"""Settings of the full symbolic mode (``sketch.symbolic``)."""
from dataclasses import dataclass, fields

from vericcl.errors import SemanticError


@dataclass(frozen=True)
class FullSymbolicSettings:
    enabled: bool = False
    # Loop passes modelled by the progress check. MSCCL runs the whole step
    # program once per pass; dependency flags and connection FIFOs carry over
    # from one pass to the next, so two passes expose cross-pass waits.
    loop_iterations: int = 2
    # Connection buffer of the pinned runtime: NCCL_STEPS (8) slots; a step of
    # a full MSCCL chunk occupies MSCCL_CHUNKSTEPS (4) of them, a smaller step
    # fewer. The messages a connection holds follow from the checked size.
    fifo_slots: int = 8
    chunk_steps: int = 4
    max_related_operations: int = 48
    max_order_hints: int = 32
    max_plan_choices: int = 6

    def __post_init__(self):
        for name in ('loop_iterations', 'fifo_slots', 'chunk_steps', 'max_related_operations', 'max_order_hints',
                     'max_plan_choices'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise SemanticError('sketch.symbolic.%s must be a positive integer' % name)


def settings(inputs) -> FullSymbolicSettings:
    config = getattr(inputs, 'request_config', None) or {}
    raw = (config.get('sketch') or {}).get('symbolic') or {}
    return FullSymbolicSettings(**{item.name: raw[item.name] for item in fields(FullSymbolicSettings)
                                   if raw.get(item.name) is not None})


def enabled(inputs) -> bool:
    config = getattr(inputs, 'request_config', None) or {}
    raw = (config.get('sketch') or {}).get('symbolic') or {}
    return bool(raw.get('enabled', False))
