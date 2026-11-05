from dataclasses import replace

import pytest

from vericcl.verification.bdd_order import analyze_tb_order
from vericcl.verification.model import ValidationStatus
from vericcl.xml.threadblocks import ThreadblockProgram

from tests.unit.verification.bdd_helpers import tb_order_case


pytestmark = pytest.mark.phase05


def test_later_step_ready_first_produces_swap_hint():
    program, schedule = tb_order_case()

    result = analyze_tb_order(program, schedule)

    assert result.status is ValidationStatus.VALID
    assert len(result.hints) == 1
    hint = result.hints[0]
    assert hint.tb_id == "r00000000-tb00000000"
    assert hint.earlier_step_id == "slow-send"
    assert hint.later_step_id == "fast-send"
    assert hint.earlier_step_index == 0
    assert hint.later_step_index == 1
    assert hint.earlier_ready_time_us == pytest.approx(5.0)
    assert hint.later_ready_time_us == pytest.approx(0.0)


def test_necessary_order_suppresses_swap_hint():
    program, schedule = tb_order_case(necessary_order=True)

    result = analyze_tb_order(program, schedule)

    assert result.status is ValidationStatus.VALID
    assert result.hints == ()


def test_xml_semantic_predecessor_suppresses_swap_hint():
    program, schedule = tb_order_case()
    block = program.threadblocks[0]
    slow, fast = block.steps
    fast = replace(
        fast,
        semantic_predecessor_node_ids=("slow",),
    )
    block = replace(block, steps=(slow, fast))
    program = ThreadblockProgram(
        threadblocks=(block,),
        steps_by_id={slow.step_id: slow, fast.step_id: fast},
        transfer_steps={},
        node_steps={"slow": (slow.step_id,), "fast": (fast.step_id,)},
        referenced_step_ids=frozenset(),
        inversion_count=1,
    )

    result = analyze_tb_order(program, schedule)

    assert result.status is ValidationStatus.VALID
    assert result.hints == ()


def test_order_backend_exception_becomes_analysis_error(monkeypatch):
    program, schedule = tb_order_case()

    def fail_backend(*args, **kwargs):
        raise RuntimeError("backend failed")

    monkeypatch.setattr(
        "vericcl.verification.bdd_order.CompactBDD",
        fail_backend,
    )
    result = analyze_tb_order(program, schedule)

    assert result.status is ValidationStatus.ANALYSIS_ERROR
    assert result.hints == ()
    assert result.code == "tb_order_bdd_analysis_error"
