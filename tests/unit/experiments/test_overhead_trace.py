from types import SimpleNamespace
import pytest


def test_trace_replay_selects_inplace_samples_and_discards_warmup():
    from vericcl.experiments.overhead_trace import select_inplace_records
    records = {rank: tuple(SimpleNamespace(iteration=i) for i in range(42))
               for rank in range(2)}
    selected = select_inplace_records(records)
    assert [record.iteration for record in selected[0]] == list(range(22, 42))


def test_trace_replay_rejects_mismatched_rank_invocations():
    from vericcl.experiments.overhead_trace import select_inplace_records
    with pytest.raises(ValueError, match="invocations"):
        select_inplace_records({0: (SimpleNamespace(iteration=0),),
                               1: (SimpleNamespace(iteration=1),)})


def test_trace_replay_rejects_different_xml(tmp_path):
    from vericcl.experiments.overhead_trace import verify_xml_identity
    path = tmp_path / "schedule.xml"
    path.write_text("wrong XML")
    with pytest.raises(ValueError, match="XML identity"):
        verify_xml_identity(path, "0" * 64)
