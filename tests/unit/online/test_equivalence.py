import pytest

from tests.unit.online.test_probe_jobs import probe_context
from tests.unit.online.test_component_measurements import mathematical_native_contract
from vericcl.performance.initial import initial_cost_model
from vericcl.verification.online.equivalence import ProbeClasses
from vericcl.verification.online.probe_jobs import ProbeFlow, mode_jobs, plan_probes


def classes_context(path, **options):
    path.mkdir(exist_ok=True)
    context = probe_context(path, **options)
    context.classes = ProbeClasses(context.inputs, context.topology)
    return context


def _member_link(context):
    classes = context.classes.link_classes()
    for pairs in classes.values():
        if len(pairs) > 1:
            return pairs[0], pairs[1]
    pytest.skip('fixture topology has no isomorphic links')


def test_each_link_class_is_planned_once_and_every_link_has_a_class(tmp_path):
    (tmp_path / 'full').mkdir()
    full = list(plan_probes(probe_context(tmp_path / 'full', channels=2, ranks=3)))
    context = classes_context(tmp_path / 'classes', channels=2, ranks=3)
    reduced = list(plan_probes(context))
    classes = context.classes.link_classes()
    assert sorted(pair for pairs in classes.values() for pair in pairs) == sorted(
        (key.src_rank, key.dst_rank) for key in context.topology.links)
    planned = {(job.flows[0].src_rank, job.flows[0].dst_rank) for job in reduced if job.category == 'link'}
    assert len(planned) == len(classes)
    assert len(reduced) < len(full)
    assert len({job.job_id for job in reduced}) == len(reduced)


def test_measured_representative_is_published_to_every_member(tmp_path, monkeypatch):
    from vericcl.verification.online import campaign
    context = classes_context(tmp_path, channels=1, ranks=3)
    jobs = [job for job in plan_probes(context) if job.category in {'link', 'copy'}]
    calls = []

    def measure(job, *_):
        calls.append(job.job_id)
        return mathematical_native_contract(job)
    monkeypatch.setattr(campaign, 'execute_probe', measure)
    result = campaign.run_calibration(context, initial_cost_model(context.inputs), jobs=jobs)
    assert len(calls) == len(jobs)
    size = context.inputs.hyperparameters.slice_size_bytes
    for key in context.topology.links:
        pieces = result.model.services['link:%d:%d' % (key.src_rank, key.dst_rank)]
        assert any(piece.source == 'measured' and piece.covers(size) for piece in pieces)
    for rank in range(context.inputs.rank_count):
        assert any(piece.source == 'measured' for piece in result.model.services['copy:%d' % rank])
    signatures = result.model.metadata['calibration_signatures']
    representative, member = _member_link(context)
    assert signatures['service:link:%d:%d' % member] == signatures['service:link:%d:%d' % representative]


def test_known_class_publishes_a_new_member_mode_without_another_launch(tmp_path, monkeypatch):
    from vericcl.verification.online import campaign
    context = classes_context(tmp_path, channels=2, ranks=3)
    representative, member = _member_link(context)
    calls = []

    def measure(job, *_):
        calls.append(job.flows)
        return mathematical_native_contract(job)
    monkeypatch.setattr(campaign, 'execute_probe', measure)
    first = mode_jobs(context, [tuple(ProbeFlow('send', *representative, channel) for channel in (0, 1))])
    measured = campaign.run_calibration(context, initial_cost_model(context.inputs), jobs=first)
    assert len(calls) == len(first)
    second = mode_jobs(context, [tuple(ProbeFlow('send', *member, channel) for channel in (0, 1))])
    assert {job.flows for job in second} == {job.flows for job in first}
    calls.clear()
    result = campaign.run_calibration(context, measured.model, jobs=second)
    assert not calls
    mode = tuple(sorted('send:%d:%d:c%d' % (*member, channel) for channel in (0, 1)))
    published = [profile for profile in result.model.profiles if profile.mode == mode]
    assert published and all(profile.source == 'measured' for profile in published)
    assert all(set(profile.flow_resources[flow]) >= {'link:%d:%d' % member} for profile in published for flow in mode)
