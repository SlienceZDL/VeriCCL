"""Analysis inputs (hotspot report, repair strategies, causes) in full symbolic mode."""
from dataclasses import replace

from .cache import execution_for
from .hotspots import locate
from .order import tb_order_hints


def analysis_inputs(bundle, lowering, model, backend, index):
    """``bundle`` carries the prediction under ``model``; ``index`` is the causal index."""
    from vericcl.verification.bottlenecks import BottleneckCause
    from vericcl.verification.hotspots import hotspot_strategies
    execution = execution_for(bundle, lowering.inputs, backend=backend, lease=lowering.lease)
    report = locate(execution, bundle, lowering.inputs, lowering.topology, model, backend=backend)
    strategies = tuple(hotspot_strategies(report, bundle, replace(lowering, model=model)))
    hints = tb_order_hints(execution, lowering.inputs, lease=lowering.lease)
    strategies += tuple(('order', record) for record, _ in hints)
    prediction = bundle.prediction
    causes = tuple(BottleneckCause(
        'hotspot:' + item.operation_id, 'scheduling_hotspot', (item.operation_id,), item.related_operations,
        (), (item.congested_resource,),
        {'service_us': prediction.end_times[item.operation_id] - prediction.start_times[item.operation_id],
         'start_us': prediction.start_times[item.operation_id], 'finish_us': prediction.end_times[item.operation_id],
         'resource_competitors': list(item.related_operations), 'completion_tails': list(index.tails),
         'contention_us': item.contention_us, 'wait_us': item.wait_us, 'buckets': list(item.buckets),
         'alternative_links': [list(link) for link in item.alternative_links],
         'on_critical_path': item.on_critical_path, 'downstream_operations': item.downstream_operations,
         'score': item.score, **item.evidence})
        for item in report.hotspots if item.operation_id in prediction.end_times)
    causes += tuple(BottleneckCause(
        record.cause_ids[0], 'tb_order_inversion', record.operation_ids, (), (), (),
        {'gain_us': gain, 'run_first': record.orders[0][0], 'evidence_level': 'full_symbolic_execution_bdd'})
        for record, gain in hints)
    return report, strategies, causes
