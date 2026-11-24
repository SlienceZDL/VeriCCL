"""Full symbolic execution and BDD reasoning, behind the switch ``sketch.symbolic.enabled``.

The switch is off by default and the ``sketch.symbolic`` section is optional:
without it the request, its input hash and every code path are exactly the
standard VeriCCL ones. With ``enabled: true`` three places use this package:

* the correctness gate of every lowered candidate (``checks.full_check``):
  the emitted XML is parsed on its own, executed symbolically over per-rank
  buffer state on a runtime-faithful progress model (threadblock order,
  dependency flags, send/receive pairing, bounded connection FIFOs, several
  loop passes), and every check is a BDD computation;
* hotspot localization and threadblock-order hints of the offline analysis
  (``hotspots.locate``, ``order.tb_order_hints``);
* the related-schedule scope and repair-plan screening of scheduling-bottleneck
  repair (``repair.scope_and_plans``), which explores the plan decisions as
  BDD path conditions.

Everything else (synthesis, lowering, the cost model and its simulator,
calibration, measurement, acceptance) is unchanged.
"""
