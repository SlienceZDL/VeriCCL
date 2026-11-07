"""Online measurement exports, resolved lazily (PEP 562).

Every calibration launch runs several ``python -m
vericcl.verification.online.remote`` helpers (prepare, one per rank, cleanup,
collect). Eagerly importing the calibration, trace and pipeline modules here
cost each helper about a third of a second. Public names are unchanged.
"""
from importlib import import_module

_EXPORTS = {
    "CalibrationCache": "cache",
    "EnvironmentSignature": "cache",
    "environment_signature_sha256": "cache",
    "CalibrationPoint": "calibration",
    "CalibrationRequest": "calibration",
    "CalibrationResult": "calibration",
    "apply_calibration_to_topology": "calibration",
    "calibration_point_from_trace": "calibration",
    "derive_calibrated_curve": "calibration",
    "CalibrationBenchmark": "calibration_xml",
    "build_calibration_artifact": "calibration_xml",
    "build_calibration_artifacts": "calibration_xml",
    "build_calibration_benchmark": "calibration_xml",
    "NcclTestMeasurement": "model",
    "NcclTestRequest": "model",
    "NcclTestRun": "model",
    "PerformanceStatistics": "model",
    "NcclTestsHelpValidator": "nccl_tests",
    "build_nccl_tests_command": "nccl_tests",
    "build_nccl_tests_trace_command": "nccl_tests",
    "parse_nccl_tests_output": "nccl_tests",
    "PerformanceHistory": "statistics",
    "summarize_runs": "statistics",
    "AlignedTimestamp": "clock_sync",
    "ClockAlignment": "clock_sync",
    "ClockOrdering": "clock_sync",
    "ClockSyncSample": "clock_sync",
    "ClockTransform": "clock_sync",
    "align_clocks": "clock_sync",
    "parse_clock_sync_output": "clock_sync",
    "BottleneckRecord": "trace_analysis",
    "PhysicalTransferInterval": "trace_analysis",
    "StepWaitAnalysis": "trace_analysis",
    "TraceAnalysis": "trace_analysis",
    "WaitClass": "trace_analysis",
    "WaitDurations": "trace_analysis",
    "analyze_trace": "trace_analysis",
    "decompose_waits": "trace_analysis",
    "pair_endpoints": "trace_analysis",
    "RawStepTraceRecord": "trace_format",
    "StepTraceRecord": "trace_format",
    "encode_raw_trace": "trace_format",
    "parse_trace": "trace_format",
    "CommandExecutor": "runner",
    "NcclTestsRunner": "runner",
    "ProcessRequest": "runner",
    "ProcessResult": "runner",
    "SubprocessCommandExecutor": "runner",
    "TraceCollectionRequest": "runner",
    "TraceCollectionResult": "runner",
    "collect_trace_files": "runner",
    "process_environment": "runner",
    "CalibrationPlan": "pipeline",
    "OnlineCalibrationOutcome": "pipeline",
    "OnlineContext": "pipeline",
    "OnlineStageStatus": "pipeline",
    "OnlineTuningEvidence": "pipeline",
    "OnlineValidationResult": "pipeline",
    "attach_online_result_to_tuning_context": "pipeline",
    "run_online_validation": "pipeline",
}

__all__ = [
    "CalibrationCache",
    "CalibrationBenchmark",
    "CalibrationPlan",
    "CalibrationPoint",
    "CalibrationRequest",
    "CalibrationResult",
    "AlignedTimestamp",
    "BottleneckRecord",
    "ClockAlignment",
    "ClockOrdering",
    "ClockSyncSample",
    "ClockTransform",
    "CommandExecutor",
    "EnvironmentSignature",
    "NcclTestMeasurement",
    "NcclTestRequest",
    "NcclTestRun",
    "NcclTestsHelpValidator",
    "NcclTestsRunner",
    "OnlineCalibrationOutcome",
    "OnlineContext",
    "OnlineStageStatus",
    "OnlineTuningEvidence",
    "OnlineValidationResult",
    "PerformanceHistory",
    "PerformanceStatistics",
    "PhysicalTransferInterval",
    "ProcessRequest",
    "ProcessResult",
    "RawStepTraceRecord",
    "StepTraceRecord",
    "StepWaitAnalysis",
    "SubprocessCommandExecutor",
    "TraceAnalysis",
    "TraceCollectionRequest",
    "TraceCollectionResult",
    "WaitClass",
    "WaitDurations",
    "align_clocks",
    "attach_online_result_to_tuning_context",
    "analyze_trace",
    "build_nccl_tests_command",
    "build_nccl_tests_trace_command",
    "build_calibration_artifact",
    "build_calibration_artifacts",
    "build_calibration_benchmark",
    "apply_calibration_to_topology",
    "calibration_point_from_trace",
    "derive_calibrated_curve",
    "decompose_waits",
    "encode_raw_trace",
    "environment_signature_sha256",
    "collect_trace_files",
    "parse_nccl_tests_output",
    "parse_clock_sync_output",
    "parse_trace",
    "pair_endpoints",
    "process_environment",
    "run_online_validation",
    "summarize_runs",
]


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is not None:
        value = getattr(import_module(__name__ + "." + module), name)
        globals()[name] = value
        return value
    if not name.startswith("__"):
        try:
            return import_module(__name__ + "." + name)
        except ModuleNotFoundError as error:
            if error.name != __name__ + "." + name:
                raise
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))


def __dir__():
    return sorted(set(globals()) | set(__all__))
