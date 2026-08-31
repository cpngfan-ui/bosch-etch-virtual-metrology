"""Data preparation and modeling tools for BOSCH etch virtual metrology."""

from .features import (
    FeatureExtractionError,
    FeatureResult,
    FeatureTables,
    build_feature_tables,
    extract_run_features,
    predictor_feature_names,
    segment_process_run,
    select_longest_contiguous_block,
)
from .io import (
    ProcessRun,
    WaferCohort,
    build_wafer_cohort,
    canonical_experiment_key,
    load_mean_si_etch_targets,
    load_process_decoder,
    load_process_runs,
)

__all__ = [
    "FeatureExtractionError",
    "FeatureResult",
    "FeatureTables",
    "ProcessRun",
    "WaferCohort",
    "build_feature_tables",
    "build_wafer_cohort",
    "canonical_experiment_key",
    "extract_run_features",
    "load_mean_si_etch_targets",
    "load_process_decoder",
    "load_process_runs",
    "predictor_feature_names",
    "segment_process_run",
    "select_longest_contiguous_block",
]

__version__ = "0.1.0"
