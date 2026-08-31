"""Target-free, cycle-aware feature engineering for process telemetry."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .constants import (
    ACTIVE_WINDOW_PROXY,
    CONDITIONING_SURFACES,
    CYCLE_SUMMARY_CHANNELS,
    EXPECTED_CYCLE_COUNT,
    EXPECTED_PREDICTOR_COUNT,
    LONG_PHASE_PROXY,
    SHORT_PHASE_PROXY,
    SLOW_STATE_CHANNELS,
    STABLE_CYCLE_START,
    STABLE_CYCLE_STOP,
)
from .io import ProcessRun, WaferCohort


class FeatureExtractionError(ValueError):
    """Raised when a run does not satisfy audited segmentation invariants."""


@dataclass(frozen=True)
class Component:
    """Half-open sample-index interval for one contiguous high state."""

    start: int
    stop: int


@dataclass(frozen=True)
class CycleSegment:
    """One Gas4-anchored process unit and its phase-proxy components."""

    index: int
    start: int
    stop: int
    short_component: Component
    long_components: tuple[Component, ...]


@dataclass(frozen=True)
class SegmentationResult:
    """Trimmed signal block, 100 cycle segments, and target-free QA."""

    run: ProcessRun
    times: NDArray[np.float64]
    values: NDArray[np.float32]
    block_start: int
    block_stop: int
    sample_interval_seconds: float
    cycles: tuple[CycleSegment, ...]
    source_active_component: Component
    qa: Mapping[str, float | int | bool]


@dataclass(frozen=True)
class FeatureResult:
    """Predictor values and non-predictive QA for one wafer run."""

    experiment_key: str
    predictors: Mapping[str, float]
    qa: Mapping[str, float | int | bool]


@dataclass(frozen=True)
class FeatureTables:
    """Feature tables for all 96 runs and the 88 matched runs."""

    all_predictors: pd.DataFrame
    all_qa: pd.DataFrame
    matched_predictors: pd.DataFrame
    matched_targets: pd.Series
    metadata: pd.DataFrame


@dataclass(frozen=True)
class BlockSelection:
    """The longest approximately regular timestamp block."""

    start: int
    stop: int
    sample_interval_seconds: float
    split_threshold_seconds: float
    largest_gap_seconds: float


def _alias(channel: str) -> str:
    return channel.removeprefix("Stat3_Etch_MV_")


def predictor_feature_names() -> tuple[str, ...]:
    """Return the deterministic 81-column predictor schema."""

    names: list[str] = []
    for channel in CYCLE_SUMMARY_CHANNELS:
        alias = _alias(channel)
        names.extend(
            (
                f"cycle__{alias}__level_median",
                f"cycle__{alias}__phase_contrast_median",
                f"cycle__{alias}__phase_contrast_mad",
                f"cycle__{alias}__late_minus_early",
            )
        )
    names.extend(
        (
            "timing__short_duration_median_s",
            "timing__short_duration_iqr_s",
            "timing__long_duration_median_s",
            "timing__long_duration_iqr_s",
            "timing__cycle_period_median_s",
            "timing__cycle_period_iqr_s",
            "timing__extra_long_components",
            "timing__abnormal_cycle_intervals",
            "timing__source_active_span_s",
            "timing__startup_interval_s",
        )
    )
    for channel in SLOW_STATE_CHANNELS:
        alias = _alias(channel)
        names.extend(
            (
                f"slow__{alias}__active_median",
                f"slow__{alias}__late_minus_early",
            )
        )
    names.extend(
        (
            "context__conditioning_count",
            "context__conditioning_surface_chuck",
            "context__conditioning_surface_si",
            "context__conditioning_surface_sio2",
            "context__wafer_order",
        )
    )
    if len(names) != EXPECTED_PREDICTOR_COUNT:
        raise AssertionError(
            f"Predictor schema has {len(names)} columns, expected {EXPECTED_PREDICTOR_COUNT}"
        )
    return tuple(names)


def select_longest_contiguous_block(
    times: NDArray[np.float64],
    *,
    gap_floor_seconds: float = 1.0,
    gap_multiple: float = 5.0,
) -> BlockSelection:
    """Select the longest block not interrupted by an abnormal timestamp gap.

    In the audited data this removes one isolated final row from 69 runs and
    leaves 27 runs unchanged.  The general longest-block rule also handles a
    future prefix or internal fragment without mutating raw data.
    """

    values = np.asarray(times, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise FeatureExtractionError("At least two one-dimensional timestamps are required")
    differences = np.diff(values)
    if np.any(differences <= 0):
        raise FeatureExtractionError("Process timestamps must be strictly increasing")
    sample_interval = float(np.median(differences))
    threshold = max(float(gap_floor_seconds), float(gap_multiple) * sample_interval)
    cuts = np.flatnonzero(differences > threshold)

    starts = np.r_[0, cuts + 1]
    stops = np.r_[cuts + 1, values.size]
    lengths = stops - starts
    winner = int(np.argmax(lengths))
    return BlockSelection(
        start=int(starts[winner]),
        stop=int(stops[winner]),
        sample_interval_seconds=sample_interval,
        split_threshold_seconds=threshold,
        largest_gap_seconds=float(differences.max()),
    )


def _component_duration(
    times: NDArray[np.float64], component: Component, sample_interval: float
) -> float:
    return float(times[component.stop - 1] - times[component.start] + sample_interval)


def _connected_components(
    mask: NDArray[np.bool_],
    times: NDArray[np.float64],
    sample_interval: float,
    *,
    min_duration_seconds: float = 0.6,
) -> tuple[Component, ...]:
    padded = np.r_[False, np.asarray(mask, dtype=bool), False]
    starts = np.flatnonzero(padded[1:] & ~padded[:-1])
    stops = np.flatnonzero(~padded[1:] & padded[:-1])
    components = tuple(
        Component(int(start), int(stop)) for start, stop in zip(starts, stops, strict=True)
    )
    return tuple(
        component
        for component in components
        if _component_duration(times, component, sample_interval) >= min_duration_seconds
    )


def _midpoint_threshold(signal: NDArray[np.float32]) -> float:
    low, high = np.quantile(np.asarray(signal, dtype=np.float64), (0.05, 0.95))
    if not high > low:
        raise FeatureExtractionError("Phase-proxy signal does not have two distinguishable levels")
    return float((low + high) / 2.0)


def _active_window_threshold(signal: NDArray[np.float32]) -> float:
    """Threshold the high-duty-cycle source-power signal.

    The main source RF is on for roughly the whole BOSCH window, so its 5th
    percentile can equal its 95th percentile for short or already-cropped
    traces.  The audited rule is therefore one half of the positive 95th
    percentile rather than the Gas4/Gas5 two-level midpoint.
    """

    high = float(np.quantile(np.asarray(signal, dtype=np.float64), 0.95))
    if high <= 0:
        raise FeatureExtractionError("Active-window proxy has no positive high state")
    return 0.5 * high


def segment_process_run(run: ProcessRun) -> SegmentationResult:
    """Trim and segment one run using the audited Gas4/Gas5 signal structure."""

    block = select_longest_contiguous_block(run.times)
    times = np.asarray(run.times[block.start : block.stop], dtype=np.float64)
    times = times - times[0]
    values = np.asarray(run.values[block.start : block.stop], dtype=np.float32)
    dt = block.sample_interval_seconds

    short_signal = values[:, run.channel_index(SHORT_PHASE_PROXY)]
    long_signal = values[:, run.channel_index(LONG_PHASE_PROXY)]
    source_signal = values[:, run.channel_index(ACTIVE_WINDOW_PROXY)]

    short_threshold = _midpoint_threshold(short_signal)
    long_threshold = _midpoint_threshold(long_signal)
    source_threshold = _active_window_threshold(source_signal)

    short_components = _connected_components(
        short_signal > short_threshold, times, dt
    )
    long_components = _connected_components(long_signal > long_threshold, times, dt)
    source_components = _connected_components(source_signal > source_threshold, times, dt)
    source_active = (
        max(source_components, key=lambda item: item.stop - item.start)
        if source_components
        else None
    )

    cycles: list[CycleSegment] = []
    empty_long_cycles = 0
    if len(short_components) == EXPECTED_CYCLE_COUNT:
        for index, short_component in enumerate(short_components):
            stop = (
                short_components[index + 1].start
                if index + 1 < len(short_components)
                else values.shape[0]
            )
            within_cycle = tuple(
                component
                for component in long_components
                if component.start >= short_component.stop and component.start < stop
            )
            if not within_cycle:
                empty_long_cycles += 1
            cycles.append(
                CycleSegment(
                    index=index,
                    start=short_component.start,
                    stop=stop,
                    short_component=short_component,
                    long_components=within_cycle,
                )
            )

    segmentation_pass = (
        len(short_components) == EXPECTED_CYCLE_COUNT
        and empty_long_cycles == 0
        and source_active is not None
    )
    qa: dict[str, float | int | bool] = {
        "raw_rows": int(run.times.size),
        "used_rows": int(times.size),
        "head_rows_removed": int(block.start),
        "tail_rows_removed": int(run.times.size - block.stop),
        "sample_interval_seconds": dt,
        "largest_gap_seconds": block.largest_gap_seconds,
        "short_threshold": short_threshold,
        "long_threshold": long_threshold,
        "source_threshold": source_threshold,
        "n_short_components": len(short_components),
        "n_long_components": len(long_components),
        "n_empty_long_cycles": empty_long_cycles,
        "segmentation_pass": segmentation_pass,
    }
    if not segmentation_pass or source_active is None:
        raise FeatureExtractionError(
            f"Segmentation failed for {run.experiment_key}: "
            f"short={len(short_components)}, empty_long={empty_long_cycles}, "
            f"source_components={len(source_components)}"
        )

    return SegmentationResult(
        run=run,
        times=times,
        values=values,
        block_start=block.start,
        block_stop=block.stop,
        sample_interval_seconds=dt,
        cycles=tuple(cycles),
        source_active_component=source_active,
        qa=qa,
    )


def _component_vector_mean(
    segmentation: SegmentationResult, component: Component
) -> tuple[NDArray[np.float64], float]:
    times = segmentation.times[component.start : component.stop]
    values = np.asarray(
        segmentation.values[component.start : component.stop], dtype=np.float64
    )
    if times.size == 1:
        return values[0], segmentation.sample_interval_seconds
    weights = np.r_[np.diff(times), segmentation.sample_interval_seconds]
    return np.average(values, axis=0, weights=weights), float(weights.sum())


def _union_vector_mean(
    segmentation: SegmentationResult, components: tuple[Component, ...]
) -> NDArray[np.float64]:
    weighted_sum = np.zeros(segmentation.values.shape[1], dtype=np.float64)
    total_weight = 0.0
    for component in components:
        component_mean, component_weight = _component_vector_mean(segmentation, component)
        weighted_sum += component_mean * component_weight
        total_weight += component_weight
    if total_weight <= 0:
        raise FeatureExtractionError("A cycle phase has no positive-duration samples")
    return weighted_sum / total_weight


def _iqr(values: NDArray[np.float64]) -> float:
    lower, upper = np.quantile(values, (0.25, 0.75))
    return float(upper - lower)


def _mad(values: NDArray[np.float64]) -> float:
    median = np.median(values)
    return float(np.median(np.abs(values - median)))


def extract_run_features(run: ProcessRun) -> FeatureResult:
    """Extract the fixed 81 predictors and separate QA for one wafer run."""

    segmentation = segment_process_run(run)
    stable_cycles = segmentation.cycles[STABLE_CYCLE_START:STABLE_CYCLE_STOP]
    if len(stable_cycles) != STABLE_CYCLE_STOP - STABLE_CYCLE_START:
        raise FeatureExtractionError("Stable cycle window is incomplete")

    short_matrix = np.vstack(
        [
            _component_vector_mean(segmentation, cycle.short_component)[0]
            for cycle in stable_cycles
        ]
    )
    long_matrix = np.vstack(
        [_union_vector_mean(segmentation, cycle.long_components) for cycle in stable_cycles]
    )
    overall_matrix = (short_matrix + long_matrix) / 2.0
    contrast_matrix = long_matrix - short_matrix

    predictors: dict[str, float] = {}
    for channel in CYCLE_SUMMARY_CHANNELS:
        column = run.channel_index(channel)
        alias = _alias(channel)
        overall = overall_matrix[:, column]
        contrast = contrast_matrix[:, column]
        predictors[f"cycle__{alias}__level_median"] = float(np.median(overall))
        predictors[f"cycle__{alias}__phase_contrast_median"] = float(
            np.median(contrast)
        )
        predictors[f"cycle__{alias}__phase_contrast_mad"] = _mad(contrast)
        predictors[f"cycle__{alias}__late_minus_early"] = float(
            np.mean(overall[-10:]) - np.mean(overall[:10])
        )

    short_durations = np.asarray(
        [
            _component_duration(
                segmentation.times,
                cycle.short_component,
                segmentation.sample_interval_seconds,
            )
            for cycle in stable_cycles
        ],
        dtype=np.float64,
    )
    long_durations = np.asarray(
        [
            sum(
                _component_duration(
                    segmentation.times, component, segmentation.sample_interval_seconds
                )
                for component in cycle.long_components
            )
            for cycle in stable_cycles
        ],
        dtype=np.float64,
    )
    stable_starts = np.asarray(
        [segmentation.times[cycle.short_component.start] for cycle in stable_cycles],
        dtype=np.float64,
    )
    periods = np.diff(stable_starts)
    all_long_component_count = sum(len(cycle.long_components) for cycle in segmentation.cycles)
    extra_long_components = all_long_component_count - len(segmentation.cycles)
    abnormal_periods = int(np.count_nonzero(np.abs(periods - 6.0) > 0.3))
    active_span = _component_duration(
        segmentation.times,
        segmentation.source_active_component,
        segmentation.sample_interval_seconds,
    )
    startup_interval = float(
        segmentation.times[segmentation.cycles[1].short_component.start]
        - segmentation.times[segmentation.cycles[0].short_component.start]
    )
    predictors.update(
        {
            "timing__short_duration_median_s": float(np.median(short_durations)),
            "timing__short_duration_iqr_s": _iqr(short_durations),
            "timing__long_duration_median_s": float(np.median(long_durations)),
            "timing__long_duration_iqr_s": _iqr(long_durations),
            "timing__cycle_period_median_s": float(np.median(periods)),
            "timing__cycle_period_iqr_s": _iqr(periods),
            "timing__extra_long_components": float(extra_long_components),
            "timing__abnormal_cycle_intervals": float(abnormal_periods),
            "timing__source_active_span_s": active_span,
            "timing__startup_interval_s": startup_interval,
        }
    )

    for channel in SLOW_STATE_CHANNELS:
        column = run.channel_index(channel)
        alias = _alias(channel)
        overall = overall_matrix[:, column]
        predictors[f"slow__{alias}__active_median"] = float(np.median(overall))
        predictors[f"slow__{alias}__late_minus_early"] = float(
            np.mean(overall[-10:]) - np.mean(overall[:10])
        )

    predictors["context__conditioning_count"] = float(run.conditioning_count)
    for surface in CONDITIONING_SURFACES:
        predictors[f"context__conditioning_surface_{surface}"] = float(
            run.conditioning_surface == surface
        )
    predictors["context__wafer_order"] = float(run.wafer_order)

    expected_names = predictor_feature_names()
    if tuple(predictors) != expected_names:
        raise AssertionError("Feature insertion order differs from the declared schema")
    values = np.fromiter(predictors.values(), dtype=np.float64)
    if not np.isfinite(values).all():
        raise FeatureExtractionError(f"Non-finite predictor generated for {run.experiment_key}")

    qa = dict(segmentation.qa)
    qa.update(
        {
            "extra_long_components": int(extra_long_components),
            "abnormal_cycle_intervals": abnormal_periods,
            "source_active_span_seconds": active_span,
        }
    )
    return FeatureResult(run.experiment_key, predictors, qa)


def build_feature_tables(cohort: WaferCohort) -> FeatureTables:
    """Extract all-run features and align the 88-run supervised subset."""

    results = [extract_run_features(cohort.runs[key]) for key in cohort.process_keys]
    all_predictors = pd.DataFrame(
        [result.predictors for result in results],
        index=[result.experiment_key for result in results],
        columns=predictor_feature_names(),
        dtype=float,
    )
    all_predictors.index.name = "experiment_key"
    all_qa = pd.DataFrame(
        [result.qa for result in results],
        index=[result.experiment_key for result in results],
    )
    all_qa.index.name = "experiment_key"

    metadata = pd.DataFrame(
        [
            {
                "experiment_key": key,
                "lot_number": run.lot_number,
                "wafer_order": run.wafer_order,
                "conditioning_count": run.conditioning_count,
                "conditioning_surface": run.conditioning_surface,
                "has_target": key in cohort.targets.index,
            }
            for key, run in sorted(cohort.runs.items())
        ]
    ).set_index("experiment_key")

    matched_keys = list(cohort.matched_keys)
    matched_predictors = all_predictors.loc[matched_keys].copy()
    matched_targets = cohort.targets.loc[matched_keys, "mean_si_etch_um"].copy()
    matched_targets.name = "mean_si_etch_um"
    return FeatureTables(
        all_predictors=all_predictors,
        all_qa=all_qa,
        matched_predictors=matched_predictors,
        matched_targets=matched_targets,
        metadata=metadata,
    )
