"""Tests for decoded cohort construction and target-free cycle features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bosch_vm.constants import (
    ACTIVE_WINDOW_PROXY,
    COMMON_PROCESS_CHANNELS,
    EXPECTED_PREDICTOR_COUNT,
    LONG_PHASE_PROXY,
    SHORT_PHASE_PROXY,
)
from bosch_vm.features import (
    build_feature_tables,
    extract_run_features,
    segment_process_run,
    select_longest_contiguous_block,
)
from bosch_vm.io import ProcessRun, build_wafer_cohort, load_mean_si_etch_targets


def _synthetic_run(*, isolated_tail: bool = True, extra_long_pulse: bool = True) -> ProcessRun:
    dt = 0.2
    main_times = np.arange(0.0, 635.0, dt, dtype=np.float64)
    values = np.zeros((main_times.size, len(COMMON_PROCESS_CHANNELS)), dtype=np.float32)

    # Give every non-proxy channel a deterministic level and mild cycle-scale drift.
    for column in range(values.shape[1]):
        values[:, column] = 1.0 + column + 0.001 * main_times

    short = np.zeros(main_times.size, dtype=np.float32)
    long = np.zeros(main_times.size, dtype=np.float32)
    starts = [10.0] + [30.0 + 6.0 * index for index in range(99)]
    for index, start in enumerate(starts):
        short_duration = 11.8 if index == 0 else 1.4
        short[(main_times >= start) & (main_times < start + short_duration)] = 300.0
        long_start = start + short_duration + 0.4
        next_start = starts[index + 1] if index + 1 < len(starts) else 629.0
        # The audited first cycle contains a short ignition pulse and may have
        # a separate retry pulse before cycle 2; stable cycles fill the normal
        # long-phase part of their six-second windows.
        long_stop = long_start + 1.4 if index == 0 else next_start - 0.2
        long[(main_times >= long_start) & (main_times < long_stop)] = 600.0

    if extra_long_pulse:
        # A startup/retry pulse in cycle 1, after the paired long phase.
        long[(main_times >= 27.0) & (main_times < 29.0)] = 600.0

    values[:, COMMON_PROCESS_CHANNELS.index(SHORT_PHASE_PROXY)] = short
    values[:, COMMON_PROCESS_CHANNELS.index(LONG_PHASE_PROXY)] = long
    source = np.zeros(main_times.size, dtype=np.float32)
    source[(main_times >= 18.0) & (main_times < 629.0)] = 2800.0
    values[:, COMMON_PROCESS_CHANNELS.index(ACTIVE_WINDOW_PROXY)] = source

    times = main_times
    if isolated_tail:
        times = np.r_[times, times[-1] + 42.0]
        values = np.vstack([values, values[-1]])

    return ProcessRun(
        experiment_key="2024-07-05_01",
        group_name="Day_2024_07_05_Wafer_01",
        times=times,
        values=values,
        channels=COMMON_PROCESS_CHANNELS,
        lot_number=2,
        wafer_order=1,
        conditioning_count=1,
        conditioning_surface="chuck",
    )


def test_longest_contiguous_block_removes_only_isolated_tail() -> None:
    run = _synthetic_run(isolated_tail=True)
    block = select_longest_contiguous_block(run.times)
    assert block.start == 0
    assert block.stop == run.times.size - 1
    assert block.sample_interval_seconds == pytest.approx(0.2)
    assert block.largest_gap_seconds == pytest.approx(42.0)


def test_segmentation_finds_100_anchors_and_keeps_retry_as_qa() -> None:
    segmentation = segment_process_run(_synthetic_run(extra_long_pulse=True))
    assert len(segmentation.cycles) == 100
    assert segmentation.qa["segmentation_pass"] is True
    assert segmentation.qa["tail_rows_removed"] == 1
    assert sum(len(cycle.long_components) for cycle in segmentation.cycles) == 101


def test_feature_schema_is_81_predictors_and_excludes_qa() -> None:
    result = extract_run_features(_synthetic_run())
    assert len(result.predictors) == EXPECTED_PREDICTOR_COUNT == 81
    assert all(np.isfinite(list(result.predictors.values())))
    assert "tail_rows_removed" in result.qa
    assert "tail_rows_removed" not in result.predictors
    assert result.predictors["timing__extra_long_components"] == 1.0
    assert result.predictors["context__conditioning_surface_chuck"] == 1.0


def test_target_is_arithmetic_mean_of_exactly_89_sites(tmp_path: Path) -> None:
    rows: list[dict[str, float | int | str]] = []
    for wafer_order in (1, 2):
        for site in range(89):
            rows.append(
                {
                    "experiment_key": f"2024-07-05_{wafer_order:02d}",
                    "lot_number": 2,
                    "wafer_number": wafer_order,
                    "X": site,
                    "Y": 0,
                    "si_etch": wafer_order + site / 100.0,
                }
            )
    path = tmp_path / "metrology.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    targets = load_mean_si_etch_targets(path)
    assert targets.shape[0] == 2
    assert targets.loc["2024-07-05_01", "n_sites"] == 89
    assert targets.loc["2024-07-05_01", "n_direct_postox_sites"] == 0
    assert np.isnan(
        targets.loc["2024-07-05_01", "mean_si_etch_direct_postox_only_um"]
    )
    assert targets.loc["2024-07-05_01", "mean_si_etch_um"] == pytest.approx(
        np.mean([1 + site / 100.0 for site in range(89)])
    )


def test_real_cohort_and_feature_counts_when_raw_data_are_present() -> None:
    raw_dir = Path(__file__).parents[1] / "data" / "raw" / "zenodo_17122442"
    if not (raw_dir / "Process_data.nc").exists():
        pytest.skip("Downloaded Zenodo raw data are not present")

    cohort = build_wafer_cohort(raw_dir)
    assert len(cohort.runs) == 96
    assert len(cohort.matched_keys) == 88
    assert len(cohort.process_only_keys) == 8
    assert cohort.targets["n_interpolated_postox_sites"].sum() == 157
    assert (
        cohort.targets["n_direct_postox_sites"]
        + cohort.targets["n_interpolated_postox_sites"]
        == 89
    ).all()

    tables = build_feature_tables(cohort)
    assert tables.all_predictors.shape == (96, 81)
    assert tables.matched_predictors.shape == (88, 81)
    assert tables.matched_targets.shape == (88,)
    assert tables.all_qa["segmentation_pass"].all()
    assert not tables.all_predictors.isna().any().any()
