"""Read-only ingestion and cohort construction for the BOSCH etch data."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .constants import (
    COMMON_PROCESS_CHANNELS,
    EXPECTED_MATCHED_RUNS,
    EXPECTED_PROCESS_RUNS,
    EXPECTED_SPATIAL_SITES,
    LOT_BY_DATE,
    METROLOGY_89_FILENAME,
    PROCESS_DATA_FILENAME,
    PROCESS_DICTIONARY_FILENAME,
)

_PROCESS_GROUP_RE = re.compile(
    r"^Day_(?P<year>\d{4})_(?P<month>\d{2})_(?P<day>\d{2})_Wafer_(?P<wafer>\d{2})$"
)


@dataclass(frozen=True)
class ProcessRun:
    """One wafer's decoded process telemetry and pre-known context.

    ``values`` is ordered exactly as ``channels``.  The public signal units are
    not supplied, so this object intentionally carries no inferred units.
    """

    experiment_key: str
    group_name: str
    times: NDArray[np.float64]
    values: NDArray[np.float32]
    channels: tuple[str, ...]
    lot_number: int
    wafer_order: int
    conditioning_count: int
    conditioning_surface: str

    def channel_index(self, channel: str) -> int:
        """Return the column index for a raw channel name."""

        try:
            return self.channels.index(channel)
        except ValueError as exc:
            raise KeyError(f"Channel {channel!r} is absent from {self.experiment_key}") from exc

    def channel(self, channel: str) -> NDArray[np.float32]:
        """Return one decoded signal without copying the full process matrix."""

        return self.values[:, self.channel_index(channel)]


@dataclass(frozen=True)
class WaferCohort:
    """All process runs plus the subset having published 89-site targets."""

    runs: Mapping[str, ProcessRun]
    targets: pd.DataFrame

    @property
    def process_keys(self) -> tuple[str, ...]:
        """Canonical keys for all decoded process runs."""

        return tuple(sorted(self.runs))

    @property
    def matched_keys(self) -> tuple[str, ...]:
        """Keys having both process telemetry and a target."""

        return tuple(sorted(set(self.runs).intersection(self.targets.index)))

    @property
    def process_only_keys(self) -> tuple[str, ...]:
        """Process runs without a published 89-site target map."""

        return tuple(sorted(set(self.runs).difference(self.targets.index)))


def canonical_experiment_key(group_name: str) -> str:
    """Convert a NetCDF group name to ``YYYY-MM-DD_NN``."""

    match = _PROCESS_GROUP_RE.fullmatch(group_name)
    if match is None:
        raise ValueError(f"Unexpected process group name: {group_name!r}")
    parts = match.groupdict()
    return f"{parts['year']}-{parts['month']}-{parts['day']}_{parts['wafer']}"


def _decode_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def load_process_decoder(path: str | Path) -> NDArray[np.float32]:
    """Load and validate the float codebook used by 16-bit process arrays."""

    with h5py.File(Path(path), "r") as handle:
        if "data" not in handle:
            raise ValueError(f"No 'data' decoder dataset in {path}")
        decoder = np.asarray(handle["data"][:], dtype=np.float32)
    if decoder.ndim != 1 or decoder.size == 0:
        raise ValueError("Process decoder must be a non-empty one-dimensional array")
    if not np.isfinite(decoder).all():
        raise ValueError("Process decoder contains NaN or infinite values")
    return decoder


def load_process_runs(
    process_path: str | Path,
    dictionary_path: str | Path,
    *,
    common_only: bool = True,
) -> dict[str, ProcessRun]:
    """Decode all wafer groups from ``Process_data.nc``.

    Parameters
    ----------
    process_path:
        Path to the immutable process NetCDF file.
    dictionary_path:
        Path to ``Dictionary_process.nc``.
    common_only:
        Select and order the audited 31-channel common schema.  This is the
        safe default because the first date has 44 channels and later dates 31.
    """

    decoder = load_process_decoder(dictionary_path)
    runs: dict[str, ProcessRun] = {}

    with h5py.File(Path(process_path), "r") as handle:
        for group_name in sorted(handle.keys()):
            group = handle[group_name]
            required = {"data", "feature", "times"}
            missing_objects = required.difference(group.keys())
            if missing_objects:
                raise ValueError(f"{group_name} lacks datasets {sorted(missing_objects)}")

            feature_names = tuple(_decode_text(value) for value in group["feature"][:])
            encoded = np.asarray(group["data"][:])
            times = np.asarray(group["times"][:], dtype=np.float64)

            if encoded.ndim != 2 or encoded.shape[0] != times.size:
                raise ValueError(f"Inconsistent data/time shape for {group_name}")
            if encoded.shape[1] != len(feature_names):
                raise ValueError(f"Inconsistent data/feature shape for {group_name}")
            if encoded.size and int(encoded.max()) >= decoder.size:
                raise ValueError(f"Encoded index exceeds decoder length for {group_name}")

            decoded = decoder[encoded]
            if common_only:
                missing_channels = set(COMMON_PROCESS_CHANNELS).difference(feature_names)
                if missing_channels:
                    raise ValueError(
                        f"{group_name} lacks common channels {sorted(missing_channels)}"
                    )
                indices = [feature_names.index(name) for name in COMMON_PROCESS_CHANNELS]
                decoded = decoded[:, indices]
                selected_channels = COMMON_PROCESS_CHANNELS
            else:
                selected_channels = feature_names

            key = canonical_experiment_key(group_name)
            date = key[:10]
            wafer_order = int(key[-2:])
            if date not in LOT_BY_DATE:
                raise ValueError(f"No lot context registered for experiment date {date}")
            lot = LOT_BY_DATE[date]

            runs[key] = ProcessRun(
                experiment_key=key,
                group_name=group_name,
                times=times,
                values=np.asarray(decoded, dtype=np.float32),
                channels=tuple(selected_channels),
                lot_number=lot.lot_number,
                wafer_order=wafer_order,
                conditioning_count=lot.conditioning_count,
                conditioning_surface=lot.conditioning_surface,
            )

    return runs


def load_mean_si_etch_targets(path: str | Path) -> pd.DataFrame:
    """Build one arithmetic-mean Si-etch target per published 89-site map.

    The result is indexed by ``experiment_key`` and includes audit columns.
    This is an arithmetic mean of the 89 published values, not an area-weighted
    wafer mean.
    """

    frame = pd.read_csv(Path(path))
    required = {
        "experiment_key",
        "lot_number",
        "wafer_number",
        "X",
        "Y",
        "si_etch",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"89-point metrology CSV lacks columns {sorted(missing)}")
    if frame[list(required)].isna().any().any():
        raise ValueError("Required 89-point target fields contain missing values")

    rows: list[dict[str, float | int | str]] = []
    for key, wafer in frame.groupby("experiment_key", sort=True):
        site_count = len(wafer)
        unique_site_count = len(wafer[["X", "Y"]].drop_duplicates())
        if site_count != EXPECTED_SPATIAL_SITES or unique_site_count != EXPECTED_SPATIAL_SITES:
            raise ValueError(
                f"{key} has {site_count} rows and {unique_site_count} unique sites; "
                f"expected {EXPECTED_SPATIAL_SITES}"
            )
        lot_values = wafer["lot_number"].unique()
        wafer_values = wafer["wafer_number"].unique()
        if len(lot_values) != 1 or len(wafer_values) != 1:
            raise ValueError(f"Inconsistent lot/wafer identifiers for {key}")
        rows.append(
            {
                "experiment_key": str(key),
                "lot_number": int(lot_values[0]),
                "wafer_order": int(wafer_values[0]),
                "n_sites": site_count,
                "mean_si_etch_um": float(wafer["si_etch"].mean()),
                "si_etch_std_um": float(wafer["si_etch"].std(ddof=1)),
                "si_etch_min_um": float(wafer["si_etch"].min()),
                "si_etch_max_um": float(wafer["si_etch"].max()),
            }
        )

        row = rows[-1]
        optional_direct_columns = {"stepheight", "postox_thickness_nan"}
        if optional_direct_columns.issubset(wafer.columns):
            direct_mask = wafer[list(optional_direct_columns)].notna().all(axis=1)
            direct_values = (
                wafer.loc[direct_mask, "stepheight"]
                - wafer.loc[direct_mask, "postox_thickness_nan"]
            )
            row["n_direct_postox_sites"] = int(direct_mask.sum())
            row["n_interpolated_postox_sites"] = int((~direct_mask).sum())
            row["mean_si_etch_direct_postox_only_um"] = (
                float(direct_values.mean()) if not direct_values.empty else float("nan")
            )
        else:
            # Keep a stable schema for small synthetic fixtures that only carry
            # the published target column.
            row["n_direct_postox_sites"] = 0
            row["n_interpolated_postox_sites"] = 0
            row["mean_si_etch_direct_postox_only_um"] = float("nan")

    return pd.DataFrame(rows).set_index("experiment_key").sort_index()


def build_wafer_cohort(
    raw_dir: str | Path,
    *,
    validate_expected_counts: bool = True,
) -> WaferCohort:
    """Construct the 96-run process cohort and 88-run matched target subset."""

    root = Path(raw_dir)
    runs = load_process_runs(
        root / PROCESS_DATA_FILENAME,
        root / PROCESS_DICTIONARY_FILENAME,
        common_only=True,
    )
    targets = load_mean_si_etch_targets(root / METROLOGY_89_FILENAME)
    cohort = WaferCohort(runs=runs, targets=targets)

    target_only = set(targets.index).difference(runs)
    if target_only:
        raise ValueError(f"Metrology keys without process data: {sorted(target_only)}")
    for key in cohort.matched_keys:
        run = runs[key]
        row = targets.loc[key]
        if run.lot_number != int(row["lot_number"]):
            raise ValueError(f"Lot mismatch for {key}")
        if run.wafer_order != int(row["wafer_order"]):
            raise ValueError(f"Wafer-order mismatch for {key}")

    if validate_expected_counts:
        if len(cohort.runs) != EXPECTED_PROCESS_RUNS:
            raise ValueError(f"Expected {EXPECTED_PROCESS_RUNS} process runs, got {len(runs)}")
        if len(cohort.matched_keys) != EXPECTED_MATCHED_RUNS:
            raise ValueError(
                f"Expected {EXPECTED_MATCHED_RUNS} matched runs, got {len(cohort.matched_keys)}"
            )
    return cohort
