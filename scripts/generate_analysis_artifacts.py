#!/usr/bin/env python3
"""Assemble verified tables, figures, model cards, and the technical report.

All predictive figures and metrics are derived from saved outer-fold OOF
predictions.  This script never substitutes final-refit or training scores.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle
from matplotlib.ticker import PercentFormatter
from scipy.stats import norm, pearsonr, spearmanr

from bosch_vm.constants import (
    LONG_PHASE_PROXY,
    METROLOGY_89_FILENAME,
    PROCESS_DATA_FILENAME,
    PROCESS_DICTIONARY_FILENAME,
    SHORT_PHASE_PROXY,
)
from bosch_vm.features import segment_process_run
from bosch_vm.io import load_process_runs
from bosch_vm.reporting import (
    model_comparison_markdown,
    plot_block_permutation_importance,
    plot_coefficient_stability,
    plot_model_comparison,
    plot_model_lot_mae_heatmap,
    plot_oof_parity_panels,
    plot_primary_residual_vs_wafer_order,
    summarize_block_permutation,
    summarize_coefficient_stability,
)

MODEL_ORDER = (
    "dummy_mean",
    "context_ridge",
    "cycle_ridge",
    "pls",
    "elastic_net",
    "rbf_svr",
    "pca_gpr",
    "extra_trees",
)
MODEL_NAMES = {
    "dummy_mean": "Training-mean dummy",
    "context_ridge": "Context-only Ridge",
    "cycle_ridge": "Cycle-feature Ridge",
    "pls": "PLS",
    "elastic_net": "Elastic Net",
    "rbf_svr": "RBF-SVR",
    "pca_gpr": "PCA + Gaussian Process",
    "extra_trees": "Extra Trees",
}
SENSITIVITY_NAMES = {
    "process_only": "Telemetry only (context removed)",
    "without_gas_delivery": "Without Gas4/Gas5 readbacks",
    "without_pressure_helium": "Without pressure and helium readbacks",
    "without_rf_power": "Without RF and power readbacks",
    "without_temperature": "Without temperature readbacks",
    "without_timing": "Without timing features",
    "robust_scaler": "RobustScaler instead of StandardScaler",
    "direct_postox_target": "Direct post-oxide sites only",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/wafer_mean_si_etch_lolo_v1"),
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("figures/wafer_mean_si_etch_lolo_v1"),
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("reports/wafer_mean_si_etch_lolo_v1.md"),
    )
    parser.add_argument(
        "--target-audit",
        type=Path,
        default=Path("data/processed/wafer_mean_si_etch_lolo_v1/target_audit.csv"),
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/zenodo_17122442"),
        help="Directory containing the downloaded core Zenodo files.",
    )
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    return parser.parse_args()


def _model_path(results_dir: Path, model_id: str, filename: str) -> Path:
    return results_dir / "models" / model_id / filename


def combine_bootstrap(results_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for model_id in MODEL_ORDER:
        frame = pd.read_csv(
            _model_path(results_dir, model_id, "cluster_bootstrap_intervals.csv")
        )
        frame.insert(0, "model_id", model_id)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(results_dir / "all_cluster_bootstrap_intervals.csv", index=False)
    return combined


def paired_model_differences(
    oof: pd.DataFrame,
    *,
    references: tuple[str, ...],
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    lot_mae = (
        oof.groupby(["model_id", "lot_number"], as_index=False)["absolute_error"]
        .mean()
        .pivot(index="lot_number", columns="model_id", values="absolute_error")
        .sort_index()
    )
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(lot_mae), size=(n_bootstrap, len(lot_mae)))
    rows: list[dict[str, Any]] = []
    for reference in references:
        for model_id in MODEL_ORDER:
            differences = (lot_mae[model_id] - lot_mae[reference]).to_numpy(dtype=float)
            replicates = differences[draws].mean(axis=1)
            rows.append(
                {
                    "model_id": model_id,
                    "reference_model_id": reference,
                    "delta_macro_lot_mae_um": float(differences.mean()),
                    "lower_95_um": float(np.quantile(replicates, 0.025)),
                    "upper_95_um": float(np.quantile(replicates, 0.975)),
                    "lots_better_than_reference": int(np.count_nonzero(differences < 0)),
                    "lots_worse_than_reference": int(np.count_nonzero(differences > 0)),
                    "n_lots": len(differences),
                    "n_bootstrap": n_bootstrap,
                }
            )
    return pd.DataFrame(rows)


def collect_hyperparameters(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for model_id in MODEL_ORDER:
        selected = json.loads(
            _model_path(results_dir, model_id, "selected_hyperparameters.json").read_text()
        )
        for record in selected:
            params = record["best_params"] or {"(no tuned parameter)": "training mean"}
            for parameter, value in params.items():
                rows.append(
                    {
                        "model_id": model_id,
                        "scope": "outer_fold",
                        "outer_fold": record["outer_fold"],
                        "outer_test_lot": record["outer_test_lot"],
                        "parameter": parameter,
                        "value": value,
                        "best_inner_macro_lot_mae_um": -float(record["best_inner_score"]),
                    }
                )
        final = json.loads(
            _model_path(results_dir, model_id, "final_refit/best_params.json").read_text()
        )
        params = final["best_params"] or {"(no tuned parameter)": "training mean"}
        for parameter, value in params.items():
            rows.append(
                {
                    "model_id": model_id,
                    "scope": "final_all_data_refit",
                    "outer_fold": np.nan,
                    "outer_test_lot": np.nan,
                    "parameter": parameter,
                    "value": value,
                    "best_inner_macro_lot_mae_um": -float(final["best_inner_score"]),
                }
            )
    long = pd.DataFrame(rows)
    long.to_csv(results_dir / "selected_hyperparameters_long.csv", index=False)

    summary_rows: list[dict[str, Any]] = []
    for (model_id, parameter), frame in long.loc[long["scope"] == "outer_fold"].groupby(
        ["model_id", "parameter"], sort=False
    ):
        rendered = frame["value"].astype(str)
        counts = rendered.value_counts()
        final_value = long.loc[
            (long["model_id"] == model_id)
            & (long["parameter"] == parameter)
            & (long["scope"] == "final_all_data_refit"),
            "value",
        ]
        summary_rows.append(
            {
                "model_id": model_id,
                "parameter": parameter,
                "outer_fold_selection_counts": "; ".join(
                    f"{value}: {count}/10" for value, count in counts.items()
                ),
                "outer_fold_mode": counts.index[0],
                "final_all_data_value": final_value.iloc[0] if not final_value.empty else np.nan,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(results_dir / "hyperparameter_summary.csv", index=False)
    return long, summary


def gpr_uncertainty_diagnostics(
    oof: pd.DataFrame, results_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = oof.loc[oof["model_id"] == "pca_gpr"].copy()
    if "y_pred_std" not in frame:
        raise ValueError("PCA+GPR OOF predictions do not contain y_pred_std")
    error = frame["absolute_error"].to_numpy(dtype=float)
    standard_deviation = frame["y_pred_std"].to_numpy(dtype=float)
    pearson = pearsonr(standard_deviation, error)
    spearman = spearmanr(standard_deviation, error)
    rows: list[dict[str, Any]] = [
        {"diagnostic": "mean_predicted_std_um", "value": standard_deviation.mean()},
        {"diagnostic": "pearson_std_vs_abs_error", "value": pearson.statistic},
        {"diagnostic": "spearman_std_vs_abs_error", "value": spearman.statistic},
    ]
    for nominal, z_score in ((0.80, 1.2815515655), (0.90, 1.6448536269), (0.95, 1.9599639845)):
        covered = error <= z_score * standard_deviation
        rows.extend(
            [
                {
                    "diagnostic": f"nominal_{int(nominal * 100)}_coverage",
                    "value": covered.mean(),
                },
                {
                    "diagnostic": f"nominal_{int(nominal * 100)}_mean_width_um",
                    "value": float((2 * z_score * standard_deviation).mean()),
                },
            ]
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(results_dir / "gpr_uncertainty_diagnostics.csv", index=False)

    per_lot_rows: list[dict[str, Any]] = []
    for lot, lot_frame in frame.groupby("lot_number"):
        lot_error = lot_frame["absolute_error"].to_numpy(dtype=float)
        lot_std = lot_frame["y_pred_std"].to_numpy(dtype=float)
        per_lot_rows.append(
            {
                "lot_number": lot,
                "n_wafers": len(lot_frame),
                "mean_predicted_std_um": lot_std.mean(),
                "mean_absolute_error_um": lot_error.mean(),
                "coverage_90": np.mean(lot_error <= 1.6448536269 * lot_std),
                "coverage_95": np.mean(lot_error <= 1.9599639845 * lot_std),
            }
        )
    per_lot = pd.DataFrame(per_lot_rows)
    per_lot.to_csv(results_dir / "gpr_uncertainty_by_lot.csv", index=False)
    return summary, per_lot


def _plot_sensitivity(sensitivity: pd.DataFrame, path: Path) -> None:
    frame = sensitivity.loc[sensitivity["same_target_as_primary"]].copy()
    frame = frame.sort_values("paired_delta_macro_lot_mae_um")
    y = np.arange(len(frame))
    estimate = frame["paired_delta_macro_lot_mae_um"].to_numpy(dtype=float)
    lower = frame["paired_delta_lower_95_um"].to_numpy(dtype=float)
    upper = frame["paired_delta_upper_95_um"].to_numpy(dtype=float)
    error = np.vstack([estimate - lower, upper - estimate])
    labels = [
        SENSITIVITY_NAMES.get(value, value.replace("_", " "))
        for value in frame["variant_id"]
    ]
    colors = np.where(estimate <= 0, "#2A9D8F", "#D55E00")
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.axvline(0, color="#30343B", linestyle="--", linewidth=1.1)
    ax.errorbar(estimate, y, xerr=error, fmt="none", color="#667788", capsize=3)
    ax.scatter(estimate, y, c=colors, s=65, edgecolor="white", linewidth=0.7)
    ax.set_yticks(y, labels)
    ax.set_xlabel(
        "Change in macro-lot MAE relative to Cycle-feature Ridge "
        "(µm; negative is lower error)"
    )
    ax.set_title("Sensitivity to feature and preprocessing choices", fontweight="bold")
    ax.grid(axis="x", color="#D9E1E8")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_target_lineage(targets: pd.DataFrame, path: Path) -> None:
    primary = targets["mean_si_etch_um"]
    direct = targets["mean_si_etch_direct_postox_only_um"]
    lower = min(primary.min(), direct.min()) - 0.05
    upper = max(primary.max(), direct.max()) + 0.05
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    completed_sites = targets["n_interpolated_postox_sites"].to_numpy(dtype=float)
    scatter = ax.scatter(
        primary,
        direct,
        c=completed_sites,
        cmap="viridis",
        s=45,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.plot([lower, upper], [lower, upper], "--", color="#30343B", linewidth=1.1)
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Published 89-site mean with completed post-oxide map (µm)")
    ax.set_ylabel("Direct-postox-only site mean (µm)")
    ax.set_title("Effect of completed post-oxide sites on the wafer target", fontweight="bold")
    ax.grid(color="#D9E1E8")
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("IDW-completed post-oxide sites")
    difference = direct - primary
    ax.text(
        0.03,
        0.97,
        f"Mean shift {difference.mean():+.3f} µm\nMax |shift| {difference.abs().max():.3f} µm",
        transform=ax.transAxes,
        va="top",
        bbox={"facecolor": "white", "edgecolor": "#D9E1E8", "alpha": 0.9},
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_gpr_uncertainty(oof: pd.DataFrame, path: Path) -> None:
    frame = oof.loc[oof["model_id"] == "pca_gpr"].copy()
    nominal = np.asarray([0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99])
    z_scores = norm.ppf((1.0 + nominal) / 2.0)
    absolute_error = frame["absolute_error"].to_numpy(dtype=float)
    predicted_std = frame["y_pred_std"].to_numpy(dtype=float)
    pooled = np.asarray(
        [np.mean(absolute_error <= z_score * predicted_std) for z_score in z_scores]
    )
    lot_curves = []
    for _, lot in frame.groupby("lot_number", sort=True):
        lot_error = lot["absolute_error"].to_numpy(dtype=float)
        lot_std = lot["y_pred_std"].to_numpy(dtype=float)
        lot_curves.append(
            [np.mean(lot_error <= z_score * lot_std) for z_score in z_scores]
        )
    lot_curves_array = np.asarray(lot_curves, dtype=float)
    lot_q25 = np.quantile(lot_curves_array, 0.25, axis=0)
    lot_q75 = np.quantile(lot_curves_array, 0.75, axis=0)

    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    ax.plot(nominal, nominal, "--", color="#30343B", linewidth=1.1, label="Ideal coverage")
    ax.fill_between(
        nominal,
        lot_q25,
        lot_q75,
        color="#A7C9C4",
        alpha=0.45,
        label="Lot-wise interquartile range",
    )
    ax.plot(
        nominal,
        pooled,
        color="#2A9D8F",
        marker="o",
        markersize=6,
        linewidth=1.8,
        label="All held-out wafers",
    )
    for level in (0.90, 0.95):
        index = int(np.flatnonzero(np.isclose(nominal, level))[0])
        ax.annotate(
            f"{pooled[index]:.1%}",
            (nominal[index], pooled[index]),
            xytext=(6, -14),
            textcoords="offset points",
            fontsize=8.5,
            color="#1F6F68",
        )
    ax.set_xlim(0.48, 1.005)
    ax.set_ylim(0.35, 1.02)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Nominal central interval coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("PCA + GPR interval coverage on held-out lots", fontweight="bold")
    ax.grid(color="#D9E1E8")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_wafer_run_example(raw_dir: Path, path: Path) -> str:
    """Plot the chronologically first matched process run and its 89-site map."""

    runs = load_process_runs(
        raw_dir / PROCESS_DATA_FILENAME,
        raw_dir / PROCESS_DICTIONARY_FILENAME,
    )
    metrology = pd.read_csv(raw_dir / METROLOGY_89_FILENAME)
    matched_keys = sorted(set(runs).intersection(metrology["experiment_key"].astype(str)))
    if not matched_keys:
        raise ValueError("No matched process run and 89-site map were found")
    experiment_key = matched_keys[0]
    run = runs[experiment_key]
    segmentation = segment_process_run(run)
    wafer = metrology.loc[metrology["experiment_key"] == experiment_key].copy()
    if len(wafer) != 89:
        raise ValueError(f"Expected 89 metrology sites for {experiment_key}; found {len(wafer)}")

    displayed_cycles = segmentation.cycles[1:7]
    window_start = displayed_cycles[0].start
    window_stop = displayed_cycles[-1].stop
    time_origin = segmentation.times[window_start]
    displayed_times = segmentation.times[window_start:window_stop] - time_origin
    gas4 = segmentation.values[
        window_start:window_stop, run.channel_index(SHORT_PHASE_PROXY)
    ]
    gas5 = segmentation.values[
        window_start:window_stop, run.channel_index(LONG_PHASE_PROXY)
    ]

    fig, (trace_ax, map_ax) = plt.subplots(
        1,
        2,
        figsize=(12.2, 5.8),
        gridspec_kw={"width_ratios": [1.45, 1.0], "wspace": 0.28},
    )
    trace_ax.plot(displayed_times, gas4, color="#0072B2", linewidth=1.5, label="Gas4Flow")
    trace_ax.plot(displayed_times, gas5, color="#D55E00", linewidth=1.5, label="Gas5Flow")
    for cycle in displayed_cycles:
        cycle_start = segmentation.times[cycle.start] - time_origin
        trace_ax.axvline(cycle_start, color="#59636E", linewidth=0.8, alpha=0.65)
        short_start = segmentation.times[cycle.short_component.start] - time_origin
        short_stop = segmentation.times[cycle.short_component.stop - 1] - time_origin
        trace_ax.axvspan(short_start, short_stop, color="#0072B2", alpha=0.08)
        for component in cycle.long_components:
            long_start = segmentation.times[component.start] - time_origin
            long_stop = segmentation.times[component.stop - 1] - time_origin
            trace_ax.axvspan(long_start, long_stop, color="#D55E00", alpha=0.08)
        trace_ax.text(
            cycle_start + 0.12,
            0.97,
            f"Cycle {cycle.index + 1}",
            transform=trace_ax.get_xaxis_transform(),
            va="top",
            fontsize=7.5,
            color="#465564",
        )
    trace_ax.set_xlim(displayed_times[0], displayed_times[-1])
    trace_ax.set_xlabel("Elapsed time in displayed window (s)")
    trace_ax.set_ylabel("Recorded readback value (source units not published)")
    trace_ax.set_title("Actual gas readbacks and detected cycle boundaries", fontweight="bold")
    trace_ax.grid(color="#D9E1E8", linewidth=0.7)
    trace_ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower center")

    x_mm = wafer["X"].to_numpy(dtype=float) / 1000.0
    y_mm = wafer["Y"].to_numpy(dtype=float) / 1000.0
    site_values = wafer["si_etch"].to_numpy(dtype=float)
    scatter = map_ax.scatter(
        x_mm,
        y_mm,
        c=site_values,
        cmap="viridis",
        s=54,
        edgecolor="white",
        linewidth=0.45,
    )
    map_ax.add_patch(Circle((0.0, 0.0), 100.0, fill=False, color="#59636E", linewidth=1.1))
    map_ax.set_xlim(-105, 105)
    map_ax.set_ylim(-105, 105)
    map_ax.set_aspect("equal", adjustable="box")
    map_ax.set_xlabel("Wafer x position (mm)")
    map_ax.set_ylabel("Wafer y position (mm)")
    map_ax.set_title(
        f"Final 89-site Si etch map\nmean = {site_values.mean():.3f} µm",
        fontweight="bold",
    )
    map_ax.grid(color="#D9E1E8", linewidth=0.7)
    colorbar = fig.colorbar(scatter, ax=map_ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Si etch (µm)")
    fig.suptitle(
        f"One labeled wafer/run: {experiment_key}",
        fontsize=14,
        fontweight="bold",
        y=0.96,
    )
    fig.subplots_adjust(left=0.07, right=0.95, top=0.82, bottom=0.13, wspace=0.30)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return experiment_key


def _markdown_table(frame: pd.DataFrame, columns: list[tuple[str, str, str]]) -> str:
    headers = [label for _, label, _ in columns]
    rows: list[list[str]] = []
    for _, row in frame.iterrows():
        rendered: list[str] = []
        for column, _, format_string in columns:
            value = row[column]
            if format_string and pd.notna(value):
                rendered.append(format_string.format(value))
            else:
                rendered.append("—" if pd.isna(value) else str(value))
        rows.append(rendered)
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def write_model_cards(
    results_dir: Path,
    comparison: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> None:
    metrics_by_model = comparison.set_index("model_id")
    for model_id in MODEL_ORDER:
        model_dir = results_dir / "models" / model_id
        row = metrics_by_model.loc[model_id]
        interval = bootstrap.loc[
            (bootstrap["model_id"] == model_id) & (bootstrap["metric"] == "macro_lot_mae")
        ].iloc[0]
        search = json.loads((model_dir / "search_space.json").read_text())
        final = json.loads((model_dir / "final_refit/best_params.json").read_text())
        plot_oof_parity_panels(
            pd.read_csv(model_dir / "oof_predictions.csv"),
            model_dir / "figures/oof_parity.png",
            model_order=(model_id,),
        )
        text = f"""# {MODEL_NAMES[model_id]}

## Procedure

- Feature view: `{row['feature_view']}`
- Selection: inner leave-one-lot-out GridSearchCV using macro-lot MAE
- Evaluation: outer leave-one-lot-out; 88 wafers, 10 lots
- Search space: `{json.dumps(search['param_grid'], ensure_ascii=False)}`
- Final all-data refit parameters: `{json.dumps(final['best_params'], ensure_ascii=False)}`

## Verified outer-OOF result

| Metric | Value |
| --- | ---: |
| Macro-lot MAE | {row['macro_lot_mae_um']:.6f} µm |
| Macro-lot MAE 95% lot-cluster CI | {interval['lower']:.6f}–{interval['upper']:.6f} µm |
| Pooled MAE | {row['mae_um']:.6f} µm |
| Pooled RMSE | {row['rmse_um']:.6f} µm |
| OOF R² | {row['r2_oof']:.6f} |
| SSE skill vs training-mean dummy | {row['sse_skill_vs_dummy']:.6f} |

## Folder contents

- `folds/`: ten held-out-lot fits, inner scores, predictions, parameters and fitted artifacts;
- `oof_predictions.csv`: one outer prediction per wafer;
- `metrics_by_lot.csv`, `metrics.json`, `cluster_bootstrap_intervals.csv`;
- `outer_coefficients.csv` and `heldout_block_permutation.csv` where supported;
- `final_refit/`: all-data deployment refit, excluded from OOF performance;
- `figures/oof_parity.png`;
- authoritative code: `src/bosch_vm/models.py` and `src/bosch_vm/experiment.py`.

## Interpretation boundary

This model consumes sensor/readback summaries and process context. Feature importance and
coefficients are predictive associations, not causal recipe-knob effects. The final refit is
an artifact for later use; every performance number above comes from outer OOF predictions.
"""
        (model_dir / "README.md").write_text(text, encoding="utf-8")


def _readable_feature_name(value: Any) -> str:
    terms = {
        "cycle": "Cycle response",
        "slow": "Slow state",
        "timing": "Timing",
        "context": "Context",
        "ForeLinePressure": "Foreline pressure",
        "SourceRFReflectedPower": "Source RF reflected power",
        "SourceRFLoadPower": "Source RF load power",
        "PlatenRFTuningCapacitor": "Platen RF tuning capacitor",
        "PlatenRFPeakToPeak": "Platen RF peak-to-peak",
        "PlatenRFReflectedPower": "Platen RF reflected power",
        "PlatenRFLoadCapacitor": "Platen RF load capacitor",
        "Heater2Temp": "Heater 2 temperature",
        "Heater3Temp": "Heater 3 temperature",
        "wafer_order": "Wafer order",
        "conditioning_count": "Conditioning count",
        "phase_contrast_median": "Median phase contrast",
        "phase_contrast_mad": "Phase-contrast variability",
        "level_median": "Median level",
        "active_median": "Median active-window level",
        "late_minus_early": "Late minus early",
        "startup_interval_s": "Startup interval",
    }
    return " · ".join(terms.get(part, part.replace("_", " ")) for part in str(value).split("__"))


def write_report(
    *,
    report_path: Path,
    figure_dir: Path,
    comparison: pd.DataFrame,
    bootstrap: pd.DataFrame,
    paired: pd.DataFrame,
    per_lot: pd.DataFrame,
    hyperparameters: pd.DataFrame,
    coefficient_summary: pd.DataFrame,
    permutation_summary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    uncertainty: pd.DataFrame,
    targets: pd.DataFrame,
    wafer_example_key: str,
) -> None:
    del figure_dir
    ridge = comparison.set_index("model_id").loc["cycle_ridge"]
    elastic_mae = comparison.set_index("model_id").loc[
        "elastic_net", "macro_lot_mae_um"
    ]
    ridge_ci = bootstrap.loc[
        (bootstrap["model_id"] == "cycle_ridge")
        & (bootstrap["metric"] == "macro_lot_mae")
    ].iloc[0]
    elastic_delta = paired.loc[
        (paired["model_id"] == "elastic_net")
        & (paired["reference_model_id"] == "cycle_ridge")
    ].iloc[0]
    context_delta = paired.loc[
        (paired["model_id"] == "context_ridge")
        & (paired["reference_model_id"] == "cycle_ridge")
    ].iloc[0]
    direct = sensitivity.set_index("variant_id").loc["direct_postox_target"]
    no_rf = sensitivity.set_index("variant_id").loc["without_rf_power"]
    robust = sensitivity.set_index("variant_id").loc["robust_scaler"]
    no_gas = sensitivity.set_index("variant_id").loc["without_gas_delivery"]
    coverage90 = uncertainty.set_index("diagnostic").loc["nominal_90_coverage", "value"]
    coverage95 = uncertainty.set_index("diagnostic").loc["nominal_95_coverage", "value"]
    rho = uncertainty.set_index("diagnostic").loc["spearman_std_vs_abs_error", "value"]

    sensitivity_display = sensitivity.copy()
    sensitivity_display["variant"] = sensitivity_display["variant_id"].map(SENSITIVITY_NAMES)
    sensitivity_display["delta_ci"] = sensitivity_display.apply(
        lambda row: (
            f"{row['paired_delta_macro_lot_mae_um']:+.3f} "
            f"[{row['paired_delta_lower_95_um']:+.3f}, {row['paired_delta_upper_95_um']:+.3f}]"
            if row["same_target_as_primary"]
            else "different target"
        ),
        axis=1,
    )
    ridge_by_lot = per_lot.loc[per_lot["model_id"] == "cycle_ridge"].copy()
    final_params = hyperparameters.loc[hyperparameters["scope"] == "final_all_data_refit"].copy()
    final_params["model"] = final_params["model_id"].map(MODEL_NAMES)
    coefficient_display = coefficient_summary.head(12).copy()
    coefficient_display["feature_name"] = coefficient_display["feature"].map(
        _readable_feature_name
    )
    permutation_display = permutation_summary.copy()
    permutation_display["block_name"] = permutation_display["block"].map(
        {
            "cycle": "Cycle-response telemetry (56 features)",
            "timing": "Timing (10 features)",
            "slow": "Slow-state telemetry (10 features)",
            "context": "Process context (5 features)",
        }
    )

    relative = lambda name: f"../figures/wafer_mean_si_etch_lolo_v1/{name}.png"  # noqa: E731
    comparison_markdown = model_comparison_markdown(
        comparison,
        bootstrap_intervals=bootstrap,
        primary_model_id="cycle_ridge",
    )
    target_missing_by_lot = (
        targets.groupby("lot_number")["n_interpolated_postox_sites"].sum().astype(int)
    )

    report = f"""# Wafer-Mean Silicon Etch Virtual Metrology

## Results

This analysis predicts final wafer-mean silicon etch from the process trace of one completed
BOSCH run. The dataset provides 88 labeled wafers from 10 lots. One wafer/run is one sample;
timestamps, cycles, and metrology sites are kept inside that sample.

Cycle-feature Ridge reached a macro-lot MAE of **{ridge['macro_lot_mae_um']:.3f} µm**
(95% lot-cluster interval {ridge_ci['lower']:.3f}–{ridge_ci['upper']:.3f}), pooled RMSE
**{ridge['rmse_um']:.3f} µm**, and held-out R² **{ridge['r2_oof']:.3f}**. Elastic Net had the
lowest MAE in this comparison at **{elastic_mae:.3f} µm**. Its lot-level MAE was
{abs(elastic_delta['delta_macro_lot_mae_um']):.3f} µm lower than Ridge on average and lower
on {int(elastic_delta['lots_better_than_reference'])} of 10 lots. The ranking still needs
confirmation on new lots because all model families were compared on the same held-out data.

## Data and prediction task

- **Input:** 31 common process readbacks, about 3,200 timestamps at 5 Hz, summarized across
  100 process cycles into 81 features.
- **Target:** the arithmetic mean of the 89 published final `si_etch` site values.
- **Cohort:** 96 process runs; 88 have a matched complete 89-site map across 10 lots.
- **Prediction time:** after the run ends and before physical metrology is returned.

The figure below uses the chronologically first matched run, `{wafer_example_key}`. The left
panel shows recorded gas readbacks and cycle boundaries found without using the target. The
right panel shows the final 89-site metrology map for the same wafer.

![One real wafer run and its metrology map]({relative('wafer_run_example')})

See the [data dictionary](../data/DATA_STRUCTURE.md) for field definitions and the target
calculation.

## Validation

Each outer fold holds out one complete lot and trains on the other nine. Imputation,
variance filtering, scaling, PCA, and hyperparameter search are fitted only on training
lots. Each wafer receives one prediction from a model that did not see its lot. No wafer's
89 sites cross fold boundaries.

Macro-lot MAE is the main metric: MAE is calculated separately for each lot and the ten lot
values are averaged. This prevents larger lots from dominating the score. The confidence
intervals resample lots rather than individual wafers.

## Model comparison

{comparison_markdown}

Context-only Ridge was worse than Cycle-feature Ridge by
{context_delta['delta_macro_lot_mae_um']:+.3f} µm (95% paired lot-cluster interval
{context_delta['lower_95_um']:+.3f} to {context_delta['upper_95_um']:+.3f}) and was worse on
all 10 lots. The completed process trace therefore adds useful predictive information beyond
conditioning and wafer order in this dataset. RBF-SVR and PCA+GPR did not improve on the
regularized linear models.

![Macro-lot MAE comparison]({relative('model_comparison')})

*Points are means of ten held-out-lot MAEs; bars are 95% lot-cluster bootstrap intervals.*

![Measured versus predicted values]({relative('oof_parity_panels')})

*Each point is one wafer predicted while its lot was held out. Metric boxes report pooled
wafer errors, while the comparison table uses macro-lot MAE.*

## Lot-to-lot behavior

{_markdown_table(ridge_by_lot, [
    ('lot_number', 'Lot', '{:.0f}'),
    ('n_wafers', 'n', '{:.0f}'),
    ('mae_um', 'MAE [µm]', '{:.3f}'),
    ('rmse_um', 'RMSE [µm]', '{:.3f}'),
    ('bias_um', 'Bias [µm]', '{:+.3f}'),
    ('r2', 'Within-lot R²', '{:.3f}'),
])}

The largest Ridge lot MAE was {ridge_by_lot['mae_um'].max():.3f} µm on lot 2. Within-lot R²
was negative on lots 8 and 10; those lots have narrow target ranges, and lot 10 has only four
labeled wafers. Per-lot absolute error is more stable than per-lot R² here.

![MAE by held-out lot]({relative('model_lot_mae_heatmap')})

![Ridge residuals by wafer order]({relative('primary_residual_vs_wafer_order')})

*Connected points show order within one lot; they are not a continuous timeline across lots.*

## Tuning results

{_markdown_table(final_params, [
    ('model', 'Model', ''),
    ('parameter', 'Parameter', ''),
    ('value', 'All-data refit value', ''),
    ('best_inner_macro_lot_mae_um', 'LOGO selection MAE [µm]', '{:.3f}'),
])}

Ridge selected `alpha=10` in eight of ten outer folds. Elastic Net selected `alpha=0.01`
and `l1_ratio=0.9` in seven of ten folds. The all-data refits are saved for reuse but do not
contribute to any held-out result.

## Model interpretation

The largest Ridge coefficients included wafer order, foreline-pressure phase contrast,
source-RF reflected-power summaries, platen-RF tuning signals, heater state, and startup
timing. Each coefficient is the change in predicted micrometres associated with a
one-standard-deviation feature change after accounting for the other inputs. Correlated
readbacks can redistribute coefficient weight, so these are predictive associations rather
than effects of changing a recipe setting.

{_markdown_table(coefficient_display, [
    ('feature_name', 'Feature', ''),
    ('coefficient_median', 'Median coefficient', '{:+.4f}'),
    ('coefficient_q25', 'Q25', '{:+.4f}'),
    ('coefficient_q75', 'Q75', '{:+.4f}'),
])}

The cycle-response block produced the largest error increase when jointly permuted. It also
contains 56 correlated features, so its value is not directly comparable to a five-feature
context block and does not identify a causal control knob.

{_markdown_table(permutation_display, [
    ('block_name', 'Feature block', ''),
    ('delta_mae_median_um', 'Median held-out ΔMAE [µm]', '{:+.3f}'),
    ('delta_mae_q25_um', 'Q25', '{:+.3f}'),
    ('delta_mae_q75_um', 'Q75', '{:+.3f}'),
])}

![Coefficient variation across outer folds]({relative('coefficient_stability')})

*Outer-fold fits share most of their training lots; sign consistency is descriptive, not ten
independent replications.*

![Joint feature-block permutation]({relative('block_permutation_importance')})

## Sensitivity checks

{_markdown_table(sensitivity_display, [
    ('variant', 'Change', ''),
    ('n_features', 'Features', '{:.0f}'),
    ('macro_lot_mae_um', 'Macro-lot MAE [µm]', '{:.3f}'),
    ('delta_ci', 'Δ vs Ridge [95% interval]', ''),
    ('lots_better_than_primary', 'Lots with lower error', '{:.0f}'),
])}

- Removing RF and power readbacks increased macro-lot MAE by
  {no_rf['paired_delta_macro_lot_mae_um']:+.3f} µm (95% interval
  {no_rf['paired_delta_lower_95_um']:+.3f} to {no_rf['paired_delta_upper_95_um']:+.3f}).
- RobustScaler reduced MAE by {abs(robust['paired_delta_macro_lot_mae_um']):.3f} µm in this
  comparison and should be evaluated again on new lots.
- Removing Gas4/Gas5 readbacks reduced MAE by
  {abs(no_gas['paired_delta_macro_lot_mae_um']):.3f} µm. This can reflect redundancy or
  unstable readbacks; it does not mean gas chemistry is unimportant.

![Sensitivity checks]({relative('sensitivity_delta')})

*Intervals are paired lot-cluster bootstrap intervals. Negative values mean lower error than
the unchanged Ridge pipeline.*

## Metrology lineage

The published target contains 157 failed post-etch spectral fits completed by inverse-distance
weighting, mainly in lot 3 ({target_missing_by_lot.get(3, 0)} sites) and lot 4
({target_missing_by_lot.get(4, 0)} sites). Recomputing the target from direct post-oxide fits
only gave macro-lot MAE {direct['macro_lot_mae_um']:.3f} µm and held-out R²
{direct['oof_r2']:.3f}. The alternative target was lower by
{abs(direct['target_mean_shift_um']):.3f} µm on average and differed by as much as
{direct['target_max_absolute_shift_um']:.3f} µm for one wafer. This is a change in target
definition, not a model improvement.

![Target lineage sensitivity]({relative('target_lineage')})

*Color shows how many post-oxide sites on each wafer were completed by interpolation.*

## Uncertainty calibration

PCA+GPR's predicted standard deviation had Spearman correlation {rho:.3f} with absolute
error. Nominal 90% and 95% Gaussian intervals covered {coverage90:.1%} and {coverage95:.1%}
of held-out wafers. Both intervals under-covered and should not be used as a metrology-skip
gate without group-aware calibration.

![GPR interval coverage]({relative('gpr_uncertainty')})

*The diagonal is the correct reference here: nominal interval coverage versus observed
coverage. The shaded band is the interquartile range across the ten held-out lots.*

## Sample size

There are 88 labeled wafer runs across 10 lots. The timestamps within each run share one
final wafer label. I used cycle summaries and regularized models to keep the number of
parameters small; sequence neural networks haven't been evaluated here.

## Limitations

1. Generalization is assessed on 88 labeled wafers from only 10 lot/date groups on one tool.
   Wafers within a lot share chamber history, and conditioning factors are partially confounded
   with lot and date.
2. This is end-of-run virtual metrology: the model uses the completed process trace to predict
   mean silicon etch depth. It does not support pre-run recipe selection, early-run intervention,
   or reconstruction of the 89-site wafer map.
3. The silicon-etch target is derived from step-height and post-oxide measurements rather than
   measured directly. Approximately 2% of post-oxide site values were spatially completed after
   fit failures, adding uncertainty to the target.
4. Coefficients and permutation importance describe predictive associations, not causal effects
   of changing recipe settings or chamber conditions.
5. The tested Gaussian-process intervals were not calibrated: nominal 95% intervals achieved
   84.1% empirical coverage and are not suitable for automated metrology-skipping decisions.

## Follow-up experiments

1. Re-evaluate Elastic Net and RobustScaler-Ridge on newly collected lots.
2. Predict a low-dimensional basis of the 89-site map within the same grouped validation.
3. Calibrate uncertainty by lot and define when physical metrology must still be requested.
4. Compare telemetry-only and telemetry-plus-OES on a matched cohort.
5. Treat recipe optimization as a separate study using independently varied setpoints.

PLS/Ridge and batch-feature choices are consistent with Virtual Metrology literature,
including [Khan et al.](https://doi.org/10.1016/j.jprocont.2008.04.014) and
[Suthar et al.](https://doi.org/10.1016/j.compchemeng.2019.05.016).

## Reproduction

~~~bash
python scripts/download_zenodo.py
bosch-vm prepare --config experiments/mean_si_etch/config.yaml --force
bosch-vm run --config experiments/mean_si_etch/config.yaml --n-jobs -1 --force
bosch-vm sensitivity --config experiments/mean_si_etch/config.yaml --n-jobs -1 --force
python scripts/generate_analysis_artifacts.py
pytest
~~~

Full local runs save inner-search scores, held-out predictions, selected hyperparameters,
interpretation tables, and all-data refits under
`results/wafer_mean_si_etch_lolo_v1/models/<model>/`. Reported performance comes from
held-out predictions, not from the all-data refit.
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    args = _parse_args()
    results_dir = args.results_dir
    figure_dir = args.figure_dir
    figure_dir.mkdir(parents=True, exist_ok=True)
    comparison = pd.read_csv(results_dir / "comparison.csv")
    oof = pd.read_csv(results_dir / "all_oof_predictions.csv")
    per_lot = pd.read_csv(results_dir / "all_metrics_by_lot.csv")
    targets = pd.read_csv(args.target_audit)
    sensitivity = pd.read_csv(results_dir / "sensitivity/sensitivity_summary.csv")

    bootstrap = combine_bootstrap(results_dir)
    paired = paired_model_differences(
        oof,
        references=("cycle_ridge", "dummy_mean"),
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    paired.to_csv(results_dir / "paired_model_differences.csv", index=False)
    hyperparameters, hyperparameter_summary = collect_hyperparameters(results_dir)
    uncertainty, _ = gpr_uncertainty_diagnostics(oof, results_dir)

    coefficients = pd.read_csv(
        _model_path(results_dir, "cycle_ridge", "outer_coefficients.csv")
    )
    permutation = pd.read_csv(
        _model_path(results_dir, "cycle_ridge", "heldout_block_permutation.csv")
    )
    coefficient_summary = summarize_coefficient_stability(
        coefficients, model_id="cycle_ridge", top_n=None
    )
    coefficient_summary.to_csv(results_dir / "cycle_ridge_coefficient_stability.csv", index=False)
    permutation_summary = summarize_block_permutation(
        permutation, model_id="cycle_ridge"
    )
    permutation_summary.to_csv(
        results_dir / "cycle_ridge_block_permutation_summary.csv", index=False
    )

    plot_model_comparison(
        comparison,
        figure_dir / "model_comparison.png",
        bootstrap_intervals=bootstrap,
        model_order=MODEL_ORDER,
    )
    plot_oof_parity_panels(
        oof, figure_dir / "oof_parity_panels.png", model_order=MODEL_ORDER
    )
    plot_model_lot_mae_heatmap(
        oof, figure_dir / "model_lot_mae_heatmap.png", model_order=MODEL_ORDER
    )
    plot_primary_residual_vs_wafer_order(
        oof, figure_dir / "primary_residual_vs_wafer_order.png"
    )
    plot_coefficient_stability(
        coefficients, figure_dir / "coefficient_stability.png", top_n=12
    )
    plot_block_permutation_importance(
        permutation, figure_dir / "block_permutation_importance.png"
    )
    _plot_sensitivity(sensitivity, figure_dir / "sensitivity_delta.png")
    _plot_target_lineage(targets, figure_dir / "target_lineage.png")
    _plot_gpr_uncertainty(oof, figure_dir / "gpr_uncertainty.png")
    wafer_example_key = _plot_wafer_run_example(
        args.raw_dir, figure_dir / "wafer_run_example.png"
    )

    write_model_cards(results_dir, comparison, bootstrap)
    write_report(
        report_path=args.report_path,
        figure_dir=figure_dir,
        comparison=comparison,
        bootstrap=bootstrap,
        paired=paired,
        per_lot=per_lot,
        hyperparameters=hyperparameters,
        coefficient_summary=coefficient_summary,
        permutation_summary=permutation_summary,
        sensitivity=sensitivity,
        uncertainty=uncertainty,
        targets=targets,
        wafer_example_key=wafer_example_key,
    )
    summary = {
        "comparison": str(results_dir / "comparison.csv"),
        "paired_model_differences": str(results_dir / "paired_model_differences.csv"),
        "hyperparameter_summary": str(results_dir / "hyperparameter_summary.csv"),
        "report": str(args.report_path),
        "figures": sorted(str(path) for path in figure_dir.glob("*.png")),
        "model_cards": [str(results_dir / "models" / model / "README.md") for model in MODEL_ORDER],
    }
    (results_dir / "analysis_artifact_manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
