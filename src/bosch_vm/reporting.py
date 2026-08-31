"""Reporting utilities for wafer-level virtual metrology results.

Every predictive-performance figure in this module is built from outer-fold
predictions.  The plotting functions intentionally refuse a prediction table
without an ``outer_fold`` column so that training or inner-CV scores cannot be
mistaken for held-out performance.

The public API accepts :class:`pandas.DataFrame` objects and explicit output
paths; it never discovers result files implicitly.  This keeps report creation
reproducible and makes the provenance of each figure visible to the caller.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

UM = "µm"
PRIMARY_COLOR = "#0B3C5D"
SECONDARY_COLOR = "#2A9D8F"
DUMMY_COLOR = "#8D99AE"
POSITIVE_COLOR = "#D55E00"
NEGATIVE_COLOR = "#0072B2"
GRID_COLOR = "#D9E1E8"
TEXT_COLOR = "#243442"
MODEL_DISPLAY_NAMES = {
    "dummy_mean": "Training-mean dummy",
    "train_mean_dummy": "Training-mean dummy",
    "context_ridge": "Context-only Ridge",
    "cycle_ridge": "Cycle-feature Ridge",
    "pls": "PLS",
    "elastic_net": "Elastic Net",
    "elasticnet": "Elastic Net",
    "rbf_svr": "RBF-SVR",
    "pca_gpr": "PCA + Gaussian process",
    "extra_trees": "Extra Trees",
}
FEATURE_DISPLAY_TERMS = {
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
BLOCK_DISPLAY_NAMES = {
    "cycle": "Cycle-response telemetry (56 features)",
    "timing": "Timing (10 features)",
    "slow": "Slow-state telemetry (10 features)",
    "context": "Process context (5 features)",
}

@dataclass(frozen=True)
class PlotArtifact:
    """A saved figure and the caption that should accompany it."""

    path: Path
    caption: str


@dataclass(frozen=True)
class ReportArtifacts:
    """Paths created by :func:`generate_report_artifacts`."""

    report_path: Path
    figures: Mapping[str, PlotArtifact]


def plot_model_comparison(
    comparison: pd.DataFrame,
    output_path: str | Path,
    *,
    bootstrap_intervals: pd.DataFrame | None = None,
    primary_model_id: str = "cycle_ridge",
    model_order: Sequence[str] | None = None,
) -> PlotArtifact:
    """Plot macro-lot MAE with lot-cluster bootstrap intervals.

    ``comparison`` must contain one row per model. Confidence intervals can
    either be columns on ``comparison`` or rows of a long-form bootstrap table
    with ``metric``, ``lower_95`` and ``upper_95`` columns.
    """

    frame = _normalize_comparison(comparison)
    if model_order is None:
        frame = frame.sort_values(["macro_lot_mae_um", "model_id"], kind="stable")
    else:
        frame = _ordered_models(frame, model_order)
    frame = frame.reset_index(drop=True)
    frame = _merge_macro_mae_intervals(frame, bootstrap_intervals)

    labels = [_display_model_name(value) for value in frame["model_id"]]
    y = np.arange(len(frame), dtype=float)
    colors = [_model_color(model_id) for model_id in frame["model_id"]]

    height = max(4.8, 0.55 * len(frame) + 1.5)
    with _report_style():
        fig, ax = plt.subplots(figsize=(9.2, height))

        lower = frame["macro_lot_mae_ci_lower_um"].to_numpy(dtype=float)
        upper = frame["macro_lot_mae_ci_upper_um"].to_numpy(dtype=float)
        estimate = frame["macro_lot_mae_um"].to_numpy(dtype=float)
        valid_ci = np.isfinite(lower) & np.isfinite(upper)
        if valid_ci.any():
            error = np.vstack(
                [
                    np.maximum(0.0, estimate[valid_ci] - lower[valid_ci]),
                    np.maximum(0.0, upper[valid_ci] - estimate[valid_ci]),
                ]
            )
            ax.errorbar(
                estimate[valid_ci],
                y[valid_ci],
                xerr=error,
                fmt="none",
                ecolor="#6C7A89",
                elinewidth=1.5,
                capsize=3,
                zorder=1,
            )
        ax.scatter(estimate, y, c=colors, s=72, edgecolor="white", linewidth=0.8, zorder=2)
        finite_bounds = np.r_[estimate, lower[np.isfinite(lower)], upper[np.isfinite(upper)]]
        span = max(float(finite_bounds.max() - finite_bounds.min()), 0.05)
        label_offset = 0.018 * span
        for row, value, lo, hi in zip(y, estimate, lower, upper, strict=True):
            label_x = hi if np.isfinite(hi) else value
            interval = (
                f"{value:.3f}  [{lo:.3f}, {hi:.3f}]"
                if np.isfinite(lo) and np.isfinite(hi)
                else f"{value:.3f}"
            )
            ax.text(
                label_x + label_offset,
                row,
                interval,
                va="center",
                ha="left",
                fontsize=8.6,
                color="#465564",
            )
        ax.set_yticks(y, labels)
        ax.set_ylim(len(frame) - 0.35, -0.65)
        ax.set_xlabel(f"Macro-lot mean absolute error ({UM}; lower is better)")
        ax.set_title("Held-out-lot model comparison", fontweight="bold", pad=13)
        ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
        ax.grid(axis="y", visible=False)
        x_left = min(0.0, float(finite_bounds.min()) - 0.03 * span)
        x_right = float(finite_bounds.max()) + 0.40 * span
        ax.set_xlim(x_left, x_right)
        fig.tight_layout()

    caption = (
        "Each point is the mean of the ten held-out-lot MAEs; bars show 95% "
        "lot-cluster bootstrap intervals."
    )
    return _save_figure(fig, output_path, caption)


def plot_oof_parity_panels(
    oof_predictions: pd.DataFrame,
    output_path: str | Path,
    *,
    model_order: Sequence[str] | None = None,
) -> PlotArtifact:
    """Draw one measured-versus-predicted panel per model from outer OOF rows."""

    frame = _validate_oof_predictions(oof_predictions)
    model_ids = _model_ids(frame, model_order)
    ncols = min(4, max(1, len(model_ids)))
    nrows = math.ceil(len(model_ids) / ncols)
    all_values = np.r_[frame["y_true"].to_numpy(), frame["y_pred"].to_numpy()]
    lower, upper = _square_limits(all_values)

    with _report_style():
        fig, axes_array = plt.subplots(
            nrows,
            ncols,
            figsize=(4.15 * ncols, 3.85 * nrows + 1.0),
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        axes = list(axes_array.ravel())
        for ax, model_id in zip(axes, model_ids, strict=False):
            group = frame.loc[frame["model_id"] == model_id]
            ax.scatter(
                group["y_true"],
                group["y_pred"],
                s=31,
                color=SECONDARY_COLOR,
                alpha=0.78,
                edgecolor="white",
                linewidth=0.45,
            )
            ax.plot([lower, upper], [lower, upper], color="#30343B", linestyle="--", linewidth=1.2)
            ax.set_xlim(lower, upper)
            ax.set_ylim(lower, upper)
            mae = float(np.mean(np.abs(group["y_pred"] - group["y_true"])))
            rmse = float(np.sqrt(np.mean(np.square(group["y_pred"] - group["y_true"]))))
            r2 = _safe_r2(group["y_true"], group["y_pred"])
            metrics = (
                f"n={len(group)}\nPooled MAE={mae:.3f} {UM}\n"
                f"Pooled RMSE={rmse:.3f} {UM}\n$R^2$={r2:.2f}"
            )
            ax.text(
                0.04,
                0.96,
                metrics,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8.5,
                bbox={
                    "boxstyle": "round,pad=0.35",
                    "facecolor": "white",
                    "alpha": 0.9,
                    "edgecolor": GRID_COLOR,
                },
            )
            ax.set_title(_display_model_name(model_id), fontweight="bold")
            ax.grid(color=GRID_COLOR, linewidth=0.75)
            ax.set_aspect("equal", adjustable="box")
        for ax in axes[len(model_ids) :]:
            ax.set_visible(False)
        for row in range(nrows):
            axes_array[row, 0].set_ylabel(f"Held-out-lot prediction ({UM})")
        for col in range(ncols):
            axes_array[-1, col].set_xlabel(f"Measured wafer mean ({UM})")

        fig.suptitle(
            "Measured versus held-out-lot predictions",
            y=0.985,
            fontsize=16,
            fontweight="bold",
            color=TEXT_COLOR,
        )
        fig.tight_layout(rect=(0.02, 0.03, 0.99, 0.94))

    caption = (
        "Each point is one wafer predicted while its entire lot was held out; the dashed "
        "line marks perfect agreement."
    )
    return _save_figure(fig, output_path, caption)


def plot_model_lot_mae_heatmap(
    oof_predictions: pd.DataFrame,
    output_path: str | Path,
    *,
    model_order: Sequence[str] | None = None,
) -> PlotArtifact:
    """Plot model-by-held-out-lot MAE, derived only from outer OOF rows."""

    frame = _validate_oof_predictions(oof_predictions)
    lot_column = _resolve_column(frame, ("lot_number", "outer_test_lot", "lot"), "lot")
    grouped = (
        frame.assign(_absolute_error=(frame["y_pred"] - frame["y_true"]).abs())
        .groupby(["model_id", lot_column], sort=True, as_index=False)["_absolute_error"]
        .mean()
        .rename(columns={"_absolute_error": "mae_um"})
    )
    pivot = grouped.pivot(index="model_id", columns=lot_column, values="mae_um")
    order = _model_ids(frame, model_order)
    pivot = pivot.reindex(order)

    width = max(9.0, 0.78 * pivot.shape[1] + 3.1)
    height = max(5.0, 0.55 * pivot.shape[0] + 2.5)
    with _report_style():
        fig, ax = plt.subplots(figsize=(width, height))
        values = pivot.to_numpy(dtype=float)
        image = ax.imshow(values, aspect="auto", cmap="YlGnBu", interpolation="nearest")
        ax.set_xticks(np.arange(pivot.shape[1]), [str(value) for value in pivot.columns])
        ax.set_yticks(
            np.arange(pivot.shape[0]),
            [_display_model_name(value) for value in pivot.index],
        )
        ax.set_xlabel("Held-out lot")
        ax.set_ylabel("Model procedure")
        ax.set_title("MAE by model and held-out lot", fontweight="bold", pad=14)
        colorbar = fig.colorbar(image, ax=ax, pad=0.018, fraction=0.035)
        colorbar.set_label(f"MAE ({UM})")

        threshold = float(np.nanmedian(values)) if np.isfinite(values).any() else 0.0
        if values.size <= 140:
            for row in range(values.shape[0]):
                for column in range(values.shape[1]):
                    value = values[row, column]
                    if not np.isfinite(value):
                        continue
                    ax.text(
                        column,
                        row,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white" if value > threshold else "#17324D",
                    )
        ax.tick_params(axis="x", rotation=0)
        fig.tight_layout(rect=(0.02, 0.03, 0.98, 0.97))

    caption = (
        "Cell values are MAE computed only on wafers from the indicated outer-held-out lot. "
        "The ten columns expose lot-to-lot differences hidden by a pooled score."
    )
    return _save_figure(fig, output_path, caption)


def plot_primary_residual_vs_wafer_order(
    oof_predictions: pd.DataFrame,
    output_path: str | Path,
    *,
    primary_model_id: str = "cycle_ridge",
    wafer_order_column: str | None = None,
) -> PlotArtifact:
    """Plot primary-model OOF residuals over within-lot wafer order.

    Residuals follow the frozen convention ``prediction - measured``.
    """

    frame = _validate_oof_predictions(oof_predictions)
    frame = frame.loc[frame["model_id"] == primary_model_id].copy()
    if frame.empty:
        raise ValueError(f"Primary model {primary_model_id!r} is absent from OOF predictions.")
    lot_column = _resolve_column(frame, ("lot_number", "outer_test_lot", "lot"), "lot")
    if wafer_order_column is None:
        wafer_order_column = _resolve_column(
            frame, ("wafer_order", "wafer_number", "wafer_index"), "wafer order"
        )
    _require_columns(frame, (wafer_order_column,))
    frame["_residual"] = frame["residual"]

    lots = list(pd.unique(frame[lot_column]))
    palette = plt.get_cmap("tab10")
    with _report_style():
        fig, ax = plt.subplots(figsize=(10.8, 6.2))
        for index, lot in enumerate(lots):
            group = frame.loc[frame[lot_column] == lot].sort_values(wafer_order_column)
            color = palette(index % 10)
            ax.plot(
                group[wafer_order_column],
                group["_residual"],
                color=color,
                linewidth=1.05,
                alpha=0.62,
            )
            ax.scatter(
                group[wafer_order_column],
                group["_residual"],
                color=color,
                edgecolor="white",
                linewidth=0.5,
                s=38,
                label=f"Lot {lot}",
                zorder=2,
            )
        ax.axhline(0.0, color="#30343B", linestyle="--", linewidth=1.2)
        bias = float(frame["_residual"].mean())
        ax.axhline(bias, color=PRIMARY_COLOR, linestyle=":", linewidth=1.3)
        ax.text(
            0.99,
            0.04,
            f"Overall bias = {bias:+.3f} {UM}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            color=PRIMARY_COLOR,
            fontweight="bold",
        )
        ax.set_xlabel("Wafer order within lot")
        ax.set_ylabel(f"Held-out-lot residual: prediction − measured ({UM})")
        ax.set_title(
            f"{_display_model_name(primary_model_id)} residuals by wafer order",
            fontweight="bold",
            pad=13,
        )
        ax.grid(color=GRID_COLOR, linewidth=0.75)
        ax.legend(
            title="Held-out lot",
            bbox_to_anchor=(1.01, 1.0),
            loc="upper left",
            frameon=False,
            ncol=1,
        )
        fig.tight_layout(rect=(0.02, 0.03, 0.86, 0.98))

    caption = (
        "Residuals use prediction minus measured wafer-mean Si etch. Connected points show "
        "order within one lot, not a continuous timeline across lots."
    )
    return _save_figure(fig, output_path, caption)


def plot_coefficient_stability(
    coefficients: pd.DataFrame,
    output_path: str | Path,
    *,
    model_id: str = "cycle_ridge",
    top_n: int = 18,
) -> PlotArtifact:
    """Plot median and interquartile range of outer-fold standardized coefficients."""

    summary = summarize_coefficient_stability(coefficients, model_id=model_id, top_n=top_n)
    y = np.arange(len(summary), dtype=float)
    values = summary["coefficient_median"].to_numpy(dtype=float)
    lower = summary["coefficient_q25"].to_numpy(dtype=float)
    upper = summary["coefficient_q75"].to_numpy(dtype=float)
    errors = np.vstack([values - lower, upper - values])
    colors = [POSITIVE_COLOR if value >= 0 else NEGATIVE_COLOR for value in values]

    height = max(5.5, 0.43 * len(summary) + 2.3)
    with _report_style():
        fig, ax = plt.subplots(figsize=(10.6, height))
        ax.axvline(0.0, color="#30343B", linestyle="--", linewidth=1.1)
        ax.errorbar(
            values,
            y,
            xerr=errors,
            fmt="none",
            ecolor="#748392",
            elinewidth=2.0,
            capsize=3,
            zorder=1,
        )
        ax.scatter(values, y, c=colors, s=62, edgecolor="white", linewidth=0.7, zorder=2)
        ax.set_yticks(y, [_display_feature(value) for value in summary["feature"]])
        ax.set_ylim(len(summary) - 0.35, -0.65)
        ax.set_xlabel("Standardized coefficient (median and IQR across outer folds)")
        ax.set_ylabel("Predictor feature")
        ax.set_title(
            f"{_display_model_name(model_id)} coefficients across outer folds",
            fontweight="bold",
            pad=13,
        )
        ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
        ax.grid(axis="y", visible=False)
        fig.tight_layout(rect=(0.02, 0.03, 0.98, 0.98))

    caption = (
        "Points are median standardized coefficients and bars are interquartile ranges across "
        "the ten fitted outer-fold models. Those fits share most of their training lots, so "
        "sign counts are descriptive rather than independent evidence of stability."
    )
    return _save_figure(fig, output_path, caption)


def plot_block_permutation_importance(
    permutation_importance: pd.DataFrame,
    output_path: str | Path,
    *,
    model_id: str = "cycle_ridge",
) -> PlotArtifact:
    """Plot held-out physical-block permutation importance as change in MAE."""

    summary = summarize_block_permutation(permutation_importance, model_id=model_id)
    y = np.arange(len(summary), dtype=float)
    values = summary["delta_mae_median_um"].to_numpy(dtype=float)
    lower = summary["delta_mae_q25_um"].to_numpy(dtype=float)
    upper = summary["delta_mae_q75_um"].to_numpy(dtype=float)
    errors = np.vstack([values - lower, upper - values])
    colors = [POSITIVE_COLOR if value >= 0 else DUMMY_COLOR for value in values]

    height = max(4.9, 0.58 * len(summary) + 2.2)
    with _report_style():
        fig, ax = plt.subplots(figsize=(10.4, height))
        ax.axvline(0.0, color="#30343B", linestyle="--", linewidth=1.1)
        ax.errorbar(
            values,
            y,
            xerr=errors,
            fmt="none",
            ecolor="#748392",
            elinewidth=2.0,
            capsize=3,
            zorder=1,
        )
        ax.scatter(values, y, c=colors, s=68, edgecolor="white", linewidth=0.7, zorder=2)
        ax.set_yticks(
            y,
            [
                BLOCK_DISPLAY_NAMES.get(str(value), _display_feature(value))
                for value in summary["block"]
            ],
        )
        ax.set_ylim(len(summary) - 0.35, -0.65)
        ax.set_xlabel(f"Change in held-out-lot MAE after joint permutation ({UM})")
        ax.set_ylabel("Predictor block")
        ax.set_title(
            f"{_display_model_name(model_id)} reliance on feature blocks",
            fontweight="bold",
            pad=13,
        )
        ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
        ax.grid(axis="y", visible=False)
        fig.tight_layout(rect=(0.02, 0.03, 0.98, 0.98))

    caption = (
        "Each point is the median increase in outer-held-out MAE after jointly permuting a "
        "feature block; bars show the outer-fold IQR. Blocks differ in size and contain "
        "correlated readbacks, so the values measure model reliance rather than causal effects."
    )
    return _save_figure(fig, output_path, caption)


def summarize_coefficient_stability(
    coefficients: pd.DataFrame,
    *,
    model_id: str = "cycle_ridge",
    top_n: int | None = 18,
) -> pd.DataFrame:
    """Aggregate outer-fold coefficients without interpreting them causally."""

    frame = _copy_frame(coefficients, "coefficients")
    feature_column = _resolve_column(
        frame, ("feature", "feature_name", "predictor"), "coefficient feature"
    )
    value_column = _resolve_column(
        frame,
        ("standardized_coefficient", "coefficient", "coef", "value"),
        "coefficient value",
    )
    fold_column = _resolve_column(
        frame, ("outer_fold", "outer_lot"), "outer-fold identifier"
    )
    if fold_column != "outer_fold":
        frame = frame.rename(columns={fold_column: "outer_fold"})
    if "model_id" in frame:
        frame = frame.loc[frame["model_id"] == model_id]
    if frame.empty:
        raise ValueError(f"No coefficient rows are available for model {model_id!r}.")
    frame[value_column] = pd.to_numeric(frame[value_column], errors="raise")

    def sign_consistency(values: pd.Series) -> float:
        array = values.to_numpy(dtype=float)
        median = float(np.median(array))
        if median > 0:
            return float(np.mean(array > 0))
        if median < 0:
            return float(np.mean(array < 0))
        return float(np.mean(array == 0))

    summary = (
        frame.groupby(feature_column, sort=False)[value_column]
        .agg(
            coefficient_median="median",
            coefficient_q25=lambda values: values.quantile(0.25),
            coefficient_q75=lambda values: values.quantile(0.75),
            outer_folds="count",
            sign_consistency=sign_consistency,
        )
        .reset_index()
        .rename(columns={feature_column: "feature"})
    )
    summary["_magnitude"] = summary["coefficient_median"].abs()
    summary = summary.sort_values(["_magnitude", "feature"], ascending=[False, True])
    if top_n is not None:
        if top_n < 1:
            raise ValueError("top_n must be positive or None.")
        summary = summary.head(top_n)
    return summary.drop(columns="_magnitude").reset_index(drop=True)


def summarize_block_permutation(
    permutation_importance: pd.DataFrame,
    *,
    model_id: str = "cycle_ridge",
) -> pd.DataFrame:
    """Aggregate held-out block permutation increases in MAE across outer folds."""

    frame = _copy_frame(permutation_importance, "permutation_importance")
    block_column = _resolve_column(
        frame, ("block", "feature_block", "physical_block", "group"), "feature block"
    )
    value_column = _resolve_column(
        frame,
        (
            "delta_mae_um",
            "mae_increase_um",
            "permutation_delta_mae_um",
            "importance_um",
            "importance",
        ),
        "permutation MAE change",
    )
    fold_column = _resolve_column(
        frame, ("outer_fold", "outer_lot"), "outer-fold identifier"
    )
    if fold_column != "outer_fold":
        frame = frame.rename(columns={fold_column: "outer_fold"})
    if "model_id" in frame:
        frame = frame.loc[frame["model_id"] == model_id]
    if frame.empty:
        raise ValueError(f"No block-permutation rows are available for model {model_id!r}.")
    frame[value_column] = pd.to_numeric(frame[value_column], errors="raise")
    fold_level = (
        frame.groupby(["outer_fold", block_column], sort=False, as_index=False)[value_column]
        .mean()
    )
    summary = (
        fold_level.groupby(block_column, sort=False)[value_column]
        .agg(
            delta_mae_median_um="median",
            delta_mae_q25_um=lambda values: values.quantile(0.25),
            delta_mae_q75_um=lambda values: values.quantile(0.75),
            outer_folds="count",
        )
        .reset_index()
        .rename(columns={block_column: "block"})
        .sort_values(["delta_mae_median_um", "block"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return summary


def model_comparison_markdown(
    comparison: pd.DataFrame,
    *,
    bootstrap_intervals: pd.DataFrame | None = None,
    primary_model_id: str = "cycle_ridge",
) -> str:
    """Return a compact Markdown table of held-out-lot results."""

    frame = _normalize_comparison(comparison)
    frame = _merge_macro_mae_intervals(frame, bootstrap_intervals)
    frame = frame.sort_values(["macro_lot_mae_um", "model_id"], kind="stable").reset_index(
        drop=True
    )
    headers = [
        "Rank",
        "Model",
        f"Macro-lot MAE [{UM}] (95% CI)",
        f"Pooled RMSE [{UM}]",
        "OOF R²",
    ]
    include_skill = "sse_skill_vs_dummy" in frame
    if include_skill:
        headers.append("SSE skill vs dummy")
    rows: list[list[str]] = []
    for index, row in frame.iterrows():
        model_id = str(row["model_id"])
        label = _display_model_name(model_id)
        lower = row["macro_lot_mae_ci_lower_um"]
        upper = row["macro_lot_mae_ci_upper_um"]
        if np.isfinite(lower) and np.isfinite(upper):
            mae = f"{row['macro_lot_mae_um']:.3f} ({lower:.3f}–{upper:.3f})"
        else:
            mae = f"{row['macro_lot_mae_um']:.3f} (CI unavailable)"
        values = [
            str(index + 1),
            label,
            mae,
            f"{row['pooled_rmse_um']:.3f}",
            _format_number(row["oof_r2"], 3),
        ]
        if include_skill:
            values.append(_format_number(row.get("sse_skill_vs_dummy", np.nan), 3))
        rows.append(values)
    return _markdown_table(headers, rows)


def build_markdown_report(
    comparison: pd.DataFrame,
    figures: Mapping[str, PlotArtifact | str | Path],
    *,
    bootstrap_intervals: pd.DataFrame | None = None,
    primary_model_id: str = "cycle_ridge",
    title: str = "Wafer-Mean Si Etch Virtual Metrology Results",
    methodology: Sequence[str] | None = None,
    findings: Sequence[str] | None = None,
    limitations: Sequence[str] | None = None,
    report_directory: str | Path | None = None,
) -> str:
    """Build a self-contained Markdown report body from verified result tables."""

    frame = _normalize_comparison(comparison)
    primary = frame.loc[frame["model_id"] == primary_model_id]
    if len(primary) != 1:
        raise ValueError(
            f"Expected exactly one comparison row for primary model {primary_model_id!r}."
        )
    primary_row = primary.iloc[0]
    observed_lowest = frame.sort_values("macro_lot_mae_um", kind="stable").iloc[0]

    default_methodology = [
        (
            "One complete wafer process run is one supervised sample; the target is the "
            "arithmetic mean of the 89 published final Si-etch site values."
        ),
        (
            "Every performance value uses nested leave-one-lot-out validation: the outer "
            "lot supplies test wafers and inner leave-one-lot-out folds tune hyperparameters."
        ),
        (
            "The primary metric is macro-lot MAE, which gives each of the 10 lots equal "
            "weight. Pooled RMSE and OOF R² are secondary diagnostics."
        ),
        "All preprocessing and model selection are fitted inside the corresponding training folds.",
    ]
    default_findings = [
        (
            f"{_display_model_name(primary_model_id)} has held-out-lot macro-lot MAE "
            f"{primary_row['macro_lot_mae_um']:.3f} {UM}, "
            f"pooled RMSE {primary_row['pooled_rmse_um']:.3f} {UM}, and "
            f"OOF R² {_format_number(primary_row['oof_r2'], 3)}."
        ),
        (
            "The lowest MAE in the observed comparison is "
            f"{_display_model_name(observed_lowest['model_id'])} "
            f"at {observed_lowest['macro_lot_mae_um']:.3f} {UM}. Because model families were "
            "compared on the same 10 outer folds, new lots are needed to confirm this ranking."
        ),
    ]
    default_limitations = [
        (
            "Only 10 lots are available, so lot-cluster intervals and apparent model "
            "rankings are uncertain."
        ),
        (
            "The target range is narrow; R² can be unstable, so absolute errors in "
            "micrometres carry more decision value."
        ),
        "Sensor readbacks are observations of chamber state, not controllable recipe knobs.",
        (
            "Coefficients and permutation importance show predictive association, not "
            "causal process effects."
        ),
    ]

    lines = [
        f"# {title}",
        "",
        (
            "> Evaluation boundary: post-run, pre-metrology virtual metrology. All "
            "reported predictive metrics are outer-fold OOF values."
        ),
        "",
        "## Experimental protocol",
        "",
        *_markdown_bullets(methodology or default_methodology),
        "",
        "## Model comparison",
        "",
        model_comparison_markdown(
            frame,
            bootstrap_intervals=bootstrap_intervals,
            primary_model_id=primary_model_id,
        ),
        "",
        "## Main findings",
        "",
        *_markdown_bullets(findings or default_findings),
        "",
        "## Figures",
        "",
    ]
    base = Path(report_directory) if report_directory is not None else None
    for name, artifact in figures.items():
        if isinstance(artifact, PlotArtifact):
            path = artifact.path
            caption = artifact.caption
        else:
            path = Path(artifact)
            caption = "Figure generated from saved held-out-lot result tables."
        rendered_path = _relative_markdown_path(path, base)
        lines.extend(
            [
                f"### {_display_feature(name)}",
                "",
                f"![{_display_feature(name)}]({rendered_path})",
                "",
                f"*{caption}*",
                "",
            ]
        )
    lines.extend(
        [
            "## Limitations and interpretation boundary",
            "",
            *_markdown_bullets(limitations or default_limitations),
            "",
            "## Reproducibility note",
            "",
            (
                "The report consumes saved outer-OOF prediction, bootstrap, coefficient, "
                "and held-out permutation tables. It does not use fitted-training metrics."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_markdown_report(
    output_path: str | Path,
    comparison: pd.DataFrame,
    figures: Mapping[str, PlotArtifact | str | Path],
    **kwargs: Any,
) -> Path:
    """Write :func:`build_markdown_report` output using UTF-8."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs.setdefault("report_directory", path.parent)
    body = build_markdown_report(comparison, figures, **kwargs)
    path.write_text(body, encoding="utf-8")
    return path


def generate_report_artifacts(
    comparison: pd.DataFrame,
    oof_predictions: pd.DataFrame,
    output_dir: str | Path,
    *,
    bootstrap_intervals: pd.DataFrame | None = None,
    coefficients: pd.DataFrame | None = None,
    permutation_importance: pd.DataFrame | None = None,
    primary_model_id: str = "cycle_ridge",
    model_order: Sequence[str] | None = None,
    report_name: str = "modeling_report.md",
) -> ReportArtifacts:
    """Generate the standard figure set and a Markdown report.

    The caller is responsible for loading and concatenating source tables; this
    function performs no implicit result-file discovery.
    """

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    figures: dict[str, PlotArtifact] = {}
    figures["model_comparison"] = plot_model_comparison(
        comparison,
        directory / "model_comparison.png",
        bootstrap_intervals=bootstrap_intervals,
        primary_model_id=primary_model_id,
        model_order=model_order,
    )
    figures["oof_parity_panels"] = plot_oof_parity_panels(
        oof_predictions,
        directory / "oof_parity_panels.png",
        model_order=model_order,
    )
    figures["model_lot_mae_heatmap"] = plot_model_lot_mae_heatmap(
        oof_predictions,
        directory / "model_lot_mae_heatmap.png",
        model_order=model_order,
    )
    figures["primary_residual_vs_wafer_order"] = plot_primary_residual_vs_wafer_order(
        oof_predictions,
        directory / "primary_residual_vs_wafer_order.png",
        primary_model_id=primary_model_id,
    )
    if coefficients is not None and not coefficients.empty:
        figures["coefficient_stability"] = plot_coefficient_stability(
            coefficients,
            directory / "coefficient_stability.png",
            model_id=primary_model_id,
        )
    if permutation_importance is not None and not permutation_importance.empty:
        figures["block_permutation_importance"] = plot_block_permutation_importance(
            permutation_importance,
            directory / "block_permutation_importance.png",
            model_id=primary_model_id,
        )

    report_path = write_markdown_report(
        directory / report_name,
        comparison,
        figures,
        bootstrap_intervals=bootstrap_intervals,
        primary_model_id=primary_model_id,
    )
    return ReportArtifacts(report_path=report_path, figures=figures)


def _normalize_comparison(comparison: pd.DataFrame) -> pd.DataFrame:
    frame = _copy_frame(comparison, "comparison")
    _require_columns(frame, ("model_id", "macro_lot_mae_um"))
    rmse_column = _resolve_column(frame, ("pooled_rmse_um", "rmse_um"), "pooled RMSE")
    r2_column = _resolve_column(frame, ("oof_r2", "r2_oof"), "OOF R-squared")
    frame = frame.rename(columns={rmse_column: "pooled_rmse_um", r2_column: "oof_r2"})
    for column in ("macro_lot_mae_um", "pooled_rmse_um", "oof_r2"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if frame["model_id"].duplicated().any():
        duplicates = sorted(frame.loc[frame["model_id"].duplicated(), "model_id"].unique())
        raise ValueError(f"comparison must contain one row per model; duplicates: {duplicates}")
    return frame


def _merge_macro_mae_intervals(
    comparison: pd.DataFrame, bootstrap_intervals: pd.DataFrame | None
) -> pd.DataFrame:
    frame = comparison.copy()
    lower_aliases = (
        "macro_lot_mae_ci_lower_um",
        "macro_lot_mae_lower_95",
        "macro_lot_mae_um_lower_95",
        "macro_lot_mae_ci_low_um",
    )
    upper_aliases = (
        "macro_lot_mae_ci_upper_um",
        "macro_lot_mae_upper_95",
        "macro_lot_mae_um_upper_95",
        "macro_lot_mae_ci_high_um",
    )
    lower_column = next((column for column in lower_aliases if column in frame), None)
    upper_column = next((column for column in upper_aliases if column in frame), None)
    if lower_column is not None and upper_column is not None:
        frame["macro_lot_mae_ci_lower_um"] = pd.to_numeric(frame[lower_column], errors="coerce")
        frame["macro_lot_mae_ci_upper_um"] = pd.to_numeric(frame[upper_column], errors="coerce")
        return frame

    frame["macro_lot_mae_ci_lower_um"] = np.nan
    frame["macro_lot_mae_ci_upper_um"] = np.nan
    if bootstrap_intervals is None or bootstrap_intervals.empty:
        return frame
    bootstrap = _copy_frame(bootstrap_intervals, "bootstrap_intervals")
    _require_columns(bootstrap, ("model_id", "metric"))
    lower = _resolve_column(
        bootstrap, ("lower_95", "ci_lower", "lower", "q025"), "bootstrap lower bound"
    )
    upper = _resolve_column(
        bootstrap, ("upper_95", "ci_upper", "upper", "q975"), "bootstrap upper bound"
    )
    wanted = bootstrap.loc[
        bootstrap["metric"]
        .astype(str)
        .str.lower()
        .isin({"macro_lot_mae_um", "macro_lot_mae", "macro-lot-mae"}),
        ["model_id", lower, upper],
    ].copy()
    if wanted["model_id"].duplicated().any():
        raise ValueError("bootstrap_intervals contains duplicate macro-lot MAE rows per model.")
    wanted = wanted.rename(columns={lower: "_bootstrap_lower", upper: "_bootstrap_upper"})
    merged = frame.merge(wanted, on="model_id", how="left", validate="one_to_one")
    merged["macro_lot_mae_ci_lower_um"] = pd.to_numeric(
        merged.pop("_bootstrap_lower"), errors="coerce"
    )
    merged["macro_lot_mae_ci_upper_um"] = pd.to_numeric(
        merged.pop("_bootstrap_upper"), errors="coerce"
    )
    return merged


def _validate_oof_predictions(oof_predictions: pd.DataFrame) -> pd.DataFrame:
    frame = _copy_frame(oof_predictions, "oof_predictions")
    _require_columns(frame, ("model_id", "outer_fold", "y_true", "y_pred"))
    for column in ("y_true", "y_pred"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise ValueError(f"OOF column {column!r} must contain only finite values.")
    if frame["outer_fold"].isna().any():
        raise ValueError("Every OOF prediction must identify its outer_fold.")
    expected_residual = frame["y_pred"] - frame["y_true"]
    if "residual" in frame:
        frame["residual"] = pd.to_numeric(frame["residual"], errors="raise")
        if not np.allclose(
            frame["residual"].to_numpy(dtype=float),
            expected_residual.to_numpy(dtype=float),
            rtol=1e-8,
            atol=1e-10,
        ):
            raise ValueError(
                "OOF residual must use the frozen convention prediction - measured "
                "(y_pred - y_true)."
            )
    else:
        frame["residual"] = expected_residual
    if "experiment_key" in frame:
        duplicates = frame.duplicated(["model_id", "experiment_key"], keep=False)
        if duplicates.any():
            raise ValueError("Each wafer must appear exactly once per model in OOF predictions.")
    return frame


def _ordered_models(frame: pd.DataFrame, model_order: Sequence[str]) -> pd.DataFrame:
    order = list(model_order)
    present = set(frame["model_id"])
    missing = [model_id for model_id in order if model_id not in present]
    if missing:
        raise ValueError(f"model_order references absent models: {missing}")
    remainder = [model_id for model_id in frame["model_id"] if model_id not in set(order)]
    categories = [*order, *remainder]
    ranked = frame.assign(
        _model_order=pd.Categorical(frame["model_id"], categories=categories, ordered=True)
    )
    return ranked.sort_values("_model_order", kind="stable").drop(columns="_model_order")


def _model_ids(frame: pd.DataFrame, model_order: Sequence[str] | None) -> list[str]:
    present = list(pd.unique(frame["model_id"].astype(str)))
    if model_order is None:
        return present
    missing = [model_id for model_id in model_order if model_id not in present]
    if missing:
        raise ValueError(f"model_order references absent models: {missing}")
    return [*model_order, *[model_id for model_id in present if model_id not in model_order]]


def _resolve_column(frame: pd.DataFrame, aliases: Sequence[str], label: str) -> str:
    for column in aliases:
        if column in frame:
            return column
    raise ValueError(f"Missing {label} column; accepted names are {list(aliases)}.")


def _copy_frame(value: pd.DataFrame, label: str) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"{label} must be a pandas DataFrame; got {type(value).__name__}.")
    return value.copy()


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _display_model_name(value: Any) -> str:
    text = str(value)
    return MODEL_DISPLAY_NAMES.get(text, text.replace("_", " ").strip().title())


def _display_feature(value: Any) -> str:
    parts = str(value).split("__")
    return " · ".join(
        FEATURE_DISPLAY_TERMS.get(part, " ".join(part.replace("_", " ").split()))
        for part in parts
    )


def _model_color(model_id: Any) -> str:
    text = str(model_id).lower()
    if "dummy" in text or "mean" in text:
        return DUMMY_COLOR
    return SECONDARY_COLOR


def _safe_r2(y_true: pd.Series, y_pred: pd.Series) -> float:
    true = y_true.to_numpy(dtype=float)
    predicted = y_pred.to_numpy(dtype=float)
    denominator = float(np.square(true - true.mean()).sum())
    if denominator <= 0:
        return float("nan")
    return float(1.0 - np.square(true - predicted).sum() / denominator)


def _square_limits(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Cannot determine plotting limits from empty/nonfinite values.")
    lower = float(finite.min())
    upper = float(finite.max())
    spread = upper - lower
    pad = 0.06 * spread if spread > 0 else max(abs(lower) * 0.02, 0.1)
    return lower - pad, upper + pad


class _report_style:
    """Small context-manager wrapper to keep report styling local."""

    def __enter__(self) -> None:
        self._context = plt.rc_context(
            {
                "figure.facecolor": "white",
                "axes.facecolor": "white",
                "axes.edgecolor": "#AEB8C2",
                "axes.labelcolor": TEXT_COLOR,
                "axes.titlecolor": TEXT_COLOR,
                "text.color": TEXT_COLOR,
                "xtick.color": "#465564",
                "ytick.color": "#465564",
                "font.family": "DejaVu Sans",
                "font.size": 10,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "savefig.facecolor": "white",
            }
        )
        self._context.__enter__()

    def __exit__(self, *args: Any) -> None:
        self._context.__exit__(*args)


def _save_figure(fig: Figure, output_path: str | Path, caption: str) -> PlotArtifact:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return PlotArtifact(path=path, caption=caption)


def _format_number(value: Any, decimals: int) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number:.{decimals}f}" if np.isfinite(number) else "—"


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    def escape(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(escape(value) for value in headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(escape(value) for value in row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def _markdown_bullets(values: Sequence[str]) -> list[str]:
    return [f"- {value}" for value in values]


def _relative_markdown_path(path: Path, base: Path | None) -> str:
    if base is None:
        return path.as_posix()
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        import os

        return Path(os.path.relpath(path.resolve(), base.resolve())).as_posix()


__all__ = [
    "PlotArtifact",
    "ReportArtifacts",
    "build_markdown_report",
    "generate_report_artifacts",
    "model_comparison_markdown",
    "plot_block_permutation_importance",
    "plot_coefficient_stability",
    "plot_model_comparison",
    "plot_model_lot_mae_heatmap",
    "plot_oof_parity_panels",
    "plot_primary_residual_vs_wafer_order",
    "summarize_block_permutation",
    "summarize_coefficient_stability",
    "write_markdown_report",
]
