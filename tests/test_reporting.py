"""Smoke tests for outer-OOF report figures and Markdown helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bosch_vm.reporting import (
    generate_report_artifacts,
    model_comparison_markdown,
    plot_oof_parity_panels,
    summarize_block_permutation,
    summarize_coefficient_stability,
)


def _synthetic_reporting_inputs() -> tuple[pd.DataFrame, ...]:
    rng = np.random.default_rng(20260830)
    models = ["train_mean_dummy", "cycle_ridge", "extra_trees"]
    rows: list[dict[str, float | int | str]] = []
    for model_index, model_id in enumerate(models):
        scale = [0.31, 0.17, 0.21][model_index]
        for lot in range(1, 11):
            for wafer_order in range(1, 5):
                target = 43.4 + 0.07 * lot + 0.018 * wafer_order
                error = rng.normal(0.015 * (model_index - 1), scale)
                rows.append(
                    {
                        "experiment_key": f"lot{lot:02d}_w{wafer_order:02d}",
                        "model_id": model_id,
                        "lot_number": lot,
                        "wafer_order": wafer_order,
                        "outer_fold": lot,
                        "y_true": target,
                        "y_pred": target + error,
                        "residual": error,
                    }
                )
    oof = pd.DataFrame(rows)

    comparison_rows = []
    bootstrap_rows = []
    for model_id, group in oof.groupby("model_id", sort=False):
        error = group["y_pred"] - group["y_true"]
        lot_mae = error.abs().groupby(group["lot_number"]).mean()
        rmse = float(np.sqrt(np.mean(np.square(error))))
        denominator = np.square(group["y_true"] - group["y_true"].mean()).sum()
        r2 = float(1 - np.square(error).sum() / denominator)
        macro = float(lot_mae.mean())
        comparison_rows.append(
            {
                "model_id": model_id,
                "macro_lot_mae_um": macro,
                "pooled_rmse_um": rmse,
                "oof_r2": r2,
                "sse_skill_vs_dummy": 0.0 if model_id == "train_mean_dummy" else 0.3,
            }
        )
        bootstrap_rows.append(
            {
                "model_id": model_id,
                "metric": "macro_lot_mae_um",
                "estimate": macro,
                "lower_95": max(0.0, macro - 0.035),
                "upper_95": macro + 0.045,
            }
        )

    coefficient_rows = []
    for fold in range(1, 11):
        for index, feature in enumerate(
            [
                "gas__level",
                "pressure__drift",
                "source_rf__delta",
                "platen_rf__mad",
                "timing__cycle_period",
                "context__wafer_order",
            ]
        ):
            coefficient_rows.append(
                {
                    "model_id": "cycle_ridge",
                    "outer_fold": fold,
                    "feature": feature,
                    "standardized_coefficient": (-1) ** index
                    * (0.35 - 0.035 * index + rng.normal(0, 0.025)),
                }
            )

    permutation_rows = []
    for fold in range(1, 11):
        for index, block in enumerate(
            ["gas", "pressure_helium", "platen_rf", "source_rf", "timing", "context"]
        ):
            permutation_rows.append(
                {
                    "model_id": "cycle_ridge",
                    "outer_fold": fold,
                    "block": block,
                    "delta_mae_um": 0.12 - 0.016 * index + rng.normal(0, 0.012),
                }
            )

    return (
        pd.DataFrame(comparison_rows),
        oof,
        pd.DataFrame(bootstrap_rows),
        pd.DataFrame(coefficient_rows),
        pd.DataFrame(permutation_rows),
    )


def test_generate_report_artifacts_smoke(tmp_path: Path) -> None:
    comparison, oof, bootstrap, coefficients, permutation = _synthetic_reporting_inputs()
    artifacts = generate_report_artifacts(
        comparison,
        oof,
        tmp_path / "report",
        bootstrap_intervals=bootstrap,
        coefficients=coefficients,
        permutation_importance=permutation,
    )

    assert artifacts.report_path.exists()
    assert set(artifacts.figures) == {
        "model_comparison",
        "oof_parity_panels",
        "model_lot_mae_heatmap",
        "primary_residual_vs_wafer_order",
        "coefficient_stability",
        "block_permutation_importance",
    }
    for artifact in artifacts.figures.values():
        assert artifact.path.exists()
        assert artifact.path.stat().st_size > 10_000
        assert artifact.caption
        assert "All performance values" not in artifact.caption

    report = artifacts.report_path.read_text(encoding="utf-8")
    assert "outer-fold OOF" in report
    assert "Cycle-feature Ridge has" in report
    assert "prediction minus measured" in report
    assert "predictive association" in report
    assert "model families were compared" in report


def test_comparison_aliases_and_summaries() -> None:
    comparison, _, bootstrap, coefficients, permutation = _synthetic_reporting_inputs()
    aliased = comparison.rename(columns={"pooled_rmse_um": "rmse_um", "oof_r2": "r2_oof"})
    table = model_comparison_markdown(aliased, bootstrap_intervals=bootstrap)
    assert "Macro-lot MAE" in table
    assert "95% CI" in table
    assert "Cycle-feature Ridge" in table
    assert "Role" not in table

    coefficient_summary = summarize_coefficient_stability(coefficients, top_n=3)
    assert len(coefficient_summary) == 3
    assert coefficient_summary["outer_folds"].eq(10).all()
    assert coefficient_summary["sign_consistency"].between(0, 1).all()

    permutation_summary = summarize_block_permutation(permutation)
    assert len(permutation_summary) == 6
    assert permutation_summary["outer_folds"].eq(10).all()


def test_prediction_plots_reject_non_oof_rows(tmp_path: Path) -> None:
    _, oof, _, _, _ = _synthetic_reporting_inputs()
    with pytest.raises(ValueError, match="outer_fold"):
        plot_oof_parity_panels(oof.drop(columns="outer_fold"), tmp_path / "bad.png")


def test_duplicate_wafer_predictions_are_rejected(tmp_path: Path) -> None:
    _, oof, _, _, _ = _synthetic_reporting_inputs()
    duplicated = pd.concat([oof, oof.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="exactly once"):
        plot_oof_parity_panels(duplicated, tmp_path / "duplicate.png")


def test_reversed_residual_convention_is_rejected(tmp_path: Path) -> None:
    _, oof, _, _, _ = _synthetic_reporting_inputs()
    reversed_sign = oof.copy()
    reversed_sign["residual"] = reversed_sign["y_true"] - reversed_sign["y_pred"]
    with pytest.raises(ValueError, match="prediction - measured"):
        plot_oof_parity_panels(reversed_sign, tmp_path / "reversed.png")


def test_generated_captions_are_figure_specific(tmp_path: Path) -> None:
    comparison, oof, bootstrap, coefficients, permutation = _synthetic_reporting_inputs()
    artifacts = generate_report_artifacts(
        comparison,
        oof,
        tmp_path / "report",
        bootstrap_intervals=bootstrap,
        coefficients=coefficients,
        permutation_importance=permutation,
    )
    captions = [artifact.caption for artifact in artifacts.figures.values()]
    assert len(captions) == len(set(captions))
    assert any("lot-cluster bootstrap" in caption for caption in captions)
    assert any("rather than causal effects" in caption for caption in captions)
