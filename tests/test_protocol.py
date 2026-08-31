from __future__ import annotations

import json

import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from bosch_vm.experiment import (
    ExperimentConfig,
    ProtocolModelSpec,
    prepare,
    run,
)


def _synthetic_wafer_table() -> pd.DataFrame:
    rows = []
    for lot in range(1, 11):
        for wafer in range(1, 3):
            x_level = lot / 10 + wafer / 100
            x_drift = wafer - 1.5
            rows.append(
                {
                    "experiment_key": f"lot{lot:02d}_wafer{wafer:02d}",
                    "lot_number": lot,
                    "wafer_number": wafer,
                    "mean_si_etch_um": 43.0 + 0.4 * x_level - 0.1 * x_drift,
                    "x_level": x_level,
                    "x_drift": x_drift,
                }
            )
    return pd.DataFrame(rows)


def _model_specs() -> list[ProtocolModelSpec]:
    return [
        ProtocolModelSpec(
            model_id="cycle_ridge",
            role="primary",
            feature_view=("x_level", "x_drift"),
            estimator=Pipeline([("scale", StandardScaler()), ("ridge", Ridge())]),
            param_grid={"ridge__alpha": [0.1, 1.0]},
        ),
        ProtocolModelSpec(
            model_id="train_mean_dummy",
            role="baseline",
            feature_view="none",
            estimator=DummyRegressor(strategy="mean"),
            param_grid={},
        ),
    ]


def test_nested_logo_produces_exactly_one_oof_prediction_per_wafer(tmp_path):
    table = _synthetic_wafer_table()
    config = ExperimentConfig(
        prepared_dir=tmp_path / "prepared",
        output_dir=tmp_path / "results",
        experiment_dir=tmp_path / "experiment",
        wafer_column="wafer_number",
        bootstrap_replicates=25,
        permutation_repetitions=2,
        n_jobs=1,
    )

    artifacts = run(config, feature_table=table, model_specs=_model_specs())

    split_manifest = pd.read_csv(artifacts.split_path)
    assert len(split_manifest) == len(table)
    assert split_manifest["experiment_key"].is_unique
    assert split_manifest["outer_fold"].nunique() == 10
    assert (split_manifest.groupby("outer_fold")["lot_number"].nunique() == 1).all()
    assert (split_manifest.groupby("outer_fold")["outer_test_lot"].nunique() == 1).all()

    for model_id in ("cycle_ridge", "train_mean_dummy"):
        model_dir = config.output_dir / "models" / model_id
        oof = pd.read_csv(model_dir / "oof_predictions.csv")
        assert len(oof) == len(table)
        assert oof["experiment_key"].is_unique
        assert set(oof["experiment_key"]) == set(table["experiment_key"])
        assert (oof["oof_occurrence"] == 1).all()
        assert oof.groupby("outer_fold")["lot_number"].nunique().eq(1).all()
        assert len(list((model_dir / "folds").glob("lot_*/model.joblib"))) == 10
        assert (model_dir / "final_refit" / "model.joblib").exists()

    ridge_inner = pd.read_csv(config.output_dir / "models" / "cycle_ridge" / "inner_scores.csv")
    assert ridge_inner["outer_fold"].nunique() == 10
    assert set(ridge_inner["n_inner_splits"]) == {9}
    assert len(ridge_inner) == 20  # two alpha candidates for each outer fold
    split_score_columns = [
        column
        for column in ridge_inner.columns
        if column.startswith("split") and column.endswith("_test_score")
    ]
    assert len(split_score_columns) == 9

    comparison = pd.read_csv(artifacts.comparison_path)
    assert set(comparison["model_id"]) == {"cycle_ridge", "train_mean_dummy"}
    assert comparison.loc[comparison["model_id"] == "cycle_ridge", "role"].item() == "primary"
    assert "delta_macro_lot_mae_vs_primary_um" in comparison

    all_oof = pd.read_csv(config.output_dir / "all_oof_predictions.csv")
    assert len(all_oof) == 2 * len(table)
    assert set(all_oof["model_id"]) == {"cycle_ridge", "train_mean_dummy"}
    all_per_lot = pd.read_csv(config.output_dir / "all_metrics_by_lot.csv")
    assert len(all_per_lot) == 20
    coefficients = pd.read_csv(
        config.output_dir / "models" / "cycle_ridge" / "outer_coefficients.csv"
    )
    assert coefficients["outer_lot"].nunique() == 10
    permutation = pd.read_csv(
        config.output_dir / "models" / "cycle_ridge" / "heldout_block_permutation.csv"
    )
    assert permutation["outer_lot"].nunique() == 10

    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["protocol"]["outer_cv"] == "LeaveOneGroupOut(lot)"
    assert manifest["protocol"]["random_sample_split"] is False
    assert manifest["counts"]["outer_predictions_per_model"] == len(table)


def test_prepare_persists_one_row_per_wafer_contract(tmp_path):
    feature_table = _synthetic_wafer_table()
    cohort = feature_table[["experiment_key", "lot_number", "wafer_number"]].copy()
    config = ExperimentConfig(prepared_dir=tmp_path / "prepared", wafer_column="wafer_number")

    artifacts = prepare(
        config,
        cohort=cohort,
        feature_table=feature_table,
        qa=pd.DataFrame({"check": ["synthetic"], "passed": [True]}),
        metadata={"feature_views": {"cycle": ["x_level", "x_drift"]}},
    )

    saved = pd.read_csv(artifacts.feature_table_path)
    assert len(saved) == len(feature_table)
    assert saved["experiment_key"].is_unique
    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
    assert metadata["n_wafers"] == 20
    assert metadata["n_lots"] == 10
    assert metadata["feature_views"]["cycle"] == ["x_level", "x_drift"]
    all_predictors = pd.read_csv(artifacts.all_predictors_path)
    assert set(all_predictors.columns) == {"experiment_key", "x_level", "x_drift"}


def test_duplicate_wafer_rows_are_rejected_before_any_split(tmp_path):
    table = _synthetic_wafer_table()
    duplicate = pd.concat([table, table.iloc[[0]]], ignore_index=True)
    config = ExperimentConfig(
        output_dir=tmp_path / "results",
        experiment_dir=tmp_path / "experiment",
        wafer_column="wafer_number",
        bootstrap_replicates=0,
        permutation_repetitions=0,
    )

    with pytest.raises(ValueError, match="one row per wafer"):
        run(config, feature_table=duplicate, model_specs=_model_specs())
