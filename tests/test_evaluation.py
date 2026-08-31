"""Tests for benchmark models and lot-aware virtual-metrology evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GridSearchCV

from bosch_vm.evaluation import (
    cluster_bootstrap_evaluation,
    collect_outer_fold_coefficients,
    evaluate_predictions,
    extract_outer_fold_coefficients,
    feature_block_permutation_importance,
    regression_metrics,
    squared_error_skill,
)
from bosch_vm.models import FROZEN_SEARCH_GRIDS, MODEL_IDS, build_model_configs


def _regression_data(
    *, n_samples: int = 36, n_features: int = 6, seed: int = 7
) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(size=(n_samples, n_features)),
        columns=[f"x_{index}" for index in range(n_features)],
    )
    y = 1.8 * X["x_0"].to_numpy() - 0.7 * X["x_1"].to_numpy()
    y += rng.normal(scale=0.08, size=n_samples)
    return X, y


def test_eight_model_configs_and_preprocessing_order() -> None:
    configs = build_model_configs(random_state=11)
    assert tuple(configs) == MODEL_IDS
    assert len(configs) == 8
    assert configs["dummy_mean"].feature_view == "context"
    assert configs["context_ridge"].feature_view == "context"
    assert all(
        config.feature_view == "cycle_context"
        for model_id, config in configs.items()
        if model_id not in {"dummy_mean", "context_ridge"}
    )

    for model_id, config in configs.items():
        step_names = tuple(config.estimator.named_steps)
        if model_id == "dummy_mean":
            assert step_names == ("model",)
            continue
        assert step_names[0:2] == ("imputer", "variance")
        if "scaler" in step_names:
            assert step_names.index("variance") < step_names.index("scaler")
        assert step_names[-1] == "model"

    with pytest.raises(TypeError):
        FROZEN_SEARCH_GRIDS["cycle_ridge"]["model__alpha"] = (999.0,)  # type: ignore[index]
    mutable = configs["cycle_ridge"].mutable_param_grid()
    mutable["model__alpha"].append(999.0)
    assert 999.0 not in FROZEN_SEARCH_GRIDS["cycle_ridge"]["model__alpha"]
    assert configs["cycle_ridge"].fresh_estimator() is not configs["cycle_ridge"].estimator


def test_pls_scalar_predict_shape_and_gpr_standard_deviation_interface() -> None:
    X, y = _regression_data()
    X["constant_in_outer_train"] = 4.0
    configs = build_model_configs(random_state=5)

    pls = configs["pls"].fresh_estimator().set_params(model__n_components=2)
    pls.fit(X, y)
    assert pls.predict(X).shape == (len(X),)
    assert pls.named_steps["variance"].get_support().sum() == X.shape[1] - 1

    gpr = configs["pca_gpr"].fresh_estimator().set_params(pca__n_components=2)
    gpr.fit(X, y)
    transformed = gpr[:-1].transform(X)
    mean, standard_deviation = gpr.named_steps["model"].predict(
        transformed, return_std=True
    )
    assert mean.shape == standard_deviation.shape == (len(X),)
    assert np.all(np.isfinite(standard_deviation))
    assert np.all(standard_deviation >= 0.0)


def test_regression_metrics_and_lot_macro_weighting() -> None:
    y_true = np.array([0.0, 1.0, 2.0, 3.0])
    y_pred = np.array([0.0, 2.0, 1.0, 3.0])
    dummy_pred = np.array([0.5, 0.5, 2.5, 2.5])
    lots = np.array(["lot_a", "lot_a", "lot_b", "lot_b"])

    metrics = regression_metrics(y_true, y_pred.reshape(-1, 1))
    assert metrics["mae"] == pytest.approx(0.5)
    assert metrics["rmse"] == pytest.approx(np.sqrt(0.5))
    assert metrics["bias"] == pytest.approx(0.0)
    assert metrics["r2"] == pytest.approx(0.6)
    assert metrics["sse"] == pytest.approx(2.0)
    assert squared_error_skill(y_true, y_pred, dummy_pred) == pytest.approx(-1.0)

    evaluation = evaluate_predictions(
        y_true, y_pred, lots, dummy_pred=dummy_pred
    )
    assert evaluation.overall["n_lots"] == 2
    assert evaluation.overall["squared_error_skill"] == pytest.approx(-1.0)
    assert evaluation.per_lot["lot"].tolist() == ["lot_a", "lot_b"]
    assert evaluation.per_lot["mae"].tolist() == pytest.approx([0.5, 0.5])
    assert evaluation.macro["mae"] == pytest.approx(0.5)
    assert evaluation.macro["squared_error_skill"] == pytest.approx(-1.0)


def test_undefined_r2_and_zero_denominator_skill_are_nan() -> None:
    assert np.isnan(regression_metrics([2.0, 2.0], [2.0, 2.0])["r2"])
    assert np.isnan(squared_error_skill([1.0, 2.0], [0.0, 0.0], [1.0, 2.0]))


def test_cluster_bootstrap_is_deterministic_and_preserves_point_estimates() -> None:
    y_true = np.array([0.0, 0.2, 1.0, 1.3, 2.0, 2.4])
    y_pred = np.array([0.1, 0.1, 1.1, 1.0, 2.2, 2.5])
    dummy_pred = np.array([1.4, 1.4, 1.1, 1.1, 0.5, 0.5])
    lots = np.repeat([1, 2, 3], 2)

    first = cluster_bootstrap_evaluation(
        y_true,
        y_pred,
        lots,
        dummy_pred=dummy_pred,
        n_bootstrap=120,
        random_state=17,
        expected_n_lots=3,
    )
    second = cluster_bootstrap_evaluation(
        y_true,
        y_pred,
        lots,
        dummy_pred=dummy_pred,
        n_bootstrap=120,
        random_state=17,
        expected_n_lots=3,
    )
    pd.testing.assert_frame_equal(first.replicates, second.replicates)
    assert first.replicates.shape == (120, 12)
    assert np.isfinite(first.replicates["macro_lot_mae"]).all()

    evaluation = evaluate_predictions(
        y_true, y_pred, lots, dummy_pred=dummy_pred
    )
    estimates = first.summary.set_index("metric")["estimate"]
    assert estimates["overall_mae"] == pytest.approx(evaluation.overall["mae"])
    assert estimates["macro_lot_mae"] == pytest.approx(evaluation.macro["mae"])
    assert first.summary["n_bootstrap"].eq(120).all()
    assert (first.summary["lower"] <= first.summary["upper"]).all()

    with pytest.raises(ValueError, match="Expected 10 lots"):
        cluster_bootstrap_evaluation(
            y_true, y_pred, lots, n_bootstrap=2
        )


def test_coefficients_align_constant_columns_across_outer_folds() -> None:
    X, y = _regression_data(n_samples=30, n_features=4)
    X.insert(2, "constant_in_fold", 9.0)
    ridge = build_model_configs()["cycle_ridge"].fresh_estimator()
    search = GridSearchCV(
        ridge,
        param_grid={"model__alpha": [0.1, 1.0]},
        scoring="neg_mean_absolute_error",
        cv=3,
    ).fit(X, y)

    coefficients = extract_outer_fold_coefficients(
        search,
        model_id="cycle_ridge",
        outer_lot=4,
        target_names=["mean_si_etch_um"],
    )
    assert coefficients["feature"].tolist() == X.columns.tolist()
    constant = coefficients.loc[coefficients["feature"] == "constant_in_fold"].iloc[0]
    assert constant["selected"] == np.bool_(False)
    assert constant["coefficient"] == pytest.approx(0.0)
    assert set(coefficients["coefficient_scale"]) == {"standardized_x_original_y"}

    collected = collect_outer_fold_coefficients(
        {4: search, 7: search},
        model_id="cycle_ridge",
        target_names=["mean_si_etch_um"],
    )
    assert set(collected["outer_lot"]) == {4, 7}
    assert len(collected) == 2 * X.shape[1]

    unsupported = extract_outer_fold_coefficients(
        build_model_configs()["rbf_svr"].fresh_estimator().fit(X, y),
        model_id="rbf_svr",
        outer_lot=4,
    )
    assert unsupported.empty


def test_pls_coefficient_orientation_matches_selected_feature_count() -> None:
    X, y = _regression_data(n_samples=28, n_features=5)
    X["constant"] = 1.0
    pls = build_model_configs()["pls"].fresh_estimator().set_params(
        model__n_components=2
    )
    pls.fit(X, y)
    coefficients = extract_outer_fold_coefficients(
        pls, model_id="pls", outer_lot="lot_9"
    )
    assert len(coefficients) == X.shape[1]
    assert coefficients["selected"].sum() == X.shape[1] - 1
    assert np.isfinite(coefficients["coefficient"]).all()


def test_feature_block_permutation_finds_signal_without_mutating_input() -> None:
    rng = np.random.default_rng(23)
    n_samples = 90
    X = pd.DataFrame(
        {
            "cycle_signal": rng.normal(size=n_samples),
            "cycle_companion": rng.normal(size=n_samples),
            "context_noise": rng.normal(size=n_samples),
        }
    )
    y = 3.0 * X["cycle_signal"].to_numpy() + rng.normal(scale=0.08, size=n_samples)
    lots = np.repeat(["a", "b", "c"], n_samples // 3)
    original = X.copy(deep=True)
    estimator = build_model_configs()["cycle_ridge"].fresh_estimator().fit(X, y)

    result = feature_block_permutation_importance(
        estimator,
        X,
        y,
        {
            "cycle": ["cycle_signal", "cycle_companion"],
            "context": ["context_noise"],
        },
        n_repeats=20,
        random_state=29,
        groups=lots,
    )
    repeated = feature_block_permutation_importance(
        estimator,
        X,
        y,
        {
            "cycle": ["cycle_signal", "cycle_companion"],
            "context": ["context_noise"],
        },
        n_repeats=20,
        random_state=29,
        groups=lots,
    )
    pd.testing.assert_frame_equal(X, original)
    np.testing.assert_allclose(result.importances, repeated.importances)
    summary = result.to_frame().set_index("block")
    assert summary.loc["cycle", "importance_mean"] > 2.0
    assert summary.loc["cycle", "importance_mean"] > (
        10.0 * abs(summary.loc["context", "importance_mean"])
    )
    assert not result.importances.flags.writeable
    assert result.to_long_frame(model_id="cycle_ridge", outer_lot=2).shape[0] == 40


def test_within_lot_permutation_does_not_shuffle_a_lot_constant_feature() -> None:
    lots = np.repeat([0, 1, 2], 8)
    X = pd.DataFrame(
        {
            "lot_constant": lots.astype(float),
            "noise": np.linspace(-1.0, 1.0, lots.size),
        }
    )
    y = lots.astype(float)
    estimator = build_model_configs()["cycle_ridge"].fresh_estimator().fit(X, y)
    result = feature_block_permutation_importance(
        estimator,
        X,
        y,
        {"context": ["lot_constant"]},
        groups=lots,
        n_repeats=5,
    )
    np.testing.assert_allclose(result.importances, 0.0, atol=1e-15)
