"""Pure, lot-aware evaluation utilities for scalar virtual metrology.

The functions in this module consume predictions that have already been produced by an
outer validation loop.  They do not fit models, choose hyperparameters, or split data.  This
separation makes it harder to accidentally evaluate on data used for model selection.

All bootstrap resampling is at the lot level.  Feature-block permutation is intentionally an
estimator-scoring primitive: callers remain responsible for passing only held-out rows.
Permutation importance and coefficients describe predictive association, not causality.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import check_scoring
from sklearn.pipeline import Pipeline

DEFAULT_RANDOM_STATE = 20260830

MetricValue = float | int
UnsupportedPolicy = Literal["empty", "raise"]

_COEFFICIENT_COLUMNS = (
    "model_id",
    "outer_lot",
    "target",
    "feature",
    "coefficient",
    "selected",
    "coefficient_scale",
)


def _as_numeric_vector(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    """Coerce a scalar-response vector, accepting sklearn's ``(n, 1)`` output."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 2 and 1 in array.shape:
        array = array.reshape(-1)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _as_group_vector(values: ArrayLike, *, n_samples: int) -> NDArray[Any]:
    array = np.asarray(values, dtype=object)
    if array.ndim == 2 and 1 in array.shape:
        array = array.reshape(-1)
    if array.ndim != 1 or array.size != n_samples:
        raise ValueError("groups must be one-dimensional and match the number of samples")
    if bool(np.asarray(pd.isna(array)).any()):
        raise ValueError("groups must not contain missing values")
    return array


def _validate_prediction_vectors(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    dummy_pred: ArrayLike | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None]:
    truth = _as_numeric_vector(y_true, name="y_true")
    prediction = _as_numeric_vector(y_pred, name="y_pred")
    if prediction.size != truth.size:
        raise ValueError("y_pred must match the number of y_true samples")
    if dummy_pred is None:
        return truth, prediction, None
    dummy = _as_numeric_vector(dummy_pred, name="dummy_pred")
    if dummy.size != truth.size:
        raise ValueError("dummy_pred must match the number of y_true samples")
    return truth, prediction, dummy


def squared_error_skill(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    dummy_pred: ArrayLike,
) -> float:
    """Return ``1 - SSE(model) / SSE(dummy)``.

    The result is ``NaN`` when the supplied dummy has zero squared error, because skill is
    then mathematically undefined.  Negative values are valid and mean the model is worse
    than the fold-training-mean baseline under squared error.
    """

    truth, prediction, dummy = _validate_prediction_vectors(y_true, y_pred, dummy_pred)
    assert dummy is not None
    model_sse = float(np.square(prediction - truth).sum())
    dummy_sse = float(np.square(dummy - truth).sum())
    if dummy_sse == 0.0:
        return float("nan")
    return 1.0 - model_sse / dummy_sse


def regression_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, MetricValue]:
    """Compute scalar-response metrics with an explicit residual sign convention.

    ``bias`` is mean ``prediction - measurement``.  R-squared is returned as ``NaN`` for a
    single observation or a constant target, where it is not mathematically defined.
    """

    truth, prediction, _ = _validate_prediction_vectors(y_true, y_pred)
    residual = prediction - truth
    absolute_error = np.abs(residual)
    squared_error = np.square(residual)
    target_sst = float(np.square(truth - truth.mean()).sum())
    sse = float(squared_error.sum())
    r2 = float("nan") if target_sst == 0.0 else 1.0 - sse / target_sst
    return {
        "n_samples": int(truth.size),
        "mae": float(absolute_error.mean()),
        "mse": float(squared_error.mean()),
        "rmse": float(np.sqrt(squared_error.mean())),
        "median_absolute_error": float(np.median(absolute_error)),
        "bias": float(residual.mean()),
        "r2": r2,
        "sse": sse,
    }


@dataclass(frozen=True)
class RegressionEvaluation:
    """Overall, per-lot, and equal-lot-weighted metrics for outer predictions."""

    overall: Mapping[str, MetricValue]
    per_lot: pd.DataFrame
    macro: Mapping[str, MetricValue]


def evaluate_predictions(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    lots: ArrayLike,
    *,
    dummy_pred: ArrayLike | None = None,
) -> RegressionEvaluation:
    """Evaluate one complete set of outer out-of-fold predictions.

    Rows are pooled for ``overall`` metrics.  ``macro`` metrics first score each lot and
    then give every lot equal weight.  If supplied, ``dummy_pred`` must contain the baseline
    prediction appropriate to each row's outer fold, not a global mean fitted on all rows.
    """

    truth, prediction, dummy = _validate_prediction_vectors(y_true, y_pred, dummy_pred)
    group_array = _as_group_vector(lots, n_samples=truth.size)
    unique_lots = pd.unique(group_array)

    overall = regression_metrics(truth, prediction)
    overall["n_lots"] = int(unique_lots.size)
    if dummy is not None:
        overall["dummy_sse"] = float(np.square(dummy - truth).sum())
        overall["squared_error_skill"] = squared_error_skill(truth, prediction, dummy)

    rows: list[dict[str, Any]] = []
    for lot in unique_lots:
        mask = group_array == lot
        metrics = regression_metrics(truth[mask], prediction[mask])
        row: dict[str, Any] = {"lot": lot, **metrics}
        if dummy is not None:
            row["dummy_mse"] = float(np.square(dummy[mask] - truth[mask]).mean())
            row["dummy_sse"] = float(np.square(dummy[mask] - truth[mask]).sum())
            row["squared_error_skill"] = squared_error_skill(
                truth[mask], prediction[mask], dummy[mask]
            )
        rows.append(row)
    per_lot = pd.DataFrame(rows)

    macro: dict[str, MetricValue] = {
        "n_lots": int(unique_lots.size),
        "mae": float(per_lot["mae"].mean()),
        "rmse": float(per_lot["rmse"].mean()),
        "median_absolute_error": float(per_lot["median_absolute_error"].mean()),
        "bias": float(per_lot["bias"].mean()),
        "worst_lot_mae": float(per_lot["mae"].max()),
        "best_lot_mae": float(per_lot["mae"].min()),
    }
    if dummy is not None:
        dummy_macro_mse = float(per_lot["dummy_mse"].mean())
        macro["dummy_mse"] = dummy_macro_mse
        macro["squared_error_skill"] = (
            float("nan")
            if dummy_macro_mse == 0.0
            else 1.0 - float(per_lot["mse"].mean()) / dummy_macro_mse
        )
    return RegressionEvaluation(overall=overall, per_lot=per_lot, macro=macro)


@dataclass(frozen=True)
class ClusterBootstrapResult:
    """Point estimates, percentile intervals, and bootstrap replicates."""

    summary: pd.DataFrame
    replicates: pd.DataFrame


def _bootstrap_point_estimates(
    evaluation: RegressionEvaluation,
) -> dict[str, float]:
    estimates = {
        "overall_mae": float(evaluation.overall["mae"]),
        "overall_rmse": float(evaluation.overall["rmse"]),
        "overall_median_absolute_error": float(
            evaluation.overall["median_absolute_error"]
        ),
        "overall_bias": float(evaluation.overall["bias"]),
        "overall_r2": float(evaluation.overall["r2"]),
        "macro_lot_mae": float(evaluation.macro["mae"]),
        "macro_lot_rmse": float(evaluation.macro["rmse"]),
        "macro_lot_median_absolute_error": float(
            evaluation.macro["median_absolute_error"]
        ),
        "macro_lot_bias": float(evaluation.macro["bias"]),
        "worst_lot_mae": float(evaluation.macro["worst_lot_mae"]),
    }
    if "squared_error_skill" in evaluation.overall:
        estimates["overall_squared_error_skill"] = float(
            evaluation.overall["squared_error_skill"]
        )
        estimates["macro_lot_squared_error_skill"] = float(
            evaluation.macro["squared_error_skill"]
        )
    return estimates


def _finite_percentile_summary(
    values: NDArray[np.float64], *, confidence_level: float
) -> tuple[float, float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")
    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(finite, (alpha / 2.0, 1.0 - alpha / 2.0))
    standard_deviation = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
    return float(lower), float(upper), standard_deviation


def cluster_bootstrap_evaluation(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    lots: ArrayLike,
    *,
    dummy_pred: ArrayLike | None = None,
    n_bootstrap: int = 10_000,
    confidence_level: float = 0.95,
    random_state: int = DEFAULT_RANDOM_STATE,
    expected_n_lots: int | None = 10,
) -> ClusterBootstrapResult:
    """Bootstrap complete lots and return percentile intervals for aggregate metrics.

    Each replicate draws ``n_lots`` lots with replacement and retains every wafer from each
    selected lot.  Repeated selections count as separate clusters for equal-lot macro
    metrics.  ``expected_n_lots=10`` guards the reference public-data experiment; pass ``None``
    for a different cohort or an explicit smaller value in unit tests.
    """

    if not isinstance(n_bootstrap, int) or isinstance(n_bootstrap, bool) or n_bootstrap < 1:
        raise ValueError("n_bootstrap must be a positive integer")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be strictly between zero and one")

    truth, prediction, dummy = _validate_prediction_vectors(y_true, y_pred, dummy_pred)
    group_array = _as_group_vector(lots, n_samples=truth.size)
    unique_lots = pd.unique(group_array)
    n_lots = int(unique_lots.size)
    if expected_n_lots is not None and n_lots != expected_n_lots:
        raise ValueError(f"Expected {expected_n_lots} lots, found {n_lots}")
    if n_lots < 2:
        raise ValueError("At least two lots are required for a cluster bootstrap")

    lot_indices = [np.flatnonzero(group_array == lot) for lot in unique_lots]
    lot_n = np.asarray([indices.size for indices in lot_indices], dtype=np.float64)
    residual = prediction - truth
    absolute_error = np.abs(residual)
    squared_error = np.square(residual)
    lot_absolute_error = np.asarray(
        [absolute_error[indices].sum() for indices in lot_indices], dtype=np.float64
    )
    lot_squared_error = np.asarray(
        [squared_error[indices].sum() for indices in lot_indices], dtype=np.float64
    )
    lot_residual = np.asarray(
        [residual[indices].sum() for indices in lot_indices], dtype=np.float64
    )
    lot_target_sum = np.asarray(
        [truth[indices].sum() for indices in lot_indices], dtype=np.float64
    )
    lot_target_square_sum = np.asarray(
        [np.square(truth[indices]).sum() for indices in lot_indices], dtype=np.float64
    )
    lot_mae = lot_absolute_error / lot_n
    lot_mse = lot_squared_error / lot_n
    lot_rmse = np.sqrt(lot_mse)
    lot_bias = lot_residual / lot_n
    lot_median_absolute_error = np.asarray(
        [np.median(absolute_error[indices]) for indices in lot_indices], dtype=np.float64
    )
    lot_dummy_squared_error = (
        None
        if dummy is None
        else np.asarray(
            [np.square(dummy[indices] - truth[indices]).sum() for indices in lot_indices],
            dtype=np.float64,
        )
    )

    metric_names = list(
        _bootstrap_point_estimates(
            evaluate_predictions(truth, prediction, group_array, dummy_pred=dummy)
        )
    )
    replicate_values = np.empty((n_bootstrap, len(metric_names)), dtype=np.float64)
    metric_index = {name: index for index, name in enumerate(metric_names)}
    rng = np.random.default_rng(random_state)

    for bootstrap_index in range(n_bootstrap):
        draws = rng.integers(0, n_lots, size=n_lots)
        sampled_n = float(lot_n[draws].sum())
        sampled_sse = float(lot_squared_error[draws].sum())
        sampled_y_sum = float(lot_target_sum[draws].sum())
        sampled_sst = float(
            lot_target_square_sum[draws].sum() - sampled_y_sum**2 / sampled_n
        )
        sampled_absolute_error = np.concatenate(
            [absolute_error[lot_indices[index]] for index in draws]
        )

        values = {
            "overall_mae": float(lot_absolute_error[draws].sum() / sampled_n),
            "overall_rmse": float(np.sqrt(sampled_sse / sampled_n)),
            "overall_median_absolute_error": float(np.median(sampled_absolute_error)),
            "overall_bias": float(lot_residual[draws].sum() / sampled_n),
            "overall_r2": (
                float("nan") if sampled_sst <= 0.0 else 1.0 - sampled_sse / sampled_sst
            ),
            "macro_lot_mae": float(lot_mae[draws].mean()),
            "macro_lot_rmse": float(lot_rmse[draws].mean()),
            "macro_lot_median_absolute_error": float(
                lot_median_absolute_error[draws].mean()
            ),
            "macro_lot_bias": float(lot_bias[draws].mean()),
            "worst_lot_mae": float(lot_mae[draws].max()),
        }
        if lot_dummy_squared_error is not None:
            sampled_dummy_sse = float(lot_dummy_squared_error[draws].sum())
            sampled_dummy_macro_mse = float(
                (lot_dummy_squared_error[draws] / lot_n[draws]).mean()
            )
            values["overall_squared_error_skill"] = (
                float("nan")
                if sampled_dummy_sse == 0.0
                else 1.0 - sampled_sse / sampled_dummy_sse
            )
            values["macro_lot_squared_error_skill"] = (
                float("nan")
                if sampled_dummy_macro_mse == 0.0
                else 1.0 - float(lot_mse[draws].mean()) / sampled_dummy_macro_mse
            )
        for metric_name, value in values.items():
            replicate_values[bootstrap_index, metric_index[metric_name]] = value

    replicates = pd.DataFrame(replicate_values, columns=metric_names)
    point_estimates = _bootstrap_point_estimates(
        evaluate_predictions(truth, prediction, group_array, dummy_pred=dummy)
    )
    summary_rows: list[dict[str, MetricValue | str]] = []
    for metric_name in metric_names:
        lower, upper, standard_deviation = _finite_percentile_summary(
            replicates[metric_name].to_numpy(dtype=np.float64),
            confidence_level=confidence_level,
        )
        summary_rows.append(
            {
                "metric": metric_name,
                "estimate": point_estimates[metric_name],
                "lower": lower,
                "upper": upper,
                "bootstrap_std": standard_deviation,
                "n_bootstrap": n_bootstrap,
                "confidence_level": confidence_level,
            }
        )
    return ClusterBootstrapResult(
        summary=pd.DataFrame(summary_rows),
        replicates=replicates,
    )


def _unwrap_estimator(estimator: BaseEstimator) -> BaseEstimator:
    """Unwrap a fitted search object without importing an orchestration class."""

    return getattr(estimator, "best_estimator_", estimator)


def _empty_coefficient_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=list(_COEFFICIENT_COLUMNS))


def _coefficient_matrix(
    coefficient: ArrayLike, *, n_features: int
) -> NDArray[np.float64]:
    values = np.asarray(coefficient, dtype=np.float64)
    if values.ndim == 1:
        if values.size != n_features:
            raise ValueError("Estimator coefficient count does not match selected features")
        return values.reshape(1, -1)
    if values.ndim != 2:
        raise ValueError("Estimator coefficients must be one- or two-dimensional")
    if values.shape[1] == n_features:
        return values
    if values.shape[0] == n_features:
        return values.T
    raise ValueError("Estimator coefficient shape does not match selected features")


def extract_outer_fold_coefficients(
    fitted_estimator: BaseEstimator,
    *,
    model_id: str,
    outer_lot: Any,
    feature_names: Sequence[str] | None = None,
    target_names: Sequence[str] | None = None,
    on_unsupported: UnsupportedPolicy = "empty",
) -> pd.DataFrame:
    """Extract aligned Ridge/PLS/Elastic-Net coefficients from one fitted outer fold.

    Constant columns removed by ``VarianceThreshold`` are restored with coefficient zero and
    ``selected=False`` so folds can be compared on a common schema.  Coefficients are on the
    standardized-X/original-y scale used by these pipelines; they are associations only.
    """

    estimator = _unwrap_estimator(fitted_estimator)
    if not isinstance(estimator, Pipeline):
        raise TypeError("fitted_estimator must resolve to a fitted sklearn Pipeline")
    final_model = estimator.steps[-1][1]
    supported = isinstance(final_model, Ridge | ElasticNet | PLSRegression)
    if not supported or not hasattr(final_model, "coef_"):
        if on_unsupported == "empty":
            return _empty_coefficient_frame()
        if on_unsupported == "raise":
            raise TypeError(
                "Coefficient extraction supports fitted Ridge, ElasticNet, and PLSRegression"
            )
        raise ValueError("on_unsupported must be either 'empty' or 'raise'")

    if feature_names is None:
        fitted_names = getattr(estimator, "feature_names_in_", None)
        if fitted_names is not None:
            original_names = np.asarray(fitted_names, dtype=object)
        else:
            n_original = int(estimator.n_features_in_)
            original_names = np.asarray(
                [f"feature_{index}" for index in range(n_original)], dtype=object
            )
    else:
        original_names = np.asarray(tuple(feature_names), dtype=object)
        expected = int(estimator.n_features_in_)
        if original_names.size != expected:
            raise ValueError(
                f"feature_names contains {original_names.size} names; expected {expected}"
            )

    variance = estimator.named_steps.get("variance")
    if variance is None:
        support = np.ones(original_names.size, dtype=bool)
    else:
        support = np.asarray(variance.get_support(), dtype=bool)
        if support.size != original_names.size:
            raise ValueError("VarianceThreshold support does not match the original schema")
    selected_names = original_names[support]
    matrix = _coefficient_matrix(final_model.coef_, n_features=selected_names.size)

    if target_names is None:
        resolved_targets = tuple(
            "target" if matrix.shape[0] == 1 else f"target_{index}"
            for index in range(matrix.shape[0])
        )
    else:
        resolved_targets = tuple(target_names)
        if len(resolved_targets) != matrix.shape[0]:
            raise ValueError(
                f"target_names contains {len(resolved_targets)} names; "
                f"expected {matrix.shape[0]}"
            )

    coefficient_scale = (
        "standardized_x_original_y"
        if "scaler" in estimator.named_steps
        else "original_x_original_y"
    )
    rows: list[dict[str, Any]] = []
    for target_index, target_name in enumerate(resolved_targets):
        aligned = np.zeros(original_names.size, dtype=np.float64)
        aligned[support] = matrix[target_index]
        for feature_index, feature_name in enumerate(original_names):
            rows.append(
                {
                    "model_id": model_id,
                    "outer_lot": outer_lot,
                    "target": target_name,
                    "feature": str(feature_name),
                    "coefficient": float(aligned[feature_index]),
                    "selected": bool(support[feature_index]),
                    "coefficient_scale": coefficient_scale,
                }
            )
    return pd.DataFrame(rows, columns=list(_COEFFICIENT_COLUMNS))


def collect_outer_fold_coefficients(
    fitted_estimators: Mapping[Any, BaseEstimator],
    *,
    model_id: str,
    feature_names: Sequence[str] | Mapping[Any, Sequence[str]] | None = None,
    target_names: Sequence[str] | None = None,
    on_unsupported: UnsupportedPolicy = "empty",
) -> pd.DataFrame:
    """Concatenate coefficient tables from fitted outer folds."""

    frames: list[pd.DataFrame] = []
    per_fold_names = isinstance(feature_names, Mapping)
    for outer_lot, estimator in fitted_estimators.items():
        names = feature_names.get(outer_lot) if per_fold_names else feature_names
        frames.append(
            extract_outer_fold_coefficients(
                estimator,
                model_id=model_id,
                outer_lot=outer_lot,
                feature_names=names,
                target_names=target_names,
                on_unsupported=on_unsupported,
            )
        )
    nonempty = [frame for frame in frames if not frame.empty]
    if not nonempty:
        return _empty_coefficient_frame()
    return pd.concat(nonempty, ignore_index=True)


@dataclass(frozen=True)
class BlockPermutationResult:
    """Held-out feature-block permutation scores for one fitted estimator."""

    baseline_score: float
    block_names: tuple[str, ...]
    importances: NDArray[np.float64]
    scoring_name: str

    def __post_init__(self) -> None:
        values = np.array(self.importances, dtype=np.float64, copy=True)
        if values.ndim != 2 or values.shape[0] != len(self.block_names):
            raise ValueError("importances must have shape (n_blocks, n_repeats)")
        values.setflags(write=False)
        object.__setattr__(self, "importances", values)

    def to_frame(self) -> pd.DataFrame:
        """Return one summary row per feature block."""

        n_repeats = self.importances.shape[1]
        standard_deviation = self.importances.std(axis=1, ddof=1 if n_repeats > 1 else 0)
        return pd.DataFrame(
            {
                "block": self.block_names,
                "importance_mean": self.importances.mean(axis=1),
                "importance_std": standard_deviation,
                "baseline_score": self.baseline_score,
                "n_repeats": n_repeats,
                "scoring": self.scoring_name,
            }
        )

    def to_long_frame(
        self, *, model_id: str | None = None, outer_lot: Any | None = None
    ) -> pd.DataFrame:
        """Return repeat-level importances suitable for aggregation across outer folds."""

        block = np.repeat(np.asarray(self.block_names, dtype=object), self.importances.shape[1])
        repeat = np.tile(np.arange(self.importances.shape[1]), len(self.block_names))
        frame = pd.DataFrame(
            {
                "block": block,
                "repeat": repeat,
                "importance": self.importances.reshape(-1),
                "baseline_score": self.baseline_score,
                "scoring": self.scoring_name,
            }
        )
        if outer_lot is not None:
            frame.insert(0, "outer_lot", outer_lot)
        if model_id is not None:
            frame.insert(0, "model_id", model_id)
        return frame


def _resolve_feature_blocks(
    X: Any,
    feature_blocks: Mapping[str, Sequence[str | int]],
    *,
    feature_names: Sequence[str] | None,
) -> tuple[tuple[str, ...], tuple[NDArray[np.int64], ...]]:
    if not feature_blocks:
        raise ValueError("feature_blocks must contain at least one block")
    n_features = int(X.shape[1])
    if isinstance(X, pd.DataFrame):
        resolved_names = tuple(str(name) for name in X.columns)
    elif feature_names is None:
        resolved_names = tuple(f"feature_{index}" for index in range(n_features))
    else:
        resolved_names = tuple(feature_names)
        if len(resolved_names) != n_features:
            raise ValueError("feature_names must match the number of X columns")
    name_to_position = {name: index for index, name in enumerate(resolved_names)}
    if len(name_to_position) != n_features:
        raise ValueError("Feature names must be unique")

    block_names: list[str] = []
    block_positions: list[NDArray[np.int64]] = []
    used_positions: set[int] = set()
    for block_name, selectors in feature_blocks.items():
        if not selectors:
            raise ValueError(f"Feature block {block_name!r} is empty")
        positions: list[int] = []
        for selector in selectors:
            if isinstance(selector, int | np.integer):
                position = int(selector)
                if position < 0 or position >= n_features:
                    raise ValueError(
                        f"Feature position {position} in block {block_name!r} is out of range"
                    )
            else:
                try:
                    position = name_to_position[str(selector)]
                except KeyError as exc:
                    raise ValueError(
                        f"Unknown feature {selector!r} in block {block_name!r}"
                    ) from exc
            if position in positions:
                raise ValueError(f"Feature {selector!r} is repeated in block {block_name!r}")
            if position in used_positions:
                raise ValueError(
                    f"Feature {selector!r} appears in more than one feature block"
                )
            positions.append(position)
        used_positions.update(positions)
        block_names.append(str(block_name))
        block_positions.append(np.asarray(positions, dtype=np.int64))
    return tuple(block_names), tuple(block_positions)


def _scoring_name(scoring: str | Callable[..., float] | None) -> str:
    if scoring is None:
        return "estimator_score"
    if isinstance(scoring, str):
        return scoring
    return getattr(scoring, "__name__", type(scoring).__name__)


def feature_block_permutation_importance(
    estimator: BaseEstimator,
    X: Any,
    y: ArrayLike,
    feature_blocks: Mapping[str, Sequence[str | int]],
    *,
    scoring: str | Callable[..., float] | None = "neg_mean_absolute_error",
    n_repeats: int = 50,
    random_state: int = DEFAULT_RANDOM_STATE,
    groups: ArrayLike | None = None,
    feature_names: Sequence[str] | None = None,
) -> BlockPermutationResult:
    """Jointly permute physical feature blocks on held-out data.

    All columns in a block use the same row permutation, preserving their within-block
    relationships.  When ``groups`` is supplied, permutations occur independently within
    each group.  Positive importance means permutation worsened a higher-is-better sklearn
    score.  The output is predictive importance and must not be interpreted as a causal
    effect or recipe-knob sensitivity.
    """

    if not isinstance(n_repeats, int) or isinstance(n_repeats, bool) or n_repeats < 1:
        raise ValueError("n_repeats must be a positive integer")
    if not hasattr(X, "shape") or len(X.shape) != 2:
        raise ValueError("X must be a two-dimensional table or array")
    truth = _as_numeric_vector(y, name="y")
    if int(X.shape[0]) != truth.size:
        raise ValueError("X and y must contain the same number of samples")
    if groups is None:
        permutation_groups = (np.arange(truth.size, dtype=np.int64),)
    else:
        group_array = _as_group_vector(groups, n_samples=truth.size)
        permutation_groups = tuple(
            np.flatnonzero(group_array == group) for group in pd.unique(group_array)
        )

    block_names, block_positions = _resolve_feature_blocks(
        X, feature_blocks, feature_names=feature_names
    )
    scorer = check_scoring(estimator, scoring=scoring)
    baseline_score = float(scorer(estimator, X, truth))
    importances = np.empty((len(block_names), n_repeats), dtype=np.float64)
    rng = np.random.default_rng(random_state)
    dataframe_input = isinstance(X, pd.DataFrame)

    for block_index, positions in enumerate(block_positions):
        for repeat_index in range(n_repeats):
            permuted = X.copy(deep=True) if dataframe_input else np.array(X, copy=True)
            for row_indices in permutation_groups:
                shuffled_indices = rng.permutation(row_indices)
                if dataframe_input:
                    permuted.iloc[row_indices, positions] = X.iloc[
                        shuffled_indices, positions
                    ].to_numpy()
                else:
                    permuted[np.ix_(row_indices, positions)] = np.asarray(X)[
                        np.ix_(shuffled_indices, positions)
                    ]
            permuted_score = float(scorer(estimator, permuted, truth))
            importances[block_index, repeat_index] = baseline_score - permuted_score

    return BlockPermutationResult(
        baseline_score=baseline_score,
        block_names=block_names,
        importances=importances,
        scoring_name=_scoring_name(scoring),
    )
