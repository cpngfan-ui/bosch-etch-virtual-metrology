"""Model definitions for the scalar virtual-metrology benchmark.

This module deliberately defines estimators and their small search spaces without
orchestrating cross-validation.  Callers are responsible for selecting the feature view
declared by :class:`ModelConfig` and for passing lot groups to their inner search.

Every learned preprocessing operation is inside the returned ``Pipeline``.  In particular,
imputation, scaling, and PCA must be refit inside every training fold.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

DEFAULT_RANDOM_STATE = 20260830

ModelId = Literal[
    "dummy_mean",
    "context_ridge",
    "cycle_ridge",
    "pls",
    "elastic_net",
    "rbf_svr",
    "pca_gpr",
    "extra_trees",
]
FeatureView = Literal["context", "cycle_context"]

MODEL_IDS: tuple[ModelId, ...] = (
    "dummy_mean",
    "context_ridge",
    "cycle_ridge",
    "pls",
    "elastic_net",
    "rbf_svr",
    "pca_gpr",
    "extra_trees",
)


def _freeze_grid(grid: Mapping[str, tuple[Any, ...]]) -> Mapping[str, tuple[Any, ...]]:
    return MappingProxyType(dict(grid))


_RIDGE_GRID = _freeze_grid({"model__alpha": (0.01, 0.1, 1.0, 10.0, 100.0)})

FROZEN_SEARCH_GRIDS: Mapping[ModelId, Mapping[str, tuple[Any, ...]]] = MappingProxyType(
    {
        "dummy_mean": _freeze_grid({}),
        "context_ridge": _RIDGE_GRID,
        "cycle_ridge": _RIDGE_GRID,
        "pls": _freeze_grid({"model__n_components": (1, 2, 3, 5)}),
        "elastic_net": _freeze_grid(
            {
                "model__alpha": (0.001, 0.01, 0.1),
                "model__l1_ratio": (0.1, 0.5, 0.9),
            }
        ),
        "rbf_svr": _freeze_grid(
            {
                "model__C": (0.3, 1.0, 3.0, 10.0),
                "model__gamma": ("scale", 0.1),
                "model__epsilon": (0.05, 0.1),
            }
        ),
        "pca_gpr": _freeze_grid({"pca__n_components": (2, 3, 5)}),
        "extra_trees": _freeze_grid(
            {
                "model__max_depth": (2, 4),
                "model__min_samples_leaf": (3, 5, 8),
                "model__max_features": (0.5, 1.0),
            }
        ),
    }
)


@dataclass(frozen=True)
class ModelConfig:
    """One benchmark learner and the feature view it is allowed to consume."""

    model_id: ModelId
    role: str
    feature_view: FeatureView
    estimator: Pipeline
    param_grid: Mapping[str, tuple[Any, ...]]
    probabilistic: bool = False

    def mutable_param_grid(self) -> dict[str, list[Any]]:
        """Return a fresh GridSearchCV-compatible copy of the frozen search grid."""

        return {name: list(values) for name, values in self.param_grid.items()}

    def fresh_estimator(self) -> Pipeline:
        """Return an unfitted clone so configurations can safely be reused across folds."""

        return clone(self.estimator)


def _median_imputer() -> SimpleImputer:
    # Keeping empty columns preserves the declared feature schema in a rare all-missing fold.
    return SimpleImputer(strategy="median", keep_empty_features=True)


def _scaled_pipeline(model: Any) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", _median_imputer()),
            ("variance", VarianceThreshold()),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def _gpr_pipeline(random_state: int) -> Pipeline:
    kernel = (
        ConstantKernel(1.0, constant_value_bounds=(1e-2, 1e2))
        * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5)
        + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1e1))
    )
    return Pipeline(
        steps=[
            ("imputer", _median_imputer()),
            ("variance", VarianceThreshold()),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=2, svd_solver="full")),
            (
                "model",
                GaussianProcessRegressor(
                    kernel=kernel,
                    normalize_y=True,
                    n_restarts_optimizer=0,
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_model_configs(
    *, random_state: int = DEFAULT_RANDOM_STATE
) -> dict[ModelId, ModelConfig]:
    """Build fresh definitions for all eight benchmark configurations.

    The returned pipelines do not select columns.  An experiment runner must supply the
    ``context`` or ``cycle_context`` matrix indicated by ``feature_view``.  Lot/date/wafer
    identifiers must not be included in either feature view.
    """

    configs: dict[ModelId, ModelConfig] = {
        "dummy_mean": ModelConfig(
            model_id="dummy_mean",
            role="baseline",
            feature_view="context",
            estimator=Pipeline(steps=[("model", DummyRegressor(strategy="mean"))]),
            param_grid=FROZEN_SEARCH_GRIDS["dummy_mean"],
        ),
        "context_ridge": ModelConfig(
            model_id="context_ridge",
            role="context_ablation",
            feature_view="context",
            estimator=_scaled_pipeline(Ridge()),
            param_grid=FROZEN_SEARCH_GRIDS["context_ridge"],
        ),
        "cycle_ridge": ModelConfig(
            model_id="cycle_ridge",
            role="reference_model",
            feature_view="cycle_context",
            estimator=_scaled_pipeline(Ridge()),
            param_grid=FROZEN_SEARCH_GRIDS["cycle_ridge"],
        ),
        "pls": ModelConfig(
            model_id="pls",
            role="model_comparison",
            feature_view="cycle_context",
            # X scaling is explicit in the pipeline; PLS still centers X and y internally.
            estimator=_scaled_pipeline(PLSRegression(n_components=2, scale=False)),
            param_grid=FROZEN_SEARCH_GRIDS["pls"],
        ),
        "elastic_net": ModelConfig(
            model_id="elastic_net",
            role="model_comparison",
            feature_view="cycle_context",
            estimator=_scaled_pipeline(ElasticNet(max_iter=50_000, selection="cyclic")),
            param_grid=FROZEN_SEARCH_GRIDS["elastic_net"],
        ),
        "rbf_svr": ModelConfig(
            model_id="rbf_svr",
            role="model_comparison",
            feature_view="cycle_context",
            estimator=_scaled_pipeline(SVR(kernel="rbf", cache_size=512)),
            param_grid=FROZEN_SEARCH_GRIDS["rbf_svr"],
        ),
        "pca_gpr": ModelConfig(
            model_id="pca_gpr",
            role="model_comparison",
            feature_view="cycle_context",
            estimator=_gpr_pipeline(random_state),
            param_grid=FROZEN_SEARCH_GRIDS["pca_gpr"],
            probabilistic=True,
        ),
        "extra_trees": ModelConfig(
            model_id="extra_trees",
            role="model_comparison",
            feature_view="cycle_context",
            estimator=Pipeline(
                steps=[
                    ("imputer", _median_imputer()),
                    ("variance", VarianceThreshold()),
                    (
                        "model",
                        ExtraTreesRegressor(
                            n_estimators=500,
                            criterion="squared_error",
                            random_state=random_state,
                            n_jobs=1,
                        ),
                    ),
                ]
            ),
            param_grid=FROZEN_SEARCH_GRIDS["extra_trees"],
        ),
    }
    return configs


def get_model_config(
    model_id: ModelId, *, random_state: int = DEFAULT_RANDOM_STATE
) -> ModelConfig:
    """Return one fresh model configuration by its frozen identifier."""

    try:
        return build_model_configs(random_state=random_state)[model_id]
    except KeyError as exc:  # pragma: no cover - guarded by the ModelId type for typed callers.
        allowed = ", ".join(MODEL_IDS)
        raise ValueError(f"Unknown model_id {model_id!r}; expected one of: {allowed}") from exc
