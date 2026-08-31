"""Pre-specified robustness analyses for the wafer-mean Si-etch benchmark.

The suite reuses the same nested leave-one-lot-out protocol as the main
experiment.  Input-block removal quantifies predictive information, not a
causal effect of manipulating a recipe knob.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from .experiment import ExperimentConfig, ProtocolModelSpec, run
from .io import build_wafer_cohort
from .models import build_model_configs


@dataclass(frozen=True)
class SensitivityVariant:
    """One fixed alternative to the main cycle-aware Ridge procedure."""

    variant_id: str
    description: str
    excluded_block: str | None = None
    robust_scaler: bool = False
    direct_postox_target: bool = False


SENSITIVITY_VARIANTS: tuple[SensitivityVariant, ...] = (
    SensitivityVariant(
        "process_only",
        "Remove all pre-known context; retain process telemetry summaries.",
        excluded_block="context",
    ),
    SensitivityVariant(
        "without_gas_delivery",
        "Remove anonymous gas-flow and gas-phase timing summaries.",
        excluded_block="gas_delivery",
    ),
    SensitivityVariant(
        "without_pressure_helium",
        "Remove pressure, foreline-pressure, and helium summaries.",
        excluded_block="pressure_helium",
    ),
    SensitivityVariant(
        "without_rf_power",
        "Remove platen/source RF, DC-bias, and RF-active-span summaries.",
        excluded_block="rf_power",
    ),
    SensitivityVariant(
        "without_temperature",
        "Remove heater-temperature summaries.",
        excluded_block="temperature",
    ),
    SensitivityVariant(
        "without_timing",
        "Remove the complete ten-feature cycle timing/startup block.",
        excluded_block="timing",
    ),
    SensitivityVariant(
        "robust_scaler",
        "Replace StandardScaler with RobustScaler; keep the learner and grid fixed.",
        robust_scaler=True,
    ),
    SensitivityVariant(
        "direct_postox_target",
        "Recompute wafer means only over sites with non-missing raw post-etch fits.",
        direct_postox_target=True,
    ),
)


def feature_block(feature: str) -> str:
    """Assign an 81-feature name to a target-free physical information block."""

    lower = feature.lower()
    if feature.startswith("context__"):
        return "context"
    if feature.startswith("timing__"):
        if any(token in lower for token in ("short_", "long_", "extra_long")):
            return "gas_delivery"
        if "source_active" in lower:
            return "rf_power"
        return "timing"
    if "gas" in lower:
        return "gas_delivery"
    if "pressure" in lower or "helium" in lower:
        return "pressure_helium"
    if any(token in lower for token in ("platen", "sourcerf", "dcbias")):
        return "rf_power"
    if "heater" in lower or "temp" in lower:
        return "temperature"
    return "other_process"


def _ridge_spec(
    *,
    feature_columns: tuple[str, ...],
    random_state: int,
    robust_scaler: bool,
) -> ProtocolModelSpec:
    primary = build_model_configs(random_state=random_state)["cycle_ridge"]
    if robust_scaler:
        estimator = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("variance", VarianceThreshold()),
                ("scaler", RobustScaler()),
                ("model", Ridge()),
            ]
        )
    else:
        estimator = primary.fresh_estimator()
    return ProtocolModelSpec(
        model_id="cycle_ridge",
        role="sensitivity_procedure",
        feature_view="explicit_sensitivity_columns",
        feature_columns=feature_columns,
        estimator=estimator,
        param_grid=primary.mutable_param_grid(),
    )


def _paired_lot_bootstrap(
    primary: pd.DataFrame,
    variant: pd.DataFrame,
    *,
    lot_column: str,
    n_bootstrap: int,
    random_state: int,
) -> dict[str, float]:
    primary_loss = primary.groupby(lot_column)["absolute_error"].mean().sort_index()
    variant_loss = variant.groupby(lot_column)["absolute_error"].mean().sort_index()
    aligned = pd.concat(
        [primary_loss.rename("primary"), variant_loss.rename("variant")], axis=1
    ).dropna()
    difference = (aligned["variant"] - aligned["primary"]).to_numpy(dtype=float)
    rng = np.random.default_rng(random_state)
    draws = rng.integers(0, len(difference), size=(n_bootstrap, len(difference)))
    replicates = difference[draws].mean(axis=1)
    return {
        "paired_delta_macro_lot_mae_um": float(difference.mean()),
        "paired_delta_lower_95_um": float(np.quantile(replicates, 0.025)),
        "paired_delta_upper_95_um": float(np.quantile(replicates, 0.975)),
        "lots_better_than_primary": int(np.count_nonzero(difference < 0)),
        "lots_worse_than_primary": int(np.count_nonzero(difference > 0)),
    }


def run_sensitivity_suite(
    config: ExperimentConfig,
    *,
    overwrite: bool = False,
) -> Path:
    """Run all Ridge robustness variants and return the summary path."""

    feature_path = config.prepared_dir / "feature_table.csv"
    if not feature_path.exists():
        raise FileNotFoundError(f"Prepared feature table not found: {feature_path}")
    primary_prediction_path = (
        config.output_dir / "models" / "cycle_ridge" / "oof_predictions.csv"
    )
    primary_metric_path = config.output_dir / "models" / "cycle_ridge" / "metrics.json"
    if not primary_prediction_path.exists() or not primary_metric_path.exists():
        raise FileNotFoundError("Run the main eight-model experiment before sensitivity analysis")

    table = pd.read_csv(feature_path)
    predictor_columns = tuple(column for column in table if "__" in column)
    if len(predictor_columns) != 81:
        raise ValueError(f"Expected 81 predictors, found {len(predictor_columns)}")
    block_by_feature = {feature: feature_block(feature) for feature in predictor_columns}
    primary_predictions = pd.read_csv(primary_prediction_path)

    cohort = build_wafer_cohort(config.raw_dir)
    direct_target = cohort.targets["mean_si_etch_direct_postox_only_um"]
    sensitivity_root = config.output_dir / "sensitivity"
    sensitivity_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    per_lot_rows: list[pd.DataFrame] = []
    for variant_index, variant in enumerate(SENSITIVITY_VARIANTS):
        variant_table = table.copy()
        feature_columns = predictor_columns
        same_target_as_primary = not variant.direct_postox_target
        if variant.excluded_block is not None:
            if variant.excluded_block == "timing":
                feature_columns = tuple(
                    feature
                    for feature in predictor_columns
                    if not feature.startswith("timing__")
                )
            else:
                feature_columns = tuple(
                    feature
                    for feature in predictor_columns
                    if block_by_feature[feature] != variant.excluded_block
                )
        if variant.direct_postox_target:
            aligned_target = direct_target.reindex(variant_table[config.id_column])
            if aligned_target.isna().any():
                raise ValueError("Direct-postox sensitivity target is incomplete after key join")
            variant_table[config.target_column] = aligned_target.to_numpy(dtype=float)

        variant_output = sensitivity_root / variant.variant_id
        variant_config = replace(
            config,
            output_dir=variant_output,
            overwrite=overwrite,
            permutation_repetitions=0,
        )
        spec = _ridge_spec(
            feature_columns=feature_columns,
            random_state=config.seed,
            robust_scaler=variant.robust_scaler,
        )
        artifacts = run(variant_config, feature_table=variant_table, model_specs=[spec])
        comparison = pd.read_csv(artifacts.comparison_path).iloc[0]
        predictions = pd.read_csv(
            variant_output / "models" / "cycle_ridge" / "oof_predictions.csv"
        )
        metrics_by_lot = pd.read_csv(
            variant_output / "models" / "cycle_ridge" / "metrics_by_lot.csv"
        )
        metrics_by_lot.insert(0, "variant_id", variant.variant_id)
        per_lot_rows.append(metrics_by_lot)

        paired: dict[str, float | int] = {
            "paired_delta_macro_lot_mae_um": float("nan"),
            "paired_delta_lower_95_um": float("nan"),
            "paired_delta_upper_95_um": float("nan"),
            "lots_better_than_primary": 0,
            "lots_worse_than_primary": 0,
        }
        if same_target_as_primary:
            paired = _paired_lot_bootstrap(
                primary_predictions,
                predictions,
                lot_column=config.lot_column,
                n_bootstrap=config.bootstrap_replicates,
                random_state=config.seed + 1000 + variant_index,
            )
        primary_truth = primary_predictions.set_index(config.id_column)["y_true"]
        variant_truth = predictions.set_index(config.id_column)["y_true"]
        aligned_truth = pd.concat(
            [primary_truth.rename("primary"), variant_truth.rename("variant")], axis=1
        ).dropna()
        target_difference = aligned_truth["variant"] - aligned_truth["primary"]
        summary_rows.append(
            {
                "variant_id": variant.variant_id,
                "description": variant.description,
                "same_target_as_primary": same_target_as_primary,
                "excluded_block": variant.excluded_block,
                "n_features": len(feature_columns),
                "macro_lot_mae_um": float(comparison["macro_lot_mae_um"]),
                "pooled_mae_um": float(comparison["mae_um"]),
                "pooled_rmse_um": float(comparison["rmse_um"]),
                "oof_r2": float(comparison["r2_oof"]),
                "target_mean_shift_um": float(target_difference.mean()),
                "target_max_absolute_shift_um": float(target_difference.abs().max()),
                **paired,
                "artifact_dir": str(variant_output),
            }
        )

    summary_path = sensitivity_root / "sensitivity_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.concat(per_lot_rows, ignore_index=True).to_csv(
        sensitivity_root / "sensitivity_metrics_by_lot.csv", index=False
    )
    pd.DataFrame(
        [
            {"feature": feature, "physical_block": block}
            for feature, block in block_by_feature.items()
        ]
    ).to_csv(sensitivity_root / "feature_block_map.csv", index=False)
    return summary_path


__all__ = [
    "SENSITIVITY_VARIANTS",
    "SensitivityVariant",
    "feature_block",
    "run_sensitivity_suite",
]
