"""Lot-grouped experiment orchestration for wafer-level virtual metrology.

The orchestration in this module deliberately treats a wafer run, not a
timestamp or metrology site, as the supervised sample.  Both model tuning and
outer evaluation use ``LeaveOneGroupOut`` with the lot number as the group.

The data-, feature-, model-, and evaluation-specific modules are imported only
when their functionality is needed.  Their expected interfaces are:

* ``bosch_vm.io.build_wafer_cohort(raw_dir) -> WaferCohort``
* ``bosch_vm.features.build_feature_tables(cohort) -> FeatureTables``
* ``bosch_vm.models.build_model_configs(random_state) -> Mapping[str, ModelConfig]``
* evaluation helpers for lot-aware metrics, cluster bootstrap, coefficient
  extraction, and held-out feature-block permutation

The small adapters below intentionally accept either dataclass-like objects or
mappings so that those modules can evolve without weakening the split protocol.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import logging
import math
import os
import platform
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut

DEFAULT_RAW_DIR = Path("data/raw/zenodo_17122442")
DEFAULT_PREPARED_DIR = Path("data/processed/wafer_mean_si_etch_lolo_v1")
DEFAULT_OUTPUT_DIR = Path("results/wafer_mean_si_etch_lolo_v1")
DEFAULT_EXPERIMENT_DIR = Path("experiments/mean_si_etch")


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration shared by ``prepare`` and ``run``.

    ``feature_views`` can map a model's view name to an explicit list of
    feature columns.  The feature builder's metadata may provide the same
    mapping.  Explicit mappings are preferable to name-based inference.
    """

    raw_dir: Path = DEFAULT_RAW_DIR
    prepared_dir: Path = DEFAULT_PREPARED_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    experiment_dir: Path = DEFAULT_EXPERIMENT_DIR
    id_column: str = "experiment_key"
    lot_column: str = "lot_number"
    wafer_column: str = "wafer_order"
    target_column: str = "mean_si_etch_um"
    primary_model_id: str = "cycle_ridge"
    scoring: str | Callable[..., float] = "neg_mean_absolute_error"
    seed: int = 1729
    n_jobs: int = 1
    bootstrap_replicates: int = 2_000
    permutation_repetitions: int = 50
    overwrite: bool = False
    context_columns: tuple[str, ...] = (
        "context__conditioning_count",
        "context__conditioning_surface_chuck",
        "context__conditioning_surface_si",
        "context__conditioning_surface_sio2",
        "context__wafer_order",
    )
    feature_views: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> ExperimentConfig:
        if not values:
            return cls()
        values = _flatten_project_config(values)
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"Unknown experiment configuration keys: {unknown}")
        normalized = dict(values)
        for key in ("raw_dir", "prepared_dir", "output_dir", "experiment_dir"):
            if key in normalized:
                normalized[key] = Path(normalized[key])
        if "context_columns" in normalized:
            normalized["context_columns"] = tuple(normalized["context_columns"])
        if "feature_views" in normalized:
            normalized["feature_views"] = {
                str(name): tuple(columns) for name, columns in normalized["feature_views"].items()
            }
        return cls(**normalized)


@dataclass(frozen=True)
class ProtocolModelSpec:
    """Minimal model specification consumed by the orchestrator."""

    model_id: str
    estimator: Any
    param_grid: Mapping[str, Sequence[Any]] | Sequence[Mapping[str, Sequence[Any]]]
    feature_view: str | Sequence[str] = "all"
    role: str = "secondary"
    scoring: str | Callable[..., float] | None = None
    feature_columns: tuple[str, ...] | None = None
    probabilistic: bool = False


@dataclass(frozen=True)
class PreparedArtifacts:
    prepared_dir: Path
    cohort_path: Path
    feature_table_path: Path
    all_predictors_path: Path
    target_audit_path: Path
    qa_path: Path
    metadata_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class RunArtifacts:
    output_dir: Path
    comparison_path: Path
    manifest_path: Path
    split_path: Path
    log_path: Path


def prepare(
    config: ExperimentConfig | Mapping[str, Any] | None = None,
    *,
    cohort: Any | None = None,
    feature_table: pd.DataFrame | None = None,
    qa: pd.DataFrame | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> PreparedArtifacts:
    """Build and persist the canonical wafer-level modeling table.

    Dataframes may be injected for tests or alternate ingestion frontends.  In
    normal use the function calls ``io.build_wafer_cohort`` followed by
    ``features.build_feature_tables``.
    """

    cfg = _coerce_config(config)
    prepared_dir = cfg.prepared_dir
    prepared_dir.mkdir(parents=True, exist_ok=True)

    paths = PreparedArtifacts(
        prepared_dir=prepared_dir,
        cohort_path=prepared_dir / "cohort.csv",
        feature_table_path=prepared_dir / "feature_table.csv",
        all_predictors_path=prepared_dir / "all_predictors.csv",
        target_audit_path=prepared_dir / "target_audit.csv",
        qa_path=prepared_dir / "feature_qa.csv",
        metadata_path=prepared_dir / "feature_metadata.json",
        manifest_path=prepared_dir / "prepare_manifest.json",
    )
    _guard_known_outputs(
        [
            paths.cohort_path,
            paths.feature_table_path,
            paths.all_predictors_path,
            paths.target_audit_path,
            paths.qa_path,
            paths.metadata_path,
            paths.manifest_path,
        ],
        overwrite=cfg.overwrite,
    )

    cohort_object: Any = cohort
    if cohort is None:
        io_module = importlib.import_module("bosch_vm.io")
        cohort_object = io_module.build_wafer_cohort(cfg.raw_dir)

    feature_metadata: dict[str, Any] = dict(metadata or {})
    all_predictors: pd.DataFrame | None = None
    target_audit: pd.DataFrame | None = None
    if feature_table is None:
        features_module = importlib.import_module("bosch_vm.features")
        built = features_module.build_feature_tables(cohort_object)
        required_attributes = {
            "all_predictors",
            "matched_predictors",
            "matched_targets",
            "metadata",
            "all_qa",
        }
        missing_attributes = sorted(
            name for name in required_attributes if not hasattr(built, name)
        )
        if missing_attributes:
            raise TypeError(
                f"features.build_feature_tables result lacks attributes {missing_attributes}."
            )

        predictors = _indexed_frame(built.matched_predictors, cfg.id_column)
        all_predictors = _indexed_frame(built.all_predictors, cfg.id_column).reset_index()
        targets = _indexed_series(built.matched_targets, cfg.id_column, cfg.target_column)
        built_metadata = _indexed_frame(built.metadata, cfg.id_column)
        matched_metadata = built_metadata.reindex(predictors.index)
        metadata_columns = [
            column
            for column in (cfg.lot_column, cfg.wafer_column)
            if column in matched_metadata.columns
        ]
        feature_table = (
            predictors.join(matched_metadata[metadata_columns], how="left")
            .join(targets, how="left")
            .reset_index()
        )
        qa = _indexed_frame(built.all_qa, cfg.id_column).reset_index()
        cohort = built_metadata.reset_index()
        if hasattr(cohort_object, "targets"):
            target_audit = _indexed_frame(cohort_object.targets, cfg.id_column).reset_index()

        predictor_columns = list(predictors.columns)
        context_columns = [
            column for column in predictor_columns if column.startswith("context__")
        ]
        feature_metadata.setdefault(
            "feature_views",
            {
                "context": context_columns,
                "cycle_context": predictor_columns,
            },
        )
        feature_metadata.setdefault("predictor_columns", predictor_columns)
    else:
        feature_table = _ensure_dataframe(feature_table, "feature_table")

    if cohort is None:
        if isinstance(cohort_object, pd.DataFrame):
            cohort = cohort_object
        else:
            raise TypeError(
                "Unable to serialize cohort metadata; the feature builder must provide metadata."
            )

    qa = pd.DataFrame() if qa is None else _ensure_dataframe(qa, "qa")
    feature_table = _validate_feature_table(feature_table, cfg)
    cohort = _validate_cohort(cohort, cfg)

    identity_columns = [
        column
        for column in (
            cfg.id_column,
            cfg.lot_column,
            cfg.wafer_column,
            cfg.target_column,
        )
        if column in feature_table.columns
    ]
    predictor_columns = [
        column for column in feature_table.columns if column not in identity_columns
    ]
    feature_table = feature_table.loc[:, identity_columns + predictor_columns]
    if all_predictors is None:
        all_predictors = feature_table.loc[:, [cfg.id_column] + predictor_columns].copy()
    if target_audit is None:
        audit_columns = [
            column
            for column in (
                cfg.id_column,
                cfg.lot_column,
                cfg.wafer_column,
                cfg.target_column,
            )
            if column in feature_table.columns
        ]
        target_audit = feature_table.loc[:, audit_columns].copy()

    cohort_ids = set(cohort[cfg.id_column].astype(str))
    feature_ids = set(feature_table[cfg.id_column].astype(str))
    if not feature_ids.issubset(cohort_ids):
        unexpected = sorted(feature_ids - cohort_ids)[:10]
        raise ValueError(f"Feature table contains wafer IDs absent from the cohort: {unexpected}")

    sort_columns = [cfg.lot_column]
    if cfg.wafer_column in feature_table.columns:
        sort_columns.append(cfg.wafer_column)
    feature_table = feature_table.sort_values(sort_columns, kind="stable").reset_index(drop=True)

    _write_dataframe(cohort, paths.cohort_path)
    _write_dataframe(feature_table, paths.feature_table_path)
    _write_dataframe(all_predictors, paths.all_predictors_path)
    _write_dataframe(target_audit, paths.target_audit_path)
    _write_dataframe(qa, paths.qa_path)

    feature_metadata.update(
        {
            "id_column": cfg.id_column,
            "lot_column": cfg.lot_column,
            "wafer_column": cfg.wafer_column,
            "target_column": cfg.target_column,
            "n_wafers": int(len(feature_table)),
            "n_lots": int(feature_table[cfg.lot_column].nunique()),
            "prepared_at_utc": _utc_now(),
        }
    )
    _write_json(feature_metadata, paths.metadata_path)

    manifest = {
        "status": "complete",
        "protocol_unit": "one wafer equals one process run equals one supervised sample",
        "configuration": _config_manifest(cfg),
        "counts": {
            "cohort_rows": int(len(cohort)),
            "feature_rows": int(len(feature_table)),
            "lots": int(feature_table[cfg.lot_column].nunique()),
        },
        "artifacts": {
            path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in (
                paths.cohort_path,
                paths.feature_table_path,
                paths.all_predictors_path,
                paths.target_audit_path,
                paths.qa_path,
                paths.metadata_path,
            )
        },
        "created_at_utc": _utc_now(),
    }
    _write_json(manifest, paths.manifest_path)
    return paths


def run(
    config: ExperimentConfig | Mapping[str, Any] | None = None,
    *,
    feature_table: pd.DataFrame | None = None,
    model_specs: Sequence[Any] | None = None,
) -> RunArtifacts:
    """Run nested lot-held-out evaluation and persist a complete audit trail."""

    cfg = _coerce_config(config)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = RunArtifacts(
        output_dir=cfg.output_dir,
        comparison_path=cfg.output_dir / "comparison.csv",
        manifest_path=cfg.output_dir / "manifest.json",
        split_path=cfg.output_dir / "splits" / "outer_lolo.csv",
        log_path=cfg.experiment_dir / "run.log",
    )
    _guard_known_outputs(
        [artifacts.comparison_path, artifacts.manifest_path], overwrite=cfg.overwrite
    )
    artifacts.split_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts.log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = _make_logger(artifacts.log_path)
    started_at = _utc_now()

    manifest: dict[str, Any] = {
        "status": "running",
        "started_at_utc": started_at,
        "configuration": _config_manifest(cfg),
        "protocol": {
            "sample_unit": "wafer_run",
            "outer_cv": "LeaveOneGroupOut(lot)",
            "inner_cv": "LeaveOneGroupOut(lot) within each outer-training set",
            "search": "GridSearchCV",
            "random_sample_split": False,
            "primary_model_id": cfg.primary_model_id,
        },
        "environment": _environment_manifest(),
    }
    _write_json(manifest, artifacts.manifest_path)

    try:
        if feature_table is None:
            feature_path = cfg.prepared_dir / "feature_table.csv"
            if not feature_path.exists():
                raise FileNotFoundError(
                    f"Prepared feature table not found: {feature_path}. Run prepare first."
                )
            feature_table = pd.read_csv(feature_path)
        table = _validate_feature_table(feature_table, cfg)
        metadata = _load_feature_metadata(cfg.prepared_dir)

        specs = _load_model_specs(model_specs, cfg.seed)
        specs = [_coerce_model_spec(spec) for spec in specs]
        _validate_model_specs(specs, cfg.primary_model_id)

        logger.info(
            "Starting nested LOLO run with %d wafers, %d lots, and %d model procedures.",
            len(table),
            table[cfg.lot_column].nunique(),
            len(specs),
        )

        split_manifest = _build_outer_split_manifest(table, cfg)
        _write_dataframe(split_manifest, artifacts.split_path)

        comparison_rows: list[dict[str, Any]] = []
        model_predictions: dict[str, pd.DataFrame] = {}
        per_lot_frames: list[pd.DataFrame] = []
        model_manifests: list[dict[str, Any]] = []

        for spec in specs:
            logger.info("Evaluating model procedure %s (%s).", spec.model_id, spec.role)
            result = _run_model_nested_lolo(
                table=table,
                spec=spec,
                cfg=cfg,
                metadata=metadata,
                logger=logger,
            )
            model_predictions[spec.model_id] = result["predictions"]
            per_lot_frames.append(result["per_lot"])
            metrics = result["metrics"]
            comparison_rows.append(
                {
                    "model_id": spec.model_id,
                    "role": spec.role,
                    "feature_view": _feature_view_label(spec.feature_view),
                    **metrics,
                }
            )
            model_manifests.append(result["manifest"])

        comparison = pd.DataFrame(comparison_rows)
        comparison = _add_primary_comparisons(
            comparison=comparison,
            predictions=model_predictions,
            cfg=cfg,
        )
        comparison = comparison.sort_values(
            ["macro_lot_mae_um", "model_id"], kind="stable"
        ).reset_index(drop=True)
        _write_dataframe(comparison, artifacts.comparison_path)

        all_oof = pd.concat(model_predictions.values(), ignore_index=True)
        _write_dataframe(all_oof, cfg.output_dir / "all_oof_predictions.csv")
        all_per_lot = pd.concat(per_lot_frames, ignore_index=True)
        _write_dataframe(all_per_lot, cfg.output_dir / "all_metrics_by_lot.csv")

        primary_differences = _paired_lot_differences(
            model_predictions, primary_model_id=cfg.primary_model_id, cfg=cfg
        )
        _write_dataframe(primary_differences, cfg.output_dir / "primary_lot_differences.csv")

        manifest.update(
            {
                "status": "complete",
                "completed_at_utc": _utc_now(),
                "counts": {
                    "wafers": int(len(table)),
                    "lots": int(table[cfg.lot_column].nunique()),
                    "models": int(len(specs)),
                    "outer_predictions_per_model": int(len(table)),
                },
                "input": {
                    "prepared_feature_table": str(cfg.prepared_dir / "feature_table.csv"),
                    "prepared_feature_sha256": _optional_sha256(
                        cfg.prepared_dir / "feature_table.csv"
                    ),
                },
                "models": model_manifests,
                "artifacts": {
                    "comparison": str(artifacts.comparison_path),
                    "outer_splits": str(artifacts.split_path),
                    "primary_lot_differences": str(cfg.output_dir / "primary_lot_differences.csv"),
                    "all_oof_predictions": str(cfg.output_dir / "all_oof_predictions.csv"),
                    "all_metrics_by_lot": str(cfg.output_dir / "all_metrics_by_lot.csv"),
                    "log": str(artifacts.log_path),
                },
            }
        )
        _write_json(manifest, artifacts.manifest_path)
        logger.info("Completed nested LOLO run; comparison saved to %s.", artifacts.comparison_path)
        return artifacts
    except Exception as exc:
        manifest.update(
            {
                "status": "failed",
                "failed_at_utc": _utc_now(),
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        _write_json(manifest, artifacts.manifest_path)
        logger.exception("Experiment failed.")
        raise
    finally:
        _close_logger(logger)


def _run_model_nested_lolo(
    *,
    table: pd.DataFrame,
    spec: ProtocolModelSpec,
    cfg: ExperimentConfig,
    metadata: Mapping[str, Any],
    logger: logging.Logger,
) -> dict[str, Any]:
    model_dir = cfg.output_dir / "models" / _safe_name(spec.model_id)
    model_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        {
            "model_id": spec.model_id,
            "role": spec.role,
            "feature_view": _feature_view_label(spec.feature_view),
            "scoring": spec.scoring or cfg.scoring,
            "probabilistic": spec.probabilistic,
            "param_grid": spec.param_grid,
            "selection_protocol": "inner LeaveOneGroupOut(lot) GridSearchCV",
        },
        model_dir / "search_space.json",
    )

    feature_columns, feature_source = _resolve_feature_columns(
        table=table, spec=spec, cfg=cfg, metadata=metadata
    )
    if feature_columns:
        X = table.loc[:, feature_columns].copy()
    else:
        # Dummy estimators still need a two-dimensional design matrix.
        X = pd.DataFrame({"__constant__": np.zeros(len(table), dtype=float)}, index=table.index)
    y = pd.to_numeric(table[cfg.target_column], errors="raise").to_numpy(dtype=float)
    groups = table[cfg.lot_column].to_numpy()

    outer = LeaveOneGroupOut()
    fold_predictions: list[pd.DataFrame] = []
    fold_scores: list[pd.DataFrame] = []
    fold_coefficients: list[pd.DataFrame] = []
    fold_permutations: list[pd.DataFrame] = []
    selected_parameters: list[dict[str, Any]] = []
    fold_summaries: list[dict[str, Any]] = []

    for outer_index, (train_index, test_index) in enumerate(
        outer.split(X, y, groups=groups), start=1
    ):
        train_lots = pd.unique(groups[train_index])
        test_lots = pd.unique(groups[test_index])
        if len(test_lots) != 1:
            raise AssertionError("Outer LeaveOneGroupOut fold must contain exactly one test lot.")
        if set(train_lots) & set(test_lots):
            raise AssertionError("Lot leakage detected between outer train and test sets.")
        if len(train_lots) < 2:
            raise ValueError("Nested LOGO requires at least three lots in the complete dataset.")

        test_lot = test_lots[0]
        fold_name = f"lot_{_safe_name(test_lot)}"
        fold_dir = model_dir / "folds" / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)

        X_train = X.iloc[train_index]
        y_train = y[train_index]
        groups_train = groups[train_index]
        X_test = X.iloc[test_index]

        inner = LeaveOneGroupOut()
        inner_splits = list(inner.split(X_train, y_train, groups=groups_train))
        expected_inner_splits = len(pd.unique(groups_train))
        if len(inner_splits) != expected_inner_splits:
            raise AssertionError("Inner CV did not hold out each outer-training lot exactly once.")

        search = GridSearchCV(
            estimator=clone(spec.estimator),
            param_grid=spec.param_grid,
            scoring=spec.scoring or cfg.scoring,
            cv=inner_splits,
            refit=True,
            n_jobs=cfg.n_jobs,
            return_train_score=True,
            error_score="raise",
        )
        search.fit(X_train, y_train)
        prediction_std: np.ndarray | None = None
        if spec.probabilistic:
            prediction, prediction_std = _predict_mean_and_std(search.best_estimator_, X_test)
        else:
            prediction = np.asarray(search.predict(X_test), dtype=float).reshape(-1)
        if prediction.shape[0] != len(test_index):
            raise AssertionError("Estimator returned the wrong number of outer-test predictions.")

        fold_pred = table.iloc[test_index][
            [cfg.id_column, cfg.lot_column]
            + ([cfg.wafer_column] if cfg.wafer_column in table.columns else [])
        ].copy()
        fold_pred["y_true"] = y[test_index]
        fold_pred["y_pred"] = prediction
        if prediction_std is not None:
            fold_pred["y_pred_std"] = prediction_std
        fold_pred["residual"] = fold_pred["y_pred"] - fold_pred["y_true"]
        fold_pred["absolute_error"] = fold_pred["residual"].abs()
        fold_pred["outer_fold"] = outer_index
        fold_pred["outer_test_lot"] = test_lot
        fold_pred["model_id"] = spec.model_id
        fold_pred["role"] = spec.role
        fold_predictions.append(fold_pred)
        _write_dataframe(fold_pred, fold_dir / "outer_predictions.csv")

        inner_scores = pd.DataFrame(search.cv_results_)
        inner_scores.insert(0, "model_id", spec.model_id)
        inner_scores.insert(1, "outer_fold", outer_index)
        inner_scores.insert(2, "outer_test_lot", test_lot)
        inner_scores.insert(3, "n_inner_splits", len(inner_splits))
        fold_scores.append(inner_scores)
        _write_dataframe(inner_scores, fold_dir / "inner_scores.csv")

        best_params = _jsonable(search.best_params_)
        parameter_record = {
            "outer_fold": outer_index,
            "outer_test_lot": _jsonable(test_lot),
            "best_index": int(search.best_index_),
            "best_inner_score": float(search.best_score_),
            "best_params": best_params,
        }
        selected_parameters.append(parameter_record)
        _write_json(parameter_record, fold_dir / "best_params.json")
        joblib.dump(search.best_estimator_, fold_dir / "model.joblib")

        coefficients = _outer_coefficients(
            search.best_estimator_,
            model_id=spec.model_id,
            outer_lot=test_lot,
            feature_columns=(feature_columns if feature_columns else ("__constant__",)),
            target_name=cfg.target_column,
        )
        if not coefficients.empty:
            fold_coefficients.append(coefficients)
            _write_dataframe(coefficients, fold_dir / "coefficients.csv")

        permutation = _heldout_block_permutation(
            estimator=search.best_estimator_,
            X_test=X_test,
            y_test=y[test_index],
            groups_test=groups[test_index],
            feature_columns=(feature_columns if feature_columns else ("__constant__",)),
            model_id=spec.model_id,
            outer_lot=test_lot,
            cfg=cfg,
            random_state=cfg.seed + outer_index,
        )
        if not permutation.empty:
            fold_permutations.append(permutation)
            _write_dataframe(permutation, fold_dir / "heldout_block_permutation.csv")

        fold_summary = {
            "outer_fold": outer_index,
            "outer_test_lot": _jsonable(test_lot),
            "train_lots": [_jsonable(value) for value in train_lots],
            "test_lots": [_jsonable(value) for value in test_lots],
            "n_train_wafers": int(len(train_index)),
            "n_test_wafers": int(len(test_index)),
            "n_inner_splits": int(len(inner_splits)),
            "feature_columns": list(feature_columns) if feature_columns else ["__constant__"],
            "model_artifact": str(fold_dir / "model.joblib"),
        }
        fold_summaries.append(fold_summary)
        _write_json(fold_summary, fold_dir / "fold_manifest.json")
        logger.info(
            "%s: held out lot %s (%d wafers); best inner score %.6f.",
            spec.model_id,
            test_lot,
            len(test_index),
            search.best_score_,
        )

    predictions = pd.concat(fold_predictions, ignore_index=True)
    predictions = _validate_oof_coverage(predictions, table, cfg, spec.model_id)
    inner_scores = pd.concat(fold_scores, ignore_index=True)
    metrics = _compute_metrics(predictions, cfg)
    per_lot = _metrics_by_lot(predictions, cfg)
    per_lot.insert(0, "model_id", spec.model_id)
    per_lot.insert(1, "role", spec.role)

    _write_dataframe(predictions, model_dir / "oof_predictions.csv")
    _write_dataframe(inner_scores, model_dir / "inner_scores.csv")
    _write_dataframe(per_lot, model_dir / "metrics_by_lot.csv")
    _write_json(metrics, model_dir / "metrics.json")
    _write_json(selected_parameters, model_dir / "selected_hyperparameters.json")

    coefficients = (
        pd.concat(fold_coefficients, ignore_index=True)
        if fold_coefficients
        else pd.DataFrame(
            columns=[
                "model_id",
                "outer_lot",
                "target",
                "feature",
                "coefficient",
                "selected",
                "coefficient_scale",
            ]
        )
    )
    _write_dataframe(coefficients, model_dir / "outer_coefficients.csv")
    permutations = (
        pd.concat(fold_permutations, ignore_index=True)
        if fold_permutations
        else pd.DataFrame(
            columns=[
                "model_id",
                "outer_lot",
                "block",
                "repeat",
                "importance",
                "baseline_score",
                "scoring",
            ]
        )
    )
    _write_dataframe(permutations, model_dir / "heldout_block_permutation.csv")

    final_refit = _fit_final_all_data_model(
        X=X,
        y=y,
        groups=groups,
        spec=spec,
        cfg=cfg,
        model_dir=model_dir,
    )

    bootstrap = _cluster_bootstrap(predictions, cfg, spec.model_id)
    _write_dataframe(bootstrap, model_dir / "cluster_bootstrap_intervals.csv")

    model_manifest = {
        "model_id": spec.model_id,
        "role": spec.role,
        "deployment_point": "post_run_pre_metrology",
        "target": cfg.target_column,
        "interpretation_limit": (
            "Predictive associations and held-out importance are not causal recipe-knob effects."
        ),
        "feature_view": _feature_view_label(spec.feature_view),
        "feature_columns": list(feature_columns) if feature_columns else ["__constant__"],
        "feature_column_source": feature_source,
        "outer_folds": int(len(fold_summaries)),
        "oof_wafers": int(len(predictions)),
        "folds": fold_summaries,
        "metrics": metrics,
        "artifacts": {
            "search_space": str(model_dir / "search_space.json"),
            "oof_predictions": str(model_dir / "oof_predictions.csv"),
            "inner_scores": str(model_dir / "inner_scores.csv"),
            "metrics_by_lot": str(model_dir / "metrics_by_lot.csv"),
            "selected_hyperparameters": str(model_dir / "selected_hyperparameters.json"),
            "cluster_bootstrap_intervals": str(model_dir / "cluster_bootstrap_intervals.csv"),
            "outer_coefficients": str(model_dir / "outer_coefficients.csv"),
            "heldout_block_permutation": str(model_dir / "heldout_block_permutation.csv"),
            "final_refit_model": str(model_dir / "final_refit" / "model.joblib"),
        },
        "final_refit": final_refit,
    }
    _write_json(model_manifest, model_dir / "model_manifest.json")
    _write_json(model_manifest, model_dir / "model_card.json")
    return {
        "predictions": predictions,
        "per_lot": per_lot,
        "metrics": metrics,
        "manifest": model_manifest,
    }


def _predict_mean_and_std(fitted_estimator: Any, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return GPR mean/std after explicitly applying pipeline transforms."""

    if hasattr(fitted_estimator, "named_steps") and "model" in fitted_estimator.named_steps:
        latent = fitted_estimator[:-1].transform(X)
        result = fitted_estimator.named_steps["model"].predict(latent, return_std=True)
    else:
        result = fitted_estimator.predict(X, return_std=True)
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError("Probabilistic model must return (prediction_mean, prediction_std).")
    prediction = np.asarray(result[0], dtype=float).reshape(-1)
    prediction_std = np.asarray(result[1], dtype=float).reshape(-1)
    if prediction.shape != prediction_std.shape:
        raise ValueError("Probabilistic mean and standard deviation shapes do not match.")
    if not np.isfinite(prediction_std).all() or np.any(prediction_std < 0):
        raise ValueError("Predictive standard deviations must be finite and non-negative.")
    return prediction, prediction_std


def _outer_coefficients(
    estimator: Any,
    *,
    model_id: str,
    outer_lot: Any,
    feature_columns: Sequence[str],
    target_name: str,
) -> pd.DataFrame:
    try:
        evaluation = importlib.import_module("bosch_vm.evaluation")
    except ImportError:
        return pd.DataFrame()
    extractor = getattr(evaluation, "extract_outer_fold_coefficients", None)
    if extractor is None:
        return pd.DataFrame()
    try:
        return extractor(
            estimator,
            model_id=model_id,
            outer_lot=outer_lot,
            feature_names=feature_columns,
            target_names=(target_name,),
            on_unsupported="empty",
        )
    except TypeError:
        # A direct DummyRegressor or another non-pipeline estimator has no
        # coefficient representation; this is not an experiment failure.
        return pd.DataFrame()


def _heldout_block_permutation(
    *,
    estimator: Any,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    groups_test: np.ndarray,
    feature_columns: Sequence[str],
    model_id: str,
    outer_lot: Any,
    cfg: ExperimentConfig,
    random_state: int,
) -> pd.DataFrame:
    if cfg.permutation_repetitions <= 0 or tuple(feature_columns) == ("__constant__",):
        return pd.DataFrame()
    try:
        evaluation = importlib.import_module("bosch_vm.evaluation")
    except ImportError:
        return pd.DataFrame()
    permutation_function = getattr(evaluation, "feature_block_permutation_importance", None)
    if permutation_function is None:
        return pd.DataFrame()
    blocks = _default_feature_blocks(feature_columns)
    if not blocks:
        return pd.DataFrame()
    result = permutation_function(
        estimator,
        X_test,
        y_test,
        blocks,
        scoring=cfg.scoring,
        n_repeats=cfg.permutation_repetitions,
        random_state=random_state,
        groups=groups_test,
        feature_names=feature_columns,
    )
    return result.to_long_frame(model_id=model_id, outer_lot=outer_lot)


def _default_feature_blocks(
    feature_columns: Sequence[str],
) -> dict[str, tuple[str, ...]]:
    """Group held-out permutation columns without consulting any target."""

    blocks: dict[str, list[str]] = {}
    for feature in feature_columns:
        if feature == "__constant__":
            continue
        block = str(feature).split("__", maxsplit=1)[0] or "other"
        blocks.setdefault(block, []).append(str(feature))
    return {name: tuple(columns) for name, columns in blocks.items()}


def _fit_final_all_data_model(
    *,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    spec: ProtocolModelSpec,
    cfg: ExperimentConfig,
    model_dir: Path,
) -> dict[str, Any]:
    """Fit a deployment artifact after evaluation; it never contributes OOF metrics."""

    final_dir = model_dir / "final_refit"
    final_dir.mkdir(parents=True, exist_ok=True)
    full_logo = LeaveOneGroupOut()
    inner_splits = list(full_logo.split(X, y, groups=groups))
    search = GridSearchCV(
        estimator=clone(spec.estimator),
        param_grid=spec.param_grid,
        scoring=spec.scoring or cfg.scoring,
        cv=inner_splits,
        refit=True,
        n_jobs=cfg.n_jobs,
        return_train_score=True,
        error_score="raise",
    )
    search.fit(X, y)
    joblib.dump(search.best_estimator_, final_dir / "model.joblib")
    scores = pd.DataFrame(search.cv_results_)
    scores.insert(0, "model_id", spec.model_id)
    scores.insert(1, "n_lot_splits", len(inner_splits))
    _write_dataframe(scores, final_dir / "inner_scores.csv")
    summary = {
        "evaluation_use": False,
        "purpose": "all-data deployment refit after nested OOF evaluation",
        "n_wafers": int(len(y)),
        "n_lots": int(len(pd.unique(groups))),
        "best_index": int(search.best_index_),
        "best_inner_score": float(search.best_score_),
        "best_params": _jsonable(search.best_params_),
        "model_artifact": str(final_dir / "model.joblib"),
    }
    _write_json(summary, final_dir / "best_params.json")
    return summary


def _coerce_config(
    config: ExperimentConfig | Mapping[str, Any] | None,
) -> ExperimentConfig:
    if config is None:
        return ExperimentConfig()
    if isinstance(config, ExperimentConfig):
        return config
    if isinstance(config, Mapping):
        return ExperimentConfig.from_mapping(config)
    raise TypeError("config must be an ExperimentConfig, a mapping, or None.")


def _flatten_project_config(values: Mapping[str, Any]) -> dict[str, Any]:
    """Map the checked-in nested experiment YAML onto config fields."""

    raw = dict(values)
    nested_keys = {"paths", "task", "validation", "features", "models"}
    if not (nested_keys & set(raw)):
        return raw

    flattened = {
        key: value for key, value in raw.items() if key in ExperimentConfig.__dataclass_fields__
    }
    paths = raw.get("paths") or {}
    task = raw.get("task") or {}
    validation = raw.get("validation") or {}
    if not all(isinstance(section, Mapping) for section in (paths, task, validation)):
        raise ValueError("Project config paths/task/validation sections must be mappings.")
    aliases = {
        "raw_dir": paths.get("raw_dir"),
        "prepared_dir": paths.get("processed_dir"),
        "output_dir": paths.get("results_dir"),
        "experiment_dir": paths.get("experiment_dir"),
        "target_column": task.get("target_name"),
        "primary_model_id": task.get("primary_model"),
        "scoring": validation.get("selection_metric"),
        "bootstrap_replicates": validation.get("cluster_bootstrap_repetitions"),
        "permutation_repetitions": validation.get("permutation_repetitions"),
        "seed": raw.get("seed"),
    }
    flattened.update({key: value for key, value in aliases.items() if value is not None})
    return flattened


def _indexed_frame(frame: Any, index_name: str) -> pd.DataFrame:
    result = _ensure_dataframe(frame, "feature-builder dataframe")
    if index_name in result.columns:
        result = result.set_index(index_name)
    if result.index.name != index_name:
        result.index = result.index.astype(str)
        result.index.name = index_name
    if result.index.has_duplicates:
        raise ValueError(f"{index_name} must be unique in feature-builder output.")
    return result


def _indexed_series(series: Any, index_name: str, value_name: str) -> pd.Series:
    if not isinstance(series, pd.Series):
        raise TypeError("Feature-builder targets must be a pandas Series.")
    result = series.copy()
    result.index.name = index_name
    result.name = value_name
    if result.index.has_duplicates:
        raise ValueError(f"{index_name} must be unique in feature-builder targets.")
    return result


def _coerce_model_spec(spec: Any) -> ProtocolModelSpec:
    if isinstance(spec, ProtocolModelSpec):
        return spec
    if isinstance(spec, Mapping):
        values = dict(spec)
    else:
        values = {
            name: getattr(spec, name)
            for name in (
                "model_id",
                "estimator",
                "param_grid",
                "feature_view",
                "role",
                "scoring",
                "feature_columns",
                "probabilistic",
            )
            if hasattr(spec, name)
        }
    required = {"model_id", "estimator", "param_grid"}
    missing = sorted(required - set(values))
    if missing:
        raise TypeError(f"Model specification is missing required attributes: {missing}")
    if hasattr(spec, "fresh_estimator"):
        values["estimator"] = spec.fresh_estimator()
    if hasattr(spec, "mutable_param_grid"):
        values["param_grid"] = spec.mutable_param_grid()
    elif isinstance(values.get("param_grid"), Mapping):
        values["param_grid"] = {
            name: list(candidates) for name, candidates in values["param_grid"].items()
        }
    if values.get("feature_columns") is not None:
        values["feature_columns"] = tuple(values["feature_columns"])
    return ProtocolModelSpec(**values)


def _load_model_specs(model_specs: Sequence[Any] | None, seed: int) -> Sequence[Any]:
    if model_specs is not None:
        return model_specs
    models_module = importlib.import_module("bosch_vm.models")
    configs = models_module.build_model_configs(random_state=seed)
    return list(configs.values()) if isinstance(configs, Mapping) else list(configs)


def _validate_model_specs(specs: Sequence[ProtocolModelSpec], primary_model_id: str) -> None:
    if not specs:
        raise ValueError("At least one model specification is required.")
    ids = [spec.model_id for spec in specs]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Model IDs must be unique; got {ids}.")
    if primary_model_id not in ids:
        raise ValueError(f"Primary model {primary_model_id!r} is absent from model specs: {ids}.")
    if primary_model_id != "cycle_ridge":
        raise ValueError("The reference model is fixed to 'cycle_ridge'.")
    for spec in specs:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", spec.model_id):
            raise ValueError(f"Unsafe model ID: {spec.model_id!r}")


def _validate_cohort(cohort: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    required = {cfg.id_column, cfg.lot_column}
    missing = sorted(required - set(cohort.columns))
    if missing:
        raise ValueError(f"Cohort is missing required columns: {missing}")
    result = cohort.copy()
    if result[cfg.id_column].isna().any() or result[cfg.lot_column].isna().any():
        raise ValueError("Cohort wafer IDs and lot IDs may not be missing.")
    if result[cfg.id_column].duplicated().any():
        duplicates = result.loc[result[cfg.id_column].duplicated(), cfg.id_column].tolist()
        raise ValueError(f"Cohort must contain one row per wafer; duplicates: {duplicates[:10]}")
    return result


def _validate_feature_table(feature_table: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    feature_table = _ensure_dataframe(feature_table, "feature_table").copy()
    required = {cfg.id_column, cfg.lot_column, cfg.target_column}
    missing = sorted(required - set(feature_table.columns))
    if missing:
        raise ValueError(f"Feature table is missing required columns: {missing}")
    if feature_table.empty:
        raise ValueError("Feature table is empty.")
    if feature_table[cfg.id_column].isna().any():
        raise ValueError("Wafer IDs may not be missing.")
    if feature_table[cfg.id_column].duplicated().any():
        duplicates = feature_table.loc[
            feature_table[cfg.id_column].duplicated(keep=False), cfg.id_column
        ].astype(str)
        raise ValueError(
            "Feature table must contain exactly one row per wafer; duplicate IDs include "
            f"{sorted(duplicates.unique())[:10]}."
        )
    if feature_table[cfg.lot_column].isna().any():
        raise ValueError("Lot IDs may not be missing.")
    if feature_table[cfg.lot_column].nunique() < 3:
        raise ValueError("Nested leave-one-lot-out evaluation requires at least three lots.")
    target = pd.to_numeric(feature_table[cfg.target_column], errors="coerce")
    if target.isna().any() or not np.isfinite(target.to_numpy(dtype=float)).all():
        raise ValueError("Target must be finite and numeric for every wafer.")
    feature_table[cfg.target_column] = target.astype(float)
    return feature_table.reset_index(drop=True)


def _build_outer_split_manifest(table: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    groups = table[cfg.lot_column].to_numpy()
    outer = LeaveOneGroupOut()
    rows: list[pd.DataFrame] = []
    split_input = np.zeros((len(table), 1), dtype=float)
    for fold, (_, test_index) in enumerate(
        outer.split(split_input, table[cfg.target_column], groups=groups), start=1
    ):
        test_lots = pd.unique(groups[test_index])
        if len(test_lots) != 1:
            raise AssertionError("An outer LOGO fold must hold out exactly one lot.")
        frame = table.iloc[test_index][
            [cfg.id_column, cfg.lot_column]
            + ([cfg.wafer_column] if cfg.wafer_column in table.columns else [])
        ].copy()
        frame["outer_fold"] = fold
        frame["outer_test_lot"] = test_lots[0]
        rows.append(frame)
    manifest = pd.concat(rows, ignore_index=True)
    if len(manifest) != len(table) or manifest[cfg.id_column].duplicated().any():
        raise AssertionError("Outer LOGO did not assign each wafer to exactly one test fold.")
    return manifest


def _resolve_feature_columns(
    *,
    table: pd.DataFrame,
    spec: ProtocolModelSpec,
    cfg: ExperimentConfig,
    metadata: Mapping[str, Any],
) -> tuple[tuple[str, ...], str]:
    if spec.feature_columns is not None:
        columns = tuple(spec.feature_columns)
        source = "model_spec.feature_columns"
    elif isinstance(spec.feature_view, Sequence) and not isinstance(spec.feature_view, str):
        columns = tuple(str(column) for column in spec.feature_view)
        source = "model_spec.feature_view_columns"
    else:
        view = str(spec.feature_view)
        configured_views = {
            **{
                str(name): tuple(columns)
                for name, columns in (metadata.get("feature_views") or {}).items()
            },
            **{str(name): tuple(columns) for name, columns in cfg.feature_views.items()},
        }
        if view in configured_views:
            columns = configured_views[view]
            source = "feature_view_mapping"
        elif view.lower() in {"none", "constant", "dummy"}:
            columns = ()
            source = "constant_dummy_view"
        elif view.lower() == "context":
            columns = tuple(column for column in cfg.context_columns if column in table.columns)
            source = "config.context_columns"
        else:
            resolved = _feature_columns_from_module(table, view)
            if resolved is not None:
                columns = tuple(resolved)
                source = "bosch_vm.features resolver"
            else:
                excluded = {
                    cfg.id_column,
                    cfg.lot_column,
                    cfg.wafer_column,
                    cfg.target_column,
                    "has_target",
                    "si_etch",
                    "stepheight",
                    "postox_thickness",
                    "postox_thickness_nan",
                    "oxide_etch",
                }
                columns = tuple(column for column in table.columns if column not in excluded)
                source = "fallback_all_nonprotected_columns"

    missing = sorted(set(columns) - set(table.columns))
    if missing:
        raise ValueError(f"Model {spec.model_id!r} references missing feature columns: {missing}")
    forbidden = {
        cfg.id_column,
        cfg.lot_column,
        cfg.target_column,
        "si_etch",
        "stepheight",
        "postox_thickness",
        "postox_thickness_nan",
        "oxide_etch",
    }
    leaked = sorted(set(columns) & forbidden)
    if leaked:
        raise ValueError(
            f"Model {spec.model_id!r} includes identifiers, grouping variables, or "
            f"post-metrology target components as predictors: {leaked}"
        )
    if len(columns) != len(set(columns)):
        raise ValueError(f"Model {spec.model_id!r} has duplicate feature columns.")
    return columns, source


def _feature_columns_from_module(table: pd.DataFrame, view: str) -> Sequence[str] | None:
    try:
        module = importlib.import_module("bosch_vm.features")
    except ImportError:
        return None
    for function_name in ("feature_columns_for_view", "get_feature_columns"):
        function = getattr(module, function_name, None)
        if function is None:
            continue
        parameters = inspect.signature(function).parameters
        if list(parameters)[:2] == ["table", "view"]:
            return function(table, view)
        if list(parameters)[:2] == ["view", "table"]:
            return function(view, table)
        try:
            return function(table=table, view=view)
        except TypeError:
            # Compatibility fallback for a one-argument resolver.
            return function(view)
    return None


def _validate_oof_coverage(
    predictions: pd.DataFrame,
    table: pd.DataFrame,
    cfg: ExperimentConfig,
    model_id: str,
) -> pd.DataFrame:
    expected = table[cfg.id_column].astype(str)
    observed = predictions[cfg.id_column].astype(str)
    counts = observed.value_counts()
    duplicates = counts[counts != 1]
    missing = sorted(set(expected) - set(observed))
    unexpected = sorted(set(observed) - set(expected))
    if len(predictions) != len(table) or not duplicates.empty or missing or unexpected:
        raise AssertionError(
            f"OOF coverage failure for {model_id}: rows={len(predictions)} "
            f"expected={len(table)}, nonunit_counts={duplicates.to_dict()}, "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}."
        )
    predictions = predictions.copy()
    predictions["oof_occurrence"] = 1
    return predictions.sort_values("outer_fold", kind="stable").reset_index(drop=True)


def _compute_metrics(predictions: pd.DataFrame, cfg: ExperimentConfig) -> dict[str, float | int]:
    try:
        evaluation_module = importlib.import_module("bosch_vm.evaluation")
    except ImportError:
        evaluation_module = None
    evaluator = (
        getattr(evaluation_module, "evaluate_predictions", None)
        if evaluation_module is not None
        else None
    )
    if evaluator is not None:
        evaluated = evaluator(
            predictions["y_true"],
            predictions["y_pred"],
            predictions[cfg.lot_column],
        )
        return {
            "n_wafers": int(evaluated.overall["n_samples"]),
            "n_lots": int(evaluated.overall["n_lots"]),
            "mae_um": float(evaluated.overall["mae"]),
            "rmse_um": float(evaluated.overall["rmse"]),
            "macro_lot_mae_um": float(evaluated.macro["mae"]),
            "macro_lot_rmse_um": float(evaluated.macro["rmse"]),
            "bias_um": float(evaluated.overall["bias"]),
            "r2_oof": float(evaluated.overall["r2"]),
            "median_absolute_error_um": float(evaluated.overall["median_absolute_error"]),
        }
    external = _optional_evaluation_function("compute_metrics")
    if external is not None:
        result = _call_with_supported_keywords(external, predictions, dummy_pred=None)
        if isinstance(result, Mapping):
            normalized = {str(key): _jsonable(value) for key, value in result.items()}
            # Preserve a stable core schema even if the external helper uses aliases.
            fallback = _internal_metrics(predictions, cfg)
            return {**fallback, **normalized}
    return _internal_metrics(predictions, cfg)


def _internal_metrics(predictions: pd.DataFrame, cfg: ExperimentConfig) -> dict[str, float | int]:
    y_true = predictions["y_true"].to_numpy(dtype=float)
    y_pred = predictions["y_pred"].to_numpy(dtype=float)
    residual = y_pred - y_true
    lot_mae = predictions.groupby(cfg.lot_column, sort=False).apply(
        lambda frame: mean_absolute_error(frame["y_true"], frame["y_pred"]),
        include_groups=False,
    )
    lot_rmse = predictions.groupby(cfg.lot_column, sort=False).apply(
        lambda frame: math.sqrt(mean_squared_error(frame["y_true"], frame["y_pred"])),
        include_groups=False,
    )
    r2 = float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan")
    return {
        "n_wafers": int(len(predictions)),
        "n_lots": int(predictions[cfg.lot_column].nunique()),
        "mae_um": float(mean_absolute_error(y_true, y_pred)),
        "rmse_um": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "macro_lot_mae_um": float(lot_mae.mean()),
        "macro_lot_rmse_um": float(lot_rmse.mean()),
        "bias_um": float(np.mean(y_pred - y_true)),
        "r2_oof": r2,
        "median_absolute_error_um": float(np.median(np.abs(residual))),
    }


def _metrics_by_lot(predictions: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    try:
        evaluation_module = importlib.import_module("bosch_vm.evaluation")
    except ImportError:
        evaluation_module = None
    evaluator = (
        getattr(evaluation_module, "evaluate_predictions", None)
        if evaluation_module is not None
        else None
    )
    if evaluator is not None:
        evaluated = evaluator(
            predictions["y_true"],
            predictions["y_pred"],
            predictions[cfg.lot_column],
        )
        per_lot = evaluated.per_lot.rename(
            columns={
                "lot": cfg.lot_column,
                "n_samples": "n_wafers",
                "mae": "mae_um",
                "rmse": "rmse_um",
                "bias": "bias_um",
                "median_absolute_error": "median_absolute_error_um",
                "r2": "r2",
            }
        )
        return per_lot
    external = _optional_evaluation_function("metrics_by_lot")
    if external is not None:
        result = _call_with_supported_keywords(external, predictions)
        if isinstance(result, pd.DataFrame):
            return result

    rows: list[dict[str, Any]] = []
    for lot, frame in predictions.groupby(cfg.lot_column, sort=True):
        y_true = frame["y_true"].to_numpy(dtype=float)
        y_pred = frame["y_pred"].to_numpy(dtype=float)
        rows.append(
            {
                cfg.lot_column: lot,
                "n_wafers": int(len(frame)),
                "target_mean_um": float(np.mean(y_true)),
                "target_std_um": float(np.std(y_true, ddof=1)) if len(frame) > 1 else np.nan,
                "mae_um": float(mean_absolute_error(y_true, y_pred)),
                "rmse_um": float(math.sqrt(mean_squared_error(y_true, y_pred))),
                "bias_um": float(np.mean(y_pred - y_true)),
            }
        )
    return pd.DataFrame(rows)


def _cluster_bootstrap(
    predictions: pd.DataFrame, cfg: ExperimentConfig, model_id: str
) -> pd.DataFrame:
    try:
        evaluation_module = importlib.import_module("bosch_vm.evaluation")
    except ImportError:
        evaluation_module = None
    bootstrap_evaluator = (
        getattr(evaluation_module, "cluster_bootstrap_evaluation", None)
        if evaluation_module is not None
        else None
    )
    if bootstrap_evaluator is not None and cfg.bootstrap_replicates > 0:
        result = bootstrap_evaluator(
            predictions["y_true"],
            predictions["y_pred"],
            predictions[cfg.lot_column],
            n_bootstrap=cfg.bootstrap_replicates,
            random_state=cfg.seed,
            expected_n_lots=None,
        )
        return result.summary
    external = _optional_evaluation_function("cluster_bootstrap")
    if external is not None:
        result = _call_with_supported_keywords(
            external,
            predictions,
            group_col=cfg.lot_column,
            lot_col=cfg.lot_column,
            n_bootstrap=cfg.bootstrap_replicates,
            n_boot=cfg.bootstrap_replicates,
            seed=cfg.seed,
            random_state=cfg.seed,
        )
        if isinstance(result, pd.DataFrame):
            return result
        if isinstance(result, Mapping):
            return pd.DataFrame([result])

    if cfg.bootstrap_replicates <= 0:
        return pd.DataFrame(columns=["metric", "estimate", "lower_95", "upper_95", "replicates"])
    lot_values = list(pd.unique(predictions[cfg.lot_column]))
    seed_offset = int(hashlib.sha256(model_id.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(cfg.seed + seed_offset)
    draws: dict[str, list[float]] = {
        "mae_um": [],
        "rmse_um": [],
        "macro_lot_mae_um": [],
    }
    frames = {lot: predictions.loc[predictions[cfg.lot_column] == lot] for lot in lot_values}
    for _ in range(cfg.bootstrap_replicates):
        sampled_lots = rng.choice(lot_values, size=len(lot_values), replace=True)
        sampled_frames = [frames[lot] for lot in sampled_lots]
        sampled = pd.concat(sampled_frames, ignore_index=True)
        draws["mae_um"].append(float(mean_absolute_error(sampled["y_true"], sampled["y_pred"])))
        draws["rmse_um"].append(
            float(math.sqrt(mean_squared_error(sampled["y_true"], sampled["y_pred"])))
        )
        draws["macro_lot_mae_um"].append(
            float(
                np.mean(
                    [
                        mean_absolute_error(frames[lot]["y_true"], frames[lot]["y_pred"])
                        for lot in sampled_lots
                    ]
                )
            )
        )

    point = _internal_metrics(predictions, cfg)
    rows = []
    for metric, values in draws.items():
        array = np.asarray(values, dtype=float)
        rows.append(
            {
                "metric": metric,
                "estimate": point[metric],
                "lower_95": float(np.quantile(array, 0.025)),
                "upper_95": float(np.quantile(array, 0.975)),
                "replicates": int(cfg.bootstrap_replicates),
                "cluster": cfg.lot_column,
            }
        )
    return pd.DataFrame(rows)


def _add_primary_comparisons(
    *,
    comparison: pd.DataFrame,
    predictions: Mapping[str, pd.DataFrame],
    cfg: ExperimentConfig,
) -> pd.DataFrame:
    result = comparison.copy()
    primary_row = result.loc[result["model_id"] == cfg.primary_model_id]
    if len(primary_row) != 1:
        raise AssertionError("Comparison must contain exactly one primary model row.")
    primary_macro = float(primary_row.iloc[0]["macro_lot_mae_um"])
    result["delta_macro_lot_mae_vs_primary_um"] = (
        result["macro_lot_mae_um"].astype(float) - primary_macro
    )

    dummy_candidates = [
        model_id
        for model_id in predictions
        if "dummy" in model_id.lower() or "mean" in model_id.lower()
    ]
    if dummy_candidates:
        dummy_id = dummy_candidates[0]
        dummy = predictions[dummy_id][[cfg.id_column, "y_pred"]].rename(
            columns={"y_pred": "dummy_pred"}
        )
        skills: list[float] = []
        for model_id in result["model_id"]:
            model = predictions[model_id][[cfg.id_column, "y_true", "y_pred"]]
            aligned = model.merge(dummy, on=cfg.id_column, validate="one_to_one")
            model_sse = float(np.square(aligned["y_true"] - aligned["y_pred"]).sum())
            dummy_sse = float(np.square(aligned["y_true"] - aligned["dummy_pred"]).sum())
            skills.append(float(1.0 - model_sse / dummy_sse) if dummy_sse > 0 else np.nan)
        result["sse_skill_vs_dummy"] = skills
        result["dummy_model_id"] = dummy_id
    return result


def _paired_lot_differences(
    predictions: Mapping[str, pd.DataFrame], *, primary_model_id: str, cfg: ExperimentConfig
) -> pd.DataFrame:
    primary = predictions[primary_model_id]
    primary_loss = primary.groupby(cfg.lot_column, sort=True)["absolute_error"].mean()
    rows: list[dict[str, Any]] = []
    for model_id, frame in predictions.items():
        loss = frame.groupby(cfg.lot_column, sort=True)["absolute_error"].mean()
        aligned = pd.concat(
            [primary_loss.rename("primary_mae_um"), loss.rename("model_mae_um")], axis=1
        )
        for lot, values in aligned.iterrows():
            rows.append(
                {
                    cfg.lot_column: lot,
                    "primary_model_id": primary_model_id,
                    "model_id": model_id,
                    "primary_mae_um": float(values["primary_mae_um"]),
                    "model_mae_um": float(values["model_mae_um"]),
                    "model_minus_primary_mae_um": float(
                        values["model_mae_um"] - values["primary_mae_um"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def _optional_evaluation_function(name: str) -> Callable[..., Any] | None:
    try:
        module = importlib.import_module("bosch_vm.evaluation")
    except ImportError:
        return None
    function = getattr(module, name, None)
    return function if callable(function) else None


def _call_with_supported_keywords(
    function: Callable[..., Any], first_argument: Any, **candidate_kwargs: Any
) -> Any:
    signature = inspect.signature(function)
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    kwargs = {
        name: value
        for name, value in candidate_kwargs.items()
        if accepts_kwargs or name in signature.parameters
    }
    return function(first_argument, **kwargs)


def _load_feature_metadata(prepared_dir: Path) -> Mapping[str, Any]:
    path = prepared_dir / "feature_metadata.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, Mapping) else {}


def _guard_known_outputs(paths: Sequence[Path], *, overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        rendered = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite existing experiment artifacts: {rendered}. "
            "Use a new output directory or explicitly enable overwrite."
        )


def _write_dataframe(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(value), handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if callable(value):
        return getattr(value, "__name__", repr(value))
    return value


def _config_manifest(cfg: ExperimentConfig) -> Mapping[str, Any]:
    return _jsonable(asdict(cfg))


def _environment_manifest() -> Mapping[str, Any]:
    try:
        import sklearn

        sklearn_version = sklearn.__version__
    except ImportError:  # pragma: no cover - sklearn is a declared dependency
        sklearn_version = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn_version,
    }


def _make_logger(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"bosch_vm.experiment.{id(path)}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def _close_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _optional_sha256(path: Path) -> str | None:
    return _sha256(path) if path.exists() else None


def _safe_name(value: Any) -> str:
    rendered = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    if not rendered:
        raise ValueError(f"Cannot create a safe artifact name from {value!r}.")
    return rendered


def _feature_view_label(view: str | Sequence[str]) -> str:
    if isinstance(view, str):
        return view
    return ",".join(str(value) for value in view)


def _ensure_dataframe(value: Any, label: str) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"{label} must be a pandas DataFrame; got {type(value).__name__}.")
    return value.copy()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "ExperimentConfig",
    "PreparedArtifacts",
    "ProtocolModelSpec",
    "RunArtifacts",
    "prepare",
    "run",
]
