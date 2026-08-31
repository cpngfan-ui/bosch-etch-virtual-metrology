# Experiment Plan: Wafer-Mean Silicon Etch Depth

## Prediction task

Use the telemetry from one completed BOSCH etch run, together with context known before the
run, to estimate that wafer's mean final silicon etch depth before physical metrology is
available.

For wafer \(i\), the response is

\[
y_i = \frac{1}{89}\sum_{j=1}^{89}\texttt{si\_etch}_{ij}.
\]

This is the arithmetic mean of the published 89-site map, not an area-weighted wafer mean.
The deployment point is after etching and before ex-situ metrology, so the full process trace
is available. The experiment does not address pre-run recipe optimization.

## Cohort

- 88 wafer runs with matched process telemetry and 89-site metrology
- 10 lot/date groups
- 31 process channels shared by all runs
- about 3,200 time samples per run at a nominal 5 Hz
- no intermediate etch-depth measurement for individual BOSCH cycles

One wafer run is one labeled row in the model table. Dependence among wafers from the same
lot is handled by grouped validation.

## Predictors

The full feature view has 81 columns:

| Block | Count | Construction |
|---|---:|---|
| Cycle response | 56 | Four summaries for each of 14 process channels |
| Timing | 10 | Cycle period, phase duration, active span, and timing irregularity |
| Slow state | 10 | Level and late-minus-early change for five slow channels |
| Context | 5 | Conditioning count, conditioning surface, and wafer order |

The feature extractor keeps the longest contiguous 5 Hz block and uses `Gas4Flow` to locate
100 cycles. `Gas4Flow` and `Gas5Flow` are treated as phase markers; the public dictionary does
not identify their chemical assignments. Stable-cycle summaries use cycles 2 through 99 to
reduce startup and shutdown influence. Lot number, date, and experiment key are not model
inputs.

## Validation

The outer loop is ten-fold leave-one-lot-out evaluation. In each fold, all wafers from one
lot are held out and the other nine lots form the training set. This produces one held-out
prediction for each of the 88 wafers.

Hyperparameters are selected inside each outer-training set with a second leave-one-lot-out
loop. Imputation, variance filtering, scaling, and PCA are fitted only on the corresponding
training fold. The inner selection score is mean absolute error.

The main summary is macro-lot MAE:

\[
\frac{1}{10}\sum_{\ell=1}^{10}\operatorname{MAE}_{\ell}.
\]

Additional summaries are pooled RMSE, pooled out-of-fold \(R^2\), bias, worst-lot MAE, and
lot-cluster bootstrap intervals. Lot-level results remain important because there are only
ten validation groups and their sizes are unequal.

## Model comparison

| ID | Input | Model | Purpose |
|---|---|---|---|
| `dummy_mean` | Ignored | Training-fold mean | Minimum baseline |
| `context_ridge` | 5 context features | Ridge | Telemetry ablation |
| `cycle_ridge` | All 81 features | Ridge | Reference regularized linear model |
| `pls` | All 81 features | PLS regression | Latent linear model for correlated inputs |
| `elastic_net` | All 81 features | Elastic Net | Sparse regularized linear comparison |
| `rbf_svr` | All 81 features | RBF support-vector regression | Smooth nonlinear comparison |
| `pca_gpr` | PCA of all 81 features | Matérn Gaussian process | Nonlinear prediction with model-based uncertainty |
| `extra_trees` | All 81 features | Constrained Extra Trees | Tree-interaction comparison |

The exact search grids, seed, paths, and feature settings are stored in `config.yaml` and in
the generated experiment manifests. `cycle_ridge` is the reference model used for detailed
feature and sensitivity diagnostics; the cross-validated comparison still reports every
model.

## Diagnostics

- context-only versus telemetry-plus-context performance
- held-out error by lot and wafer order
- coefficient sign and magnitude across overlapping outer folds
- held-out feature-block permutation for model reliance
- input-block and scaling sensitivity tests
- sensitivity of the target to interpolated post-oxide measurements
- empirical coverage of Gaussian-process intervals

Coefficients and permutation scores describe model association, not causal effects or
controllable recipe settings.

## Sequence-model decision

The roughly 3,200 timestamps in a trace share one final wafer label. With 88 labeled runs
across 10 lots, a raw LSTM or Transformer would add substantially more parameters and tuning
choices than this data can support. This benchmark therefore preserves time structure with
cycle-aligned summaries and low-capacity models.
