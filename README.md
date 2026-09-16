# BOSCH Etch Virtual Metrology

Predicting mean silicon etch depth from the sensor trace of a completed BOSCH etch run.
The data are public: 88 labeled wafers from 10 lots on one 200 mm tool, with 31 process
channels and 89 metrology sites per wafer.

I compared eight model setups using leave-one-lot-out validation. Elastic Net had the
lowest macro-lot MAE, **0.0921 µm**. Ridge, the reference model for the feature analysis,
reached **0.1239 µm**. The models use the full run trace, so predictions are made after
etching, before the physical measurement is returned.

![Error on held-out lots](figures/wafer_mean_si_etch_lolo_v1/model_comparison.png)

## Data

The [source dataset](https://doi.org/10.5281/zenodo.17122442) includes 96 process traces.
88 can be matched to complete metrology maps; those are the runs used here.
Each trace has about 3,200 timestamps at 5 Hz and covers 100 etch/passivation cycles.
The target is the arithmetic mean of the 89 published `si_etch` values for that wafer.

Here is one run, `2024-07-02_01`. On the left are two gas-flow readbacks and the detected
cycle boundaries. On the right are the measurements used to calculate its target.

![Process trace and metrology for one wafer](figures/wafer_mean_si_etch_lolo_v1/wafer_run_example.png)

The 89 sites and thousands of timestamps still give us one labeled sample per wafer.
Wafers in a lot share conditioning and chamber history, so I keep each lot together
when splitting the data.

File schemas, channel definitions, missing runs, and the target calculation are in
[data/DATA_STRUCTURE.md](data/DATA_STRUCTURE.md).

## Features and validation

I keep the longest contiguous timestamp block in each trace, detect the repeating cycles,
and summarize cycles 2–99 to reduce the effect of startup and shutdown. This gives 81
features per wafer:

| Features | Count |
| --- | ---: |
| Cycle summaries of gas, pressure, helium, and RF readbacks | 56 |
| Phase durations, cycle timing, and startup behavior | 10 |
| Levels and drift of thermal or slow-flow channels | 10 |
| Conditioning and wafer order | 5 |

The cycle summaries capture signal level, the contrast between phases, variability of
that contrast, and change from early to late cycles. `Gas4Flow` and `Gas5Flow` mark the
phases. Their chemical identities are not given in the public files.

There are only 88 labeled runs, so I used cycle summaries with regularized regression
and a few nonlinear baselines. I haven't trained a sequence neural network on these data.

Each outer fold holds out one lot and trains on the other nine. Hyperparameter tuning
uses a second leave-one-lot-out loop within those nine lots. Preprocessing is fitted
inside the training folds, including imputation, scaling, and PCA where used.

Macro-lot MAE is the average of the ten held-out-lot MAEs; each lot gets equal weight.
The intervals in the figure come from 10,000 bootstrap samples of whole lots.

Implementation: [features.py](src/bosch_vm/features.py),
[config.yaml](experiments/mean_si_etch/config.yaml),
[experiment plan](experiments/mean_si_etch/EXPERIMENT_PLAN.md).

## Results

All scores below use outer-fold predictions.

| Model | Macro-lot MAE (µm) | Pooled RMSE (µm) | Out-of-fold R² |
| --- | ---: | ---: | ---: |
| Elastic Net | 0.0921 | 0.1151 | 0.917 |
| PLS | 0.1112 | 0.1415 | 0.875 |
| Cycle-feature Ridge | 0.1239 | 0.1556 | 0.848 |
| Extra Trees | 0.1400 | 0.1640 | 0.832 |
| PCA + Gaussian process | 0.1633 | 0.2101 | 0.724 |
| RBF-SVR | 0.1756 | 0.2209 | 0.694 |
| Context-only Ridge | 0.1958 | 0.2447 | 0.625 |
| Training-mean baseline | 0.3475 | 0.4044 | -0.024 |

Elastic Net beat Ridge on 9 of 10 lots. PLS also did well; the nonlinear models did not
improve on the linear models in this comparison. These model families were compared on
the same ten outer folds, so the ranking still needs checking on new lots.

A few results from the Ridge analysis:

- Using only conditioning and wafer order raised macro-lot MAE by 0.0719 µm. Error was
  higher on all ten lots.
- Removing RF/power features raised macro-lot MAE by 0.0519 µm, with higher error on
  nine lots. These measurements help prediction; the ablation doesn't tell us what
  would happen if an RF setting were changed.
- Switching to `RobustScaler` lowered macro-lot MAE by 0.0188 µm. This was a sensitivity
  check, so the scaler needs to be included in a new nested comparison before choosing it.

The [report](reports/wafer_mean_si_etch_lolo_v1.md) has per-lot errors, residual plots,
coefficient analysis, and uncertainty checks. Run notes are in the
[experiment log](experiments/mean_si_etch/EXPERIMENT_LOG.md).

## Limits

The data come from one tool and only ten lots. Conditioning, lot, and date are partly
confounded. I haven't tested transfer to another tool or new production lots.

This model predicts a wafer mean from a finished run. It doesn't predict the full spatial
map or choose recipe settings. The inputs are recorded readbacks, not a table of
independently varied control settings.

The target also has measurement uncertainty: silicon etch is derived from step-height
and post-oxide measurements, and about 2% of the post-oxide site values were interpolated
after fit failures.

The Gaussian-process uncertainty estimates need more work. Nominal 95% intervals covered
only 84.1% of held-out wafers. I wouldn't use those intervals to decide which physical
measurements to skip.

Next I'd check the model and scaler choices on new lots. Within the existing data, the
89-site spatial map and the unused optical emission spectra are possible extensions;
neither is modeled here yet.

## Running it

Python 3.11 or newer:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Download the source files and check their published hashes:

```bash
python scripts/download_zenodo.py
```

Build the features, run the models and sensitivity checks, then generate the report:

```bash
bosch-vm prepare --config experiments/mean_si_etch/config.yaml --force
bosch-vm run --config experiments/mean_si_etch/config.yaml --n-jobs -1 --force
bosch-vm sensitivity --config experiments/mean_si_etch/config.yaml --n-jobs -1 --force
python scripts/generate_analysis_artifacts.py
```

Tests and lint:

```bash
pytest
ruff check src tests scripts
```

Raw files stay unchanged. The repository includes compact result tables and figures;
feature tables, fitted models, and full tuning outputs are generated locally.

## Files

- [src/bosch_vm/](src/bosch_vm/): data loading, features, models, evaluation, and reporting
- [experiments/](experiments/): configuration and run notes
- [tests/](tests/): data checks, split and leakage checks, metrics, CLI, and reporting tests
- [results/](results/): saved predictions, metrics, and split assignments
- [references/](references/): papers and documentation used for this analysis

## Source and license

Sayyed et al., *A Multi-Model Dataset for BOSCH Plasma-Etching: Optical Emission Spectra,
Process Parameters, and Wafer Measurements for Data-Driven Plasma Modeling* (2025),
[Zenodo 17122442](https://doi.org/10.5281/zenodo.17122442), CC BY 4.0.

[Data provenance](data/PROVENANCE.md) lists the files, checksums, and exclusions.
The download script retrieves raw data from Zenodo; the repository does not redistribute it.
Code is [MIT licensed](LICENSE). See [NOTICE.md](NOTICE.md) and [CITATION.cff](CITATION.cff).
