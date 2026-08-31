# Results

The tracked files under `wafer_mean_si_etch_lolo_v1/` are the compact outputs needed to check
the published results:

- the eight-model comparison and lot-bootstrap intervals;
- outer out-of-fold predictions and per-lot metrics;
- paired model differences and selected hyperparameters;
- Gaussian-process interval diagnostics;
- Ridge coefficient and feature-block summaries;
- sensitivity-analysis results; and
- the exact outer lot split manifest.

Reported performance is calculated from outer leave-one-lot-out predictions. The final
all-data refit is stored separately during a local run and is not used for evaluation.

A full run also writes inner-search tables, fold models, detailed diagnostics, and serialized
estimators under this result directory. Those larger generated artifacts are excluded from
Git and can be rebuilt from the experiment configuration and `src/bosch_vm/`:

```bash
bosch-vm run --config experiments/mean_si_etch/config.yaml --n-jobs -1 --force
bosch-vm sensitivity --config experiments/mean_si_etch/config.yaml --n-jobs -1 --force
```
