# Modeling Sources

These sources informed the dataset interpretation, model choices, grouped evaluation, and
feature-importance cautions used in the benchmark.

## Dataset

- Sayyed et al., [A Multi-Model Dataset for BOSCH Plasma-Etching](https://doi.org/10.5281/zenodo.17122442).
  The Zenodo record supplies the process traces, metrology tables, experiment description,
  wafer layout, and source checksums.

## Virtual metrology and feature-based models

- Khan, Moyne, and Tilbury, [Virtual metrology and feedback control using recursive
  PLS](https://doi.org/10.1016/j.jprocont.2008.04.014). This supports PLS as a compact baseline
  for correlated process measurements.
- Kang et al., [A virtual metrology system for semiconductor
  manufacturing](https://doi.org/10.1016/j.eswa.2009.05.053).
- Suthar et al., [Feature-based virtual
  metrology](https://doi.org/10.1016/j.compchemeng.2019.05.016). This work motivates converting
  batch traces into fixed process features rather than treating timestamps as independent
  samples.
- Susto et al., [Least-angle regression and clustering for virtual
  metrology](https://doi.org/10.1002/asmb.1948).

## Comparison models

- Geurts, Ernst, and Wehenkel, [Extremely randomized
  trees](https://doi.org/10.1007/s10994-006-6226-1).
- Drucker et al., [Support vector
  regression](https://papers.neurips.cc/paper/1996/hash/d38901788c533e8286cb6400b40b386d-Abstract.html).

## Validation and interpretation

- Scikit-learn, [Grouped cross-validation and
  LeaveOneGroupOut](https://scikit-learn.org/stable/modules/cross_validation.html#leave-one-group-out).
- Varma and Simon, [Bias in error estimation when using cross-validation for model
  selection](https://brb.nci.nih.gov/techreport/Varma-Simon-CrossValid.pdf). This is the basis
  for separating outer evaluation from inner hyperparameter selection.
- Scikit-learn, [Permutation
  importance](https://scikit-learn.org/stable/modules/permutation_importance.html).
- Scikit-learn, [Causal interpretation warning for predictive
  models](https://scikit-learn.org/stable/auto_examples/inspection/plot_causal_interpretation.html).
  Coefficients and permutation scores in this project are interpreted as predictive
  associations, not intervention effects.
