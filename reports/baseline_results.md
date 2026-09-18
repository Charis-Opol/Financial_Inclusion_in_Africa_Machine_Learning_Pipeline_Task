# Phase 3 — Untuned Baseline Results

Single stratified 5-fold CV (country + target stratified), class-weight imbalance handling only, F1/precision/recall/accuracy at the default 0.5 threshold (not yet tuned -- see `reports/ablation_study_report.md` for Phase 4's tuned results). Accuracy is included for completeness, not as the metric to optimize against -- see `evaluation/metrics.py`'s module docstring for why it's misleading at this dataset's ~14% positive rate.

## Results table

| Model | Imbalance strategy | PR-AUC | ROC-AUC | F1 | Precision | Recall | Accuracy |
|---|---|---|---|---|---|---|---|
| LogisticRegression | class-weight | 0.559 ± 0.019 | 0.852 ± 0.006 | 0.496 ± 0.008 | 0.374 ± 0.007 | 0.737 ± 0.012 | 0.789 ± 0.004 |
| XGBoost | class-weight | 0.551 ± 0.017 | 0.846 ± 0.005 | 0.508 ± 0.005 | 0.399 ± 0.005 | 0.700 ± 0.011 | 0.809 ± 0.003 |
| PyTorchMLP (2-layer) | class-weight | 0.580 ± 0.019 | 0.863 ± 0.006 | 0.506 ± 0.013 | 0.377 ± 0.018 | 0.773 ± 0.020 | 0.787 ± 0.016 |

## Per-country breakdown

### LogisticRegression

| Country | PR-AUC | F1 |
|---|---|---|
| Kenya | 0.643 ± 0.031 | 0.551 ± 0.015 |
| Rwanda | 0.433 ± 0.018 | 0.412 ± 0.011 |
| Tanzania | 0.571 ± 0.043 | 0.495 ± 0.029 |
| Uganda | 0.529 ± 0.057 | 0.460 ± 0.040 |

### XGBoost

| Country | PR-AUC | F1 |
|---|---|---|
| Kenya | 0.610 ± 0.023 | 0.561 ± 0.008 |
| Rwanda | 0.438 ± 0.019 | 0.428 ± 0.005 |
| Tanzania | 0.587 ± 0.030 | 0.515 ± 0.026 |
| Uganda | 0.483 ± 0.062 | 0.454 ± 0.034 |

### PyTorchMLP (2-layer)

| Country | PR-AUC | F1 |
|---|---|---|
| Kenya | 0.645 ± 0.031 | 0.545 ± 0.011 |
| Rwanda | 0.461 ± 0.011 | 0.438 ± 0.010 |
| Tanzania | 0.616 ± 0.024 | 0.522 ± 0.023 |
| Uganda | 0.539 ± 0.059 | 0.467 ± 0.040 |

## Confusion matrices (summed across the 5 outer-validation folds)

### LogisticRegression

| Actual \ Predicted | Pred No | Pred Yes |
|---|---|---|
| Actual No | 16124 | 4088 |
| Actual Yes | 870 | 2442 |

### XGBoost

| Actual \ Predicted | Pred No | Pred Yes |
|---|---|---|
| Actual No | 16722 | 3490 |
| Actual Yes | 994 | 2318 |

### PyTorchMLP (2-layer)

| Actual \ Predicted | Pred No | Pred Yes |
|---|---|---|
| Actual No | 15957 | 4255 |
| Actual Yes | 751 | 2561 |
