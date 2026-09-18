# Phase 3 — Untuned Baseline Results

Single stratified 5-fold CV (country + target stratified), class-weight imbalance handling only, F1 at the default 0.5 threshold (not yet tuned -- see `reports/ablation_study_report.md` for Phase 4's tuned results).

## Results table

| Model | Imbalance strategy | PR-AUC | F1 | ROC-AUC |
|---|---|---|---|---|
| LogisticRegression | class-weight | 0.559 ± 0.019 | 0.496 ± 0.008 | 0.852 ± 0.006 |
| XGBoost | class-weight | 0.551 ± 0.017 | 0.508 ± 0.005 | 0.846 ± 0.005 |
| PyTorchMLP (2-layer) | class-weight | 0.580 ± 0.019 | 0.506 ± 0.013 | 0.863 ± 0.006 |

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
