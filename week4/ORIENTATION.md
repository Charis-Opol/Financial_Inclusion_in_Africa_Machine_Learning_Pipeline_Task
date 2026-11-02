# Orientation: what exists and how Week 4 reuses it

Written at Phase 0, from reading the repo at commit `48462c6`. Nothing outside `week4/` was changed. Numbers quoted from existing reports are tagged `[VERIFY]`, because rule 2 requires numbers in write-ups to come from `week4/results/`. Phases 1 and 2 will regenerate them there.

## What "Week 2 baseline" means here

The repo's own labels differ from the brief. `Implementation_plan.md` organises modelling into Phases 1 to 5, and the README labels Week 3 as explainability. I treat the **"Week 2 baseline" as the tuned nested-CV results from `scripts/run_tuning_and_ablation.py`**, reported in `reports/ablation_study_report.md`: XGBoost/class-weight and the 2-layer PyTorch MLP/class-weight. This is an assumption for you to confirm.

## Dataset

| Item | Location | Notes |
|---|---|---|
| Labelled data | `data/raw/Train_v2.csv` | Gitignored. 23,524 rows `[VERIFY]`. Target `bank_account`. |
| Unlabelled test | `data/raw/Test_v2.csv` | Gitignored. No labels, so not usable for evaluation. |
| Loader | `src/fin_inclusion/data/loader.py` | `DataLoader`, `TARGET_COLUMN`. Key is `uniqueid + country + year`. |
| EDA | `reports/eda_summary.md` | Prevalence per country. |

Ad hoc count from the raw file, to be regenerated in Phase 1 `[VERIFY]`:

| Country | Survey year | Rows | Positives | Prevalence |
|---|---|---|---|---|
| Kenya | 2018 | 6,068 | 1,521 | 25.1% |
| Rwanda | 2016 | 8,735 | 1,003 | 11.5% |
| Tanzania | 2017 | 6,620 | 607 | 9.2% |
| Uganda | 2018 | 2,101 | 181 | 8.6% |

## Baseline code

| Purpose | File | Reuse in Week 4 |
|---|---|---|
| Nested CV | `src/fin_inclusion/evaluation/cv_runner.py` | E1 calls `run_nested_cv` as is. It returns per-fold metrics only and no OOF predictions, so Week 4 needs its own wrapper for OOF output. |
| Metrics | `src/fin_inclusion/evaluation/metrics.py` | Import `compute_classification_metrics`, which uses `average_precision_score`. |
| Reporting | `src/fin_inclusion/evaluation/reporting.py` | Import where it fits (`summarize_metrics`, `compare_paired_pr_auc`). Not yet read in full. |
| XGBoost | `src/fin_inclusion/models/xgboost_model.py` | Import `XGBoostModel`. |
| MLP | `src/fin_inclusion/models/pytorch_mlp.py` | Import `PyTorchMLP` for ERM. It has BCE with sample weights, Adam, and no early stopping, and its training loop has no group-balanced sampling or per-group loss. Group DRO therefore needs a new trainer in `week4/src/`, reusing the architecture if possible. |
| Tuning | `src/fin_inclusion/tuning/optuna_search.py` | `tune_xgboost`, `tune_pytorch_mlp`: TPE, 3 inner folds, objective is pooled mean PR-AUC. RQ3 needs a worst-country objective, so that goes in a new function. |
| Imbalance | `src/fin_inclusion/imbalance/strategies.py` | Import `ClassWeightStrategy`, `NoImbalanceStrategy`. |
| Preprocessing | `src/fin_inclusion/preprocessing/pipeline_factory.py`, `encoders.py`, `cleaning.py` | Import the pipeline builders and fit them inside folds. |
| Experiment script | `scripts/run_tuning_and_ablation.py` | Rerun in Phase 2 for E1, with outputs sent to `week4/results/` and not over `reports/`. Settings: 3 inner folds, 15 XGBoost trials, 8 PyTorch trials, 6 tuning epochs, 20 final epochs. |
| Config | `config/config.yaml` | Seed 42, 5 outer folds. |

## Week 3 SHAP

| File | Reuse |
|---|---|
| `src/fin_inclusion/interpretability/README.md` | Map of the package. |
| `model_adapters.py`, `shap_explainers.py`, `shapley.py`, `grouping.py`, `sampling.py`, `bootstrap.py` in the same folder | E6 reuses adapters, explainers and the grouping to raw features where they fit. The new parts are per-country rankings and the size-matched noise baseline. These modules have not been read line by line yet; that happens before E6. |
| `scripts/run_xai_shap.py`, `reports/xai_decisions_log.md` | Earlier design decisions to stay consistent with. |

## Differences the Week 4 harness has to handle

1. **Baseline preprocessing is fitted before CV.** `scripts/run_tuning_and_ablation.py:169` and `:172` fit the XGBoost and PyTorch pipelines on all training rows and only then run CV. Rule 5 requires fitting inside folds. E1 reproduces the baseline as it stands, and the Week 4 harness fits inside folds. Phase 2 measures the difference.
2. **Imbalance handling differs by model.** In the baseline, the XGBoost class-weight arm uses tuned `scale_pos_weight` while the MLP uses class weights. TabPFN has neither by default, as far as I know so far. Rule 5 asks for one harness, so Phase 3 has to define what "identical" means here. To be settled with the installed TabPFN version.
3. **Tuning effort differs by model** (15 trials against 8). E2 either keeps the baseline settings and says so, or equalises them. Decided in Phase 3.
4. **Environments.** `.venv` lacks torch. `.venv-xai` has torch 2.12.1, xgboost 3.3.0, shap 0.52.0, optuna 5.0.0, scikit-learn 1.7.2. Neither has `tabpfn`. Phase 1 creates a separate environment, with `week4/requirements.txt`, and keeps it out of git with a `week4/.gitignore`, because the root `.gitignore` lists only `.venv/`, `.venv-xai/` and `venv/`.
5. **No GPU here.** `nvidia-smi` is unavailable and the machine has 16 CPU threads. This affects TabPFN runtime, which Phase 1 measures.

## Week 4 layout

```text
week4/
  RESEARCH_FRAMEWORK.md  HYPOTHESES.md  ORIENTATION.md
  DECISIONS.md  RESEARCH_LOG.md  STATUS.md
  src/  notebooks/  tests/
  results/  results/runs/
  figures/
  paper/  literature_index.md  LITERATURE_GAPS.md
```
