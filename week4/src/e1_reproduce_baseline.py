"""E1: reproduce the baseline XGBoost and MLP, and measure the effect of pre-CV preprocessing.

Two variants of the two baseline arms (XGBoost/class-weight, 2-layer MLP/class-weight), seed 42:

  as_reported  The baseline's own code path. Imports `run_xgboost_arm` and `run_pytorch_arm` from
               scripts/run_tuning_and_ablation.py and calls `run_nested_cv` unchanged, with
               preprocessing fitted on all labelled rows before CV, as the baseline did.
  in_fold      Same splits, tuning budget and threshold step, but preprocessing is fitted on each
               outer training fold only. Rule 5 requires this for all new Week 4 work.

The baseline script's `main()` is never called, because it writes to reports/. Outputs:
  week4/results/baseline_reproduction.csv   long format, one row per arm/variant/scope/fold/metric
  week4/results/runs/e1_<arm>_<variant>.json

Run from the repo root:
  week4/.venv/Scripts/python.exe week4/src/e1_reproduce_baseline.py
"""
from __future__ import annotations

import importlib.util
import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from fin_inclusion.evaluation.cv_runner import (
    POSITIVE_LABEL,
    NestedFoldResult,
    _stratify_key,
)
from fin_inclusion.evaluation.metrics import compute_classification_metrics, compute_confusion_matrix
from fin_inclusion.evaluation.threshold_selection import select_f1_maximizing_threshold
from fin_inclusion.imbalance.strategies import ClassWeightStrategy, NoImbalanceStrategy
from fin_inclusion.models.pytorch_mlp import PyTorchMLP
from fin_inclusion.models.xgboost_model import XGBoostModel
from fin_inclusion.preprocessing.pipeline_factory import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    build_pytorch_pipeline,
    build_xgboost_pipeline,
)
from fin_inclusion.tuning.optuna_search import tune_pytorch_mlp, tune_xgboost
from run_record import REPO_ROOT, RESULTS_DIR, save_run

SEED = 42
N_OUTER = 5
ARM_KEYS = {"xgboost_class_weight": "xgb_cw", "mlp_2layer_class_weight": "pt2_cw"}
PT_NUMERIC_COLUMNS = NUMERIC_COLUMNS + ["household_size_outlier"]


def load_baseline_script():
    """Imports the baseline script as a module. Its `main()` is guarded, so nothing runs or writes."""
    path = REPO_ROOT / "scripts/run_tuning_and_ablation.py"
    spec = importlib.util.spec_from_file_location("baseline_tuning_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def nested_cv_in_fold(train, y, country, tune_and_build, n_outer_folds=N_OUTER, seed=SEED) -> list[NestedFoldResult]:
    """The logic of `run_nested_cv`, except preprocessing is fitted on the outer training fold only.

    `tune_and_build(train_fold, val_fold)` fits the preprocessing and returns the transformed frames,
    `tune_fn(X, y, country)`, `build_model(params)` and an imbalance strategy factory. Keep this in
    step with fin_inclusion/evaluation/cv_runner.py: only the preprocessing line differs.
    """
    skf = StratifiedKFold(n_splits=n_outer_folds, shuffle=True, random_state=seed)
    results = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(train, _stratify_key(y, country))):
        X_train, X_val, tune_fn, build_model, strategy_factory = tune_and_build(
            train.iloc[train_idx], train.iloc[val_idx]
        )
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        best_params = tune_fn(X_train, y_train, country.iloc[train_idx])

        X_thr_train, X_thr_val, y_thr_train, y_thr_val = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=seed
        )
        X_res, y_res, weight = strategy_factory().apply(X_thr_train, y_thr_train)
        threshold_model = build_model(best_params)
        threshold_model.fit(X_res, y_res, sample_weight=weight)
        threshold, _ = select_f1_maximizing_threshold(
            (y_thr_val == POSITIVE_LABEL).astype(int).to_numpy(), threshold_model.predict_proba(X_thr_val)
        )

        X_res, y_res, weight = strategy_factory().apply(X_train, y_train)
        final_model = build_model(best_params)
        final_model.fit(X_res, y_res, sample_weight=weight)
        scores = final_model.predict_proba(X_val)
        y_binary = (y_val == POSITIVE_LABEL).astype(int).to_numpy()
        countries = country.iloc[val_idx].to_numpy()
        results.append(
            NestedFoldResult(
                fold=fold,
                best_params=best_params,
                threshold=threshold,
                metrics=compute_classification_metrics(y_binary, scores, threshold),
                country_metrics={
                    c: compute_classification_metrics(y_binary[countries == c], scores[countries == c], threshold)
                    for c in sorted(set(countries))
                },
                confusion_matrix=compute_confusion_matrix(y_binary, scores, threshold),
            )
        )
    return results


def xgb_fold_setup(baseline):
    def setup(train_fold, val_fold):
        pipeline = build_xgboost_pipeline().fit(train_fold)
        X_train = pipeline.transform(train_fold).drop(columns=baseline.NON_FEATURE_COLUMNS)
        X_val = pipeline.transform(val_fold).drop(columns=baseline.NON_FEATURE_COLUMNS)

        def tune_fn(X, y, country):
            return tune_xgboost(
                X, y, country, imbalance_strategy_name="class_weight",
                n_inner_folds=baseline.N_INNER_FOLDS, n_trials=baseline.XGB_N_TRIALS, seed=SEED,
            )

        return X_train, X_val, tune_fn, lambda params: XGBoostModel(seed=SEED, **params), NoImbalanceStrategy

    return setup


def mlp_fold_setup(baseline):
    def setup(train_fold, val_fold):
        pipeline = build_pytorch_pipeline().fit(train_fold)
        X_train = pipeline.transform(train_fold).drop(columns=baseline.NON_FEATURE_COLUMNS)
        X_val = pipeline.transform(val_fold).drop(columns=baseline.NON_FEATURE_COLUMNS)
        vocab_sizes = pipeline.named_steps["encode"].vocab_sizes_

        def tune_fn(X, y, country):
            return tune_pytorch_mlp(
                X, y, country,
                categorical_columns=CATEGORICAL_COLUMNS, vocab_sizes=vocab_sizes, numeric_columns=PT_NUMERIC_COLUMNS,
                imbalance_strategy_name="class_weight", n_layers=2,
                n_inner_folds=baseline.N_INNER_FOLDS, n_trials=baseline.PT_N_TRIALS,
                tuning_epochs=baseline.PT_TUNING_EPOCHS, seed=SEED,
            )

        def build_model(params):
            return PyTorchMLP(
                categorical_columns=CATEGORICAL_COLUMNS, vocab_sizes=vocab_sizes, numeric_columns=PT_NUMERIC_COLUMNS,
                epochs=baseline.PT_FINAL_EPOCHS, seed=SEED, **params,
            )

        return X_train, X_val, tune_fn, build_model, ClassWeightStrategy

    return setup


def run_variant(arm: str, variant: str, baseline, train, y, country) -> list[NestedFoldResult]:
    if variant == "in_fold":
        setup = xgb_fold_setup(baseline) if arm == "xgboost_class_weight" else mlp_fold_setup(baseline)
        return nested_cv_in_fold(train, y, country, setup)
    # The baseline's own preprocessing, fitted on all rows before CV (its lines 169 to 175).
    if arm == "xgboost_class_weight":
        X = build_xgboost_pipeline().fit_transform(train).drop(columns=baseline.NON_FEATURE_COLUMNS)
        return baseline.run_xgboost_arm(arm, X, y, country, "class_weight", N_OUTER, SEED)
    pipeline = build_pytorch_pipeline().fit(train)
    X = pipeline.transform(train).drop(columns=baseline.NON_FEATURE_COLUMNS)
    return baseline.run_pytorch_arm(
        arm, X, y, country, CATEGORICAL_COLUMNS, pipeline.named_steps["encode"].vocab_sizes_,
        PT_NUMERIC_COLUMNS, 2, "class_weight", N_OUTER, SEED,
    )


def reported_fold_values(arm: str) -> tuple[list[float], list[float], list[dict]]:
    """Fold PR-AUC, thresholds and tuned parameters saved by the original run, unrounded."""
    raw = json.loads((REPO_ROOT / "reports/ablation_raw_results.json").read_text(encoding="utf-8"))
    key = ARM_KEYS[arm]
    return raw["fold_pr_auc"][key], raw["thresholds"][key], raw["best_params_per_fold"][key]


def to_rows(arm: str, variant: str, results: list[NestedFoldResult]) -> list[dict]:
    reported_pr, reported_thresholds, _ = reported_fold_values(arm)
    rows = []

    def add(scope, fold, metric, value, reported=None):
        rows.append(
            {
                "arm": arm, "variant": variant, "scope": scope, "fold": fold, "metric": metric,
                "reproduced": value, "reported": reported,
                "difference": None if reported is None else value - reported,
            }
        )

    for result in results:
        for metric, value in result.metrics.items():
            add("pooled", result.fold, metric, float(value), reported_pr[result.fold] if metric == "pr_auc" else None)
        add("pooled", result.fold, "threshold", float(result.threshold), reported_thresholds[result.fold])
        for country_name, metrics in result.country_metrics.items():
            for metric, value in metrics.items():
                add(country_name, result.fold, metric, float(value))
    return rows


def main() -> None:
    baseline = load_baseline_script()
    from fin_inclusion.data.loader import TARGET_COLUMN, DataLoader

    train = DataLoader(
        raw_train_path=REPO_ROOT / "data/raw/Train_v2.csv",
        raw_test_path=REPO_ROOT / "data/raw/Test_v2.csv",
    ).load_train()
    y, country = train[TARGET_COLUMN], train["country"]

    output = RESULTS_DIR / "baseline_reproduction.csv"
    all_rows: list[dict] = []
    for arm in ARM_KEYS:
        for variant in ("as_reported", "in_fold"):
            start = time.perf_counter()
            results = run_variant(arm, variant, baseline, train, y, country)
            seconds = time.perf_counter() - start
            all_rows.extend(to_rows(arm, variant, results))
            pd.DataFrame(all_rows).to_csv(output, index=False)
            pr_auc = [r.metrics["pr_auc"] for r in results]
            save_run(
                f"e1_{arm}_{variant}",
                config={"seed": SEED, "n_outer": N_OUTER, "variant": variant, "arm": arm},
                seed=SEED,
                metrics={
                    "seconds": round(seconds, 1),
                    "fold_pr_auc": [float(v) for v in pr_auc],
                    "mean_pr_auc": float(np.mean(pr_auc)),
                    "std_pr_auc": float(np.std(pr_auc)),
                    "thresholds": [float(r.threshold) for r in results],
                    "best_params_per_fold": [r.best_params for r in results],
                },
            )
            print(f"{arm} {variant}: mean PR-AUC {np.mean(pr_auc):.6f} in {seconds:.0f}s", flush=True)


if __name__ == "__main__":
    main()
