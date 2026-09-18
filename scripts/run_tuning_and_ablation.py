"""Phase 4: nested CV hyperparameter tuning, imbalance-strategy ablation,
depth ablation, statistical comparison, and SHAP interpretability.

Requires `notebooks/02_smote_distribution_eda.ipynb` to have already run and
cleared SMOTE for use (Phase 4.3's hard gate) -- it has (see that notebook's
summary): no disqualifying red flag, two documented caveats carried into
`reports/ablation_study_report.md`.

TUNING BUDGET is deliberately modest (documented here, not silently
under-run) for interactive tractability: n_trials/n_inner_folds/epochs below
were calibrated against measured fit times on this dataset/machine so the
full grid finishes in roughly an hour rather than several. The concession is
search *breadth*, not evaluation honesty -- every number this script reports
still comes from genuinely nested CV (`run_nested_cv`): tuning and threshold
selection never touch the outer validation fold they're then scored against.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Callable

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "config"))
from settings import load_settings  # noqa: E402

from fin_inclusion.data.loader import DataLoader, TARGET_COLUMN  # noqa: E402
from fin_inclusion.evaluation.cv_runner import NestedFoldResult, run_nested_cv  # noqa: E402
from fin_inclusion.evaluation.pr_curve import plot_pr_curve_overlay  # noqa: E402
from fin_inclusion.evaluation.reporting import (  # noqa: E402
    aggregate_confusion_matrix,
    compare_paired_pr_auc,
    confusion_matrix_markdown,
    country_table_markdown,
    fold_pr_auc_scores,
    full_results_table_markdown,
    full_results_table_row,
    summarize_country_metrics,
    summarize_metrics,
)
from fin_inclusion.imbalance.strategies import (  # noqa: E402
    BaseImbalanceStrategy,
    ClassWeightStrategy,
    NoImbalanceStrategy,
    SMOTEStrategy,
)
from fin_inclusion.interpretability.shap_analysis import (  # noqa: E402
    compute_shap_explanation,
    global_importance,
    plot_dependence,
    plot_global_importance,
)
from fin_inclusion.models.pytorch_mlp import PyTorchMLP  # noqa: E402
from fin_inclusion.models.xgboost_model import XGBoostModel  # noqa: E402
from fin_inclusion.preprocessing.pipeline_factory import (  # noqa: E402
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    build_pytorch_pipeline,
    build_xgboost_pipeline,
    make_country_target_stratify_key,
)
from fin_inclusion.tuning.optuna_search import tune_pytorch_mlp, tune_xgboost  # noqa: E402

NON_FEATURE_COLUMNS = ["uniqueid", "year", TARGET_COLUMN]

N_INNER_FOLDS = 3
XGB_N_TRIALS = 15
PT_N_TRIALS = 8
PT_TUNING_EPOCHS = 6
PT_FINAL_EPOCHS = 20


def _xgb_strategy_factory(strategy_name: str, seed: int) -> Callable[[], BaseImbalanceStrategy]:
    if strategy_name == "class_weight":
        return NoImbalanceStrategy
    return lambda: SMOTEStrategy(seed=seed)


def _pt_strategy_factory(strategy_name: str, seed: int) -> Callable[[], BaseImbalanceStrategy]:
    if strategy_name == "class_weight":
        return ClassWeightStrategy
    return lambda: SMOTEStrategy(seed=seed)


def run_xgboost_arm(
    label: str,
    X: pd.DataFrame,
    y: pd.Series,
    country: pd.Series,
    strategy_name: str,
    n_outer_folds: int,
    seed: int,
) -> list[NestedFoldResult]:
    def tune_fn(X_train, y_train, country_train):
        return tune_xgboost(
            X_train, y_train, country_train,
            imbalance_strategy_name=strategy_name,
            n_inner_folds=N_INNER_FOLDS, n_trials=XGB_N_TRIALS, seed=seed,
        )

    def build_model(params):
        return XGBoostModel(seed=seed, **params)

    t0 = time.time()
    results = run_nested_cv(
        X=X, y=y, country=country, tune_fn=tune_fn, build_model=build_model,
        imbalance_strategy_factory=_xgb_strategy_factory(strategy_name, seed),
        n_outer_folds=n_outer_folds, seed=seed,
    )
    print(f"[{label}] done in {time.time() - t0:.0f}s")
    return results


def run_pytorch_arm(
    label: str,
    X: pd.DataFrame,
    y: pd.Series,
    country: pd.Series,
    categorical_columns: list[str],
    vocab_sizes: dict[str, int],
    numeric_columns: list[str],
    n_layers: int,
    strategy_name: str,
    n_outer_folds: int,
    seed: int,
) -> list[NestedFoldResult]:
    def tune_fn(X_train, y_train, country_train):
        return tune_pytorch_mlp(
            X_train, y_train, country_train,
            categorical_columns=categorical_columns, vocab_sizes=vocab_sizes, numeric_columns=numeric_columns,
            imbalance_strategy_name=strategy_name, n_layers=n_layers,
            n_inner_folds=N_INNER_FOLDS, n_trials=PT_N_TRIALS, tuning_epochs=PT_TUNING_EPOCHS, seed=seed,
        )

    def build_model(params):
        return PyTorchMLP(
            categorical_columns=categorical_columns, vocab_sizes=vocab_sizes, numeric_columns=numeric_columns,
            epochs=PT_FINAL_EPOCHS, seed=seed, **params,
        )

    t0 = time.time()
    results = run_nested_cv(
        X=X, y=y, country=country, tune_fn=tune_fn, build_model=build_model,
        imbalance_strategy_factory=_pt_strategy_factory(strategy_name, seed),
        n_outer_folds=n_outer_folds, seed=seed,
    )
    print(f"[{label}] done in {time.time() - t0:.0f}s")
    return results


def main() -> None:
    settings = load_settings()
    settings.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    settings.paths.figures_dir.mkdir(parents=True, exist_ok=True)
    seed = settings.seed
    n_outer_folds = settings.cv.n_outer_folds

    loader = DataLoader(raw_train_path=settings.paths.raw_train, raw_test_path=settings.paths.raw_test)
    train = loader.load_train()
    y = train[TARGET_COLUMN]
    country = train["country"]
    print(f"Loaded train: {train.shape}")

    xgb_encoded = build_xgboost_pipeline().fit_transform(train)
    X_xgb = xgb_encoded.drop(columns=NON_FEATURE_COLUMNS)

    pt_pipeline = build_pytorch_pipeline().fit(train)
    X_pt = pt_pipeline.transform(train).drop(columns=NON_FEATURE_COLUMNS)
    vocab_sizes = pt_pipeline.named_steps["encode"].vocab_sizes_
    pt_numeric_columns = NUMERIC_COLUMNS + ["household_size_outlier"]

    all_results: dict[str, list[NestedFoldResult]] = {}

    print("\n=== XGBoost x class-weight ===")
    all_results["xgb_cw"] = run_xgboost_arm("XGBoost/class-weight", X_xgb, y, country, "class_weight", n_outer_folds, seed)
    print("\n=== XGBoost x SMOTE ===")
    all_results["xgb_smote"] = run_xgboost_arm("XGBoost/SMOTE", X_xgb, y, country, "smote", n_outer_folds, seed)

    print("\n=== PyTorch 2-layer x class-weight ===")
    all_results["pt2_cw"] = run_pytorch_arm(
        "PyTorch2/class-weight", X_pt, y, country, CATEGORICAL_COLUMNS, vocab_sizes, pt_numeric_columns,
        2, "class_weight", n_outer_folds, seed,
    )
    print("\n=== PyTorch 2-layer x SMOTE ===")
    all_results["pt2_smote"] = run_pytorch_arm(
        "PyTorch2/SMOTE", X_pt, y, country, CATEGORICAL_COLUMNS, vocab_sizes, pt_numeric_columns,
        2, "smote", n_outer_folds, seed,
    )

    pt2_cw_mean = summarize_metrics(all_results["pt2_cw"])["pr_auc"][0]
    pt2_smote_mean = summarize_metrics(all_results["pt2_smote"])["pr_auc"][0]
    pt_best_strategy = "class_weight" if pt2_cw_mean >= pt2_smote_mean else "smote"
    print(f"\nPyTorch 2-layer best strategy: {pt_best_strategy} (class-weight={pt2_cw_mean:.3f}, smote={pt2_smote_mean:.3f})")

    print(f"\n=== PyTorch 3-layer x {pt_best_strategy} (best strategy from 2-layer) ===")
    all_results["pt3_best"] = run_pytorch_arm(
        "PyTorch3/best", X_pt, y, country, CATEGORICAL_COLUMNS, vocab_sizes, pt_numeric_columns,
        3, pt_best_strategy, n_outer_folds, seed,
    )

    row_labels = {
        "xgb_cw": ("XGBoost", "class-weight"),
        "xgb_smote": ("XGBoost", "resampling (SMOTE)"),
        "pt2_cw": ("PyTorch (2-layer)", "class-weight"),
        "pt2_smote": ("PyTorch (2-layer)", "resampling (SMOTE)"),
        "pt3_best": ("PyTorch (3-layer)", f"best strategy ({pt_best_strategy})"),
    }
    rows = [
        full_results_table_row(model_name, strategy_name, summarize_metrics(all_results[key]))
        for key, (model_name, strategy_name) in row_labels.items()
    ]
    table_md = full_results_table_markdown(rows)
    print("\n" + table_md)

    confusion_tables = {k: aggregate_confusion_matrix(v) for k, v in all_results.items()}
    for key, (model_name, strategy_name) in row_labels.items():
        print(f"\n{model_name} ({strategy_name}) confusion matrix (summed across outer folds):")
        print(confusion_matrix_markdown(confusion_tables[key]))

    grid_means = {k: summarize_metrics(v)["pr_auc"][0] for k, v in all_results.items()}
    xgb_best_key = "xgb_cw" if grid_means["xgb_cw"] >= grid_means["xgb_smote"] else "xgb_smote"
    pt_best_key = max(("pt2_cw", "pt2_smote", "pt3_best"), key=lambda k: grid_means[k])
    overall_best_key = max(grid_means, key=lambda k: grid_means[k])
    print(f"\nXGBoost-best: {xgb_best_key}, PyTorch-best: {pt_best_key}, overall-best: {overall_best_key}")

    stats_result = compare_paired_pr_auc(
        fold_pr_auc_scores(all_results[xgb_best_key]), fold_pr_auc_scores(all_results[pt_best_key])
    )
    print(f"XGBoost-best vs PyTorch-best paired test: {stats_result}")

    country_df = summarize_country_metrics(all_results[overall_best_key])
    country_md = country_table_markdown(country_df)
    print("\nPer-country breakdown (overall best model):\n" + country_md)

    print("\n=== PR curve overlay: XGBoost-best vs PyTorch-best ===")
    key_to_strategy_name = {
        "xgb_cw": "class_weight", "xgb_smote": "smote",
        "pt2_cw": "class_weight", "pt2_smote": "smote", "pt3_best": pt_best_strategy,
    }
    key_to_X = {"xgb_cw": X_xgb, "xgb_smote": X_xgb, "pt2_cw": X_pt, "pt2_smote": X_pt, "pt3_best": X_pt}

    def _build_for_key(key: str, params: dict):
        if key.startswith("xgb"):
            return XGBoostModel(seed=seed, **params)
        p = dict(params)
        p["hidden_dims"] = tuple(p["hidden_dims"])
        return PyTorchMLP(
            categorical_columns=CATEGORICAL_COLUMNS, vocab_sizes=vocab_sizes,
            numeric_columns=pt_numeric_columns, epochs=PT_FINAL_EPOCHS, seed=seed, **p,
        )

    stratify_df = pd.DataFrame({"country": country.to_numpy(), "target": y.to_numpy()})
    pr_curve_stratify_key = make_country_target_stratify_key(stratify_df, country_col="country", target_col="target")
    idx_train, idx_val = train_test_split(train.index, test_size=0.2, stratify=pr_curve_stratify_key, random_state=seed)
    y_val_binary = (y.loc[idx_val] == "Yes").astype(int).to_numpy()

    curves = {}
    for key in (xgb_best_key, pt_best_key):
        model_name, strategy_name_label = row_labels[key]
        X = key_to_X[key]
        strategy_name = key_to_strategy_name[key]
        strategy = (
            _xgb_strategy_factory(strategy_name, seed)() if key.startswith("xgb")
            else _pt_strategy_factory(strategy_name, seed)()
        )
        params = all_results[key][0].best_params  # fold 0's tuned hyperparameters, representative
        X_res, y_res, sw = strategy.apply(X.loc[idx_train], y.loc[idx_train])
        model = _build_for_key(key, params).fit(X_res, y_res, sample_weight=sw)
        curves[f"{model_name} ({strategy_name_label})"] = (y_val_binary, model.predict_proba(X.loc[idx_val]))

    pr_curve_path = settings.paths.figures_dir / "pr_curve_overlay.png"
    average_precisions = plot_pr_curve_overlay(curves, pr_curve_path)
    print(f"Average precision (single 80/20 split, illustrative): {average_precisions}")
    print(f"Wrote {pr_curve_path}")

    print("\n=== SHAP: refitting the winning XGBoost arm on full train ===")
    xgb_strategy_name = "class_weight" if xgb_best_key == "xgb_cw" else "smote"
    production_params = tune_xgboost(
        X_xgb, y, country, imbalance_strategy_name=xgb_strategy_name,
        n_inner_folds=N_INNER_FOLDS, n_trials=XGB_N_TRIALS, seed=seed,
    )
    production_strategy = _xgb_strategy_factory(xgb_strategy_name, seed)()
    X_res, y_res, sw = production_strategy.apply(X_xgb, y)
    production_xgb = XGBoostModel(seed=seed, **production_params).fit(X_res, y_res, sample_weight=sw)

    explanation = compute_shap_explanation(production_xgb, X_xgb, max_samples=2000, seed=seed)
    importance = global_importance(explanation)
    print("Top 10 SHAP feature importances:\n" + importance.head(10).to_string())
    plot_global_importance(explanation, settings.paths.figures_dir / "shap_global_importance.png")
    for feature in ["cellphone_access_Yes", "age_of_respondent"]:
        if feature in X_xgb.columns:
            plot_dependence(explanation, feature, settings.paths.figures_dir / f"shap_dependence_{feature}.png")

    output = {
        "grid_means_pr_auc": grid_means,
        "pt_best_strategy_for_3layer": pt_best_strategy,
        "xgb_best_key": xgb_best_key,
        "pt_best_key": pt_best_key,
        "overall_best_key": overall_best_key,
        "results_table_md": table_md,
        "country_table_md": country_md,
        "confusion_matrices": {k: v.tolist() for k, v in confusion_tables.items()},
        "pr_curve_average_precision": average_precisions,
        "stats": vars(stats_result),
        "shap_top10": importance.head(10).to_dict(),
        "fold_pr_auc": {k: fold_pr_auc_scores(v) for k, v in all_results.items()},
        "thresholds": {k: [r.threshold for r in v] for k, v in all_results.items()},
        "best_params_per_fold": {k: [r.best_params for r in v] for k, v in all_results.items()},
        "production_xgb_params": production_params,
    }
    out_path = settings.paths.reports_dir / "ablation_raw_results.json"
    out_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
