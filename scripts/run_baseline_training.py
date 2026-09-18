"""Phase 3: untuned baselines through a single stratified 5-fold CV.

Establishes the honest floor Phase 4's tuning gains are measured against.
Default hyperparameters only (no tuning); class-weighting only (SMOTE is
Phase 4); F1 at the default 0.5 threshold (explicitly *not* tuned yet).

Reuses the Phase 2 preprocessing pipelines, fit once on the full train set
-- Phase 3's fold loop is about model comparison under a fixed feature
representation, not about re-validating the preprocessing fit itself (that
non-leakage requirement is scoped to Phase 4's SMOTE arm and inner tuning,
per the Implementation Plan).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "config"))
from settings import load_settings  # noqa: E402

from fin_inclusion.data.loader import DataLoader, TARGET_COLUMN  # noqa: E402
from fin_inclusion.evaluation.cv_runner import run_stratified_cv  # noqa: E402
from fin_inclusion.evaluation.reporting import (  # noqa: E402
    country_table_markdown,
    results_table_markdown,
    results_table_row,
    summarize_country_metrics,
    summarize_metrics,
)
from fin_inclusion.imbalance.strategies import ClassWeightStrategy  # noqa: E402
from fin_inclusion.models.logistic_regression import LogisticRegressionModel  # noqa: E402
from fin_inclusion.models.pytorch_mlp import PyTorchMLP  # noqa: E402
from fin_inclusion.models.xgboost_model import XGBoostModel  # noqa: E402
from fin_inclusion.preprocessing.pipeline_factory import (  # noqa: E402
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    build_pytorch_pipeline,
    build_xgboost_pipeline,
)

ID_COLUMNS = ["uniqueid", "year"]
NON_FEATURE_COLUMNS = ID_COLUMNS + [TARGET_COLUMN]
IMBALANCE_STRATEGY_LABEL = "class-weight"


def main() -> None:
    settings = load_settings()
    settings.paths.reports_dir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(
        raw_train_path=settings.paths.raw_train, raw_test_path=settings.paths.raw_test
    )
    train = loader.load_train()
    print(f"Loaded train: {train.shape}")

    y = train[TARGET_COLUMN]
    country = train["country"]

    xgb_encoded = build_xgboost_pipeline().fit_transform(train)
    X_tabular = xgb_encoded.drop(columns=NON_FEATURE_COLUMNS)

    pytorch_pipeline = build_pytorch_pipeline().fit(train)
    pt_encoded = pytorch_pipeline.transform(train)
    X_pytorch = pt_encoded.drop(columns=NON_FEATURE_COLUMNS)
    vocab_sizes = pytorch_pipeline.named_steps["encode"].vocab_sizes_
    pytorch_numeric_columns = NUMERIC_COLUMNS + ["household_size_outlier"]

    strategy = ClassWeightStrategy()
    n_folds = settings.cv.n_outer_folds

    model_specs = [
        ("LogisticRegression", X_tabular, lambda: LogisticRegressionModel(seed=settings.seed)),
        ("XGBoost", X_tabular, lambda: XGBoostModel(seed=settings.seed)),
        (
            "PyTorchMLP (2-layer)",
            X_pytorch,
            lambda: PyTorchMLP(
                categorical_columns=CATEGORICAL_COLUMNS,
                vocab_sizes=vocab_sizes,
                numeric_columns=pytorch_numeric_columns,
                seed=settings.seed,
            ),
        ),
    ]

    rows = []
    country_tables = {}
    for name, X, model_factory in model_specs:
        print(f"\nRunning {n_folds}-fold CV: {name} ({IMBALANCE_STRATEGY_LABEL})")
        results = run_stratified_cv(
            X=X,
            y=y,
            country=country,
            model_factory=model_factory,
            imbalance_strategy=strategy,
            n_folds=n_folds,
            threshold=0.5,
            seed=settings.seed,
        )
        summary = summarize_metrics(results)
        print({k: f"{v[0]:.3f} +/- {v[1]:.3f}" for k, v in summary.items()})
        rows.append(results_table_row(name, IMBALANCE_STRATEGY_LABEL, summary))
        country_tables[name] = summarize_country_metrics(results)

    table_md = results_table_markdown(rows)
    print("\n" + table_md)

    report_lines = [
        "# Phase 3 — Untuned Baseline Results",
        "",
        "Single stratified 5-fold CV (country + target stratified), class-weight "
        "imbalance handling only, F1 at the default 0.5 threshold (not yet tuned -- "
        "see `reports/ablation_study_report.md` for Phase 4's tuned results).",
        "",
        "## Results table",
        "",
        table_md,
        "",
        "## Per-country breakdown",
        "",
    ]
    for name, _, _ in model_specs:
        report_lines.append(f"### {name}")
        report_lines.append("")
        report_lines.append(country_table_markdown(country_tables[name]))
        report_lines.append("")

    report_path = settings.paths.reports_dir / "baseline_results.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nWrote {report_path}")


if __name__ == "__main__":
    main()
