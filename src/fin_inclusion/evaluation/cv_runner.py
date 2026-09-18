"""Stratified CV orchestration (README §6, Phase 3.5 / Phase 4.1).

`run_stratified_cv` is the non-nested loop Phase 3 uses for untuned
baselines. `run_nested_cv` wraps the same fold-splitting logic with an
inner tuning step per outer fold (Phase 4): for each outer fold, `tune_fn`
is called on the outer *training* split only (never the outer validation
split) to pick hyperparameters, a further held-out slice of that same
training split picks the F1-maximizing threshold (Phase 4.5), and only
then is the final model refit on the full outer-training split and scored
once against the untouched outer-validation split. The outer validation
fold is never touched by tuning or threshold selection at any point.

Folds are stratified on `country`+`target` jointly (`make_country_target_
stratify_key`, Phase 2.8), per EDA 1.1/1.2's finding that country is the
dominant structural effect -- stratifying on target alone would let a
fold's country mix drift and confound the per-country breakdown below.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from fin_inclusion.evaluation.metrics import compute_classification_metrics
from fin_inclusion.evaluation.threshold_selection import select_f1_maximizing_threshold
from fin_inclusion.imbalance.strategies import BaseImbalanceStrategy
from fin_inclusion.models.base_model import BaseModel
from fin_inclusion.preprocessing.pipeline_factory import make_country_target_stratify_key

POSITIVE_LABEL = "Yes"


@dataclass(frozen=True)
class CVFoldResult:
    fold: int
    metrics: dict[str, float]
    country_metrics: dict[str, dict[str, float]]


@dataclass(frozen=True)
class NestedFoldResult:
    fold: int
    best_params: dict
    threshold: float
    metrics: dict[str, float]
    country_metrics: dict[str, dict[str, float]]


def _stratify_key(y: pd.Series, country: pd.Series) -> pd.Series:
    stratify_df = pd.DataFrame({"country": country.to_numpy(), "target": y.to_numpy()})
    return make_country_target_stratify_key(stratify_df, country_col="country", target_col="target")


def run_stratified_cv(
    X: pd.DataFrame,
    y: pd.Series,
    country: pd.Series,
    model_factory: Callable[[], BaseModel],
    imbalance_strategy: BaseImbalanceStrategy,
    n_folds: int = 5,
    threshold: float = 0.5,
    seed: int = 42,
) -> list[CVFoldResult]:
    """Fits a fresh `model_factory()` model per fold; never reuses a fitted model across folds."""
    stratify_key = _stratify_key(y, country)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    results = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, stratify_key)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        country_val = country.iloc[val_idx]

        X_res, y_res, sample_weight = imbalance_strategy.apply(X_train, y_train)

        model = model_factory()
        model.fit(X_res, y_res, sample_weight=sample_weight)
        y_score = model.predict_proba(X_val)
        y_val_binary = (y_val == POSITIVE_LABEL).astype(int).to_numpy()

        fold_metrics = compute_classification_metrics(y_val_binary, y_score, threshold)
        country_metrics = {
            c: compute_classification_metrics(
                y_val_binary[country_val.to_numpy() == c],
                y_score[country_val.to_numpy() == c],
                threshold,
            )
            for c in sorted(country_val.unique())
        }
        results.append(CVFoldResult(fold=fold, metrics=fold_metrics, country_metrics=country_metrics))

    return results


def run_nested_cv(
    X: pd.DataFrame,
    y: pd.Series,
    country: pd.Series,
    tune_fn: Callable[[pd.DataFrame, pd.Series, pd.Series], dict],
    build_model: Callable[[dict], BaseModel],
    imbalance_strategy_factory: Callable[[], BaseImbalanceStrategy],
    n_outer_folds: int = 5,
    threshold_holdout_frac: float = 0.2,
    seed: int = 42,
) -> list[NestedFoldResult]:
    """Nested CV (Phase 4): tunes, selects a threshold, and refits within
    each outer-training fold only; scores once against the untouched outer
    validation fold.

    `tune_fn(X_train, y_train, country_train)` returns the best
    hyperparameters (its own inner CV/Optuna search happens inside it, on
    the outer-training fold it's given). `build_model(params)` constructs a
    fresh, unfit model from those hyperparameters.
    """
    stratify_key = _stratify_key(y, country)
    skf = StratifiedKFold(n_splits=n_outer_folds, shuffle=True, random_state=seed)

    results = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, stratify_key)):
        X_train_outer, X_val_outer = X.iloc[train_idx], X.iloc[val_idx]
        y_train_outer, y_val_outer = y.iloc[train_idx], y.iloc[val_idx]
        country_train_outer, country_val_outer = country.iloc[train_idx], country.iloc[val_idx]

        best_params = tune_fn(X_train_outer, y_train_outer, country_train_outer)

        X_thr_train, X_thr_val, y_thr_train, y_thr_val = train_test_split(
            X_train_outer,
            y_train_outer,
            test_size=threshold_holdout_frac,
            stratify=y_train_outer,
            random_state=seed,
        )
        threshold_strategy = imbalance_strategy_factory()
        X_thr_res, y_thr_res, thr_sample_weight = threshold_strategy.apply(X_thr_train, y_thr_train)
        threshold_model = build_model(best_params)
        threshold_model.fit(X_thr_res, y_thr_res, sample_weight=thr_sample_weight)
        y_thr_score = threshold_model.predict_proba(X_thr_val)
        y_thr_binary = (y_thr_val == POSITIVE_LABEL).astype(int).to_numpy()
        threshold, _ = select_f1_maximizing_threshold(y_thr_binary, y_thr_score)

        final_strategy = imbalance_strategy_factory()
        X_res, y_res, sample_weight = final_strategy.apply(X_train_outer, y_train_outer)
        final_model = build_model(best_params)
        final_model.fit(X_res, y_res, sample_weight=sample_weight)

        y_val_score = final_model.predict_proba(X_val_outer)
        y_val_binary = (y_val_outer == POSITIVE_LABEL).astype(int).to_numpy()
        fold_metrics = compute_classification_metrics(y_val_binary, y_val_score, threshold)
        country_val_arr = country_val_outer.to_numpy()
        country_metrics = {
            c: compute_classification_metrics(
                y_val_binary[country_val_arr == c], y_val_score[country_val_arr == c], threshold
            )
            for c in sorted(country_val_outer.unique())
        }

        results.append(
            NestedFoldResult(
                fold=fold,
                best_params=best_params,
                threshold=threshold,
                metrics=fold_metrics,
                country_metrics=country_metrics,
            )
        )

    return results
