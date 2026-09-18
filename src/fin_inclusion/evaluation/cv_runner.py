"""Stratified CV orchestration (README §6, Phase 3.5).

Non-nested at this stage -- Phase 3 evaluates untuned baselines, so there's
no inner hyperparameter search to nest. Phase 4 wraps this same fold loop
with an inner Optuna search per outer fold; the loop itself doesn't change.

Folds are stratified on `country`+`target` jointly (`make_country_target_
stratify_key`, Phase 2.8), per EDA 1.1/1.2's finding that country is the
dominant structural effect -- stratifying on target alone would let a
fold's country mix drift and confound the per-country breakdown below.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from fin_inclusion.evaluation.metrics import compute_classification_metrics
from fin_inclusion.imbalance.strategies import BaseImbalanceStrategy
from fin_inclusion.models.base_model import BaseModel
from fin_inclusion.preprocessing.pipeline_factory import make_country_target_stratify_key

POSITIVE_LABEL = "Yes"


@dataclass(frozen=True)
class CVFoldResult:
    fold: int
    metrics: dict[str, float]
    country_metrics: dict[str, dict[str, float]]


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
    stratify_df = pd.DataFrame({"country": country.to_numpy(), "target": y.to_numpy()})
    stratify_key = make_country_target_stratify_key(stratify_df, country_col="country", target_col="target")
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
