"""Imbalance-handling strategies, common interface (README §9).

`apply` returns `(X, y, sample_weight)` uniformly across strategies:
`ClassWeightStrategy` leaves `X`/`y` untouched and returns per-sample
weights; a future `SMOTEStrategy` (Phase 4) would instead resample `X`/`y`
and return `sample_weight=None`. Every `BaseModel.fit` accepts
`sample_weight`, so `evaluation/cv_runner.py` can swap strategies without
touching model or CV code.

Kept out of `preprocessing/` because it must be re-instantiated fresh
inside every training fold, never fit across fold boundaries -- SMOTE
fit on anything but the training-fold split would leak (Phase 4.2).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_sample_weight


class BaseImbalanceStrategy(ABC):
    @abstractmethod
    def apply(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series, np.ndarray | None]: ...


class ClassWeightStrategy(BaseImbalanceStrategy):
    """Inverse-frequency sample weights (sklearn's `"balanced"` formula).

    Phase 3's imbalance-handling arm; kept as one arm of the Phase 4
    ablation alongside SMOTE.
    """

    def apply(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
        sample_weight = compute_sample_weight(class_weight="balanced", y=y)
        return X, y, sample_weight


class NoImbalanceStrategy(BaseImbalanceStrategy):
    """Passes `X`/`y` through unchanged, no `sample_weight`.

    Used for XGBoost's class-weight ablation arm specifically: that arm's
    imbalance handling is `scale_pos_weight`, XGBoost's own native lever,
    tuned as a regular hyperparameter (Phase 4.4) -- applying a generic
    `sample_weight` on top of it would double-correct for imbalance.
    """

    def apply(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series, None]:
        return X, y, None


class SMOTEStrategy(BaseImbalanceStrategy):
    """Resamples the minority class via SMOTE, applied directly to the
    already-encoded feature matrix (one-hot or embedding-index+scaled).

    This is a known simplification, not a categorical-aware variant
    (`SMOTENC`): interpolating between one-hot rows or between embedding
    *indices* can produce fractional values that don't correspond to a
    real category. That's deliberate -- the Phase 4.3 post-SMOTE EDA
    checkpoint (`notebooks/02_smote_distribution_eda.ipynb`) exists
    specifically to check whether this produces implausible synthetic
    respondents before this strategy is trusted as an ablation arm, rather
    than silently assuming a fancier resampler would be fine.

    Must be instantiated fresh per fold (never fit across a fold boundary)
    -- `apply` is only ever called on a fold's *training* split.
    """

    def __init__(self, seed: int = 42, k_neighbors: int = 5):
        self.seed = seed
        self.k_neighbors = k_neighbors

    def apply(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series, None]:
        smote = SMOTE(random_state=self.seed, k_neighbors=self.k_neighbors)
        X_res, y_res = smote.fit_resample(X, y)
        return X_res, y_res, None
