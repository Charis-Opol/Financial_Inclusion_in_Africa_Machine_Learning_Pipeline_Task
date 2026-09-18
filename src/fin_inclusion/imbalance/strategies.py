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
