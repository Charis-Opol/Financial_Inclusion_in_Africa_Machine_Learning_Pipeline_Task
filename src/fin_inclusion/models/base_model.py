"""Shared model interface (README §7: Open/Closed + Liskov Substitution).

Every model family (`LogisticRegressionModel`, `XGBoostModel`, `PyTorchMLP`)
implements this ABC, so `evaluation/cv_runner.py` can drive any of them
through the same fit/score loop without knowing which one it's running, and
a new model type can be added without touching the CV runner.

`predict_proba` deliberately returns a 1D array of P(target == positive
class), not sklearn's `(n, 2)` convention -- this repo's target is always
binary and every consumer (`evaluation/metrics.py`) wants a single score
per row.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd


class BaseModel(ABC):
    @abstractmethod
    def fit(
        self, X: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray | None = None
    ) -> "BaseModel":
        """Fits on `X`/`y`. `sample_weight`, when given, implements an
        imbalance strategy (`imbalance/strategies.py`) -- every model family
        must accept it, even if internally it just forwards to the
        underlying library's own `sample_weight` support."""

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns a 1D array of P(target == positive class), one per row of `X`."""

    @abstractmethod
    def get_params(self) -> dict:
        """Hyperparameters this model was constructed with (for logging/reporting)."""

    @abstractmethod
    def save(self, path: Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseModel": ...
