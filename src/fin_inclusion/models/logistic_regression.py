"""Logistic regression -- the sanity-check floor (README §5, Phase 3.2).

Deliberately untuned defaults: this model's job is to confirm every other
model beats a linear separator, not to be competitive itself. Per EDA 1.5,
`age_of_respondent`'s relationship with the target is non-monotonic, which
a linear-in-the-logit model on raw age can't capture -- this is expected
to show up as this model's weakest point relative to the tree/MLP models.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from fin_inclusion.models.base_model import BaseModel

POSITIVE_LABEL = "Yes"


class LogisticRegressionModel(BaseModel):
    def __init__(self, seed: int = 42, max_iter: int = 1000):
        self.seed = seed
        self.max_iter = max_iter
        self._model = LogisticRegression(max_iter=self.max_iter, random_state=self.seed)

    def fit(
        self, X: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray | None = None
    ) -> "LogisticRegressionModel":
        y_binary = (y == POSITIVE_LABEL).astype(int)
        self._model.fit(X, y_binary, sample_weight=sample_weight)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(X)[:, 1]

    def get_params(self) -> dict:
        return self._model.get_params()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._model, f)

    @classmethod
    def load(cls, path: Path) -> "LogisticRegressionModel":
        instance = cls()
        with open(path, "rb") as f:
            instance._model = pickle.load(f)
        return instance
