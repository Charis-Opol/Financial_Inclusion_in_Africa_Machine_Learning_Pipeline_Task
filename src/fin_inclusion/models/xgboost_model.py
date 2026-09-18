"""XGBoost -- the primary baseline (README §5, Phase 3.3).

Default hyperparameters only; tuning is explicitly deferred to Phase 4 so
this baseline stays a clean, untuned reference point.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from fin_inclusion.models.base_model import BaseModel

POSITIVE_LABEL = "Yes"


class XGBoostModel(BaseModel):
    def __init__(self, seed: int = 42):
        self.seed = seed
        self._model = XGBClassifier(random_state=self.seed, eval_metric="logloss")

    def fit(
        self, X: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray | None = None
    ) -> "XGBoostModel":
        y_binary = (y == POSITIVE_LABEL).astype(int)
        self._model.fit(X, y_binary, sample_weight=sample_weight)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(X)[:, 1]

    def get_params(self) -> dict:
        return self._model.get_params()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(path))

    @classmethod
    def load(cls, path: Path) -> "XGBoostModel":
        instance = cls()
        instance._model.load_model(str(path))
        return instance
