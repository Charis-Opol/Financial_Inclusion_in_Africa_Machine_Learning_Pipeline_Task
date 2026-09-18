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
    """`**xgb_params` forwards directly to `XGBClassifier` -- Phase 4's Optuna
    search passes `max_depth`, `learning_rate`, `n_estimators`,
    `min_child_weight`, `subsample`, `colsample_bytree`, and, for the
    class-weight ablation arm, `scale_pos_weight` (XGBoost's native
    imbalance lever, used instead of generic `sample_weight` for that arm
    specifically -- see `imbalance/strategies.py`'s `NoImbalanceStrategy`)."""

    def __init__(self, seed: int = 42, **xgb_params):
        self.seed = seed
        self.xgb_params = xgb_params
        self._model = XGBClassifier(random_state=self.seed, eval_metric="logloss", **xgb_params)

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

    @property
    def booster(self) -> XGBClassifier:
        """The underlying fitted `XGBClassifier`, for `interpretability/shap_analysis.py`."""
        return self._model

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(path))

    @classmethod
    def load(cls, path: Path) -> "XGBoostModel":
        instance = cls()
        instance._model.load_model(str(path))
        return instance
