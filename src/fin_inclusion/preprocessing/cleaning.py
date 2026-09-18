"""Missingness-in-disguise recoding and outlier flagging.

Both transformers are stateless (the values/thresholds they act on are
fixed domain constants from the Phase 1 EDA, not learned from data), but
still implement the `fit`/`transform` interface so they compose inside a
`sklearn.pipeline.Pipeline` alongside the (stateful) encoders.
"""
from __future__ import annotations

import pandas as pd

from fin_inclusion.preprocessing.base import BasePreprocessor

DISGUISED_MISSING_VALUES: dict[str, str] = {
    "marital_status": "Dont know",
    "education_level": "Other/Dont know/RTA",
    "job_type": "Dont Know/Refuse to answer",
}
"""Placeholder categories found in EDA 1.3. Two of three showed a
bank_account='Yes' rate well above baseline (25.0%/31.4% vs 14.1%
overall) -- kept as an explicit category, not imputed or dropped."""

UNKNOWN_LABEL = "unknown"

HOUSEHOLD_SIZE_OUTLIER_THRESHOLD = 15
"""EDA 1.6: 8 rows exceed this, spread across countries with plausible
respondent ages -- flagged, not capped or dropped."""


class MissingnessRecoder(BasePreprocessor):
    """Recodes disguised-missingness placeholders to a canonical `"unknown"` category.

    Per Phase 2.2: the placeholder strings (e.g. `"Dont know"`) are kept as
    their own category level -- this only normalizes the label so every
    affected column encodes "respondent declined/didn't know" the same way,
    instead of leaving column-specific placeholder text for the encoders to
    treat as arbitrary, unrelated categories.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "MissingnessRecoder":
        return self

    def __sklearn_is_fitted__(self) -> bool:
        # Recodes a fixed set of known placeholder strings -- no state to learn from X.
        return True

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, placeholder in DISGUISED_MISSING_VALUES.items():
            if col in X.columns:
                X[col] = X[col].replace(placeholder, UNKNOWN_LABEL)
        return X


class HouseholdSizeOutlierFlagger(BasePreprocessor):
    """Adds a `household_size_outlier` boolean flag; never drops or caps rows.

    Per Phase 2.3: the 8 rows found in EDA 1.6 are plausible extended
    households, not data-entry errors, so they stay in the dataset with
    this flag available as a feature rather than being silently removed.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "HouseholdSizeOutlierFlagger":
        return self

    def __sklearn_is_fitted__(self) -> bool:
        # Flags against a fixed threshold -- no state to learn from X.
        return True

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["household_size_outlier"] = X["household_size"] > HOUSEHOLD_SIZE_OUTLIER_THRESHOLD
        return X
