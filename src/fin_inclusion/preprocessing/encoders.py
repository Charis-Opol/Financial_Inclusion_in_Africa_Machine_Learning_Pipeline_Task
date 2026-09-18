"""Model-family-specific encoding branches (Phase 2.4/2.5/2.6).

`OneHotBranch` (XGBoost path) and `EmbeddingIndexBranch` (PyTorch path) both
fit their category vocabulary on train only and route any category seen
only at transform time to an explicit unknown bucket/index, warning via
`UnseenCategoryWarning` rather than either crashing or silently mis-encoding
(Phase 2.5 -- the test set has no target to catch this indirectly, per EDA
1.8). `NumericScaler` implements Phase 2.6: numeric features are scaled for
the PyTorch branch only, since trees are scale-invariant and MLPs are not.

Every transformer here takes and returns a full DataFrame: the columns it
owns are replaced/added in place, everything else passes through
untouched, so these compose freely inside a `Pipeline`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fin_inclusion.preprocessing.base import BasePreprocessor, warn_if_unseen


class OneHotBranch(BasePreprocessor):
    """One-hot encodes `columns`, fit on train only; unseen categories -> all-zero row."""

    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "OneHotBranch":
        self.encoder_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.encoder_.fit(X[self.columns])
        self.known_categories_ = {
            col: set(cats) for col, cats in zip(self.columns, self.encoder_.categories_)
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        warn_if_unseen(X, self.columns, self.known_categories_)
        encoded = pd.DataFrame(
            self.encoder_.transform(X[self.columns]),
            columns=self.encoder_.get_feature_names_out(self.columns),
            index=X.index,
        )
        return pd.concat([X.drop(columns=self.columns), encoded], axis=1)


class EmbeddingIndexBranch(BasePreprocessor):
    """Integer-indexes `columns` for `nn.Embedding` lookup, fit on train only.

    Index 0 is reserved for the unknown bucket (both truly-unseen categories
    at transform time, and, defensively, any missing value); known
    categories are indexed 1..n. `vocab_sizes_` (n categories + 1) is what
    Phase 3's `PyTorchMLP` uses to size each embedding layer.
    """

    UNKNOWN_INDEX = 0

    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "EmbeddingIndexBranch":
        self.category_maps_ = {
            col: {cat: i + 1 for i, cat in enumerate(sorted(X[col].dropna().unique()))}
            for col in self.columns
        }
        self.known_categories_ = {col: set(m) for col, m in self.category_maps_.items()}
        return self

    @property
    def vocab_sizes_(self) -> dict[str, int]:
        return {col: len(mapping) + 1 for col, mapping in self.category_maps_.items()}

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        warn_if_unseen(X, self.columns, self.known_categories_)
        X = X.copy()
        for col in self.columns:
            X[col] = X[col].map(self.category_maps_[col]).fillna(self.UNKNOWN_INDEX).astype(int)
        return X


class NumericScaler(BasePreprocessor):
    """Standardizes `columns` using train-fitted mean/std (PyTorch branch only)."""

    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "NumericScaler":
        self.scaler_ = StandardScaler()
        self.scaler_.fit(X[self.columns])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.columns] = self.scaler_.transform(X[self.columns]).astype(np.float64)
        return X
