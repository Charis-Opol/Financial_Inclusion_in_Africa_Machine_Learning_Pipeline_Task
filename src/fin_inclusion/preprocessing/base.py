"""Shared preprocessing interface and unseen-category handling infrastructure.

`BasePreprocessor` is the Interface Segregation anchor for this package
(README §7): every transformer implements only `fit`/`transform`, nothing
more, and is `sklearn`-compatible (usable inside a `Pipeline`) via
`BaseEstimator`/`TransformerMixin`.

`UnseenCategoryWarning` and `unseen_categories` implement Phase 2.5's
"fail loudly, not silently" requirement: encoders fit on train only route
unseen categories to an explicit unknown bucket/index at transform time,
but must also surface that it happened, since the test set has no target
to catch a silent mis-encoding indirectly (Phase 1.8).
"""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class BasePreprocessor(ABC, BaseEstimator, TransformerMixin):
    """`fit`/`transform` interface every preprocessing transformer implements."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "BasePreprocessor": ...

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame: ...


class UnseenCategoryWarning(UserWarning):
    """Raised when `transform` sees a category absent at `fit` time."""


def unseen_categories(
    X: pd.DataFrame, columns: list[str], known: dict[str, set]
) -> dict[str, list[str]]:
    """Per-column categories in `X` not present in `known` (the fit-time vocabulary)."""
    unseen: dict[str, list[str]] = {}
    for col in columns:
        diff = sorted(set(X[col].dropna().unique()) - known[col])
        if diff:
            unseen[col] = diff
    return unseen


def warn_if_unseen(X: pd.DataFrame, columns: list[str], known: dict[str, set]) -> None:
    diff = unseen_categories(X, columns, known)
    if diff:
        warnings.warn(
            f"Unseen categories at transform time, routed to the unknown bucket: {diff}",
            UnseenCategoryWarning,
            stacklevel=3,
        )
