"""Read-only data-quality checks: key uniqueness and category consistency.

Kept separate from `loader.py` (which enforces hard invariants that should
never be violated) because these functions produce *reports* for EDA and
for Phase 2's unseen-category handling, rather than raising on failure.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fin_inclusion.data.loader import COMPOSITE_KEY


@dataclass(frozen=True)
class KeyUniquenessReport:
    """Compares `uniqueid` alone vs. the composite key as a candidate key."""

    n_rows: int
    n_duplicates_on_uniqueid_alone: int
    n_duplicates_on_composite_key: int

    @property
    def uniqueid_alone_is_unique(self) -> bool:
        return self.n_duplicates_on_uniqueid_alone == 0

    @property
    def composite_key_is_unique(self) -> bool:
        return self.n_duplicates_on_composite_key == 0


def check_key_uniqueness(df: pd.DataFrame) -> KeyUniquenessReport:
    """Confirms `uniqueid` is not a global key but `uniqueid+country+year` is."""
    return KeyUniquenessReport(
        n_rows=len(df),
        n_duplicates_on_uniqueid_alone=int(df.duplicated(subset=["uniqueid"]).sum()),
        n_duplicates_on_composite_key=int(df.duplicated(subset=COMPOSITE_KEY).sum()),
    )


@dataclass(frozen=True)
class CategoryConsistencyReport:
    """Per-column categories present in test but absent from train.

    These are the categories that would be "unseen" to an encoder fit on
    train only — exactly the case Phase 2's encoders must handle
    explicitly (fail loudly / route to an "unknown" bucket) rather than
    silently mis-encode.
    """

    unseen_in_test: dict[str, list[str]]

    @property
    def has_unseen_categories(self) -> bool:
        return any(len(v) > 0 for v in self.unseen_in_test.values())


def check_train_test_category_consistency(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    categorical_columns: list[str],
) -> CategoryConsistencyReport:
    """For each categorical column, finds test categories absent from train."""
    unseen: dict[str, list[str]] = {}
    for col in categorical_columns:
        train_categories = set(train_df[col].unique())
        test_categories = set(test_df[col].unique())
        diff = sorted(test_categories - train_categories)
        unseen[col] = diff
    return CategoryConsistencyReport(unseen_in_test=unseen)


def cardinality_audit(df: pd.DataFrame, categorical_columns: list[str]) -> pd.Series:
    """Unique level counts per categorical column, sorted descending.

    Directly informs the Phase 2 encoding decision (one-hot vs. embeddings):
    high-cardinality columns are the ones where embeddings pay off most.
    """
    return pd.Series(
        {col: df[col].nunique() for col in categorical_columns}
    ).sort_values(ascending=False)
