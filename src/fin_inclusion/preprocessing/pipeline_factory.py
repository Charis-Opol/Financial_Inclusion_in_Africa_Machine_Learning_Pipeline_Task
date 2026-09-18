"""Assembles the cleaning + encoding steps into one `Pipeline` per model family.

Both branches share the same cleaning steps (missingness recoding, outlier
flagging) and diverge only at encoding, per the Phase 2.4/2.6 decisions:
XGBoost gets one-hot categoricals and raw numerics (trees are
scale-invariant); PyTorch gets embedding-index categoricals and scaled
numerics.
"""
from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline

from fin_inclusion.preprocessing.cleaning import HouseholdSizeOutlierFlagger, MissingnessRecoder
from fin_inclusion.preprocessing.encoders import EmbeddingIndexBranch, NumericScaler, OneHotBranch

CATEGORICAL_COLUMNS: list[str] = [
    "country",
    "location_type",
    "cellphone_access",
    "gender_of_respondent",
    "relationship_with_head",
    "marital_status",
    "education_level",
    "job_type",
]
"""Per EDA 1.7's cardinality audit; `country` is included deliberately per
Phase 2.8 -- kept in the pooled feature set, not treated as an ID column."""

NUMERIC_COLUMNS: list[str] = ["household_size", "age_of_respondent"]


def build_xgboost_pipeline(
    categorical_columns: list[str] = CATEGORICAL_COLUMNS,
) -> Pipeline:
    """XGBoost branch: missingness recode + outlier flag + one-hot categoricals."""
    return Pipeline(
        steps=[
            ("missingness", MissingnessRecoder()),
            ("outlier_flag", HouseholdSizeOutlierFlagger()),
            ("encode", OneHotBranch(columns=categorical_columns)),
        ]
    )


def build_pytorch_pipeline(
    categorical_columns: list[str] = CATEGORICAL_COLUMNS,
    numeric_columns: list[str] = NUMERIC_COLUMNS,
) -> Pipeline:
    """PyTorch branch: missingness recode + outlier flag + scaled numerics + embedding indices."""
    return Pipeline(
        steps=[
            ("missingness", MissingnessRecoder()),
            ("outlier_flag", HouseholdSizeOutlierFlagger()),
            ("scale", NumericScaler(columns=numeric_columns)),
            ("encode", EmbeddingIndexBranch(columns=categorical_columns)),
        ]
    )


def make_country_target_stratify_key(
    df: pd.DataFrame, country_col: str = "country", target_col: str | None = None
) -> pd.Series:
    """Joint `country`+`target` label for stratified CV (Phase 2.8).

    Per EDA 1.1/1.2, country is the dominant structural effect (account
    ownership varies ~3x by country) -- stratifying by target alone would
    let a fold's country mix drift and confound per-country reporting.
    `target_col=None` (the test set, which withholds the target) stratifies
    on `country` alone.
    """
    if target_col is None:
        return df[country_col].astype(str)
    return df[country_col].astype(str) + "_" + df[target_col].astype(str)
