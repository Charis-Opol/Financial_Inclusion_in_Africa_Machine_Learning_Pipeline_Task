"""Raw data loading with schema and composite-key enforcement.

Caveat this module exists to encode: ``uniqueid`` is **not** a global
primary key in this dataset. It repeats across country/year combinations
(the same ``uniqueid`` values are reused independently within each of the
four national surveys). The real composite key is
``uniqueid + country + year``. Any downstream join or merge must use the
composite key — joining on ``uniqueid`` alone will silently produce a
many-to-many match across countries.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

EXPECTED_COLUMNS: dict[str, str] = {
    "country": "object",
    "year": "int64",
    "uniqueid": "object",
    "location_type": "object",
    "cellphone_access": "object",
    "household_size": "int64",
    "age_of_respondent": "int64",
    "gender_of_respondent": "object",
    "relationship_with_head": "object",
    "marital_status": "object",
    "education_level": "object",
    "job_type": "object",
}
"""Columns/dtypes common to both train and test. Train additionally has
`bank_account` (the target), which test withholds."""

TARGET_COLUMN = "bank_account"
COMPOSITE_KEY = ["uniqueid", "country", "year"]


class SchemaError(ValueError):
    """Raised when a loaded frame doesn't match the expected schema."""


class DuplicateKeyError(ValueError):
    """Raised when the composite key (`uniqueid`+`country`+`year`) is not unique."""


@dataclass(frozen=True)
class DataLoader:
    """Loads raw train/test CSVs, enforcing schema and key uniqueness.

    Deliberately read-only: no cleaning, encoding, or transformation
    happens here (that's `preprocessing/`). This class's only job is to
    guarantee that whatever comes out of it is schema-valid and keyed
    correctly, so every downstream module can assume that without
    re-checking it.
    """

    raw_train_path: Path
    raw_test_path: Path

    def load_train(self) -> pd.DataFrame:
        df = pd.read_csv(self.raw_train_path)
        self._validate_schema(df, expect_target=True)
        self._validate_composite_key(df)
        return df

    def load_test(self) -> pd.DataFrame:
        df = pd.read_csv(self.raw_test_path)
        self._validate_schema(df, expect_target=False)
        self._validate_composite_key(df)
        return df

    @staticmethod
    def _validate_schema(df: pd.DataFrame, *, expect_target: bool) -> None:
        required = dict(EXPECTED_COLUMNS)
        if expect_target:
            required[TARGET_COLUMN] = "object"

        missing = [col for col in required if col not in df.columns]
        if missing:
            raise SchemaError(f"Missing expected columns: {missing}")

        mismatched = {
            col: (str(df[col].dtype), dtype)
            for col, dtype in required.items()
            if str(df[col].dtype) != dtype
        }
        if mismatched:
            raise SchemaError(f"Column dtype mismatch (actual, expected): {mismatched}")

    @staticmethod
    def _validate_composite_key(df: pd.DataFrame) -> None:
        n_dupes = df.duplicated(subset=COMPOSITE_KEY).sum()
        if n_dupes:
            raise DuplicateKeyError(
                f"{n_dupes} rows violate the composite key {COMPOSITE_KEY}. "
                "This should be impossible for this dataset — investigate "
                "before trusting any downstream join."
            )
