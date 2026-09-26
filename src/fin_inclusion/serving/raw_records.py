"""Reads raw survey CSVs into the record dicts the serving encoder consumes.

Stdlib `csv` rather than pandas so offline tooling (ONNX verification,
benchmarks, drift checks) runs in the lightweight serving environment.
"""
from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path


def read_raw_records(
    csv_path: Path, categorical_columns: Iterable[str], numeric_columns: Iterable[str]
) -> list[dict[str, object]]:
    """Returns one dict per row with only the model's input columns, numerics as int.

    Raises (rather than skipping rows) on a missing column or a non-integer
    numeric value, so a malformed file can't quietly shrink the evaluation set.
    """
    categorical_columns, numeric_columns = list(categorical_columns), list(numeric_columns)
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in categorical_columns + numeric_columns if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{csv_path} lacks required columns {missing}")
        records = []
        for line_no, row in enumerate(reader, start=2):
            try:
                record: dict[str, object] = {c: row[c] for c in categorical_columns}
                record |= {c: int(row[c]) for c in numeric_columns}
            except ValueError as exc:
                raise ValueError(f"{csv_path}:{line_no}: {exc}") from exc
            records.append(record)
    return records
