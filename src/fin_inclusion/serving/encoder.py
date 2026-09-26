"""Numpy-only re-implementation of the fitted XGBoost preprocessing branch.

Why not just unpickle the fitted sklearn `Pipeline`? Three reasons:
  1. It would drag pandas + sklearn (and pickle's version-coupling) into the
     serving image and the request path, for what is ultimately a ~40-slot
     lookup table.
  2. The pandas pipeline costs milliseconds per single-row call; this costs
     microseconds.
  3. The same flat float32 vector is exactly what the ONNX graph consumes, so
     native and ONNX inference share one encoder.

Correctness is not taken on trust: `scripts/train_production_model.py`
refuses to write the model artifact unless this encoder reproduces the real
pipeline's output bit-for-bit on the full training set.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

HOUSEHOLD_SIZE_OUTLIER_FEATURE = "household_size_outlier"


class FeatureContractError(ValueError):
    """The metadata and the model's feature list disagree -- artifact is unusable."""


class UnknownCategoryError(ValueError):
    """A categorical value outside the training vocabulary reached the encoder.

    The API schema rejects these with a 422 before encoding, so seeing this at
    serving time means the schema and the artifact have drifted apart -- a
    server-side bug, not bad client input.
    """


@dataclass(frozen=True)
class FeatureEncoder:
    """Maps one raw survey record to the model's ordered float32 feature vector.

    Built once from `metadata.json`; every lookup table is precomputed in
    `from_metadata` so `encode` is just dict lookups and array stores.
    """

    feature_columns: tuple[str, ...]
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    disguised_missing_values: Mapping[str, str]
    unknown_label: str
    household_size_outlier_threshold: int
    numeric_index: Mapping[str, int]
    outlier_index: int
    # (column, raw value as sent by the client) -> position of its one-hot slot
    onehot_index: Mapping[tuple[str, str], int]

    @classmethod
    def from_metadata(cls, metadata: Mapping) -> "FeatureEncoder":
        feature_columns = tuple(metadata["feature_columns"])
        position = {name: i for i, name in enumerate(feature_columns)}
        disguised = dict(metadata["disguised_missing_values"])
        unknown_label = metadata["unknown_label"]

        missing = [c for c in [*metadata["numeric_columns"], HOUSEHOLD_SIZE_OUTLIER_FEATURE] if c not in position]
        if missing:
            raise FeatureContractError(f"Model feature list lacks expected columns: {missing}")

        onehot_index: dict[tuple[str, str], int] = {}
        for column, raw_levels in metadata["raw_categorical_levels"].items():
            for raw_value in raw_levels:
                # Mirrors MissingnessRecoder: the column's placeholder answer is
                # renamed to "unknown" *before* one-hot encoding, so it lands in
                # the `<column>_unknown` slot, not a `<column>_Dont know` one.
                encoded_level = unknown_label if disguised.get(column) == raw_value else raw_value
                feature_name = f"{column}_{encoded_level}"
                if feature_name not in position:
                    raise FeatureContractError(
                        f"No model feature {feature_name!r} for {column}={raw_value!r}"
                    )
                onehot_index[(column, raw_value)] = position[feature_name]

        n_mapped = len(metadata["numeric_columns"]) + 1 + len(set(onehot_index.values()))
        if n_mapped != len(feature_columns):
            raise FeatureContractError(
                f"Encoder covers {n_mapped} of the model's {len(feature_columns)} features; "
                "metadata and model are out of sync"
            )

        return cls(
            feature_columns=feature_columns,
            numeric_columns=tuple(metadata["numeric_columns"]),
            categorical_columns=tuple(metadata["raw_categorical_levels"]),
            disguised_missing_values=disguised,
            unknown_label=unknown_label,
            household_size_outlier_threshold=int(metadata["household_size_outlier_threshold"]),
            numeric_index={c: position[c] for c in metadata["numeric_columns"]},
            outlier_index=position[HOUSEHOLD_SIZE_OUTLIER_FEATURE],
            onehot_index=onehot_index,
        )

    @property
    def n_features(self) -> int:
        return len(self.feature_columns)

    def encode_into(self, record: Mapping[str, object], out: np.ndarray) -> None:
        """Writes `record`'s features into the zeroed row `out` (no allocation)."""
        for column, index in self.numeric_index.items():
            out[index] = record[column]
        out[self.outlier_index] = record["household_size"] > self.household_size_outlier_threshold
        for column in self.categorical_columns:
            key = (column, record[column])
            try:
                out[self.onehot_index[key]] = 1.0
            except KeyError:
                raise UnknownCategoryError(f"{column}={record[column]!r} is not in the training vocabulary") from None

    def encode_batch(self, records: Sequence[Mapping[str, object]]) -> np.ndarray:
        """Encodes records into an `(n, n_features)` float32 matrix, in model column order."""
        matrix = np.zeros((len(records), self.n_features), dtype=np.float32)
        for row, record in zip(matrix, records):
            self.encode_into(record, row)
        # Pydantic already rejects NaN/Inf at the boundary; this guards any
        # other caller (batch jobs, benchmarks) from silently scoring garbage.
        if not np.isfinite(matrix).all():
            raise ValueError("Encoded feature matrix contains NaN/Inf")
        return matrix
