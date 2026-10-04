"""Aggregating encoded-column attributions back to the 10 raw survey features.

The two models see the same respondent in different encodings: XGBoost gets
40 columns (one-hot categoricals, raw numerics, an outlier flag), the MLP
gets a 6-dim embedding per categorical plus 3 numeric inputs. The raw
features are the only unit the two can be compared in, and the unit the
serving contract is written in.

Summing is valid in this sense: Shapley values, Integrated Gradients and
Feature Ablation are all additive decompositions of the model output, so the
summed columns are still an additive decomposition and efficiency /
completeness is preserved exactly.

What summing does NOT guarantee: that grouped per-column Shapley values
equal the Shapley values of the game played directly on the 10 raw features.
Treating 40 one-hot columns as 40 players averages over coalitions in which
one column of a categorical comes from the instance and another from the
background (two levels "hot" at once, or none) -- inputs the 10-player game
never evaluates. The two agree when the model has no interactions between
columns of the same categorical and can differ otherwise. That gap is
measured, not assumed away: the exact-Shapley oracle is defined on the raw
features and grouped TreeSHAP is compared against it.

`household_size_outlier` is not an 11th feature: it is a deterministic
function of `household_size` (> 15), so its attribution belongs to
`household_size`.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from fin_inclusion.interpretability.attribution import Attribution

DERIVED_FEATURE_PARENTS: dict[str, str] = {"household_size_outlier": "household_size"}
"""Model inputs computed from another raw feature (`preprocessing/cleaning.py`'s
`HouseholdSizeOutlierFlagger`), mapped to the raw feature they derive from."""


@dataclass(frozen=True, eq=False)
class FeatureGrouping:
    """Which raw feature each encoded column belongs to.

    `group_index[k]` is the position in `raw_features` of the raw feature
    that encoded column `encoded_names[k]` carries.
    """

    raw_features: tuple[str, ...]
    encoded_names: tuple[str, ...]
    group_index: np.ndarray

    def __post_init__(self) -> None:
        group_index = np.asarray(self.group_index, dtype=np.int64)
        if group_index.shape != (len(self.encoded_names),):
            raise ValueError(
                f"group_index has shape {group_index.shape} for {len(self.encoded_names)} encoded columns"
            )
        if len(set(self.raw_features)) != len(self.raw_features):
            raise ValueError(f"raw features are not unique: {self.raw_features}")
        if group_index.size and (group_index.min() < 0 or group_index.max() >= len(self.raw_features)):
            raise ValueError("group_index refers to a raw feature that does not exist")
        # A raw feature with no encoded column would silently get a zero
        # attribution and read as "the model ignores it".
        empty = sorted(set(self.raw_features) - {self.raw_features[i] for i in group_index})
        if empty:
            raise ValueError(f"No encoded column maps to raw feature(s) {empty}")
        object.__setattr__(self, "group_index", group_index)

    @classmethod
    def for_onehot(
        cls,
        encoded_columns: Sequence[str],
        categorical_columns: Sequence[str],
        numeric_columns: Sequence[str],
        derived_parents: Mapping[str, str] = DERIVED_FEATURE_PARENTS,
    ) -> "FeatureGrouping":
        """Grouping for the XGBoost branch's columns (`OneHotBranch` names
        them `<column>_<level>`). Raw features are ordered categoricals
        first, then numerics."""
        raw_features = (*categorical_columns, *numeric_columns)
        position = {name: i for i, name in enumerate(raw_features)}
        # Longest prefix first, so a categorical whose name is a prefix of
        # another's can never claim the other's columns.
        by_length = sorted(categorical_columns, key=len, reverse=True)

        group_index = []
        for column in encoded_columns:
            if column in position and column in numeric_columns:
                parent = column
            elif column in derived_parents:
                parent = derived_parents[column]
            else:
                parent = next((c for c in by_length if column.startswith(f"{c}_")), None)
            if parent not in position:
                raise ValueError(f"Encoded column {column!r} does not belong to any raw feature")
            group_index.append(position[parent])
        return cls(raw_features=raw_features, encoded_names=tuple(encoded_columns), group_index=np.array(group_index))

    @classmethod
    def for_embeddings(
        cls,
        categorical_columns: Sequence[str],
        embedding_dim: int,
        numeric_inputs: Sequence[str],
        derived_parents: Mapping[str, str] = DERIVED_FEATURE_PARENTS,
    ) -> "FeatureGrouping":
        """Grouping for the MLP's first-layer input: `embedding_dim` slots
        per categorical (in `categorical_columns` order), then
        `numeric_inputs` -- the layout `_MLPNet.forward` concatenates.
        `numeric_inputs` is the model's own numeric input list, which
        includes derived inputs such as the outlier flag."""
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        raw_numerics = [c for c in numeric_inputs if c not in derived_parents]
        raw_features = (*categorical_columns, *raw_numerics)
        position = {name: i for i, name in enumerate(raw_features)}

        encoded_names: list[str] = []
        group_index: list[int] = []
        for column in categorical_columns:
            encoded_names += [f"{column}[emb{k}]" for k in range(embedding_dim)]
            group_index += [position[column]] * embedding_dim
        for column in numeric_inputs:
            parent = derived_parents.get(column, column)
            if parent not in position:
                raise ValueError(f"Derived input {column!r} has parent {parent!r}, which is not a model input")
            encoded_names.append(column)
            group_index.append(position[parent])
        return cls(raw_features=raw_features, encoded_names=tuple(encoded_names), group_index=np.array(group_index))

    @property
    def n_encoded(self) -> int:
        return len(self.encoded_names)

    def columns_of(self, raw_feature: str) -> list[str]:
        """The encoded columns that carry `raw_feature`."""
        target = self.raw_features.index(raw_feature)
        return [name for name, group in zip(self.encoded_names, self.group_index) if group == target]

    def aggregate(self, values: np.ndarray) -> np.ndarray:
        """Sums `[n, n_encoded]` attributions into `[n, n_raw_features]`."""
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.n_encoded:
            raise ValueError(f"Expected values of shape [n, {self.n_encoded}], got {values.shape}")
        grouped = np.zeros((values.shape[0], len(self.raw_features)), dtype=np.float64)
        np.add.at(grouped, (slice(None), self.group_index), values)
        return grouped


def group_attribution(attribution: Attribution, grouping: FeatureGrouping) -> Attribution:
    """`attribution` over encoded columns -> the same attribution over raw
    features. The base value is untouched: grouping moves contributions
    between columns, it does not change their total."""
    if attribution.feature_names != grouping.encoded_names:
        raise ValueError(
            "Attribution columns do not match the grouping's encoded columns "
            f"(first mismatch at position {_first_mismatch(attribution.feature_names, grouping.encoded_names)})"
        )
    return replace(
        attribution,
        values=grouping.aggregate(attribution.values),
        feature_names=grouping.raw_features,
        metadata={**attribution.metadata, "grouped_from_n_encoded_columns": grouping.n_encoded},
    )


def _first_mismatch(a: Sequence[str], b: Sequence[str]) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def mean_abs_importance(attribution: Attribution) -> pd.Series:
    """Global importance: mean |attribution| per feature over the explained
    instances, in `feature_names` order (not sorted -- callers that rank
    sort explicitly, so two methods' Series stay aligned by default).

    Units are those of `attribution.target_space` (logits here). The
    absolute value is taken per instance before averaging, so a feature that
    pushes some respondents up and others down still registers as important.
    """
    return pd.Series(np.abs(attribution.values).mean(axis=0), index=list(attribution.feature_names))


def importance_share(importance: pd.Series) -> pd.Series:
    """Rescales a non-negative importance vector to sum to 1.

    Methods measured against different references produce attributions on
    different scales (a far-off baseline inflates every value), so raw
    magnitudes are not comparable across methods; each feature's share of
    the method's own total is.
    """
    values = importance.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("importance_share needs finite, non-negative importances (take |attribution| first)")
    total = values.sum()
    if total <= 0:
        raise ValueError("All importances are zero; a share of nothing is undefined")
    return importance / total
