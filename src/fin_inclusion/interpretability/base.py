"""Shared explainer interface (README §7: Open/Closed + Liskov Substitution).

Every explanation method (TreeSHAP, KernelSHAP, the exact-Shapley oracle,
Integrated Gradients, Feature Ablation) implements this ABC, so the scripts
that save artifacts, draw plots and compare methods can drive any of them
through the same call without knowing which one it is, and a new method can
be added without editing an existing one.

The contract is deliberately stated in terms of RAW survey rows: the two
models encode the same respondent differently (40 one-hot columns for
XGBoost, 8 embeddings + 3 numerics for the MLP), so the raw features are the
only unit in which their explanations can be compared. Each implementation
owns the encoding and the aggregation back to raw features
(`grouping.FeatureGrouping`).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from fin_inclusion.interpretability.attribution import Attribution


class Explainer(ABC):
    @abstractmethod
    def explain(self, X_raw: pd.DataFrame) -> Attribution:
        """Attributes the model's output for each row of `X_raw` (raw,
        unencoded survey columns) to the raw input features.

        The returned `Attribution` has one column per raw feature and states
        its own reference and target space; implementations must not return
        encoded-column attributions for the caller to aggregate."""
