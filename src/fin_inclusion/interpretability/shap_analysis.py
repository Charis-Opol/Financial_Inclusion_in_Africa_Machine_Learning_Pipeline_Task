"""SHAP global importance + dependence plots on the tuned XGBoost model (Phase 4.8).

Explains the XGBoost model's learned associations, not causal drivers of
financial inclusion (README §10, Known Limitations). Doubles as a sanity
check: `cellphone_access`, `education_level`, `country` are expected to
dominate per Phase 1's bivariate EDA and the Phase 1.11 Cramer's V ranking
-- this confirms the tuned model learned something consistent with the EDA,
rather than something spurious.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: every plot here is saved to disk, never shown interactively
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from fin_inclusion.models.xgboost_model import XGBoostModel


def compute_shap_explanation(
    model: XGBoostModel, X: pd.DataFrame, max_samples: int | None = 2000, seed: int = 42
) -> shap.Explanation:
    """`max_samples` subsamples `X` for tractability; SHAP's exact tree
    explainer is O(rows x trees x depth), unnecessary to run on the full
    training set to get a stable global importance ranking."""
    X_sample = X if max_samples is None or len(X) <= max_samples else X.sample(max_samples, random_state=seed)
    explainer = shap.TreeExplainer(model.booster)
    return explainer(X_sample)


def global_importance(explanation: shap.Explanation) -> pd.Series:
    """Mean |SHAP value| per feature, sorted descending."""
    values = np.abs(explanation.values).mean(axis=0)
    return pd.Series(values, index=explanation.feature_names).sort_values(ascending=False)


def plot_global_importance(explanation: shap.Explanation, output_path: Path, max_display: int = 15) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shap.plots.bar(explanation, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_dependence(explanation: shap.Explanation, feature: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shap.plots.scatter(explanation[:, feature], show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
