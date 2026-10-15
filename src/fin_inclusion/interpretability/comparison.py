"""Agreement between two sets of attributions for the same respondents.

Used whenever one explanation is set against another: two baselines for the
same method (Phase 4), two methods on the same model, or two models
(Phase 6). Agreement is measured on what a reader would take away:

  - the global ranking of features (by mean |attribution|),
  - the overlap of the top-k features,
  - for each respondent, whether the two put the features in the same order
    and name the same most important feature,
  - whether contributions point the same way.

Agreement between two explanations says nothing about whether either is
correct; that needs a ground truth or a faithfulness test.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau, spearmanr

from fin_inclusion.interpretability.attribution import Attribution


def _aligned(a: Attribution, b: Attribution) -> tuple[np.ndarray, np.ndarray]:
    if a.feature_names != b.feature_names:
        raise ValueError("Attributions are over different features or in a different order")
    if a.n_instances != b.n_instances:
        raise ValueError(f"Attributions cover different numbers of rows ({a.n_instances} vs {b.n_instances})")
    rows_a, rows_b = a.metadata.get("rows"), b.metadata.get("rows")
    if rows_a is not None and rows_b is not None and rows_a != rows_b:
        raise ValueError("Attributions are for different rows")
    return a.values, b.values


def global_agreement(a_values: np.ndarray, b_values: np.ndarray, top_k: tuple[int, ...] = (3, 5)) -> dict:
    """Agreement of the two global rankings (mean |attribution| per feature)."""
    importance_a, importance_b = np.abs(a_values).mean(axis=0), np.abs(b_values).mean(axis=0)
    order_a, order_b = np.argsort(-importance_a), np.argsort(-importance_b)
    result = {
        "spearman": float(spearmanr(importance_a, importance_b).statistic),
        "kendall": float(kendalltau(importance_a, importance_b).statistic),
    }
    for k in top_k:
        result[f"top{k}_overlap"] = len(set(order_a[:k]) & set(order_b[:k])) / k
    return result


def instance_rank_correlations(a_values: np.ndarray, b_values: np.ndarray) -> np.ndarray:
    """Per respondent: Spearman correlation between the two |attribution|
    vectors, i.e. do the two put the features in the same order of
    importance for this person. NaN for a row where either is constant."""
    ranks_a = np.argsort(np.argsort(-np.abs(a_values), axis=1), axis=1).astype(float)
    ranks_b = np.argsort(np.argsort(-np.abs(b_values), axis=1), axis=1).astype(float)
    ranks_a -= ranks_a.mean(axis=1, keepdims=True)
    ranks_b -= ranks_b.mean(axis=1, keepdims=True)
    denominator = np.sqrt((ranks_a**2).sum(axis=1) * (ranks_b**2).sum(axis=1))
    constant = (np.ptp(np.abs(a_values), axis=1) == 0) | (np.ptp(np.abs(b_values), axis=1) == 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = (ranks_a * ranks_b).sum(axis=1) / denominator
    rho[constant] = np.nan
    return rho


def summarize_agreement(a: Attribution, b: Attribution) -> dict:
    """Every agreement measure for two aligned attributions."""
    a_values, b_values = _aligned(a, b)
    rho = instance_rank_correlations(a_values, b_values)
    nonzero = (a_values != 0) & (b_values != 0)
    return {
        "n": int(len(a_values)),
        "global": global_agreement(a_values, b_values),
        "instance_spearman": {
            "median": float(np.nanmedian(rho)),
            "q25": float(np.nanquantile(rho, 0.25)),
            "q75": float(np.nanquantile(rho, 0.75)),
            "share_below_0.5": float(np.nanmean(rho < 0.5)),
        },
        "same_top_feature_share": float(
            (np.abs(a_values).argmax(axis=1) == np.abs(b_values).argmax(axis=1)).mean()
        ),
        "sign_agreement_share": float((np.sign(a_values[nonzero]) == np.sign(b_values[nonzero])).mean()),
        "mean_abs_difference": float(np.abs(a_values - b_values).mean()),
    }
