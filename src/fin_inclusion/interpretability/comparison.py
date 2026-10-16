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


SHARED_ZERO_TOLERANCE = 1e-9


def shared_zeros(a_values: np.ndarray, b_values: np.ndarray) -> np.ndarray:
    """Cells where BOTH attributions are (numerically) exactly zero.

    With a single baseline point, a feature on which the respondent already
    equals the baseline gets exactly zero from Ablation, IG and point-Shapley
    alike -- whatever the model. Two such explanations then "agree" on those
    features by construction, which inflates rank correlations."""
    return (np.abs(a_values) < SHARED_ZERO_TOLERANCE) & (np.abs(b_values) < SHARED_ZERO_TOLERANCE)


def instance_rank_correlations_excluding_shared_zeros(a_values: np.ndarray, b_values: np.ndarray) -> np.ndarray:
    """Per respondent: Spearman correlation of |attribution| over only the
    features that are not zero in both. NaN when fewer than 3 remain."""
    keep = ~shared_zeros(a_values, b_values)
    rho = np.full(len(a_values), np.nan)
    for i in range(len(a_values)):
        if keep[i].sum() >= 3:
            a_row, b_row = np.abs(a_values[i, keep[i]]), np.abs(b_values[i, keep[i]])
            if np.ptp(a_row) > 0 and np.ptp(b_row) > 0:
                rho[i] = spearmanr(a_row, b_row).statistic
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


AGREEMENT_METRICS = (
    "global_spearman", "global_kendall", "top3_overlap", "top5_overlap",
    "instance_spearman_median", "instance_spearman_median_excluding_shared_zeros",
    "same_top_feature_share", "sign_agreement_share",
)


def agreement_with_ci(a_values: np.ndarray, b_values: np.ndarray, reps: int, seed: int, level: float = 0.95) -> dict:
    """Every agreement metric for two aligned `[n, features]` arrays, each with
    a percentile bootstrap interval over respondents."""
    from fin_inclusion.interpretability.bootstrap import bootstrap_ci

    rho = instance_rank_correlations(a_values, b_values)
    rho_informative = instance_rank_correlations_excluding_shared_zeros(a_values, b_values)
    same_top = np.abs(a_values).argmax(axis=1) == np.abs(b_values).argmax(axis=1)
    nonzero = (a_values != 0) & (b_values != 0)
    same_sign = ((np.sign(a_values) == np.sign(b_values)) & nonzero).sum(axis=1)
    counted = nonzero.sum(axis=1)

    def statistic(rows: np.ndarray) -> np.ndarray:
        g = global_agreement(a_values[rows], b_values[rows])
        return np.array([
            g["spearman"], g["kendall"], g["top3_overlap"], g["top5_overlap"],
            np.nanmedian(rho[rows]), np.nanmedian(rho_informative[rows]), same_top[rows].mean(), same_sign[rows].sum() / max(counted[rows].sum(), 1),
        ])

    result = bootstrap_ci(statistic, len(a_values), reps, seed, level)
    return {
        "n": int(len(a_values)),
        **{name: {"estimate": float(result.estimate[k]), "ci": [float(result.lower[k]), float(result.upper[k])]}
           for k, name in enumerate(AGREEMENT_METRICS)},
        "mean_shared_zero_features_per_row": float(shared_zeros(a_values, b_values).sum(axis=1).mean()),
        "instance_spearman_values": rho,  # for histograms; dropped before saving to JSON
    }


def per_feature_disagreement(a_values: np.ndarray, b_values: np.ndarray) -> np.ndarray:
    """Per feature: mean |a - b| relative to the average size of the two,
    i.e. how much the two disagree about this feature, in units of the
    feature's own typical attribution (0 = identical; 2 = opposite)."""
    scale = (np.abs(a_values).mean(axis=0) + np.abs(b_values).mean(axis=0)) / 2
    return np.abs(a_values - b_values).mean(axis=0) / np.where(scale > 0, scale, np.nan)


def cramers_v(a: np.ndarray, b: np.ndarray) -> float:
    """Cramer's V between two categorical columns (0 = independent, 1 = one
    determines the other). Plain (not bias-corrected) V: with ~23,500 rows
    the small-sample bias is negligible."""
    from scipy.stats import chi2_contingency

    _, a_codes = np.unique(a, return_inverse=True)
    _, b_codes = np.unique(b, return_inverse=True)
    table = np.zeros((a_codes.max() + 1, b_codes.max() + 1))
    np.add.at(table, (a_codes, b_codes), 1)
    if min(table.shape) < 2:
        return 0.0
    chi2 = chi2_contingency(table, correction=False)[0]
    return float(np.sqrt(chi2 / (table.sum() * (min(table.shape) - 1))))
