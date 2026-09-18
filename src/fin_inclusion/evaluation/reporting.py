"""Turns `CVFoldResult` lists into the results tables from README §6.

Every metric is reported as mean ± std across folds, never a single
number (README §6: "a model at 0.65 ± 0.02 PR-AUC is a materially
different result than 0.65 ± 0.15, even with an identical headline
number").
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: every plot here is saved to disk, never shown interactively
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from fin_inclusion.evaluation.cv_runner import CVFoldResult, NestedFoldResult

METRIC_NAMES = ("pr_auc", "roc_auc", "f1", "precision", "recall", "accuracy")


def summarize_metrics(
    results: list[CVFoldResult] | list[NestedFoldResult], metric_names: tuple[str, ...] = METRIC_NAMES
) -> dict[str, tuple[float, float]]:
    """Mean, std (across folds) per metric."""
    return {
        m: (
            float(np.mean([r.metrics[m] for r in results])),
            float(np.std([r.metrics[m] for r in results])),
        )
        for m in metric_names
    }


def summarize_country_metrics(
    results: list[CVFoldResult] | list[NestedFoldResult], metric_names: tuple[str, ...] = ("pr_auc", "f1")
) -> pd.DataFrame:
    """Mean ± std per country per metric, aggregated across folds (README §6 secondary reporting)."""
    countries = sorted(results[0].country_metrics)
    rows = []
    for country in countries:
        row: dict[str, float | str] = {"country": country}
        for metric in metric_names:
            values = np.array([r.country_metrics[country][metric] for r in results])
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std())
        rows.append(row)
    return pd.DataFrame(rows)


def results_table_row(
    model_name: str, imbalance_strategy: str, summary: dict[str, tuple[float, float]]
) -> dict[str, str]:
    """One row of the README §6 core results table (`Model | Imbalance strategy | PR-AUC | F1 | ROC-AUC`)."""
    row = {"model": model_name, "imbalance_strategy": imbalance_strategy}
    for metric in ("pr_auc", "f1", "roc_auc"):
        mean, std = summary[metric]
        row[metric] = f"{mean:.3f} ± {std:.3f}"
    return row


def country_table_markdown(
    country_summary: pd.DataFrame, metric_names: tuple[str, ...] = ("pr_auc", "f1")
) -> str:
    """Renders `summarize_country_metrics`'s output as a `mean ± std` markdown table."""
    headers = ["Country"] + [m.upper().replace("_", "-") for m in metric_names]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for _, row in country_summary.iterrows():
        cells = [str(row["country"])]
        cells += [f"{row[f'{m}_mean']:.3f} ± {row[f'{m}_std']:.3f}" for m in metric_names]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def results_table_markdown(rows: list[dict[str, str]]) -> str:
    """Builds the README §6 markdown table by hand (no `tabulate` dependency)."""
    headers = ["Model", "Imbalance strategy", "PR-AUC", "F1", "ROC-AUC"]
    keys = ["model", "imbalance_strategy", "pr_auc", "f1", "roc_auc"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[k]) for k in keys) + " |")
    return "\n".join(lines)


FULL_METRIC_NAMES = ("pr_auc", "roc_auc", "f1", "precision", "recall", "accuracy")


def full_results_table_row(
    model_name: str, imbalance_strategy: str, summary: dict[str, tuple[float, float]]
) -> dict[str, str]:
    """Like `results_table_row`, but every metric (incl. precision/recall/
    accuracy) rather than just the README §6 headline three."""
    row = {"model": model_name, "imbalance_strategy": imbalance_strategy}
    for metric in FULL_METRIC_NAMES:
        mean, std = summary[metric]
        row[metric] = f"{mean:.3f} ± {std:.3f}"
    return row


def full_results_table_markdown(rows: list[dict[str, str]]) -> str:
    """`full_results_table_row`'s markdown table -- PR-AUC/ROC-AUC/F1/
    Precision/Recall/Accuracy. Accuracy is included for completeness, not as
    a metric to optimize against: see `metrics.py`'s module docstring for
    why it's misleading at this dataset's ~14% positive rate."""
    headers = ["Model", "Imbalance strategy", "PR-AUC", "ROC-AUC", "F1", "Precision", "Recall", "Accuracy"]
    keys = ["model", "imbalance_strategy", "pr_auc", "roc_auc", "f1", "precision", "recall", "accuracy"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[k]) for k in keys) + " |")
    return "\n".join(lines)


def aggregate_confusion_matrix(results: list[CVFoldResult] | list[NestedFoldResult]) -> np.ndarray:
    """Sums the per-fold confusion matrices into one aggregate over every
    outer-validation fold combined -- every prediction made across the CV
    run counted exactly once (each row's outer fold holds out a disjoint
    slice, so summing doesn't double-count anything)."""
    return np.sum([r.confusion_matrix for r in results], axis=0)


def confusion_matrix_markdown(cm: np.ndarray, labels: tuple[str, str] = ("No", "Yes")) -> str:
    """Renders a 2x2 `[[TN, FP], [FN, TP]]` confusion matrix (rows=actual, columns=predicted)."""
    tn, fp, fn, tp = cm.ravel()
    headers = ["Actual \\ Predicted", f"Pred {labels[0]}", f"Pred {labels[1]}"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
        f"| Actual {labels[0]} | {tn} | {fp} |",
        f"| Actual {labels[1]} | {fn} | {tp} |",
    ]
    return "\n".join(lines)


def plot_confusion_matrix_heatmap(
    cm: np.ndarray,
    output_path: Path,
    title: str,
    labels: tuple[str, str] = ("No", "Yes"),
) -> None:
    """Saves a single-hue (count = magnitude, not polarity) annotated heatmap."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar_kws={"label": "Count"},
        xticklabels=[f"Pred {l}" for l in labels],
        yticklabels=[f"Actual {l}" for l in labels],
        square=True,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def fold_pr_auc_scores(results: list[CVFoldResult] | list[NestedFoldResult]) -> list[float]:
    return [r.metrics["pr_auc"] for r in results]


@dataclass(frozen=True)
class PairedComparisonResult:
    """Both tests are cheap to compute given the 5 fold-level scores already
    exist -- reported together (README §6 names either as acceptable) rather
    than picking one. Wilcoxon is the more robust choice at n=5 (no normality
    assumption); the paired t-test is included for comparability."""

    ttest_statistic: float
    ttest_p_value: float
    wilcoxon_statistic: float
    wilcoxon_p_value: float


def compare_paired_pr_auc(scores_a: list[float], scores_b: list[float]) -> PairedComparisonResult:
    """Paired comparison of fold-level PR-AUC scores between two models
    (Phase 4.7) -- `scores_a`/`scores_b` must be the same folds, same order."""
    t_stat, t_p = stats.ttest_rel(scores_a, scores_b)
    w_stat, w_p = stats.wilcoxon(scores_a, scores_b)
    return PairedComparisonResult(
        ttest_statistic=float(t_stat),
        ttest_p_value=float(t_p),
        wilcoxon_statistic=float(w_stat),
        wilcoxon_p_value=float(w_p),
    )
