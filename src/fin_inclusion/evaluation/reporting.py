"""Turns `CVFoldResult` lists into the results tables from README §6.

Every metric is reported as mean ± std across folds, never a single
number (README §6: "a model at 0.65 ± 0.02 PR-AUC is a materially
different result than 0.65 ± 0.15, even with an identical headline
number").
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fin_inclusion.evaluation.cv_runner import CVFoldResult

METRIC_NAMES = ("pr_auc", "roc_auc", "f1", "precision", "recall")


def summarize_metrics(
    results: list[CVFoldResult], metric_names: tuple[str, ...] = METRIC_NAMES
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
    results: list[CVFoldResult], metric_names: tuple[str, ...] = ("pr_auc", "f1")
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
