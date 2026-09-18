"""PR curve overlay plotting (README §6 secondary reporting, Phase 4 ablation report).

Not part of the nested-CV metrics themselves -- `run_nested_cv` scores each
outer fold once, at one threshold, which doesn't give a full precision/recall
curve. This module fits the already-tuned winning models once more on a
held-out split purely to draw the curve; it never feeds back into the
reported PR-AUC/F1/ROC-AUC numbers.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

CURVE_COLORS = {0: "#4C72B0", 1: "#C44E52", 2: "#55A868", 3: "#8172B2"}


def plot_pr_curve_overlay(
    curves: dict[str, tuple[np.ndarray, np.ndarray]], output_path: Path
) -> dict[str, float]:
    """`curves`: label -> (y_true, y_score). Returns each label's average precision."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    average_precisions = {}
    for i, (label, (y_true, y_score)) in enumerate(curves.items()):
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        average_precisions[label] = float(ap)
        ax.plot(recall, precision, color=CURVE_COLORS[i % len(CURVE_COLORS)], linewidth=2, label=f"{label} (AP={ap:.3f})")

    positive_rate = float(np.mean(next(iter(curves.values()))[0]))
    ax.axhline(positive_rate, color="gray", linestyle="--", linewidth=1, label=f"No-skill baseline ({positive_rate:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve: XGBoost-best vs. PyTorch-best")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return average_precisions
