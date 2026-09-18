"""Classification metrics (README §6): PR-AUC primary, ROC-AUC/F1/precision/
recall/accuracy alongside, plus the confusion matrix.

A single function so every caller (CV runner, ablation report) computes
metrics identically. `threshold` defaults to 0.5 in Phase 3 (explicitly
*not* tuned yet); Phase 4 passes the F1-maximizing threshold selected on
inner validation data instead.

`accuracy` is computed and reported, but deliberately isn't README §6's
headline metric: at this dataset's ~14% positive rate, a model that always
predicts "No" scores ~86% accuracy while catching zero true positives --
the same reasoning §6 already gives for preferring PR-AUC over ROC-AUC
applies even more strongly to accuracy.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """`y_true` is 0/1; `y_score` is P(positive class) in [0, 1]."""
    y_pred = (y_score >= threshold).astype(int)
    return {
        "pr_auc": average_precision_score(y_true, y_score),
        "roc_auc": roc_auc_score(y_true, y_score),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def compute_confusion_matrix(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5
) -> np.ndarray:
    """2x2 array `[[TN, FP], [FN, TP]]` (sklearn's `labels=[0, 1]` ordering)."""
    y_pred = (y_score >= threshold).astype(int)
    return confusion_matrix(y_true, y_pred, labels=[0, 1])
