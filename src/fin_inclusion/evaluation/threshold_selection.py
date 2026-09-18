"""F1-maximizing threshold search on inner validation data (README §6, Phase 4.5).

Selected honestly: only ever called on a fold's *inner* validation split,
never on the outer validation fold the final metrics are computed on --
the outer fold must stay unseen by every step of tuning, threshold
selection included.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve


def select_f1_maximizing_threshold(
    y_true: np.ndarray, y_score: np.ndarray
) -> tuple[float, float]:
    """Scans every threshold implied by `y_score`'s distinct values; returns `(threshold, f1)`."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if len(thresholds) == 0:
        return 0.5, 0.0

    f1_scores = 2 * precision * recall / (precision + recall + 1e-12)
    f1_scores = f1_scores[:-1]  # last point has no corresponding threshold
    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])
