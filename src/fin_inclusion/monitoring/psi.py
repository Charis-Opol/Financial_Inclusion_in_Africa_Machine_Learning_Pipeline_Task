"""Population Stability Index (PSI) and the binning it depends on.

    PSI = sum_i (actual_i - expected_i) * ln(actual_i / expected_i)

over matching bins of a reference ("expected") and a recent ("actual")
distribution. Standard reading: < 0.10 stable, 0.10-0.25 moderate shift
(investigate), > 0.25 significant shift (act: the model is scoring a
population it wasn't trained on).

Binning is fixed by the *reference* and reused for every window, so values
are comparable over time:
  - numeric: bin edges at the reference deciles (deduplicated -- discrete
    features like household_size have heavy ties), open-ended at both ends
    so out-of-range values still land in a bin;
  - categorical: one bin per training level plus an "(unseen)" bin, which is
    empty in the reference by construction.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

PSI_STABLE_BELOW = 0.10
PSI_SIGNIFICANT_ABOVE = 0.25
# Empty bins make ln(a/e) infinite; flooring proportions at a small epsilon
# is the conventional fix. It caps a single empty bin's contribution
# instead of letting one unseen category produce PSI = inf.
EPSILON = 1e-4
UNSEEN_LABEL = "(unseen)"
DECILES = np.linspace(0.1, 0.9, 9)


def psi(expected: Sequence[float], actual: Sequence[float]) -> float:
    e = np.maximum(np.asarray(expected, dtype=np.float64), EPSILON)
    a = np.maximum(np.asarray(actual, dtype=np.float64), EPSILON)
    if e.shape != a.shape:
        raise ValueError(f"Bin count mismatch: expected {e.shape}, actual {a.shape}")
    return float(np.sum((a - e) * np.log(a / e)))


def per_bin_contributions(expected: Sequence[float], actual: Sequence[float]) -> np.ndarray:
    e = np.maximum(np.asarray(expected, dtype=np.float64), EPSILON)
    a = np.maximum(np.asarray(actual, dtype=np.float64), EPSILON)
    return (a - e) * np.log(a / e)


def interpret(value: float) -> str:
    if value < PSI_STABLE_BELOW:
        return "stable"
    if value <= PSI_SIGNIFICANT_ABOVE:
        return "moderate"
    return "significant"


def decile_edges(values: np.ndarray) -> list[float]:
    # inverted_cdf: every edge is an actually observed value. Linear
    # interpolation would put edges like 1.5 between integer household sizes,
    # creating bins no real value can ever fall into.
    return np.unique(np.quantile(np.asarray(values, dtype=np.float64), DECILES, method="inverted_cdf")).tolist()


def numeric_proportions(values: Sequence[float], edges: Sequence[float]) -> np.ndarray:
    """Share of `values` in each of the len(edges)+1 bins (-inf, e1], (e1, e2], ..., (ek, inf)."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("Cannot compute proportions of an empty sample")
    bins = np.searchsorted(np.asarray(edges), values, side="left")
    return np.bincount(bins, minlength=len(edges) + 1) / values.size


def numeric_bin_labels(edges: Sequence[float]) -> list[str]:
    bounds = [-math.inf, *edges, math.inf]
    return [f"({_fmt(lo)}, {_fmt(hi)}]" for lo, hi in zip(bounds[:-1], bounds[1:])]


def categorical_proportions(values: Sequence[str], levels: Sequence[str]) -> np.ndarray:
    """Share of `values` per level, with a trailing bin for anything not in `levels`."""
    if len(values) == 0:
        raise ValueError("Cannot compute proportions of an empty sample")
    index = {level: i for i, level in enumerate(levels)}
    counts = np.zeros(len(levels) + 1)
    for value in values:
        counts[index.get(value, len(levels))] += 1
    return counts / len(values)


def categorical_bin_labels(levels: Sequence[str]) -> list[str]:
    return [*levels, UNSEEN_LABEL]


def _fmt(x: float) -> str:
    if math.isinf(x):
        return "-inf" if x < 0 else "inf"
    return f"{x:g}" if abs(x) >= 1 else f"{x:.3f}"
