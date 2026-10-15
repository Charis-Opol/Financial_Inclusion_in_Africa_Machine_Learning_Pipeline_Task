"""Bootstrap confidence intervals over explained instances.

Any number summarising a set of respondents (a feature's mean |attribution|,
a rank correlation between two methods) depends on which respondents were
sampled. Resampling the instances with replacement and recomputing the
statistic shows how much. This matters most for Uganda-only results, where
the sample is small.

The interval reflects sampling of respondents only. It says nothing about
whether the model, the background set or the method is right.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BootstrapResult:
    """`estimate` on the full sample, with a percentile interval from the resamples."""

    estimate: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    level: float
    reps: int
    n_instances: int


def bootstrap_ci(
    statistic: Callable[[np.ndarray], np.ndarray],
    n_instances: int,
    reps: int,
    seed: int,
    level: float = 0.95,
) -> BootstrapResult:
    """Percentile bootstrap for `statistic(row_positions)`.

    `statistic` receives an integer array of row positions (the full range
    for the point estimate, a resample with replacement for each replicate)
    and returns a scalar or 1-D array. Taking positions rather than data
    lets one call resample several aligned arrays at once (two methods'
    attributions for the same respondents, say).
    """
    if n_instances < 2:
        raise ValueError(f"Need at least 2 instances to bootstrap, got {n_instances}")
    if not 0 < level < 1:
        raise ValueError(f"level must be in (0, 1), got {level}")
    rng = np.random.default_rng(seed)
    estimate = np.asarray(statistic(np.arange(n_instances)), dtype=np.float64)
    replicates = np.stack(
        [np.asarray(statistic(rng.integers(0, n_instances, size=n_instances)), dtype=np.float64) for _ in range(reps)]
    )
    tail = (1 - level) / 2
    # nan-aware: a resample can make a statistic undefined (e.g. a rank
    # correlation when a resample is constant); those replicates are skipped.
    lower, upper = np.nanquantile(replicates, [tail, 1 - tail], axis=0)
    return BootstrapResult(estimate=estimate, lower=lower, upper=upper, level=level, reps=reps, n_instances=n_instances)


def bootstrap_mean_abs_importance(values: np.ndarray, reps: int, seed: int, level: float = 0.95) -> BootstrapResult:
    """Mean |attribution| per feature for `[n, n_features]` values, with a bootstrap CI."""
    absolute = np.abs(np.asarray(values, dtype=np.float64))
    return bootstrap_ci(lambda rows: absolute[rows].mean(axis=0), len(absolute), reps, seed, level)
