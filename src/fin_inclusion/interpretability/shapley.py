"""Exact Shapley values by brute force: the ground truth other methods are checked against.

With only 10 raw features there are 2^10 = 1,024 coalitions, few enough to
evaluate every one. No sampling, no regression, no tree-specific algorithm:
this is the definition of the Shapley value computed directly, so when a
library method disagrees with it, the library method (or the game it is
playing) is what needs explaining.

The game, stated exactly:

  - Players: the raw survey features (not encoded columns).
  - Value of a coalition S for respondent x (INTERVENTIONAL value function):
        v(S) = mean over background rows b of f(x_S, b_notS)
    i.e. the features in S take the respondent's values, the rest take a
    background row's values, and the model's logit is averaged over the
    background. Features are replaced independently of each other, which
    breaks correlations between them: the model is asked about hybrid rows
    that may not exist in the population (a known limitation of the
    interventional choice, and the price of not needing a density model).
  - Shapley value of feature j:
        phi_j = sum over S not containing j of
                |S|! (M - |S| - 1)! / M!  *  ( v(S + j) - v(S) )
  - Base value: v(empty set) = mean logit over the background.

Efficiency holds exactly: sum_j phi_j + v(empty) = v(all) = f(x).
"""
from __future__ import annotations

from collections.abc import Callable
from math import factorial

import numpy as np
import pandas as pd

from fin_inclusion.interpretability.attribution import LOGIT_SPACE, Attribution
from fin_inclusion.interpretability.base import Explainer
from fin_inclusion.interpretability.model_adapters import RawFeatureModel

MAX_PLAYERS = 16
"""2^M coalitions x background rows are scored per instance; beyond this the
brute force stops being a sensible tool."""


def coalition_masks(n_players: int) -> np.ndarray:
    """`[2^M, M]` boolean matrix; row `s` has player `j` in the coalition iff bit `j` of `s` is set."""
    return ((np.arange(2**n_players)[:, None] >> np.arange(n_players)) & 1).astype(bool)


def shapley_from_coalition_values(values: np.ndarray, n_players: int) -> np.ndarray:
    """Shapley values from `values[s]` = v(coalition with bitmask `s`), for all 2^M coalitions."""
    masks = coalition_masks(n_players)
    sizes = masks.sum(axis=1)
    weight = np.array(
        [factorial(s) * factorial(n_players - s - 1) / factorial(n_players) for s in range(n_players)]
    )
    phi = np.empty(n_players)
    for j in range(n_players):
        without_j = np.flatnonzero(~masks[:, j])
        phi[j] = np.sum(weight[sizes[without_j]] * (values[without_j | (1 << j)] - values[without_j]))
    return phi


def exact_shapley_values(
    predict: Callable[[np.ndarray], np.ndarray], X: np.ndarray, background: np.ndarray
) -> tuple[np.ndarray, float]:
    """Exact interventional Shapley values of `predict` for each row of `X`.

    `X` is `[n, M]`, `background` is `[b, M]`; a feature is "absent" by
    taking a background row's value in that column. Returns `([n, M] values,
    base value)`.
    """
    X, background = np.asarray(X, dtype=np.float64), np.asarray(background, dtype=np.float64)
    n_players = X.shape[1]
    if background.ndim != 2 or background.shape[1] != n_players or len(background) == 0:
        raise ValueError(f"background must be [b, {n_players}] with b >= 1, got {background.shape}")
    if n_players > MAX_PLAYERS:
        raise ValueError(f"Brute-force Shapley over {n_players} players is intractable (limit {MAX_PLAYERS})")

    masks = coalition_masks(n_players)[:, None, :]  # [2^M, 1, M]
    phi = np.empty((len(X), n_players))
    for i, x in enumerate(X):
        hybrids = np.where(masks, x, background[None, :, :])  # [2^M, b, M]
        outputs = predict(hybrids.reshape(-1, n_players)).reshape(len(masks), len(background))
        phi[i] = shapley_from_coalition_values(outputs.mean(axis=1), n_players)
    # v(empty set) is the same for every instance: no feature comes from x.
    base_value = float(np.mean(predict(background)))
    return phi, base_value


class ExactShapleyExplainer(Explainer):
    """Brute-force interventional Shapley values on the raw features of `model`."""

    method = "exact_shapley"

    def __init__(self, model: RawFeatureModel, background_raw: pd.DataFrame):
        self.model = model
        self.background_codes = model.to_codes(background_raw)
        self.reference = f"mean logit over {len(self.background_codes)} background rows"
        self.metadata: dict = {}

    @classmethod
    def at_point(cls, model: RawFeatureModel, baseline: np.ndarray, description: str, name: str) -> "ExactShapleyExplainer":
        """Shapley values against ONE reference point (a code vector, which may
        hold `UNKNOWN_CODE` or non-integer numeric means) instead of a
        background distribution. With the same point as Integrated Gradients
        or Feature Ablation, the three share their reference exactly, so any
        remaining difference between them is the method, not the reference."""
        explainer = cls.__new__(cls)
        explainer.model = model
        explainer.background_codes = np.atleast_2d(np.asarray(baseline, dtype=np.float64))
        explainer.reference = f"one baseline point ({description})"
        explainer.metadata = {"baseline": name}
        return explainer

    def explain(self, X_raw: pd.DataFrame) -> Attribution:
        values, base_value = exact_shapley_values(
            self.model.predict_logit_codes, self.model.to_codes(X_raw), self.background_codes
        )
        return Attribution(
            values=values,
            base_value=base_value,
            feature_names=self.model.raw_features,
            method=self.method,
            model_name=self.model.name,
            target_space=LOGIT_SPACE,
            baseline_description=(
                f"{self.reference}; interventional value function (absent features take the reference's "
                f"values), exact over all {2 ** len(self.model.raw_features)} coalitions of raw features"
            ),
            metadata={"n_background": len(self.background_codes),
                      "n_coalitions": 2 ** len(self.model.raw_features), **self.metadata},
        )


# --- Pairwise interactions -----------------------------------------------------
#
# The Shapley interaction index splits each feature's Shapley value into a
# part it earns alone (the diagonal, its "main effect") and parts it earns
# only together with one other feature (off-diagonal, shared equally between
# the pair). Same game as above -- interventional, raw features, same
# background -- so the matrices are directly comparable with the attributions.


def interactions_from_coalition_values(values: np.ndarray, n_players: int) -> np.ndarray:
    """`[M, M]` Shapley interaction matrix from v(S) for all 2^M coalitions.

    Off-diagonal: Phi_ij = sum over S without i, j of
        |S|! (M - |S| - 2)! / (2 (M - 1)!) * [v(S+i+j) - v(S+i) - v(S+j) + v(S)]
    (the pair's joint effect, split equally between Phi_ij and Phi_ji).
    Diagonal: Phi_ii = phi_i - sum_{j != i} Phi_ij, so row i sums to feature
    i's Shapley value and the whole matrix sums to f(x) - base value.
    """
    masks = coalition_masks(n_players)
    sizes = masks.sum(axis=1)
    weight = np.array(
        [factorial(s) * factorial(n_players - s - 2) / (2 * factorial(n_players - 1)) for s in range(n_players - 1)]
    )
    matrix = np.zeros((n_players, n_players))
    for i in range(n_players):
        for j in range(i + 1, n_players):
            without = np.flatnonzero(~masks[:, i] & ~masks[:, j])
            bit_i, bit_j = 1 << i, 1 << j
            joint = values[without | bit_i | bit_j] - values[without | bit_i] - values[without | bit_j] + values[without]
            matrix[i, j] = matrix[j, i] = np.sum(weight[sizes[without]] * joint)
    phi = shapley_from_coalition_values(values, n_players)
    np.fill_diagonal(matrix, phi - (matrix.sum(axis=1) - np.diag(matrix)))
    return matrix


def exact_shapley_interactions(
    predict: Callable[[np.ndarray], np.ndarray], X: np.ndarray, background: np.ndarray
) -> tuple[np.ndarray, float]:
    """`([n, M, M] interaction matrices, base value)` for each row of `X`,
    by brute force over all coalitions (same game as `exact_shapley_values`)."""
    X, background = np.asarray(X, dtype=np.float64), np.asarray(background, dtype=np.float64)
    n_players = X.shape[1]
    if background.ndim != 2 or background.shape[1] != n_players or len(background) == 0:
        raise ValueError(f"background must be [b, {n_players}] with b >= 1, got {background.shape}")
    if not 2 <= n_players <= MAX_PLAYERS:
        raise ValueError(f"Interactions need 2..{MAX_PLAYERS} players, got {n_players}")
    masks = coalition_masks(n_players)[:, None, :]
    result = np.empty((len(X), n_players, n_players))
    for i, x in enumerate(X):
        hybrids = np.where(masks, x, background[None, :, :])
        outputs = predict(hybrids.reshape(-1, n_players)).reshape(len(masks), len(background))
        result[i] = interactions_from_coalition_values(outputs.mean(axis=1), n_players)
    return result, float(np.mean(predict(background)))
