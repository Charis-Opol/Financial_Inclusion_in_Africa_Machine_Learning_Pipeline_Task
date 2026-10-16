"""Faithfulness checks: does an explanation track what the model actually does?

Two explanations agreeing does not make either correct. These tests ask the
MODEL instead, by changing its input in the way an explanation says matters
and measuring the logit:

  - Deletion: remove the features an explanation ranks most important, one
    at a time; a faithful ranking moves the logit away from f(x) fastest.
  - Insertion: start from "everything removed" and add the top-ranked
    features back; a faithful ranking returns to f(x) fastest.
  - Comprehensiveness / sufficiency at k: the deletion / insertion curve at
    k features -- how much the top-k carry on their own.
  - Randomisation (Adebayo et al., 2018): scramble the MLP's weights layer by
    layer; an explanation that barely changes is not explaining the model.
  - Stability: nudge a numeric input slightly; an explanation that jumps
    while the model's output barely moves is fragile.

"Removing" a feature needs a reference, and the choice favours the methods
built on the same one: replacing it with background rows (averaged) is the
SHAP methods' own value function; replacing it with the modal point is
IG's and Ablation's. Both are run and the bias is reported, not hidden.

Everything is in logits, on raw features, through the model adapters.
"""
from __future__ import annotations

import copy
from collections.abc import Callable

import numpy as np

MAX_ROWS_PER_CALL = 400_000


def ranking(values: np.ndarray) -> np.ndarray:
    """Per row, feature positions from largest to smallest |attribution| (ties: lower position first)."""
    return np.argsort(-np.abs(values), axis=1, kind="stable")


def random_rankings(n_rows: int, n_features: int, n_permutations: int, seed: int) -> np.ndarray:
    """`[n_permutations, n_rows, n_features]` random feature orders: the no-information baseline."""
    rng = np.random.default_rng(seed)
    return np.stack([np.argsort(rng.random((n_rows, n_features)), axis=1) for _ in range(n_permutations)])


def coalition_values(
    predict: Callable[[np.ndarray], np.ndarray], X: np.ndarray, keep: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    """Mean logit when row i keeps its own values where `keep[i]` is True and
    takes each reference row's values elsewhere, averaged over the reference
    rows. With one reference row this is a single model call per row."""
    n, m = X.shape
    reference = np.atleast_2d(reference)
    per_chunk = max(1, MAX_ROWS_PER_CALL // len(reference))
    out = np.empty(n)
    for start in range(0, n, per_chunk):
        stop = min(start + per_chunk, n)
        hybrid = np.where(keep[start:stop, None, :], X[start:stop, None, :], reference[None, :, :])
        out[start:stop] = predict(hybrid.reshape(-1, m)).reshape(stop - start, len(reference)).mean(axis=1)
    return out


def removal_curves(
    predict: Callable[[np.ndarray], np.ndarray], X: np.ndarray, order: np.ndarray, reference: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """`(deletion, insertion)`, each `[n, M + 1]` logits for k = 0..M:
    deletion[k] = top-k features removed; insertion[k] = only the top-k kept."""
    n, m = X.shape
    rows = np.arange(n)[:, None]
    deletion, insertion = np.empty((n, m + 1)), np.empty((n, m + 1))
    for k in range(m + 1):
        top = np.zeros((n, m), dtype=bool)
        top[rows, order[:, :k]] = True
        deletion[:, k] = coalition_values(predict, X, ~top, reference)
        insertion[:, k] = coalition_values(predict, X, top, reference)
    return deletion, insertion


def faithfulness_scores(logit: np.ndarray, deletion: np.ndarray, insertion: np.ndarray, ks=(1, 3, 5)) -> dict:
    """Per-row scores (logits):

    - `deletion_area`: mean over k of |f(x) - f(top-k removed)|. Higher = the
      ranking finds what moves the model.
    - `insertion_gap`: mean over k of |f(x) - f(only top-k kept)|. Lower =
      the top-ranked features recover the output sooner.
    - `comprehensiveness@k` = |f(x) - f(top-k removed)| (higher better);
      `sufficiency@k` = |f(x) - f(only top-k kept)| (lower better).
    """
    removed = np.abs(logit[:, None] - deletion)
    kept = np.abs(logit[:, None] - insertion)
    scores = {"deletion_area": removed.mean(axis=1), "insertion_gap": kept.mean(axis=1)}
    for k in (k for k in ks if k < removed.shape[1]):  # k beyond the number of features is undefined
        scores[f"comprehensiveness@{k}"] = removed[:, k]
        scores[f"sufficiency@{k}"] = kept[:, k]
    return scores


# --- Randomisation ------------------------------------------------------------------

CASCADE = ("output layer", "hidden layer 2", "hidden layer 1", "embeddings")
"""Order in which the MLP's parameters are re-initialised, from the output
backwards (a "cascading" randomisation): step k randomises the first k."""


def randomised_mlp(model, steps: int, seed: int):
    """A deep copy of `model` (an `MLPRawModel`) with the first `steps` groups
    of `CASCADE` re-initialised by PyTorch's default initialisation."""
    import torch
    from torch import nn

    clone = copy.deepcopy(model)
    linears = [module for module in clone.net.mlp if isinstance(module, nn.Linear)]
    groups = {
        "output layer": [linears[-1]],
        "hidden layer 2": [linears[1]],
        "hidden layer 1": [linears[0]],
        "embeddings": list(clone.net.embeddings),
    }
    torch.manual_seed(seed)
    for name in CASCADE[:steps]:
        for module in groups[name]:
            module.reset_parameters()
    clone.net.eval()
    return clone


# --- Stability ------------------------------------------------------------------------


def numeric_nudges(
    codes: np.ndarray, positions: dict[str, int], bounds: dict[str, tuple[float, float]], step: float = 1.0
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """For each numeric feature, `+step` and `-step` copies of `codes`, kept
    within the observed range. Returns `(label, perturbed codes, valid
    mask)`; a row is invalid where the nudge would leave the range."""
    nudges = []
    for feature, j in positions.items():
        low, high = bounds[feature]
        for delta in (-step, step):
            moved = codes.copy()
            moved[:, j] = codes[:, j] + delta
            valid = (moved[:, j] >= low) & (moved[:, j] <= high)
            nudges.append((f"{feature} {delta:+g}", moved, valid))
    return nudges


def relative_change(original: np.ndarray, perturbed: np.ndarray) -> np.ndarray:
    """Per row: ||phi(x') - phi(x)||_1 / ||phi(x)||_1."""
    size = np.abs(original).sum(axis=1)
    return np.abs(perturbed - original).sum(axis=1) / np.where(size > 0, size, np.nan)
