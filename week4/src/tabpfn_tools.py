"""TabPFN v2 helpers: building the model, choosing contexts, and leak-free embeddings.

Everything here rests on what Phase 1 verified on the installed version (results/feasibility.md):
  - A plain TabPFNClassifier() is v3.5, so v2 is requested explicitly.
  - Each estimator's context can be set exactly with SUBSAMPLE_SAMPLES index lists.
  - data_source="train" embeddings, and "test" embeddings of rows also in the context, leak labels.
    Only rows absent from the context may be embedded.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

V2_ROW_LIMIT = 10_000
DEFAULT_N_ESTIMATORS = 8  # The installed default (DEFAULT_N_ESTIMATORS in tabpfn/preprocessing/ensemble.py).


def make_v2(
    *,
    device: str,
    seed: int,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    context_lists: list[np.ndarray] | None = None,
    categorical_indices: list[int] | None = None,
) -> TabPFNClassifier:
    """TabPFN v2 with default settings. `ignore_pretraining_limits` is needed for CPU and for
    passing a full fold to fit; each context still stays within the row limit (D1.3)."""
    inference_config = {"SUBSAMPLE_SAMPLES": context_lists} if context_lists is not None else None
    return TabPFNClassifier.create_default_for_version(
        ModelVersion.V2,
        device=device,
        random_state=seed,
        n_estimators=n_estimators,
        ignore_pretraining_limits=True,
        categorical_features_indices=categorical_indices,
        inference_config=inference_config,
    )


def _deal(members: np.ndarray, take: int, n_lists: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Deals `take` rows per list from one shuffled stratum, wrapping round, so lists differ
    and, when take * n_lists >= len(members), every member is used at least once."""
    shuffled = rng.permutation(members)
    return [
        np.take(shuffled, np.arange(i * take, i * take + take), mode="wrap") if take else np.array([], dtype=int)
        for i in range(n_lists)
    ]


def stratified_context_lists(
    strata: pd.Series, n_lists: int, max_rows: int = V2_ROW_LIMIT, seed: int = 0
) -> list[np.ndarray]:
    """D1.2: one row-position array per estimator, each keeping every stratum's share of the rows."""
    rng = np.random.default_rng(seed)
    fraction = min(1.0, max_rows / len(strata))
    chosen: list[list[np.ndarray]] = [[] for _ in range(n_lists)]
    for _, members in sorted(strata.reset_index(drop=True).groupby(strata.to_numpy()).indices.items()):
        take = int(np.floor(len(members) * fraction))
        for estimator, rows in enumerate(_deal(members, take, n_lists, rng)):
            chosen[estimator].append(rows)
    return [np.sort(np.concatenate(rows)) for rows in chosen]


def balanced_context_lists(
    country: pd.Series, y_binary: np.ndarray, n_lists: int, max_rows: int = V2_ROW_LIMIT, seed: int = 0
) -> list[np.ndarray]:
    """H1 (D1.7): every country contributes the same number of rows to each estimator's context.

    The number is set by the smallest country, capped so the context fits the row limit. Within
    a country, its own class ratio is kept, so only the country mix changes, not within-country
    class balance. Group balance is what H1 tests.
    """
    rng = np.random.default_rng(seed)
    countries = country.to_numpy()
    names = sorted(set(countries))
    per_country = min(min(int((countries == name).sum()) for name in names), max_rows // len(names))
    chosen: list[list[np.ndarray]] = [[] for _ in range(n_lists)]
    for name in names:
        in_country = np.flatnonzero(countries == name)
        for label in (0, 1):
            members = in_country[y_binary[in_country] == label]
            take = int(np.floor(per_country * len(members) / len(in_country)))
            for estimator, rows in enumerate(_deal(members, take, n_lists, rng)):
                chosen[estimator].append(rows)
    return [np.sort(np.concatenate(rows)) for rows in chosen]


def predict_scores(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    *,
    context_lists: list[np.ndarray],
    categorical_indices: list[int],
    device: str,
    seed: int,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
) -> np.ndarray:
    """Positive-class probability for validation rows. Validation rows are never in the context."""
    model = make_v2(
        device=device,
        seed=seed,
        n_estimators=n_estimators,
        context_lists=context_lists,
        categorical_indices=categorical_indices,
    )
    model.fit(X_train, y_train)
    return model.predict_proba(X_val)[:, 1]


def _mean_over_estimators(embeddings: np.ndarray) -> np.ndarray:
    """(n_estimators, n_rows, dim) to (n_rows, dim), as the get_embeddings documentation shows."""
    return embeddings.mean(axis=0) if embeddings.ndim == 3 else embeddings


ModelFactory = Callable[[list[np.ndarray] | None], TabPFNClassifier]


def embed_new_rows(
    X_context: np.ndarray, y_context: np.ndarray, X_new: np.ndarray, strata: pd.Series,
    make_model: ModelFactory, *, n_estimators: int, max_rows: int, seed: int,
) -> np.ndarray:
    """Embeds rows that are NOT in the context. This is the only call that produces embeddings."""
    lists = stratified_context_lists(strata, n_estimators, max_rows=max_rows, seed=seed)
    model = make_model(lists).fit(X_context, y_context)
    return _mean_over_estimators(model.get_embeddings(X_new, data_source="test"))


def out_of_fold_embeddings(
    X: np.ndarray, y: np.ndarray, strata: pd.Series, make_model: ModelFactory,
    *, n_estimators: int, max_rows: int, seed: int, n_folds: int = 5,
) -> np.ndarray:
    """Embeddings for training rows, each made by a model whose context excludes that row (D1.5)."""
    strata = strata.reset_index(drop=True)
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    result: np.ndarray | None = None
    for fit_idx, held_idx in splitter.split(X, strata):
        embedded = embed_new_rows(
            X[fit_idx], y[fit_idx], X[held_idx], strata.iloc[fit_idx], make_model,
            n_estimators=n_estimators, max_rows=max_rows, seed=seed,
        )
        if result is None:
            result = np.zeros((len(X), embedded.shape[1]), dtype=np.float32)
        result[held_idx] = embedded
    return result
