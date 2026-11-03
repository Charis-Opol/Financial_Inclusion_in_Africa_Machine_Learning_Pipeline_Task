"""H2 leakage tests, H1 and row-limit context tests. The first is the one the brief requires."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from tabpfn_tools import (
    balanced_context_lists,
    embed_new_rows,
    make_v2,
    out_of_fold_embeddings,
    stratified_context_lists,
)


class RecordingModel:
    """Stands in for TabPFN and records which rows were in each context and which were embedded."""

    log: list[dict] = []

    def __init__(self, lists):
        self.lists = lists

    def fit(self, X, y):
        self.context_ids = set(X[:, 0].astype(int).tolist())
        return self

    def get_embeddings(self, X, data_source="test"):
        RecordingModel.log.append(
            {"data_source": data_source, "context": self.context_ids, "embedded": X[:, 0].astype(int).tolist()}
        )
        return np.repeat(X[:, :1][None, ...], 2, axis=0)


def _rows(n=200):
    rng = np.random.default_rng(0)
    # Column 0 is a unique row id, so the recording model can identify every row.
    X = np.column_stack([np.arange(n), rng.normal(size=n)]).astype(np.float32)
    y = (np.arange(n) % 4 == 0).astype(int)
    strata = pd.Series(np.where(np.arange(n) % 2 == 0, "A_", "B_")) + pd.Series(y).astype(str)
    return X, y, strata


def test_no_row_is_ever_embedded_by_a_model_with_it_in_context():
    """Required. Checks the actual row sets passed to the model, for every internal fold."""
    RecordingModel.log = []
    X, y, strata = _rows()
    out_of_fold_embeddings(X, y, strata, RecordingModel, n_estimators=2, max_rows=1000, seed=0, n_folds=5)
    assert len(RecordingModel.log) == 5
    embedded_rows = []
    for call in RecordingModel.log:
        assert call["data_source"] == "test"  # "train" tokens leak labels (results/feasibility.md).
        assert not call["context"] & set(call["embedded"])
        embedded_rows.extend(call["embedded"])
    assert sorted(embedded_rows) == list(range(len(X)))  # Each row embedded exactly once.


def test_validation_rows_are_never_in_the_context():
    RecordingModel.log = []
    X, y, strata = _rows()
    train, new = X[:150], X[150:]
    embed_new_rows(train, y[:150], new, strata.iloc[:150], RecordingModel, n_estimators=2, max_rows=1000, seed=0)
    call = RecordingModel.log[0]
    assert call["data_source"] == "test"
    assert not call["context"] & set(call["embedded"])


def _head_auc(embeddings, y):
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_predict(LogisticRegression(max_iter=2000), embeddings, y, cv=folds, method="predict_proba")[:, 1]
    return roc_auc_score(y, scores)


@pytest.mark.slow
def test_real_tabpfn_embeddings_show_no_label_leakage_on_noise():
    """Required, on the real model and the project's own function. Noise labels: skill means leakage."""
    aucs = []
    for seed in range(4):
        rng = np.random.default_rng(200 + seed)
        X = rng.normal(size=(400, 10)).astype(np.float32)
        y = rng.integers(0, 2, size=400)
        strata = pd.Series(y).astype(str)

        def make_model(lists):
            return make_v2(device="cpu", seed=seed, n_estimators=1, context_lists=lists)

        embeddings = out_of_fold_embeddings(X, y, strata, make_model, n_estimators=1, max_rows=1000, seed=seed)
        aucs.append(_head_auc(embeddings, y))
    # Train tokens scored 1.000 and in-context test tokens 0.955 in Phase 1. Chance is 0.5.
    assert np.mean(aucs) < 0.60, aucs


def test_stratified_lists_keep_each_stratum_share_within_the_limit():
    strata = pd.Series(["K_Yes"] * 500 + ["K_No"] * 1500 + ["U_Yes"] * 50 + ["U_No"] * 950)
    lists = stratified_context_lists(strata, n_lists=8, max_rows=1000, seed=0)
    assert len(lists) == 8
    for rows in lists:
        assert len(rows) <= 1000
        assert len(set(rows.tolist())) == len(rows)
        counts = strata.iloc[rows].value_counts(normalize=True)
        for name, share in strata.value_counts(normalize=True).items():
            assert abs(counts[name] - share) < 0.01


def test_stratified_lists_differ_and_use_every_row_across_the_ensemble():
    strata = pd.Series(["a"] * 3000 + ["b"] * 3000)
    lists = stratified_context_lists(strata, n_lists=8, max_rows=2000, seed=0)
    assert not np.array_equal(lists[0], lists[1])
    assert set(np.concatenate(lists).tolist()) == set(range(6000))


def test_stratified_lists_are_reproducible():
    strata = pd.Series(["a"] * 300 + ["b"] * 700)
    first, second = (stratified_context_lists(strata, 3, 500, seed=4) for _ in range(2))
    assert all(np.array_equal(a, b) for a, b in zip(first, second))


def test_balanced_lists_give_equal_rows_per_country_and_keep_class_ratio():
    country = pd.Series(["Kenya"] * 5000 + ["Uganda"] * 1000)
    y = np.concatenate(
        [(np.arange(5000) % 4 == 0).astype(int), (np.arange(1000) % 10 == 0).astype(int)]
    )
    lists = balanced_context_lists(country, y, n_lists=4, max_rows=10_000, seed=0)
    for rows in lists:
        kenya, uganda = rows[rows < 5000], rows[rows >= 5000]
        assert abs(len(kenya) - len(uganda)) <= 2
        assert abs(y[kenya].mean() - 0.25) < 0.01
        assert abs(y[uganda].mean() - 0.10) < 0.01
    assert not np.array_equal(lists[0][lists[0] < 5000], lists[1][lists[1] < 5000])


def test_balanced_lists_respect_the_row_limit():
    country = pd.Series(["A"] * 9000 + ["B"] * 9000)
    y = (np.arange(18000) % 5 == 0).astype(int)
    for rows in balanced_context_lists(country, y, n_lists=2, max_rows=10_000, seed=0):
        assert len(rows) <= 10_000
