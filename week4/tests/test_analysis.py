"""Tests of the Phase 4 statistics: the bootstrap PR-AUC and the stratified resampling."""
import numpy as np
from sklearn.metrics import average_precision_score

from analysis import bootstrap_counts, holm, weighted_ap


def test_weighted_ap_equals_sklearn_with_ties_and_weights():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    scores = np.round(rng.random(500), 2)  # Rounded, so many scores tie.
    counts = rng.integers(0, 4, (3, 500))
    expected = [average_precision_score(y, scores, sample_weight=c) for c in counts]
    assert np.allclose(weighted_ap(y, scores, counts), expected)
    assert np.isclose(weighted_ap(y, scores, np.ones((1, 500)))[0], average_precision_score(y, scores))


def test_bootstrap_keeps_every_stratum_size_in_every_replicate():
    strata = np.array(["a"] * 30 + ["b"] * 70)
    counts = bootstrap_counts(strata, replicates=50, seed=0)
    assert (counts[:, :30].sum(axis=1) == 30).all()
    assert (counts[:, 30:].sum(axis=1) == 70).all()
    assert np.array_equal(counts, bootstrap_counts(strata, replicates=50, seed=0))


def test_holm_matches_hand_calculation():
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
