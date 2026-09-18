import numpy as np
import pytest

from fin_inclusion.evaluation.threshold_selection import select_f1_maximizing_threshold


def test_finds_perfect_threshold_under_perfect_separation():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

    threshold, f1 = select_f1_maximizing_threshold(y_true, y_score)

    assert f1 == pytest.approx(1.0)
    assert 0.3 < threshold <= 0.7


def test_finds_better_threshold_than_naive_default_under_skew():
    rng = np.random.default_rng(0)
    y_true = np.concatenate([np.zeros(90), np.ones(10)]).astype(int)
    y_score = np.concatenate([rng.uniform(0, 0.6, 90), rng.uniform(0.3, 0.9, 10)])

    threshold, f1 = select_f1_maximizing_threshold(y_true, y_score)

    from sklearn.metrics import f1_score

    default_f1 = f1_score(y_true, (y_score >= 0.5).astype(int), zero_division=0)
    assert f1 >= default_f1


def test_returns_threshold_and_f1_as_floats():
    y_true = np.array([0, 1, 0, 1])
    y_score = np.array([0.2, 0.6, 0.4, 0.8])

    threshold, f1 = select_f1_maximizing_threshold(y_true, y_score)

    assert isinstance(threshold, float)
    assert isinstance(f1, float)
