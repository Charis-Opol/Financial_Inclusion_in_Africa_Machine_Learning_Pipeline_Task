import numpy as np
import pandas as pd
import pytest

from fin_inclusion.imbalance.strategies import (
    BaseImbalanceStrategy,
    ClassWeightStrategy,
    NoImbalanceStrategy,
    SMOTEStrategy,
)


def test_base_imbalance_strategy_is_abstract():
    with pytest.raises(TypeError):
        BaseImbalanceStrategy()


def test_class_weight_strategy_leaves_X_and_y_unchanged():
    X = pd.DataFrame({"a": [1, 2, 3, 4]})
    y = pd.Series(["Yes", "No", "No", "No"])

    X_out, y_out, _ = ClassWeightStrategy().apply(X, y)

    pd.testing.assert_frame_equal(X_out, X)
    pd.testing.assert_series_equal(y_out, y)


def test_class_weight_strategy_weights_minority_class_higher():
    y = pd.Series(["Yes", "Yes", "No", "No", "No", "No", "No", "No", "No", "No"])

    _, _, weights = ClassWeightStrategy().apply(pd.DataFrame(index=y.index), y)

    yes_weight = weights[y == "Yes"][0]
    no_weight = weights[y == "No"][0]
    assert yes_weight == pytest.approx(10 / (2 * 2))
    assert no_weight == pytest.approx(10 / (2 * 8))
    assert yes_weight > no_weight


def test_no_imbalance_strategy_leaves_everything_unchanged():
    X = pd.DataFrame({"a": [1, 2, 3]})
    y = pd.Series(["Yes", "No", "No"])

    X_out, y_out, sample_weight = NoImbalanceStrategy().apply(X, y)

    pd.testing.assert_frame_equal(X_out, X)
    pd.testing.assert_series_equal(y_out, y)
    assert sample_weight is None


def _imbalanced_frame(n_majority: int = 40, n_minority: int = 10):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "num": rng.normal(size=n_majority + n_minority),
            "onehot_a": rng.integers(0, 2, size=n_majority + n_minority),
        }
    )
    y = pd.Series(["No"] * n_majority + ["Yes"] * n_minority)
    return X, y


def test_smote_strategy_balances_the_classes():
    X, y = _imbalanced_frame()

    X_res, y_res, sample_weight = SMOTEStrategy(seed=0).apply(X, y)

    counts = y_res.value_counts()
    assert counts["Yes"] == counts["No"]
    assert sample_weight is None


def test_smote_strategy_only_adds_rows_does_not_remove_originals():
    X, y = _imbalanced_frame()

    X_res, y_res, _ = SMOTEStrategy(seed=0).apply(X, y)

    assert len(X_res) >= len(X)
    assert len(X_res) == len(y_res)


def test_smote_strategy_preserves_columns():
    X, y = _imbalanced_frame()

    X_res, _, _ = SMOTEStrategy(seed=0).apply(X, y)

    assert list(X_res.columns) == list(X.columns)


def test_smote_strategy_is_deterministic_given_seed():
    X, y = _imbalanced_frame()

    first_X, first_y, _ = SMOTEStrategy(seed=3).apply(X, y)
    second_X, second_y, _ = SMOTEStrategy(seed=3).apply(X, y)

    pd.testing.assert_frame_equal(first_X, second_X)
    pd.testing.assert_series_equal(first_y, second_y)
