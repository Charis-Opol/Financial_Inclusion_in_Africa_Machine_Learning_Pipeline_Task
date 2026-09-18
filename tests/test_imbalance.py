import pandas as pd
import pytest

from fin_inclusion.imbalance.strategies import BaseImbalanceStrategy, ClassWeightStrategy


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
