import numpy as np
import pandas as pd

from fin_inclusion.interpretability.shap_analysis import (
    compute_shap_explanation,
    global_importance,
    plot_dependence,
    plot_global_importance,
)
from fin_inclusion.models.xgboost_model import XGBoostModel


def _fitted_model_and_data(n: int = 60):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "strong_signal": rng.normal(size=n),
            "noise": rng.normal(size=n),
        }
    )
    y = pd.Series(np.where(X["strong_signal"] > 0, "Yes", "No"))
    model = XGBoostModel(seed=0).fit(X, y)
    return model, X


def test_compute_shap_explanation_shape_matches_input():
    model, X = _fitted_model_and_data()

    explanation = compute_shap_explanation(model, X, max_samples=None)

    assert explanation.values.shape == (len(X), X.shape[1])


def test_max_samples_subsamples():
    model, X = _fitted_model_and_data(n=60)

    explanation = compute_shap_explanation(model, X, max_samples=10, seed=0)

    assert explanation.values.shape[0] == 10


def test_global_importance_ranks_strong_signal_above_noise():
    model, X = _fitted_model_and_data()
    explanation = compute_shap_explanation(model, X, max_samples=None)

    importance = global_importance(explanation)

    assert list(importance.index)[0] == "strong_signal"
    assert importance["strong_signal"] > importance["noise"]


def test_plot_global_importance_writes_file(tmp_path):
    model, X = _fitted_model_and_data()
    explanation = compute_shap_explanation(model, X, max_samples=None)
    output_path = tmp_path / "importance.png"

    plot_global_importance(explanation, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_dependence_writes_file(tmp_path):
    model, X = _fitted_model_and_data()
    explanation = compute_shap_explanation(model, X, max_samples=None)
    output_path = tmp_path / "dependence.png"

    plot_dependence(explanation, "strong_signal", output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
