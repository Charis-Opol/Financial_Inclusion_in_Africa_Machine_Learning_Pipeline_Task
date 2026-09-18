import numpy as np
import pandas as pd

from fin_inclusion.tuning.optuna_search import tune_pytorch_mlp, tune_xgboost

XGB_PARAM_KEYS = {
    "max_depth",
    "learning_rate",
    "n_estimators",
    "min_child_weight",
    "subsample",
    "colsample_bytree",
}


def _tabular_data(n_per_group: int = 12):
    rng = np.random.default_rng(0)
    countries, targets = [], []
    for country in ["Kenya", "Uganda"]:
        for target in ["Yes", "No"]:
            countries += [country] * n_per_group
            targets += [target] * n_per_group
    n = len(countries)
    X = pd.DataFrame({"feat_a": rng.normal(size=n), "feat_b": rng.integers(0, 2, size=n)})
    y = pd.Series(targets)
    country = pd.Series(countries)
    return X, y, country


def _pytorch_data(n_per_group: int = 12):
    rng = np.random.default_rng(1)
    countries, targets = [], []
    for country in ["Kenya", "Uganda"]:
        for target in ["Yes", "No"]:
            countries += [country] * n_per_group
            targets += [target] * n_per_group
    n = len(countries)
    X = pd.DataFrame(
        {
            "country": rng.integers(1, 3, size=n),
            "job_type": rng.integers(1, 5, size=n),
            "age_of_respondent": rng.normal(size=n),
        }
    )
    y = pd.Series(targets)
    country = pd.Series(countries)
    return X, y, country


class TestTuneXGBoost:
    def test_class_weight_arm_includes_scale_pos_weight(self):
        X, y, country = _tabular_data()

        best_params = tune_xgboost(
            X, y, country, imbalance_strategy_name="class_weight",
            n_inner_folds=2, n_trials=2, seed=0,
        )

        assert XGB_PARAM_KEYS.issubset(best_params)
        assert "scale_pos_weight" in best_params

    def test_smote_arm_excludes_scale_pos_weight(self):
        X, y, country = _tabular_data()

        best_params = tune_xgboost(
            X, y, country, imbalance_strategy_name="smote",
            n_inner_folds=2, n_trials=2, seed=0,
        )

        assert XGB_PARAM_KEYS.issubset(best_params)
        assert "scale_pos_weight" not in best_params


class TestTunePyTorchMLP:
    def test_two_layer_search_returns_two_hidden_dims(self):
        X, y, country = _pytorch_data()

        best_params = tune_pytorch_mlp(
            X, y, country,
            categorical_columns=["country", "job_type"],
            vocab_sizes={"country": 3, "job_type": 5},
            numeric_columns=["age_of_respondent"],
            imbalance_strategy_name="class_weight",
            n_layers=2, n_inner_folds=2, n_trials=2, tuning_epochs=1, seed=0,
        )

        assert len(best_params["hidden_dims"]) == 2
        assert {"embedding_dim", "dropout", "lr", "weight_decay"}.issubset(best_params)

    def test_three_layer_search_returns_three_hidden_dims(self):
        X, y, country = _pytorch_data()

        best_params = tune_pytorch_mlp(
            X, y, country,
            categorical_columns=["country", "job_type"],
            vocab_sizes={"country": 3, "job_type": 5},
            numeric_columns=["age_of_respondent"],
            imbalance_strategy_name="smote",
            n_layers=3, n_inner_folds=2, n_trials=2, tuning_epochs=1, seed=0,
        )

        assert len(best_params["hidden_dims"]) == 3
