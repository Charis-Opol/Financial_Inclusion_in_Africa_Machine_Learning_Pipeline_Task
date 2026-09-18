import numpy as np
import pandas as pd
import pytest

from fin_inclusion.models.base_model import BaseModel
from fin_inclusion.models.logistic_regression import LogisticRegressionModel
from fin_inclusion.models.pytorch_mlp import PyTorchMLP
from fin_inclusion.models.xgboost_model import XGBoostModel


def _tabular_X(n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({"feat_a": rng.normal(size=n), "feat_b": rng.integers(0, 2, size=n)})


def _y(n: int = 40) -> pd.Series:
    rng = np.random.default_rng(1)
    return pd.Series(rng.choice(["Yes", "No"], size=n, p=[0.3, 0.7]))


def _pytorch_X(n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    return pd.DataFrame(
        {
            "country": rng.integers(1, 4, size=n),
            "job_type": rng.integers(1, 6, size=n),
            "age_of_respondent": rng.normal(size=n),
            "household_size": rng.normal(size=n),
        }
    )


def test_base_model_is_abstract():
    with pytest.raises(TypeError):
        BaseModel()


class TestLogisticRegressionModel:
    def test_predict_proba_shape_and_range(self):
        X, y = _tabular_X(), _y()
        model = LogisticRegressionModel(seed=0).fit(X, y)

        scores = model.predict_proba(X)

        assert scores.shape == (len(X),)
        assert (scores >= 0).all() and (scores <= 1).all()

    def test_accepts_sample_weight(self):
        X, y = _tabular_X(), _y()
        weights = np.ones(len(X))

        LogisticRegressionModel(seed=0).fit(X, y, sample_weight=weights)

    def test_get_params_returns_dict(self):
        model = LogisticRegressionModel(seed=0)
        assert isinstance(model.get_params(), dict)

    def test_save_load_round_trip(self, tmp_path):
        X, y = _tabular_X(), _y()
        model = LogisticRegressionModel(seed=0).fit(X, y)
        path = tmp_path / "lr.pkl"

        model.save(path)
        loaded = LogisticRegressionModel.load(path)

        np.testing.assert_allclose(loaded.predict_proba(X), model.predict_proba(X))


class TestXGBoostModel:
    def test_predict_proba_shape_and_range(self):
        X, y = _tabular_X(), _y()
        model = XGBoostModel(seed=0).fit(X, y)

        scores = model.predict_proba(X)

        assert scores.shape == (len(X),)
        assert (scores >= 0).all() and (scores <= 1).all()

    def test_accepts_sample_weight(self):
        X, y = _tabular_X(), _y()
        weights = np.ones(len(X))

        XGBoostModel(seed=0).fit(X, y, sample_weight=weights)

    def test_save_load_round_trip(self, tmp_path):
        X, y = _tabular_X(), _y()
        model = XGBoostModel(seed=0).fit(X, y)
        path = tmp_path / "xgb.json"

        model.save(path)
        loaded = XGBoostModel.load(path)

        np.testing.assert_allclose(loaded.predict_proba(X), model.predict_proba(X))


class TestPyTorchMLP:
    def _model(self, **overrides) -> PyTorchMLP:
        params = dict(
            categorical_columns=["country", "job_type"],
            vocab_sizes={"country": 4, "job_type": 6},
            numeric_columns=["age_of_respondent", "household_size"],
            hidden_dims=(8, 4),
            embedding_dim=3,
            epochs=2,
            batch_size=8,
            seed=0,
        )
        params.update(overrides)
        return PyTorchMLP(**params)

    def test_predict_proba_shape_and_range(self):
        X, y = _pytorch_X(), _y()
        model = self._model().fit(X, y)

        scores = model.predict_proba(X)

        assert scores.shape == (len(X),)
        assert (scores >= 0).all() and (scores <= 1).all()

    def test_accepts_sample_weight(self):
        X, y = _pytorch_X(), _y()
        weights = np.ones(len(X), dtype="float32")

        self._model().fit(X, y, sample_weight=weights)

    def test_hidden_dims_controls_depth(self):
        two_layer = self._model(hidden_dims=(8, 4)).get_params()
        three_layer = self._model(hidden_dims=(8, 4, 2)).get_params()

        assert len(two_layer["hidden_dims"]) == 2
        assert len(three_layer["hidden_dims"]) == 3

    def test_save_load_round_trip(self, tmp_path):
        X, y = _pytorch_X(), _y()
        model = self._model().fit(X, y)
        path = tmp_path / "mlp.pt"

        model.save(path)
        loaded = PyTorchMLP.load(path)

        np.testing.assert_allclose(loaded.predict_proba(X), model.predict_proba(X), atol=1e-6)

    def test_raises_if_predict_before_fit(self):
        with pytest.raises(AssertionError):
            self._model().predict_proba(_pytorch_X())
