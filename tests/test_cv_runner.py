import numpy as np
import pandas as pd

from fin_inclusion.evaluation.cv_runner import run_nested_cv, run_stratified_cv
from fin_inclusion.imbalance.strategies import ClassWeightStrategy
from fin_inclusion.models.logistic_regression import LogisticRegressionModel

N_FOLDS = 3
METRIC_KEYS = {"pr_auc", "roc_auc", "f1", "precision", "recall"}


def _synthetic_data(n_per_group: int = 12):
    rng = np.random.default_rng(0)
    countries, targets = [], []
    for country in ["Kenya", "Uganda"]:
        for target in ["Yes", "No"]:
            countries += [country] * n_per_group
            targets += [target] * n_per_group
    n = len(countries)
    X = pd.DataFrame({"feat": rng.normal(size=n)})
    y = pd.Series(targets)
    country = pd.Series(countries)
    return X, y, country


def _run(seed: int = 42):
    X, y, country = _synthetic_data()
    return run_stratified_cv(
        X=X,
        y=y,
        country=country,
        model_factory=lambda: LogisticRegressionModel(seed=0),
        imbalance_strategy=ClassWeightStrategy(),
        n_folds=N_FOLDS,
        seed=seed,
    )


def test_returns_one_result_per_fold():
    results = _run()
    assert len(results) == N_FOLDS
    assert [r.fold for r in results] == list(range(N_FOLDS))


def test_fold_metrics_have_expected_keys():
    results = _run()
    for r in results:
        assert set(r.metrics) == METRIC_KEYS


def test_country_metrics_cover_every_country():
    results = _run()
    for r in results:
        assert set(r.country_metrics) == {"Kenya", "Uganda"}
        for country_metrics in r.country_metrics.values():
            assert set(country_metrics) == METRIC_KEYS


def test_same_seed_is_reproducible():
    first = _run(seed=7)
    second = _run(seed=7)

    assert [r.metrics for r in first] == [r.metrics for r in second]


class TestNestedCV:
    N_OUTER_FOLDS = 3

    def _tune_fn(self, X_train, y_train, country_train):
        # Trivial stand-in for an Optuna search: no hyperparameters to tune,
        # just confirms the callback receives the outer-training split only.
        assert len(X_train) < len(_synthetic_data(20)[0])
        return {}

    def _build_model(self, params):
        return LogisticRegressionModel(seed=0, **params)

    def test_returns_one_result_per_outer_fold(self):
        X, y, country = _synthetic_data(n_per_group=20)

        results = run_nested_cv(
            X=X,
            y=y,
            country=country,
            tune_fn=self._tune_fn,
            build_model=self._build_model,
            imbalance_strategy_factory=ClassWeightStrategy,
            n_outer_folds=self.N_OUTER_FOLDS,
            seed=42,
        )

        assert len(results) == self.N_OUTER_FOLDS
        for r in results:
            assert set(r.metrics) == METRIC_KEYS
            assert isinstance(r.best_params, dict)
            assert isinstance(r.threshold, float)
            assert set(r.country_metrics) == {"Kenya", "Uganda"}

    def test_outer_validation_fold_never_reaches_tune_fn(self):
        X, y, country = _synthetic_data(n_per_group=20)
        val_indices_seen_by_tune_fn = []

        def spying_tune_fn(X_train, y_train, country_train):
            val_indices_seen_by_tune_fn.append(set(X_train.index))
            return {}

        skf_results = run_nested_cv(
            X=X,
            y=y,
            country=country,
            tune_fn=spying_tune_fn,
            build_model=self._build_model,
            imbalance_strategy_factory=ClassWeightStrategy,
            n_outer_folds=self.N_OUTER_FOLDS,
            seed=42,
        )

        assert len(val_indices_seen_by_tune_fn) == self.N_OUTER_FOLDS
        assert len(skf_results) == self.N_OUTER_FOLDS
