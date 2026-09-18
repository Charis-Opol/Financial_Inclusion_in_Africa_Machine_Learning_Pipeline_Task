import numpy as np
import pandas as pd

from fin_inclusion.evaluation.cv_runner import run_stratified_cv
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
