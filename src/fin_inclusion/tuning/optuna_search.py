"""Inner-loop hyperparameter search per model family (README §6, Phase 4.1/4.4).

Each `tune_*` function is a `cv_runner.run_nested_cv`-compatible `tune_fn`:
called once per outer fold with that fold's *training* split only, it runs
its own inner stratified CV (reusing `run_stratified_cv`) inside an Optuna
study and returns the best hyperparameters. The outer validation fold this
function is never shown never enters this module at all -- `run_nested_cv`
only ever calls it with the outer-training split.
"""
from __future__ import annotations

import optuna
import pandas as pd

from fin_inclusion.evaluation.cv_runner import run_stratified_cv
from fin_inclusion.evaluation.reporting import summarize_metrics
from fin_inclusion.imbalance.strategies import ClassWeightStrategy, NoImbalanceStrategy, SMOTEStrategy
from fin_inclusion.models.pytorch_mlp import PyTorchMLP
from fin_inclusion.models.xgboost_model import XGBoostModel

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _mean_pr_auc(X, y, country, model_factory, imbalance_strategy, n_inner_folds, seed) -> float:
    results = run_stratified_cv(
        X=X,
        y=y,
        country=country,
        model_factory=model_factory,
        imbalance_strategy=imbalance_strategy,
        n_folds=n_inner_folds,
        seed=seed,
    )
    return summarize_metrics(results, metric_names=("pr_auc",))["pr_auc"][0]


def tune_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    country: pd.Series,
    imbalance_strategy_name: str,
    n_inner_folds: int = 3,
    n_trials: int = 15,
    seed: int = 42,
) -> dict:
    """`imbalance_strategy_name`: `"class_weight"` (tunes `scale_pos_weight`,
    XGBoost's native lever -- see `NoImbalanceStrategy`) or `"smote"`."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }
        if imbalance_strategy_name == "class_weight":
            params["scale_pos_weight"] = trial.suggest_float("scale_pos_weight", 1.0, 10.0, log=True)
            strategy = NoImbalanceStrategy()
        else:
            strategy = SMOTEStrategy(seed=seed)

        return _mean_pr_auc(
            X, y, country,
            model_factory=lambda: XGBoostModel(seed=seed, **params),
            imbalance_strategy=strategy,
            n_inner_folds=n_inner_folds,
            seed=seed,
        )

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return dict(study.best_params)


def tune_pytorch_mlp(
    X: pd.DataFrame,
    y: pd.Series,
    country: pd.Series,
    categorical_columns: list[str],
    vocab_sizes: dict[str, int],
    numeric_columns: list[str],
    imbalance_strategy_name: str,
    n_layers: int = 2,
    n_inner_folds: int = 3,
    n_trials: int = 15,
    tuning_epochs: int = 8,
    seed: int = 42,
) -> dict:
    """`n_layers` sets the MLP depth being tuned (2 for Phase 4's main grid,
    3 for the depth-ablation arm) -- same search space, just more hidden-dim
    dimensions sampled. `imbalance_strategy_name`: `"class_weight"` or
    `"smote"`."""

    def objective(trial: optuna.Trial) -> float:
        hidden_dims = tuple(
            trial.suggest_int(f"hidden_dim_{i + 1}", 8, 128) for i in range(n_layers)
        )
        params = dict(
            hidden_dims=hidden_dims,
            embedding_dim=trial.suggest_int("embedding_dim", 4, 16),
            dropout=trial.suggest_float("dropout", 0.0, 0.5),
            lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        )
        strategy = (
            ClassWeightStrategy() if imbalance_strategy_name == "class_weight" else SMOTEStrategy(seed=seed)
        )

        return _mean_pr_auc(
            X, y, country,
            model_factory=lambda: PyTorchMLP(
                categorical_columns=categorical_columns,
                vocab_sizes=vocab_sizes,
                numeric_columns=numeric_columns,
                epochs=tuning_epochs,
                seed=seed,
                **params,
            ),
            imbalance_strategy=strategy,
            n_inner_folds=n_inner_folds,
            seed=seed,
        )

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)

    best_params = dict(study.best_params)
    best_params["hidden_dims"] = tuple(best_params.pop(f"hidden_dim_{i + 1}") for i in range(n_layers))
    return best_params
