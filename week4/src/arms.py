"""Every model variant in the study, each described as an `Arm` for the shared harness.

An arm says how to prepare a fold, what to tune, and how to fit and predict. It knows nothing
about folds, seeds across runs, or saving results, so all arms are treated the same way.

Search spaces and trial budgets match the baseline (fin_inclusion/tuning/optuna_search.py).
"""
from __future__ import annotations

import json

import numpy as np
import optuna
import pandas as pd

from fin_inclusion.models.pytorch_mlp import PyTorchMLP
from fin_inclusion.models.xgboost_model import XGBoostModel
from group_dro import GroupDROMLP
from harness import (
    PT_NUMERIC_COLUMNS,
    Prepared,
    RunConfig,
    class_weights,
    country_label_key,
    country_weights,
    fold_paths,
    group_ids,
    prepare_matrix,
    prepare_mlp,
    prepare_xgb,
)
from hybrids import fit_head, fold_embeddings, prepare_embeddings
from tabpfn_tools import balanced_context_lists, predict_scores, stratified_context_lists

STEP_SIZE_RANGE = (1e-3, 1.0)  # Log scale. Includes the 0.01 default reported in koh2021wilds (p.66).
ADJUSTMENT_CHOICES = [0.0, 1.0, 2.0, 4.0]


def _dro_history_extra(history) -> dict:
    return {"q_by_epoch": history.q_by_epoch, "group_loss_by_epoch": history.group_loss_by_epoch}


class Arm:
    name = ""
    selection = "pooled"  # Inner-loop criterion: "pooled" or "worst_country". None if nothing is tuned.
    params_from: str | None = None  # Reuse this parent arm's tuned parameters instead of tuning.
    overrides: dict = {}

    def trials(self, config: RunConfig) -> int:
        return 0

    def suggest(self, trial: optuna.Trial) -> dict:
        return {}

    def finalize_params(self, params: dict) -> dict:
        return params

    def fold_context(self, train, val, use_country, config, seed, design, fold_name) -> dict:
        return {}

    def prepare(self, train: pd.DataFrame, val: pd.DataFrame, use_country: bool, context: dict) -> Prepared:
        raise NotImplementedError

    def fit_predict(self, params, prepared, config, seed, tuning) -> tuple[np.ndarray, dict]:
        raise NotImplementedError

    def inherited_params(self, config: RunConfig, design: str, seed: int, fold_name: str) -> dict:
        """A sweep arm takes its parent's tuned parameters for the same fold, changing only `overrides`."""
        _, parent_meta = fold_paths(config, design, self.params_from, seed, fold_name)
        if not parent_meta.exists():
            raise FileNotFoundError(f"{self.name} needs {self.params_from} to finish first: {parent_meta}")
        params = json.loads(parent_meta.read_text(encoding="utf-8"))["params"]
        return {**params, **self.overrides}


class XGBoostArm(Arm):
    """Tuned XGBoost. `weighting` is "balanced", "balanced_country" or "tuned_scale_pos_weight"."""

    def __init__(self, name: str, weighting: str):
        self.name, self.weighting = name, weighting

    def trials(self, config):
        return config.xgb_trials

    def suggest(self, trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }
        if self.weighting == "tuned_scale_pos_weight":
            params["scale_pos_weight"] = trial.suggest_float("scale_pos_weight", 1.0, 10.0, log=True)
        return params

    def prepare(self, train, val, use_country, context):
        return prepare_xgb(train, val, use_country)

    def fit_predict(self, params, prepared, config, seed, tuning):
        weight = None
        if self.weighting == "balanced":
            weight = class_weights(prepared.y_train)
        elif self.weighting == "balanced_country":
            weight = class_weights(prepared.y_train) * country_weights(prepared.country_train)
        model = XGBoostModel(seed=seed, **params).fit(prepared.X_train, prepared.y_train, sample_weight=weight)
        return model.predict_proba(prepared.X_val), {}


class MLPArm(Arm):
    """The baseline MLP search space. `weighting` is "balanced" or "balanced_country"."""

    def __init__(self, name: str, weighting: str, selection: str):
        self.name, self.weighting, self.selection = name, weighting, selection

    def trials(self, config):
        return config.mlp_trials

    def suggest(self, trial):
        return {
            "hidden_dim_1": trial.suggest_int("hidden_dim_1", 8, 128),
            "hidden_dim_2": trial.suggest_int("hidden_dim_2", 8, 128),
            "embedding_dim": trial.suggest_int("embedding_dim", 4, 16),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        }

    def finalize_params(self, params):
        return dict(params)

    def prepare(self, train, val, use_country, context):
        return prepare_mlp(train, val, use_country)

    def _model_args(self, params, prepared, config, seed, tuning) -> dict:
        args = {k: v for k, v in params.items() if not k.startswith("hidden_dim_")}
        args["hidden_dims"] = (params["hidden_dim_1"], params["hidden_dim_2"])
        return {
            **args,
            "categorical_columns": prepared.categorical_columns,
            "vocab_sizes": prepared.vocab_sizes,
            "numeric_columns": PT_NUMERIC_COLUMNS,
            "epochs": config.tuning_epochs if tuning else config.final_epochs,
            "seed": seed,
        }

    def _weights(self, prepared) -> np.ndarray:
        weight = class_weights(prepared.y_train)
        if self.weighting == "balanced_country":
            weight = weight * country_weights(prepared.country_train)
        return weight

    def fit_predict(self, params, prepared, config, seed, tuning):
        model = PyTorchMLP(**self._model_args(params, prepared, config, seed, tuning))
        model.fit(prepared.X_train, prepared.y_train, sample_weight=self._weights(prepared))
        return model.predict_proba(prepared.X_val), {}


class GroupDROArm(MLPArm):
    """The same MLP trained with Group DRO. Class weights match the ERM MLP (rule 5, brief item 3).

    `group_by_label` makes the groups country by label instead of country, for E7.
    """

    def __init__(self, name: str, group_by_label: bool = False, params_from: str | None = None, overrides: dict | None = None):
        super().__init__(name, weighting="balanced", selection="worst_country")
        self.group_by_label = group_by_label
        self.params_from = params_from
        self.overrides = overrides or {}

    def suggest(self, trial):
        return {
            **super().suggest(trial),
            "step_size": trial.suggest_float("step_size", *STEP_SIZE_RANGE, log=True),
            "adjustment_c": trial.suggest_categorical("adjustment_c", ADJUSTMENT_CHOICES),
        }

    def fit_predict(self, params, prepared, config, seed, tuning):
        groups = group_ids(prepared.country_train, prepared.y_train if self.group_by_label else None)
        model = GroupDROMLP(**self._model_args(params, prepared, config, seed, tuning))
        model.fit(prepared.X_train, prepared.y_train, sample_weight=self._weights(prepared), groups=groups)
        extra = {} if tuning else _dro_history_extra(model.history_)
        return model.predict_proba(prepared.X_val), extra


class TabPFNArm(Arm):
    """TabPFN v2 with default settings and no tuning. `context` is "stratified" (D1.2) or "balanced" (H1)."""

    selection = None

    def __init__(self, name: str, context: str):
        self.name, self.context = name, context

    def prepare(self, train, val, use_country, context):
        return prepare_matrix(train, val, use_country)

    def fit_predict(self, params, prepared, config, seed, tuning):
        if self.context == "stratified":
            strata = prepared.country_train.astype(str) + "_" + prepared.y_train.astype(str)
            lists = stratified_context_lists(strata, config.tabpfn_estimators, config.context_rows, seed)
        else:
            lists = balanced_context_lists(
                prepared.country_train, prepared.y_binary, config.tabpfn_estimators, config.context_rows, seed
            )
        scores = predict_scores(
            prepared.X_train, prepared.y_binary, prepared.X_val,
            context_lists=lists, categorical_indices=list(range(len(prepared.categorical_columns))),
            device=config.device, seed=seed, n_estimators=config.tabpfn_estimators,
        )
        return scores, {"context_rows_per_estimator": sorted({len(rows) for rows in lists})}


class EmbeddingHeadArm(Arm):
    """H2: a head on TabPFN embeddings. `objective` is "erm", "reweighted" or "group_dro"."""

    selection = "worst_country"  # The same criterion as the MLP arms of E3.

    def __init__(self, name: str, objective: str):
        self.name, self.objective = name, objective

    def trials(self, config):
        return config.mlp_trials

    def suggest(self, trial):
        params = {
            "hidden_dim": trial.suggest_int("hidden_dim", 8, 128),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        }
        if self.objective == "group_dro":
            params["step_size"] = trial.suggest_float("step_size", *STEP_SIZE_RANGE, log=True)
            params["adjustment_c"] = trial.suggest_categorical("adjustment_c", ADJUSTMENT_CHOICES)
        return params

    def fold_context(self, train, val, use_country, config, seed, design, fold_name):
        return fold_embeddings(train, val, use_country, config, seed, design, fold_name)

    def prepare(self, train, val, use_country, context):
        return prepare_embeddings(train, val, context)

    def fit_predict(self, params, prepared, config, seed, tuning):
        weight = class_weights(prepared.y_train)
        if self.objective == "reweighted":
            weight = weight * country_weights(prepared.country_train)
        groups = group_ids(prepared.country_train) if self.objective == "group_dro" else None
        epochs = config.tuning_epochs if tuning else config.final_epochs
        scores, history = fit_head(prepared, weight, params, epochs, seed, groups=groups)
        extra = _dro_history_extra(history) if history is not None and not tuning else {}
        return scores, extra


def build_arms() -> dict[str, Arm]:
    arms: list[Arm] = [
        # E2: main benchmark. Selection on pooled inner PR-AUC, as pre-registered.
        TabPFNArm("tabpfn_v2", context="stratified"),
        XGBoostArm("xgb", weighting="balanced"),
        XGBoostArm("xgb_spw", weighting="tuned_scale_pos_weight"),  # The baseline's handling, as a sensitivity arm.
        MLPArm("mlp_erm", weighting="balanced", selection="pooled"),
        # E3: all MLP arms selected on worst-country inner PR-AUC (D0.4).
        MLPArm("mlp_erm_wc", weighting="balanced", selection="worst_country"),
        MLPArm("mlp_rw", weighting="balanced_country", selection="worst_country"),
        GroupDROArm("mlp_dro"),
        XGBoostArm("xgb_rw", weighting="balanced_country"),
        # E3 ablations: the parent's tuned parameters, with one setting changed. Nothing is tuned.
        *[
            GroupDROArm(f"mlp_dro_step_{step}", params_from="mlp_dro", overrides={"step_size": step})
            for step in (0.0, 0.001, 0.01, 0.1, 1.0)
        ],
        *[
            GroupDROArm(f"mlp_dro_c_{c}", params_from="mlp_dro", overrides={"adjustment_c": c})
            for c in ADJUSTMENT_CHOICES
        ],
        # E5: hybrids. h2_erm is a control and is not counted as a hybrid in RQ4 (D0.6).
        TabPFNArm("h1_balanced", context="balanced"),
        EmbeddingHeadArm("h2_erm", objective="erm"),
        EmbeddingHeadArm("h2_rw", objective="reweighted"),
        EmbeddingHeadArm("h2_dro", objective="group_dro"),
        # E7: group definition.
        GroupDROArm("mlp_dro_country_label", group_by_label=True),
    ]
    return {arm.name: arm for arm in arms}
