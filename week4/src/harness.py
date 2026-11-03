"""The one harness every model goes through (rule 5).

Same folds for a given seed, preprocessing fitted inside each fold (outer and inner), tuning in the
inner loop only, and raw out-of-fold scores saved per fold so runs can resume and every metric is
computed later from the same files.

A model is an `Arm` (see arms.py), which only has to say how to prepare a fold, what to tune and how
to fit and predict. An arm cannot reach outer validation labels: `fit_predict` gets features for the
validation rows, and labels only for the training rows.
"""
from __future__ import annotations

import json
import os
import platform
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

from fin_inclusion.data.loader import TARGET_COLUMN, DataLoader
from fin_inclusion.evaluation.cv_runner import POSITIVE_LABEL
from fin_inclusion.preprocessing.pipeline_factory import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    build_pytorch_pipeline,
    build_xgboost_pipeline,
    make_country_target_stratify_key,
)
from run_record import REPO_ROOT, RESULTS_DIR, _git_commit

optuna.logging.set_verbosity(optuna.logging.WARNING)

N_OUTER = 5
N_INNER = 3
ALWAYS_DROPPED = ["uniqueid", "year", TARGET_COLUMN]
PT_NUMERIC_COLUMNS = NUMERIC_COLUMNS + ["household_size_outlier"]


@dataclass(frozen=True)
class RunConfig:
    """Budgets for a run. Baseline values for a real run (scripts/run_tuning_and_ablation.py);
    `smoke()` shrinks everything to check that every code path runs."""

    device: str = "cpu"  # Only TabPFN uses this. XGBoost and the MLPs stay on CPU (see CODE_WALKTHROUGH).
    n_outer: int = N_OUTER
    n_inner: int = N_INNER
    xgb_trials: int = 15
    mlp_trials: int = 8
    tuning_epochs: int = 6
    final_epochs: int = 20
    tabpfn_estimators: int = 8
    h2_estimators: int = 2  # D1.6 on CPU. run_all.py --h2-estimators raises it (D3.4: 8 on the GPU machine).
    context_rows: int = 10_000
    output_dir: Path = RESULTS_DIR / "oof"
    cache_dir: Path = RESULTS_DIR / "cache"
    row_limit: int | None = None

    @classmethod
    def smoke(cls, device: str = "cpu") -> "RunConfig":
        return cls(
            device=device, n_outer=2, n_inner=2, xgb_trials=2, mlp_trials=2, tuning_epochs=1, final_epochs=1,
            tabpfn_estimators=1, h2_estimators=1, context_rows=300,
            output_dir=RESULTS_DIR / "smoke" / "oof", cache_dir=RESULTS_DIR / "smoke" / "cache", row_limit=2400,
        )


def load_labelled(config: RunConfig | None = None) -> pd.DataFrame:
    """Labelled rows, with the row's position as its index. That index is the row id everywhere."""
    frame = DataLoader(
        raw_train_path=REPO_ROOT / "data/raw/Train_v2.csv",
        raw_test_path=REPO_ROOT / "data/raw/Test_v2.csv",
    ).load_train().reset_index(drop=True)
    if config is not None and config.row_limit is not None:
        fraction = config.row_limit / len(frame)
        key = make_country_target_stratify_key(frame, target_col=TARGET_COLUMN)
        frame = frame.groupby(key, group_keys=False).sample(frac=fraction, random_state=0).sort_index()
    return frame


def country_label_key(frame: pd.DataFrame) -> pd.Series:
    return make_country_target_stratify_key(frame, target_col=TARGET_COLUMN)


def outer_folds(frame: pd.DataFrame, seed: int, n_splits: int = N_OUTER) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Nested-CV outer folds, as positions into `frame`. The same seed gives the same folds to every arm."""
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (train_idx, val_idx) in enumerate(splitter.split(frame, country_label_key(frame))):
        yield f"fold{fold}", train_idx, val_idx


def loco_folds(frame: pd.DataFrame) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Leave-one-country-out. The split has no randomness; the seed only affects the model."""
    countries = frame["country"].to_numpy()
    for name in sorted(set(countries)):
        yield f"heldout_{name}", np.flatnonzero(countries != name), np.flatnonzero(countries == name)


@dataclass
class Prepared:
    """One fold after preprocessing. Holds no validation labels."""

    X_train: pd.DataFrame | np.ndarray
    X_val: pd.DataFrame | np.ndarray
    y_train: pd.Series
    country_train: pd.Series
    categorical_columns: list[str] = field(default_factory=list)
    vocab_sizes: dict[str, int] = field(default_factory=dict)
    pipeline: object | None = None  # The fitted preprocessing, kept for E6 explanations.

    @property
    def y_binary(self) -> np.ndarray:
        return (self.y_train == POSITIVE_LABEL).to_numpy(dtype=int)


def _active_categoricals(use_country: bool) -> list[str]:
    return list(CATEGORICAL_COLUMNS) if use_country else [c for c in CATEGORICAL_COLUMNS if c != "country"]


def prepare_xgb(train: pd.DataFrame, val: pd.DataFrame, use_country: bool) -> Prepared:
    """One-hot branch, fitted on `train` only."""
    columns = _active_categoricals(use_country)
    pipeline = build_xgboost_pipeline(categorical_columns=columns).fit(train)
    # With country inactive the encoder leaves the raw column in place, so it is removed here.
    drop = ALWAYS_DROPPED + ([] if use_country else ["country"])
    return Prepared(
        X_train=pipeline.transform(train).drop(columns=drop),
        X_val=pipeline.transform(val).drop(columns=drop, errors="ignore"),
        y_train=train[TARGET_COLUMN],
        country_train=train["country"],
        pipeline=pipeline,
    )


def prepare_mlp(train: pd.DataFrame, val: pd.DataFrame, use_country: bool) -> Prepared:
    """Embedding-index branch with scaled numerics, fitted on `train` only."""
    columns = _active_categoricals(use_country)
    pipeline = build_pytorch_pipeline(categorical_columns=columns).fit(train)
    drop = ALWAYS_DROPPED + ([] if use_country else ["country"])
    return Prepared(
        X_train=pipeline.transform(train).drop(columns=drop),
        X_val=pipeline.transform(val).drop(columns=drop, errors="ignore"),
        y_train=train[TARGET_COLUMN],
        country_train=train["country"],
        categorical_columns=columns,
        vocab_sizes=pipeline.named_steps["encode"].vocab_sizes_,
        pipeline=pipeline,
    )


def prepare_matrix(train: pd.DataFrame, val: pd.DataFrame, use_country: bool) -> Prepared:
    """The MLP branch's output as a float matrix, for TabPFN (D1.10). Categoricals come first."""
    prepared = prepare_mlp(train, val, use_country)
    order = prepared.categorical_columns + PT_NUMERIC_COLUMNS
    prepared.X_train = prepared.X_train[order].to_numpy(dtype=np.float32)
    prepared.X_val = prepared.X_val[order].to_numpy(dtype=np.float32)
    return prepared


def class_weights(y: pd.Series) -> np.ndarray:
    """Balanced class weights, the same formula as the baseline's ClassWeightStrategy."""
    return compute_sample_weight(class_weight="balanced", y=y)


def within_country_class_weights(y: pd.Series, country: pd.Series) -> np.ndarray:
    """Balanced class weights computed inside each country (D3.3).

    Each country keeps its total weight (its row count), but its two classes weigh equally within
    it, so countries' losses are compared on the same class balance.
    """
    weight = np.empty(len(y), dtype=np.float64)
    countries = country.to_numpy()
    for name in np.unique(countries):
        rows = countries == name
        weight[rows] = compute_sample_weight(class_weight="balanced", y=y.to_numpy()[rows])
    return weight


def country_weights(country: pd.Series) -> np.ndarray:
    """Inverse country frequency, scaled to mean 1, so every country has equal total weight."""
    values = country.to_numpy()
    names, counts = np.unique(values, return_counts=True)
    per_country = dict(zip(names, len(values) / (len(names) * counts)))
    return np.array([per_country[v] for v in values], dtype=np.float64)


def group_ids(country: pd.Series, y: pd.Series | None = None) -> np.ndarray:
    """Integer group per row: country, or country by label for E7. Ids are contiguous from 0."""
    labels = country.astype(str) if y is None else country.astype(str) + "_" + y.astype(str)
    return np.unique(labels.to_numpy(), return_inverse=True)[1]


def selection_score(scores: np.ndarray, y_binary: np.ndarray, country: np.ndarray, selection: str) -> float:
    """The inner-loop criterion: pooled PR-AUC, or the lowest per-country PR-AUC."""
    if selection == "pooled":
        return float(average_precision_score(y_binary, scores))
    if selection == "worst_country":
        return min(
            float(average_precision_score(y_binary[country == c], scores[country == c]))
            for c in sorted(set(country))
            if y_binary[country == c].sum() > 0
        )
    raise ValueError(f"Unknown selection criterion: {selection}")


def tune(arm, train: pd.DataFrame, use_country: bool, config: RunConfig, seed: int, context: dict) -> dict:
    """Inner loop: an Optuna search scored on inner out-of-fold predictions from `train` only."""
    n_trials = arm.trials(config)
    if n_trials == 0:
        return {}
    y_binary = (train[TARGET_COLUMN] == POSITIVE_LABEL).to_numpy(dtype=int)
    countries = train["country"].to_numpy()
    splitter = StratifiedKFold(n_splits=config.n_inner, shuffle=True, random_state=seed)
    splits = list(splitter.split(train, country_label_key(train)))

    def objective(trial: optuna.Trial) -> float:
        params = arm.suggest(trial)
        scores = np.zeros(len(train))
        for fit_idx, val_idx in splits:
            prepared = arm.prepare(train.iloc[fit_idx], train.iloc[val_idx], use_country, context)
            scores[val_idx] = arm.fit_predict(params, prepared, config, seed, tuning=True)[0]
        return selection_score(scores, y_binary, countries, arm.selection)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return arm.finalize_params(dict(study.best_params))


def fold_paths(config: RunConfig, design: str, arm_name: str, seed: int, fold_name: str) -> tuple[Path, Path]:
    folder = config.output_dir / design / arm_name / f"seed{seed}"
    return folder / f"{fold_name}.csv.gz", folder / f"{fold_name}.json"


def _write_atomically(path: Path, write: Callable[[Path], None]) -> None:
    """A dropped session must not leave a half-written file that looks finished."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp" + path.suffix)
    write(temporary)
    os.replace(temporary, path)


def run_arm(arm, design: str, seed: int, frame: pd.DataFrame, config: RunConfig) -> None:
    """Runs one arm for one seed. Finished folds are skipped, so a dropped session can resume."""
    if design == "nested":
        folds, use_country = outer_folds(frame, seed, config.n_outer), True
    elif design == "loco":
        folds, use_country = loco_folds(frame), False
    else:
        raise ValueError(f"Unknown design: {design}")

    for fold_name, train_idx, val_idx in folds:
        scores_path, meta_path = fold_paths(config, design, arm.name, seed, fold_name)
        if meta_path.exists():
            print(f"skip   {design} {arm.name} seed{seed} {fold_name}", flush=True)
            continue
        start = time.perf_counter()
        train, val = frame.iloc[train_idx], frame.iloc[val_idx]
        context = arm.fold_context(train, val, use_country, config, seed, design, fold_name)
        params = (
            arm.inherited_params(config, design, seed, fold_name)
            if arm.params_from
            else tune(arm, train, use_country, config, seed, context)
        )
        prepared = arm.prepare(train, val.drop(columns=[TARGET_COLUMN]), use_country, context)
        scores, extra = arm.fit_predict(params, prepared, config, seed, tuning=False)

        output = pd.DataFrame(
            {
                "row_id": val.index,
                "country": val["country"].to_numpy(),
                "y": (val[TARGET_COLUMN] == POSITIVE_LABEL).to_numpy(dtype=int),
                "score": scores,
            }
        )
        _write_atomically(scores_path, lambda p: output.to_csv(p, index=False, compression="gzip"))
        meta = {
            "arm": arm.name, "design": design, "seed": seed, "fold": fold_name,
            "train_rows": len(train), "val_rows": len(val),
            "params": params, "selection": arm.selection, "device": config.device,
            "budget": {
                "tabpfn_estimators": config.tabpfn_estimators, "h2_estimators": config.h2_estimators,
                "context_rows": config.context_rows, "final_epochs": config.final_epochs,
            },
            "torch_threads": torch.get_num_threads(), "host": platform.node(),
            "seconds": round(time.perf_counter() - start, 1),
            "git_commit": _git_commit(), "extra": extra,
        }
        # The metadata goes last: its existence is what marks the fold as finished.
        _write_atomically(meta_path, lambda p: p.write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8"))
        print(f"done   {design} {arm.name} seed{seed} {fold_name} in {meta['seconds']}s", flush=True)
