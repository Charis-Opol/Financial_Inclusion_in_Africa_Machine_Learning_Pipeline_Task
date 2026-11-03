"""Phase 1 toy checks of what the installed TabPFN really does (rule 3).

No check here measures predictive performance on the study's labels:
  - the behaviour checks use synthetic data;
  - the runtime check uses real feature rows with RANDOMLY PERMUTED labels and only
    records seconds, so it cannot reveal how TabPFN will score.

Results accumulate in week4/results/runs/phase1_tabpfn_feasibility.json, one key per
section, so sections can be run or rerun separately:
  week4/.venv/Scripts/python.exe week4/src/tabpfn_feasibility.py versions limits schema
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import tabpfn
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion
from tabpfn.finetuning import FinetunedTabPFNClassifier
from tabpfn.settings import settings

from fin_inclusion.data.loader import TARGET_COLUMN, DataLoader
from fin_inclusion.preprocessing.pipeline_factory import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    build_pytorch_pipeline,
    build_xgboost_pipeline,
)
from run_record import REPO_ROOT, RUNS_DIR, _git_commit
from tabpfn_tools import stratified_context_lists

OUTPUT = RUNS_DIR / "phase1_tabpfn_feasibility.json"
SEED = 0
MODEL_FEATURES = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS + ["household_size_outlier"]


def v2_classifier(**overrides) -> TabPFNClassifier:
    """TabPFN v2 specifically. A plain TabPFNClassifier() would use the gated default version."""
    options = {"device": "cpu", "random_state": SEED, "ignore_pretraining_limits": True}
    return TabPFNClassifier.create_default_for_version(ModelVersion.V2, **{**options, **overrides})


def load_train() -> pd.DataFrame:
    return DataLoader(
        raw_train_path=REPO_ROOT / "data/raw/Train_v2.csv",
        raw_test_path=REPO_ROOT / "data/raw/Test_v2.csv",
    ).load_train()


def noise_data(n: int, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Features and labels that are independent, so any skill on these labels is leakage."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 10)), rng.integers(0, 2, size=n)


def signal_data(n: int, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Labels that truly depend on the features, as a positive control."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 10))
    score = X[:, 0] + X[:, 1] * X[:, 2] + 0.5 * rng.normal(size=n)
    return X, (score > np.median(score)).astype(int)


def head_auc(embeddings: np.ndarray, y: np.ndarray) -> float:
    """AUC of a logistic head under 5-fold CV: how well an embedding reveals y."""
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    head = LogisticRegression(max_iter=2000, C=1.0)
    scores = cross_val_predict(head, embeddings, y, cv=folds, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, scores))


def squeeze_estimators(embeddings: np.ndarray) -> np.ndarray:
    """Average over estimators, giving (n_rows, embedding_dim)."""
    return embeddings.mean(axis=0) if embeddings.ndim == 3 else embeddings


def out_of_fold_embeddings(X: np.ndarray, y: np.ndarray, n_folds: int = 5, **model_args) -> np.ndarray:
    """Each row is embedded by a model whose context excludes that row."""
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    result = None
    for fit_idx, held_idx in folds.split(X, y):
        model = v2_classifier(**model_args).fit(X[fit_idx], y[fit_idx])
        embedded = squeeze_estimators(model.get_embeddings(X[held_idx], data_source="test"))
        if result is None:
            result = np.zeros((len(X), embedded.shape[1]))
        result[held_idx] = embedded
    return result


# ---------------------------------------------------------------- sections


def section_versions() -> dict:
    """Installed versions, the default model version, and the checkpoint v2 resolves to."""
    model = v2_classifier(n_estimators=1)
    X, y = signal_data(60)
    start = time.perf_counter()
    model.fit(X, y).predict_proba(X[:5])
    checkpoint = Path(model.model_path if isinstance(model.model_path, (str, Path)) else model.model_path[0])
    return {
        "tabpfn": tabpfn.__version__,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "python": platform.python_version(),
        "cuda_available": torch.cuda.is_available(),
        "cpu_threads": torch.get_num_threads(),
        "package_default_model_version": str(settings.tabpfn.model_version),
        "v2_checkpoint": checkpoint.name,
        "v2_checkpoint_mb": round(checkpoint.stat().st_size / 1e6, 1) if checkpoint.exists() else None,
        "first_fit_predict_seconds_including_download": round(time.perf_counter() - start, 1),
        "login_prompted": False,
    }


def section_limits() -> dict:
    """What the installed code actually enforces for v2 on CPU."""

    def attempt(n_rows: int, n_features: int = 5, n_classes: int = 2, **args) -> str:
        rng = np.random.default_rng(SEED)
        X = rng.normal(size=(n_rows, n_features))
        y = np.arange(n_rows) % n_classes
        try:
            v2_classifier(n_estimators=1, **args).fit(X, y)
            return "accepted"
        except Exception as error:  # noqa: BLE001  The message is the finding here.
            return f"{type(error).__name__}: {str(error).splitlines()[0][:160]}"

    strict = {"ignore_pretraining_limits": False}
    outcomes = {
        "cpu_1000_rows_strict": attempt(1000, **strict),
        "cpu_1001_rows_strict": attempt(1001, **strict),
        "cpu_10001_rows_ignore_limits": attempt(10_001),
        "classes_11_ignore_limits": attempt(200, n_classes=11),
    }
    settings.tabpfn.allow_cpu_large_dataset = True
    try:
        outcomes["cpu_allowed_10000_rows_strict"] = attempt(10_000, **strict)
        outcomes["cpu_allowed_10001_rows_strict"] = attempt(10_001, **strict)
        outcomes["cpu_allowed_501_features_strict"] = attempt(200, n_features=501, **strict)
    finally:
        settings.tabpfn.allow_cpu_large_dataset = False
    return outcomes


def section_schema() -> dict:
    """Encoded feature counts, and category levels unseen when a country is held out. No labels used."""
    train = load_train()
    xgb = build_xgboost_pipeline().fit_transform(train)
    non_features = {"uniqueid", "year", TARGET_COLUMN}
    pt = build_pytorch_pipeline().fit(train).transform(train)
    result = {
        "xgboost_encoded_columns": len([c for c in xgb.columns if c not in non_features]),
        "pytorch_branch_columns": len([c for c in pt.columns if c not in non_features]),
        "model_features_for_tabpfn": MODEL_FEATURES,
        "loco_unseen_levels": {},
    }
    # Same cleaning as the pipelines; a level only the held-out country has is "unseen".
    cleaned = build_pytorch_pipeline().named_steps["missingness"].transform(train)
    for country in sorted(train["country"].unique()):
        held = cleaned["country"] == country
        unseen = {}
        for column in [c for c in CATEGORICAL_COLUMNS if c != "country"]:
            levels = sorted(set(cleaned.loc[held, column]) - set(cleaned.loc[~held, column]))
            if levels:
                unseen[column] = {level: int((cleaned.loc[held, column] == level).sum()) for level in levels}
        result["loco_unseen_levels"][country] = unseen
    return result


def section_embeddings() -> dict:
    """Shapes, and the decisive leakage test: on pure-noise labels nothing should predict y."""
    n = 600
    X_noise, y_noise = noise_data(n)
    X_extra, _ = noise_data(50, seed=SEED + 1)
    model = v2_classifier(n_estimators=1).fit(X_noise, y_noise)
    train_tokens = model.get_embeddings(X_noise, data_source="train")
    test_tokens_same_rows = model.get_embeddings(X_noise, data_source="test")
    test_tokens_new_rows = model.get_embeddings(X_extra, data_source="test")
    multi = v2_classifier(n_estimators=3).fit(X_noise[:200], y_noise[:200])
    train_from_new_rows = model.get_embeddings(X_extra, data_source="train")

    # Does a row's own label reach its own embedding? Flip one label and compare.
    flipped = y_noise.copy()
    flipped[0] = 1 - flipped[0]
    flipped_model = v2_classifier(n_estimators=1).fit(X_noise, flipped)
    train_flipped = flipped_model.get_embeddings(X_noise, data_source="train")
    test_flipped = flipped_model.get_embeddings(X_extra, data_source="test")
    train_a, train_b = squeeze_estimators(train_tokens), squeeze_estimators(train_flipped)
    other_rows = np.linalg.norm(train_a[1:] - train_b[1:], axis=1)

    X_signal, y_signal = signal_data(n)
    signal_model = v2_classifier(n_estimators=1).fit(X_signal[:400], y_signal[:400])
    held_scores = signal_model.predict_proba(X_signal[400:])[:, 1]

    return {
        "shape_one_estimator_train_tokens": list(train_tokens.shape),
        "shape_one_estimator_test_tokens": list(test_tokens_new_rows.shape),
        "shape_three_estimators_test_tokens": list(multi.get_embeddings(X_extra, data_source="test").shape),
        "train_tokens_shape_when_passed_other_rows": list(train_from_new_rows.shape),
        "noise_labels_head_auc": {
            "A_train_tokens_of_context_rows": head_auc(squeeze_estimators(train_tokens), y_noise),
            "B_test_tokens_row_also_in_context": head_auc(squeeze_estimators(test_tokens_same_rows), y_noise),
            "C_out_of_fold_test_tokens": head_auc(out_of_fold_embeddings(X_noise, y_noise, n_estimators=1), y_noise),
        },
        "signal_labels_head_auc": {
            "C_out_of_fold_test_tokens": head_auc(out_of_fold_embeddings(X_signal, y_signal, n_estimators=1), y_signal),
            "direct_tabpfn_prediction_on_held_out_rows": float(roc_auc_score(y_signal[400:], held_scores)),
        },
        "flip_one_label_embedding_change": {
            "flipped_row_train_token_l2": float(np.linalg.norm(train_a[0] - train_b[0])),
            "other_rows_train_token_median_l2": float(np.median(other_rows)),
            "test_tokens_median_l2": float(
                np.median(
                    np.linalg.norm(
                        squeeze_estimators(test_tokens_new_rows) - squeeze_estimators(test_flipped), axis=1
                    )
                )
            ),
        },
    }


def section_embeddings_null() -> dict:
    """Is the out-of-fold AUC on noise really chance? Repeat over noise draws, with a raw-feature reference."""
    rows = []
    for seed in range(8):
        X, y = noise_data(600, seed=100 + seed)
        rows.append(
            {
                "noise_seed": 100 + seed,
                "out_of_fold_embedding_head_auc": head_auc(out_of_fold_embeddings(X, y, n_estimators=1), y),
                "raw_feature_head_auc": head_auc(X, y),
            }
        )
    frame = pd.DataFrame(rows)
    return {
        "repeats": rows,
        "out_of_fold_mean": float(frame["out_of_fold_embedding_head_auc"].mean()),
        "out_of_fold_sd": float(frame["out_of_fold_embedding_head_auc"].std()),
        "raw_mean": float(frame["raw_feature_head_auc"].mean()),
        "raw_sd": float(frame["raw_feature_head_auc"].std()),
    }


def section_context() -> dict:
    """H1 and the row-limit handling: can we control exactly which rows are the context?"""
    X, y = signal_data(600)
    groups = np.arange(600) % 4
    balanced = np.concatenate([np.flatnonzero(groups == g)[:50] for g in range(4)])
    query = X[500:520]

    subset = v2_classifier(n_estimators=1).fit(X[balanced], y[balanced])
    full = v2_classifier(n_estimators=1).fit(X, y)

    # Per-estimator explicit row lists: an ensemble in which estimators see different rows.
    lists = [np.arange(0, 300), np.arange(300, 600)]
    listed = v2_classifier(n_estimators=2, inference_config={"SUBSAMPLE_SAMPLES": lists}).fit(X, y)
    train_tokens = listed.get_embeddings(X[:5], data_source="train")

    equivalent = v2_classifier(n_estimators=1, inference_config={"SUBSAMPLE_SAMPLES": [balanced]}).fit(X, y)
    return {
        "subset_before_fit_context_rows": len(balanced),
        "subset_changes_prediction_vs_full": bool(
            not np.allclose(subset.predict_proba(query), full.predict_proba(query))
        ),
        "explicit_lists_train_token_shape": list(train_tokens.shape),
        "explicit_list_vs_subset_before_fit_max_abs_diff": float(
            np.abs(equivalent.predict_proba(query) - subset.predict_proba(query)).max()
        ),
        "methods_available": ["balanced", "stratified", "majority_downsample", "explicit index lists"],
    }


class _RecordingFinetuner(FinetunedTabPFNClassifier):
    """Overrides the loss hook on the fine-tuner, to test whether a custom loss can be used."""

    forward_calls = 0
    query_batch_fields: tuple[str, ...] = ()
    query_size_seen = 0

    def _forward_with_loss(self, batch):  # noqa: ANN001, ANN202
        type(self).forward_calls += 1
        type(self).query_batch_fields = tuple(batch.__dataclass_fields__)
        type(self).query_size_seen = int(batch.y_query.shape[1])
        return super()._forward_with_loss(batch)


def _finetune_once(X: np.ndarray, y: np.ndarray, n_samples: int) -> tuple[_RecordingFinetuner, float]:
    """One epoch on v2 on CPU, with the validation pass and final model kept as small as possible."""
    _RecordingFinetuner.forward_calls = 0
    model = _RecordingFinetuner(
        device="cpu",
        epochs=1,
        model_version=ModelVersion.V2,
        n_finetune_ctx_plus_query_samples=n_samples,
        n_estimators_finetune=1,
        n_estimators_validation=1,
        n_estimators_final_inference=1,
        validation_split_ratio=0.2,
        random_state=SEED,
        extra_classifier_kwargs={"ignore_pretraining_limits": True},
    )
    start = time.perf_counter()
    model.fit(X, y)
    return model, time.perf_counter() - start


def section_finetune() -> dict:
    """Does fine-tuning run on CPU on v2, can the loss be replaced, and does a batch know row groups?"""
    X, y = signal_data(300)
    model, seconds = _finetune_once(X[:250], y[:250], n_samples=200)
    result: dict = {
        "import_path": "tabpfn.finetuning.FinetunedTabPFNClassifier",
        "ran_on_cpu": True,
        "toy_seconds_one_epoch_250_rows": round(seconds, 1),
        "loss_hook_called_times": _RecordingFinetuner.forward_calls,
        "batch_fields": list(_RecordingFinetuner.query_batch_fields),
        "query_rows_in_one_batch": _RecordingFinetuner.query_size_seen,
        "batch_has_row_ids_or_groups": any(
            name in {"row_ids", "groups", "group", "indices", "sample_weight"}
            for name in _RecordingFinetuner.query_batch_fields
        ),
        "toy_auc_after_finetuning_on_synthetic": float(roc_auc_score(y[250:], model.predict_proba(X[250:])[:, 1])),
    }
    X_large, y_large = signal_data(2500)
    _, seconds = _finetune_once(X_large, y_large, n_samples=2000)
    result["toy_seconds_one_epoch_2500_rows"] = round(seconds, 1)
    result["steps_in_2500_row_epoch"] = _RecordingFinetuner.forward_calls
    return result


def _timed_ladder(plan: list[tuple[int, int, int]], partial_key: str) -> list[dict]:
    """Seconds for fit+predict on real feature rows with permuted labels. No score is computed."""
    train = load_train()
    pipeline = build_pytorch_pipeline().fit(train)
    X = pipeline.transform(train)[MODEL_FEATURES].to_numpy(dtype=np.float32)
    rng = np.random.default_rng(SEED)
    y = rng.permutation(train[TARGET_COLUMN].eq("Yes").to_numpy().astype(int))
    categorical = list(range(len(CATEGORICAL_COLUMNS)))

    rows = []
    budget_seconds = 2400  # Stop the ladder once one step takes longer than this.
    for context, query, estimators in plan:
        order = rng.permutation(len(X))
        fit_idx, query_idx = order[:context], order[context : context + query]
        model = v2_classifier(n_estimators=estimators, categorical_features_indices=categorical)
        start = time.perf_counter()
        model.fit(X[fit_idx], y[fit_idx])
        fitted = time.perf_counter()
        model.predict_proba(X[query_idx])
        done = time.perf_counter()
        rows.append(
            {
                "context_rows": context,
                "query_rows": query,
                "n_estimators": estimators,
                "fit_seconds": round(fitted - start, 1),
                "predict_seconds": round(done - fitted, 1),
                "total_seconds": round(done - start, 1),
            }
        )
        print(rows[-1], flush=True)
        _store(partial_key, rows)
        if done - start > budget_seconds:
            rows.append({"stopped": f"last step exceeded {budget_seconds} seconds"})
            break
    return rows


def section_runtime() -> dict:
    """First ladder. A stray process of mine competed for CPU until step 7 (see RESEARCH_LOG)."""
    plan = [
        (500, 1000, 1),
        (500, 1000, 8),
        (1000, 1000, 8),
        (2000, 1000, 8),
        (5000, 1000, 8),
        (5000, 4705, 8),
        (10000, 1000, 8),
        (10000, 4705, 8),
    ]
    return {"steps": _timed_ladder(plan, "runtime_partial"), "cpu_threads": torch.get_num_threads()}


def section_runtime_clean() -> dict:
    """Rerun at a real outer test-fold size, with nothing else competing for CPU."""
    plan = [(1000, 4705, 8), (2000, 4705, 8), (5000, 4705, 8)]
    return {"steps": _timed_ladder(plan, "runtime_clean_partial"), "cpu_threads": torch.get_num_threads()}


def section_runtime_planned() -> dict:
    """The planned E2 setting on one real-sized outer fold. Permuted labels, and no score is computed."""
    train = load_train()
    X = build_pytorch_pipeline().fit(train).transform(train)[MODEL_FEATURES].to_numpy(dtype=np.float32)
    strata = train["country"].astype(str) + "_" + train[TARGET_COLUMN].astype(str)
    rng = np.random.default_rng(SEED)
    y = rng.permutation(train[TARGET_COLUMN].eq("Yes").to_numpy().astype(int))
    fit_idx, query_idx = next(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, strata)
    )
    lists = stratified_context_lists(strata.iloc[fit_idx], 8, 10_000, SEED)
    model = v2_classifier(
        n_estimators=8,
        categorical_features_indices=list(range(len(CATEGORICAL_COLUMNS))),
        inference_config={"SUBSAMPLE_SAMPLES": lists},
    )
    start = time.perf_counter()
    model.fit(X[fit_idx], y[fit_idx])
    fitted = time.perf_counter()
    model.predict_proba(X[query_idx])
    done = time.perf_counter()
    covered = len(set(np.concatenate(lists).tolist()))
    return {
        "train_fold_rows": len(fit_idx),
        "query_rows": len(query_idx),
        "n_estimators": 8,
        "rows_per_estimator": sorted({len(chosen) for chosen in lists}),
        "train_rows_used_by_at_least_one_estimator": covered,
        "fit_seconds": round(fitted - start, 1),
        "predict_seconds": round(done - fitted, 1),
        "total_seconds": round(done - start, 1),
    }


SECTIONS = {
    "versions": section_versions,
    "limits": section_limits,
    "schema": section_schema,
    "embeddings": section_embeddings,
    "embeddings_null": section_embeddings_null,
    "context": section_context,
    "finetune": section_finetune,
    "runtime": section_runtime,
    "runtime_clean": section_runtime_clean,
    "runtime_planned": section_runtime_planned,
}


def _store(key: str, value: object) -> None:
    record = json.loads(OUTPUT.read_text(encoding="utf-8")) if OUTPUT.exists() else {}
    record[key] = value
    record["git_commit"] = _git_commit()
    record["installed"] = subprocess.check_output(
        [str(Path(__file__).resolve().parents[1] / ".venv/Scripts/python.exe"), "-c", "import tabpfn;print(tabpfn.__version__)"],
        text=True,
    ).strip()
    OUTPUT.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sections", nargs="*", default=list(SECTIONS))
    for name in parser.parse_args().sections:
        print(f"=== {name}", flush=True)
        start = time.perf_counter()
        try:
            value = SECTIONS[name]()
            status = "ok"
        except Exception:  # noqa: BLE001  A failed check is itself a feasibility finding.
            value = {"failed": traceback.format_exc()[-1800:]}
            status = "FAILED"
        _store(name, {"status": status, "seconds": round(time.perf_counter() - start, 1), **value})
        print(json.dumps(value, indent=1, default=str)[:3000], flush=True)


if __name__ == "__main__":
    main()
