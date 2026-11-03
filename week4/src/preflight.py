"""Checks a machine is ready for the Phase 4 runs, before any long job starts.

Stops with an error if the code differs from its commit, the data is missing, or a requested GPU
is not visible. Then compares TabPFN v2 on the GPU (fp16 autocast, the package default there)
with the CPU (float32, as in Phase 1) on synthetic labels, and times one planned E2 fit.
Writes week4/results/runs/phase4_preflight_<host>.json.

  week4/.venv/bin/python week4/src/preflight.py --device cuda
"""
from __future__ import annotations

import argparse
import platform
import sys
import time

import numpy as np
import tabpfn
import torch
from sklearn.metrics import roc_auc_score
from tabpfn.utils import infer_autocast_inference_mode

from harness import RunConfig, country_label_key, load_labelled, outer_folds, prepare_matrix
from run_record import REPO_ROOT, _git_commit, save_run
from tabpfn_tools import predict_scores, stratified_context_lists

AGREEMENT_CONTEXT_ROWS = 2_000
AGREEMENT_QUERY_ROWS = 1_000


def fail(message: str) -> None:
    sys.exit(f"PREFLIGHT FAILED: {message}")


def synthetic_labels(X: np.ndarray, seed: int) -> np.ndarray:
    """Labels from a fixed random rule on the real features, so scores vary but no real label is used."""
    rng = np.random.default_rng(seed)
    z = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    logits = z @ rng.normal(size=X.shape[1]) - 1.5
    return (rng.random(len(X)) < 1 / (1 + np.exp(-logits))).astype(int)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-timing", action="store_true", help="Skip the full-size E2 fit")
    parser.add_argument("--allow-dirty", action="store_true", help="Only for testing this script before a commit")
    args = parser.parse_args()
    record: dict = {"host": platform.node(), "python": platform.python_version()}

    commit = _git_commit()
    if commit.endswith("-dirty") and not args.allow_dirty:
        fail(f"week4/src differs from commit {commit}. Run from a clean clone so results trace to a commit.")
    record["git_commit"] = commit

    for name in ("Train_v2.csv", "Test_v2.csv"):
        if not (REPO_ROOT / "data/raw" / name).exists():
            fail(f"data/raw/{name} is missing")
    frame = load_labelled()
    if len(frame) != 23_524:
        fail(f"expected 23,524 labelled rows (results/data_audit.md), found {len(frame)}")
    record["rows"] = len(frame)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        fail(f"--device {args.device}, but torch {torch.__version__} cannot see a GPU (driver or CUDA build?)")
    record.update(
        torch=torch.__version__, tabpfn=tabpfn.__version__, device=args.device,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        gpu_memory_gb=round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if torch.cuda.is_available() else None,
        cpu_threads=torch.get_num_threads(),
        tabpfn_autocast_on_device=infer_autocast_inference_mode([torch.device(args.device)], enable=None),
    )
    print(record, flush=True)

    # One outer fold of seed 0, real features, synthetic labels.
    _, train_idx, val_idx = next(outer_folds(frame, seed=0))
    train, val = frame.iloc[train_idx], frame.iloc[val_idx]
    prepared = prepare_matrix(train, val, use_country=True)
    categorical = list(range(len(prepared.categorical_columns)))
    y_all = synthetic_labels(np.vstack([prepared.X_train, prepared.X_val]), seed=0)
    y_train, y_val = y_all[: len(train)], y_all[len(train):]

    strata = country_label_key(train).reset_index(drop=True)
    small = stratified_context_lists(strata, 8, max_rows=AGREEMENT_CONTEXT_ROWS, seed=0)
    query = prepared.X_val[:AGREEMENT_QUERY_ROWS]
    scores = {}
    for device in dict.fromkeys(["cpu", args.device]):
        start = time.perf_counter()
        scores[device] = predict_scores(
            prepared.X_train, y_train, query, context_lists=small,
            categorical_indices=categorical, device=device, seed=0,
        )
        record[f"agreement_seconds_{device}"] = round(time.perf_counter() - start, 1)
    if args.device != "cpu":
        diff = np.abs(scores["cpu"] - scores[args.device])
        record["agreement"] = {
            "context_rows": AGREEMENT_CONTEXT_ROWS, "query_rows": len(query), "labels": "synthetic",
            "max_abs_score_difference": float(diff.max()),
            "mean_abs_score_difference": float(diff.mean()),
            "auc_cpu": float(roc_auc_score(y_val[: len(query)], scores["cpu"])),
            f"auc_{args.device}": float(roc_auc_score(y_val[: len(query)], scores[args.device])),
        }
        print("agreement", record["agreement"], flush=True)

    if not args.skip_timing:
        # The planned E2 setting (D1.2): 8 stratified contexts of about 10,000 rows, all validation rows.
        lists = stratified_context_lists(strata, 8, max_rows=RunConfig().context_rows, seed=0)
        start = time.perf_counter()
        predict_scores(
            prepared.X_train, y_train, prepared.X_val, context_lists=lists,
            categorical_indices=categorical, device=args.device, seed=0,
        )
        seconds = time.perf_counter() - start
        record["e2_fit_seconds"] = round(seconds, 1)
        # Rough upper-end projection, scaled by fit count only: E2 25 fits, E4 20, E5a 25, and H2's
        # 45 nested and LOCO folds at 6 fits each (5 out-of-fold, 1 for validation rows). Not a measurement.
        record["tabpfn_hours_projection_8_estimators"] = round(seconds * (25 + 20 + 25 + 6 * 45) / 3600, 1)
        print(f"one planned E2 fit on {args.device}: {seconds:.1f}s", flush=True)

    path = save_run(f"phase4_preflight_{platform.node()}", {"device": args.device}, record)
    print(f"PREFLIGHT PASSED. Record: {path}", flush=True)


if __name__ == "__main__":
    main()
