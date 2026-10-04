"""XAI Phase 2: obtain and verify the two models to explain, and fix the samples.

Writes:

    reports/xai/model_verification.json   held-out PR-AUC gate + parity checks
    reports/xai/sample_sets.json          background / explanation row positions
    models/xai/mlp.pt, mlp_metadata.json  retrained MLP (gitignored)

Why a gate: Phase 4 never saved the PyTorch model, so the MLP explained here
is a retrain. An explanation of a model that does not perform like the one
in the ablation report would be an explanation of something else, so nothing
is saved unless held-out PR-AUC lands within `PR_AUC_TOLERANCE` of the
reported nested-CV mean -- for both models.

What "held-out" means here. Neither final model has a labelled hold-out of
its own: the shipped XGBoost was fit on all 23,524 labelled rows, the MLP is
refit the same way to match it, and `Test_v2.csv` has no labels. The gate
therefore re-runs the report's outer 5-fold CV (same seed, same country x
target stratification, so the same folds) with each model's FINAL
hyperparameters fixed:

  - MLP: all five outer folds of Phase 4 selected identical hyperparameters,
    so this repeats the reported procedure exactly and should reproduce the
    reported per-fold numbers, not just the mean.
  - XGBoost: the shipped `production_xgb_params` were tuned on the full
    training set, i.e. on rows that are validation rows in this CV. The
    result is the held-out performance of the shipped configuration, mildly
    optimistic as an estimate, and it is compared with a nested-CV number
    whose per-fold hyperparameters differed. Stated in the output.
"""
from __future__ import annotations

import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "config"))
from settings import load_settings  # noqa: E402

from fin_inclusion.data.loader import TARGET_COLUMN, DataLoader  # noqa: E402
from fin_inclusion.evaluation.cv_runner import CVFoldResult, run_stratified_cv  # noqa: E402
from fin_inclusion.imbalance.strategies import ClassWeightStrategy, NoImbalanceStrategy  # noqa: E402
from fin_inclusion.interpretability.model_adapters import MLPRawModel, XGBoostRawModel, sha256_of  # noqa: E402
from fin_inclusion.interpretability.sampling import build_sample_sets, composition_table  # noqa: E402
from fin_inclusion.models.pytorch_mlp import PyTorchMLP  # noqa: E402
from fin_inclusion.models.xgboost_model import XGBoostModel  # noqa: E402
from fin_inclusion.preprocessing.pipeline_factory import (  # noqa: E402
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    build_pytorch_pipeline,
    build_xgboost_pipeline,
)

ROOT = Path(__file__).resolve().parent.parent
PRODUCTION_DIR = ROOT / "models" / "production"
NON_FEATURE_COLUMNS = ["uniqueid", "year", TARGET_COLUMN]
RAW_FEATURES = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS

MLP_ARM_KEY = "pt2_cw"
XGB_ARM_KEY = "xgb_cw"
# Same final-fit epoch count Phase 4 used (`run_tuning_and_ablation.PT_FINAL_EPOCHS`).
MLP_EPOCHS = 20
# The band the reported numbers must be reproduced within (PR-AUC, absolute).
PR_AUC_TOLERANCE = 0.02
# Refit-vs-shipped and adapter-vs-pipeline score agreement (sigmoid scale).
SCORE_ATOL = 1e-6


class VerificationGateError(RuntimeError):
    """A model does not reproduce its reported performance; nothing was saved."""


def mlp_hyperparameters(phase4: dict) -> dict:
    """The tuned MLP configuration, which must be unambiguous.

    Phase 4 tuned per outer fold; a retrain needs one configuration. If the
    folds ever disagreed, choosing among them would be a modelling decision
    that belongs in the decisions log, not a silent `[0]`.
    """
    per_fold = phase4["best_params_per_fold"][MLP_ARM_KEY]
    if any(p != per_fold[0] for p in per_fold):
        raise VerificationGateError(
            f"{MLP_ARM_KEY} folds selected different hyperparameters; pick one deliberately and log why"
        )
    params = dict(per_fold[0])
    params["hidden_dims"] = tuple(params["hidden_dims"])
    return params


def gate_summary(results: list[CVFoldResult], reported_folds: list[float], focus_country: str) -> dict:
    folds = [float(r.metrics["pr_auc"]) for r in results]
    reported_mean, mean = float(np.mean(reported_folds)), float(np.mean(folds))
    return {
        "reported_fold_pr_auc": reported_folds,
        "reported_mean_pr_auc": reported_mean,
        "fold_pr_auc": folds,
        "mean_pr_auc": mean,
        "std_pr_auc": float(np.std(folds)),
        "abs_diff_of_means": abs(mean - reported_mean),
        "max_abs_fold_diff": float(np.max(np.abs(np.array(folds) - np.array(reported_folds)))),
        f"{focus_country.lower()}_fold_pr_auc": [float(r.country_metrics[focus_country]["pr_auc"]) for r in results],
        f"{focus_country.lower()}_mean_pr_auc": float(
            np.mean([r.country_metrics[focus_country]["pr_auc"] for r in results])
        ),
        "tolerance": PR_AUC_TOLERANCE,
        "within_band": abs(mean - reported_mean) <= PR_AUC_TOLERANCE,
    }


def unknown_embedding_rows(model: MLPRawModel) -> dict:
    """Norm of each embedding's index-0 ("unknown") row beside the trained rows.

    No training row uses index 0, so that row is shaped only by its random
    initialisation and weight decay. Recorded because an IG "unknown"
    baseline starts its integration path at exactly this vector.
    """
    rows = {}
    for column, embedding in zip(model.categorical_columns, model.net.embeddings):
        norms = embedding.weight.detach().norm(dim=1).numpy()
        rows[column] = {
            "unknown_row_norm": float(norms[0]),
            "mean_trained_row_norm": float(norms[1:].mean()),
            "min_trained_row_norm": float(norms[1:].min()),
        }
    return rows


def main() -> None:
    settings = load_settings()
    seed, xai = settings.seed, settings.xai
    xai.artifacts_dir.mkdir(parents=True, exist_ok=True)

    phase4 = json.loads((settings.paths.reports_dir / "ablation_raw_results.json").read_text(encoding="utf-8"))
    train = DataLoader(settings.paths.raw_train, settings.paths.raw_test).load_train()
    y, country = train[TARGET_COLUMN], train["country"]
    y_binary = (y == "Yes").astype(int).to_numpy()
    X_raw = train[RAW_FEATURES]
    data_sha = sha256_of(settings.paths.raw_train)
    print(f"Loaded train: {train.shape}, sha256 {data_sha[:12]}")

    # --- The shipped XGBoost, loaded through its own metadata ---------------
    xgb_model = XGBoostRawModel.load(PRODUCTION_DIR)
    if xgb_model.metadata["training_data_sha256"] != data_sha:
        raise VerificationGateError("data/raw/Train_v2.csv is not the file the shipped model was trained on")
    xgb_params = xgb_model.metadata["hyperparameters"]
    X_xgb = build_xgboost_pipeline().fit_transform(train).drop(columns=NON_FEATURE_COLUMNS)
    if not np.array_equal(xgb_model.encode(X_raw), X_xgb.to_numpy(dtype=np.float32)):
        raise VerificationGateError("XGBoost raw-input adapter does not reproduce the training pipeline's encoding")
    xgb_scores = xgb_model.predict_score(X_raw)
    print(f"Loaded {xgb_model.name}; adapter encoding matches the pipeline on all {len(train)} rows")

    # The CV below evaluates "fit XGBoost with the production params". Show
    # that this procedure IS the shipped artifact: a full-data refit must
    # score the same as the booster on disk.
    refit_scores = XGBoostModel(seed=seed, **xgb_params).fit(X_xgb, y).predict_proba(X_xgb)
    refit_max_diff = float(np.max(np.abs(refit_scores - xgb_scores)))
    print(f"Refit-vs-shipped max |score diff|: {refit_max_diff:.2e}")

    # --- Verification gate: outer 5-fold CV, final hyperparameters ----------
    pt_pipeline = build_pytorch_pipeline().fit(train)
    X_pt = pt_pipeline.transform(train).drop(columns=NON_FEATURE_COLUMNS)
    vocab_sizes = pt_pipeline.named_steps["encode"].vocab_sizes_
    numeric_inputs = NUMERIC_COLUMNS + ["household_size_outlier"]
    mlp_params = mlp_hyperparameters(phase4)

    def build_mlp() -> PyTorchMLP:
        return PyTorchMLP(
            categorical_columns=CATEGORICAL_COLUMNS, vocab_sizes=vocab_sizes, numeric_columns=numeric_inputs,
            epochs=MLP_EPOCHS, seed=seed, **mlp_params,
        )

    print("\n=== Gate: XGBoost, production params, 5-fold CV ===")
    xgb_cv = run_stratified_cv(
        X_xgb, y, country, model_factory=lambda: XGBoostModel(seed=seed, **xgb_params),
        imbalance_strategy=NoImbalanceStrategy(), n_folds=settings.cv.n_outer_folds, seed=seed,
    )
    xgb_gate = gate_summary(xgb_cv, phase4["fold_pr_auc"][XGB_ARM_KEY], "Uganda")
    print(f"  folds {np.round(xgb_gate['fold_pr_auc'], 4)} mean {xgb_gate['mean_pr_auc']:.4f} "
          f"(reported {xgb_gate['reported_mean_pr_auc']:.4f})")

    print("=== Gate: MLP, tuned params, class-weight, 5-fold CV ===")
    mlp_cv = run_stratified_cv(
        X_pt, y, country, model_factory=build_mlp,
        imbalance_strategy=ClassWeightStrategy(), n_folds=settings.cv.n_outer_folds, seed=seed,
    )
    mlp_gate = gate_summary(mlp_cv, phase4["fold_pr_auc"][MLP_ARM_KEY], "Uganda")
    print(f"  folds {np.round(mlp_gate['fold_pr_auc'], 4)} mean {mlp_gate['mean_pr_auc']:.4f} "
          f"(reported {mlp_gate['reported_mean_pr_auc']:.4f})")

    verification = {
        "training_data_sha256": data_sha,
        "seed": seed,
        "cv": f"{settings.cv.n_outer_folds}-fold, stratified on country x target, same seed as Phase 4's outer loop",
        "xgboost": {
            **xgb_gate,
            "model_version": xgb_model.metadata["model_version"],
            "hyperparameters": xgb_params,
            "evaluation": "held-out folds, shipped production_xgb_params fixed",
            "caveat": "production params were tuned on the full training set (mildly optimistic); the reported "
                      "number is nested CV with per-fold tuned params, so the folds are not expected to match exactly",
            "adapter_encoding_matches_pipeline": True,
            "refit_vs_shipped_max_abs_score_diff": refit_max_diff,
            "refit_reproduces_shipped": refit_max_diff <= SCORE_ATOL,
        },
        "mlp": {
            **mlp_gate,
            "hyperparameters": {**mlp_params, "hidden_dims": list(mlp_params["hidden_dims"]), "epochs": MLP_EPOCHS},
            "hyperparameter_choice": f"all {len(phase4['best_params_per_fold'][MLP_ARM_KEY])} outer folds of "
                                     f"{MLP_ARM_KEY} selected this same configuration",
            "evaluation": "held-out folds, same procedure as the reported nested CV's final refit",
        },
    }
    verification_path = xai.artifacts_dir / "model_verification.json"
    verification_path.write_text(json.dumps(verification, indent=2), encoding="utf-8")

    failures = [name for name in ("xgboost", "mlp") if not verification[name]["within_band"]]
    if not verification["xgboost"]["refit_reproduces_shipped"]:
        failures.append("xgboost refit does not reproduce the shipped booster")
    if failures:
        raise VerificationGateError(
            f"Verification gate FAILED for {failures}; see {verification_path}. No model or sample set was written."
        )
    print(f"Gate passed (tolerance +/-{PR_AUC_TOLERANCE}). Wrote {verification_path}")

    # --- Final MLP: refit on every labelled row, as the shipped XGBoost was --
    print("\n=== Refitting the MLP on the full training set ===")
    X_res, y_res, sample_weight = ClassWeightStrategy().apply(X_pt, y)
    mlp = build_mlp().fit(X_res, y_res, sample_weight=sample_weight)
    mlp_threshold = statistics.median(phase4["thresholds"][MLP_ARM_KEY])
    mlp_model = MLPRawModel.from_fitted(
        mlp, pt_pipeline, decision_threshold=mlp_threshold,
        metadata={
            "imbalance_strategy": "class_weight",
            "seed": seed,
            "training_rows": int(len(train)),
            "training_data_sha256": data_sha,
            "threshold_source": f"median of {MLP_ARM_KEY} per-outer-fold F1-optimal thresholds",
            "hyperparameter_source": f"reports/ablation_raw_results.json best_params_per_fold[{MLP_ARM_KEY}]",
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    )

    cat, num = mlp_model.encode(X_raw)
    if not (np.array_equal(cat, X_pt[CATEGORICAL_COLUMNS].to_numpy(dtype=np.int64))
            and np.array_equal(num, X_pt[numeric_inputs].to_numpy(dtype=np.float32))):
        raise VerificationGateError("MLP raw-input adapter does not reproduce the training pipeline's encoding")
    mlp_scores = mlp_model.predict_score(X_raw)
    adapter_max_diff = float(np.max(np.abs(mlp_scores - mlp.predict_proba(X_pt))))
    if adapter_max_diff > SCORE_ATOL:
        raise VerificationGateError(f"MLP adapter scores differ from the model's by {adapter_max_diff:.2e}")

    metadata = mlp_model.save(xai.model_dir)
    reloaded = MLPRawModel.load(xai.model_dir)
    reload_max_diff = float(np.max(np.abs(reloaded.predict_logit(X_raw) - mlp_model.predict_logit(X_raw))))
    if reload_max_diff != 0.0:
        raise VerificationGateError(f"Reloaded MLP logits differ from the saved model's by {reload_max_diff:.2e}")
    print(f"Saved {xai.model_dir / 'mlp.pt'} (sha256 {metadata['weights_sha256'][:12]}, threshold {mlp_threshold:.4f})")

    verification["mlp"] |= {
        "weights_sha256": metadata["weights_sha256"],
        "torch_version": torch.__version__,
        "decision_threshold": mlp_threshold,
        "adapter_encoding_matches_pipeline": True,
        "adapter_vs_model_max_abs_score_diff": adapter_max_diff,
        "reload_max_abs_logit_diff": reload_max_diff,
        "unknown_embedding_rows": unknown_embedding_rows(mlp_model),
    }
    # In-sample, and labelled as such: both final models have seen every row.
    verification["in_sample_not_a_performance_estimate"] = {
        "xgboost_pr_auc": float(average_precision_score(y_binary, xgb_scores)),
        "mlp_pr_auc": float(average_precision_score(y_binary, mlp_scores)),
        "xgboost_share_predicted_yes": float((xgb_scores >= xgb_model.decision_threshold).mean()),
        "mlp_share_predicted_yes": float((mlp_scores >= mlp_threshold).mean()),
    }
    verification_path.write_text(json.dumps(verification, indent=2), encoding="utf-8")

    # --- Background and explanation sets -------------------------------------
    sets = build_sample_sets(
        country, y, background_size=xai.background_size,
        explanation_sample_size=xai.explanation_sample_size, seed=seed, training_data_sha256=data_sha,
    )
    xgb_logit, mlp_logit = xgb_model.predict_logit(X_raw), mlp_model.predict_logit(X_raw)
    composition = {
        "background": composition_table(sets.background, country, y),
        "pooled": composition_table(sets.pooled, country, y),
        "focus": composition_table(sets.focus, country, y),
        "background_and_pooled_overlap": int(np.intersect1d(sets.background, sets.pooled).size),
        "background_and_focus_overlap": int(np.intersect1d(sets.background, sets.focus).size),
        "positive_rate": {
            "all_training_rows": float(y_binary.mean()),
            "background": float(y_binary[sets.background].mean()),
            "pooled": float(y_binary[sets.pooled].mean()),
            "focus": float(y_binary[sets.focus].mean()),
        },
        # How well the background stands in for the population: its mean
        # logit is the base value every interventional attribution starts from.
        "mean_logit": {
            "xgboost": {"all_training_rows": float(xgb_logit.mean()), "background": float(xgb_logit[sets.background].mean())},
            "mlp": {"all_training_rows": float(mlp_logit.mean()), "background": float(mlp_logit[sets.background].mean())},
        },
    }
    sample_sets_path = xai.artifacts_dir / "sample_sets.json"
    sets.save(sample_sets_path, composition)
    print(f"\nBackground {sets.background.size} rows, pooled {sets.pooled.size}, "
          f"{sets.focus_country} {sets.focus.size}, explained in total {sets.explanation.size}")
    print(f"Wrote {sample_sets_path}")


if __name__ == "__main__":
    main()
