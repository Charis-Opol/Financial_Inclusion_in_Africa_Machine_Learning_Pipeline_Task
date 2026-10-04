"""XAI Phase 2: obtain and verify the two models to explain, and fix the samples.

The script is five steps, in this order (see `main`):

    1. load the training data
    2. load the shipped XGBoost and check it is the model we think it is
    3. GATE: do both models perform as the ablation report says? (5-fold CV)
    4. refit the MLP on all rows and save it
    5. choose the background rows and the rows to explain

Writes:

    reports/xai/model_verification.json   held-out PR-AUC gate + parity checks
    reports/xai/sample_sets.json          background / explanation row positions
    models/xai/mlp.pt, mlp_metadata.json  retrained MLP (gitignored)

Why a gate: Phase 4 never saved the PyTorch model, so the MLP explained here
is a retrain. An explanation of a model that does not perform like the one
in the ablation report would be an explanation of something else, so steps 4
and 5 do not run unless held-out PR-AUC lands within `PR_AUC_TOLERANCE` of
the reported nested-CV mean -- for both models.

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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "config"))
from settings import Settings, load_settings  # noqa: E402

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
# The MLP's numeric inputs, in the order Phase 4 trained it with.
MLP_NUMERIC_INPUTS = NUMERIC_COLUMNS + ["household_size_outlier"]
FOCUS_COUNTRY = "Uganda"

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


# --- Step 1: the training data ----------------------------------------------


@dataclass(frozen=True)
class TrainingData:
    train: pd.DataFrame
    sha256: str

    @property
    def X_raw(self) -> pd.DataFrame:
        """The 10 raw survey fields, as the adapters expect them."""
        return self.train[RAW_FEATURES]

    @property
    def y(self) -> pd.Series:
        return self.train[TARGET_COLUMN]

    @property
    def y_binary(self) -> np.ndarray:
        return (self.y == "Yes").astype(int).to_numpy()

    @property
    def country(self) -> pd.Series:
        return self.train["country"]


def load_training_data(settings: Settings) -> TrainingData:
    train = DataLoader(settings.paths.raw_train, settings.paths.raw_test).load_train()
    data = TrainingData(train=train, sha256=sha256_of(settings.paths.raw_train))
    print(f"Loaded train: {train.shape}, sha256 {data.sha256[:12]}")
    return data


# --- Step 2: the shipped XGBoost --------------------------------------------


def load_and_check_shipped_xgboost(data: TrainingData, seed: int) -> tuple[XGBoostRawModel, pd.DataFrame, dict]:
    """Loads `models/production/` and confirms three things about it:
    it was trained on this data file; our raw-input adapter encodes rows
    exactly as the training pipeline did; and refitting with the recorded
    hyperparameters reproduces it (so the CV in step 3, which refits with
    those hyperparameters, is evaluating the procedure that made this file).

    Returns the adapter, the pipeline-encoded training matrix, and the checks.
    """
    model = XGBoostRawModel.load(PRODUCTION_DIR)
    if model.metadata["training_data_sha256"] != data.sha256:
        raise VerificationGateError("data/raw/Train_v2.csv is not the file the shipped model was trained on")

    X_encoded = build_xgboost_pipeline().fit_transform(data.train).drop(columns=NON_FEATURE_COLUMNS)
    if not np.array_equal(model.encode(data.X_raw), X_encoded.to_numpy(dtype=np.float32)):
        raise VerificationGateError("XGBoost raw-input adapter does not reproduce the training pipeline's encoding")
    print(f"Loaded {model.name}; adapter encoding matches the pipeline on all {len(data.train)} rows")

    refit = XGBoostModel(seed=seed, **model.metadata["hyperparameters"]).fit(X_encoded, data.y)
    refit_max_diff = float(np.max(np.abs(refit.predict_proba(X_encoded) - model.predict_score(data.X_raw))))
    print(f"Refit-vs-shipped max |score diff|: {refit_max_diff:.2e}")

    checks = {
        "adapter_encoding_matches_pipeline": True,
        "refit_vs_shipped_max_abs_score_diff": refit_max_diff,
        "refit_reproduces_shipped": refit_max_diff <= SCORE_ATOL,
    }
    return model, X_encoded, checks


# --- Step 3: the verification gate ------------------------------------------


@dataclass(frozen=True)
class MLPSetup:
    """Everything needed to build the MLP the way Phase 4 did."""

    pipeline: Pipeline
    X_encoded: pd.DataFrame
    params: dict
    seed: int

    def build(self) -> PyTorchMLP:
        return PyTorchMLP(
            categorical_columns=CATEGORICAL_COLUMNS,
            vocab_sizes=self.pipeline.named_steps["encode"].vocab_sizes_,
            numeric_columns=MLP_NUMERIC_INPUTS,
            epochs=MLP_EPOCHS, seed=self.seed, **self.params,
        )


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


def prepare_mlp(data: TrainingData, phase4: dict, seed: int) -> MLPSetup:
    pipeline = build_pytorch_pipeline().fit(data.train)
    X_encoded = pipeline.transform(data.train).drop(columns=NON_FEATURE_COLUMNS)
    return MLPSetup(pipeline=pipeline, X_encoded=X_encoded, params=mlp_hyperparameters(phase4), seed=seed)


def gate_summary(results: list[CVFoldResult], reported_folds: list[float]) -> dict:
    """Reproduced vs reported PR-AUC for one model, and whether it is within the band."""
    folds = [float(r.metrics["pr_auc"]) for r in results]
    focus_folds = [float(r.country_metrics[FOCUS_COUNTRY]["pr_auc"]) for r in results]
    reported_mean, mean = float(np.mean(reported_folds)), float(np.mean(folds))
    return {
        "reported_fold_pr_auc": reported_folds,
        "reported_mean_pr_auc": reported_mean,
        "fold_pr_auc": folds,
        "mean_pr_auc": mean,
        "std_pr_auc": float(np.std(folds)),
        "abs_diff_of_means": abs(mean - reported_mean),
        "max_abs_fold_diff": float(np.max(np.abs(np.array(folds) - np.array(reported_folds)))),
        f"{FOCUS_COUNTRY.lower()}_fold_pr_auc": focus_folds,
        f"{FOCUS_COUNTRY.lower()}_mean_pr_auc": float(np.mean(focus_folds)),
        "tolerance": PR_AUC_TOLERANCE,
        "within_band": abs(mean - reported_mean) <= PR_AUC_TOLERANCE,
    }


def run_gate(
    data: TrainingData, X_xgb: pd.DataFrame, xgb_params: dict, mlp: MLPSetup, phase4: dict, settings: Settings
) -> tuple[dict, dict]:
    """Held-out 5-fold PR-AUC for each model with its final hyperparameters."""
    cv = dict(n_folds=settings.cv.n_outer_folds, seed=settings.seed)

    print("\n=== Gate: XGBoost, production params, 5-fold CV ===")
    xgb_results = run_stratified_cv(
        X_xgb, data.y, data.country, model_factory=lambda: XGBoostModel(seed=settings.seed, **xgb_params),
        imbalance_strategy=NoImbalanceStrategy(), **cv,
    )
    xgb_gate = gate_summary(xgb_results, phase4["fold_pr_auc"][XGB_ARM_KEY])
    _print_gate(xgb_gate)

    print("=== Gate: MLP, tuned params, class-weight, 5-fold CV ===")
    mlp_results = run_stratified_cv(
        mlp.X_encoded, data.y, data.country, model_factory=mlp.build,
        imbalance_strategy=ClassWeightStrategy(), **cv,
    )
    mlp_gate = gate_summary(mlp_results, phase4["fold_pr_auc"][MLP_ARM_KEY])
    _print_gate(mlp_gate)
    return xgb_gate, mlp_gate


def _print_gate(gate: dict) -> None:
    print(f"  folds {np.round(gate['fold_pr_auc'], 4)} mean {gate['mean_pr_auc']:.4f} "
          f"(reported {gate['reported_mean_pr_auc']:.4f})")


def raise_if_gate_failed(verification: dict, verification_path: Path) -> None:
    failures = [name for name in ("xgboost", "mlp") if not verification[name]["within_band"]]
    if not verification["xgboost"]["refit_reproduces_shipped"]:
        failures.append("xgboost refit does not reproduce the shipped booster")
    if failures:
        raise VerificationGateError(
            f"Verification gate FAILED for {failures}; see {verification_path}. No model or sample set was written."
        )
    print(f"Gate passed (tolerance +/-{PR_AUC_TOLERANCE}). Wrote {verification_path}")


# --- Step 4: the final MLP ---------------------------------------------------


def refit_and_save_mlp(data: TrainingData, mlp: MLPSetup, phase4: dict, model_dir: Path) -> tuple[MLPRawModel, dict]:
    """Fits the MLP on every labelled row (as the shipped XGBoost was), checks
    the raw-input adapter against the training pipeline, and saves it.

    Returns the adapter and the checks.
    """
    print("\n=== Refitting the MLP on the full training set ===")
    X_res, y_res, sample_weight = ClassWeightStrategy().apply(mlp.X_encoded, data.y)
    fitted = mlp.build().fit(X_res, y_res, sample_weight=sample_weight)
    threshold = statistics.median(phase4["thresholds"][MLP_ARM_KEY])
    model = MLPRawModel.from_fitted(
        fitted, mlp.pipeline, decision_threshold=threshold,
        metadata={
            "imbalance_strategy": "class_weight",
            "seed": mlp.seed,
            "training_rows": int(len(data.train)),
            "training_data_sha256": data.sha256,
            "threshold_source": f"median of {MLP_ARM_KEY} per-outer-fold F1-optimal thresholds",
            "hyperparameter_source": f"reports/ablation_raw_results.json best_params_per_fold[{MLP_ARM_KEY}]",
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    )

    cat, num = model.encode(data.X_raw)
    if not (np.array_equal(cat, mlp.X_encoded[CATEGORICAL_COLUMNS].to_numpy(dtype=np.int64))
            and np.array_equal(num, mlp.X_encoded[MLP_NUMERIC_INPUTS].to_numpy(dtype=np.float32))):
        raise VerificationGateError("MLP raw-input adapter does not reproduce the training pipeline's encoding")
    adapter_max_diff = float(np.max(np.abs(model.predict_score(data.X_raw) - fitted.predict_proba(mlp.X_encoded))))
    if adapter_max_diff > SCORE_ATOL:
        raise VerificationGateError(f"MLP adapter scores differ from the model's by {adapter_max_diff:.2e}")

    metadata = model.save(model_dir)
    reloaded = MLPRawModel.load(model_dir)
    reload_max_diff = float(np.max(np.abs(reloaded.predict_logit(data.X_raw) - model.predict_logit(data.X_raw))))
    if reload_max_diff != 0.0:
        raise VerificationGateError(f"Reloaded MLP logits differ from the saved model's by {reload_max_diff:.2e}")
    print(f"Saved {model_dir / 'mlp.pt'} (sha256 {metadata['weights_sha256'][:12]}, threshold {threshold:.4f})")

    checks = {
        "weights_sha256": metadata["weights_sha256"],
        "torch_version": torch.__version__,
        "decision_threshold": threshold,
        "adapter_encoding_matches_pipeline": True,
        "adapter_vs_model_max_abs_score_diff": adapter_max_diff,
        "reload_max_abs_logit_diff": reload_max_diff,
        "unknown_embedding_rows": unknown_embedding_rows(model),
    }
    return model, checks


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


def in_sample_summary(data: TrainingData, xgb_model: XGBoostRawModel, mlp_model: MLPRawModel) -> dict:
    """In-sample, and labelled as such: both final models have seen every row."""
    xgb_scores, mlp_scores = xgb_model.predict_score(data.X_raw), mlp_model.predict_score(data.X_raw)
    return {
        "xgboost_pr_auc": float(average_precision_score(data.y_binary, xgb_scores)),
        "mlp_pr_auc": float(average_precision_score(data.y_binary, mlp_scores)),
        "xgboost_share_predicted_yes": float((xgb_scores >= xgb_model.decision_threshold).mean()),
        "mlp_share_predicted_yes": float((mlp_scores >= mlp_model.decision_threshold).mean()),
    }


# --- Step 5: background and explanation sets --------------------------------


def build_and_save_sample_sets(
    data: TrainingData, xgb_model: XGBoostRawModel, mlp_model: MLPRawModel, settings: Settings
) -> None:
    xai = settings.xai
    sets = build_sample_sets(
        data.country, data.y, background_size=xai.background_size,
        explanation_sample_size=xai.explanation_sample_size, seed=settings.seed,
        training_data_sha256=data.sha256, focus_country=FOCUS_COUNTRY,
    )
    xgb_logit, mlp_logit = xgb_model.predict_logit(data.X_raw), mlp_model.predict_logit(data.X_raw)
    composition = {
        "background": composition_table(sets.background, data.country, data.y),
        "pooled": composition_table(sets.pooled, data.country, data.y),
        "focus": composition_table(sets.focus, data.country, data.y),
        "background_and_pooled_overlap": int(np.intersect1d(sets.background, sets.pooled).size),
        "background_and_focus_overlap": int(np.intersect1d(sets.background, sets.focus).size),
        "positive_rate": {
            "all_training_rows": float(data.y_binary.mean()),
            "background": float(data.y_binary[sets.background].mean()),
            "pooled": float(data.y_binary[sets.pooled].mean()),
            "focus": float(data.y_binary[sets.focus].mean()),
        },
        # How well the background stands in for the population: its mean
        # logit is the base value every interventional attribution starts from.
        "mean_logit": {
            "xgboost": {"all_training_rows": float(xgb_logit.mean()), "background": float(xgb_logit[sets.background].mean())},
            "mlp": {"all_training_rows": float(mlp_logit.mean()), "background": float(mlp_logit[sets.background].mean())},
        },
    }
    path = xai.artifacts_dir / "sample_sets.json"
    sets.save(path, composition)
    print(f"\nBackground {sets.background.size} rows, pooled {sets.pooled.size}, "
          f"{sets.focus_country} {sets.focus.size}, explained in total {sets.explanation.size}")
    print(f"Wrote {path}")


# --- Orchestration ------------------------------------------------------------


def main() -> None:
    settings = load_settings()
    xai = settings.xai
    xai.artifacts_dir.mkdir(parents=True, exist_ok=True)
    verification_path = xai.artifacts_dir / "model_verification.json"
    phase4 = json.loads((settings.paths.reports_dir / "ablation_raw_results.json").read_text(encoding="utf-8"))

    # 1. Data
    data = load_training_data(settings)

    # 2. The shipped XGBoost
    xgb_model, X_xgb, xgb_checks = load_and_check_shipped_xgboost(data, settings.seed)
    xgb_params = xgb_model.metadata["hyperparameters"]

    # 3. Gate: both models must perform as reported before anything is saved
    mlp = prepare_mlp(data, phase4, settings.seed)
    xgb_gate, mlp_gate = run_gate(data, X_xgb, xgb_params, mlp, phase4, settings)
    verification = {
        "training_data_sha256": data.sha256,
        "seed": settings.seed,
        "cv": f"{settings.cv.n_outer_folds}-fold, stratified on country x target, same seed as Phase 4's outer loop",
        "xgboost": {
            **xgb_gate,
            "model_version": xgb_model.metadata["model_version"],
            "hyperparameters": xgb_params,
            "evaluation": "held-out folds, shipped production_xgb_params fixed",
            "caveat": "production params were tuned on the full training set (mildly optimistic); the reported "
                      "number is nested CV with per-fold tuned params, so the folds are not expected to match exactly",
            **xgb_checks,
        },
        "mlp": {
            **mlp_gate,
            "hyperparameters": {**mlp.params, "hidden_dims": list(mlp.params["hidden_dims"]), "epochs": MLP_EPOCHS},
            "hyperparameter_choice": f"all {len(phase4['best_params_per_fold'][MLP_ARM_KEY])} outer folds of "
                                     f"{MLP_ARM_KEY} selected this same configuration",
            "evaluation": "held-out folds, same procedure as the reported nested CV's final refit",
        },
    }
    verification_path.write_text(json.dumps(verification, indent=2), encoding="utf-8")
    raise_if_gate_failed(verification, verification_path)

    # 4. The final MLP
    mlp_model, mlp_checks = refit_and_save_mlp(data, mlp, phase4, xai.model_dir)
    verification["mlp"] |= mlp_checks
    verification["in_sample_not_a_performance_estimate"] = in_sample_summary(data, xgb_model, mlp_model)
    verification_path.write_text(json.dumps(verification, indent=2), encoding="utf-8")

    # 5. Which rows are the reference, and which are explained
    build_and_save_sample_sets(data, xgb_model, mlp_model, settings)


if __name__ == "__main__":
    main()
