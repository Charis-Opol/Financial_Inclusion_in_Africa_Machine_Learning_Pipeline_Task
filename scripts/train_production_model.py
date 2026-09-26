"""Refits the shipped model (XGBoost, class-weight arm) and writes the serving artifact.

Phase 4 (`run_tuning_and_ablation.py`) tuned `production_xgb_params` and
refit the winning arm on the full training set, but only kept that model in
memory for SHAP -- nothing was persisted. This script reproduces that exact
refit (same params, same seed, no re-tuning) and writes:

    models/production/model.json     XGBoost booster (native JSON format)
    models/production/metadata.json  everything the service needs to encode
                                     inputs and interpret outputs without
                                     importing pandas/sklearn

Before saving, it proves the numpy-only serving encoder reproduces the real
sklearn preprocessing pipeline bit-for-bit on every training row, and that
booster predictions on the encoder's output match the training-time model's.
If either check fails, nothing is written.
"""
from __future__ import annotations

import hashlib
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xgboost

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "config"))
from settings import load_settings  # noqa: E402

from fin_inclusion.data.loader import TARGET_COLUMN, DataLoader  # noqa: E402
from fin_inclusion.models.xgboost_model import XGBoostModel  # noqa: E402
from fin_inclusion.preprocessing.cleaning import (  # noqa: E402
    DISGUISED_MISSING_VALUES,
    HOUSEHOLD_SIZE_OUTLIER_THRESHOLD,
    UNKNOWN_LABEL,
)
from fin_inclusion.preprocessing.pipeline_factory import (  # noqa: E402
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    build_xgboost_pipeline,
)
from fin_inclusion.serving.encoder import FeatureEncoder  # noqa: E402

# Same exclusion list Phase 4 trained with: `year` is fully determined by
# `country` (one survey year per country), `uniqueid` is an identifier.
NON_FEATURE_COLUMNS = ["uniqueid", "year", TARGET_COLUMN]
ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "models" / "production"
WINNING_ARM_KEY = "xgb_cw"


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_phase4_results(reports_dir: Path) -> tuple[dict, float]:
    """Returns (production hyperparameters, deployment decision threshold).

    The threshold is the median of the per-outer-fold F1-optimal thresholds,
    as recommended in ablation_study_report.md §3.4 -- any single fold's value
    is too noisy (0.34-0.75 spread) to trust on its own.
    """
    results = json.loads((reports_dir / "ablation_raw_results.json").read_text(encoding="utf-8"))
    if results["overall_best_key"] != WINNING_ARM_KEY:
        raise RuntimeError(
            f"Phase 4 winner is {results['overall_best_key']!r}, not {WINNING_ARM_KEY!r}; "
            "this script only knows how to refit the XGBoost class-weight arm"
        )
    threshold = statistics.median(results["thresholds"][WINNING_ARM_KEY])
    return results["production_xgb_params"], threshold


def verify_encoder_parity(encoder: FeatureEncoder, raw_records: list[dict], X_pipeline: np.ndarray) -> None:
    encoded = encoder.encode_batch(raw_records)
    if encoded.shape != X_pipeline.shape or not np.array_equal(encoded, X_pipeline):
        mismatched_rows = np.flatnonzero((encoded != X_pipeline).any(axis=1))
        raise AssertionError(
            f"Serving encoder diverges from the training pipeline on {mismatched_rows.size} rows "
            f"(first: {mismatched_rows[:5].tolist()}); refusing to write the artifact"
        )


def main() -> None:
    settings = load_settings()
    params, threshold = load_phase4_results(settings.paths.reports_dir)

    train = DataLoader(settings.paths.raw_train, settings.paths.raw_test).load_train()
    X = build_xgboost_pipeline().fit_transform(train).drop(columns=NON_FEATURE_COLUMNS)
    y = train[TARGET_COLUMN]
    print(f"Training on {X.shape[0]} rows x {X.shape[1]} features with params {params}")

    # Class-weight arm: imbalance is handled by `scale_pos_weight` inside
    # `params` (NoImbalanceStrategy), so no sample_weight here -- matches Phase 4.
    model = XGBoostModel(seed=settings.seed, **params).fit(X, y)

    metadata = {
        "model_family": "xgboost",
        "imbalance_strategy": "class_weight",
        "xgboost_version": xgboost.__version__,
        "hyperparameters": params,
        "seed": settings.seed,
        "decision_threshold": threshold,
        "threshold_source": "median of per-outer-fold F1-optimal thresholds (ablation_study_report.md §3.4)",
        "positive_label": "Yes",
        "target": TARGET_COLUMN,
        "feature_columns": list(X.columns),
        "numeric_columns": NUMERIC_COLUMNS,
        "raw_categorical_levels": {c: sorted(train[c].unique().tolist()) for c in CATEGORICAL_COLUMNS},
        "disguised_missing_values": DISGUISED_MISSING_VALUES,
        "unknown_label": UNKNOWN_LABEL,
        "household_size_outlier_threshold": HOUSEHOLD_SIZE_OUTLIER_THRESHOLD,
        "numeric_observed_range": {
            c: [int(train[c].min()), int(train[c].max())] for c in NUMERIC_COLUMNS
        },
        "training_rows": int(len(train)),
        "training_data_sha256": sha256_of(settings.paths.raw_train),
    }

    raw_records = train[CATEGORICAL_COLUMNS + NUMERIC_COLUMNS].to_dict(orient="records")
    X_pipeline = X.to_numpy(dtype=np.float32)
    encoder = FeatureEncoder.from_metadata(metadata)
    verify_encoder_parity(encoder, raw_records, X_pipeline)
    print(f"Encoder parity OK on all {len(raw_records)} training rows")

    # The service calls the raw booster (no sklearn wrapper); make sure that
    # path scores identically to the wrapper the model was trained through.
    wrapper_proba = model.predict_proba(X)
    booster_proba = model.booster.get_booster().inplace_predict(encoder.encode_batch(raw_records))
    np.testing.assert_allclose(booster_proba, wrapper_proba, rtol=0, atol=1e-7)
    print("Booster-on-encoded-input parity OK")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACT_DIR / "model.json"
    model.save(model_path)
    model_sha = sha256_of(model_path)
    metadata |= {
        # Content-addressed: the same params + data + library version always
        # produce the same version string, and any change produces a new one.
        "model_version": f"xgb-cw-{model_sha[:12]}",
        "model_sha256": model_sha,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (ARTIFACT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {model_path} and metadata.json (version {metadata['model_version']}, threshold {threshold:.4f})")


if __name__ == "__main__":
    main()
