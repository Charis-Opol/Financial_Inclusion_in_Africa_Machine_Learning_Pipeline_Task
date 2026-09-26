"""Builds the drift-check reference profile from the model's own training data.

    python -m fin_inclusion.monitoring.reference_profile [--data data/raw/Train_v2.csv]

Writes `models/production/reference_profile.json`. Run it once after
`scripts/train_production_model.py`; the drift check refuses to run against
a profile built for a different model version.

Why per-country segments as well as a pooled one: training pools four
national surveys, but a deployment may serve one country. Ugandan-only
traffic compared against the pooled reference would show huge "drift" on
`country` and every country-correlated feature on day one -- a false alarm.
`drift_check --country Uganda` compares Ugandan traffic to Ugandan training
rows instead.

The reference also stores the distribution of model *scores* on the
training data: score drift can reveal a shift spread thinly across many
features that no single-feature PSI crosses a threshold for.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fin_inclusion.monitoring import psi
from fin_inclusion.serving.predictor import Predictor
from fin_inclusion.serving.raw_records import read_raw_records

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCORE_FEATURE = "prediction_score"
POOLED_SEGMENT = "all"


def build_segment_profile(records: list[dict], scores: np.ndarray, metadata: dict, exclude: set[str]) -> dict:
    features: dict[str, dict] = {}
    for column in metadata["numeric_columns"]:
        values = np.array([r[column] for r in records], dtype=np.float64)
        edges = psi.decile_edges(values)
        features[column] = {"kind": "numeric", "edges": edges, "proportions": psi.numeric_proportions(values, edges).tolist()}
    for column, levels in metadata["raw_categorical_levels"].items():
        if column in exclude:
            continue
        values = [r[column] for r in records]
        features[column] = {"kind": "categorical", "levels": levels, "proportions": psi.categorical_proportions(values, levels).tolist()}
    score_edges = psi.decile_edges(scores)
    features[SCORE_FEATURE] = {"kind": "numeric", "edges": score_edges, "proportions": psi.numeric_proportions(scores, score_edges).tolist()}
    return {
        "n": len(records),
        "positive_rate": float(np.mean(scores >= metadata["decision_threshold"])),
        "features": features,
    }


def build_reference_profile(predictor: Predictor, data_path: Path) -> dict:
    metadata = predictor.metadata
    data_sha = hashlib.sha256(data_path.read_bytes()).hexdigest()
    if data_sha != metadata["training_data_sha256"]:
        raise ValueError(
            f"{data_path} is not the file model {predictor.model_version} was trained on "
            "(sha256 mismatch); a reference built from other data would make every PSI meaningless"
        )

    records = read_raw_records(data_path, metadata["raw_categorical_levels"], metadata["numeric_columns"])
    scores = predictor.booster.inplace_predict(predictor.encoder.encode_batch(records), validate_features=False)

    segments = {POOLED_SEGMENT: build_segment_profile(records, scores, metadata, exclude=set())}
    countries = np.array([r["country"] for r in records])
    for country in metadata["raw_categorical_levels"]["country"]:
        mask = countries == country
        segment_records = [r for r, keep in zip(records, mask) if keep]
        segments[country] = build_segment_profile(segment_records, scores[mask], metadata, exclude={"country"})

    return {
        "model_version": predictor.model_version,
        "training_data_sha256": data_sha,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "psi_thresholds": {"stable_below": psi.PSI_STABLE_BELOW, "significant_above": psi.PSI_SIGNIFICANT_ABOVE},
        "segments": segments,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the PSI reference profile from training data.")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models" / "production")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "raw" / "Train_v2.csv")
    args = parser.parse_args()

    predictor = Predictor.load(args.model_dir)
    profile = build_reference_profile(predictor, args.data)
    out = args.model_dir / "reference_profile.json"
    out.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    sizes = {name: seg["n"] for name, seg in profile["segments"].items()}
    print(f"Wrote {out} for model {profile['model_version']}; segment sizes {sizes}")


if __name__ == "__main__":
    main()
