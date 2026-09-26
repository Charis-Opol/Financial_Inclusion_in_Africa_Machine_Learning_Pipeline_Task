"""Loads the model artifact once and turns validated records into predictions.

Loading is where every "this artifact can't be trusted" condition is caught
-- missing files, a checksum mismatch, a feature list that disagrees with the
metadata, an API schema that disagrees with the training vocabulary. Each
raises `ModelLoadError` with a specific message, so a bad deploy surfaces as
an unhealthy service with a clear reason in the logs, never as a service
that starts up and quietly scores inputs wrong.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xgboost

from fin_inclusion.serving.encoder import FeatureContractError, FeatureEncoder
from fin_inclusion.serving.schemas import SurveyRecord, categorical_literals

logger = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    """The artifact is missing, corrupt, or inconsistent with this service's code."""


class InferenceError(RuntimeError):
    """The model failed or produced an unusable output for a valid input."""


@dataclass(frozen=True)
class Prediction:
    probability: float
    label: str
    confidence: float
    warnings: list[str]


def check_schema_matches_artifact(metadata: dict) -> None:
    """Fails if the API's allowed category values differ from the model's vocabulary.

    Values the schema allows but the model never saw would be encoded as
    all-zero one-hot rows (silent mis-scoring); values the model knows but the
    schema lacks would be wrongly rejected. Either way, refuse to start.
    """
    schema_levels = categorical_literals()
    artifact_levels = {col: set(levels) for col, levels in metadata["raw_categorical_levels"].items()}
    if schema_levels != artifact_levels:
        diff = {
            col: {
                "schema_only": sorted(schema_levels.get(col, set()) - artifact_levels.get(col, set())),
                "model_only": sorted(artifact_levels.get(col, set()) - schema_levels.get(col, set())),
            }
            for col in schema_levels.keys() | artifact_levels.keys()
            if schema_levels.get(col) != artifact_levels.get(col)
        }
        raise ModelLoadError(f"API schema and model vocabulary disagree: {diff}")


class Predictor:
    def __init__(self, booster: xgboost.Booster, encoder: FeatureEncoder, metadata: dict):
        self._booster = booster
        self._encoder = encoder
        self.metadata = metadata
        self.model_version: str = metadata["model_version"]
        self.threshold: float = float(metadata["decision_threshold"])
        self._positive_label: str = metadata["positive_label"]
        self._max_seen = {col: hi for col, (_, hi) in metadata["numeric_observed_range"].items()}

    @classmethod
    def load(cls, model_dir: Path, inference_threads: int = 1) -> "Predictor":
        model_path, metadata_path = model_dir / "model.json", model_dir / "metadata.json"
        for path in (model_path, metadata_path):
            if not path.is_file():
                raise ModelLoadError(
                    f"Missing artifact file {path.resolve()} -- run scripts/train_production_model.py"
                )

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        actual_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if actual_sha != metadata["model_sha256"]:
            raise ModelLoadError(
                f"model.json checksum {actual_sha[:12]} does not match metadata "
                f"({metadata['model_sha256'][:12]}); files are from different training runs"
            )

        booster = xgboost.Booster()
        booster.load_model(str(model_path))
        booster.set_param({"nthread": inference_threads})

        if list(booster.feature_names or []) != metadata["feature_columns"]:
            raise ModelLoadError("Booster feature names/order differ from metadata.feature_columns")
        check_schema_matches_artifact(metadata)
        try:
            encoder = FeatureEncoder.from_metadata(metadata)
        except FeatureContractError as exc:
            raise ModelLoadError(str(exc)) from exc

        predictor = cls(booster, encoder, metadata)
        predictor._smoke_test()
        logger.info(
            "Loaded model %s (%d features, threshold %.4f, xgboost %s)",
            predictor.model_version, encoder.n_features, predictor.threshold, xgboost.__version__,
        )
        return predictor

    def _smoke_test(self) -> None:
        """Scores one synthetic row so a broken booster fails at startup, not on the first user."""
        levels = self.metadata["raw_categorical_levels"]
        record = {col: values[0] for col, values in levels.items()}
        record |= {col: lo for col, (lo, _) in self.metadata["numeric_observed_range"].items()}
        try:
            self._score(self._encoder.encode_batch([record]))
        except InferenceError as exc:
            raise ModelLoadError(f"Startup smoke test failed: {exc}") from exc

    def _score(self, features: np.ndarray) -> np.ndarray:
        # validate_features=False: column order was verified once at load
        # time, and the encoder always emits that order, so re-checking names
        # on every call would be redundant work in the hot path.
        try:
            scores = self._booster.inplace_predict(features, validate_features=False)
        except xgboost.core.XGBoostError as exc:
            raise InferenceError(f"XGBoost prediction failed: {exc}") from exc
        if scores.shape != (features.shape[0],):
            raise InferenceError(f"Expected {features.shape[0]} scores, got shape {scores.shape}")
        if not np.isfinite(scores).all():
            raise InferenceError("Model produced non-finite scores")
        return scores

    def predict(self, record: SurveyRecord) -> Prediction:
        """Scores one validated record. CPU-bound -- call from a worker thread."""
        raw = record.model_dump()
        try:
            features = self._encoder.encode_batch([raw])
        except ValueError as exc:
            # The schema already vetted the input, so an encoding failure means
            # the service itself is inconsistent -> server error, not a 422.
            raise InferenceError(f"Feature encoding failed for a schema-valid record: {exc}") from exc

        probability = float(self._score(features)[0])
        is_positive = probability >= self.threshold
        return Prediction(
            probability=probability,
            label=self._positive_label if is_positive else "No",
            confidence=probability if is_positive else 1.0 - probability,
            warnings=self._out_of_distribution_warnings(raw),
        )

    def _out_of_distribution_warnings(self, raw: dict) -> list[str]:
        return [
            f"{col}={raw[col]} exceeds the training maximum of {hi}; prediction is an extrapolation"
            for col, hi in self._max_seen.items()
            if raw[col] > hi
        ]
