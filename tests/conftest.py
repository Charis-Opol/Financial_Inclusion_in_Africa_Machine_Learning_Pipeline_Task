"""Shared synthetic fixtures for the XAI tests (no dependence on data/raw/).

A small survey in the real raw schema, and tiny models trained on it and
written in the same artifact formats as the real ones, so the adapters and
explainers are exercised end to end without the real data or models.
"""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from fin_inclusion.preprocessing.cleaning import (
    DISGUISED_MISSING_VALUES,
    HOUSEHOLD_SIZE_OUTLIER_THRESHOLD,
    UNKNOWN_LABEL,
)
from fin_inclusion.preprocessing.pipeline_factory import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    build_pytorch_pipeline,
    build_xgboost_pipeline,
)

RAW_FEATURES = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS
SURVEY_LEVELS = {
    "country": ["Kenya", "Rwanda", "Tanzania", "Uganda"],
    "location_type": ["Rural", "Urban"],
    "cellphone_access": ["No", "Yes"],
    "gender_of_respondent": ["Female", "Male"],
    "relationship_with_head": ["Child", "Head of Household", "Spouse"],
    "marital_status": ["Dont know", "Married/Living together", "Single/Never Married"],
    "education_level": ["No formal education", "Other/Dont know/RTA", "Primary education"],
    "job_type": ["Dont Know/Refuse to answer", "Farming and Fishing", "Self employed"],
}


def make_survey(n: int = 240, seed: int = 0) -> pd.DataFrame:
    """Raw-schema rows with every level present, an outlier household, and a
    target that depends on the features so fitted models are not constant."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({c: rng.choice(levels, size=n) for c, levels in SURVEY_LEVELS.items()})
    frame["household_size"] = rng.integers(1, 10, size=n)
    frame.loc[0, "household_size"] = 18
    frame["age_of_respondent"] = rng.integers(16, 90, size=n)
    signal = (frame["cellphone_access"] == "Yes").astype(int) + (frame["location_type"] == "Urban").astype(int)
    frame["bank_account"] = np.where(signal + rng.normal(scale=0.6, size=n) > 1.0, "Yes", "No")
    frame["uniqueid"] = [f"uniqueid_{i}" for i in range(n)]
    frame["year"] = 2018
    return frame


def model_features(encoded: pd.DataFrame) -> pd.DataFrame:
    return encoded.drop(columns=["uniqueid", "year", "bank_account"])


@pytest.fixture(scope="session")
def survey() -> pd.DataFrame:
    return make_survey()


@pytest.fixture(scope="session")
def write_xgb_artifact(survey):
    """Factory: trains a small XGBoost on `survey` and writes it in the
    production artifact format. Returns `(model_dir, encoded X, fitted model)`."""
    from fin_inclusion.models.xgboost_model import XGBoostModel

    def write(model_dir, **xgb_params):
        X = model_features(build_xgboost_pipeline().fit_transform(survey))
        model = XGBoostModel(seed=0, **xgb_params).fit(X, survey["bank_account"])
        model.save(model_dir / "model.json")
        metadata = {
            "model_version": "xgb-test",
            "decision_threshold": 0.4,
            "feature_columns": list(X.columns),
            "numeric_columns": NUMERIC_COLUMNS,
            "raw_categorical_levels": {c: sorted(survey[c].unique().tolist()) for c in CATEGORICAL_COLUMNS},
            "disguised_missing_values": DISGUISED_MISSING_VALUES,
            "unknown_label": UNKNOWN_LABEL,
            "household_size_outlier_threshold": HOUSEHOLD_SIZE_OUTLIER_THRESHOLD,
            "model_sha256": hashlib.sha256((model_dir / "model.json").read_bytes()).hexdigest(),
        }
        (model_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        return model_dir, X, model

    return write


@pytest.fixture(scope="session")
def xgb_artifact(write_xgb_artifact, survey, tmp_path_factory):
    model_dir, X, model = write_xgb_artifact(tmp_path_factory.mktemp("xgb_artifact"), n_estimators=20, max_depth=3)
    return model_dir, survey, X, model


@pytest.fixture(scope="session")
def mlp_fitted(survey):
    pytest.importorskip("torch", reason="PyTorch is required for the MLP adapter")
    from fin_inclusion.interpretability.model_adapters import MLPRawModel
    from fin_inclusion.models.pytorch_mlp import PyTorchMLP

    pipeline = build_pytorch_pipeline().fit(survey)
    X = model_features(pipeline.transform(survey))
    numeric_inputs = NUMERIC_COLUMNS + ["household_size_outlier"]
    model = PyTorchMLP(
        categorical_columns=CATEGORICAL_COLUMNS,
        vocab_sizes=pipeline.named_steps["encode"].vocab_sizes_,
        numeric_columns=numeric_inputs,
        hidden_dims=(8, 4),
        embedding_dim=3,
        dropout=0.3,
        epochs=3,
        batch_size=32,
        seed=0,
    ).fit(X, survey["bank_account"])
    adapter = MLPRawModel.from_fitted(model, pipeline, decision_threshold=0.6, metadata={"seed": 0})
    return adapter, model, survey, X, numeric_inputs
