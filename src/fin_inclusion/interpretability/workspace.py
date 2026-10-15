"""Loads everything an XAI script needs, with the checks done once.

Every analysis script (SHAP, Captum, figures, comparison) and the dashboard
start from the same four things: the training rows, the two verified models,
and the fixed sample sets. Loading them here means each script cannot
accidentally pair a model with a different data file or a different set of
rows than the others used.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fin_inclusion.data.loader import TARGET_COLUMN, DataLoader
from fin_inclusion.interpretability.model_adapters import (
    ArtifactError,
    MLPRawModel,
    RawFeatureModel,
    XGBoostRawModel,
    sha256_of,
)
from fin_inclusion.interpretability.sampling import SampleSets

SAMPLE_SETS_FILE = "sample_sets.json"


@dataclass(frozen=True)
class Workspace:
    train: pd.DataFrame
    xgboost: XGBoostRawModel
    mlp: MLPRawModel
    sets: SampleSets

    @property
    def models(self) -> dict[str, RawFeatureModel]:
        """Both models under short, stable keys used in artifact file names."""
        return {"xgboost": self.xgboost, "mlp": self.mlp}

    @property
    def raw_features(self) -> tuple[str, ...]:
        return self.xgboost.raw_features

    @property
    def X_raw(self) -> pd.DataFrame:
        """The 10 raw survey fields for every training row."""
        return self.train[list(self.raw_features)]

    @property
    def y_binary(self) -> np.ndarray:
        return (self.train[TARGET_COLUMN] == "Yes").astype(int).to_numpy()

    @property
    def country(self) -> np.ndarray:
        return self.train["country"].to_numpy()

    @property
    def background(self) -> pd.DataFrame:
        """The reference rows every interventional method averages over."""
        return self.X_raw.iloc[self.sets.background]

    def rows(self, positions: np.ndarray) -> pd.DataFrame:
        return self.X_raw.iloc[positions]

    @property
    def sample_masks(self) -> dict[str, np.ndarray]:
        """Boolean masks over the explanation set: the pooled sample, and the
        focus country (key: its lower-case name). Pooled statistics use only
        the pooled sample (decisions log D2.6)."""
        rows = self.sets.explanation
        return {"pooled": np.isin(rows, self.sets.pooled),
                self.sets.focus_country.lower(): np.isin(rows, self.sets.focus)}


def load_workspace(
    raw_train_path: Path, raw_test_path: Path, production_dir: Path, xai_model_dir: Path, artifacts_dir: Path
) -> Workspace:
    """Loads data, models and sample sets, refusing any mismatch between them."""
    train = DataLoader(raw_train_path, raw_test_path).load_train()
    data_sha = sha256_of(raw_train_path)
    xgboost, mlp = XGBoostRawModel.load(production_dir), MLPRawModel.load(xai_model_dir)
    for model in (xgboost, mlp):
        if model.metadata["training_data_sha256"] != data_sha:
            raise ArtifactError(f"{model.name} was trained on a different file than {raw_train_path}")
    if xgboost.raw_features != mlp.raw_features:
        raise ArtifactError("The two models disagree on the raw feature list/order")
    if xgboost.categorical_levels != mlp.categorical_levels:
        raise ArtifactError("The two models disagree on the categorical vocabulary")
    sets = SampleSets.load(artifacts_dir / SAMPLE_SETS_FILE, training_data_sha256=data_sha)
    return Workspace(train=train, xgboost=xgboost, mlp=mlp, sets=sets)
