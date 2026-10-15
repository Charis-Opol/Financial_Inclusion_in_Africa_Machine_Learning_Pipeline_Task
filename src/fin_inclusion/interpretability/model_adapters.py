"""Raw-input -> logit adapters for the two models being explained.

Every explanation method needs to ask the same question of a model: "what is
your output for these respondents?", where a respondent is the 10 raw survey
fields of the serving contract. The two models answer through different
encodings (40 one-hot/numeric columns for XGBoost; 8 embedding indices + 3
numerics for the MLP), so each adapter owns its model's encoding and exposes
one method, `predict_logit(X_raw)`. Perturbation-based explainers (exact
Shapley, KernelSHAP, Feature Ablation) are then written once, against
`RawFeatureModel`, and work on either model.

Outputs are logits (the pre-sigmoid margin), the space every attribution in
this repo is computed in. The sigmoid of that logit is a ranking score, not
a calibrated probability: both models were trained with class weighting.

Loading is strict in the same way as `serving/predictor.py`: a checksum that
does not match its metadata raises, rather than explaining whatever file
happens to be on disk.
"""
from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost

from fin_inclusion.serving.encoder import FeatureEncoder, UnknownCategoryError

MLP_WEIGHTS_FILE = "mlp.pt"
MLP_METADATA_FILE = "mlp_metadata.json"

UNKNOWN_CODE = -1
"""A categorical code meaning "a category the model never saw". It exists
only so an "unknown" BASELINE can be expressed in codes: XGBoost encodes it
as an all-zero one-hot block, the MLP as embedding index 0 -- exactly what
each training pipeline does with an unseen category. `to_codes` never
produces it: real input with an unseen category is still an error."""


class ArtifactError(RuntimeError):
    """A model artifact is missing, corrupt, or inconsistent with its metadata."""


def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sigmoid(logit: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logit))


def _require_columns(X_raw: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [c for c in columns if c not in X_raw.columns]
    if missing:
        raise KeyError(f"Raw input is missing feature column(s) {missing}")


def _lookup(X_raw: pd.DataFrame, column: str, table: Mapping[str, int]) -> np.ndarray:
    """Maps a raw categorical column through `table`, refusing unseen values.

    The training pipeline routes an unseen category to an "unknown" encoding
    and warns; an explanation of that silently mis-encoded row would be an
    explanation of the wrong input, so here it is an error.
    """
    codes = X_raw[column].map(table)
    if codes.isna().any():
        unseen = sorted(X_raw.loc[codes.isna(), column].astype(str).unique())
        raise UnknownCategoryError(f"{column} has value(s) outside the training vocabulary: {unseen}")
    return codes.to_numpy(dtype=np.int64)


class RawFeatureModel(ABC):
    """A fitted binary classifier addressed in raw survey features."""

    name: str
    raw_features: tuple[str, ...]
    categorical_levels: Mapping[str, tuple[str, ...]]
    """Raw (as-collected) levels of each categorical, i.e. the values a
    perturbation may legitimately put in that column."""
    decision_threshold: float
    """Score cut-off for a "Yes" prediction, on the sigmoid scale. Model-
    specific: each model's own median per-fold F1-optimal threshold."""

    # --- "Codes": raw rows as one plain number per raw feature ---------------
    #
    # Perturbation methods (exact Shapley, KernelSHAP, Feature Ablation) build
    # millions of hybrid rows by swapping single fields between a respondent
    # and a reference row. Doing that on DataFrames of strings is slow, so a
    # raw row is also representable as a float vector with one entry per raw
    # feature: a categorical holds the position of its level in
    # `categorical_levels[column]`, a numeric holds its value. Swapping an
    # entry between two code vectors is exactly swapping that survey field.

    def to_codes(self, X_raw: pd.DataFrame) -> np.ndarray:
        """`[n, n_raw_features]` float64 codes for raw rows, in `raw_features` order."""
        _require_columns(X_raw, self.raw_features)
        columns = []
        for feature in self.raw_features:
            if feature in self.categorical_levels:
                table = {level: i for i, level in enumerate(self.categorical_levels[feature])}
                columns.append(_lookup(X_raw, feature, table).astype(np.float64))
            else:
                columns.append(X_raw[feature].to_numpy(dtype=np.float64))
        return np.column_stack(columns)

    def from_codes(self, codes: np.ndarray) -> pd.DataFrame:
        """Inverse of `to_codes`: codes back to readable raw rows."""
        codes = np.asarray(codes)
        frame = {}
        for j, feature in enumerate(self.raw_features):
            if feature in self.categorical_levels:
                frame[feature] = np.asarray(self.categorical_levels[feature], dtype=object)[codes[:, j].astype(np.int64)]
            else:
                frame[feature] = codes[:, j]
        return pd.DataFrame(frame)

    @abstractmethod
    def predict_logit_codes(self, codes: np.ndarray) -> np.ndarray:
        """Pre-sigmoid model output for `[n, n_raw_features]` codes, one float64 per row."""

    def predict_logit(self, X_raw: pd.DataFrame) -> np.ndarray:
        """Pre-sigmoid model output, one float64 per row of `X_raw`."""
        return self.predict_logit_codes(self.to_codes(X_raw))

    def predict_score(self, X_raw: pd.DataFrame) -> np.ndarray:
        """Sigmoid of the logit: a ranking score in [0, 1], not a calibrated probability."""
        return _sigmoid(self.predict_logit(X_raw))


class XGBoostRawModel(RawFeatureModel):
    """The shipped XGBoost booster, driven by its own `metadata.json`."""

    def __init__(self, booster: xgboost.Booster, metadata: Mapping):
        self.booster = booster
        self.metadata = dict(metadata)
        self.name = f"xgboost ({metadata['model_version']})"
        self.feature_columns: tuple[str, ...] = tuple(metadata["feature_columns"])
        self.categorical_columns: tuple[str, ...] = tuple(metadata["raw_categorical_levels"])
        self.numeric_columns: tuple[str, ...] = tuple(metadata["numeric_columns"])
        self.raw_features = (*self.categorical_columns, *self.numeric_columns)
        self.categorical_levels = {c: tuple(v) for c, v in metadata["raw_categorical_levels"].items()}
        self.decision_threshold = float(metadata["decision_threshold"])

        # The serving encoder already derives (and validates) the raw-value ->
        # one-hot-slot table from the metadata; reuse it so the explained
        # model and the served model cannot encode differently.
        encoder = FeatureEncoder.from_metadata(metadata)
        self._numeric_index = dict(encoder.numeric_index)
        self._outlier_index = encoder.outlier_index
        self._outlier_threshold = encoder.household_size_outlier_threshold
        # Per categorical: one-hot column position of each level, indexed by level code.
        self._slot = {
            column: np.array([encoder.onehot_index[(column, level)] for level in levels], dtype=np.int64)
            for column, levels in self.categorical_levels.items()
        }
        self._feature_position = {feature: j for j, feature in enumerate(self.raw_features)}

    @classmethod
    def load(cls, model_dir: Path) -> "XGBoostRawModel":
        model_path, metadata_path = Path(model_dir) / "model.json", Path(model_dir) / "metadata.json"
        for path in (model_path, metadata_path):
            if not path.is_file():
                raise ArtifactError(f"Missing artifact file {path} -- run scripts/train_production_model.py")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        actual_sha = sha256_of(model_path)
        if actual_sha != metadata["model_sha256"]:
            raise ArtifactError(
                f"model.json checksum {actual_sha[:12]} does not match metadata ({metadata['model_sha256'][:12]})"
            )
        booster = xgboost.Booster()
        booster.load_model(str(model_path))
        if list(booster.feature_names or []) != metadata["feature_columns"]:
            raise ArtifactError("Booster feature names/order differ from metadata.feature_columns")
        return cls(booster, metadata)

    def encode(self, X_raw: pd.DataFrame) -> np.ndarray:
        """`[n, 40]` float32 matrix in the booster's column order. Vectorised
        equivalent of `FeatureEncoder.encode_batch` (parity is tested), since
        perturbation methods score millions of rows."""
        return self.encode_codes(self.to_codes(X_raw))

    def encode_codes(self, codes: np.ndarray) -> np.ndarray:
        codes = np.asarray(codes, dtype=np.float64)
        position = self._feature_position
        matrix = np.zeros((len(codes), len(self.feature_columns)), dtype=np.float32)
        for column, index in self._numeric_index.items():
            matrix[:, index] = codes[:, position[column]]
        # The outlier flag is recomputed from whatever household_size the row
        # now holds, so a perturbed household_size never keeps a stale flag.
        matrix[:, self._outlier_index] = codes[:, position["household_size"]] > self._outlier_threshold
        for column in self.categorical_columns:
            level = codes[:, position[column]].astype(np.int64)
            known = level != UNKNOWN_CODE  # unknown: leave the whole block at zero
            matrix[np.flatnonzero(known), self._slot[column][level[known]]] = 1.0
        if not np.isfinite(matrix).all():
            raise ValueError("Encoded feature matrix contains NaN/Inf")
        return matrix

    def predict_logit_encoded(self, matrix: np.ndarray) -> np.ndarray:
        margin = self.booster.inplace_predict(matrix, predict_type="margin", validate_features=False)
        return np.asarray(margin, dtype=np.float64)

    def predict_logit_codes(self, codes: np.ndarray) -> np.ndarray:
        return self.predict_logit_encoded(self.encode_codes(codes))


class MLPRawModel(RawFeatureModel):
    """The PyTorch MLP plus the fitted preprocessing state it was trained behind.

    The state (category -> embedding index, numeric mean/scale) is held as
    plain dicts rather than a pickled sklearn `Pipeline`, so the artifact is
    JSON + weights and the encoding is vectorised.
    """

    def __init__(self, model, preprocessing: Mapping, decision_threshold: float, metadata: Mapping | None = None):
        self.model = model
        self.preprocessing = dict(preprocessing)
        self.metadata = dict(metadata or {})
        self.name = "pytorch_mlp"
        self.categorical_columns: tuple[str, ...] = tuple(model.categorical_columns)
        self.numeric_inputs: tuple[str, ...] = tuple(model.numeric_columns)
        self.numeric_columns: tuple[str, ...] = tuple(preprocessing["numeric_mean"])
        self.raw_features = (*self.categorical_columns, *self.numeric_columns)
        self.decision_threshold = float(decision_threshold)

        disguised = preprocessing["disguised_missing_values"]
        unknown = preprocessing["unknown_label"]
        self._index: dict[str, dict[str, int]] = {}
        levels: dict[str, tuple[str, ...]] = {}
        for column in self.categorical_columns:
            cleaned = dict(preprocessing["category_maps"][column])
            # Mirror MissingnessRecoder: the raw placeholder answer is what a
            # respondent actually gives; it shares the `unknown` level's index.
            raw_table = {level: index for level, index in cleaned.items() if level != unknown}
            if column in disguised:
                if unknown not in cleaned:
                    raise ArtifactError(f"{column}: no {unknown!r} level for placeholder {disguised[column]!r}")
                raw_table[disguised[column]] = cleaned[unknown]
            self._index[column] = raw_table
            levels[column] = tuple(sorted(raw_table))
        self.categorical_levels = levels
        # Per categorical: embedding index of each level, indexed by level code.
        self._embedding_index = {
            column: np.array([self._index[column][level] for level in levels[column]], dtype=np.int64)
            for column in self.categorical_columns
        }
        self._feature_position = {feature: j for j, feature in enumerate(self.raw_features)}

    @classmethod
    def from_fitted(cls, model, pipeline, decision_threshold: float, metadata: Mapping | None = None) -> "MLPRawModel":
        """Captures the state of a fitted `build_pytorch_pipeline()` pipeline."""
        from fin_inclusion.preprocessing.cleaning import (
            DISGUISED_MISSING_VALUES,
            HOUSEHOLD_SIZE_OUTLIER_THRESHOLD,
            UNKNOWN_LABEL,
        )

        scale_step, encode_step = pipeline.named_steps["scale"], pipeline.named_steps["encode"]
        preprocessing = {
            "category_maps": {c: {str(k): int(v) for k, v in m.items()} for c, m in encode_step.category_maps_.items()},
            "numeric_mean": dict(zip(scale_step.columns, map(float, scale_step.scaler_.mean_))),
            "numeric_scale": dict(zip(scale_step.columns, map(float, scale_step.scaler_.scale_))),
            "disguised_missing_values": dict(DISGUISED_MISSING_VALUES),
            "unknown_label": UNKNOWN_LABEL,
            "unknown_index": int(encode_step.UNKNOWN_INDEX),
            "household_size_outlier_threshold": HOUSEHOLD_SIZE_OUTLIER_THRESHOLD,
        }
        return cls(model, preprocessing, decision_threshold, metadata)

    @property
    def net(self):
        """The underlying `nn.Module` (embeddings + MLP), for gradient-based
        attribution. Returns logits; callers must put it in `eval()` mode."""
        if self.model._net is None:
            raise ArtifactError("The MLP has not been fitted or loaded")
        return self.model._net

    def encode(self, X_raw: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """`(cat [n, 8] int64 embedding indices, num [n, 3] float32)`, in the
        column orders `_MLPNet.forward` expects."""
        return self.encode_codes(self.to_codes(X_raw))

    def encode_codes(self, codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        codes = np.asarray(codes, dtype=np.float64)
        position = self._feature_position
        unknown_index = int(self.preprocessing["unknown_index"])
        columns = []
        for c in self.categorical_columns:
            level = codes[:, position[c]].astype(np.int64)
            # UNKNOWN_CODE -> the reserved never-seen index; known codes -> their embedding row.
            columns.append(np.where(level == UNKNOWN_CODE, unknown_index,
                                    self._embedding_index[c][np.maximum(level, 0)]))
        cat = np.column_stack(columns)
        mean, scale = self.preprocessing["numeric_mean"], self.preprocessing["numeric_scale"]
        threshold = self.preprocessing["household_size_outlier_threshold"]
        numeric = []
        for name in self.numeric_inputs:
            if name == "household_size_outlier":
                # Recomputed from the row's current household_size (never stale).
                numeric.append((codes[:, position["household_size"]] > threshold).astype(np.float64))
            else:
                numeric.append((codes[:, position[name]] - mean[name]) / scale[name])
        return cat, np.column_stack(numeric).astype(np.float32)

    def predict_logit_codes(self, codes: np.ndarray) -> np.ndarray:
        import torch

        cat, num = self.encode_codes(codes)
        net = self.net
        net.eval()  # dropout off: explanations must be of the deterministic model
        with torch.no_grad():
            logit = net(torch.from_numpy(cat), torch.from_numpy(num))
        return logit.numpy().astype(np.float64)

    def save(self, model_dir: Path) -> dict:
        """Writes `mlp.pt` and `mlp_metadata.json` (preprocessing state,
        hyperparameters, threshold, weights checksum). Returns the metadata."""
        import torch

        model_dir = Path(model_dir)
        weights_path = model_dir / MLP_WEIGHTS_FILE
        self.model.save(weights_path)
        params = self.model.get_params()
        metadata = {
            **self.metadata,
            "model_family": "pytorch_mlp",
            "torch_version": torch.__version__,
            "hyperparameters": {**params, "hidden_dims": list(params["hidden_dims"])},
            "categorical_columns": list(self.categorical_columns),
            "numeric_inputs": list(self.numeric_inputs),
            "vocab_sizes": dict(self.model.vocab_sizes),
            "decision_threshold": self.decision_threshold,
            "preprocessing": self.preprocessing,
            "weights_sha256": sha256_of(weights_path),
        }
        (model_dir / MLP_METADATA_FILE).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self.metadata = metadata
        return metadata

    @classmethod
    def load(cls, model_dir: Path) -> "MLPRawModel":
        from fin_inclusion.models.pytorch_mlp import PyTorchMLP

        weights_path, metadata_path = Path(model_dir) / MLP_WEIGHTS_FILE, Path(model_dir) / MLP_METADATA_FILE
        for path in (weights_path, metadata_path):
            if not path.is_file():
                raise ArtifactError(f"Missing artifact file {path} -- run scripts/run_xai_models.py")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        # Checked BEFORE loading: `PyTorchMLP.load` unpickles the checkpoint,
        # so only a file this repo wrote (and recorded) is ever opened.
        actual_sha = sha256_of(weights_path)
        if actual_sha != metadata["weights_sha256"]:
            raise ArtifactError(
                f"mlp.pt checksum {actual_sha[:12]} does not match metadata ({metadata['weights_sha256'][:12]})"
            )
        model = PyTorchMLP.load(weights_path)
        return cls(model, metadata["preprocessing"], metadata["decision_threshold"], metadata)
