"""SHAP-library explainers, each returning raw-feature `Attribution`s.

Three explainers from the `shap` package, all measured against the same
background rows and all in logit space:

  - `TreeShapExplainer` (XGBoost): TreeSHAP with interventional perturbation.
    Exact and fast, but its players are the 40 ENCODED columns; the result
    is summed back to the 10 raw features.
  - `KernelShapExplainer` (any model): model-agnostic, plays the game
    directly on the 10 raw features. With enough samples to enumerate all
    2^10 coalitions it is exact; with fewer it is an estimate.
  - `GradientShapExplainer` (MLP): expected gradients in the MLP's embedding
    space. A sampled approximation, so efficiency holds only approximately.

All three use the INTERVENTIONAL value function: a feature is "absent" by
taking its value from a background row, independently of the other
features. `shapley.py` computes the same quantity by brute force and is the
check on these.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from fin_inclusion.interpretability.attribution import LOGIT_SPACE, Attribution
from fin_inclusion.interpretability.base import Explainer
from fin_inclusion.interpretability.grouping import FeatureGrouping, group_attribution
from fin_inclusion.interpretability.model_adapters import MLPRawModel, RawFeatureModel, XGBoostRawModel


class TreeShapExplainer(Explainer):
    """Interventional TreeSHAP on the shipped booster, grouped to raw features."""

    method = "tree_shap"

    def __init__(self, model: XGBoostRawModel, background_raw: pd.DataFrame):
        self.model = model
        self.n_background = len(background_raw)
        self.grouping = FeatureGrouping.for_onehot(
            model.feature_columns, model.categorical_columns, model.numeric_columns
        )
        # Passed as an explicit masker with max_samples = the full background:
        # given a bare array, TreeExplainer silently subsamples it to 100
        # rows, which would give TreeSHAP a different reference (and base
        # value) from every other method here.
        background = shap.maskers.Independent(model.encode(background_raw), max_samples=self.n_background)
        self._explainer = shap.TreeExplainer(
            model.booster,
            data=background,
            feature_perturbation="interventional",
            model_output="raw",  # the margin, i.e. the logit
        )

    def explain_encoded(self, X_raw: pd.DataFrame) -> Attribution:
        """Attributions over the 40 encoded columns, before grouping."""
        values = self._explainer.shap_values(self.model.encode(X_raw), check_additivity=True)
        return Attribution(
            values=values,
            base_value=float(np.ravel(self._explainer.expected_value)[0]),
            feature_names=self.model.feature_columns,
            method=self.method,
            model_name=self.model.name,
            target_space=LOGIT_SPACE,
            baseline_description=(
                f"mean logit over {self.n_background} background rows; interventional value function; "
                f"players are the {len(self.model.feature_columns)} encoded columns, summed to raw features"
            ),
            metadata={"n_background": self.n_background, "players": "encoded columns"},
        )

    def explain(self, X_raw: pd.DataFrame) -> Attribution:
        return group_attribution(self.explain_encoded(X_raw), self.grouping)


class KernelShapExplainer(Explainer):
    """KernelSHAP on the raw-feature wrapper of any model.

    `nsamples` is the number of coalitions evaluated per instance. With M
    raw features there are 2^M - 2 non-trivial coalitions; at or above that,
    `shap` enumerates them all and the result is the exact Shapley value.
    `nsamples=None` asks for exactly that.

    `l1_reg=False`: shap's default applies a feature-selection step to the
    regression, which zeroes small attributions. With 10 features there is
    nothing to gain from it and it would bias the comparison with the oracle.
    """

    method = "kernel_shap"

    def __init__(self, model: RawFeatureModel, background_raw: pd.DataFrame, nsamples: int | None = None, seed: int = 42):
        self.model = model
        self.background_codes = model.to_codes(background_raw)
        self.n_coalitions = 2 ** len(model.raw_features) - 2
        self.nsamples = self.n_coalitions if nsamples is None else int(nsamples)
        self.seed = seed
        self._explainer = shap.KernelExplainer(model.predict_logit_codes, self.background_codes)

    @property
    def is_exact(self) -> bool:
        return self.nsamples >= self.n_coalitions

    def explain(self, X_raw: pd.DataFrame) -> Attribution:
        # KernelExplainer draws coalitions from numpy's global RNG and offers
        # no seed argument, so the global state is set (and restored) here.
        state = np.random.get_state()
        np.random.seed(self.seed)
        try:
            values = self._explainer.shap_values(
                self.model.to_codes(X_raw), nsamples=self.nsamples, l1_reg=False, silent=True
            )
        finally:
            np.random.set_state(state)
        coverage = "all coalitions enumerated (exact)" if self.is_exact else f"{self.nsamples} sampled coalitions"
        return Attribution(
            values=values,
            base_value=float(np.ravel(self._explainer.expected_value)[0]),
            feature_names=self.model.raw_features,
            method=self.method,
            model_name=self.model.name,
            target_space=LOGIT_SPACE,
            baseline_description=(
                f"mean logit over {len(self.background_codes)} background rows; interventional value "
                f"function on raw features; {coverage}"
            ),
            metadata={
                "n_background": len(self.background_codes), "nsamples": self.nsamples,
                "exact": self.is_exact, "seed": self.seed, "players": "raw features",
            },
        )


class GradientShapExplainer(Explainer):
    """Expected gradients (`shap.GradientExplainer`) on the MLP.

    Embedding indices are integers and have no gradient, so the explained
    function is the MLP *after* the embedding lookup: its input is the
    51-dim vector of concatenated embeddings and numeric inputs. Each
    background row is embedded the same way and serves as a reference point;
    the 51 attributions are then summed per raw feature.

    This is a Monte Carlo estimate (`nsamples` random reference/interpolation
    draws per instance), so attributions sum to the output only
    approximately; the efficiency error is reported, not hidden.
    """

    method = "gradient_shap"

    def __init__(self, model: MLPRawModel, background_raw: pd.DataFrame, nsamples: int = 200, seed: int = 42):
        self.model = model
        self.nsamples = nsamples
        self.seed = seed
        self.n_background = len(background_raw)
        net = model.net
        net.eval()
        self.grouping = FeatureGrouping.for_embeddings(
            model.categorical_columns, net.embeddings[0].embedding_dim, model.numeric_inputs
        )
        self._background = self._embed(background_raw)
        self._explainer = shap.GradientExplainer(net.mlp, self._background)
        import torch

        with torch.no_grad():
            self._base_value = float(net.mlp(self._background).mean())

    def _embed(self, X_raw: pd.DataFrame):
        """Raw rows -> the `[n, 51]` tensor `_MLPNet.forward` feeds to its first Linear layer."""
        import torch

        cat, num = self.model.encode(X_raw)
        cat, num = torch.from_numpy(cat), torch.from_numpy(num)
        with torch.no_grad():
            parts = [emb(cat[:, i]) for i, emb in enumerate(self.model.net.embeddings)]
            return torch.cat([*parts, num], dim=1)

    def explain(self, X_raw: pd.DataFrame) -> Attribution:
        values = self._explainer.shap_values(self._embed(X_raw), nsamples=self.nsamples, rseed=self.seed)
        if isinstance(values, list):
            values = values[0]
        values = np.asarray(values, dtype=np.float64).reshape(len(X_raw), self.grouping.n_encoded)
        encoded = Attribution(
            values=values,
            base_value=self._base_value,
            feature_names=self.grouping.encoded_names,
            method=self.method,
            model_name=self.model.name,
            target_space=LOGIT_SPACE,
            baseline_description=(
                f"mean logit over {self.n_background} background rows; expected gradients in embedding "
                f"space ({self.nsamples} reference/interpolation draws per instance), summed to raw features"
            ),
            metadata={"n_background": self.n_background, "nsamples": self.nsamples, "seed": self.seed},
        )
        return group_attribution(encoded, self.grouping)
