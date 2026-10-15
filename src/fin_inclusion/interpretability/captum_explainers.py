"""Captum explainers: Integrated Gradients, Feature Ablation, and layer/neuron attribution.

Unlike the SHAP methods, which average over a background set, every method
here compares a respondent with ONE reference point, the baseline. Two
baselines are used throughout, because the answer depends on the choice and
that dependence is itself a result:

  - "unknown": every categorical set to the category the model never saw
    (MLP: embedding index 0; XGBoost: an all-zero one-hot block);
  - "modal": every categorical set to its most frequent training level.

Numeric features take their training mean in both.

What each method measures (all in logits):

  - Integrated Gradients (MLP only): moves the input in a straight line from
    the baseline to the respondent and adds up the gradient along the way.
    Category indices cannot be moved continuously, so the line is drawn
    between their EMBEDDING VECTORS (LayerIntegratedGradients on each
    embedding layer) and between the scaled numeric inputs. Contributions
    add up to f(respondent) - f(baseline) up to a "convergence delta" that
    shrinks with more integration steps.
  - Feature Ablation (both models): replaces one raw feature at a time with
    its baseline value and records how much the logit drops. Each feature is
    removed ALONE -- no averaging over which other features are present --
    so contributions need not add up to anything.
  - Layer attribution (MLP only): which hidden units the output flows
    through. Hidden units have no meaning of their own; this describes the
    network's mechanics, not anything about respondents.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import torch
from captum.attr import FeatureAblation, LayerConductance, LayerGradientXActivation, LayerIntegratedGradients
from torch import nn

from fin_inclusion.interpretability.attribution import LOGIT_SPACE, Attribution
from fin_inclusion.interpretability.base import Explainer
from fin_inclusion.interpretability.grouping import FeatureGrouping, group_attribution
from fin_inclusion.interpretability.model_adapters import UNKNOWN_CODE, MLPRawModel, RawFeatureModel

BASELINES = ("unknown", "modal")


def baseline_codes(model: RawFeatureModel, training_codes: np.ndarray, kind: str) -> np.ndarray:
    """The baseline as a code vector (one entry per raw feature).

    Categoricals: `UNKNOWN_CODE`, or the most frequent training level.
    Numerics: the training mean (for the MLP that is 0 after scaling).
    """
    if kind not in BASELINES:
        raise ValueError(f"baseline must be one of {BASELINES}, got {kind!r}")
    codes = np.empty(len(model.raw_features))
    for j, feature in enumerate(model.raw_features):
        column = training_codes[:, j]
        if feature not in model.categorical_levels:
            codes[j] = column.mean()
        elif kind == "unknown":
            codes[j] = UNKNOWN_CODE
        else:
            codes[j] = np.bincount(column.astype(np.int64)).argmax()  # ties -> lowest code
    return codes


def describe_baseline(model: RawFeatureModel, codes: np.ndarray, kind: str) -> str:
    if kind == "unknown":
        categorical = "every categorical set to a never-seen category"
    else:
        levels = [f"{f}={model.categorical_levels[f][int(c)]}" for f, c in zip(model.raw_features, codes)
                  if f in model.categorical_levels]
        categorical = "every categorical at its most frequent training level (" + ", ".join(levels) + ")"
    numerics = ", ".join(f"{f}={c:.2f}" for f, c in zip(model.raw_features, codes) if f not in model.categorical_levels)
    return f"{kind} baseline: {categorical}; numerics at the training mean ({numerics})"


# --- Views of the MLP that Captum can differentiate ----------------------------


class CategoricalInputNet(nn.Module):
    """`_MLPNet.forward(cat, num)`, unchanged, except that the numeric inputs
    pass through a named identity layer. That gives LayerIntegratedGradients a
    layer to integrate the numerics on, alongside the embedding layers. Shares
    the original modules (and weights); no copy is made."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.embeddings = net.embeddings
        self.numeric = nn.Identity()
        self.mlp = net.mlp

    def forward(self, cat: torch.Tensor, num: torch.Tensor) -> torch.Tensor:
        parts = [embedding(cat[:, i]) for i, embedding in enumerate(self.embeddings)]
        parts.append(self.numeric(num))
        return self.mlp(torch.cat(parts, dim=1)).squeeze(-1)


class EmbeddedInputNet(nn.Module):
    """The MLP after the embedding lookup: input is the 51-dim vector of
    concatenated embeddings and numerics. Everything here is continuous, so
    gradient methods can interpolate on it directly."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.mlp = net.mlp

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.mlp(z).squeeze(-1)


def mlp_tensors(model: MLPRawModel, codes: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    cat, num = model.encode_codes(np.atleast_2d(codes))
    return torch.from_numpy(cat), torch.from_numpy(num)


def embed(model: MLPRawModel, codes: np.ndarray) -> torch.Tensor:
    """Codes -> the `[n, 51]` input of the MLP's first Linear layer."""
    cat, num = mlp_tensors(model, codes)
    with torch.no_grad():
        parts = [embedding(cat[:, i]) for i, embedding in enumerate(model.net.embeddings)]
        return torch.cat([*parts, num], dim=1)


def hidden_layers(net: nn.Module) -> dict[str, nn.Module]:
    """The ReLU outputs of the MLP, i.e. the hidden units' activations."""
    relus = [module for module in net.mlp if isinstance(module, nn.ReLU)]
    return {f"hidden_{i + 1}": relu for i, relu in enumerate(relus)}


# --- Integrated Gradients --------------------------------------------------------


class IntegratedGradientsExplainer(Explainer):
    """LayerIntegratedGradients on the 8 embedding layers + the numeric inputs, jointly."""

    method = "integrated_gradients"

    def __init__(self, model: MLPRawModel, baseline: np.ndarray, baseline_name: str, n_steps: int):
        self.model, self.baseline, self.baseline_name, self.n_steps = model, baseline, baseline_name, n_steps
        net = model.net
        net.eval()  # dropout off: attributions must be of the deterministic model
        self.wrapper = CategoricalInputNet(net)
        self._ig = LayerIntegratedGradients(self.wrapper, [*self.wrapper.embeddings, self.wrapper.numeric])
        self.grouping = FeatureGrouping.for_embeddings(
            model.categorical_columns, net.embeddings[0].embedding_dim, model.numeric_inputs
        )
        self.base_value = float(model.predict_logit_codes(baseline[None])[0])
        self.description = describe_baseline(model, baseline, baseline_name)

    def explain_with_delta(self, X_raw: pd.DataFrame) -> tuple[Attribution, np.ndarray]:
        """Attributions over the 51 embedding dimensions and numeric inputs
        (before grouping), and each row's convergence delta:
        sum(attributions) - (f(x) - f(baseline))."""
        cat, num = mlp_tensors(self.model, self.model.to_codes(X_raw))
        base_cat, base_num = mlp_tensors(self.model, self.baseline)
        layer_attributions, delta = self._ig.attribute(
            (cat, num),
            baselines=(base_cat.expand_as(cat), base_num.expand_as(num)),
            n_steps=self.n_steps,
            return_convergence_delta=True,
        )
        values = torch.cat([a.detach() for a in layer_attributions], dim=1).double().numpy()
        attribution = Attribution(
            values=values,
            base_value=self.base_value,
            feature_names=self.grouping.encoded_names,
            method=self.method,
            model_name=self.model.name,
            target_space=LOGIT_SPACE,
            baseline_description=f"{self.description}; straight-line path in embedding space, "
                                 f"{self.n_steps} Gauss-Legendre steps",
            metadata={"baseline": self.baseline_name, "n_steps": self.n_steps},
        )
        return attribution, delta.detach().double().numpy()

    def explain(self, X_raw: pd.DataFrame) -> Attribution:
        encoded, delta = self.explain_with_delta(X_raw)
        grouped = group_attribution(encoded, self.grouping)
        return replace(grouped, metadata={
            **grouped.metadata,
            "max_abs_convergence_delta": float(np.abs(delta).max()),
            "mean_abs_convergence_delta": float(np.abs(delta).mean()),
        })


# --- Feature Ablation --------------------------------------------------------------


class FeatureAblationExplainer(Explainer):
    """Captum FeatureAblation on the raw-feature wrapper of any model.

    The feature mask is the 10 raw features, so ablating `education_level`
    replaces the whole categorical (all its one-hot columns, or its whole
    embedding) at once, and ablating `household_size` also resets the
    outlier flag derived from it.
    """

    method = "feature_ablation"

    def __init__(self, model: RawFeatureModel, baseline: np.ndarray, baseline_name: str):
        self.model, self.baseline, self.baseline_name = model, baseline, baseline_name
        self.base_value = float(model.predict_logit_codes(baseline[None])[0])
        self.description = describe_baseline(model, baseline, baseline_name)
        self._ablation = FeatureAblation(self._forward)

    def _forward(self, codes: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(self.model.predict_logit_codes(codes.detach().numpy()))

    def explain(self, X_raw: pd.DataFrame) -> Attribution:
        codes = torch.from_numpy(self.model.to_codes(X_raw))
        n_features = codes.shape[1]
        values = self._ablation.attribute(
            codes,
            baselines=torch.from_numpy(self.baseline[None]),
            feature_mask=torch.arange(n_features)[None],
            perturbations_per_eval=n_features,
        )
        return Attribution(
            values=values.double().numpy(),
            base_value=self.base_value,
            feature_names=self.model.raw_features,
            method=self.method,
            model_name=self.model.name,
            target_space=LOGIT_SPACE,
            baseline_description=f"{self.description}; each raw feature replaced ALONE, value = f(x) - "
                                 "f(x with that feature at the baseline); not an additive decomposition",
            metadata={"baseline": self.baseline_name},
        )


# --- Layer and neuron attribution ---------------------------------------------------


def hidden_unit_attributions(
    model: MLPRawModel, X_raw: pd.DataFrame, baseline: np.ndarray, n_steps: int
) -> dict[str, dict[str, np.ndarray]]:
    """Per hidden layer: each unit's activation, LayerConductance (how much of
    f(x) - f(baseline) flows through the unit) with its convergence delta, and
    LayerGradientXActivation (activation x local gradient; no baseline).

    Returns `{layer name: {"activation", "conductance", "conductance_delta",
    "gradient_x_activation"}}`, each `[n, units]` except the delta `[n]`.
    """
    net = model.net
    net.eval()
    wrapper = EmbeddedInputNet(net)
    z = embed(model, model.to_codes(X_raw))
    z_baseline = embed(model, baseline).expand_as(z)
    results: dict[str, dict[str, np.ndarray]] = {}
    for name, layer in hidden_layers(net).items():
        captured: list[torch.Tensor] = []
        hook = layer.register_forward_hook(lambda _m, _i, output: captured.append(output.detach()))
        with torch.no_grad():
            wrapper(z)
        hook.remove()
        conductance, delta = LayerConductance(wrapper, layer).attribute(
            z, baselines=z_baseline, n_steps=n_steps, return_convergence_delta=True
        )
        gradient_x_activation = LayerGradientXActivation(wrapper, layer).attribute(z)
        results[name] = {
            "activation": captured[0].double().numpy(),
            "conductance": conductance.detach().double().numpy(),
            "conductance_delta": delta.detach().double().numpy(),
            "gradient_x_activation": gradient_x_activation.detach().double().numpy(),
        }
    return results


def unit_names(layer: str, n_units: int) -> tuple[str, ...]:
    return tuple(f"{layer}[unit{u:03d}]" for u in range(n_units))
