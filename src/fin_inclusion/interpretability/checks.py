"""Efficiency / completeness checks on attributions.

The methods used here all promise the same accounting identity under
different names -- "efficiency" for Shapley values, "completeness" for
Integrated Gradients:

    sum_j attribution[i, j] + base_value == model_output[i]

It is the one property of an explanation that can be verified without
ground truth, so it is checked on every run and its error is reported, not
assumed. A violation means the attributions do not explain the output they
claim to: a wrong target space (probability vs logit), a base value from a
different background, too few integration steps, or a model wrapper that
scores differently from the model being explained.

Passing says the attributions add up. It does not say they are faithful to
the model's reasoning, and nothing here bears on causality.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fin_inclusion.interpretability.attribution import Attribution


class EfficiencyError(AssertionError):
    """Attributions do not sum to the model output within tolerance."""


@dataclass(frozen=True)
class EfficiencyReport:
    """Per-run summary of the residual
    `attributions.sum(axis=1) + base_value - model_output`."""

    method: str
    model_name: str
    n_instances: int
    atol: float
    max_abs_error: float
    mean_abs_error: float
    n_violations: int

    @property
    def passed(self) -> bool:
        return self.n_violations == 0

    def raise_if_failed(self) -> None:
        if not self.passed:
            raise EfficiencyError(
                f"{self.method} on {self.model_name}: {self.n_violations}/{self.n_instances} instances "
                f"violate efficiency (max |error| {self.max_abs_error:.3g} > atol {self.atol:.3g})"
            )

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "model_name": self.model_name,
            "n_instances": self.n_instances,
            "atol": self.atol,
            "max_abs_error": self.max_abs_error,
            "mean_abs_error": self.mean_abs_error,
            "n_violations": self.n_violations,
            "passed": self.passed,
        }


def efficiency_residual(attribution: Attribution, model_output: np.ndarray) -> np.ndarray:
    """`attributions.sum(axis=1) + base_value - model_output`, per instance.

    `model_output` must be in `attribution.target_space` (logits, for every
    method in this repo) and row-aligned with the attribution.
    """
    model_output = np.asarray(model_output, dtype=np.float64)
    if model_output.shape != (attribution.n_instances,):
        raise ValueError(
            f"model_output must have shape ({attribution.n_instances},), got {model_output.shape}"
        )
    if not np.isfinite(model_output).all():
        raise ValueError("model_output contains NaN/Inf")
    return attribution.values.sum(axis=1) + attribution.base_values - model_output


def check_efficiency(attribution: Attribution, model_output: np.ndarray, atol: float) -> EfficiencyReport:
    """Measures how closely `attribution` accounts for `model_output`.

    `atol` is an absolute tolerance in target-space units and has no
    default on purpose: what counts as "equal" depends on the method (exact
    Shapley is limited by float precision, sampled KernelSHAP by its
    regression fit, Integrated Gradients by its step count), so each caller
    must state the tolerance it is claiming.
    """
    if not atol > 0:
        raise ValueError(f"atol must be positive, got {atol}")
    abs_error = np.abs(efficiency_residual(attribution, model_output))
    return EfficiencyReport(
        method=attribution.method,
        model_name=attribution.model_name,
        n_instances=attribution.n_instances,
        atol=float(atol),
        max_abs_error=float(abs_error.max(initial=0.0)),
        mean_abs_error=float(abs_error.mean()) if abs_error.size else 0.0,
        n_violations=int((abs_error > atol).sum()),
    )
