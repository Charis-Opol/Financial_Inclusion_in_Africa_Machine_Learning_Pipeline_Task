"""The one result type every explanation method returns.

SHAP, Integrated Gradients and Feature Ablation come from different
libraries with different output objects. Comparing them (Task 4) is only
meaningful if they are reduced to the same thing first: a matrix of
per-instance, per-feature contributions plus the reference those
contributions are measured against. `Attribution` is that common form, and
it carries the facts a number is meaningless without -- which model, which
output space, which baseline -- so they cannot be separated from the values
when an artifact is saved, reloaded, plotted or quoted in the report.

An attribution describes what the MODEL learned to rely on. It is not a
causal statement about what drives bank account ownership.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

LOGIT_SPACE = "logit"
"""Every method in this repo explains the pre-sigmoid margin. Attributions
are additive there; they are not additive in probability, and the
probability itself is an uncalibrated ranking score (class-weighted
training), so a "probability points" reading would mislead twice over."""

_FORMAT_VERSION = 1


@dataclass(frozen=True, eq=False)
class Attribution:
    """Additive feature contributions for `n` instances.

    `values[i, j]` is feature `j`'s contribution to instance `i`'s model
    output in `target_space`, relative to `base_value`. For a method that
    satisfies efficiency/completeness, `values[i].sum() + base_value` equals
    the model output for instance `i` (`checks.check_efficiency` verifies it
    rather than assuming it).

    `base_value` is the model output at the reference: the mean output over
    the background set for interventional SHAP, the output at the baseline
    input for Integrated Gradients and Feature Ablation. It is a scalar when
    one reference serves every instance, or an array of length `n` when the
    reference differs per instance.

    `baseline_description` says in words what that reference is and which
    value function was used (interventional vs conditional); `metadata`
    holds anything else worth keeping with the numbers (seed, n_steps,
    convergence deltas, the background set's row ids) and must be
    JSON-serialisable so the artifact is self-describing on disk.
    """

    values: np.ndarray
    base_value: float | np.ndarray
    feature_names: tuple[str, ...]
    method: str
    model_name: str
    target_space: str
    baseline_description: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError(f"values must be 2-D [n_instances, n_features], got shape {values.shape}")
        feature_names = tuple(self.feature_names)
        if len(feature_names) != values.shape[1]:
            raise ValueError(
                f"{len(feature_names)} feature names for {values.shape[1]} attribution columns"
            )
        if len(set(feature_names)) != len(feature_names):
            raise ValueError(f"feature names are not unique: {feature_names}")
        if not np.isfinite(values).all():
            raise ValueError(f"{self.method} attributions for {self.model_name} contain NaN/Inf")

        base_value = np.asarray(self.base_value, dtype=np.float64)
        if base_value.ndim == 0:
            base_value = float(base_value)
        elif base_value.shape != (values.shape[0],):
            raise ValueError(
                f"base_value must be a scalar or have shape ({values.shape[0]},), got {base_value.shape}"
            )
        if not np.isfinite(base_value).all():
            raise ValueError("base_value contains NaN/Inf")

        for name in ("method", "model_name", "target_space", "baseline_description"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be stated; an attribution without it cannot be interpreted")

        # frozen=True blocks plain assignment; normalising here means every
        # consumer sees the same dtypes however the explainer built the object.
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "base_value", base_value)
        object.__setattr__(self, "feature_names", feature_names)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def n_instances(self) -> int:
        return self.values.shape[0]

    @property
    def n_features(self) -> int:
        return self.values.shape[1]

    @property
    def base_values(self) -> np.ndarray:
        """`base_value` broadcast to one entry per instance."""
        return np.broadcast_to(np.asarray(self.base_value, dtype=np.float64), (self.n_instances,))

    def to_frame(self, index: pd.Index | None = None) -> pd.DataFrame:
        return pd.DataFrame(self.values, columns=list(self.feature_names), index=index)

    def select(self, rows: np.ndarray) -> "Attribution":
        """The same attribution restricted to `rows` (boolean mask or integer
        positions) -- how a Uganda-only view is taken from a pooled run
        without re-explaining anything."""
        rows = np.asarray(rows)
        base_value = self.base_value if np.ndim(self.base_value) == 0 else self.base_value[rows]
        return replace(self, values=self.values[rows], base_value=base_value)

    def save(self, path: Path) -> None:
        """Writes one `.npz` holding the values and, as JSON, everything
        needed to interpret them. No pickle: the artifact can be loaded
        without executing code."""
        path = Path(path)
        if path.suffix != ".npz":
            raise ValueError(f"Attribution artifacts are .npz files, got {path.name!r}")
        try:
            description = json.dumps(
                {
                    "format_version": _FORMAT_VERSION,
                    "feature_names": list(self.feature_names),
                    "method": self.method,
                    "model_name": self.model_name,
                    "target_space": self.target_space,
                    "baseline_description": self.baseline_description,
                    "metadata": self.metadata,
                }
            )
        except TypeError as exc:
            raise TypeError(f"Attribution.metadata must be JSON-serialisable: {exc}") from exc
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, values=self.values, base_value=np.asarray(self.base_value), description=np.array(description)
        )

    @classmethod
    def load(cls, path: Path) -> "Attribution":
        with np.load(Path(path), allow_pickle=False) as archive:
            description = json.loads(str(archive["description"]))
            values, base_value = archive["values"], archive["base_value"]
        if description["format_version"] != _FORMAT_VERSION:
            raise ValueError(
                f"{path} has attribution format {description['format_version']}, expected {_FORMAT_VERSION}"
            )
        return cls(
            values=values,
            base_value=base_value,
            feature_names=tuple(description["feature_names"]),
            method=description["method"],
            model_name=description["model_name"],
            target_space=description["target_space"],
            baseline_description=description["baseline_description"],
            metadata=description["metadata"],
        )
