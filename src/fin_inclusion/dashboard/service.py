"""Everything the dashboard computes, with no Streamlit in it.

The UI (`app.py`) only lays out widgets and charts; validation, scoring and
explaining happen here, so they can be tested without a browser and reused
from a notebook.

Inputs are validated with the serving API's own `SurveyRecord` schema, so the
dashboard accepts exactly what `/predict` accepts: the same 10 fields, the
same category spellings, the same numeric ranges. An unseen category is an
error, never silently encoded as "unknown".

Local explanations are computed live for one respondent; global views come
from the artifacts the analysis scripts saved, so the dashboard never shows
a number the report cannot trace.
"""
from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from fin_inclusion.interpretability.attribution import Attribution
from fin_inclusion.interpretability.model_adapters import RawFeatureModel

MODEL_LABELS = {"xgboost": "XGBoost (shipped model)", "mlp": "PyTorch MLP (comparison model)"}
BASELINE_LABELS = {
    "modal": "Modal: every answer at its most common value",
    "unknown": "Unknown: every category never seen in training",
}
LIMITATIONS = (
    "**Read these explanations with care.**\n\n"
    "- They describe what the **model** learned to rely on, not what causes someone to have a bank account.\n"
    "- The score is a **ranking score, not a calibrated probability** (both models were trained with class "
    "weighting).\n"
    "- **Uganda is a small segment** (2,101 of 23,524 training rows, 181 account holders); its results are the "
    "least certain.\n"
    "- Every number depends on the **reference** it is measured against; switching the baseline can reorder a "
    "single respondent's explanation.\n"
    "- The models were trained on four national surveys from 2016-2018 pooled together."
)


@dataclass(frozen=True)
class MethodSpec:
    label: str
    models: tuple[str, ...]
    uses_baseline: bool
    description: str


METHODS: dict[str, MethodSpec] = {
    "tree_shap": MethodSpec(
        "TreeSHAP", ("xgboost",), False,
        "Shapley values computed from the trees, compared with 200 reference respondents.",
    ),
    "exact_shap": MethodSpec(
        "Exact SHAP", ("xgboost", "mlp"), False,
        "Shapley values computed exactly over every combination of the 10 fields, compared with 200 reference "
        "respondents. The ground truth the other SHAP methods approximate.",
    ),
    "integrated_gradients": MethodSpec(
        "Integrated Gradients", ("mlp",), True,
        "Adds up the MLP's gradient along a straight path from one baseline respondent to this one.",
    ),
    "feature_ablation": MethodSpec(
        "Feature Ablation", ("xgboost", "mlp"), True,
        "Resets one field at a time to the baseline and measures how far the score moves. Not additive.",
    ),
}


class InvalidRecord(ValueError):
    """The input does not satisfy the serving API's schema."""

    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


def validate_record(raw: Mapping[str, object]) -> dict:
    """Validates one respondent exactly as the API does; returns the clean record."""
    from pydantic import ValidationError

    from fin_inclusion.serving.schemas import SurveyRecord

    try:
        return SurveyRecord(**dict(raw)).model_dump()
    except ValidationError as exc:
        raise InvalidRecord(
            [f"{'.'.join(map(str, e['loc'])) or 'record'}: {e['msg']}" for e in exc.errors()]
        ) from None


def allowed_values() -> dict[str, tuple[str, ...]]:
    """The API's allowed values for every categorical field, in a stable order."""
    from fin_inclusion.serving.schemas import categorical_literals

    return {name: tuple(sorted(values)) for name, values in categorical_literals().items()}


@dataclass(frozen=True)
class LocalExplanation:
    model_key: str
    method: str
    baseline: str | None
    record: dict
    logit: float
    score: float
    threshold: float
    attribution: Attribution
    warnings: list[str] = field(default_factory=list)

    @property
    def predicted_yes(self) -> bool:
        return self.score >= self.threshold

    def table(self) -> pd.DataFrame:
        """One row per field: its value and contribution, largest |contribution| first."""
        frame = pd.DataFrame({
            "field": list(self.attribution.feature_names),
            # as text: one column mixing numbers and category names cannot be displayed as a table
            "value": [str(self.record[f]) for f in self.attribution.feature_names],
            "contribution": self.attribution.values[0],
        })
        return frame.reindex(frame["contribution"].abs().sort_values(ascending=False).index).reset_index(drop=True)


class ExplanationService:
    """Scores and explains one respondent at a time with either model."""

    def __init__(
        self,
        models: Mapping[str, RawFeatureModel],
        background: pd.DataFrame,
        baselines: Mapping[str, Mapping[str, np.ndarray]],
        ig_steps: int,
        training_max: Mapping[str, float],
    ):
        self.models = dict(models)
        self.background = background
        self.baselines = baselines
        self.ig_steps = ig_steps
        self.training_max = dict(training_max)
        self._explainers: dict[tuple, object] = {}

    @classmethod
    def from_workspace(cls, ws, ig_steps: int) -> "ExplanationService":
        from fin_inclusion.interpretability.captum_explainers import BASELINES, baseline_codes

        training_codes = ws.mlp.to_codes(ws.X_raw)
        baselines = {key: {kind: baseline_codes(model, training_codes, kind) for kind in BASELINES}
                     for key, model in ws.models.items()}
        training_max = {name: float(ws.X_raw[name].max()) for name in ("household_size", "age_of_respondent")}
        return cls(ws.models, ws.background, baselines, ig_steps, training_max)

    def methods_for(self, model_key: str) -> list[str]:
        return [name for name, spec in METHODS.items() if model_key in spec.models]

    def _explainer(self, model_key: str, method: str, baseline: str | None):
        key = (model_key, method, baseline if METHODS[method].uses_baseline else None)
        if key not in self._explainers:
            model = self.models[model_key]
            if method == "tree_shap":
                from fin_inclusion.interpretability.shap_explainers import TreeShapExplainer

                explainer = TreeShapExplainer(model, self.background)
            elif method == "exact_shap":
                from fin_inclusion.interpretability.shapley import ExactShapleyExplainer

                explainer = ExactShapleyExplainer(model, self.background)
            elif method == "integrated_gradients":
                from fin_inclusion.interpretability.captum_explainers import IntegratedGradientsExplainer

                explainer = IntegratedGradientsExplainer(model, self.baselines[model_key][baseline], baseline,
                                                         self.ig_steps)
            else:
                from fin_inclusion.interpretability.captum_explainers import FeatureAblationExplainer

                explainer = FeatureAblationExplainer(model, self.baselines[model_key][baseline], baseline)
            self._explainers[key] = explainer
        return self._explainers[key]

    def explain(self, raw: Mapping[str, object], model_key: str, method: str, baseline: str = "modal") -> LocalExplanation:
        if model_key not in self.models:
            raise KeyError(f"Unknown model {model_key!r}; choose from {list(self.models)}")
        if method not in self.methods_for(model_key):
            raise ValueError(f"{METHODS[method].label} is not available for {MODEL_LABELS[model_key]}")
        record = validate_record(raw)
        model = self.models[model_key]
        X = pd.DataFrame([record])[list(model.raw_features)]
        logit = float(model.predict_logit(X)[0])
        attribution = self._explainer(model_key, method, baseline).explain(X)
        warnings = [f"{name} = {record[name]} is above the training maximum ({self.training_max[name]:g}); the "
                    f"score is an extrapolation" for name in self.training_max if record[name] > self.training_max[name]]
        return LocalExplanation(
            model_key=model_key, method=method, baseline=baseline if METHODS[method].uses_baseline else None,
            record=record, logit=logit, score=float(1 / (1 + np.exp(-logit))),
            threshold=model.decision_threshold, attribution=attribution, warnings=warnings,
        )

    def what_if(self, raw: Mapping[str, object], field_name: str, value: object, model_key: str, method: str,
                baseline: str = "modal") -> tuple[LocalExplanation, LocalExplanation]:
        """The explanation before and after changing one field."""
        before = self.explain(raw, model_key, method, baseline)
        after = self.explain({**before.record, field_name: value}, model_key, method, baseline)
        return before, after

    def compare_methods(self, raw: Mapping[str, object], model_key: str, baseline: str = "modal") -> pd.DataFrame:
        """This respondent's contribution per field under every method available for the model."""
        columns = {}
        for method in self.methods_for(model_key):
            explanation = self.explain(raw, model_key, method, baseline)
            columns[METHODS[method].label] = explanation.attribution.values[0]
        names = list(self.models[model_key].raw_features)
        return pd.DataFrame(columns, index=names)


# --- Saved, global artifacts ------------------------------------------------------------


class SavedResults:
    """Read-only access to what the analysis scripts wrote."""

    def __init__(self, artifacts_dir: Path, figures_dir: Path):
        self.artifacts_dir, self.figures_dir = Path(artifacts_dir), Path(figures_dir)
        required = ["shap_summary.json", "captum_summary.json", "internals_summary.json", "comparison_table.csv"]
        missing = [name for name in required if not (self.artifacts_dir / name).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing analysis artifacts {missing} in {self.artifacts_dir}; run the scripts/run_xai_*.py "
                "scripts first (see src/fin_inclusion/interpretability/README.md)"
            )
        self._internals = json.loads((self.artifacts_dir / "internals_summary.json").read_text(encoding="utf-8"))
        self.shap = json.loads((self.artifacts_dir / "shap_summary.json").read_text(encoding="utf-8"))

    def importance_shares(self, model_key: str) -> pd.DataFrame:
        """Share of importance per field, every saved method for this model, pooled and Uganda."""
        suffix = " · XGB" if model_key == "xgboost" else " · MLP"
        global_ = self._internals["global_importance_all_methods"]
        rows = []
        for sample in ("pooled", "uganda"):
            for feature, by_method in global_[sample]["shares"].items():
                for method, share in by_method.items():
                    if method.endswith(suffix):
                        rows.append({"sample": sample, "field": feature, "method": method[: -len(suffix)],
                                     "share": share})
        return pd.DataFrame(rows)

    def importance_with_ci(self, model_key: str) -> pd.DataFrame:
        """Mean |attribution| with 95% bootstrap intervals for the model's main SHAP method."""
        key = "tree_shap__xgboost" if model_key == "xgboost" else "kernel_shap__mlp"
        rows = []
        for sample, block in self.shap["global_importance"][key].items():
            for feature, values in block["features"].items():
                rows.append({"sample": sample, "field": feature, "mean_abs": values["mean_abs"],
                             "ci_low": values["ci_low"], "ci_high": values["ci_high"], "n": block["n"]})
        return pd.DataFrame(rows)

    def comparison_table(self) -> pd.DataFrame:
        with open(self.artifacts_dir / "comparison_table.csv", encoding="utf-8") as handle:
            return pd.DataFrame(list(csv.DictReader(handle)))

    def figure(self, name: str) -> Path | None:
        path = self.figures_dir / name
        return path if path.exists() else None

    def local_examples(self) -> dict[str, dict]:
        """The rule-chosen local examples from the SHAP phase (training rows, with their actual label)."""
        return {name: {"rule": e["rule"], "has_account": e["has_account"], "features": e["features"]}
                for name, e in self.shap["local_examples"].items()}


def held_out_examples(raw_test_path: Path, n: int, seed: int) -> list[dict]:
    """`n` respondents from the unlabelled test file (never seen in training),
    keeping only rows the API schema accepts."""
    from fin_inclusion.data.loader import EXPECTED_COLUMNS

    frame = pd.read_csv(raw_test_path)
    fields = [c for c in EXPECTED_COLUMNS if c not in ("year", "uniqueid")]
    sample = frame.sample(n=min(n * 2, len(frame)), random_state=seed)
    examples = []
    for _, row in sample.iterrows():
        record = {k: (int(v) if isinstance(v, (np.integer,)) else v) for k, v in row[fields].items()}
        try:
            examples.append(validate_record(record))
        except InvalidRecord:
            continue
        if len(examples) == n:
            break
    return examples
