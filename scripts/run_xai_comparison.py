"""XAI Phase 6 (Task 4): SHAP vs Captum -- where they agree, where they diverge, and why.

Two experiments, kept apart so a METHOD difference is never mistaken for a
MODEL difference:

  A. Same model (the MLP): SHAP vs Integrated Gradients vs Feature Ablation.
  B. Different models: TreeSHAP on XGBoost vs IG on the MLP -- what do the two
     candidate models rely on? -- plus same-method model comparisons that
     separate the model effect from the method effect.

The script is seven steps, in this order (see `main`):

    1. load the models, data, sample sets and every saved attribution
    2. ALIGN REFERENCES: exact Shapley values against the same single
       baseline points IG and Ablation use, so the reference can be held
       fixed while the method changes
    3. agreement for every comparison (rank correlations, top-k overlap,
       per-respondent rank correlation, sign agreement), each with a
       bootstrap interval, pooled and Uganda
    4. explaining disagreement with evidence: reference, value function
       (interactions), correlated features (Cramer's V), gradient saturation
    5. faithfulness: deletion/insertion curves, comprehensiveness and
       sufficiency, against two removal references
    6. sanity checks: weight randomisation and input-perturbation stability
    7. figures, comparison table, summary

Writes:

    reports/xai/comparison_table.csv          one row per comparison x sample
    reports/xai/comparison_summary.json       every number quoted later
    reports/xai/attributions/exact_shapley_point__*.npz, *_interactions_point_*
    reports/figures/xai/comparison_*.png, faithfulness_*.png

Agreement is not correctness. Where a ground truth exists -- the exact
Shapley value for a given reference -- methods are measured against it; the
faithfulness and sanity checks ask the model directly. Nothing here says
anything about what causes account ownership.
"""
from __future__ import annotations

import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "config"))
from settings import Settings, load_settings  # noqa: E402

from fin_inclusion.interpretability.attribution import Attribution  # noqa: E402
from fin_inclusion.interpretability.bootstrap import bootstrap_ci  # noqa: E402
from fin_inclusion.interpretability.captum_explainers import (  # noqa: E402
    BASELINES,
    FeatureAblationExplainer,
    IntegratedGradientsExplainer,
    baseline_codes,
    describe_baseline,
    path_gradient_profile,
)
from fin_inclusion.interpretability.comparison import (  # noqa: E402
    AGREEMENT_METRICS,
    agreement_with_ci,
    cramers_v,
    instance_rank_correlations,
    instance_rank_correlations_excluding_shared_zeros,
    per_feature_disagreement,
)
from fin_inclusion.interpretability.faithfulness import (  # noqa: E402
    CASCADE,
    faithfulness_scores,
    numeric_nudges,
    random_rankings,
    randomised_mlp,
    ranking,
    relative_change,
    removal_curves,
)
from fin_inclusion.interpretability.grouping import FeatureGrouping  # noqa: E402
from fin_inclusion.interpretability.plots import (  # noqa: E402
    plot_curves,
    plot_forest,
    plot_identity_scatter,
    plot_labelled_scatter,
    plot_lines,
    plot_matrix_panels,
    plot_rho_histograms,
)
from fin_inclusion.interpretability.shap_explainers import TreeShapExplainer  # noqa: E402
from fin_inclusion.interpretability.shapley import ExactShapleyExplainer, exact_shapley_interactions  # noqa: E402
from fin_inclusion.interpretability.workspace import Workspace, load_workspace  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PRODUCTION_DIR = ROOT / "models" / "production"
CHUNK_ROWS = 500

# (experiment, id, label, attribution A, attribution B)
COMPARISONS = [
    ("A", "A1", "Exact SHAP vs IG (modal)", "kernel_shap__mlp", "integrated_gradients__mlp__modal"),
    ("A", "A2", "Exact SHAP vs IG (unknown)", "kernel_shap__mlp", "integrated_gradients__mlp__unknown"),
    ("A", "A3", "Exact SHAP vs Ablation (modal)", "kernel_shap__mlp", "feature_ablation__mlp__modal"),
    ("A", "A4", "Exact SHAP vs Ablation (unknown)", "kernel_shap__mlp", "feature_ablation__mlp__unknown"),
    ("A", "A5", "IG vs Ablation (both modal)", "integrated_gradients__mlp__modal", "feature_ablation__mlp__modal"),
    ("A", "A6", "Exact SHAP vs GradientSHAP", "kernel_shap__mlp", "gradient_shap__mlp"),
    ("A-aligned", "R1", "SHAP: background vs modal point (reference only)",
     "kernel_shap__mlp", "exact_shapley_point__mlp__modal"),
    ("A-aligned", "M1", "SHAP vs IG at the modal point (method only)",
     "exact_shapley_point__mlp__modal", "integrated_gradients__mlp__modal"),
    ("A-aligned", "M2", "SHAP vs Ablation at the modal point (method only)",
     "exact_shapley_point__mlp__modal", "feature_ablation__mlp__modal"),
    ("A-aligned", "M3", "SHAP vs IG at the unknown point (method only)",
     "exact_shapley_point__mlp__unknown", "integrated_gradients__mlp__unknown"),
    ("A-aligned", "M4", "SHAP vs Ablation at the unknown point (method only)",
     "exact_shapley_point__mlp__unknown", "feature_ablation__mlp__unknown"),
    ("B", "B1", "TreeSHAP (XGBoost) vs IG modal (MLP)", "tree_shap__xgboost", "integrated_gradients__mlp__modal"),
    ("B", "B2", "TreeSHAP (XGBoost) vs IG unknown (MLP)", "tree_shap__xgboost", "integrated_gradients__mlp__unknown"),
    ("B", "B3", "Exact SHAP: XGBoost vs MLP (model only)", "kernel_shap__xgboost", "kernel_shap__mlp"),
    ("B", "B4", "Ablation modal: XGBoost vs MLP (model only)",
     "feature_ablation__xgboost__modal", "feature_ablation__mlp__modal"),
    ("B-aligned", "V1", "XGBoost: SHAP vs Ablation at the modal point (method only)",
     "exact_shapley_point__xgboost__modal", "feature_ablation__xgboost__modal"),
]

# Methods ranked in the faithfulness tests, per model: (attribution stem, label).
FAITHFULNESS_METHODS = {
    "xgboost": [("tree_shap__xgboost", "TreeSHAP"), ("kernel_shap__xgboost", "exact SHAP"),
                ("exact_shapley_point__xgboost__modal", "SHAP @ modal point"),
                ("feature_ablation__xgboost__modal", "Ablation modal"),
                ("feature_ablation__xgboost__unknown", "Ablation unknown")],
    "mlp": [("kernel_shap__mlp", "exact SHAP"), ("gradient_shap__mlp", "GradientSHAP"),
            ("exact_shapley_point__mlp__modal", "SHAP @ modal point"),
            ("integrated_gradients__mlp__modal", "IG modal"), ("integrated_gradients__mlp__unknown", "IG unknown"),
            ("feature_ablation__mlp__modal", "Ablation modal"), ("feature_ablation__mlp__unknown", "Ablation unknown")],
}
REMOVALS = ("background", "modal point")
N_RANDOM_ORDERS = 3
RANDOMISATION_ROWS = 300
STABILITY_ROWS = 200
SATURATION_POINTS = 32
FLAT_FRACTION = 0.10  # a path step is "flat" when |df/dalpha| < 10% of that path's largest
MODEL_LABEL = {"xgboost": "XGBoost (shipped)", "mlp": "PyTorch MLP"}


class Attributions:
    """Loads saved attribution sets on demand, checking they cover the explanation rows."""

    def __init__(self, ws: Workspace, directory: Path):
        self.ws, self.directory, self._cache = ws, directory, {}

    def __getitem__(self, stem: str) -> Attribution:
        if stem not in self._cache:
            attribution = Attribution.load(self.directory / f"{stem}.npz")
            if attribution.metadata.get("rows") != self.ws.sets.explanation.tolist():
                raise ValueError(f"{stem}.npz is not for the current explanation rows; re-run its script")
            self._cache[stem] = attribution
        return self._cache[stem]


# --- Step 2: aligned references ------------------------------------------------------


def point_shapley(ws: Workspace, baselines: dict, attribution_dir: Path, force: bool) -> None:
    """Exact Shapley values against each single baseline point, both models."""
    rows = ws.sets.explanation
    for key, model in ws.models.items():
        for kind in BASELINES:
            path = attribution_dir / f"exact_shapley_point__{key}__{kind}.npz"
            if path.exists() and not force and Attribution.load(path).metadata.get("rows") == rows.tolist():
                print(f"  {path.stem}: reusing")
                continue
            codes = baselines[key][kind]
            explainer = ExactShapleyExplainer.at_point(model, codes, describe_baseline(model, codes, kind), kind)
            parts = [explainer.explain(ws.rows(rows[s:s + CHUNK_ROWS])) for s in range(0, len(rows), CHUNK_ROWS)]
            attribution = replace(parts[0], values=np.vstack([p.values for p in parts]),
                                  metadata={**parts[0].metadata, "rows": rows.tolist()})
            attribution.save(path)
            print(f"  {path.stem}: base value {attribution.base_value:+.3f}")


# --- Step 3: agreement ----------------------------------------------------------------


def agreement_table(ws: Workspace, attributions: Attributions, settings: Settings) -> tuple[list[dict], dict]:
    rows, histograms = [], {}
    for experiment, cid, label, a_stem, b_stem in COMPARISONS:
        a, b = attributions[a_stem], attributions[b_stem]
        histograms[cid] = {}
        for sample, mask in ws.sample_masks.items():
            result = agreement_with_ci(a.values[mask], b.values[mask], settings.xai.bootstrap_reps, settings.seed)
            histograms[cid][sample] = result.pop("instance_spearman_values")
            rows.append({"experiment": experiment, "id": cid, "comparison": label, "a": a_stem, "b": b_stem,
                         "reference_a": a.baseline_description.split(";")[0],
                         "reference_b": b.baseline_description.split(";")[0], "sample": sample, **result})
        pooled = rows[-2]
        print(f"  {cid} {label:58s} global rho {pooled['global_spearman']['estimate']:+.2f}  per-row median "
              f"{pooled['instance_spearman_median']['estimate']:+.2f} (excl. shared zeros "
              f"{pooled['instance_spearman_median_excluding_shared_zeros']['estimate']:+.2f})  same top "
              f"{pooled['same_top_feature_share']['estimate']:.0%}")
    return rows, histograms


def write_table(rows: list[dict], path: Path) -> None:
    columns = ["experiment", "id", "comparison", "sample", "n", "reference_a", "reference_b"]
    for metric in AGREEMENT_METRICS:
        columns += [metric, f"{metric}_ci_low", f"{metric}_ci_high"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            values = [row[c] for c in columns[:7]]
            for metric in AGREEMENT_METRICS:
                values += [round(row[metric]["estimate"], 4), round(row[metric]["ci"][0], 4),
                           round(row[metric]["ci"][1], 4)]
            writer.writerow(values)


# --- Step 4: why they disagree -----------------------------------------------------------


def value_function_evidence(ws: Workspace, attributions: Attributions, baselines: dict, attribution_dir: Path,
                            force: bool) -> tuple[dict, dict]:
    """Same reference (the modal point), different value function: Ablation
    removes a feature from the FULL set of others; Shapley averages over all
    coalitions. The difference is made only of interaction terms, so it
    should track each feature's pairwise interactions."""
    pooled = ws.sample_masks["pooled"]
    positions = ws.sets.explanation[pooled]
    report, scatter = {}, {}
    for key, model in ws.models.items():
        path = attribution_dir / f"exact_shapley_interactions_point__{key}__modal__pooled.npz"
        if path.exists() and not force:
            with np.load(path, allow_pickle=False) as archive:
                matrices = archive["interactions"]
                assert np.array_equal(archive["rows"], positions), f"{path.name} is for different rows"
        else:
            matrices, _ = exact_shapley_interactions(model.predict_logit_codes, model.to_codes(ws.rows(positions)),
                                                     baselines[key]["modal"][None])
            np.savez_compressed(path, interactions=matrices, rows=positions)
        shapley = attributions[f"exact_shapley_point__{key}__modal"].values[pooled]
        ablation = attributions[f"feature_ablation__{key}__modal"].values[pooled]
        gap = ablation - shapley
        pair_sum = matrices.sum(axis=2) - np.einsum("nii->ni", matrices)  # sum_k!=j Phi_jk
        fit = float(1 - ((gap - pair_sum) ** 2).sum() / ((gap - gap.mean()) ** 2).sum())
        report[key] = {
            "n": int(len(positions)),
            "mean_abs_gap_ablation_minus_shapley": float(np.abs(gap).mean()),
            "mean_abs_shapley": float(np.abs(shapley).mean()),
            "spearman_gap_vs_pairwise_interaction_sum": float(spearmanr(gap.ravel(), pair_sum.ravel()).statistic),
            "r2_gap_explained_by_pairwise_sum_identity": fit,
            "same_sign_share": float((np.sign(gap) == np.sign(pair_sum))[np.abs(gap) > 1e-6].mean()),
        }
        scatter[key] = (pair_sum.ravel(), gap.ravel())
        print(f"  {key}: Ablation - SHAP (modal point) vs sum of pairwise interactions: Spearman "
              f"{report[key]['spearman_gap_vs_pairwise_interaction_sum']:+.2f}, R^2 on y = x {fit:.2f}")
    return report, scatter


def correlated_feature_evidence(ws: Workspace, attributions: Attributions, table: list[dict]) -> tuple[dict, np.ndarray]:
    """Cramer's V between every pair of raw features (numerics in quartile
    bins) on the full training set, and whether the features the methods
    disagree on most are the ones most associated with others."""
    names = list(ws.raw_features)
    columns = {}
    for name in names:
        values = ws.X_raw[name]
        columns[name] = (pd.qcut(values, 4, duplicates="drop").astype(str).to_numpy()
                         if name not in ws.xgboost.categorical_levels else values.to_numpy())
    v = np.eye(len(names))
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            v[i, j] = v[j, i] = cramers_v(columns[names[i]], columns[names[j]])
    off = v - np.eye(len(names))
    max_v = off.max(axis=1)
    pooled = ws.sample_masks["pooled"]
    disagreement = {}
    for cid in ("A1", "A3", "A5", "R1", "M1", "M2"):
        _, _, _, a_stem, b_stem = next(c for c in COMPARISONS if c[1] == cid)
        disagreement[cid] = per_feature_disagreement(attributions[a_stem].values[pooled],
                                                     attributions[b_stem].values[pooled])
    raw_methods = np.mean([disagreement[c] for c in ("A1", "A3", "A5")], axis=0)
    pairs = sorted(((v[i, j], names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))),
                   reverse=True)
    report = {
        "cramers_v": {names[i]: {names[j]: float(v[i, j]) for j in range(len(names))} for i in range(len(names))},
        "numerics_binned": "quartiles of the training values",
        "strongest_pairs": [{"pair": [a, b], "v": float(x)} for x, a, b in pairs[:5]],
        "named_pairs": {
            "relationship_with_head x marital_status": float(v[names.index("relationship_with_head"),
                                                               names.index("marital_status")]),
            "country x location_type": float(v[names.index("country"), names.index("location_type")]),
        },
        "max_v_by_feature": dict(zip(names, map(float, max_v))),
        "per_feature_disagreement": {cid: dict(zip(names, map(float, d))) for cid, d in disagreement.items()},
        "mean_disagreement_A1_A3_A5_by_feature": dict(zip(names, map(float, raw_methods))),
    }
    for label, values in (("methods (A1, A3, A5 mean)", raw_methods), ("reference only (R1)", disagreement["R1"])):
        result = spearmanr(max_v, values)
        report[f"spearman_max_v_vs_disagreement__{label}"] = {"rho": float(result.statistic),
                                                              "p_value": float(result.pvalue), "n_features": len(names)}
        print(f"  Cramer's V vs disagreement, {label}: Spearman {result.statistic:+.2f} (p = {result.pvalue:.2f}, "
              f"n = {len(names)} features)")
    return report, v


def saturation_evidence(ws: Workspace, baselines: dict, attributions: Attributions, histograms: dict) -> tuple[dict, dict]:
    """Along IG's path, where is the MLP's output flat, and how far is the
    endpoint gradient from the full integral?"""
    pooled = ws.sample_masks["pooled"]
    X = ws.rows(ws.sets.explanation[pooled])
    grouping = FeatureGrouping.for_embeddings(ws.mlp.categorical_columns, ws.mlp.net.embeddings[0].embedding_dim,
                                              ws.mlp.numeric_inputs)
    report, profiles = {}, {}
    for kind in BASELINES:
        profile, endpoint = path_gradient_profile(ws.mlp, X, baselines["mlp"][kind], SATURATION_POINTS)
        magnitude = np.abs(profile)
        flat = (magnitude < FLAT_FRACTION * magnitude.max(axis=1, keepdims=True)).mean(axis=1)
        total = profile.mean(axis=1)  # = f(x) - f(baseline), up to the midpoint rule
        endpoint_total = endpoint.sum(axis=1)
        moved = np.abs(total) > 0.05
        linear_gap = np.abs(endpoint_total - total)[moved] / np.abs(total)[moved]
        ig = attributions[f"integrated_gradients__mlp__{kind}"].values[pooled]
        rho_endpoint = instance_rank_correlations(grouping.aggregate(endpoint), ig)
        m_id = "M1" if kind == "modal" else "M3"
        disagreement = 1 - histograms[m_id]["pooled"]
        report[kind] = {
            "n": int(len(X)),
            "median_flat_share_of_path": float(np.median(flat)),
            "share_rows_with_over_half_the_path_flat": float((flat > 0.5).mean()),
            "median_relative_gap_endpoint_gradient_vs_IG_total": float(np.median(linear_gap)),
            "rows_with_output_change_over_0.05": int(moved.sum()),
            "median_rank_correlation_endpoint_vs_IG": float(np.nanmedian(rho_endpoint)),
            f"spearman_flatness_vs_{m_id}_disagreement": float(spearmanr(flat, disagreement, nan_policy="omit").statistic),
        }
        profiles[kind] = np.median(magnitude / magnitude.max(axis=1, keepdims=True), axis=0)
        print(f"  IG {kind}: median {report[kind]['median_flat_share_of_path']:.0%} of the path is flat; endpoint "
              f"gradient misses {report[kind]['median_relative_gap_endpoint_gradient_vs_IG_total']:.0%} of the change "
              f"(median); flatness vs SHAP-IG disagreement Spearman "
              f"{report[kind][f'spearman_flatness_vs_{m_id}_disagreement']:+.2f}")
    return report, profiles


# --- Step 5: faithfulness ---------------------------------------------------------------


def faithfulness(ws: Workspace, attributions: Attributions, baselines: dict, settings: Settings) -> tuple[dict, dict]:
    report, curves = {}, {}
    rows = ws.sets.explanation
    for key, model in ws.models.items():
        references = {"background": model.to_codes(ws.background), "modal point": baselines[key]["modal"][None]}
        for sample, mask in ws.sample_masks.items():
            X = model.to_codes(ws.rows(rows[mask]))
            logit = model.predict_logit_codes(X)
            orders = {label: ranking(attributions[stem].values[mask]) for stem, label in FAITHFULNESS_METHODS[key]}
            randoms = random_rankings(len(X), X.shape[1], N_RANDOM_ORDERS, settings.seed)
            for removal in REMOVALS:
                entry, mean_curves = {}, {}
                for label, order in [*orders.items(), ("random order", None)]:
                    if order is None:
                        per_order = [removal_curves(model.predict_logit_codes, X, o, references[removal]) for o in randoms]
                        deletion = np.mean([d for d, _ in per_order], axis=0)
                        insertion = np.mean([i for _, i in per_order], axis=0)
                    else:
                        deletion, insertion = removal_curves(model.predict_logit_codes, X, order, references[removal])
                    scores = faithfulness_scores(logit, deletion, insertion)
                    entry[label] = {}
                    for metric, values in scores.items():
                        ci = bootstrap_ci(lambda r: values[r].mean(), len(values), settings.xai.bootstrap_reps, settings.seed)
                        entry[label][metric] = {"mean": float(ci.estimate), "ci": [float(ci.lower), float(ci.upper)]}
                    mean_curves[label] = {"deletion": np.abs(logit[:, None] - deletion).mean(axis=0),
                                          "insertion": np.abs(logit[:, None] - insertion).mean(axis=0)}
                report.setdefault(key, {}).setdefault(sample, {})[removal] = entry
                curves[(key, sample, removal)] = mean_curves
                best = max(entry, key=lambda m: entry[m]["deletion_area"]["mean"])
                print(f"  {key:8s} {sample:7s} remove->{removal:12s} best deletion: {best} "
                      f"({entry[best]['deletion_area']['mean']:.3f}); random {entry['random order']['deletion_area']['mean']:.3f}")
    return report, curves


# --- Step 6: sanity checks ---------------------------------------------------------------


def randomisation_check(ws: Workspace, baselines: dict, n_steps: int, settings: Settings) -> dict:
    """Cascading weight randomisation of the MLP: similarity between each
    method's attributions on the trained and the randomised model."""
    rng = np.random.default_rng(settings.seed)
    positions = np.sort(rng.choice(ws.sets.pooled, size=RANDOMISATION_ROWS, replace=False))
    X = ws.rows(positions)
    modal = baselines["mlp"]["modal"]

    def explain(model) -> dict[str, np.ndarray]:
        return {
            "exact SHAP": ExactShapleyExplainer(model, ws.background).explain(X).values,
            "IG modal": IntegratedGradientsExplainer(model, modal, "modal", n_steps).explain(X).values,
            "Ablation modal": FeatureAblationExplainer(model, modal, "modal").explain(X).values,
        }

    original, original_logit = explain(ws.mlp), ws.mlp.predict_logit(X)
    steps = ["trained model"] + [f"+ {name}" for name in CASCADE]
    report = {"rows": positions.tolist(), "steps": steps, "methods": {}, "model_logit_correlation": [1.0]}
    for method in original:
        report["methods"][method] = {"median_instance_rank_correlation": [1.0],
                                     "median_instance_rank_correlation_excluding_shared_zeros": [1.0],
                                     "global_rank_correlation": [1.0]}
    for step in range(1, len(CASCADE) + 1):
        model = randomised_mlp(ws.mlp, step, settings.seed)
        report["model_logit_correlation"].append(float(np.corrcoef(original_logit, model.predict_logit(X))[0, 1]))
        for method, values in explain(model).items():
            rho = instance_rank_correlations(original[method], values)
            entry = report["methods"][method]
            entry["median_instance_rank_correlation"].append(float(np.nanmedian(rho)))
            entry["median_instance_rank_correlation_excluding_shared_zeros"].append(float(np.nanmedian(
                instance_rank_correlations_excluding_shared_zeros(original[method], values))))
            entry["global_rank_correlation"].append(float(spearmanr(np.abs(original[method]).mean(0),
                                                                    np.abs(values).mean(0)).statistic))
        print(f"  {steps[step]:22s} model logit corr {report['model_logit_correlation'][-1]:+.2f}; per-row rank corr "
              + ", ".join(f"{m} {report['methods'][m]['median_instance_rank_correlation'][-1]:+.2f} (excl. shared "
                          f"zeros {report['methods'][m]['median_instance_rank_correlation_excluding_shared_zeros'][-1]:+.2f})"
                          for m in original))
    return report


def stability_check(ws: Workspace, baselines: dict, n_steps: int, settings: Settings) -> dict:
    """Nudge age or household size by one unit; how much do attributions move?"""
    rng = np.random.default_rng(settings.seed + 1)
    positions = np.sort(rng.choice(ws.sets.pooled, size=STABILITY_ROWS, replace=False))
    report = {"rows": positions.tolist(), "nudge": "age_of_respondent and household_size, each +/-1, within the "
              "observed training range", "methods": {}}
    for key, model in ws.models.items():
        codes = model.to_codes(ws.rows(positions))
        position = {name: list(model.raw_features).index(name) for name in ("age_of_respondent", "household_size")}
        bounds = {name: (float(ws.X_raw[name].min()), float(ws.X_raw[name].max())) for name in position}
        modal = baselines[key]["modal"]
        explainers = {"exact SHAP": ExactShapleyExplainer(model, ws.background),
                      "Ablation modal": FeatureAblationExplainer(model, modal, "modal")}
        if key == "xgboost":
            explainers["TreeSHAP"] = TreeShapExplainer(model, ws.background)
        else:
            explainers["IG modal"] = IntegratedGradientsExplainer(model, modal, "modal", n_steps)
        base_logit = model.predict_logit_codes(codes)
        original = {name: e.explain(model.from_codes(codes)).values for name, e in explainers.items()}
        for name, explainer in explainers.items():
            changes, rhos, logit_moves = [], [], []
            for _, moved, valid in numeric_nudges(codes, position, bounds):
                perturbed = explainer.explain(model.from_codes(moved[valid])).values
                changes.append(relative_change(original[name][valid], perturbed))
                rhos.append(instance_rank_correlations(original[name][valid], perturbed))
                logit_moves.append(np.abs(model.predict_logit_codes(moved[valid]) - base_logit[valid]))
            change, rho, move = np.concatenate(changes), np.concatenate(rhos), np.concatenate(logit_moves)
            ci = bootstrap_ci(lambda r: np.nanmedian(change[r]), len(change), settings.xai.bootstrap_reps, settings.seed)
            report["methods"][f"{name} · {key}"] = {
                "n_perturbed_rows": int(len(change)),
                "median_relative_l1_change": float(ci.estimate), "ci": [float(ci.lower), float(ci.upper)],
                "median_instance_rank_correlation": float(np.nanmedian(rho)),
                "mean_abs_logit_change": float(move.mean()),
            }
            print(f"  {name:15s} {key:8s} median relative change {ci.estimate:.3f} [{ci.lower:.3f}, {ci.upper:.3f}], "
                  f"rank corr {np.nanmedian(rho):+.2f}, mean |logit change| {move.mean():.3f}")
    return report


# --- Step 7: figures -------------------------------------------------------------------------


def make_figures(ws: Workspace, table: list[dict], histograms: dict, value_scatter: dict, v_matrix: np.ndarray,
                 correlated: dict, saturation_profiles: dict, faith: dict, curves: dict, randomisation: dict,
                 stability: dict, figures_dir: Path) -> None:
    focus = ws.sets.focus_country.lower()
    labels = {cid: f"{cid}  {label}" for _, cid, label, _, _ in COMPARISONS}
    plot_rho_histograms(
        {labels[cid]: {"pooled": h["pooled"], focus: h[focus]} for cid, h in histograms.items()},
        figures_dir / "comparison_instance_rank_correlation_histograms.png",
        title="Per-respondent agreement between explanations: rank correlation of |attribution|",
        caption="A: same model (MLP), different methods. R/M: references aligned (exact Shapley at the same baseline "
                "point as IG/Ablation), so R isolates the reference and M the method. B: different models. "
                f"Pooled n = {int(ws.sample_masks['pooled'].sum()):,}, {focus} n = {int(ws.sample_masks[focus].sum()):,}.",
    )
    for metric, xlabel, name in (("instance_spearman_median", "median per-respondent rank correlation", "instance"),
                                 ("instance_spearman_median_excluding_shared_zeros",
                                  "median per-respondent rank correlation, shared zeros excluded",
                                  "instance_no_shared_zeros"),
                                 ("global_spearman", "rank correlation of global importance", "global")):
        series = {}
        for sample in ("pooled", focus):
            chosen = [r for r in table if r["sample"] == sample]
            series[sample] = tuple(np.array(v) for v in zip(*[(r[metric]["estimate"], *r[metric]["ci"]) for r in chosen]))
        plot_forest([labels[r["id"]] for r in table if r["sample"] == "pooled"], series,
                    figures_dir / f"comparison_forest__{name}_rank_correlation.png",
                    title=f"Agreement between explanations: {xlabel}", xlabel=xlabel, reference=0.0, xlim=(-1, 1),
                    caption="Comparison IDs as in the comparison table. 1 = same ordering, 0 = unrelated.")
    for key, (pair_sum, gap) in value_scatter.items():
        plot_identity_scatter(
            pair_sum, gap, figures_dir / f"comparison_value_function__{key}.png",
            title=f"{MODEL_LABEL[key]}: why Ablation differs from SHAP at the same reference",
            xlabel="sum of the feature's pairwise Shapley interactions (logits)",
            ylabel="Ablation - Shapley value (logits)",
            caption="Both at the modal baseline point, pooled sample. Ablation removes a feature from the full set of "
                    "others; Shapley averages over coalitions. Their difference consists only of interaction terms.",
        )
    names = list(ws.raw_features)
    shown = v_matrix.copy()
    np.fill_diagonal(shown, np.nan)
    plot_matrix_panels({"Cramer's V (training data)": shown},
                       names, names, figures_dir / "comparison_cramers_v.png",
                       title="How strongly are the features associated with each other?",
                       caption="Cramer's V on all 23,524 training rows (0 = independent, 1 = one determines the other); "
                               "age and household size binned into quartiles. Diagonal left blank.",
                       diverging=False, vmax=float(np.nanmax(shown)), fmt="{:.2f}")
    max_v = np.array([correlated["max_v_by_feature"][n] for n in names])
    disagreement = np.array([correlated["mean_disagreement_A1_A3_A5_by_feature"][n] for n in names])
    rho = correlated["spearman_max_v_vs_disagreement__methods (A1, A3, A5 mean)"]
    plot_labelled_scatter(
        max_v, disagreement, names, figures_dir / "comparison_disagreement_vs_cramers_v.png",
        title="Do the methods disagree most on features tied to other features?",
        xlabel="feature's strongest Cramer's V with any other feature",
        ylabel="method disagreement on the feature (relative)",
        caption=f"Disagreement: mean |a - b| over the mean size of the two, averaged over comparisons A1, A3, A5 (MLP, "
                f"pooled). Spearman across the 10 features {rho['rho']:+.2f} (p = {rho['p_value']:.2f}); with 10 points "
                f"this can only show a strong pattern, not rule one out.",
    )
    plot_curves(
        {"MLP, IG path": {f"{kind} baseline": profile for kind, profile in saturation_profiles.items()}},
        figures_dir / "comparison_saturation__mlp.png",
        title="Along IG's path, how much does the MLP's output still change?",
        xlabel=f"position along the path (step of {SATURATION_POINTS}, baseline -> respondent)",
        ylabel="|df/dalpha| relative to its path maximum (median)",
        caption="Median over the pooled sample of the path derivative IG integrates, scaled per respondent to its own "
                "maximum. Low stretches are flat (saturated) regions where gradients carry little.",
    )
    for key in ws.models:
        for removal in REMOVALS:
            mean_curves = curves[(key, "pooled", removal)]
            plot_curves(
                {"Deletion: top-k removed (higher = more faithful)": {m: c["deletion"] for m, c in mean_curves.items()},
                 "Insertion: only top-k kept (lower = more faithful)": {m: c["insertion"] for m, c in mean_curves.items()}},
                figures_dir / f"faithfulness_curves__{key}__{removal.replace(' ', '_')}.png",
                title=f"{MODEL_LABEL[key]}: deletion and insertion, removal = {removal} (pooled)",
                xlabel="k features (in the method's order)", ylabel="mean |f(x) - f(modified x)| (logits)",
                caption=f"Removal reference: {removal}"
                        + (" (averaged over the 200 background rows: the SHAP methods' own value function, so it favours "
                           "them)" if removal == "background" else " (IG's and Ablation's own reference, so it favours "
                           "them)") + f". Random order: mean of {N_RANDOM_ORDERS} random orders. Every row is a training row.",
            )
        methods = list(faith[key]["pooled"]["background"])
        for metric, xlabel in (("deletion_area", "deletion area (higher = more faithful)"),
                               ("insertion_gap", "insertion gap (lower = more faithful)")):
            series = {removal: tuple(np.array(v) for v in zip(*[
                (faith[key]["pooled"][removal][m][metric]["mean"], *faith[key]["pooled"][removal][m][metric]["ci"])
                for m in methods])) for removal in REMOVALS}
            plot_forest(methods, series, figures_dir / f"faithfulness_{metric}__{key}.png",
                        title=f"{MODEL_LABEL[key]}: {xlabel.split(' (')[0]} by method and removal reference (pooled)",
                        xlabel=f"{xlabel}, logits",
                        caption="Mean over respondents of |f(x) - f(modified x)| averaged over k = 0..10. Each removal "
                                "reference favours the methods built on it.")
    plot_lines(
        randomisation["steps"],
        {**{m: e["median_instance_rank_correlation_excluding_shared_zeros"]
            for m, e in randomisation["methods"].items()},
         "model output (logit correlation)": randomisation["model_logit_correlation"]},
        figures_dir / "comparison_randomisation__mlp.png",
        title="Sanity check: explanations of a progressively randomised MLP",
        ylabel="similarity to the trained model's", ylim=(-0.4, 1.05),
        caption=f"Cascading re-initialisation from the output layer back to the embeddings; {RANDOMISATION_ROWS} pooled "
                "rows. Methods: median per-respondent rank correlation with the trained model's attributions, over the "
                "features not exactly zero in both (a respondent who equals the modal baseline on a feature gets 0 "
                "from IG and Ablation whatever the model; counting those would hide the change). A faithful method's "
                "similarity must fall as the model it explains is destroyed.",
    )
    methods = list(stability["methods"])
    plot_forest(methods, {"median": tuple(np.array(v) for v in zip(*[
                    (stability["methods"][m]["median_relative_l1_change"], *stability["methods"][m]["ci"]) for m in methods]))},
                figures_dir / "comparison_stability.png",
                title="Stability: how much do attributions move when age or household size changes by 1?",
                xlabel="median relative change ||phi(x') - phi(x)||1 / ||phi(x)||1",
                caption=f"{STABILITY_ROWS} pooled rows, each nudged by +/-1 in age and in household size (within the "
                        "training range). Read beside the model's own mean |logit change| in the summary.")


# --- Orchestration ---------------------------------------------------------------------------


def jsonable(table: list[dict]) -> list[dict]:
    return [{k: v for k, v in row.items()} for row in table]


def main() -> None:
    force = "--force" in sys.argv[1:]
    settings = load_settings()
    xai = settings.xai
    attribution_dir = xai.artifacts_dir / "attributions"

    # 1. Inputs
    ws = load_workspace(settings.paths.raw_train, settings.paths.raw_test, PRODUCTION_DIR, xai.model_dir,
                        xai.artifacts_dir)
    attributions = Attributions(ws, attribution_dir)
    captum = json.loads((xai.artifacts_dir / "captum_summary.json").read_text(encoding="utf-8"))
    n_steps = captum["integrated_gradients"]["n_steps"]["chosen"]
    training_codes = ws.mlp.to_codes(ws.X_raw)
    baselines = {key: {kind: baseline_codes(model, training_codes, kind) for kind in BASELINES}
                 for key, model in ws.models.items()}
    print(f"Loaded models and data; IG steps {n_steps} (from Phase 4)")

    # 2. Aligned references
    print("\n=== Exact Shapley values at the IG/Ablation baseline points ===")
    point_shapley(ws, baselines, attribution_dir, force)

    # 3. Agreement
    print("\n=== Agreement, every comparison (pooled shown) ===")
    table, histograms = agreement_table(ws, attributions, settings)
    write_table(table, xai.artifacts_dir / "comparison_table.csv")

    # 4. Why they disagree
    print("\n=== Evidence: value function (interactions) ===")
    value_report, value_scatter = value_function_evidence(ws, attributions, baselines, attribution_dir, force)
    print("=== Evidence: correlated features ===")
    correlated_report, v_matrix = correlated_feature_evidence(ws, attributions, table)
    print("=== Evidence: gradient saturation (MLP) ===")
    saturation_report, saturation_profiles = saturation_evidence(ws, baselines, attributions, histograms)

    # 5. Faithfulness
    print("\n=== Faithfulness: deletion / insertion ===")
    faith_report, curves = faithfulness(ws, attributions, baselines, settings)

    # 6. Sanity checks
    print("\n=== Weight randomisation (MLP) ===")
    randomisation_report = randomisation_check(ws, baselines, n_steps, settings)
    print("=== Stability under small input changes ===")
    stability_report = stability_check(ws, baselines, n_steps, settings)

    # 7. Figures, table, summary
    make_figures(ws, table, histograms, value_scatter, v_matrix, correlated_report, saturation_profiles,
                 faith_report, curves, randomisation_report, stability_report, xai.figures_dir)
    np.savez_compressed(xai.artifacts_dir / "comparison_instance_rank_correlations.npz",
                        **{f"{cid}__{sample}": rho for cid, by in histograms.items() for sample, rho in by.items()})
    summary = {
        "note": "agreement is not correctness; attributions describe the models, not causes; every explained row is "
                "a training row; logits throughout",
        "comparisons": [{"experiment": e, "id": c, "label": l, "a": a, "b": b} for e, c, l, a, b in COMPARISONS],
        "agreement": jsonable(table),
        "evidence": {"value_function": value_report, "correlated_features": correlated_report,
                     "gradient_saturation": saturation_report},
        "faithfulness": faith_report,
        "randomisation": randomisation_report,
        "stability": stability_report,
    }
    path = xai.artifacts_dir / "comparison_summary.json"
    path.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"\nWrote {path}, comparison_table.csv and figures to {xai.figures_dir}")


if __name__ == "__main__":
    main()
