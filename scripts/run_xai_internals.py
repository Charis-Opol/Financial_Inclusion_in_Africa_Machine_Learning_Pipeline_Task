"""XAI Phase 5 (Task 3): visualising what the models rely on, and how.

The script is six steps, in this order (see `main`):

    1. load the models, data, sample sets and every saved attribution
    2. global importance under every method, on one comparable scale
    3. attribution heatmaps: respondents x features, sorted by score
    4. average attribution for true positives vs true negatives
    5. ATTENTION MAPS: not applicable (see below); substitutes:
       a. 2-D projections of the MLP's learned category embeddings
       b. exact Shapley interaction values, and the strongest interaction
    6. summary

Writes:

    reports/xai/attributions/exact_shapley_interactions__<model>__pooled.npz
    reports/xai/internals_summary.json
    reports/figures/xai/internals_*.png

Attention maps: neither model has an attention mechanism -- XGBoost is a
tree ensemble and the PyTorch model is an MLP over embeddings -- so there
is nothing to draw and none is fabricated. Step 5 shows the closest real
internals instead: what the MLP learned for each category, and which pairs
of features the models use together.

Comparability: "importance" is each feature's share of its method's total
mean |attribution| (methods differ in scale and reference, so raw sizes are
not comparable); every signed heatmap shares one colour scale; every caption
states units (logits), reference and sample size.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "config"))
from settings import Settings, load_settings  # noqa: E402

from fin_inclusion.interpretability.attribution import Attribution  # noqa: E402
from fin_inclusion.interpretability.bootstrap import bootstrap_ci  # noqa: E402
from fin_inclusion.interpretability.grouping import importance_share, mean_abs_importance  # noqa: E402
from fin_inclusion.interpretability.plots import (  # noqa: E402
    plot_attribution_heatmap,
    plot_embedding_projection,
    plot_interaction_facets,
    plot_matrix_panels,
    plot_rank_table,
    plot_share_grid,
)
from fin_inclusion.interpretability.shapley import exact_shapley_interactions  # noqa: E402
from fin_inclusion.interpretability.workspace import Workspace, load_workspace  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PRODUCTION_DIR = ROOT / "models" / "production"

# Every attribution set compared in step 2: (artifact stem, short label).
CATALOGUE = [
    ("tree_shap__xgboost", "TreeSHAP · XGB"),
    ("kernel_shap__xgboost", "Exact SHAP · XGB"),
    ("feature_ablation__xgboost__modal", "Ablation modal · XGB"),
    ("feature_ablation__xgboost__unknown", "Ablation unknown · XGB"),
    ("kernel_shap__mlp", "Exact SHAP · MLP"),
    ("gradient_shap__mlp", "GradientSHAP · MLP"),
    ("integrated_gradients__mlp__modal", "IG modal · MLP"),
    ("integrated_gradients__mlp__unknown", "IG unknown · MLP"),
    ("feature_ablation__mlp__modal", "Ablation modal · MLP"),
    ("feature_ablation__mlp__unknown", "Ablation unknown · MLP"),
]
# The one attribution per model used where a single view is shown (steps 3-5):
# exact raw-feature Shapley values, which are also the SHAP ground truth (D3.3).
PRIMARY = {"xgboost": "kernel_shap__xgboost", "mlp": "kernel_shap__mlp"}
MODEL_LABEL = {"xgboost": "XGBoost (shipped)", "mlp": "PyTorch MLP"}
EMBEDDING_FEATURES = ["job_type", "country", "education_level"]
HEATMAP_CLIP_PERCENTILE = 99
SHAP_REFERENCE = ("Exact interventional Shapley values on the raw features; reference: mean logit over 200 "
                  "background rows.")
ATTENTION_STATEMENT = (
    "Attention maps: not applicable. Neither model has an attention mechanism (XGBoost is a tree ensemble; the "
    "PyTorch model is an MLP over category embeddings), so no attention map exists and none is drawn. Substitutes: "
    "2-D projections of the learned category embeddings, and exact Shapley interaction values."
)


# --- Step 1: inputs --------------------------------------------------------------


def load_attributions(ws: Workspace, attribution_dir: Path) -> dict[str, Attribution]:
    loaded = {}
    for stem, _ in CATALOGUE:
        attribution = Attribution.load(attribution_dir / f"{stem}.npz")
        if attribution.metadata.get("rows") != ws.sets.explanation.tolist():
            raise ValueError(f"{stem}.npz is not for the current explanation rows; re-run its script")
        if attribution.feature_names != ws.raw_features:
            raise ValueError(f"{stem}.npz is not over the raw features")
        loaded[stem] = attribution
    return loaded


# --- Step 2: every method on one scale ----------------------------------------------


def global_comparison(ws: Workspace, attributions: dict[str, Attribution], figures_dir: Path) -> dict:
    report = {}
    names = list(ws.raw_features)
    consensus_order = None
    for sample, mask in ws.sample_masks.items():
        shares = np.column_stack([
            importance_share(mean_abs_importance(attributions[stem].select(mask))).to_numpy() for stem, _ in CATALOGUE
        ])
        ranks = np.column_stack([pd.Series(-shares[:, j]).rank(method="min").to_numpy() for j in range(len(CATALOGUE))])
        if consensus_order is None:  # fixed from the pooled sample, reused for the focus country
            consensus_order = list(np.argsort(ranks.mean(axis=1), kind="stable"))
        order = consensus_order
        n = int(mask.sum())
        labels = [label for _, label in CATALOGUE]
        plot_rank_table(
            ranks[order], shares[order], [names[i] for i in order], labels,
            figures_dir / f"internals_rank_table__{sample}.png",
            title=f"Feature rank under every method ({sample}, n = {n:,}); share of importance in brackets",
            caption="Rank 1 = largest mean |attribution|. Rows ordered by mean rank across methods (pooled). "
                    "SHAP methods: mean over 200 background rows. IG and Ablation: one baseline point, modal "
                    "(most frequent answers) or unknown (never-seen categories). Ablation removes one feature at "
                    "a time and is not additive.",
        )
        plot_share_grid(
            {label.replace(" · ", "\n"): shares[:, j] for j, label in enumerate(labels)}, names, order,
            figures_dir / f"internals_global_importance_all_methods__{sample}.png",
            title=f"Share of importance, every method ({sample}, n = {n:,})",
            caption="Each panel: each feature's share of that method's total mean |attribution| (sums to 1), so "
                    "methods with different scales and references are comparable. Same feature order and scale "
                    "in every panel. References as in the rank table.",
        )
        report[sample] = {
            "n": n,
            "methods": labels,
            "ranks": {names[i]: {label: int(ranks[i, j]) for j, label in enumerate(labels)} for i in range(len(names))},
            "shares": {names[i]: {label: float(shares[i, j]) for j, label in enumerate(labels)} for i in range(len(names))},
            "features_ranked_top3_by_every_method": [
                names[i] for i in range(len(names)) if (ranks[i] <= 3).all()
            ],
        }
        print(f"  {sample}: top 3 under every method: {report[sample]['features_ranked_top3_by_every_method']}")
    report["consensus_order"] = [names[i] for i in consensus_order]
    return report


# --- Step 3: heatmaps ------------------------------------------------------------------


def heatmaps(ws: Workspace, attributions: dict[str, Attribution], figures_dir: Path) -> dict:
    views = {
        (key, sample): attributions[PRIMARY[key]].select(mask)
        for key in PRIMARY for sample, mask in ws.sample_masks.items()
    }
    vmax = float(np.percentile(np.abs(np.vstack([v.values for v in views.values()])), HEATMAP_CLIP_PERCENTILE))
    rows = ws.sets.explanation
    for (key, sample), view in views.items():
        positions = rows[ws.sample_masks[sample]]
        model = ws.models[key]
        plot_attribution_heatmap(
            view.values, view.feature_names, model.predict_score(ws.rows(positions)), ws.y_binary[positions],
            model.decision_threshold, vmax, figures_dir / f"internals_attribution_heatmap__{key}__{sample}.png",
            title=f"{MODEL_LABEL[key]}: every respondent's attributions ({sample})",
            caption=f"{SHAP_REFERENCE} Red pushes towards \"has an account\", blue away.",
        )
    return {"colour_scale_max_abs_logits": vmax, "clip_percentile": HEATMAP_CLIP_PERCENTILE,
            "attribution": {key: PRIMARY[key] for key in PRIMARY}}


# --- Step 4: true positives vs true negatives ------------------------------------------


def tp_vs_tn(ws: Workspace, attributions: dict[str, Attribution], figures_dir: Path, settings: Settings) -> dict:
    report, panels_by_model = {}, {}
    rows = ws.sets.explanation
    names = list(ws.raw_features)
    for key in PRIMARY:
        model = ws.models[key]
        report[key] = {}
        panels = {}
        for sample, mask in ws.sample_masks.items():
            values = attributions[PRIMARY[key]].values[mask]
            positions = rows[mask]
            predicted = model.predict_score(ws.rows(positions)) >= model.decision_threshold
            actual = ws.y_binary[positions] == 1
            tp, tn = actual & predicted, ~actual & ~predicted
            mean_tp, mean_tn = values[tp].mean(axis=0), values[tn].mean(axis=0)
            group = np.where(tp, 1, np.where(tn, 0, -1))
            keep = group >= 0
            kept_values, kept_group = values[keep], group[keep]
            ci = bootstrap_ci(
                lambda r: kept_values[r][kept_group[r] == 1].mean(axis=0) - kept_values[r][kept_group[r] == 0].mean(axis=0),
                int(keep.sum()), settings.xai.bootstrap_reps, settings.seed,
            )
            panels[f"{sample}: TP n = {int(tp.sum()):,}, TN n = {int(tn.sum()):,}"] = np.column_stack(
                [mean_tp, mean_tn, mean_tp - mean_tn]
            )
            report[key][sample] = {
                "n_tp": int(tp.sum()), "n_tn": int(tn.sum()), "threshold": model.decision_threshold,
                "features": {
                    name: {"mean_tp": float(mean_tp[j]), "mean_tn": float(mean_tn[j]),
                           "difference": float(mean_tp[j] - mean_tn[j]),
                           "difference_ci": [float(ci.lower[j]), float(ci.upper[j])]}
                    for j, name in enumerate(names)
                },
            }
        panels_by_model[key] = panels
    vmax = max(float(np.nanmax(np.abs(m))) for panels in panels_by_model.values() for m in panels.values())
    for key, panels in panels_by_model.items():
        plot_matrix_panels(
            panels, names, ["mean, TP", "mean, TN", "TP - TN"],
            figures_dir / f"internals_tp_vs_tn__{key}.png",
            title=f"{MODEL_LABEL[key]}: average attribution, true positives vs true negatives",
            caption=f"{SHAP_REFERENCE} TP/TN at this model's threshold ({ws.models[key].decision_threshold:.4f}); "
                    f"every row is a training row, so the split is in-sample. One colour scale (+/-{vmax:.2f}) "
                    f"shared by both models' figures.",
            diverging=True, vmax=vmax,
        )
    return report


# --- Step 5a: embedding projections ------------------------------------------------------


def embedding_projections(ws: Workspace, attributions: dict[str, Attribution], figures_dir: Path) -> dict:
    mlp = ws.mlp
    pooled = ws.sample_masks["pooled"]
    pooled_rows = ws.rows(ws.sets.explanation[pooled])
    shap_values = attributions[PRIMARY["mlp"]].values[pooled]
    panels, report = {}, {}
    for feature in EMBEDDING_FEATURES:
        weights = mlp.net.embeddings[mlp.categorical_columns.index(feature)].weight.detach().double().numpy()
        table = mlp.embedding_rows(feature)
        levels = sorted(table, key=table.get)
        trained = weights[[table[level] for level in levels]]
        centre = trained.mean(axis=0)
        _, singular, components = np.linalg.svd(trained - centre, full_matrices=False)
        explained = float((singular[:2] ** 2).sum() / (singular**2).sum())
        coords = (trained - centre) @ components[:2].T
        unknown = (weights[0] - centre) @ components[:2].T
        j = ws.raw_features.index(feature)
        mean_attribution = np.array([
            shap_values[pooled_rows[feature].to_numpy() == level, j].mean()
            if (pooled_rows[feature].to_numpy() == level).any() else np.nan
            for level in levels
        ])
        panels[feature] = {"coords": coords, "labels": levels, "colour": np.nan_to_num(mean_attribution),
                           "unknown": unknown, "explained": explained}
        report[feature] = {
            "variance_explained_by_2_components": explained,
            "levels": {level: {"pc1": float(c[0]), "pc2": float(c[1]),
                               "mean_attribution_pooled": None if np.isnan(m) else float(m),
                               "n_pooled": int((pooled_rows[feature].to_numpy() == level).sum())}
                       for level, c, m in zip(levels, coords, mean_attribution)},
            "unknown_index_projection": [float(unknown[0]), float(unknown[1])],
        }
    vmax = max(float(np.nanmax(np.abs(p["colour"]))) for p in panels.values())
    plot_embedding_projection(
        panels, vmax, figures_dir / "internals_embedding_projection__mlp.png",
        title="PyTorch MLP: learned category embeddings, projected to 2-D (substitute for attention maps)",
        caption=f"{ATTENTION_STATEMENT} Each point is one answer's 6-dim embedding vector, projected with PCA fitted "
                f"on the trained levels; x = the unused 'unknown' row, projected but not fitted. Colour: mean exact "
                f"Shapley attribution of that answer for the MLP on the pooled sample (n = 1,000; levels absent from "
                f"the sample are drawn at 0). Positions are relative within a panel; axes have no units of meaning.",
    )
    return report


# --- Step 5b: interactions ------------------------------------------------------------------


def interactions(ws: Workspace, attribution_dir: Path, attributions: dict[str, Attribution], figures_dir: Path,
                 force: bool) -> dict:
    pooled = ws.sample_masks["pooled"]
    positions = ws.sets.explanation[pooled]
    names = list(ws.raw_features)
    report, mean_abs = {}, {}
    for key, model in ws.models.items():
        path = attribution_dir / f"exact_shapley_interactions__{key}__pooled.npz"
        if path.exists() and not force:
            with np.load(path, allow_pickle=False) as archive:
                matrices, rows, base_value = archive["interactions"], archive["rows"], float(archive["base_value"])
            if not np.array_equal(rows, positions):
                raise ValueError(f"{path.name} is for different rows; re-run with --force")
            print(f"  {key}: reusing {path.name}")
        else:
            print(f"  {key}: exact Shapley interactions for {len(positions):,} rows (about 5 minutes)", flush=True)
            matrices, base_value = exact_shapley_interactions(
                model.predict_logit_codes, model.to_codes(ws.rows(positions)), model.to_codes(ws.background)
            )
            np.savez_compressed(path, interactions=matrices, rows=positions, base_value=base_value,
                                feature_names=np.array(names))
        shapley = attributions[PRIMARY[key]].values[pooled]
        consistency = float(np.abs(matrices.sum(axis=2) - shapley).max())
        averaged = np.abs(matrices).mean(axis=0)
        off = ~np.eye(len(names), dtype=bool)
        pairs = sorted(((averaged[i, j], names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))),
                       reverse=True)
        total = np.abs(matrices).sum(axis=(1, 2))
        report[key] = {
            "n": int(len(positions)),
            "rows_sum_to_shapley_values_max_abs_error": consistency,
            "share_of_abs_attribution_in_interactions": float(np.abs(matrices[:, off]).sum(axis=1).sum() / total.sum()),
            "top_pairs_mean_abs": [{"pair": [a, b], "mean_abs_each_half": float(v)} for v, a, b in pairs[:5]],
            "mean_abs_matrix": {names[i]: {names[j]: float(averaged[i, j]) for j in range(len(names))}
                                for i in range(len(names))},
        }
        shown = averaged.copy()
        np.fill_diagonal(shown, np.nan)
        mean_abs[key] = shown
        print(f"  {key}: {report[key]['share_of_abs_attribution_in_interactions']:.1%} of |attribution| is in "
              f"pairwise interactions; strongest pair {pairs[0][1]} x {pairs[0][2]} ({pairs[0][0]:.3f}); "
              f"rows sum to Shapley values within {consistency:.1e}")

        # The strongest pair, as a faceted dependence plot (rule: the top pair;
        # the feature with the larger main effect is plotted, faceted by the other).
        _, a, b = pairs[0]
        feature, partner = (a, b) if averaged[names.index(a), names.index(a)] >= averaged[names.index(b), names.index(b)] else (b, a)
        rows_raw = ws.rows(positions)
        plot_interaction_facets(
            feature, _as_levels(rows_raw[feature], ws), shapley[:, names.index(feature)],
            partner, _as_levels(rows_raw[partner], ws),
            figures_dir / f"internals_interaction_dependence__{key}.png",
            title=f"{MODEL_LABEL[key]}: does {feature}'s effect depend on {partner}? (pooled)",
            caption=f"{SHAP_REFERENCE} Pair chosen by rule: the largest mean |Shapley interaction| for this model. "
                    f"n = {len(positions):,}.",
        )
        report[key]["plotted_pair"] = {"feature": feature, "faceted_by": partner}
    vmax = max(float(np.nanmax(m)) for m in mean_abs.values())
    plot_matrix_panels(
        {MODEL_LABEL[k]: m for k, m in mean_abs.items()}, names, names,
        figures_dir / "internals_shapley_interactions.png",
        title="Exact Shapley interaction values: mean |interaction| per pair (pooled, n = 1,000)",
        caption=f"{ATTENTION_STATEMENT} Each cell: mean |Phi_ij| over respondents, the part of a feature's "
                f"attribution it earns only jointly with the other (each pair's joint effect is split equally "
                f"between the two cells). Same game as the Shapley values: interventional, raw features, 200 "
                f"background rows. Diagonal (main effects) left blank. One colour scale for both models.",
        diverging=False, vmax=vmax, fmt="{:.3f}",
    )
    return report


def _as_levels(values: pd.Series, ws: Workspace) -> np.ndarray:
    """Categorical values as they are; numeric values as quartile labels."""
    if values.name in ws.xgboost.categorical_levels:
        return values.to_numpy()
    quartile = pd.qcut(values, 4, duplicates="drop")
    return np.array([f"{values.name} {interval.left:.0f}-{interval.right:.0f}" for interval in quartile])


# --- Orchestration --------------------------------------------------------------------------


def main() -> None:
    force = "--force" in sys.argv[1:]
    settings = load_settings()
    xai = settings.xai
    attribution_dir = xai.artifacts_dir / "attributions"

    # 1. Inputs
    ws = load_workspace(settings.paths.raw_train, settings.paths.raw_test, PRODUCTION_DIR, xai.model_dir,
                        xai.artifacts_dir)
    attributions = load_attributions(ws, attribution_dir)
    print(f"Loaded {len(attributions)} attribution sets for {len(ws.sets.explanation):,} rows")

    # 2. Every method on one scale
    print("\n=== Global importance, every method ===")
    global_report = global_comparison(ws, attributions, xai.figures_dir)

    # 3. Heatmaps
    print("\n=== Attribution heatmaps ===")
    heatmap_report = heatmaps(ws, attributions, xai.figures_dir)

    # 4. True positives vs true negatives
    print("=== True positives vs true negatives ===")
    tp_tn_report = tp_vs_tn(ws, attributions, xai.figures_dir, settings)

    # 5. Substitutes for attention maps
    print("\n=== Embedding projections (MLP) ===")
    embedding_report = embedding_projections(ws, attributions, xai.figures_dir)
    print("=== Exact Shapley interactions ===")
    interaction_report = interactions(ws, attribution_dir, attributions, xai.figures_dir, force)

    # 6. Summary
    summary = {
        "attention_maps": ATTENTION_STATEMENT,
        "note": "attributions describe the models, not causes; scores are ranking scores, not calibrated "
                "probabilities; every explained row is a training row",
        "global_importance_all_methods": global_report,
        "heatmaps": heatmap_report,
        "tp_vs_tn": tp_tn_report,
        "embedding_projections": embedding_report,
        "interactions": interaction_report,
    }
    path = xai.artifacts_dir / "internals_summary.json"
    path.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"\nWrote {path} and figures to {xai.figures_dir}")


if __name__ == "__main__":
    main()
