"""XAI Phase 4 (Task 2): Captum -- Integrated Gradients, Feature Ablation, layer attribution.

The script is seven steps, in this order (see `main`):

    1. load the models, data and fixed sample sets
    2. define the two baselines ("unknown" and "modal") and score them
    3. choose the number of Integrated Gradients steps by a stated rule
    4. Integrated Gradients on the MLP, every explained row, both baselines
    5. Feature Ablation on BOTH models, every explained row, both baselines
    6. layer attribution: hidden units and embedding dimensions of the MLP
    7. how much the answers depend on the baseline; figures; summary

Writes:

    reports/xai/attributions/<method>__<model>__<baseline>.npz
    reports/xai/captum_summary.json       every number quoted later
    reports/figures/xai/captum_*.png

Every method here compares a respondent with ONE baseline point (unlike
SHAP's 200-row background), in logits, with the MLP in eval mode (dropout
off). Feature Ablation removes one feature at a time; Integrated Gradients
integrates along a straight path. Hidden units have no meaning of their own:
step 6 describes how the network computes, not anything about respondents.
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "config"))
from settings import Settings, load_settings  # noqa: E402

from fin_inclusion.interpretability.attribution import LOGIT_SPACE, Attribution  # noqa: E402
from fin_inclusion.interpretability.bootstrap import bootstrap_ci, bootstrap_mean_abs_importance  # noqa: E402
from fin_inclusion.interpretability.captum_explainers import (  # noqa: E402
    BASELINES,
    FeatureAblationExplainer,
    IntegratedGradientsExplainer,
    baseline_codes,
    describe_baseline,
    hidden_unit_attributions,
    unit_names,
)
from fin_inclusion.interpretability.checks import check_efficiency  # noqa: E402
from fin_inclusion.interpretability.comparison import summarize_agreement  # noqa: E402
from fin_inclusion.interpretability.grouping import (  # noqa: E402
    group_attribution,
    importance_share,
    mean_abs_importance,
)
from fin_inclusion.interpretability.plots import (  # noqa: E402
    plot_convergence,
    plot_importance_bars,
    plot_paired_shares,
    plot_unit_profiles,
)
from fin_inclusion.interpretability.workspace import Workspace, load_workspace  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PRODUCTION_DIR = ROOT / "models" / "production"

# Step 3's rule: the smallest step count on this grid whose largest
# |convergence delta| on the oracle rows is within DELTA_TARGET, for both
# baselines. Fixed before looking at the results.
N_STEPS_GRID = [32, 64, 128, 256, 512, 1024]
DELTA_TARGET = 0.01  # logits
CHUNK_ROWS = 500
MODEL_LABEL = {"xgboost": "XGBoost (shipped)", "mlp": "PyTorch MLP"}


# --- Step 2: baselines -----------------------------------------------------------


def define_baselines(ws: Workspace) -> tuple[dict[str, np.ndarray], dict]:
    training_codes = ws.mlp.to_codes(ws.X_raw)
    codes = {kind: baseline_codes(ws.mlp, training_codes, kind) for kind in BASELINES}
    report = {
        kind: {
            "description": describe_baseline(ws.mlp, c, kind),
            "logit": {key: float(model.predict_logit_codes(c[None])[0]) for key, model in ws.models.items()},
        }
        for kind, c in codes.items()
    }
    report["mean_training_logit"] = {key: float(model.predict_logit(ws.X_raw).mean()) for key, model in ws.models.items()}
    for kind in BASELINES:
        print(f"  {kind}: logit XGBoost {report[kind]['logit']['xgboost']:+.3f}, MLP {report[kind]['logit']['mlp']:+.3f}")
    return codes, report


# --- Step 3: integration steps ----------------------------------------------------


def choose_n_steps(ws: Workspace, baselines: dict[str, np.ndarray], oracle_rows: np.ndarray) -> tuple[int, dict]:
    X = ws.rows(oracle_rows)
    table = {kind: {} for kind in baselines}
    for n in N_STEPS_GRID:
        for kind, codes in baselines.items():
            _, delta = IntegratedGradientsExplainer(ws.mlp, codes, kind, n).explain_with_delta(X)
            table[kind][n] = {"max_abs_delta": float(np.abs(delta).max()), "mean_abs_delta": float(np.abs(delta).mean())}
    passing = [n for n in N_STEPS_GRID if all(table[k][n]["max_abs_delta"] <= DELTA_TARGET for k in baselines)]
    if not passing:
        raise RuntimeError(f"No n_steps in {N_STEPS_GRID} brings |delta| within {DELTA_TARGET}; widen the grid")
    chosen = passing[0]
    for kind in baselines:
        print(f"  {kind}: max |delta| by n_steps: "
              + ", ".join(f"{n}: {table[kind][n]['max_abs_delta']:.1e}" for n in N_STEPS_GRID))
    print(f"  chosen n_steps = {chosen} (rule: smallest with max |delta| <= {DELTA_TARGET} for both baselines)")
    report = {"grid": N_STEPS_GRID, "target_max_abs_delta": DELTA_TARGET, "rows": oracle_rows.tolist(),
              "by_baseline": {k: {str(n): v for n, v in t.items()} for k, t in table.items()}, "chosen": chosen}
    return chosen, report


# --- Steps 4-5: attributions for every explained row ------------------------------


def in_chunks(explain, ws: Workspace, positions: np.ndarray):
    for start in range(0, len(positions), CHUNK_ROWS):
        yield explain(ws.rows(positions[start:start + CHUNK_ROWS]))


def run_integrated_gradients(
    ws: Workspace, baselines: dict[str, np.ndarray], n_steps: int, out_dir: Path
) -> tuple[dict, dict, dict]:
    """IG on the MLP. Returns grouped attributions, per-dimension
    attributions, and the convergence report, all keyed by baseline."""
    grouped, encoded, report = {}, {}, {}
    rows = ws.sets.explanation
    for kind, codes in baselines.items():
        explainer = IntegratedGradientsExplainer(ws.mlp, codes, kind, n_steps)
        parts = list(in_chunks(explainer.explain_with_delta, ws, rows))
        delta = np.concatenate([d for _, d in parts])
        first = parts[0][0]
        enc = replace(first, values=np.vstack([a.values for a, _ in parts]),
                      metadata={**first.metadata, "rows": rows.tolist()})
        grp = group_attribution(enc, explainer.grouping)
        grp = replace(grp, metadata={**grp.metadata, "max_abs_convergence_delta": float(np.abs(delta).max()),
                                     "mean_abs_convergence_delta": float(np.abs(delta).mean())})
        enc.save(out_dir / f"integrated_gradients__mlp__{kind}__embedding_dims.npz")
        grp.save(out_dir / f"integrated_gradients__mlp__{kind}.npz")
        completeness = check_efficiency(grp, ws.mlp.predict_logit(ws.rows(rows)), atol=DELTA_TARGET)
        report[kind] = {
            "base_value": grp.base_value,
            "max_abs_delta": float(np.abs(delta).max()),
            "mean_abs_delta": float(np.abs(delta).mean()),
            "share_rows_within_target": float((np.abs(delta) <= DELTA_TARGET).mean()),
            "completeness_check": completeness.to_dict(),
        }
        print(f"  IG {kind}: max |delta| {report[kind]['max_abs_delta']:.2e} over {len(rows):,} rows, "
              f"{report[kind]['share_rows_within_target']:.1%} within {DELTA_TARGET}")
        grouped[kind], encoded[kind] = grp, enc
    return grouped, encoded, report


def run_feature_ablation(ws: Workspace, baselines: dict[str, np.ndarray], out_dir: Path) -> tuple[dict, dict]:
    """Ablation on both models. Returns `{(model key, baseline): Attribution}` and a
    report of how far each is from adding up (it is not meant to)."""
    attributions, report = {}, {}
    rows = ws.sets.explanation
    for key, model in ws.models.items():
        logit = model.predict_logit(ws.rows(rows))
        for kind, codes in baselines.items():
            explainer = FeatureAblationExplainer(model, codes, kind)
            parts = list(in_chunks(explainer.explain, ws, rows))
            attribution = replace(parts[0], values=np.vstack([p.values for p in parts]),
                                  metadata={**parts[0].metadata, "rows": rows.tolist()})
            attribution.save(out_dir / f"feature_ablation__{key}__{kind}.npz")
            gap = attribution.values.sum(axis=1) + attribution.base_value - logit
            report[f"{key}__{kind}"] = {
                "base_value": attribution.base_value,
                "non_additivity_mean_abs": float(np.abs(gap).mean()),
                "non_additivity_max_abs": float(np.abs(gap).max()),
                "mean_abs_logit_minus_base": float(np.abs(logit - attribution.base_value).mean()),
            }
            attributions[(key, kind)] = attribution
            print(f"  ablation {key:8s} {kind:8s}: sum of attributions misses f(x) - f(baseline) by "
                  f"{report[f'{key}__{kind}']['non_additivity_mean_abs']:.3f} logits on average")
    return attributions, report


# --- Step 6: layer attribution ------------------------------------------------------


def run_layer_attribution(
    ws: Workspace, baselines: dict[str, np.ndarray], n_steps: int, ig_encoded: dict, out_dir: Path
) -> tuple[dict, dict]:
    """Hidden units (conductance, gradient x activation) and embedding dimensions (from IG)."""
    rows = ws.sets.explanation
    X = ws.rows(rows)
    logit = ws.mlp.predict_logit(X)
    report: dict = {"hidden_units": {}, "embedding_dimensions": {}}
    unit_profiles = {}
    for kind, codes in baselines.items():
        layers = hidden_unit_attributions(ws.mlp, X, codes, n_steps)
        base_value = float(ws.mlp.predict_logit_codes(codes[None])[0])
        for name, result in layers.items():
            names = unit_names(name, result["conductance"].shape[1])
            conductance = Attribution(
                values=result["conductance"], base_value=base_value, feature_names=names,
                method="layer_conductance", model_name=ws.mlp.name, target_space=LOGIT_SPACE,
                baseline_description=f"{describe_baseline(ws.mlp, codes, kind)}; {n_steps} steps; flow of "
                                     f"f(x) - f(baseline) through each unit of {name}",
                metadata={"baseline": kind, "layer": name, "n_steps": n_steps, "rows": rows.tolist()},
            )
            conductance.save(out_dir / f"layer_conductance__mlp__{name}__{kind}.npz")
            dead = result["activation"].max(axis=0) == 0
            importance = np.abs(result["conductance"]).mean(axis=0)
            top10 = np.argsort(-importance)[:10]
            completeness = check_efficiency(conductance, logit, atol=DELTA_TARGET)
            gxa_importance = np.abs(result["gradient_x_activation"]).mean(axis=0)
            report["hidden_units"].setdefault(name, {})[kind] = {
                "n_units": int(len(importance)),
                "never_active_units": int(dead.sum()),
                "share_rows_active_by_unit_median": float(np.median((result["activation"] > 0).mean(axis=0))),
                "top10_units": [int(u) for u in top10],
                "top10_share_of_mean_abs_conductance": float(importance[top10].sum() / importance.sum()),
                "completeness_check": completeness.to_dict(),
                "spearman_conductance_vs_gradient_x_activation_unit_importance": float(
                    spearmanr(importance, gxa_importance).statistic
                ),
            }
            if kind == "modal":
                unit_profiles[name] = (importance, dead)
                Attribution(
                    values=result["gradient_x_activation"], base_value=0.0, feature_names=names,
                    method="layer_gradient_x_activation", model_name=ws.mlp.name, target_space=LOGIT_SPACE,
                    baseline_description=f"no baseline: activation x gradient of the logit at the respondent, "
                                         f"each unit of {name}; not an additive decomposition",
                    metadata={"layer": name, "rows": rows.tolist()},
                ).save(out_dir / f"layer_gradient_x_activation__mlp__{name}.npz")
            print(f"  {name} ({kind}): {int(dead.sum())}/{len(importance)} units never active, top 10 carry "
                  f"{report['hidden_units'][name][kind]['top10_share_of_mean_abs_conductance']:.0%}, "
                  f"completeness max |error| {completeness.max_abs_error:.1e}")

    # Embedding dimensions: how a feature's IG attribution spreads over its 6 dims.
    pooled = ws.sample_masks["pooled"]
    for kind, enc in ig_encoded.items():
        values = np.abs(enc.values[pooled]).mean(axis=0)
        per_feature = {}
        for feature in ws.mlp.categorical_columns:
            dims = [j for j, n in enumerate(enc.feature_names) if n.startswith(f"{feature}[emb")]
            dim_importance = values[dims]
            per_feature[feature] = {
                "mean_abs_by_dimension": [float(v) for v in dim_importance],
                "largest_dimension_share": float(dim_importance.max() / dim_importance.sum()),
            }
        report["embedding_dimensions"][kind] = per_feature
    return report, unit_profiles


# --- Step 7: baseline sensitivity -------------------------------------------------


def baseline_sensitivity(ws: Workspace, attributions: dict[str, dict[str, Attribution]], settings: Settings) -> dict:
    """For each (method, model): importance under each baseline (with
    bootstrap intervals) and how well the two baselines agree."""
    report = {}
    for label, by_baseline in attributions.items():
        report[label] = {}
        for sample, mask in ws.sample_masks.items():
            subsets = {kind: a.select(mask) for kind, a in by_baseline.items()}
            unknown, modal = subsets["unknown"], subsets["modal"]
            agreement = summarize_agreement(unknown, modal)
            importance_u, importance_m = np.abs(unknown.values), np.abs(modal.values)
            ci = bootstrap_ci(
                lambda r: spearmanr(importance_u[r].mean(axis=0), importance_m[r].mean(axis=0)).statistic,
                len(importance_u), settings.xai.bootstrap_reps, settings.seed,
            )
            agreement["global"]["spearman_ci"] = [float(ci.lower), float(ci.upper)]
            per_baseline = {}
            for kind, subset in subsets.items():
                boot = bootstrap_mean_abs_importance(subset.values, settings.xai.bootstrap_reps, settings.seed)
                share = importance_share(mean_abs_importance(subset))
                per_baseline[kind] = {
                    "ranking": list(share.sort_values(ascending=False).index),
                    "features": {
                        name: {"share": float(share[name]), "mean_abs": float(boot.estimate[j]),
                               "ci_low": float(boot.lower[j]), "ci_high": float(boot.upper[j])}
                        for j, name in enumerate(subset.feature_names)
                    },
                }
            report[label][sample] = {"by_baseline": per_baseline, "unknown_vs_modal": agreement}
        g = report[label]["pooled"]["unknown_vs_modal"]
        print(f"  {label:32s} pooled: global Spearman {g['global']['spearman']:+.2f} "
              f"[{g['global']['spearman_ci'][0]:+.2f}, {g['global']['spearman_ci'][1]:+.2f}], "
              f"same top feature {g['same_top_feature_share']:.0%}, per-row Spearman median "
              f"{g['instance_spearman']['median']:+.2f}")
    return report


# --- Figures --------------------------------------------------------------------------


def make_figures(ws: Workspace, n_steps_report: dict, ig: dict, ablation: dict, sensitivity: dict,
                 unit_profiles: dict, figures_dir: Path, settings: Settings) -> None:
    grid = n_steps_report["grid"]
    plot_convergence(
        grid, {kind: [n_steps_report["by_baseline"][kind][str(n)]["max_abs_delta"] for n in grid] for kind in BASELINES},
        figures_dir / "captum_ig_convergence.png",
        title="Integrated Gradients: worst convergence delta vs integration steps (MLP)",
        xlabel="integration steps (Gauss-Legendre)", ylabel="max |convergence delta| (logits)",
        caption=f"Convergence delta = sum of attributions - (f(x) - f(baseline)), worst case over "
                f"{len(n_steps_report['rows'])} rows. Chosen: {n_steps_report['chosen']} steps, the smallest on "
                f"the grid within the target for both baselines.",
        threshold=DELTA_TARGET,
    )
    focus = ws.sets.focus_country
    for kind in BASELINES:
        attribution = ig[kind]
        plot_importance_bars(
            {"Pooled": bootstrap_mean_abs_importance(attribution.values[ws.sample_masks["pooled"]],
                                                     settings.xai.bootstrap_reps, settings.seed),
             f"{focus} only": bootstrap_mean_abs_importance(attribution.values[ws.sample_masks[focus.lower()]],
                                                            settings.xai.bootstrap_reps, settings.seed)},
            attribution.feature_names, figures_dir / f"captum_global_importance__ig__mlp__{kind}.png",
            title=f"Global importance. PyTorch MLP: Integrated Gradients, {kind} baseline",
            caption=f"Reference: one baseline point ({attribution.baseline_description.split(';')[0]}); "
                    f"f(baseline) = {attribution.base_value:+.2f} logits. Every explained row is a training row.",
        )
    panels = {}
    for label, title in (("integrated_gradients__mlp", "IG, MLP"), ("feature_ablation__mlp", "Ablation, MLP"),
                         ("feature_ablation__xgboost", "Ablation, XGBoost")):
        by = sensitivity[label]["pooled"]["by_baseline"]
        names = list(by["modal"]["features"])
        panels[title] = (names, {f"{kind} baseline": np.array([by[kind]["features"][n]["share"] for n in names])
                                 for kind in ("modal", "unknown")})
    plot_paired_shares(
        panels, figures_dir / "captum_baseline_sensitivity.png",
        title="How much does the baseline change each feature's importance? (pooled, n = 1,000)",
        caption="Share of total mean |attribution| per feature under the modal baseline (every categorical at its "
                "most frequent level) and the unknown baseline (every categorical at a never-seen category); "
                "numerics at the training mean in both. A long line = importance that depends on the baseline.",
    )
    plot_unit_profiles(
        unit_profiles, figures_dir / "captum_hidden_units__mlp.png",
        title="PyTorch MLP: how concentrated is the computation? (LayerConductance, modal baseline)",
        caption="Mean |conductance| per hidden unit over the explained rows, units sorted. Hidden units have no "
                "meaning of their own; this describes the network's mechanics, not respondents.",
    )


# --- Orchestration ------------------------------------------------------------------------


def main() -> None:
    settings = load_settings()
    xai = settings.xai
    attribution_dir = xai.artifacts_dir / "attributions"

    # 1. Models, data, sample sets
    ws = load_workspace(settings.paths.raw_train, settings.paths.raw_test, PRODUCTION_DIR, xai.model_dir,
                        xai.artifacts_dir)
    oracle_rows = np.array(json.loads((xai.artifacts_dir / "shap_summary.json").read_text(encoding="utf-8"))
                           ["oracle"]["rows"])
    print(f"Loaded {ws.xgboost.name} and {ws.mlp.name}; explaining {len(ws.sets.explanation):,} rows")

    # 2. Baselines
    print("\n=== Baselines ===")
    baselines, baseline_report = define_baselines(ws)

    # 3. Integration steps
    print("\n=== Choosing Integrated Gradients steps ===")
    n_steps, n_steps_report = choose_n_steps(ws, baselines, oracle_rows)

    # 4. Integrated Gradients
    print("\n=== Integrated Gradients (MLP) ===")
    ig, ig_encoded, ig_report = run_integrated_gradients(ws, baselines, n_steps, attribution_dir)

    # 5. Feature Ablation
    print("\n=== Feature Ablation (both models) ===")
    ablation, ablation_report = run_feature_ablation(ws, baselines, attribution_dir)

    # 6. Layer attribution
    print("\n=== Layer attribution (MLP) ===")
    layer_report, unit_profiles = run_layer_attribution(ws, baselines, n_steps, ig_encoded, attribution_dir)

    # 7. Baseline sensitivity, figures, summary
    print("\n=== Sensitivity to the baseline: unknown vs modal ===")
    sensitivity = baseline_sensitivity(ws, {
        "integrated_gradients__mlp": ig,
        "feature_ablation__mlp": {kind: ablation[("mlp", kind)] for kind in BASELINES},
        "feature_ablation__xgboost": {kind: ablation[("xgboost", kind)] for kind in BASELINES},
    }, settings)
    make_figures(ws, n_steps_report, ig, ablation, sensitivity, unit_profiles, xai.figures_dir, settings)
    summary = {
        "target_space": "logit",
        "note": "each method compares a respondent with ONE baseline point; attributions describe the models, "
                "not causes; scores are ranking scores, not calibrated probabilities; every explained row is a "
                "training row; MLP in eval mode (dropout off)",
        "baselines": baseline_report,
        "integrated_gradients": {"n_steps": n_steps_report, "full_run": ig_report},
        "feature_ablation": ablation_report,
        "layer_attribution": layer_report,
        "baseline_sensitivity": sensitivity,
    }
    path = xai.artifacts_dir / "captum_summary.json"
    path.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"\nWrote {path} and figures to {xai.figures_dir}")


if __name__ == "__main__":
    main()
