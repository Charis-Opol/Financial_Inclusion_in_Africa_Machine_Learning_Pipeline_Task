"""XAI Phase 3 (Task 1): SHAP explanations of both models, checked against exact Shapley values.

The script is six steps, in this order (see `main`):

    1. load the models, data and fixed sample sets
    2. compute SHAP attributions for every explained row
    3. ORACLE: brute-force exact Shapley values on a few rows, and how close
       each SHAP method gets to them
    4. efficiency check: do attributions + base value equal the model's logit?
    5. global importance (pooled and Uganda, with bootstrap intervals)
    6. figures: importance bars, beeswarms, local waterfalls, dependence plots

Writes:

    reports/xai/attributions/<method>__<model>.npz   raw-feature attributions
    reports/xai/shap_summary.json                    every number quoted later
    reports/figures/xai/shap_*.png                   figures

Step 2 is the slow one (about 0.3 s per row per model for exact KernelSHAP).
Finished attributions are reused on the next run; pass --force to recompute.

Reference and value function, for every method here: the same 200 background
rows; INTERVENTIONAL value function (an absent feature takes a background
row's value, independently of the other features); logit space.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "config"))
from settings import Settings, load_settings  # noqa: E402

from fin_inclusion.interpretability.attribution import Attribution  # noqa: E402
from fin_inclusion.interpretability.base import Explainer  # noqa: E402
from fin_inclusion.interpretability.bootstrap import bootstrap_mean_abs_importance  # noqa: E402
from fin_inclusion.interpretability.checks import check_efficiency  # noqa: E402
from fin_inclusion.interpretability.grouping import importance_share, mean_abs_importance  # noqa: E402
from fin_inclusion.interpretability.plots import (  # noqa: E402
    plot_beeswarm,
    plot_dependence,
    plot_importance_bars,
    plot_waterfall,
)
from fin_inclusion.interpretability.sampling import LOCAL_EXAMPLE_RULES, select_local_examples  # noqa: E402
from fin_inclusion.interpretability.shap_explainers import (  # noqa: E402
    GradientShapExplainer,
    KernelShapExplainer,
    TreeShapExplainer,
)
from fin_inclusion.interpretability.shapley import ExactShapleyExplainer  # noqa: E402
from fin_inclusion.interpretability.workspace import Workspace, load_workspace  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PRODUCTION_DIR = ROOT / "models" / "production"

# The main SHAP result per model: the one the figures and later phases use.
PRIMARY_METHOD = {"xgboost": "tree_shap", "mlp": "kernel_shap"}
MODEL_LABEL = {"xgboost": "XGBoost (shipped)", "mlp": "PyTorch MLP"}
METHOD_LABEL = {
    "tree_shap": "TreeSHAP, grouped to raw features",
    "kernel_shap": "KernelSHAP, all coalitions enumerated (exact)",
    "gradient_shap": "GradientSHAP (expected gradients)",
}

# Efficiency tolerances, in logits. Stated per method because "equal" means
# different things: the first three are exact up to float32 model arithmetic;
# GradientSHAP is a Monte Carlo estimate and is REPORTED against its
# tolerance, not required to meet it.
EFFICIENCY_ATOL = {"exact_shapley": 1e-4, "kernel_shap": 1e-4, "tree_shap": 1e-3, "gradient_shap": 0.05}
MUST_BE_EFFICIENT = {"exact_shapley", "kernel_shap", "tree_shap"}

# 20 is included deliberately: with 10 features it is too few coalitions for
# KernelSHAP's regression to be determined (shap warns "singular"), and the
# result is meaningless. Kept in the table as the failure it is.
KERNEL_CONVERGENCE_NSAMPLES = [20, 50, 100, 200, 500]
GRADIENT_CONVERGENCE_NSAMPLES = [200, 1000, 5000]
N_DEPENDENCE_PLOTS = 3
CHUNK_ROWS = 250

# shap logs a "200 background samples could be slow" notice on every call;
# the background size is a deliberate choice (decisions log D2.5).
logging.getLogger("shap").setLevel(logging.ERROR)


# --- Step 2: attributions for every explained row ----------------------------


def explain_in_chunks(explainer: Explainer, ws: Workspace, positions: np.ndarray, label: str) -> Attribution:
    """Explains `positions` a chunk at a time (progress is printed), and
    records which rows and which background the result belongs to."""
    parts = []
    for start in range(0, len(positions), CHUNK_ROWS):
        parts.append(explainer.explain(ws.rows(positions[start:start + CHUNK_ROWS])))
        print(f"  {label}: {min(start + CHUNK_ROWS, len(positions)):,}/{len(positions):,} rows", flush=True)
    first = parts[0]
    return replace(
        first,
        values=np.vstack([p.values for p in parts]),
        metadata={**first.metadata, "rows": positions.tolist(), "background_rows": ws.sets.background.tolist()},
    )


def compute_or_reuse(path: Path, explainer_factory, ws: Workspace, positions: np.ndarray, force: bool) -> Attribution:
    """Loads a finished artifact if it is for exactly these rows and this
    background; otherwise computes and saves it."""
    label = path.stem
    if path.exists() and not force:
        cached = Attribution.load(path)
        if (cached.metadata.get("rows") == positions.tolist()
                and cached.metadata.get("background_rows") == ws.sets.background.tolist()):
            print(f"  {label}: reusing {path.name}")
            return cached
        print(f"  {label}: {path.name} is for different rows or background; recomputing")
    attribution = explain_in_chunks(explainer_factory(), ws, positions, label)
    attribution.save(path)
    return attribution


def compute_attributions(ws: Workspace, out_dir: Path, seed: int, force: bool) -> dict[tuple[str, str], Attribution]:
    """`{(method, model key): Attribution}` over all explanation rows."""
    rows, background = ws.sets.explanation, ws.background
    jobs = {
        ("tree_shap", "xgboost"): lambda: TreeShapExplainer(ws.xgboost, background),
        ("kernel_shap", "xgboost"): lambda: KernelShapExplainer(ws.xgboost, background, seed=seed),
        ("kernel_shap", "mlp"): lambda: KernelShapExplainer(ws.mlp, background, seed=seed),
        ("gradient_shap", "mlp"): lambda: GradientShapExplainer(ws.mlp, background, seed=seed),
    }
    print(f"\n=== SHAP attributions for {len(rows):,} rows ===")
    return {
        key: compute_or_reuse(out_dir / f"{key[0]}__{key[1]}.npz", factory, ws, rows, force)
        for key, factory in jobs.items()
    }


# --- Step 3: the exact-Shapley oracle ----------------------------------------


def error_stats(estimate: np.ndarray, truth: np.ndarray) -> dict:
    error = np.abs(estimate - truth)
    return {"mean_abs_error": float(error.mean()), "max_abs_error": float(error.max())}


def run_oracle(
    ws: Workspace, attributions: dict, out_dir: Path, settings: Settings
) -> tuple[dict, dict[str, Attribution]]:
    """Brute-force Shapley values on `oracle_instances` rows of the pooled
    sample, and each SHAP method's distance from them."""
    rng = np.random.default_rng(settings.seed)
    oracle_rows = np.sort(rng.choice(ws.sets.pooled, size=settings.xai.oracle_instances, replace=False))
    in_explanation = np.searchsorted(ws.sets.explanation, oracle_rows)
    X_oracle = ws.rows(oracle_rows)
    print(f"\n=== Oracle: exact Shapley on {len(oracle_rows)} rows ===")

    report: dict = {"rows": oracle_rows.tolist(), "n_coalitions": 2 ** len(ws.raw_features)}
    oracles: dict[str, Attribution] = {}
    for key, model in ws.models.items():
        oracle = ExactShapleyExplainer(model, ws.background).explain(X_oracle)
        oracle = replace(oracle, metadata={**oracle.metadata, "rows": oracle_rows.tolist(),
                                           "background_rows": ws.sets.background.tolist()})
        oracle.save(out_dir / f"exact_shapley__{key}__oracle_rows.npz")
        oracles[key] = oracle
        truth = oracle.values

        kernel = attributions[("kernel_shap", key)]
        convergence = {
            str(n): error_stats(KernelShapExplainer(model, ws.background, nsamples=n, seed=settings.seed)
                                .explain(X_oracle).values, truth)
            for n in KERNEL_CONVERGENCE_NSAMPLES
        }
        convergence[f"{kernel.metadata['nsamples']} (all coalitions)"] = error_stats(kernel.values[in_explanation], truth)
        report[key] = {
            "mean_abs_oracle_attribution": float(np.abs(truth).mean()),
            "base_value": oracle.base_value,
            "kernel_shap_base_value_abs_diff": abs(kernel.base_value - oracle.base_value),
            "kernel_shap_error_by_nsamples": convergence,
        }
        print(f"  {key}: KernelSHAP mean |error| by nsamples: "
              + ", ".join(f"{n}: {s['mean_abs_error']:.2e}" for n, s in convergence.items()))

    # XGBoost: grouped 40-column TreeSHAP against the 10-feature game.
    tree = attributions[("tree_shap", "xgboost")]
    tree_values, truth = tree.values[in_explanation], oracles["xgboost"].values
    per_feature = np.abs(tree_values - truth).mean(axis=0)
    report["xgboost"]["tree_shap_grouped_vs_oracle"] = {
        **error_stats(tree_values, truth),
        "base_value_abs_diff": abs(tree.base_value - oracles["xgboost"].base_value),
        "mean_abs_error_by_feature": dict(zip(ws.raw_features, map(float, per_feature))),
        "mean_abs_oracle_by_feature": dict(zip(ws.raw_features, map(float, np.abs(truth).mean(axis=0)))),
        "relative_mean_abs_error": float(np.abs(tree_values - truth).mean() / np.abs(truth).mean()),
        "same_sign_share": float((np.sign(tree_values) == np.sign(truth)).mean()),
        "same_top_feature_share": float(
            (np.abs(tree_values).argmax(axis=1) == np.abs(truth).argmax(axis=1)).mean()
        ),
    }
    gap = report["xgboost"]["tree_shap_grouped_vs_oracle"]
    print(f"  xgboost: grouped TreeSHAP vs oracle: mean |diff| {gap['mean_abs_error']:.4f}, "
          f"max {gap['max_abs_error']:.4f} ({gap['relative_mean_abs_error']:.1%} of mean |oracle|)")

    # MLP: GradientSHAP is a sampled estimate. More samples should shrink its
    # efficiency error (noise); its distance from the oracle need not go to
    # zero, because expected gradients is a different quantity from the
    # Shapley value. Both are recorded so the two effects can be told apart.
    truth, logit = oracles["mlp"].values, ws.mlp.predict_logit(X_oracle)
    report["mlp"]["gradient_shap_by_nsamples"] = {}
    for n in GRADIENT_CONVERGENCE_NSAMPLES:
        gradient = GradientShapExplainer(ws.mlp, ws.background, nsamples=n, seed=settings.seed).explain(X_oracle)
        efficiency = check_efficiency(gradient, logit, atol=EFFICIENCY_ATOL["gradient_shap"])
        report["mlp"]["gradient_shap_by_nsamples"][str(n)] = {
            "vs_oracle": error_stats(gradient.values, truth),
            "efficiency_mean_abs_error": efficiency.mean_abs_error,
            "efficiency_max_abs_error": efficiency.max_abs_error,
        }
    print("  mlp: GradientSHAP by nsamples (mean |error| vs oracle / mean efficiency error): "
          + ", ".join(f"{n}: {s['vs_oracle']['mean_abs_error']:.4f} / {s['efficiency_mean_abs_error']:.4f}"
                      for n, s in report["mlp"]["gradient_shap_by_nsamples"].items()))
    report["kernel_shap_note"] = (
        f"nsamples={KERNEL_CONVERGENCE_NSAMPLES[0]} is too few coalitions for {len(ws.raw_features)} features: "
        "the regression is singular and the result is invalid, not merely imprecise"
    )
    return report, oracles


# --- Step 4: efficiency -------------------------------------------------------


def check_all_efficiency(ws: Workspace, attributions: dict, oracles: dict[str, Attribution]) -> dict:
    """Attributions + base value against the model's own logit, for every artifact."""
    print("\n=== Efficiency: sum of attributions + base value == model logit ===")
    report = {}
    everything = {**attributions, **{("exact_shapley", key): oracle for key, oracle in oracles.items()}}
    for (method, key), attribution in everything.items():
        logit = ws.models[key].predict_logit(ws.rows(np.array(attribution.metadata["rows"])))
        result = check_efficiency(attribution, logit, atol=EFFICIENCY_ATOL[method])
        report[f"{method}__{key}"] = result.to_dict()
        print(f"  {method:14s} {key:8s} max |error| {result.max_abs_error:.2e} "
              f"(atol {result.atol:g}) {'ok' if result.passed else 'NOT within tolerance'}")
        if method in MUST_BE_EFFICIENT:
            result.raise_if_failed()
    return report


# --- Step 5: global importance -------------------------------------------------


def sample_masks(ws: Workspace) -> dict[str, np.ndarray]:
    """Row masks into the explanation set: pooled sample, and the focus country."""
    rows = ws.sets.explanation
    return {"pooled": np.isin(rows, ws.sets.pooled), ws.sets.focus_country.lower(): np.isin(rows, ws.sets.focus)}


def global_importance(ws: Workspace, attributions: dict, settings: Settings) -> tuple[dict, dict]:
    """Mean |attribution| per raw feature, with bootstrap intervals, per method, model and sample."""
    report, results = {}, {}
    for (method, key), attribution in attributions.items():
        for sample, mask in sample_masks(ws).items():
            subset = attribution.select(mask)
            boot = bootstrap_mean_abs_importance(subset.values, settings.xai.bootstrap_reps, settings.seed)
            share = importance_share(mean_abs_importance(subset))
            results[(method, key, sample)] = boot
            report.setdefault(f"{method}__{key}", {})[sample] = {
                "n": boot.n_instances,
                "ci_level": boot.level,
                "bootstrap_reps": boot.reps,
                "features": {
                    name: {"mean_abs": float(boot.estimate[j]), "ci_low": float(boot.lower[j]),
                           "ci_high": float(boot.upper[j]), "share": float(share[name])}
                    for j, name in enumerate(attribution.feature_names)
                },
                "ranking": list(share.sort_values(ascending=False).index),
            }
    return report, results


# --- Step 6: figures -----------------------------------------------------------


def reference_note(ws: Workspace) -> str:
    return (f"Reference: mean logit over {len(ws.sets.background)} background rows (pooled, stratified by "
            f"country and target); interventional value function.")


def plot_global_figures(ws: Workspace, attributions: dict, importance: dict, figures_dir: Path) -> None:
    focus = ws.sets.focus_country
    ordered = {name: "dark = larger" for name in ws.raw_features if name not in ws.xgboost.categorical_levels}
    ordered |= {name: f"dark = {levels[1]}" for name, levels in ws.xgboost.categorical_levels.items() if len(levels) == 2}
    codes = ws.xgboost.to_codes(ws.rows(ws.sets.explanation))
    masks = sample_masks(ws)

    for key, method in PRIMARY_METHOD.items():
        attribution = attributions[(method, key)]
        title = f"{MODEL_LABEL[key]}: {METHOD_LABEL[method]}"
        plot_importance_bars(
            {"Pooled": importance[(method, key, "pooled")], f"{focus} only": importance[(method, key, focus.lower())]},
            attribution.feature_names, figures_dir / f"shap_global_importance__{key}.png",
            title=f"Global importance. {title}",
            caption=f"{reference_note(ws)} Every explained row is a training row. {focus} is compared with the "
                    f"pooled reference, not a {focus}-only one.",
        )
        for sample, mask in masks.items():
            plot_beeswarm(
                attribution.values[mask], codes[mask], attribution.feature_names, ordered,
                figures_dir / f"shap_beeswarm__{key}__{sample}.png",
                title=f"{title} ({sample})", caption=reference_note(ws),
            )


def plot_local_figures(ws: Workspace, attributions: dict, figures_dir: Path) -> dict:
    """Waterfalls for five rule-chosen respondents; the same five for both models."""
    scores = {key: model.predict_score(ws.X_raw) for key, model in ws.models.items()}
    examples = select_local_examples(
        ws.sets.pooled, ws.sets.focus, ws.y_binary, scores["xgboost"], ws.xgboost.decision_threshold
    )
    report = {}
    for name, position in examples.items():
        row = ws.X_raw.iloc[position]
        index = int(np.searchsorted(ws.sets.explanation, position))
        report[name] = {
            "rule": LOCAL_EXAMPLE_RULES[name],
            "chosen_with": "XGBoost score and threshold; pooled sample (focus example: all focus rows)",
            "row": position,
            "country": row["country"],
            "has_account": bool(ws.y_binary[position]),
            "features": {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()},
        }
        for key, method in PRIMARY_METHOD.items():
            model, attribution = ws.models[key], attributions[(method, key)]
            score = float(scores[key][position])
            predicted = "Yes" if score >= model.decision_threshold else "No"
            report[name][key] = {
                "score": score, "threshold": model.decision_threshold, "predicted": predicted,
                "logit": float(attribution.base_value + attribution.values[index].sum()),
                "attributions": dict(zip(attribution.feature_names, map(float, attribution.values[index]))),
            }
            plot_waterfall(
                [f"{feature} = {row[feature]}" for feature in attribution.feature_names],
                attribution.values[index], attribution.base_value,
                figures_dir / f"shap_waterfall__{key}__{name}.png",
                title=f"{MODEL_LABEL[key]}: {name.replace('_', ' ')} (row {position}, {row['country']})",
                caption=f"Chosen by rule: {LOCAL_EXAMPLE_RULES[name]} (rule applied to XGBoost). Actual: "
                        f"{'has an account' if ws.y_binary[position] else 'no account'}; this model's score "
                        f"{score:.3f} vs threshold {model.decision_threshold:.4f} -> predicted {predicted}. "
                        f"{METHOD_LABEL[method]}. {reference_note(ws)} A training row.",
            )
    return report


def plot_dependence_figures(ws: Workspace, attributions: dict, importance_report: dict, figures_dir: Path) -> dict:
    """Dependence plots for the top features by pooled importance (chosen by
    rank, whatever they turn out to be), on the pooled sample."""
    key, method = "xgboost", PRIMARY_METHOD["xgboost"]
    attribution = attributions[(method, key)]
    top = importance_report[f"{method}__{key}"]["pooled"]["ranking"][:N_DEPENDENCE_PLOTS]
    mask = sample_masks(ws)["pooled"]
    rows = ws.rows(ws.sets.explanation[mask])
    report = {"rule": f"top {N_DEPENDENCE_PLOTS} features by pooled mean |{method}| on {key}", "features": {}}
    for feature in top:
        j = attribution.feature_names.index(feature)
        values, raw = attribution.values[mask, j], rows[feature].to_numpy()
        categorical = feature in ws.xgboost.categorical_levels
        plot_dependence(
            feature, raw, values, categorical, figures_dir / f"shap_dependence__{key}__{feature}.png",
            title=f"{MODEL_LABEL[key]}: attribution of {feature} by its value (pooled)",
            caption=f"{METHOD_LABEL[method]}. {reference_note(ws)}",
        )
        if categorical:
            report["features"][feature] = {
                str(level): {"n": int((raw == level).sum()), "mean_attribution": float(values[raw == level].mean()),
                             "std_attribution": float(values[raw == level].std())}
                for level in sorted(set(raw))
            }
        else:
            report["features"][feature] = {
                "correlation_with_value": float(np.corrcoef(raw.astype(float), values)[0, 1]),
                "mean_attribution_by_quartile": {
                    f"q{q + 1}": float(part.mean())
                    for q, part in enumerate(np.array_split(values[np.argsort(raw.astype(float), kind="stable")], 4))
                },
            }
    return report


# --- Orchestration --------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="recompute attributions even if artifacts exist")
    args = parser.parse_args()

    settings = load_settings()
    xai = settings.xai
    attribution_dir = xai.artifacts_dir / "attributions"

    # 1. Models, data, sample sets
    ws = load_workspace(
        settings.paths.raw_train, settings.paths.raw_test, PRODUCTION_DIR, xai.model_dir, xai.artifacts_dir
    )
    print(f"Loaded {ws.xgboost.name} and {ws.mlp.name}; background {len(ws.sets.background)} rows, "
          f"explaining {len(ws.sets.explanation):,}")

    # 2. SHAP attributions
    attributions = compute_attributions(ws, attribution_dir, settings.seed, args.force)

    # 3. Oracle
    oracle_report, oracles = run_oracle(ws, attributions, attribution_dir, settings)

    # 4. Efficiency
    efficiency_report = check_all_efficiency(ws, attributions, oracles)

    # 5. Global importance
    importance_report, importance = global_importance(ws, attributions, settings)

    # 6. Figures
    print("\n=== Figures ===")
    plot_global_figures(ws, attributions, importance, xai.figures_dir)
    local_report = plot_local_figures(ws, attributions, xai.figures_dir)
    dependence_report = plot_dependence_figures(ws, attributions, importance_report, xai.figures_dir)
    print(f"Wrote figures to {xai.figures_dir}")

    summary = {
        "target_space": "logit",
        "value_function": "interventional",
        "reference": reference_note(ws),
        "note": "attributions describe the models, not causes; scores are ranking scores, not calibrated "
                "probabilities; every explained row is a training row of both models",
        "n_rows": {"explanation": int(len(ws.sets.explanation)), "pooled": int(len(ws.sets.pooled)),
                   ws.sets.focus_country.lower(): int(len(ws.sets.focus)), "background": int(len(ws.sets.background))},
        "base_value": {f"{m}__{k}": a.base_value for (m, k), a in attributions.items()},
        "efficiency": efficiency_report,
        "oracle": oracle_report,
        "global_importance": importance_report,
        "local_examples": local_report,
        "dependence": dependence_report,
    }
    summary_path = xai.artifacts_dir / "shap_summary.json"
    summary_path.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
