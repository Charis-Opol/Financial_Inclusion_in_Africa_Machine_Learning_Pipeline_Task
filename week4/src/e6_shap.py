"""E6 (RQ6): do SHAP-measured drivers differ by country beyond resampling noise, and does Group DRO
change which features matter for the worst-off country? Rules in HYPOTHESES.md, RQ6.

Models, for seed 0 (D3.6): XGBoost (`xgb`, primary, interventional TreeSHAP), the ERM MLP
(`mlp_erm_wc`, E3's comparator) and the Group DRO MLP (`mlp_dro`), both with exact interventional
Shapley values over the raw features (Week 3's `exact_shapley_values`, which KernelSHAP
approximates). TabPFN is not included: exact Shapley values would need 1,024 x background
TabPFN predictions per row.

Each outer fold's model is refitted from its saved parameters and explains only that fold's
validation rows, so every attribution is out-of-fold. The refit is checked against the saved scores.
Background: 200 rows from the fold's training rows, stratified by country and label, as Week 3 chose
(reports/xai_decisions_log.md, D2.5). The MLPs' coalition batches are scored on --device (the GPU on the
GPU machine): about 7 h on this laptop's CPU otherwise (260 ms per row at 100 rows, measured).

Writes week4/results/e6/ and week4/figures/fig_e6_*.png.

  week4/.venv/bin/python week4/src/e6_shap.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
import torch  # noqa: E402
from scipy.stats import kendalltau, spearmanr  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fin_inclusion.interpretability.grouping import FeatureGrouping  # noqa: E402
from fin_inclusion.interpretability.model_adapters import MLPRawModel  # noqa: E402
from fin_inclusion.interpretability.plots import INK, INK_SECONDARY, SEQUENTIAL, _style  # noqa: E402
from fin_inclusion.interpretability.shapley import exact_shapley_values  # noqa: E402
from fin_inclusion.preprocessing.pipeline_factory import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS  # noqa: E402
from arms import build_arms  # noqa: E402
from harness import TARGET_COLUMN, RunConfig, country_label_key, fold_paths, load_labelled, outer_folds  # noqa: E402
from run_record import RESULTS_DIR, WEEK4_DIR, save_run  # noqa: E402

E6_SEED = 0
MODELS = {"xgb": "tree_shap", "mlp_erm_wc": "exact_shapley", "mlp_dro": "exact_shapley"}
ERM, DRO = "mlp_erm_wc", "mlp_dro"
BACKGROUND_ROWS = 200
REPEATS = 200
STATS_SEED = 2026
COUNTRIES = ("Kenya", "Rwanda", "Tanzania", "Uganda")
RANKED = [f for f in (*CATEGORICAL_COLUMNS, *NUMERIC_COLUMNS) if f != "country"]  # The 9 ranked features.


def background(train: pd.DataFrame, seed: int, n: int) -> pd.DataFrame:
    fraction = min(1.0, n / len(train))
    return train.groupby(country_label_key(train), group_keys=False).sample(frac=fraction, random_state=seed)


DEVICE = "cpu"


class DeviceMLPAdapter(MLPRawModel):
    """MLPRawModel scoring on a chosen device. The encoding and the network are unchanged."""

    device = "cpu"

    def predict_logit_codes(self, codes: np.ndarray) -> np.ndarray:
        cat, num = self.encode_codes(codes)
        net = self.net.to(self.device)
        net.eval()
        with torch.no_grad():
            logit = net(torch.from_numpy(cat).to(self.device), torch.from_numpy(num).to(self.device))
        return logit.cpu().numpy().astype(np.float64)


def mlp_adapter(model, pipeline) -> MLPRawModel:
    """Week 3's MLPRawModel.from_fitted, except that a placeholder answer (such as "Dont know") whose
    `unknown` level never occurred in this fold's training rows is left out instead of raising.
    Rows holding it then count as unseen and are skipped in `explain_fold`."""
    from fin_inclusion.preprocessing.cleaning import (
        DISGUISED_MISSING_VALUES, HOUSEHOLD_SIZE_OUTLIER_THRESHOLD, UNKNOWN_LABEL,
    )

    scale, encode = pipeline.named_steps["scale"], pipeline.named_steps["encode"]
    maps = {c: {str(k): int(v) for k, v in m.items()} for c, m in encode.category_maps_.items()}
    preprocessing = {
        "category_maps": maps,
        "numeric_mean": dict(zip(scale.columns, map(float, scale.scaler_.mean_))),
        "numeric_scale": dict(zip(scale.columns, map(float, scale.scaler_.scale_))),
        "disguised_missing_values": {c: v for c, v in DISGUISED_MISSING_VALUES.items() if UNKNOWN_LABEL in maps.get(c, {})},
        "unknown_label": UNKNOWN_LABEL,
        "unknown_index": int(encode.UNKNOWN_INDEX),
        "household_size_outlier_threshold": HOUSEHOLD_SIZE_OUTLIER_THRESHOLD,
    }
    adapter = DeviceMLPAdapter(model, preprocessing, decision_threshold=0.5)
    adapter.device = DEVICE
    return adapter


def explain_fold(arm, params, train, val, config, n_background) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    """(raw-feature attributions, feature names, refit scores, rows explained) for one outer fold."""
    prepared = arm.prepare(train, val.drop(columns=[TARGET_COLUMN]), True, {})
    model = arm.fit_model(params, prepared, config, E6_SEED, tuning=False)
    scores = model.predict_proba(prepared.X_val)
    reference = background(train, E6_SEED, n_background)
    if MODELS[arm.name] == "tree_shap":
        columns = list(prepared.X_train.columns)
        grouping = FeatureGrouping.for_onehot(columns, list(CATEGORICAL_COLUMNS), list(NUMERIC_COLUMNS))
        encode = lambda rows: prepared.pipeline.transform(rows)[columns].to_numpy(dtype=np.float32)  # noqa: E731
        # An explicit masker keeps the whole background; a bare array is subsampled to 100 (Week 3 note).
        explainer = shap.TreeExplainer(
            model.booster.get_booster(),
            data=shap.maskers.Independent(encode(reference), max_samples=len(reference)),
            feature_perturbation="interventional", model_output="raw",
        )
        values = grouping.aggregate(explainer.shap_values(prepared.X_val[columns].to_numpy(dtype=np.float32)))
        return values, list(grouping.raw_features), scores, np.ones(len(val), dtype=bool)
    adapter = mlp_adapter(model, prepared.pipeline)
    # Rows holding a level the fold never trained on cannot be coded faithfully, so they are skipped and counted.
    known = np.ones(len(val), dtype=bool)
    for column, levels in adapter.categorical_levels.items():
        known &= val[column].isin(levels).to_numpy()
    values, _ = exact_shapley_values(adapter.predict_logit_codes, adapter.to_codes(val[known]), adapter.to_codes(reference))
    return values, list(adapter.raw_features), scores, known


def attributions(model_name: str, frame: pd.DataFrame, config: RunConfig, out_dir: Path, n_background: int) -> pd.DataFrame:
    """Out-of-fold attributions for every row, cached per model."""
    path = out_dir / f"attributions_{model_name}.csv.gz"
    if path.exists():
        return pd.read_csv(path)
    arm = build_arms()[model_name]
    parts, checks = [], []
    for fold_name, train_idx, val_idx in outer_folds(frame, E6_SEED, config.n_outer):
        _, meta_path = fold_paths(config, "nested", model_name, E6_SEED, fold_name)
        scores_path = meta_path.with_suffix("").with_suffix(".csv.gz")
        params = json.loads(meta_path.read_text(encoding="utf-8"))["params"]
        train, val = frame.iloc[train_idx], frame.iloc[val_idx]
        values, names, refit, known = explain_fold(arm, params, train, val, config, n_background)
        saved = pd.read_csv(scores_path).set_index("row_id").reindex(val.index)["score"].to_numpy()
        checks.append({"fold": fold_name, "max_abs_refit_difference": float(np.abs(refit - saved).max()),
                       "rows_skipped_unseen_level": int((~known).sum())})
        part = pd.DataFrame(values, columns=names)
        part.insert(0, "row_country", val["country"].to_numpy()[known])  # "country" is a feature column.
        part.insert(0, "row_id", val.index.to_numpy()[known])
        parts.append(part)
        print(f"e6 {model_name} {fold_name}: refit max diff {checks[-1]['max_abs_refit_difference']:.2e}", flush=True)
    result = pd.concat(parts, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False, compression="gzip")
    (out_dir / f"refit_check_{model_name}.json").write_text(json.dumps(checks, indent=1), encoding="utf-8")
    return result


def importance(values: pd.DataFrame) -> np.ndarray:
    return values[RANKED].abs().mean().to_numpy()


def disagreement(a: np.ndarray, b: np.ndarray) -> dict:
    top_a, top_b = set(np.argsort(-a)[:3]), set(np.argsort(-b)[:3])
    return {"spearman": 1 - spearmanr(a, b).statistic, "kendall": 1 - kendalltau(a, b).statistic,
            "top3": 1 - len(top_a & top_b) / 3}


def summarise(draws: list[dict], label: str) -> list[dict]:
    """Mean and 95% percentile interval over repeats, for each disagreement measure."""
    frame = pd.DataFrame(draws)
    return [{"quantity": label, "measure": m, "mean": float(frame[m].mean()),
             "ci_low": float(np.percentile(frame[m], 2.5)), "ci_high": float(np.percentile(frame[m], 97.5))}
            for m in ("spearman", "kendall", "top3")]


def country_test(values: pd.DataFrame, size: int, rng: np.random.Generator) -> list[dict]:
    """Cross-country disagreement minus the within-country noise baseline, over REPEATS draws."""
    draws = []
    for _ in range(REPEATS):
        halves = {}
        for c in COUNTRIES:
            rows = rng.permutation(np.flatnonzero(values["row_country"].to_numpy() == c))[: 2 * size]
            halves[c] = (importance(values.iloc[rows[:size]]), importance(values.iloc[rows[size:]]))
        cross = [disagreement(halves[a][0], halves[b][0]) for i, a in enumerate(COUNTRIES) for b in COUNTRIES[i + 1:]]
        noise = [disagreement(*halves[c]) for c in COUNTRIES]
        draws.append({m: np.mean([d[m] for d in cross]) - np.mean([d[m] for d in noise]) for m in ("spearman", "kendall", "top3")})
    return summarise(draws, "cross-country minus noise")


def dro_test(erm: pd.DataFrame, dro: pd.DataFrame, country: str, size: int, rng: np.random.Generator) -> list[dict]:
    """In one country: ERM-vs-DRO ranking disagreement minus each model's own resampling noise, same rows."""
    erm = erm[erm["row_country"] == country].set_index("row_id")
    dro = dro[dro["row_country"] == country].set_index("row_id")
    common = erm.index.intersection(dro.index).to_numpy()
    draws = []
    for _ in range(REPEATS):
        rows = rng.permutation(common)[: 2 * size]
        first, second = rows[:size], rows[size:]
        between = disagreement(importance(dro.loc[first]), importance(erm.loc[first]))
        noise = [disagreement(importance(m.loc[first]), importance(m.loc[second])) for m in (erm, dro)]
        draws.append({k: between[k] - np.mean([n[k] for n in noise]) for k in between})
    return summarise(draws, f"Group DRO vs ERM in {country}, minus noise")


def worst_country_of(model_name: str, frame: pd.DataFrame, config: RunConfig) -> str:
    """Lowest per-country PR-AUC of the model, mean over seeds 0-4 (W as in HYPOTHESES.md)."""
    per_country = {c: [] for c in COUNTRIES}
    for seed_dir in sorted((config.output_dir / "nested" / model_name).glob("seed*")):
        scores = pd.concat(pd.read_csv(p) for p in seed_dir.glob("*.csv.gz"))
        for c in COUNTRIES:
            part = scores[scores["country"] == c]
            per_country[c].append(average_precision_score(part["y"], part["score"]))
    return min(COUNTRIES, key=lambda c: np.mean(per_country[c]))


def figure(values_by_model: dict[str, pd.DataFrame], tests: pd.DataFrame, figures_dir: Path) -> None:
    fig, axes = plt.subplots(1, len(values_by_model), figsize=(4.2 * len(values_by_model), 3.8), squeeze=False)
    for ax, (name, values) in zip(axes[0], values_by_model.items()):
        ranks = np.array([pd.Series(importance(values[values["row_country"] == c])).rank(ascending=False).to_numpy()
                          for c in COUNTRIES]).T
        ax.imshow(ranks, cmap=SEQUENTIAL.reversed(), vmin=1, vmax=len(RANKED), aspect="auto")
        for (i, j), rank in np.ndenumerate(ranks):
            ax.text(j, i, f"{rank:.0f}", ha="center", va="center", fontsize=8, color="white" if rank <= 3 else INK)
        ax.set_xticks(range(len(COUNTRIES)), COUNTRIES)
        ax.set_yticks(range(len(RANKED)), RANKED)
        ax.set_title(f"{name} ({MODELS[name].replace('_', ' ')})", loc="left")
        ax.tick_params(length=0)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.text(0.01, 0.01, "Feature rank by mean |SHAP| over each country's out-of-fold rows, seed 0 (1 = most relied on). "
             "country is excluded. Rankings describe the model, not causes of account ownership.",
             fontsize=7.5, color=INK_SECONDARY)
    figures_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(figures_dir / f"fig_e6_rankings.{ext}", dpi=200)
    plt.close(fig)

    part = tests[tests["measure"] == "spearman"].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(part) + 1.4))
    ys = np.arange(len(part))[::-1]
    for y, row in zip(ys, part.itertuples()):
        ax.plot([row.ci_low, row.ci_high], [y, y], color=INK, linewidth=2, solid_capstyle="round")
        ax.plot(row.mean, y, "o", color=INK, markersize=6, markeredgecolor="white")
    ax.set_yticks(ys, [f"{r.model}: {r.quantity}" for r in part.itertuples()])
    ax.axvline(0, color=INK_SECONDARY, linewidth=1)
    ax.set_xlabel("Disagreement (1 − Spearman ρ) beyond resampling noise")
    _style(ax, "x")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.text(0.01, 0.01, f"Mean and 95% interval over {REPEATS} size-matched draws. Above 0: rankings differ by more than "
             "two random halves of one country do. RQ6 is falsified if the interval includes or lies below 0.",
             fontsize=7.5, color=INK_SECONDARY)
    for ext in ("png", "pdf"):
        fig.savefig(figures_dir / f"fig_e6_tests.{ext}", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Smoke outputs, a small background and few repeats")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device for scoring the MLPs' Shapley coalitions")
    args = parser.parse_args()
    global REPEATS, DEVICE
    DEVICE = args.device
    config = RunConfig.smoke() if args.smoke else RunConfig()
    out_dir = RESULTS_DIR / ("smoke/e6" if args.smoke else "e6")
    figures_dir = RESULTS_DIR / "smoke/figures" if args.smoke else WEEK4_DIR / "figures"
    n_background = 5 if args.smoke else BACKGROUND_ROWS
    REPEATS = 10 if args.smoke else REPEATS

    frame = load_labelled(config)
    size = int(frame["country"].value_counts().min()) // 2  # Half the smallest country (HYPOTHESES.md).
    values = {name: attributions(name, frame, config, out_dir, n_background) for name in MODELS}

    rng = np.random.default_rng(STATS_SEED)
    rows = [{"model": name, **r} for name, v in values.items() for r in country_test(v, size, rng)]
    worst = worst_country_of(ERM, frame, config)
    rows += [{"model": f"{DRO} vs {ERM}", **r} for r in dro_test(values[ERM], values[DRO], worst, size, rng)]
    tests = pd.DataFrame(rows)
    tests.to_csv(out_dir / "rq6_tests.csv", index=False)
    pd.DataFrame([{"model": n, "country": c, **dict(zip(RANKED, importance(v[v["row_country"] == c])))}
                  for n, v in values.items() for c in COUNTRIES]).to_csv(out_dir / "mean_abs_shap_by_country.csv", index=False)
    figure(values, tests, figures_dir)
    if not args.smoke:
        save_run("e6_shap", {"seed": E6_SEED, "background_rows": BACKGROUND_ROWS, "device": DEVICE, "repeats": REPEATS, "size": size,
                             "models": MODELS}, {"worst_country_for_erm_mlp": worst, "tests": rows})
    print(tests.to_string(index=False))


if __name__ == "__main__":
    main()
