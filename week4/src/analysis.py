"""Phase 4 metrics, bootstrap intervals, paired comparisons and figures, from the saved scores.

Implements the definitions in HYPOTHESES.md, fixed before any result existed:
  - PR-AUC on each seed's out-of-fold scores (all outer folds together), then averaged over seeds.
  - P pooled, P_c per country, W = min over c of the seed-mean P_c, G = P - W,
    Ex_c = P_c - prevalence, L_c = P_c / prevalence, and W on Excess and on Lift.
  - 2,000 bootstrap replicates resampling rows within country-by-label strata. Every model and
    seed uses the same resampled rows, so differences are paired. 95% percentile intervals.
  - Seed noise: SD over the 5 seeds of the per-seed paired difference.
  - Holm correction within RQ4 (hybrids) and within each held-out country in RQ5.
It reports quantities. Verdicts are written from these tables at the next checkpoint.

Arms with missing folds are skipped and listed, so a partial run still gives partial results.
Writes week4/results/metrics/ and week4/figures/.

  week4/.venv/bin/python week4/src/analysis.py
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fin_inclusion.interpretability.plots import (  # noqa: E402  (also applies the shared figure style)
    GRID, INK, INK_MUTED, INK_SECONDARY, PALETTE, _style,
)
from harness import RunConfig, country_label_key, load_labelled  # noqa: E402
from run_all import EXPERIMENTS, SWEEP_ARMS  # noqa: E402
from run_record import RESULTS_DIR, WEEK4_DIR, save_run  # noqa: E402

REPLICATES = 2_000
BOOTSTRAP_SEED = 2026
SEEDS = (0, 1, 2, 3, 4)
COUNTRIES = ("Kenya", "Rwanda", "Tanzania", "Uganda")
COUNTRY_COLOR = dict(zip(COUNTRIES, PALETTE))  # Fixed identity colours, validated categorical slots 1-4.
DELTA = 0.01  # Smallest effect of interest (HYPOTHESES.md).
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = WEEK4_DIR / "figures"
RQ5_METHODS = ["xgb", "mlp_erm", "mlp_dro", "tabpfn_v2", "h1_balanced", "h2_rw", "h2_dro"]
HYBRIDS = ["h1_balanced", "h2_rw", "h2_dro"]  # Counted in RQ4. h2_erm is a control (D0.6).


# --- PR-AUC under bootstrap weights -------------------------------------------------------------

def weighted_ap(y: np.ndarray, scores: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Average precision for each row of `counts` (replicates x rows), each row's count acting as
    a sample weight. With all counts 1 it equals sklearn's average_precision_score (a test checks)."""
    order = np.argsort(-scores, kind="mergesort")
    y, s = y[order].astype(np.float64), scores[order]
    c = np.atleast_2d(counts)[:, order].astype(np.float64)
    tp, fp = np.cumsum(c * y, axis=1), np.cumsum(c * (1 - y), axis=1)
    ends = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]  # Last row of each tied score.
    tp, fp = tp[:, ends], fp[:, ends]
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall_steps = np.diff(tp, axis=1, prepend=0.0)
    return (recall_steps * precision).sum(axis=1) / tp[:, -1]


def bootstrap_counts(strata: np.ndarray, replicates: int, seed: int) -> np.ndarray:
    """How often each row is drawn in each replicate, resampling within every stratum."""
    rng = np.random.default_rng(seed)
    counts = np.zeros((replicates, len(strata)), dtype=np.uint16)
    for stratum in np.unique(strata):
        rows = np.flatnonzero(strata == stratum)
        draws = rng.integers(0, len(rows), size=(replicates, len(rows)))
        flat = (draws + np.arange(replicates)[:, None] * len(rows)).ravel()
        counts[:, rows] = np.bincount(flat, minlength=replicates * len(rows)).reshape(replicates, len(rows))
    return counts


# --- Loading scores -------------------------------------------------------------------------------

@dataclass
class ArmScores:
    """One arm's out-of-fold scores, one array per seed, aligned to `rows`."""

    name: str
    design: str
    rows: np.ndarray  # Row ids covered: every row (nested) or one country's rows (LOCO, per fold).
    by_seed: dict[int, np.ndarray]
    fold_ap: list[float]


def load_arm(config: RunConfig, design: str, arm: str, row_ids: np.ndarray) -> ArmScores | None:
    """All seeds of one arm, or None if any fold is missing."""
    expected = config.n_outer if design == "nested" else len(COUNTRIES)
    by_seed, fold_ap = {}, []
    for seed in SEEDS:
        folder = config.output_dir / design / arm / f"seed{seed}"
        # A fold counts only once its metadata exists, which run_arm writes last.
        files = [p.with_suffix("").with_suffix(".csv.gz") for p in sorted(folder.glob("*.json"))]
        if len(files) != expected:
            return None
        folds = [pd.read_csv(p) for p in files]
        fold_ap += [average_precision_score(f["y"], f["score"]) for f in folds]
        frame = pd.concat(folds).set_index("row_id").sort_index()
        if design == "nested" and not np.array_equal(frame.index.to_numpy(), row_ids):
            raise ValueError(f"{arm} seed {seed}: out-of-fold rows do not cover every row exactly once")
        by_seed[seed] = frame.reindex(row_ids)["score"].to_numpy()
    return ArmScores(arm, design, row_ids, by_seed, fold_ap)


# --- Metrics ------------------------------------------------------------------------------------------

class Evaluator:
    """Point values and bootstrap replicates of every metric, with the same replicates for all arms."""

    def __init__(self, frame: pd.DataFrame):
        self.y = (frame["bank_account"] == "Yes").to_numpy(dtype=int)
        self.country = frame["country"].to_numpy()
        self.counts = bootstrap_counts(country_label_key(frame).to_numpy(), REPLICATES, BOOTSTRAP_SEED)
        self.masks = {c: self.country == c for c in COUNTRIES}
        self.prevalence = {c: float(self.y[m].mean()) for c, m in self.masks.items()}

    def scope_values(self, scores: np.ndarray, scopes: dict[str, np.ndarray]) -> tuple[dict, dict]:
        """(point AP per scope, replicate APs per scope) for one seed's scores."""
        point, boot = {}, {}
        for name, mask in scopes.items():
            point[name] = float(average_precision_score(self.y[mask], scores[mask]))
            boot[name] = weighted_ap(self.y[mask], scores[mask], self.counts[:, mask])
        return point, boot

    def derived(self, ap: dict) -> dict:
        """All HYPOTHESES.md quantities from pooled and per-country AP (scalars or replicate arrays)."""
        out = {"P": ap["pooled"]}
        for c in COUNTRIES:
            out[f"P_{c}"] = ap[c]
            out[f"Ex_{c}"] = ap[c] - self.prevalence[c]
            out[f"L_{c}"] = ap[c] / self.prevalence[c]
        out["W"] = np.min([ap[c] for c in COUNTRIES], axis=0)
        out["G"] = out["P"] - out["W"]
        out["W_ex"] = np.min([out[f"Ex_{c}"] for c in COUNTRIES], axis=0)
        out["W_lift"] = np.min([out[f"L_{c}"] for c in COUNTRIES], axis=0)
        return out

    def nested(self, arm: ArmScores) -> dict:
        """Per-seed points, seed-mean point and seed-mean replicates of every metric."""
        scopes = {"pooled": np.ones(len(self.y), dtype=bool), **self.masks}
        per_seed, boots = [], []
        for seed in SEEDS:
            point, boot = self.scope_values(arm.by_seed[seed], scopes)
            per_seed.append(self.derived(point))
            boots.append(boot)
        mean_point_ap = {"pooled": np.mean([s["P"] for s in per_seed]),
                         **{c: np.mean([s[f"P_{c}"] for s in per_seed]) for c in COUNTRIES}}
        mean_boot_ap = {k: np.mean([b[k] for b in boots], axis=0) for k in scopes}
        point = self.derived(mean_point_ap)
        point["worst_country"] = COUNTRIES[int(np.argmin([mean_point_ap[c] for c in COUNTRIES]))]
        point["P_foldmean"] = float(np.mean(arm.fold_ap))
        return {"per_seed": per_seed, "point": point, "boot": self.derived(mean_boot_ap)}

    def loco(self, arm: ArmScores, held_out: str) -> dict:
        """PR-AUC in one held-out country (LOCO compares only within a country)."""
        mask = self.masks[held_out]
        per_seed, boots = [], []
        for seed in SEEDS:
            point, boot = self.scope_values(arm.by_seed[seed], {held_out: mask})
            per_seed.append({"P": point[held_out]})
            boots.append(boot[held_out])
        return {
            "per_seed": per_seed,
            "point": {"P": float(np.mean([s["P"] for s in per_seed]))},
            "boot": {"P": np.mean(boots, axis=0)},
        }


# --- Comparisons --------------------------------------------------------------------------------------

def ci(values: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted, running = np.empty(len(p_values)), 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def paired(results: dict, a: str, b: str | list[str], metric: str) -> dict:
    """a minus b (or minus the better of several b, chosen within each replicate and seed)."""
    bs = [b] if isinstance(b, str) else b
    boot = results[a]["boot"][metric] - np.max([results[x]["boot"][metric] for x in bs], axis=0)
    point = results[a]["point"][metric] - max(results[x]["point"][metric] for x in bs)
    seeds = [results[a]["per_seed"][i][metric] - max(results[x]["per_seed"][i][metric] for x in bs) for i in range(len(SEEDS))]
    low, high = ci(boot)
    return {
        "a": a, "b": " or ".join(bs), "metric": metric, "difference": float(point), "ci_low": low, "ci_high": high,
        "seed_sd": float(np.std(seeds, ddof=1)),
        "p_two_sided": float(min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()))),
        "p_greater": float((boot <= 0).mean()),
    }


MAIN_METRICS = ["P", "W", "G", "W_ex", "W_lift", "P_Uganda", "P_Rwanda"]

# (family, a, b, metrics). Every entry is pre-registered in HYPOTHESES.md or D3.3.
NESTED_COMPARISONS = [
    ("RQ1 TabPFN vs XGBoost", "tabpfn_v2", "xgb", MAIN_METRICS),
    ("RQ1 TabPFN vs MLP", "tabpfn_v2", "mlp_erm", MAIN_METRICS),
    ("RQ1 sensitivity: TabPFN vs XGBoost (tuned scale_pos_weight)", "tabpfn_v2", "xgb_spw", MAIN_METRICS),
    ("RQ3 gain: Group DRO vs ERM", "mlp_dro", "mlp_erm_wc", MAIN_METRICS),
    ("RQ3 Group DRO vs reweighting", "mlp_dro", "mlp_rw", MAIN_METRICS),
    ("RQ3 sensitivity: Group DRO vs ERM selected on pooled", "mlp_dro", "mlp_erm", MAIN_METRICS),
    ("RQ3 D3.3: Group DRO vs ERM, within-country class weights", "mlp_dro_within_cw", "mlp_erm_within_cw", MAIN_METRICS),
    ("RQ3 comparator: reweighted vs ERM XGBoost", "xgb_rw", "xgb", MAIN_METRICS),
    ("E7: country-by-label groups vs country groups", "mlp_dro_country_label", "mlp_dro", MAIN_METRICS),
]


def run_comparisons(results: dict, missing: list[str]) -> pd.DataFrame:
    rows = []
    for family, a, b, metrics in NESTED_COMPARISONS:
        if a in results and b in results:
            rows += [{"family": family, **paired(results, a, b, m)} for m in metrics]
        else:
            missing.append(f"comparison '{family}' (needs {a} and {b})")
    # RQ4: each hybrid against the better of TabPFN default and the Group DRO MLP, within each replicate.
    rq4 = [paired(results, h, ["tabpfn_v2", "mlp_dro"], "W") for h in HYBRIDS
           if all(x in results for x in (h, "tabpfn_v2", "mlp_dro"))]
    for row, adjusted in zip(rq4, holm([r["p_greater"] for r in rq4])):
        rows.append({"family": "RQ4 hybrid vs best single approach (Holm over hybrids)", **row, "p_holm": adjusted})
    if all(x in results for x in ("h2_erm", "tabpfn_v2", "mlp_dro")):
        rows.append({"family": "RQ4 control (not a hybrid): H2 ERM head", **paired(results, "h2_erm", ["tabpfn_v2", "mlp_dro"], "W")})
    return pd.DataFrame(rows)


def loco_tables(loco: dict, nested: dict, evaluator: Evaluator) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """RQ5 pairwise comparisons per held-out country with Holm, the consistency rule, and the E2-to-E4 drop."""
    pair_rows = []
    for c in COUNTRIES:
        present = [m for m in RQ5_METHODS if m in loco.get(c, {})]
        family = [paired(loco[c], a, b, "P") for i, a in enumerate(present) for b in present[i + 1:]]
        for row, adjusted in zip(family, holm([r["p_two_sided"] for r in family])):
            pair_rows.append({"held_out": c, **row, "p_holm": adjusted})
    pairs = pd.DataFrame(pair_rows)

    consistency = []
    for method in RQ5_METHODS:
        wins, beaten, countries = 0, False, 0
        for c in COUNTRIES:
            if method not in loco.get(c, {}):
                continue
            countries += 1
            here = pairs[(pairs["held_out"] == c) & ((pairs["a"] == method) | (pairs["b"] == method))]
            # Signed so that positive means `method` is better.
            signed = np.where(here["a"] == method, here["difference"], -here["difference"])
            significant = here["p_holm"].to_numpy() < 0.05
            best = max(loco[c], key=lambda m: loco[c][m]["point"]["P"] if m in RQ5_METHODS else -1) == method
            if best and len(here) and (significant & (signed > 0)).all():
                wins += 1
            if (significant & (signed < 0)).any():
                beaten = True
        consistency.append({"method": method, "held_out_countries_run": countries, "clear_wins": wins,
                            "ever_clearly_beaten": beaten, "consistently_better": wins >= 3 and not beaten})

    drop_rows = []
    for c in COUNTRIES:
        mask = evaluator.masks[c]
        for arm, result in loco.get(c, {}).items():
            if arm not in nested:
                continue
            within = nested[arm]
            boot = within["boot"][f"P_{c}"] - result["boot"]["P"]
            low, high = ci(boot)
            drop_rows.append({"country": c, "arm": arm, "within_country_cv": within["point"][f"P_{c}"],
                              "held_out": result["point"]["P"],
                              "drop": within["point"][f"P_{c}"] - result["point"]["P"], "ci_low": low, "ci_high": high,
                              "rows": int(mask.sum())})
    return pairs, pd.DataFrame(consistency), pd.DataFrame(drop_rows)


# --- Group DRO weights over training --------------------------------------------------------------

def q_history(config: RunConfig, design: str, arm: str) -> pd.DataFrame:
    """Mean over seeds and folds of q per epoch and group, from each fold's saved history."""
    rows = []
    for meta in sorted((config.output_dir / design / arm).glob("seed*/*.json")):
        extra = json.loads(meta.read_text(encoding="utf-8")).get("extra", {})
        for epoch, (q, loss) in enumerate(zip(extra.get("q_by_epoch", []), extra.get("group_loss_by_epoch", []))):
            rows += [{"epoch": epoch + 1, "group": g, "q": qg, "loss": lg} for g, (qg, lg) in enumerate(zip(q, loss))]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).groupby(["epoch", "group"], as_index=False)[["q", "loss"]].mean()


# --- Figures --------------------------------------------------------------------------------------------

def _save(fig, name: str, caption: str) -> None:
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.text(0.01, 0.01, caption, ha="left", va="bottom", fontsize=7.5, color=INK_SECONDARY, wrap=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=200)
    fig.savefig(FIGURES_DIR / f"{name}.pdf")
    plt.close(fig)


def interval_rows(ax, labels, points, lows, highs, colors=None, reference=None):
    """Horizontal point-and-interval rows, top to bottom in the order given."""
    ys = np.arange(len(labels))[::-1]
    colors = colors or [INK] * len(labels)
    for y, p, lo, hi, color in zip(ys, points, lows, highs, colors):
        ax.plot([lo, hi], [y, y], color=color, linewidth=2, solid_capstyle="round")
        ax.plot(p, y, "o", color=color, markersize=6, markeredgecolor="white", markeredgewidth=1)
    ax.set_yticks(ys, labels)
    if reference is not None:
        ax.axvline(reference, color=INK_MUTED, linewidth=1)
    _style(ax, "x")


def fig_marginal(results: dict, arms: list[str], metric: str, name: str, title: str, xlabel: str, caption: str):
    present = [a for a in arms if a in results]
    if not present:
        return
    fig, ax = plt.subplots(figsize=(6.5, 0.35 * len(present) + 1.4))
    interval_rows(ax, present, [results[a]["point"][metric] for a in present],
                  *zip(*[ci(results[a]["boot"][metric]) for a in present]))
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    _save(fig, name, caption)


def fig_per_country(results: dict, arms: list[str]):
    present = [a for a in arms if a in results]
    if not present:
        return
    fig, axes = plt.subplots(1, 4, figsize=(12, 0.32 * len(present) + 1.6), sharey=True)
    for ax, c in zip(axes, COUNTRIES):
        lows, highs = zip(*[ci(results[a]["boot"][f"P_{c}"]) for a in present])
        interval_rows(ax, present, [results[a]["point"][f"P_{c}"] for a in present], lows, highs,
                      colors=[COUNTRY_COLOR[c]] * len(present))
        ax.set_title(c, loc="left")
        ax.set_xlabel("PR-AUC")
    _save(fig, "fig_per_country_prauc",
          "PR-AUC per country, mean over 5 seeds, with 95% bootstrap intervals (2,000 replicates within country-by-label "
          "strata). Compare within a panel: prevalence differs between countries, so raw PR-AUC is not comparable across panels.")


def fig_forest(table: pd.DataFrame, name: str, title: str, caption: str, xlabel: str = "Difference in PR-AUC"):
    if table.empty:
        return
    labels = [f"{r.family.split(':')[0]}: {r.a} − {r.b} [{r.metric}]" if "metric" in table else r.a for r in table.itertuples()]
    fig, ax = plt.subplots(figsize=(8, 0.34 * len(table) + 1.5))
    interval_rows(ax, labels, table["difference"], table["ci_low"], table["ci_high"], reference=0)
    for edge in (-DELTA, DELTA):
        ax.axvline(edge, color=GRID, linewidth=1, linestyle="--")
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    _save(fig, name, caption)


def fig_sweeps(results: dict):
    groups = {
        "Step size": [(float(a.rsplit("_", 1)[1]), a) for a in SWEEP_ARMS if "_step_" in a],
        "Adjustment C": [(float(a.rsplit("_", 1)[1]), a) for a in SWEEP_ARMS if "_c_" in a],
    }
    if not any(a in results for _, items in groups.items() for _, a in items):
        return
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    for col, (label, items) in enumerate(groups.items()):
        items = [(v, a) for v, a in items if a in results]
        for row, metric in enumerate(["W", "P"]):
            ax = axes[row, col]
            xs = np.arange(len(items))
            points = [results[a]["point"][metric] for _, a in items]
            lows, highs = zip(*[ci(results[a]["boot"][metric]) for _, a in items]) if items else ([], [])
            # Reference line drawn first (lowest zorder) so it sits behind the markers instead of
            # cutting across them, which previously looked like a dot punched into the line.
            if "mlp_dro" in results:
                ref = results["mlp_dro"]["point"][metric]
                ax.axhline(ref, color=PALETTE[1], linewidth=1.3, linestyle="--", alpha=0.7, zorder=1)
                ax.annotate("tuned mlp_dro", (xs[0] if len(xs) else 0, ref),
                            textcoords="offset points", xytext=(0, 6), ha="left", fontsize=7.5, color=INK_SECONDARY,
                            zorder=4, bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85))
            ax.vlines(xs, lows, highs, color=INK, linewidth=2, zorder=2)
            ax.plot(xs, points, "o", color=INK, markersize=6, markeredgecolor="white", zorder=3)
            ax.set_xticks(xs, [f"{v:g}" for v, _ in items])
            ax.set_xlabel(label)
            ax.set_ylabel("Worst-country PR-AUC (W)" if metric == "W" else "Pooled PR-AUC (P)")
            _style(ax, "y")
    _save(fig, "fig_rq3_sweeps",
          "Group DRO ablations: each point reuses the tuned mlp_dro parameters for its fold and changes one setting. Nothing "
          "is retuned. Mean over 5 seeds with 95% bootstrap intervals. Step size 0 is plain training on group-balanced batches.")


def fig_q(config: RunConfig):
    arms = [("nested", a) for a in ("mlp_dro", "mlp_dro_within_cw", "h2_dro")]
    histories = [(a, q_history(config, d, a)) for d, a in arms]
    histories = [(a, h) for a, h in histories if not h.empty]
    if not histories:
        return
    fig, axes = plt.subplots(1, len(histories), figsize=(4 * len(histories), 3.4), sharey=True, squeeze=False)
    for ax, (arm, history) in zip(axes[0], histories):
        for g, c in enumerate(COUNTRIES):
            part = history[history["group"] == g]
            # Markers as well as colour, since countries with similar q overlap.
            ax.plot(part["epoch"], part["q"], color=COUNTRY_COLOR[c], linewidth=2, marker="os^D"[g], markersize=4, label=c)
        ax.axhline(0.25, color=INK_MUTED, linewidth=1, linestyle="--")
        ax.set_title(arm, loc="left")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Group weight q")
        _style(ax, "y")
    axes[0][0].legend(frameon=False, fontsize=8)
    _save(fig, "fig_dro_group_weights",
          "Group DRO weights q at the end of each epoch, mean over seeds and outer folds of the final fits. "
          "Uniform weighting is 0.25. Shows which countries Group DRO actually focused on.")


def fig_loco(loco: dict):
    present = [c for c in COUNTRIES if loco.get(c)]
    if not present:
        return
    methods = [m for m in RQ5_METHODS + ["h2_erm"] if any(m in loco[c] for c in present)]
    fig, axes = plt.subplots(1, len(present), figsize=(3.2 * len(present), 0.35 * len(methods) + 1.6), sharey=True, squeeze=False)
    for ax, c in zip(axes[0], present):
        shown = [m for m in methods if m in loco[c]]
        rows = {m: loco[c][m] for m in shown}
        lows, highs = zip(*[ci(r["boot"]["P"]) for r in rows.values()])
        interval_rows(ax, shown, [r["point"]["P"] for r in rows.values()], lows, highs,
                      colors=[INK_MUTED if m == "h2_erm" else COUNTRY_COLOR[c] for m in shown])
        ax.set_title(f"Held out: {c}", loc="left")
        ax.set_xlabel("PR-AUC")
    _save(fig, "fig_rq5_loco",
          "Leave-one-country-out: trained on three countries, scored on the fourth, with country dropped as a feature. "
          "Mean over 5 seeds, 95% bootstrap intervals. Compare only within a panel. h2_erm (grey) is a control, not a hybrid.")


def fig_pr_curves(arms: dict[str, ArmScores], evaluator: Evaluator):
    shown = [a for a in ("xgb", "mlp_erm", "tabpfn_v2") if a in arms]  # At most 3 series: validated all-pairs slots.
    if not shown:
        return
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4), sharey=True)
    for ax, c in zip(axes, COUNTRIES):
        mask = evaluator.masks[c]
        for color, arm in zip(PALETTE, shown):
            precision, recall, _ = precision_recall_curve(evaluator.y[mask], arms[arm].by_seed[0][mask])
            ax.plot(recall, precision, color=color, linewidth=2, label=arm)
        ax.axhline(evaluator.prevalence[c], color=INK_MUTED, linewidth=1, linestyle="--")
        ax.set_title(c, loc="left")
        ax.set_xlabel("Recall")
        _style(ax, "both")
    axes[0].set_ylabel("Precision")
    axes[0].legend(frameon=False, fontsize=8)
    _save(fig, "fig_pr_curves_seed0",
          "Precision-recall curves per country for seed 0's out-of-fold scores. Dashed line: the country's prevalence, "
          "which is the PR-AUC of a random ranking.")


# --- Main ------------------------------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Read smoke outputs (seed 0 only) and write under results/smoke/")
    args = parser.parse_args()
    global SEEDS, REPLICATES, METRICS_DIR, FIGURES_DIR
    config = RunConfig.smoke() if args.smoke else RunConfig()
    if args.smoke:
        SEEDS, REPLICATES = (0,), 200
        METRICS_DIR, FIGURES_DIR = RESULTS_DIR / "smoke/metrics", RESULTS_DIR / "smoke/figures"
    frame = load_labelled(config)
    evaluator = Evaluator(frame)
    row_ids = frame.index.to_numpy()
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []

    nested_arms = list(dict.fromkeys(a for d, arms in EXPERIMENTS.values() for a in arms if d == "nested"))
    loco_arms = list(dict.fromkeys(a for d, arms in EXPERIMENTS.values() for a in arms if d == "loco"))

    scores, nested = {}, {}
    for arm in nested_arms:
        loaded = load_arm(config, "nested", arm, row_ids)
        if loaded is None:
            missing.append(f"nested/{arm}")
            continue
        scores[arm] = loaded
        nested[arm] = evaluator.nested(loaded)
        print(f"metrics nested {arm}", flush=True)

    loco: dict[str, dict] = {}
    for arm in loco_arms:
        loaded = load_arm(config, "loco", arm, row_ids)
        if loaded is None:
            missing.append(f"loco/{arm}")
            continue
        for c in COUNTRIES:
            loco.setdefault(c, {})[arm] = evaluator.loco(loaded, c)
        print(f"metrics loco {arm}", flush=True)

    # Tables.
    marginal = []
    for arm, result in nested.items():
        for metric, value in result["point"].items():
            if metric == "worst_country":
                continue
            low, high = ci(result["boot"][metric]) if metric in result["boot"] else (np.nan, np.nan)
            seed_values = [s[metric] for s in result["per_seed"]] if metric in result["per_seed"][0] else [np.nan]
            marginal.append({"arm": arm, "metric": metric, "value": float(value), "ci_low": low, "ci_high": high,
                             "seed_sd": float(np.std(seed_values, ddof=1)) if len(seed_values) > 1 else np.nan})
    marginal = pd.DataFrame(marginal)
    marginal.to_csv(METRICS_DIR / "nested_marginal.csv", index=False)
    worst = pd.DataFrame([{"arm": a, "worst_country": r["point"]["worst_country"],
                           **{f"worst_country_seed{s}": COUNTRIES[int(np.argmin([r["per_seed"][i][f'P_{c}'] for c in COUNTRIES]))]
                              for i, s in enumerate(SEEDS)}} for a, r in nested.items()])
    worst.to_csv(METRICS_DIR / "worst_country.csv", index=False)
    comparisons = run_comparisons(nested, missing)
    comparisons.to_csv(METRICS_DIR / "nested_paired.csv", index=False)

    loco_marginal = pd.DataFrame([
        {"held_out": c, "arm": a, "P": r["point"]["P"], **dict(zip(("ci_low", "ci_high"), ci(r["boot"]["P"]))),
         "seed_sd": float(np.std([s["P"] for s in r["per_seed"]], ddof=1))}
        for c, arms in loco.items() for a, r in arms.items()
    ])
    loco_marginal.to_csv(METRICS_DIR / "loco_marginal.csv", index=False)
    pairs, consistency, drop = loco_tables(loco, nested, evaluator)
    pairs.to_csv(METRICS_DIR / "loco_paired.csv", index=False)
    consistency.to_csv(METRICS_DIR / "loco_consistency.csv", index=False)
    drop.to_csv(METRICS_DIR / "e2_to_e4_drop.csv", index=False)
    for arm in ("mlp_dro", "mlp_dro_within_cw", "h2_dro", "mlp_dro_country_label"):
        history = q_history(config, "nested", arm)
        if not history.empty:
            history.to_csv(METRICS_DIR / f"q_history_{arm}.csv", index=False)

    # Figures.
    e2 = ["tabpfn_v2", "xgb", "xgb_spw", "mlp_erm"]
    e3 = ["mlp_erm_wc", "mlp_rw", "mlp_dro", "mlp_erm_within_cw", "mlp_dro_within_cw", "xgb_rw"]
    e5 = ["h1_balanced", "h2_erm", "h2_rw", "h2_dro"]
    main_arms = e2 + e3 + e5 + ["mlp_dro_country_label"]
    bootstrap_note = "Mean over 5 seeds; 95% percentile intervals from 2,000 bootstrap replicates within country-by-label strata."
    fig_marginal(nested, e2, "P", "fig_rq1_pooled_prauc", "RQ1: pooled PR-AUC (E2)", "Pooled PR-AUC", bootstrap_note)
    fig_marginal(nested, main_arms, "W", "fig_worst_country_prauc", "Worst-country PR-AUC, W", "W = lowest per-country PR-AUC",
                 bootstrap_note + " The worst country is chosen per model and per replicate, so it can differ between models.")
    fig_marginal(nested, main_arms, "G", "fig_rq2_gap", "Gap between pooled and worst-country PR-AUC, G = P − W", "G",
                 bootstrap_note + " A smaller gap reached by lowering P is not an improvement for the worst-off country.")
    fig_per_country(nested, main_arms)
    for family, name, title in [
        ("RQ1", "fig_rq1_paired", "RQ1 and RQ2: TabPFN against XGBoost and the MLP"),
        ("RQ3", "fig_rq3_paired", "RQ3: Group DRO against ERM and reweighting"),
        ("RQ4", "fig_rq4_hybrids", "RQ4: hybrids against the better single approach"),
    ]:
        part = comparisons[comparisons["family"].str.startswith(family)] if not comparisons.empty else comparisons
        if family != "RQ4" and not part.empty:
            part = part[part["metric"].isin(["P", "W", "G"])]
        fig_forest(part, name, title, "Paired differences, a − b. " + bootstrap_note +
                   " Dashed lines: ±0.01, the smallest effect of interest. RQ4 compares each hybrid with the better of "
                   "TabPFN and the Group DRO MLP within each replicate.")
    fig_sweeps(nested)
    fig_q(config)
    fig_loco(loco)
    if not drop.empty:
        drop_table = drop.assign(a=drop["country"] + ": " + drop["arm"], difference=drop["drop"])
        fig_forest(drop_table[["a", "difference", "ci_low", "ci_high"]], "fig_e2_to_e4_drop",
                   "PR-AUC lost when a country is held out (E2 minus E4, same rows)",
                   "Within-country CV PR-AUC minus leave-one-country-out PR-AUC on the same country's rows. Not pure shift: "
                   "training size differs too. " + bootstrap_note, xlabel="Drop in PR-AUC")
    fig_pr_curves(scores, evaluator)

    summary_rows = []
    for arm in main_arms + SWEEP_ARMS:
        if arm in nested:
            p = nested[arm]["point"]
            summary_rows.append(f"| {arm} | {p['P']:.3f} | {p['W']:.3f} | {p['worst_country']} | {p['G']:.3f} | "
                                f"{p['P_Uganda']:.3f} | {p['P_Rwanda']:.3f} | {p['W_ex']:.3f} | {p['W_lift']:.2f} | {p['P_foldmean']:.3f} |")
    text = "\n".join([
        "# Phase 4 metrics summary", "",
        f"Generated by `week4/src/analysis.py`. Point values are means over seeds {list(SEEDS)}. Intervals are in "
        "`nested_marginal.csv`; paired comparisons in `nested_paired.csv`; LOCO in `loco_*.csv`.", "",
        "| Arm | P | W | Worst country | G | P Uganda | P Rwanda | W on Excess | W on Lift | Fold-mean P |",
        "|---|---|---|---|---|---|---|---|---|---|", *summary_rows, "",
        "**Not available:** " + ("; ".join(missing) if missing else "nothing; every arm finished."),
    ])
    (METRICS_DIR / "summary.md").write_text(text + "\n", encoding="utf-8")
    if args.smoke:
        print(text)
        return
    save_run("phase4_analysis", {"replicates": REPLICATES, "bootstrap_seed": BOOTSTRAP_SEED, "seeds": list(SEEDS)},
             {"arms_nested": sorted(nested), "arms_loco": sorted({a for c in loco.values() for a in c}), "missing": missing})
    print(text)


if __name__ == "__main__":
    main()
