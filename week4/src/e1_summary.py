"""Compares E1's reproduced numbers with the reported ones, and writes the mismatch note.

Reads week4/results/baseline_reproduction.csv and reports/ablation_raw_results.json (the original
run's saved values). Writes week4/results/baseline_reproduction_summary.md. No numbers are typed in.

  week4/.venv/Scripts/python.exe week4/src/e1_summary.py
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from run_record import REPO_ROOT, RESULTS_DIR

ARM_KEYS = {"xgboost_class_weight": "xgb_cw", "mlp_2layer_class_weight": "pt2_cw"}
TABLE_ROWS = {"xgboost_class_weight": ("XGBoost", "class-weight"), "mlp_2layer_class_weight": ("PyTorch (2-layer)", "class-weight")}
METRICS = ["pr_auc", "roc_auc", "f1", "precision", "recall", "accuracy"]


def reported_table(raw: dict) -> dict[tuple[str, str], dict[str, tuple[float, float]]]:
    """Parses the original run's markdown table of mean ± std (3 decimals) into numbers."""
    table = {}
    for line in raw["full_metrics_table_md"].splitlines()[2:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        pairs = [tuple(float(x) for x in re.findall(r"\d+\.\d+", cell)) for cell in cells[2:]]
        table[(cells[0], cells[1])] = dict(zip(METRICS, pairs))
    return table


def reported_countries(raw: dict) -> dict[str, tuple[float, float]]:
    rows = {}
    for line in raw["country_table_md"].splitlines()[2:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        rows[cells[0]] = tuple(float(x) for x in re.findall(r"\d+\.\d+", cells[1]))
    return rows


def md(df: pd.DataFrame) -> str:
    def cell(v):
        return f"{v:.6f}" if isinstance(v, float) else str(v)

    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def main() -> None:
    data = pd.read_csv(RESULTS_DIR / "baseline_reproduction.csv")
    raw = json.loads((REPO_ROOT / "reports/ablation_raw_results.json").read_text(encoding="utf-8"))
    table, countries = reported_table(raw), reported_countries(raw)
    pooled = data[data["scope"] == "pooled"]

    fold_rows = []
    for (arm, variant), group in pooled[pooled["metric"].isin(["pr_auc", "threshold"])].groupby(["arm", "variant"]):
        for metric in ("pr_auc", "threshold"):
            part = group[group["metric"] == metric]
            fold_rows.append(
                {
                    "arm": arm, "variant": variant, "quantity": f"fold {metric}",
                    "max_abs_difference": float(part["difference"].abs().max()),
                }
            )

    summary_rows = []
    for (arm, variant), group in pooled.groupby(["arm", "variant"]):
        for metric in METRICS:
            values = group[group["metric"] == metric]["reproduced"].to_numpy()
            mean, std = float(values.mean()), float(np.std(values))  # Population std, as reporting.py uses.
            rep_mean, rep_std = table[TABLE_ROWS[arm]][metric]
            summary_rows.append(
                {
                    "arm": arm, "variant": variant, "metric": metric,
                    "reproduced": f"{mean:.3f} ± {std:.3f}", "reported": f"{rep_mean:.3f} ± {rep_std:.3f}",
                    "same_to_3_decimals": round(mean, 3) == rep_mean and round(std, 3) == rep_std,
                }
            )

    country_rows = []
    country_part = data[(data["arm"] == "xgboost_class_weight") & (data["metric"] == "pr_auc") & (data["scope"] != "pooled")]
    for (variant, country), group in country_part.groupby(["variant", "scope"]):
        values = group["reproduced"].to_numpy()
        # Population std: summarize_country_metrics calls np.std on a NumPy array (reporting.py:51).
        mean, std = float(values.mean()), float(np.std(values))
        rep_mean, rep_std = countries[country]
        country_rows.append(
            {
                "variant": variant, "country": country,
                "reproduced": f"{mean:.3f} ± {std:.3f}", "reported": f"{rep_mean:.3f} ± {rep_std:.3f}",
                "same_to_3_decimals": round(mean, 3) == rep_mean and round(std, 3) == rep_std,
            }
        )

    effect_rows = []
    for arm, group in pooled[pooled["metric"] == "pr_auc"].groupby("arm"):
        pivot = group.pivot(index="fold", columns="variant", values="reproduced")
        change = pivot["in_fold"] - pivot["as_reported"]
        effect_rows.append(
            {
                "arm": arm,
                "as_reported_mean": float(pivot["as_reported"].mean()),
                "in_fold_mean": float(pivot["in_fold"].mean()),
                "mean_change": float(change.mean()),
                "largest_fold_change": float(change.abs().max()),
            }
        )

    runs = {
        f"{arm} {variant}": json.loads((RESULTS_DIR / "runs" / f"e1_{arm}_{variant}.json").read_text(encoding="utf-8"))
        for arm in ARM_KEYS for variant in ("as_reported", "in_fold")
    }
    params_same = {
        name: [r == p for r, p in zip(run["metrics"]["best_params_per_fold"], raw["best_params_per_fold"][ARM_KEYS[name.split()[0]]])]
        for name, run in runs.items()
    }
    timing = pd.DataFrame(
        [{"run": name, "seconds": run["metrics"]["seconds"], "tuned_params_equal_reported_per_fold": params_same[name]}
         for name, run in runs.items()]
    )

    text = "\n\n".join(
        [
            "# E1: baseline reproduction\n\nGenerated by `week4/src/e1_summary.py` from `baseline_reproduction.csv` "
            "and the original run's `reports/ablation_raw_results.json`. Do not edit by hand.",
            "## Fold-level agreement with the original run (full precision)\n\n" + md(pd.DataFrame(fold_rows)),
            "## Mean ± std against the reported table\n\n" + md(pd.DataFrame(summary_rows)),
            "## XGBoost per-country PR-AUC against the reported table\n\n" + md(pd.DataFrame(country_rows)),
            "## Effect of fitting preprocessing inside each fold\n\n" + md(pd.DataFrame(effect_rows)),
            "## Run time and tuned hyperparameters\n\n" + md(timing),
        ]
    )
    (RESULTS_DIR / "baseline_reproduction_summary.md").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
