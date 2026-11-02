"""Phase 1 data audit: counts, missingness and class balance per country, and fold sizes.

Describes the labelled data only. Nothing here trains a model. Writes:
  week4/results/data_audit.md          the readable audit
  week4/results/per_country_counts.csv rows, positives, prevalence per country
  week4/results/fold_sizes.csv         training-set sizes the models will actually see
  week4/results/runs/phase1_data_audit.json

Run from the repo root:
  week4/.venv/Scripts/python.exe week4/src/data_audit.py
"""
from __future__ import annotations

from itertools import combinations

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from fin_inclusion.data.loader import TARGET_COLUMN, DataLoader
from fin_inclusion.preprocessing.cleaning import (
    DISGUISED_MISSING_VALUES,
    HOUSEHOLD_SIZE_OUTLIER_THRESHOLD,
)
from fin_inclusion.preprocessing.pipeline_factory import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    make_country_target_stratify_key,
)
from run_record import REPO_ROOT, RESULTS_DIR, save_run

SEEDS = [0, 1, 2, 3, 4]
N_OUTER = 5
N_INNER = 3
TABPFN_V2_ROW_LIMIT = 10_000  # As published (hollmann2025accurate); feasibility.md checks the installed code.


def per_country_counts(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """One row per country plus a total, with counts as integers and prevalence as a fraction."""
    positive = train[TARGET_COLUMN].eq("Yes")
    table = train.assign(positive=positive).groupby("country").agg(
        survey_year=("year", lambda s: ",".join(str(y) for y in sorted(s.unique()))),
        rows=("positive", "size"),
        positives=("positive", "sum"),
    )
    table["negatives"] = table["rows"] - table["positives"]
    table["prevalence"] = table["positives"] / table["rows"]
    table["share_of_rows"] = table["rows"] / len(train)
    table["share_of_positives"] = table["positives"] / positive.sum()
    table["unlabelled_test_rows"] = test.groupby("country").size()
    total = pd.DataFrame(
        {
            "survey_year": "all",
            "rows": len(train),
            "positives": int(positive.sum()),
            "negatives": int((~positive).sum()),
            "prevalence": positive.mean(),
            "share_of_rows": 1.0,
            "share_of_positives": 1.0,
            "unlabelled_test_rows": len(test),
        },
        index=["ALL"],
    )
    return pd.concat([table, total]).rename_axis("country").reset_index()


def missingness(train: pd.DataFrame) -> pd.DataFrame:
    """Truly empty cells per column, overall and per country."""
    columns = [c for c in train.columns if c != "uniqueid"]
    table = train.groupby("country")[columns].apply(lambda g: g.isna().sum()).T
    table["ALL"] = train[columns].isna().sum()
    return table.rename_axis("column").reset_index()


def disguised_missingness(train: pd.DataFrame) -> pd.DataFrame:
    """Placeholder answers such as "Dont know", per country, with their positive rate."""
    rows = []
    for column, placeholder in DISGUISED_MISSING_VALUES.items():
        for country, group in [*train.groupby("country"), ("ALL", train)]:
            hit = group[group[column] == placeholder]
            rows.append(
                {
                    "column": column,
                    "placeholder": placeholder,
                    "country": country,
                    "n": len(hit),
                    "share_of_country_rows": len(hit) / len(group),
                    "positives": int(hit[TARGET_COLUMN].eq("Yes").sum()),
                }
            )
    return pd.DataFrame(rows)


def category_support(train: pd.DataFrame) -> pd.DataFrame:
    """Rows per category level per country. A zero or tiny cell matters for leave-one-country-out."""
    rows = []
    for column in [c for c in CATEGORICAL_COLUMNS if c != "country"]:
        counts = pd.crosstab(train[column], train["country"])
        for level, row in counts.iterrows():
            rows.append({"column": column, "level": level, **row.to_dict(), "min_cell": int(row.min())})
    return pd.DataFrame(rows)


def covariate_distance(train: pd.DataFrame) -> pd.DataFrame:
    """Total variation distance between country pairs per categorical feature (0 same, 1 disjoint)."""
    rows = []
    countries = sorted(train["country"].unique())
    for column in [c for c in CATEGORICAL_COLUMNS if c != "country"]:
        shares = pd.crosstab(train[column], train["country"], normalize="columns")
        row = {"column": column}
        for a, b in combinations(countries, 2):
            row[f"{a[:3]}-{b[:3]}"] = 0.5 * (shares[a] - shares[b]).abs().sum()
        rows.append(row)
    return pd.DataFrame(rows)


def rate_by_feature(train: pd.DataFrame, column: str) -> pd.DataFrame:
    """Positive rate per level per country, to see whether a feature behaves alike across countries."""
    table = train.assign(positive=train[TARGET_COLUMN].eq("Yes")).pivot_table(
        index=column, columns="country", values="positive", aggfunc="mean"
    )
    return table.rename_axis(column).reset_index()


def numeric_summary(train: pd.DataFrame) -> pd.DataFrame:
    """Median and range of each numeric column per country, plus the household-size outlier count."""
    rows = []
    for country, group in [*train.groupby("country"), ("ALL", train)]:
        row = {"country": country}
        for column in NUMERIC_COLUMNS:
            row[f"{column}_median"] = group[column].median()
            row[f"{column}_min"] = group[column].min()
            row[f"{column}_max"] = group[column].max()
        row[f"household_size_over_{HOUSEHOLD_SIZE_OUTLIER_THRESHOLD}"] = int(
            (group["household_size"] > HOUSEHOLD_SIZE_OUTLIER_THRESHOLD).sum()
        )
        rows.append(row)
    return pd.DataFrame(rows)


def fold_sizes(train: pd.DataFrame) -> pd.DataFrame:
    """Actual sizes for the outer folds, inner folds and leave-one-country-out, over seeds 0 to 4."""
    key = make_country_target_stratify_key(train, target_col=TARGET_COLUMN)
    positive = train[TARGET_COLUMN].eq("Yes")
    rows = []
    for seed in SEEDS:
        outer = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=seed)
        for fold, (fit_idx, test_idx) in enumerate(outer.split(train, key)):
            test_rows = train.iloc[test_idx]
            test_positive = positive.iloc[test_idx]
            record = {
                "design": "nested_cv_outer",
                "seed": seed,
                "fold": fold,
                "train_rows": len(fit_idx),
                "test_rows": len(test_idx),
                "test_positives": int(test_positive.sum()),
            }
            for country, group in test_rows.groupby("country"):
                record[f"test_positives_{country}"] = int(test_positive.loc[group.index].sum())
            rows.append(record)
            inner = StratifiedKFold(n_splits=N_INNER, shuffle=True, random_state=seed)
            for inner_fold, (inner_fit, inner_val) in enumerate(inner.split(fit_idx, key.iloc[fit_idx])):
                rows.append(
                    {
                        "design": "nested_cv_inner",
                        "seed": seed,
                        "fold": f"{fold}.{inner_fold}",
                        "train_rows": len(inner_fit),
                        "test_rows": len(inner_val),
                        "test_positives": int(positive.iloc[fit_idx[inner_val]].sum()),
                    }
                )
    for country in sorted(train["country"].unique()):
        held_out = train["country"].eq(country)
        rows.append(
            {
                "design": "leave_one_country_out",
                "seed": "n/a",
                "fold": country,
                "train_rows": int((~held_out).sum()),
                "test_rows": int(held_out.sum()),
                "test_positives": int(positive[held_out].sum()),
            }
        )
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame, floats: str = "{:.3f}") -> str:
    """Small Markdown table writer, to avoid adding `tabulate` as a dependency."""
    def cell(value: object) -> str:
        if isinstance(value, float):
            return floats.format(value)
        return str(value)

    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    rule = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, rule, *body])


def main() -> None:
    loader = DataLoader(
        raw_train_path=REPO_ROOT / "data/raw/Train_v2.csv",
        raw_test_path=REPO_ROOT / "data/raw/Test_v2.csv",
    )
    train, test = loader.load_train(), loader.load_test()

    counts = per_country_counts(train, test)
    folds = fold_sizes(train)
    counts.to_csv(RESULTS_DIR / "per_country_counts.csv", index=False)
    folds.to_csv(RESULTS_DIR / "fold_sizes.csv", index=False)

    size_summary = (
        folds.groupby("design")
        .agg(
            folds=("train_rows", "size"),
            min_train_rows=("train_rows", "min"),
            max_train_rows=("train_rows", "max"),
            min_test_positives=("test_positives", "min"),
        )
        .reset_index()
    )
    size_summary["max_over_published_tabpfn_limit"] = size_summary["max_train_rows"] / TABPFN_V2_ROW_LIMIT
    outer = folds[folds["design"] == "nested_cv_outer"]
    per_country_columns = [c for c in folds.columns if c.startswith("test_positives_")]
    fold_positives = pd.DataFrame(
        {
            "country": [c.removeprefix("test_positives_") for c in per_country_columns],
            "min_test_positives_per_outer_fold": [int(outer[c].min()) for c in per_country_columns],
            "max_test_positives_per_outer_fold": [int(outer[c].max()) for c in per_country_columns],
        }
    )
    support = category_support(train)
    feature_columns = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS
    excluded = [c for c in train.columns if c not in feature_columns + [TARGET_COLUMN]]

    sections = [
        "# Data audit\n\nGenerated by `week4/src/data_audit.py`. Do not edit by hand. "
        "Covers `data/raw/Train_v2.csv`, the labelled data. `Test_v2.csv` has no labels and is counted only.",
        f"## Shape\n\n- Labelled rows: {len(train)}\n- Unlabelled test rows: {len(test)}\n"
        f"- Model features: {len(feature_columns)} ({len(CATEGORICAL_COLUMNS)} categorical, "
        f"{len(NUMERIC_COLUMNS)} numeric), or {len(feature_columns) - 1} without `country`\n"
        f"- Not used as features: {', '.join(f'`{c}`' for c in excluded)}; target `{TARGET_COLUMN}`\n"
        "- The baseline encoders add a derived `household_size_outlier` flag; "
        "encoded column counts are in `feasibility.md`.",
        "## Rows, class balance and prevalence per country\n\n" + to_markdown(counts),
        "Each country has exactly one survey year, so country and survey round cannot be separated.",
        "## Truly missing cells per column\n\n" + to_markdown(missingness(train)),
        "## Disguised missingness (placeholder answers)\n\n" + to_markdown(disguised_missingness(train)),
        "## Numeric columns\n\n" + to_markdown(numeric_summary(train), floats="{:.1f}"),
        "## Sizes the models will see\n\n" + to_markdown(size_summary),
        "`max_over_published_tabpfn_limit` is the largest training set divided by "
        f"{TABPFN_V2_ROW_LIMIT}, the published TabPFN v2 row limit.",
        "## Positives per country in one outer test fold\n\n" + to_markdown(fold_positives),
        "These are the positives behind one fold's per-country PR-AUC, which is why the primary metric "
        "is computed on concatenated out-of-fold predictions.",
        "## Category levels with few rows in some country\n\n"
        "Levels with fewer than 30 rows in at least one country. These are the levels a model may barely "
        "see when that country is the only source, or meet rarely when that country is held out.\n\n"
        + to_markdown(support[support["min_cell"] < 30]),
        "## How different the countries' feature distributions are\n\n"
        "Total variation distance between country pairs: 0 means identical distributions, 1 means disjoint.\n\n"
        + to_markdown(covariate_distance(train)),
        "## Positive rate by level, per country\n\n"
        "Shows whether a feature relates to the label in the same direction across countries. "
        "This is description, not evidence about causes.\n\n"
        + "\n\n".join(
            to_markdown(rate_by_feature(train, column))
            for column in ["location_type", "cellphone_access", "gender_of_respondent", "education_level", "job_type"]
        ),
    ]
    (RESULTS_DIR / "data_audit.md").write_text("\n\n".join(sections) + "\n", encoding="utf-8")

    path = save_run(
        "phase1_data_audit",
        config={"seeds": SEEDS, "n_outer": N_OUTER, "n_inner": N_INNER},
        metrics={
            "labelled_rows": len(train),
            "per_country": counts.to_dict(orient="records"),
            "fold_size_summary": size_summary.to_dict(orient="records"),
        },
    )
    print(f"Wrote data_audit.md, per_country_counts.csv, fold_sizes.csv and {path.name}")


if __name__ == "__main__":
    main()
