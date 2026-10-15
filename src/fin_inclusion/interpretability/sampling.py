"""The background (reference) set and the explanation set.

Two fixed, seeded row selections every XAI script shares, so that every
method is measured against the same reference and on the same respondents --
otherwise a difference between two methods could be a difference between two
samples.

Background set: what "the feature is absent" means for the interventional
methods (a removed feature is replaced by its value in a background row and
the outputs are averaged). It is drawn proportionally within country x
target strata, so the base value (mean model output over the background)
approximates the model's average output over the training population -- the
natural reading of "compared with a typical respondent". An equal-allocation
design would over-represent account holders and small countries and move
that reference away from the population the model was trained on.

Explanation set: a proportional stratified sample for pooled results, plus
every row of the focus country (Uganda, the primary quantity of interest)
for country-specific results. The pooled sample is disjoint from the
background, so no explained respondent is also part of its own reference.
Pooled statistics use the stratified sample only: adding all Uganda rows to
it would over-weight Uganda roughly sixfold.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fin_inclusion.preprocessing.pipeline_factory import make_country_target_stratify_key


def stratified_positions(
    strata: pd.Series, n: int, seed: int, exclude: np.ndarray | None = None
) -> np.ndarray:
    """`n` row positions drawn proportionally within `strata`, sorted.

    `exclude` removes positions from the pool before drawing.
    """
    pool = np.arange(len(strata))
    if exclude is not None:
        pool = np.setdiff1d(pool, exclude)
    if not 0 < n < len(pool):
        raise ValueError(f"Cannot draw {n} rows from a pool of {len(pool)}")
    chosen, _ = train_test_split(
        pool, train_size=n, stratify=strata.to_numpy()[pool], random_state=seed, shuffle=True
    )
    return np.sort(chosen)


@dataclass(frozen=True, eq=False)
class SampleSets:
    """Row positions into the training frame as `DataLoader.load_train` returns it."""

    background: np.ndarray
    pooled: np.ndarray
    focus: np.ndarray
    focus_country: str
    seed: int
    training_data_sha256: str

    @property
    def explanation(self) -> np.ndarray:
        """Every row that gets explained: the pooled sample plus the focus country."""
        return np.union1d(self.pooled, self.focus)

    def save(self, path: Path, composition: dict | None = None) -> None:
        """Writes the summary first, indented for reading, then each list of
        row positions on a single line (thousands of one-number lines would
        bury the part a person actually opens the file for)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "seed": self.seed,
            "training_data_sha256": self.training_data_sha256,
            "focus_country": self.focus_country,
            "n_background": int(self.background.size),
            "n_pooled": int(self.pooled.size),
            "n_focus": int(self.focus.size),
            "n_explanation": int(self.explanation.size),
            "composition": composition or {},
        }
        row_lists = {"background": self.background, "pooled": self.pooled, "focus": self.focus}
        lines = [f' "{name}": {json.dumps(rows.tolist())}' for name, rows in row_lists.items()]
        summary_text = json.dumps(summary, indent=1).removesuffix("\n}")
        path.write_text(summary_text + ",\n" + ",\n".join(lines) + "\n}\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path, training_data_sha256: str | None = None) -> "SampleSets":
        """`training_data_sha256`, when given, must match the file the
        positions were drawn from: positions into a different file would
        silently select different respondents."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if training_data_sha256 is not None and payload["training_data_sha256"] != training_data_sha256:
            raise ValueError(f"{path} was built from a different training file than the one loaded")
        return cls(
            background=np.array(payload["background"], dtype=np.int64),
            pooled=np.array(payload["pooled"], dtype=np.int64),
            focus=np.array(payload["focus"], dtype=np.int64),
            focus_country=payload["focus_country"],
            seed=payload["seed"],
            training_data_sha256=payload["training_data_sha256"],
        )


def build_sample_sets(
    country: pd.Series,
    target: pd.Series,
    background_size: int,
    explanation_sample_size: int,
    seed: int,
    training_data_sha256: str,
    focus_country: str = "Uganda",
) -> SampleSets:
    strata = make_country_target_stratify_key(
        pd.DataFrame({"country": country.to_numpy(), "target": target.to_numpy()}), target_col="target"
    )
    focus = np.flatnonzero(country.to_numpy() == focus_country)
    if focus.size == 0:
        raise ValueError(f"No rows for focus country {focus_country!r}")
    background = stratified_positions(strata, background_size, seed)
    pooled = stratified_positions(strata, explanation_sample_size, seed, exclude=background)
    return SampleSets(
        background=background,
        pooled=pooled,
        focus=focus,
        focus_country=focus_country,
        seed=seed,
        training_data_sha256=training_data_sha256,
    )


LOCAL_EXAMPLE_RULES: dict[str, str] = {
    "confident_tp": "has an account, predicted Yes: the highest score among such rows",
    "confident_tn": "no account, predicted No: the lowest score among such rows",
    "false_negative": "has an account, predicted No: the lowest score among such rows (the most confident miss)",
    "false_positive": "no account, predicted Yes: the highest score among such rows (the most confident false alarm)",
    "focus_typical": "focus-country row whose score is closest to that country's median score",
}
"""The rule behind each local example. Examples are picked by these rules,
never by how their explanation looks; ties go to the lowest row position."""


def select_local_examples(
    candidates: np.ndarray,
    focus_candidates: np.ndarray,
    y_binary: np.ndarray,
    score: np.ndarray,
    threshold: float,
) -> dict[str, int]:
    """Row positions of the five local examples, chosen by `LOCAL_EXAMPLE_RULES`.

    `y_binary` and `score` are indexed by row position (full training
    frame). The four confusion-matrix examples come from `candidates`, the
    focus-country example from `focus_candidates`.
    """
    candidates, focus_candidates = np.sort(candidates), np.sort(focus_candidates)
    actual = y_binary[candidates] == 1
    predicted = score[candidates] >= threshold

    def pick(mask: np.ndarray, highest: bool, name: str) -> int:
        if not mask.any():
            raise ValueError(f"No candidate row satisfies the rule for {name!r}")
        pool = candidates[mask]
        scores = score[pool]
        # argmax/argmin return the first extreme, and `pool` is sorted, so ties go to the lowest position
        return int(pool[np.argmax(scores) if highest else np.argmin(scores)])

    focus_scores = score[focus_candidates]
    return {
        "confident_tp": pick(actual & predicted, True, "confident_tp"),
        "confident_tn": pick(~actual & ~predicted, False, "confident_tn"),
        "false_negative": pick(actual & ~predicted, False, "false_negative"),
        "false_positive": pick(~actual & predicted, True, "false_positive"),
        "focus_typical": int(focus_candidates[np.argmin(np.abs(focus_scores - np.median(focus_scores)))]),
    }


def composition_table(positions: np.ndarray, country: pd.Series, target: pd.Series) -> dict[str, dict[str, int]]:
    """Row counts per country and target level in a selection, for the log."""
    selected = pd.DataFrame(
        {"country": country.to_numpy()[positions], "target": target.to_numpy()[positions]}
    )
    counts = selected.groupby(["country", "target"]).size()
    return {
        c: {str(t): int(counts.get((c, t), 0)) for t in sorted(selected["target"].unique())}
        for c in sorted(selected["country"].unique())
    }
