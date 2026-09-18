# Phase 4 — Hyperparameter Tuning & Ablation Study Report

Source: `scripts/run_tuning_and_ablation.py`, `reports/ablation_raw_results.json`,
`notebooks/02_smote_distribution_eda.ipynb`. Baseline reference:
`reports/baseline_results.md` (Phase 3, untuned).

> **Status: methodology and post-SMOTE findings final; results table, PR
> curves, per-country breakdown, statistical test, and recommendation are
> pending the nested-CV run and will be filled in from
> `reports/ablation_raw_results.json` once it completes — nothing below is
> fabricated or estimated in its place.**

---

## 1. Methodology

### 1.1 Nested cross-validation

5 outer folds, stratified jointly on `country`+`target` (`make_country_
target_stratify_key`, per Phase 2.8/EDA 1.1–1.2). For each outer fold:

1. **Inner tuning** (`tuning/optuna_search.py`): an Optuna TPE search runs
   its own inner stratified CV *inside* the outer-training split only —
   the outer validation fold is never touched during tuning.
2. **Threshold selection** (`evaluation/threshold_selection.py`): a further
   80/20 stratified split of the same outer-training fold picks the
   F1-maximizing threshold, again without touching the outer validation
   fold.
3. **Final refit**: a fresh model, built with the tuned hyperparameters, is
   refit on the *full* outer-training fold.
4. **Scoring**: the refit model is scored once against the untouched outer
   validation fold, at the tuned threshold.

This is implemented generically in `evaluation/cv_runner.run_nested_cv`,
parameterized by a `tune_fn` (Phase 4.1's inner/outer separation) so the
same nested-CV driver runs every grid cell below.

### 1.2 Imbalance strategies

- **Class-weight.** For PyTorch, `ClassWeightStrategy`'s inverse-frequency
  sample weights (unchanged from Phase 3). For XGBoost specifically, this
  arm instead tunes `scale_pos_weight` — XGBoost's own native imbalance
  lever — as a regular hyperparameter, via `NoImbalanceStrategy` (so a
  generic `sample_weight` isn't applied on top and double-correcting).
- **SMOTE** (`imbalance/strategies.SMOTEStrategy`), fit *only* inside each
  inner training-fold split and the outer-training-fold's final refit —
  never on any validation split at any level (Phase 4.2). Cleared for use
  by the post-SMOTE EDA checkpoint below, with two documented caveats.

### 1.3 Hyperparameter search spaces

| Model | Tuned | Fixed |
|---|---|---|
| XGBoost | `max_depth`, `learning_rate`, `n_estimators`, `min_child_weight`, `subsample`, `colsample_bytree`, and (class-weight arm only) `scale_pos_weight` | — |
| PyTorch MLP | `hidden_dims` (per-layer, count set by the grid cell's depth), `embedding_dim`, `dropout`, `lr`, `weight_decay` | `epochs` (see 1.4), `batch_size` |

### 1.4 Tuning budget (documented tractability trade-off)

Trial counts and PyTorch epoch counts during the *search* are deliberately
modest, calibrated against measured fit times on this dataset/machine so
the full 5-grid-cell nested search finishes in roughly one to two hours
of interactive wall-clock time rather than many:

| Parameter | Value |
|---|---|
| Outer folds | 5 |
| Inner folds (per Optuna trial's CV) | 3 |
| XGBoost trials per arm | 15 |
| PyTorch trials per arm | 8 |
| PyTorch epochs during search | 6 |
| PyTorch epochs for the final refit | 20 |

This trades search *breadth* for tractability — it does not affect
evaluation *honesty*: every reported number still comes from a model that
never saw its outer validation fold during tuning, threshold selection, or
fitting. A larger trial budget would likely find marginally better
hyperparameters; it would not change the nested-CV structure's leakage
guarantees. Measured mid-run: XGBoost's SMOTE arm took noticeably longer
than its class-weight arm (SMOTE roughly 1.7x's the training-fold row
count, and Optuna's search drifted toward larger, slower XGBoost
configurations once they scored better) — expected search behavior, not a
methodology concern.

### 1.5 Statistical comparison

Paired comparison (`evaluation/reporting.compare_paired_pr_auc`) on the 5
outer-fold-level PR-AUC scores, XGBoost-best vs. PyTorch-best (the better
imbalance-strategy arm within each model family) — both a paired t-test and
a Wilcoxon signed-rank test are reported (Phase 4.7); Wilcoxon is the more
robust reading at n=5 since it makes no normality assumption.

### 1.6 Interpretability

SHAP (`interpretability/shap_analysis.py`) global importance and
dependence plots on the winning XGBoost arm, refit once more on the full
training set with its tuned hyperparameters (Phase 4.8) — a check that the
tuned model's learned associations are consistent with the Phase 1 EDA
(`cellphone_access`, `education_level`, `country` expected to dominate),
not a claim of causal drivers of financial inclusion (README §10).

---

## 2. Post-SMOTE distribution checkpoint (Phase 4.3)

Full detail: `notebooks/02_smote_distribution_eda.ipynb`. Summary:

**Decision: SMOTE cleared for use as an ablation arm, with two documented
caveats** — not an unconditional pass:

1. **XGBoost (one-hot) branch:** ~17.6% of synthetic minority rows end up
   with a *blended* country one-hot (no single country dummy at 1.0), since
   interpolating between two one-hot rows from different countries produces
   fractional values in between. The aggregate synthetic country *mix*
   still tracks the real minority distribution reasonably well (Kenya
   ~49% vs. 46%, Rwanda ~29% vs. 30%, Tanzania ~18% each, Uganda ~4% vs.
   6%), but **Uganda's synthetic rows are disproportionately more likely to
   be blended** — Uganda has the fewest real positive examples (181) for
   SMOTE's 5-nearest-neighbor search to draw same-country neighbors from.
   Since Uganda is this brief's primary country of interest, its per-country
   numbers on the SMOTE arm (§5 below) get extra scrutiny for this reason.
   Correlation structure is not meaningfully distorted beyond the expected
   one-hot mutual-exclusivity softening (max |Δr| ≈ 0.17, between two
   `education_level` dummies of the same column).
2. **PyTorch (embedding-index) branch:** no synthetic row gets an invalid
   (non-integer) category code — but only because unscaled category-code
   magnitudes (`job_type` std ≈ 3.1, `relationship_with_head` ≈ 1.9)
   dominate SMOTE's Euclidean nearest-neighbor search over the
   already-standardized numeric columns (std ≈ 1.0), which happens to bias
   neighbor selection toward same-category rows. This avoids the
   invalid-category failure mode, but for a methodological reason unrelated
   to SMOTE being category-aware (it isn't) — noted as a caveat on this
   branch's SMOTE arm, not a defect that blocks it.

Class balance: every outer-training fold went from ~14% positive to an
exact 50/50 balance after SMOTE, adding only minority rows (majority-class
row count is unchanged before/after in every fold) — confirmed against the
real per-fold splits, not just the unit-test-level synthetic case.

---

## 3. Results table

**PENDING** — filled in from `reports/ablation_raw_results.json` once the
nested-CV run completes. Target shape (README §6):

| Model | Imbalance strategy | PR-AUC | F1 | ROC-AUC |
|---|---|---|---|---|
| XGBoost | class-weight | *pending* | *pending* | *pending* |
| XGBoost | resampling (SMOTE) | *pending* | *pending* | *pending* |
| PyTorch (2-layer) | class-weight | *pending* | *pending* | *pending* |
| PyTorch (2-layer) | resampling (SMOTE) | *pending* | *pending* | *pending* |
| PyTorch (3-layer) | best strategy from above | *pending* | *pending* | *pending* |

For reference, Phase 3's **untuned** baselines (class-weight only, default
0.5 threshold): XGBoost 0.551 ± 0.017 PR-AUC, PyTorch (2-layer) 0.580 ±
0.019 PR-AUC, Logistic Regression floor 0.559 ± 0.019 PR-AUC (full detail
in `reports/baseline_results.md`) — the numbers above should be read
against this floor.

## 4. PR curve overlay

**PENDING** — XGBoost-best vs. PyTorch-best, saved to
`reports/figures/pr_curve_overlay.png` once the run completes.

## 5. Per-country breakdown (overall best model)

**PENDING.**

## 6. Statistical comparison

**PENDING** — paired t-test and Wilcoxon signed-rank on the 5 outer-fold
PR-AUC scores, XGBoost-best vs. PyTorch-best.

## 7. Interpretability (SHAP)

**PENDING** — global importance ranking and dependence plots, saved to
`reports/figures/shap_global_importance.png` and
`reports/figures/shap_dependence_*.png`.

## 8. Recommendation

**PENDING** — will be justified against PR-AUC stability (mean ± std, not
just mean), per-country fairness (Uganda specifically, given the §2 SMOTE
caveat), and interpretability tradeoffs, per the Implementation Plan's
explicit instruction not to default to "highest PR-AUC wins."
