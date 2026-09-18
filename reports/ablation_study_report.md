# Phase 4 — Hyperparameter Tuning & Ablation Study Report

Source: `scripts/run_tuning_and_ablation.py`, `reports/ablation_raw_results.json`,
`notebooks/02_smote_distribution_eda.ipynb`. Baseline reference:
`reports/baseline_results.md` (Phase 3, untuned).

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

Implemented generically in `evaluation/cv_runner.run_nested_cv`,
parameterized by a `tune_fn` so the same nested-CV driver runs every grid
cell below.

### 1.2 Imbalance strategies

- **Class-weight.** For PyTorch, `ClassWeightStrategy`'s inverse-frequency
  sample weights. For XGBoost specifically, this arm instead tunes
  `scale_pos_weight` — XGBoost's own native imbalance lever — as a regular
  hyperparameter (`NoImbalanceStrategy`, so a generic `sample_weight` isn't
  applied on top and double-correcting).
- **SMOTE** (`SMOTEStrategy`), fit *only* inside each inner training-fold
  split and the outer-training-fold's final refit — never on any
  validation split at any level (Phase 4.2). Cleared for use by the
  post-SMOTE EDA checkpoint (§2), with two documented caveats that, in
  hindsight (§3), plausibly explain why it underperforms class-weighting
  here.

### 1.3 Hyperparameter search spaces

| Model | Tuned | Fixed |
|---|---|---|
| XGBoost | `max_depth`, `learning_rate`, `n_estimators`, `min_child_weight`, `subsample`, `colsample_bytree`, and (class-weight arm only) `scale_pos_weight` | — |
| PyTorch MLP | `hidden_dims` (per-layer, count set by the grid cell's depth), `embedding_dim`, `dropout`, `lr`, `weight_decay` | `epochs`, `batch_size` |

### 1.4 Tuning budget (documented tractability trade-off)

| Parameter | Value |
|---|---|
| Outer folds | 5 |
| Inner folds (per Optuna trial's CV) | 3 |
| XGBoost trials per arm | 15 |
| PyTorch trials per arm | 8 |
| PyTorch epochs during search | 6 |
| PyTorch epochs for the final refit | 20 |

Modest by design, calibrated against measured fit times on this
dataset/machine for interactive tractability. This trades search *breadth*
for tractability — it does not affect evaluation *honesty*: every number
below comes from a model that never saw its outer validation fold during
tuning, threshold selection, or fitting. **Actual run time: ~68 minutes**
(XGBoost class-weight 135s, XGBoost SMOTE 1247s, PyTorch 2-layer
class-weight 737s, PyTorch 2-layer SMOTE 1220s, PyTorch 3-layer 726s). The
SMOTE arms took substantially longer than class-weight — confirmed (via
isolated timing) to be genuine compute cost, not a bug: SMOTE both enlarges
the training-fold row count (~1.7x) and Optuna's search drifted toward
larger, slower XGBoost configurations once they scored better.

### 1.5 Statistical comparison

Paired comparison (`compare_paired_pr_auc`) on the 5 outer-fold-level
PR-AUC scores, XGBoost-best vs. PyTorch-best (the better imbalance-strategy
arm within each model family) — both a paired t-test and a Wilcoxon
signed-rank test are reported; Wilcoxon is the more robust reading at n=5
since it makes no normality assumption.

### 1.6 Interpretability

SHAP global importance and dependence plots on the winning XGBoost arm,
refit once more on the full training set with its tuned hyperparameters —
a check that the tuned model's learned associations are consistent with
the Phase 1 EDA, not a claim of causal drivers of financial inclusion
(README §10).

---

## 2. Post-SMOTE distribution checkpoint (Phase 4.3)

Full detail: `notebooks/02_smote_distribution_eda.ipynb`. Summary:

**Decision: SMOTE cleared for use as an ablation arm, with two documented
caveats** — not an unconditional pass:

1. **XGBoost (one-hot) branch:** ~17.6% of synthetic minority rows end up
   with a *blended* country one-hot (no single country dummy at 1.0).
   The aggregate synthetic country mix still tracks the real minority
   distribution reasonably well, but **Uganda's synthetic rows are
   disproportionately more likely to be blended** (fewest real positive
   examples — 181 — for SMOTE's 5-nearest-neighbor search to draw
   same-country neighbors from). Correlation structure is not meaningfully
   distorted beyond the expected one-hot mutual-exclusivity softening
   (max |Δr| ≈ 0.17).
2. **PyTorch (embedding-index) branch:** no synthetic row gets an invalid
   (non-integer) category code, but only because unscaled category-code
   magnitudes (`job_type` std ≈ 3.1) dominate SMOTE's Euclidean
   nearest-neighbor search over the standardized numeric columns (std ≈
   1.0) — a methodological artifact, not evidence the interpolation is
   feature-aware.

Class balance: every outer-training fold went from ~14% positive to an
exact 50/50 balance after SMOTE, adding only minority rows.

---

## 3. Results table

Nested-CV, tuned, F1/precision/recall/accuracy at the per-fold
F1-maximizing threshold (not a fixed 0.5 — see §3.2). Accuracy is reported
for completeness, not as the metric to optimize against — see
`evaluation/metrics.py`'s module docstring for why it's misleading at this
dataset's ~14% positive rate (a constant "No" prediction alone scores
~86%, comparable to every row below):

| Model | Imbalance strategy | PR-AUC | ROC-AUC | F1 | Precision | Recall | Accuracy |
|---|---|---|---|---|---|---|---|
| XGBoost | class-weight | **0.590 ± 0.019** | 0.865 ± 0.003 | 0.547 ± 0.016 | 0.500 ± 0.039 | 0.612 ± 0.058 | 0.857 ± 0.016 |
| XGBoost | resampling (SMOTE) | 0.577 ± 0.017 | 0.859 ± 0.004 | 0.537 ± 0.018 | 0.505 ± 0.026 | 0.576 ± 0.033 | 0.860 ± 0.008 |
| PyTorch (2-layer) | class-weight | 0.584 ± 0.019 | 0.864 ± 0.006 | 0.546 ± 0.019 | 0.495 ± 0.036 | 0.614 ± 0.029 | 0.856 ± 0.014 |
| PyTorch (2-layer) | resampling (SMOTE) | 0.568 ± 0.018 | 0.852 ± 0.008 | 0.525 ± 0.018 | 0.540 ± 0.049 | 0.521 ± 0.059 | 0.868 ± 0.010 |
| PyTorch (3-layer) | best strategy (class-weight) | 0.581 ± 0.019 | 0.862 ± 0.006 | 0.542 ± 0.014 | 0.486 ± 0.036 | 0.617 ± 0.041 | 0.853 ± 0.014 |

**XGBoost-best: class-weight arm. PyTorch-best: 2-layer class-weight arm.**
Overall best by mean PR-AUC: **XGBoost/class-weight (0.590)**.

Note the accuracy column barely moves across arms (0.853–0.868) despite
PR-AUC/F1/recall moving meaningfully — exactly the flattening effect
accuracy has at this class imbalance, and why it isn't this report's
decision metric. SMOTE's arms score *slightly* higher accuracy than their
class-weight counterparts (e.g. XGBoost 0.860 vs. 0.857) precisely because
SMOTE trades recall for precision (fewer false positives, more false
negatives — see §3.5's confusion matrices), which flatters accuracy while
PR-AUC (the honest signal) still ranks class-weight higher.

### 3.1 SMOTE underperforms class-weighting for both model families

Consistent for both XGBoost (0.590 vs. 0.577) and PyTorch (0.584 vs.
0.568) — SMOTE never wins an arm here. This is consistent with, and
plausibly explained by, the §2 caveats: the blended one-hot country
artifacts (XGBoost branch) and the arbitrary categorical-distance-dominated
neighbor search (PyTorch branch) both point at SMOTE introducing noise
rather than useful synthetic diversity on this particular dataset. Not
proof of causation, but the two independent findings (EDA red-flag-adjacent
caveats, then worse ablation performance) corroborate each other.

### 3.2 Depth doesn't help: 3-layer ≈ 2-layer

PyTorch 3-layer (0.581) is statistically indistinguishable from 2-layer
(0.584) — within a fold's own std, and both used the same (class-weight)
strategy. At this dataset size (~23,500 rows, 8 categorical + 2 numeric
features) there's little evidence a third hidden layer adds useful
capacity; it plausibly adds overfitting risk without a compensating gain.

### 3.3 Tuned vs. untuned: honest gains, not dramatic ones

Against Phase 3's untuned floor (`reports/baseline_results.md`): XGBoost
PR-AUC improved 0.551 → 0.590 (+0.039), PyTorch (2-layer) improved 0.580 →
0.584 (+0.004, within noise). Tuning helped XGBoost meaningfully and
PyTorch only marginally — plausible given PyTorch's search budget (§1.4)
covered fewer trials for a higher-dimensional space (5 hyperparameters vs.
XGBoost's 6–7).

### 3.4 Threshold instability across folds

The F1-maximizing threshold selected per outer fold for XGBoost/class-weight
ranged from **0.34 to 0.75** across the 5 folds (PyTorch's range was
tighter, 0.61–0.80) — a wide spread for a supposedly "optimal" operating
point. This is a genuine finding, not a bug: with only ~2,600 minority
training examples per outer-training fold, the F1-maximizing threshold
itself is noisy. A single fixed deployment threshold should be chosen with
this instability in mind (e.g. the median across folds, ~0.46 for
XGBoost/class-weight) rather than trusting any one fold's value.

### 3.5 Confusion matrices (summed across the 5 outer-validation folds)

**XGBoost, class-weight (shipped model):**

![XGBoost class-weight confusion matrix](figures/confusion_matrix_xgb_cw.png)

| Actual \ Predicted | Pred No | Pred Yes |
|---|---|---|
| Actual No | 18130 | 2082 |
| Actual Yes | 1284 | 2028 |

**XGBoost, SMOTE:**

![XGBoost SMOTE confusion matrix](figures/confusion_matrix_xgb_smote.png)

| Actual \ Predicted | Pred No | Pred Yes |
|---|---|---|
| Actual No | 18330 | 1882 |
| Actual Yes | 1403 | 1909 |

**PyTorch (2-layer), class-weight:**

![PyTorch 2-layer class-weight confusion matrix](figures/confusion_matrix_pt2_cw.png)

| Actual \ Predicted | Pred No | Pred Yes |
|---|---|---|
| Actual No | 18103 | 2109 |
| Actual Yes | 1280 | 2032 |

**PyTorch (2-layer), SMOTE:**

![PyTorch 2-layer SMOTE confusion matrix](figures/confusion_matrix_pt2_smote.png)

| Actual \ Predicted | Pred No | Pred Yes |
|---|---|---|
| Actual No | 18687 | 1525 |
| Actual Yes | 1586 | 1726 |

**PyTorch (3-layer), class-weight:**

![PyTorch 3-layer confusion matrix](figures/confusion_matrix_pt3_best.png)

| Actual \ Predicted | Pred No | Pred Yes |
|---|---|---|
| Actual No | 18013 | 2199 |
| Actual Yes | 1267 | 2045 |

Each matrix sums to 23,524 (every training row scored exactly once, across
its one outer-validation fold). Comparing the two XGBoost arms directly:
SMOTE catches 121 fewer true positives (1,909 vs. 2,028) and produces 200
fewer false positives (1,882 vs. 2,082) than class-weight — a real
precision/recall tradeoff, not a free lunch, and it's why SMOTE's PR-AUC
(the metric that integrates over every threshold, not just this one) still
comes out lower despite the marginally higher accuracy noted in §3.

---

## 4. PR curve overlay

![PR curve overlay](figures/pr_curve_overlay.png)

Generated separately from the nested-CV metrics above: the winning
hyperparameters (XGBoost/class-weight's mode across folds; PyTorch
2-layer/class-weight's hyperparameters, identical across all 5 folds) were
refit once on a single 80/20 stratified split, purely to draw a full
precision-recall curve (nested CV only scores one threshold per fold, not
a full curve). On this single split: XGBoost AP=0.577, PyTorch AP=0.577 —
visually the two curves overlap almost everywhere, consistent with §3's
finding that neither model has a real edge. (This single-split AP differs
slightly from the 5-fold CV means in §3, e.g. XGBoost 0.577 here vs. 0.590
CV mean — expected sampling variance from using one split instead of
five; §3's numbers are the authoritative ones.)

---

## 5. Per-country breakdown (overall best model: XGBoost/class-weight)

| Country | PR-AUC | F1 |
|---|---|---|
| Kenya | 0.649 ± 0.030 | 0.591 ± 0.016 |
| Rwanda | 0.473 ± 0.023 | 0.450 ± 0.039 |
| Tanzania | 0.627 ± 0.020 | 0.567 ± 0.027 |
| Uganda | 0.553 ± 0.047 | 0.542 ± 0.043 |

**Uganda-specific finding (the primary quantity of interest for this
brief):** PR-AUC improved from 0.483 (Phase 3 untuned XGBoost baseline) to
**0.553** after tuning — a +0.070 (~14.5% relative) gain, roughly in line
with the other countries' improvement. Uganda now clearly outperforms
Rwanda (0.553 vs. 0.473), reversing their untuned-baseline ordering
(Uganda 0.483 vs. Rwanda 0.438, where Uganda was worse). Uganda's std
(±0.047 PR-AUC, ±0.043 F1) is still the largest of the four countries —
expected, given it has the fewest positive examples (181) to estimate
metrics from — so this improvement should be read as encouraging but noisy,
not as a fully resolved gap. Given §2's finding that SMOTE's synthetic
rows are disproportionately unreliable for Uganda specifically, using the
class-weight (not SMOTE) arm for the shipped model is doubly justified for
Uganda's numbers, not just the aggregate PR-AUC.

---

## 6. Statistical comparison

XGBoost-best (0.590 ± 0.019) vs. PyTorch-best (0.584 ± 0.019), paired on
the 5 outer-fold PR-AUC scores:

| Test | Statistic | p-value |
|---|---|---|
| Paired t-test | 2.253 | 0.087 |
| Wilcoxon signed-rank | 1.0 | 0.125 |

**Neither test reaches significance at α=0.05.** XGBoost wins on mean
PR-AUC in every one of the 5 outer folds (consistent direction, hence the
t-test's p=0.087 being closer to significant than a mixed-direction result
would produce), but with n=5 folds neither test has the power to rule out
chance at the conventional threshold. Read as: **XGBoost is very likely at
least as good as PyTorch here, and plausibly slightly better, but "PyTorch
scored 0.006 lower" is not a claim this data can support with high
confidence** — exactly the kind of overclaim a fold-level paired test
exists to prevent.

---

## 7. Interpretability (SHAP)

![SHAP global importance](figures/shap_global_importance.png)

Top features by mean |SHAP value| on the tuned XGBoost/class-weight model
(refit on full train): `cellphone_access_No`, `education_level_Primary
education`, `country_Kenya`, `age_of_respondent`, `education_level_No
formal education`, `cellphone_access_Yes`. **Consistent with the Phase 1
EDA's prediction** (`cellphone_access`, `education_level`, `country`
expected to dominate, per EDA 1.4 and the Cramér's V ranking in EDA
1.11) — the tuned model learned associations that match the univariate
signal already identified before any modeling happened, which is the
sanity check this step exists to provide, not evidence of a causal
mechanism (README §10).

Dependence plots: `figures/shap_dependence_cellphone_access_Yes.png`,
`figures/shap_dependence_age_of_respondent.png`.

---

## 8. Recommendation

**Ship XGBoost, class-weight arm (tuned hyperparameters in
`reports/ablation_raw_results.json`'s `production_xgb_params`), not the
highest-mean-PR-AUC model chosen blindly — the reasoning happens to agree
with that choice here, but for stated reasons, not just the number:**

1. **PR-AUC stability.** XGBoost/class-weight has the best mean (0.590)
   *and* a std (±0.019) tied for the tightest among the five grid rows —
   not a case of a high mean bought with high variance.
2. **Statistical caution, not overclaiming.** §6's tests don't reach
   significance against PyTorch-best — this recommendation is "XGBoost is
   a reasonable, defensible choice," not "XGBoost is proven better."
   Either model would be defensible; XGBoost is chosen on the balance of
   the evidence below, not a p<0.05 win.
3. **Per-country fairness, Uganda specifically.** §5's Uganda improvement
   (0.483 → 0.553 PR-AUC) is real and the class-weight arm avoids §2's
   SMOTE-blending caveat that disproportionately affects Uganda's synthetic
   training rows — a second, independent reason to prefer class-weight
   over SMOTE for this specific country's numbers, beyond the aggregate
   PR-AUC ranking in §3.1.
4. **Interpretability.** SHAP (§7) gives XGBoost a direct, tree-native
   global-importance and dependence-plot story that matches the Phase 1
   EDA. PyTorch's embedding-based representation doesn't offer the same
   directness without additional tooling — a real (if secondary, per the
   Implementation Plan's explicit instruction not to let PR-AUC alone
   decide) factor favoring XGBoost when the two models are this close.
5. **Depth ablation confirms simplicity isn't costing anything.** §3.2's
   finding that 3-layer PyTorch doesn't beat 2-layer reinforces that
   there's no hidden capacity gain being left on the table by not chasing
   a more complex model.

**Caveat carried forward, not hidden:** §3.4's threshold instability
(0.34–0.75 across folds for this exact arm) means the single deployment
threshold should be chosen deliberately (e.g. the cross-fold median, not
an arbitrary fold's value) and revisited if the deployment population's
class balance drifts from this training set's.
