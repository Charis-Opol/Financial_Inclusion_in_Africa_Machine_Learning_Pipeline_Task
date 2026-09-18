# Implementation Plan — Financial Inclusion in Africa Pipeline

Five phases, matching the brief. Each phase lists concrete tasks, the SOLID
module(s) it produces or touches (see `README.md` §9 for the full repo
structure), and — for the EDA and preprocessing phases specifically — what
each task is *for*, per the brief's request.

---

## Phase 1 — EDA

**Goal:** Fully characterize the data before any modeling decision is
treated as final. Every task below is chosen because it resolves a specific
open question raised during the initial pass, not because it's a generic
EDA checklist item.

| # | EDA task | What it tells us |
|---|---|---|
| 1.1 | Per-country target distribution (bar chart, exact %) | Confirms the 3x spread in account ownership across Kenya/Rwanda/Tanzania/Uganda; sets expectations for per-country reporting later |
| 1.2 | Per-country feature distributions (`education_level`, `job_type`, `cellphone_access`, etc.) | Confirms whether these differ by country (different economies) — decides whether a pooled model is defensible or whether country-conditional preprocessing is needed |
| 1.3 | Missingness-in-disguise correlation check — does `"Dont know"`/`"Other/Dont know/RTA"` correlate with `bank_account`? | Decides whether to keep as its own category (if predictive) vs. treat as noise to impute |
| 1.4 | Bivariate: `bank_account` rate by `education_level`, `job_type`, `cellphone_access`, `relationship_with_head`, `marital_status` (crosstabs / normalized bar charts) | Identifies which categoricals actually separate the classes; `cellphone_access` checked closely given the Kenya/M-Pesa hypothesis |
| 1.5 | `household_size` and `age_of_respondent` vs. target (binned) | Checks for non-linear relationships (e.g. working-age peak) that a linear model would miss but a tree/MLP could capture |
| 1.6 | `household_size` outlier sanity check (8 rows > 15) | Distinguishes legitimate extended households from data-entry errors before deciding to cap or keep |
| 1.7 | Cardinality audit: unique level counts for every categorical | Directly informs the encoding decision in Phase 2 (one-hot vs. embeddings) |
| 1.8 | Train/test category consistency check | Confirms no unseen categories will break inference-time encoding, since test has no target to catch this indirectly |
| 1.9 | `relationship_with_head` × `marital_status` crosstab | Tests the suspected redundancy (e.g. "Head" + "Widowed" vs. "Spouse" + "Married") before deciding to keep both or engineer a combined feature |
| 1.10 | `uniqueid` uniqueness check (composite key confirmation) | Confirms `uniqueid + country + year` is the real key; documented so no one joins incorrectly downstream |

**Deliverable:** `notebooks/01_eda.ipynb` + `reports/eda_summary.md`
summarizing findings and the decision each one fed into.

**Modules touched:** `src/fin_inclusion/data/loader.py`,
`data/validators.py` (read-only exploration at this stage — no
transformation logic yet).

---

## Phase 2 — Data Preparation & Preprocessing

**Goal:** Turn EDA findings into a deterministic, leakage-safe, testable
preprocessing pipeline.

| # | Task | Why |
|---|---|---|
| 2.1 | Build `DataLoader` enforcing the `uniqueid+country+year` composite key and schema | Prevents the join bug identified in EDA 1.10 from ever reaching downstream code |
| 2.2 | Recode missingness-in-disguise categories as an explicit `"unknown"`/kept-level, per the Phase 1.3 finding | Keeps potentially predictive information instead of discarding or mis-imputing it |
| 2.3 | Flag (not auto-drop) `household_size` outliers based on Phase 1.6 sanity check | Avoids silently deleting legitimate extended-household respondents |
| 2.4 | Build two parallel encoding branches: one-hot (XGBoost path) and categorical-index-for-embeddings (PyTorch path) | Different model families need different representations; decided in Phase 1.7 based on cardinality |
| 2.5 | Fit all encoders on train only; implement explicit unseen-category handling | Test set has no target to sanity-check against (Phase 1.8), so this must fail loudly, not silently, at inference |
| 2.6 | Scale numeric features for the PyTorch branch; leave raw for XGBoost | Trees are scale-invariant, MLPs are not — using one scaling regime for both would be a silent bug |
| 2.7 | (Conditional) engineer a combined `relationship_marital` feature only if Phase 1.9 crosstab shows meaningful redundancy without full overlap | Avoids engineering a feature the EDA doesn't actually support |
| 2.8 | Wire `country` into both the pooled feature set and the CV stratification key | Reflects the Phase 1.1/1.2 finding that country is the dominant structural effect |
| 2.9 | Unit tests for every transformer (`tests/test_preprocessing.py`) | Preprocessing bugs are the most common silent-failure mode in ML pipelines; SOLID's Interface Segregation only pays off if each piece is independently testable |

**Deliverable:** `src/fin_inclusion/preprocessing/` fully implemented and
tested; `data/interim/` and `data/processed/` checkpoints reproducible via
`scripts/run_preprocessing.py`.

---

## Phase 3 — Model Selection & Training (Untuned Baselines)

**Goal:** Establish honest, untuned baselines before any tuning budget is
spent, so later tuning gains are measured against a real floor.

1. Implement `BaseModel` ABC (`fit`, `predict_proba`, `get_params`,
   `save`/`load`) in `models/base_model.py`.
2. Implement `LogisticRegressionModel` — sanity-check floor.
3. Implement `XGBoostModel` (default hyperparameters, no tuning yet).
4. Implement `PyTorchMLP` (2-layer, default hyperparameters, embedding
   layers for categoricals), with `ClassWeightStrategy` as the initial
   imbalance handling (SMOTE deferred to Phase 4 per the brief's explicit
   ordering).
5. Run all three through a single stratified 5-fold CV (country + target
   stratified, non-nested at this stage — nesting is introduced in Phase 4
   once tuning enters the picture) using `evaluation/cv_runner.py`.
6. Compute PR-AUC, ROC-AUC, F1 (at default 0.5 threshold, explicitly labeled
   as *not yet* the tuned threshold), mean ± std across folds.
7. Report per-country breakdown for this untuned baseline as a reference
   point — establishes whether the Kenya/Uganda gap exists even before any
   tuning, which matters for interpreting Phase 4's results correctly.
8. Write up baseline results as the first rows of the results table defined
   in `README.md` §6.

**Deliverable:** `scripts/run_baseline_training.py`, populated baseline rows
in `reports/` results table, `tests/test_models.py`.

**Explicitly not in this phase:** hyperparameter search, SMOTE, 3-layer
MLP, SHAP, statistical testing — all deferred to Phase 4 so the baseline
stays a clean, untuned reference point.

---

## Phase 4 — Hyperparameter Tuning & Ablation Study

**Goal:** Tune honestly (nested CV), compare imbalance strategies, run the
depth ablation, and produce a standalone ablation report — in that order,
with a mandatory post-SMOTE EDA checkpoint before the ablation study itself
begins, per your instruction.

1. **Nested CV scaffolding:** wrap Phase 3's CV runner with an inner Optuna
   loop (`tuning/optuna_search.py`) per the outer/inner structure defined in
   `README.md` §6. Outer folds' validation data is never touched during
   inner tuning.
2. **Imbalance strategy implementations:**
   - `ClassWeightStrategy` (already used in Phase 3, kept as one arm).
   - `SMOTEStrategy` — fit **only** inside each inner training-fold split,
     never on validation/outer-fold data.
3. **Post-SMOTE EDA checkpoint (required before the ablation study runs):**
   - Class balance before/after SMOTE, per fold, to confirm the resampling
     behaved as expected.
   - Feature distribution comparison (real vs. synthetic minority samples)
     for the key predictors identified in Phase 1 (`cellphone_access`,
     `education_level`, `age_of_respondent`, `country`) — checks SMOTE
     isn't generating implausible synthetic respondents (e.g. mixing
     country-specific patterns nonsensically, given how strong the country
     effect is).
   - Correlation matrix before/after SMOTE — checks synthetic samples
     haven't distorted feature relationships in a way that would make the
     model learn spurious patterns.
   - Documented in `notebooks/02_smote_distribution_eda.ipynb`; any red
     flags found here directly gate whether SMOTE is trusted as an arm of
     the ablation study or reported as unreliable.
4. **Hyperparameter tuning runs** (inner Optuna search, per model family):
   - XGBoost: tree depth, learning rate, `n_estimators`,
     `min_child_weight`, `subsample`/`colsample_bytree`, `scale_pos_weight`
     (class-weight arm only).
   - PyTorch 2-layer MLP: hidden dims, learning rate, dropout, weight decay,
     embedding dimensions.
5. **Threshold tuning:** F1-maximizing threshold selected on inner
   validation data per fold, per the methodology in `README.md` §6 — run
   for every model/strategy combination, not just the eventual winner.
6. **Full ablation grid execution** (the core results table from
   `README.md` §6):
   - XGBoost × {class-weight, SMOTE}
   - PyTorch 2-layer × {class-weight, SMOTE}
   - PyTorch 3-layer × best strategy identified above (isolates depth as a
     single variable rather than re-running the full grid a third time)
7. **Statistical comparison:** paired t-test/Wilcoxon signed-rank on the 5
   fold-level PR-AUC scores, XGBoost-best vs. PyTorch-best.
8. **Interpretability:** SHAP global importance + 1–2 dependence plots on
   the tuned XGBoost model (`cellphone_access`, `education_level`, `country`
   expected to dominate, per Phase 1 bivariate findings — confirms the
   tuned model learned something consistent with the EDA).
9. **Per-country final breakdown** on the overall best model — the primary
   Uganda-relevant number this whole pipeline is built to produce.

**Ablation study report (required deliverable):**
`reports/ablation_study_report.md`, containing:
- Full results table (all rows, mean ± std).
- Post-SMOTE distribution/correlation findings from step 3, and whether
  they raised any concerns.
- PR curve overlay (XGBoost-best vs. PyTorch-best).
- Per-country breakdown table.
- Statistical test result and interpretation.
- **Recommendation section:** which model + imbalance strategy to ship,
  explicitly justified against PR-AUC stability (mean ± std, not just
  mean), per-country fairness (Uganda performance specifically), and
  interpretability tradeoffs — not just "highest PR-AUC wins."

---

## Phase 5 — Documentation

**Goal:** Make the repo self-explanatory to a reviewer who has none of this
conversation's context.

1. Finalize `README.md` (already drafted — this phase is about keeping it in
   sync with what actually got built, not writing it from scratch).
2. Docstrings on every public class/function in `src/fin_inclusion/`,
   following one consistent style (NumPy or Google docstring format —
   pick one and enforce it, e.g. via `pydocstyle` in CI).
3. `reports/eda_summary.md` finalized from Phase 1 notebook findings.
4. `reports/ablation_study_report.md` finalized from Phase 4.
5. Inline comments only where *why*, not *what*, needs explaining (e.g. why
   SMOTE is fit inside the fold, not what `SMOTE.fit_resample` does).
6. `CONTRIBUTING.md` or a "Reproducing this pipeline" section covering exact
   environment setup, seed values, and command sequence
   (`run_eda.py` → `run_preprocessing.py` → `run_baseline_training.py` →
   `run_tuning_and_ablation.py`).
7. Push to GitHub with the full structure from `README.md` §9, tag a release
   corresponding to the final ablation results so the reported numbers are
   tied to a specific, reproducible commit.

**Deliverable:** Public GitHub repo, README and ablation report both
readable standalone, no undocumented decision points remaining.

---

## Phase Ordering Notes

- Phases 1 and 2 are strictly sequential (preprocessing decisions are
  derived from EDA findings, not assumed).
- Phase 3 must complete *before* Phase 4 begins, so tuning gains are always
  measured against a real untuned floor.
- Within Phase 4, the post-SMOTE EDA checkpoint (step 3) is a hard gate
  before the ablation grid (step 6) runs — this is intentional, not just
  good practice, per your explicit instruction.
- Phase 5 runs continuously in parallel with 1–4 in practice (docstrings and
  the README are updated as each phase lands), but is listed last because
  the *final consolidated* documentation pass depends on all prior results
  existing.