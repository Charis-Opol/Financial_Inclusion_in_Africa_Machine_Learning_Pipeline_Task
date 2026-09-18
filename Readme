# Financial Inclusion in Africa — End-to-End ML Pipeline

A reproducible, SOLID-principle ML pipeline predicting bank account ownership
across Kenya, Rwanda, Tanzania, and Uganda, built with an emphasis on honest
evaluation methodology over leaderboard-chasing.

---

## 1. Project Framing

**Task from brief:** Design and implement a complete end-to-end ML pipeline on
a Uganda/Africa-relevant dataset, with clean reproducible code, proper
evaluation methodology, and solid documentation.

**Mapped to this repo:**

| Brief requirement | Decision |
|---|---|
| Dataset | Financial Inclusion in Africa (Kenya, Rwanda, Tanzania, Uganda) |
| Preprocessing/feature pipeline | `sklearn`-compatible pipeline, missingness-in-disguise handling, encoding strategy per cardinality |
| Baseline model | XGBoost (+ Logistic Regression as a sanity-check floor) |
| Deep learning variant | PyTorch MLP, 2-layer then 3-layer ablation |
| Stratified k-fold + metrics | Nested stratified 5-fold CV (country + target stratified), PR-AUC primary, F1, ROC-AUC |
| SOLID + documented code | Interface-driven modules (see Repo Structure) |

---

## 2. Dataset Decision

**Chosen dataset:** Financial Inclusion in Africa (Zindi / World Bank
Findex-derived household survey data).

**Why:** Uganda-relevant (Uganda is one of four countries), tabular
(appropriate for both XGBoost and a PyTorch MLP baseline), binary classified
target with realistic class imbalance, and rich enough categorical structure
to justify a genuine feature-engineering and encoding discussion rather than
a token one.

**Shape:**

| | Train | Test |
|---|---|---|
| Rows | 23,524 | 10,086 |
| Columns | 13 (12 features + target) | 12 (target held out) |

**Target:** `bank_account` — No 85.9% / Yes 14.1%. Moderate-severe imbalance.

---

## 3. EDA Findings That Drove Downstream Decisions

These are the findings from the initial exploratory pass, kept here because
each one directly produced a design decision elsewhere in this document.

1. **`uniqueid` is not globally unique.** It repeats across
   country/year combinations; `uniqueid + country + year` together is the
   real key (0 duplicates on that composite). → Recorded as an explicit
   warning in the data loader's docstring so no one joins on `uniqueid` alone.

2. **Missingness-in-disguise.** No literal `NaN`s, but placeholder categories
   exist inside otherwise-clean columns: `"Dont know"` (`marital_status`, 8
   rows), `"Other/Dont know/RTA"` (`education_level`, 35 rows),
   `"Dont Know/Refuse to answer"` (`job_type`, 126 rows). `.isnull().sum()`
   would miss all of these. → Decision: keep as their own category level
   rather than impute/drop, pending a check of whether they correlate with
   the target (see Phase 1, EDA task list).

3. **Strong country effect.** These are four separate national surveys
   stacked together, not one cross-sectional sample (different survey
   years per country: Kenya 2018, Rwanda 2016, Tanzania 2017, Uganda 2018).
   Account ownership varies 3x by country (Kenya 25.1%, Rwanda 11.5%,
   Tanzania 9.2%, Uganda 8.6%), most plausibly driven by Kenya's
   mobile-money (M-Pesa) penetration. → Decisions:
   - Build one pooled model with `country` as a feature (not four separate
     models), but
   - Stratify CV folds by **country + target jointly**, not target alone.
   - Report all evaluation metrics **both in aggregate and per-country**,
     since Uganda-specific performance is the primary quantity of interest
     for this brief, and a pooled model could look good in aggregate while
     under-serving Uganda if it over-fits to Kenya's larger positive class.

4. **Cardinality/encoding decision drivers.** `job_type` (10 levels) and
   `relationship_with_head` (6 levels) need real encoding decisions, not a
   default one-hot. → Decision: one-hot for the XGBoost path (tree models
   handle wide sparse one-hot fine and it keeps the pipeline
   interpretable/SHAP-friendly); learned embeddings for the PyTorch path
   (categorical embedding layers), since that is a natural strength of the
   deep learning variant and gives the ablation study something genuine to
   compare.

5. **Possible redundancy.** `relationship_with_head` and `marital_status`
   likely overlap (e.g. "Head of Household" + "Widowed" vs "Spouse" +
   "Married"). → Decision: keep both as-is initially, but run a crosstab in
   Phase 1 EDA before deciding whether to engineer a combined feature; do
   not drop either without evidence.

6. **Household size outliers.** 8 rows with `household_size` > 15. →
   Decision: sanity-check against plausible household sizes (extended
   family households do exist in this context) rather than blind capping;
   flag rather than silently remove.

7. **Train/test category consistency.** Test set has no target to
   sanity-check against, so unseen categories must be caught explicitly. →
   Decision: encoders are fit on train only, with an explicit
   "unseen category" handling strategy (a dedicated "unknown" bucket) so
   the pipeline fails loudly rather than silently at inference time.

---

## 4. Preprocessing Decisions

| Task | Decision | Why |
|---|---|---|
| Missingness-in-disguise | Kept as own category level | Plausibly informative (e.g. "don't know your job" may itself correlate with financial exclusion), not random missingness |
| Categorical encoding (XGBoost path) | One-hot | Interpretable, SHAP-compatible, trees handle sparsity well |
| Categorical encoding (PyTorch path) | Learned embeddings | Standard practice for tabular deep learning, gives ablation study a real axis to compare |
| Numeric features (`age_of_respondent`, `household_size`) | Scaled for PyTorch, left raw for XGBoost | Trees are scale-invariant; MLPs are not |
| Outlier handling (`household_size` > 15) | Flagged, not auto-capped, pending sanity check | Avoid silently discarding legitimate extended-household data |
| Country | Kept as a pooled feature + used in stratification | Strongest single predictor; needs to inform both modeling and validation design |
| Imbalance strategy | Both class-weighting and SMOTE resampling tested as an ablation axis | No a priori reason to prefer one; brief explicitly asks for an ablation study |

---

## 5. Modeling Decisions

- **Baseline:** XGBoost (primary baseline) plus a Logistic Regression floor
  for sanity-checking that any model beats a linear separator.
- **Deep learning variant:** PyTorch MLP.
  - Stage 1: 2-layer MLP, evaluated untuned.
  - Stage 2: hyperparameter-tuned 2-layer MLP.
  - Stage 3: 3-layer MLP ablation using the best imbalance strategy found
    in Stage 2, to isolate the effect of depth specifically.
- **Both model families are evaluated under both imbalance strategies**
  (class-weighting vs. SMOTE resampling), giving a 2x2-plus-depth ablation
  grid rather than a single number per model family.

---

## 6. Evaluation Methodology

**Validation scheme:** Nested stratified k-fold cross-validation.

- **Outer loop:** 5-fold, stratified jointly on `country` and `bank_account`.
  Final metrics are computed only on outer validation folds — data never
  seen during tuning.
- **Inner loop:** Optuna hyperparameter search on each outer training fold.
  SMOTE/resampling, where used, is fit **inside** the training-fold-only
  step of each inner split, never on validation or test data, to avoid
  leakage.

**Metrics:**

- **PR-AUC — primary/headline metric.** At ~14% positive rate, ROC-AUC is
  flattered by the majority class; PR-AUC is the honest signal.
- **F1** at a **tuned decision threshold** — not a default 0.5. The
  threshold that maximizes F1 on inner validation data is selected as part
  of the nested CV (tuned honestly, never peeked from the outer fold) and
  reported explicitly per fold/model in the results table.
- **Precision and recall reported separately**, not just folded into F1,
  because which one matters more here is a modeling choice (for financial
  inclusion targeting, over- vs under-predicting "has account" has
  different real-world implications) and deserves an explicit stated
  tradeoff rather than a silent default.
- **ROC-AUC** reported alongside PR-AUC for comparability with other
  published work on this dataset, not as the decision metric.
- **All metrics reported as mean ± std across the 5 outer folds**, not a
  single number — a model at 0.65 ± 0.02 PR-AUC is a materially different
  (and better) result than 0.65 ± 0.15, even with an identical headline
  number.

**Core results table** (Phase 3 rows populated as untuned baselines --
default hyperparameters, F1 at the default 0.5 threshold, not yet tuned;
see `reports/baseline_results.md` for the full per-country breakdown and
`reports/ablation_study_report.md` for Phase 4's tuned numbers):

| Model | Imbalance strategy | PR-AUC | F1 | ROC-AUC |
|---|---|---|---|---|
| XGBoost | class-weight | 0.551 ± 0.017 | 0.508 ± 0.005 | 0.846 ± 0.005 |
| XGBoost | resampling (SMOTE) | ... | ... | ... |
| PyTorch (2-layer) | class-weight | 0.580 ± 0.019 | 0.506 ± 0.013 | 0.863 ± 0.006 |
| PyTorch (2-layer) | resampling (SMOTE) | ... | ... | ... |
| PyTorch (3-layer) | best strategy from above | ... | ... | ... |

Logistic Regression (sanity-check floor, not part of the ablation grid):
PR-AUC 0.559 ± 0.019, F1 0.496 ± 0.008, ROC-AUC 0.852 ± 0.006 -- both
XGBoost and PyTorch are expected to eventually clear this once tuned;
neither does yet at default hyperparameters, which is the expected shape
of an *untuned* baseline, not a modeling problem.

**Secondary reporting:**

- **Per-country breakdown** of PR-AUC/F1 on the best model — catches a
  pooled model quietly under-performing on Uganda specifically.
- **PR curves**, XGBoost vs. PyTorch (best variants) overlaid — the visual
  centerpiece of the evaluation section; shows the full precision/recall
  tradeoff rather than one threshold's point on it.
- **SHAP** on the XGBoost model — global feature importance plus 1–2
  dependence plots on the strongest features (expected: `cellphone_access`,
  `education_level`, `country`). Doubles as a sanity check that the model
  learned something plausible.
- **Paired statistical test** (paired t-test or Wilcoxon signed-rank) across
  the 5 fold-level PR-AUC scores, XGBoost vs. PyTorch — turns "PyTorch
  scored 0.02 higher" into a defensible claim rather than noise. Cheap to
  compute given the 5 fold-level numbers already exist from nested CV.

---

## 7. Design Principles (SOLID mapping)

| Principle | How it's applied in this repo |
|---|---|
| **S**ingle Responsibility | Data loading, cleaning, feature engineering, encoding, modeling, tuning, and evaluation are each their own module — no god-class pipeline script |
| **O**pen/Closed | Model classes implement a shared `BaseModel` interface (`fit`, `predict_proba`, `get_params`); adding a new model type doesn't require editing existing model code |
| **L**iskov Substitution | Any `BaseModel` subclass (XGBoost, LR, PyTorch MLP) is swappable inside the CV/evaluation loop without the loop knowing which one it's running |
| **I**nterface Segregation | Preprocessing transformers implement only the `fit`/`transform` surface they need (`sklearn`-compatible), not a bloated shared interface |
| **D**ependency Inversion | Pipeline orchestration depends on abstractions (`BaseModel`, `BasePreprocessor`) injected via config, not concrete classes hard-coded into the training script |

---

## 8. Reproducibility

- All random seeds (`numpy`, `torch`, `sklearn`, `optuna`, SMOTE) fixed and
  centralized in `config/config.yaml` (`seed: 42`), threaded explicitly
  through every script and model constructor rather than relying on a
  global default.
- `requirements.txt` / `pyproject.toml` pins exact versions.
- `evaluation/cv_runner.py`'s `StratifiedKFold(shuffle=True, random_state=
  seed)` makes fold assignments deterministic and re-derivable from the seed
  alone; `reports/ablation_raw_results.json` additionally persists the
  per-fold tuned hyperparameters and thresholds Phase 4 actually selected,
  so the reported numbers don't require re-running the (~1-2 hour) nested
  search to audit.
- No metric is computed on data the corresponding model/threshold selection
  step has seen (`evaluation/cv_runner.run_nested_cv`'s outer validation
  fold is untouched by tuning, threshold selection, and the final refit).

### Reproducing this pipeline

Environment: Python ≥3.10 (developed against 3.13), `pip install -r
requirements.txt`, then `pip install -e .` from the repo root so `src/
fin_inclusion` is importable as `fin_inclusion`. Place `Train_v2.csv` /
`Test_v2.csv` under `data/raw/` (gitignored; not redistributed here).

Run in order — each stage's outputs are inputs to the next:

```bash
python scripts/run_eda.py                  # re-executes notebooks/01_eda.ipynb in place
python scripts/run_preprocessing.py         # data/raw -> data/interim -> data/processed
python scripts/run_baseline_training.py     # Phase 3 untuned baselines -> reports/baseline_results.md
python scripts/run_tuning_and_ablation.py   # Phase 4 nested-CV tuning + ablation -> reports/ablation_raw_results.json
```

`run_tuning_and_ablation.py` also requires `notebooks/02_smote_distribution_
eda.ipynb` to have been reviewed first (Phase 4.3's hard gate; already
cleared — see that notebook's summary and §2 of
`reports/ablation_study_report.md`). It is the slow step: the documented
tuning budget (§1.4 of the ablation report) targets roughly one to two
hours on a single CPU machine, not minutes — a deliberate breadth/time
trade-off, not an oversight.

`pytest` runs the full test suite (`tests/`) against small synthetic
fixtures — none of it depends on `data/raw/` being present.

---

## 9. Repository Structure

```
financial-inclusion-africa/
├── README.md
├── IMPLEMENTATION_PLAN.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── config/
│   ├── __init__.py
│   ├── config.yaml              # seeds, paths, CV settings, model hyperparameter search spaces
│   └── settings.py              # typed config loader (dataclass/pydantic)
│
├── data/
│   ├── raw/                     # untouched train.csv / test.csv (gitignored, README describes how to fetch)
│   ├── interim/                 # cleaned-but-unencoded checkpoints
│   └── processed/                # fully encoded, model-ready arrays/frames
│
├── src/
│   └── fin_inclusion/
│       ├── __init__.py
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── loader.py             # DataLoader: reads raw CSVs, enforces schema, documents uniqueid caveat
│       │   └── validators.py         # train/test category-consistency checks, unseen-category detection
│       │
│       ├── preprocessing/
│       │   ├── __init__.py
│       │   ├── base.py               # BasePreprocessor interface (fit/transform, sklearn-compatible)
│       │   ├── cleaning.py           # missingness-in-disguise handling, outlier flagging
│       │   ├── encoders.py           # OneHotBranch (XGBoost path), EmbeddingIndexBranch (PyTorch path)
│       │   └── pipeline_factory.py   # builds sklearn Pipeline / ColumnTransformer per model family
│       │
│       ├── features/
│       │   └── __init__.py           # empty: Phase 2.7's conditional combined-feature engineering
│       │                             # didn't trigger (EDA 1.9 found partial, not total, overlap)
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base_model.py         # BaseModel ABC: fit, predict_proba, get_params, save/load
│       │   ├── logistic_regression.py
│       │   ├── xgboost_model.py
│       │   └── pytorch_mlp.py        # configurable depth (2-layer / 3-layer), embedding layers for categoricals
│       │
│       ├── imbalance/
│       │   ├── __init__.py
│       │   └── strategies.py         # ClassWeightStrategy, SMOTEStrategy — common interface, fit on train-fold only
│       │
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── metrics.py            # PR-AUC, ROC-AUC, F1, precision/recall @ tuned threshold
│       │   ├── threshold_selection.py# inner-fold F1-maximizing threshold search
│       │   ├── cv_runner.py          # nested stratified CV orchestration (country+target stratified)
│       │   └── reporting.py          # results tables, per-country breakdown, statistical tests (paired t-test/Wilcoxon)
│       │
│       ├── tuning/
│       │   ├── __init__.py
│       │   └── optuna_search.py      # inner-loop hyperparameter search per model family
│       │
│       └── interpretability/
│           ├── __init__.py
│           └── shap_analysis.py      # SHAP importance + dependence plots on XGBoost
│
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_smote_distribution_eda.ipynb   # post-SMOTE distribution/correlation checks (Phase 4 pre-ablation)
│
├── reports/
│   ├── eda_summary.md
│   ├── baseline_results.md               # Phase 3 deliverable (untuned baselines)
│   ├── ablation_study_report.md          # Phase 4 deliverable
│   ├── ablation_raw_results.json         # machine-readable Phase 4 output the report above is written from
│   └── figures/                          # PR curves, SHAP plots, per-country bar charts
│
├── tests/
│   ├── test_loader.py
│   ├── test_validators.py
│   ├── test_preprocessing.py
│   ├── test_models.py
│   ├── test_imbalance.py
│   ├── test_metrics.py
│   ├── test_threshold_selection.py
│   ├── test_cv_runner.py
│   ├── test_optuna_search.py
│   ├── test_shap_analysis.py
│   └── test_reporting.py
│
└── scripts/
    ├── run_eda.py                    # re-executes notebooks/01_eda.ipynb in place
    ├── run_preprocessing.py
    ├── run_baseline_training.py
    └── run_tuning_and_ablation.py
```

**Structure rationale:**

- `src/fin_inclusion/` as an installable package (not loose scripts) so
  `tests/` can import it cleanly and the code is portable beyond this repo.
- `models/base_model.py` and `preprocessing/base.py` are the two interfaces
  that make the Open/Closed and Liskov Substitution claims in Section 7
  concrete rather than aspirational.
- `imbalance/strategies.py` is isolated specifically because it must be
  re-instantiated fresh inside every inner training fold — keeping it out of
  `preprocessing/` makes that leakage-avoidance requirement visible in the
  structure itself.
- `reports/ablation_study_report.md` is a named deliverable matching Phase 4
  of the implementation plan below, not an incidental notebook output.
- No separate `pipeline.py` orchestrator: each `scripts/run_*.py` *is* the
  orchestration layer for its phase, composing the same injected
  abstractions (`BaseModel`, `BaseImbalanceStrategy`, the `Pipeline`s from
  `pipeline_factory.py`) a `pipeline.py` module would have — an extra
  wrapper module would have added indirection without adding capability.

---

## 10. Known Limitations / Honesty Notes

- Four national surveys pooled into one dataset is a domain-shift risk, not
  fully resolved by country-stratified CV alone — flagged rather than hidden.
- SHAP explanations describe the XGBoost model's learned associations, not
  causal drivers of financial inclusion.
- Survey years differ by country (2016–2018), so the pooled model implicitly
  assumes reasonable temporal stability across that window; not separately
  validated.
