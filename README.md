# Financial Inclusion in Africa — End-to-End ML Pipeline

A reproducible, SOLID-principle ML pipeline predicting bank account ownership
across Kenya, Rwanda, Tanzania, and Uganda, built with an emphasis on honest
evaluation methodology over leaderboard-chasing — and taken through to a
containerized FastAPI service with ONNX export, prediction logging, and PSI
drift monitoring (§10).

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
| Model serving | FastAPI `/predict` + `/health`, strict Pydantic contract, 422/500/503 error semantics (§10.2) |
| Containerization | Multi-stage `python:3.13-slim` image, non-root, read-only, healthcheck, resource limits (§10.3) |
| ONNX export + benchmark | Parity-verified `model.onnx`, native vs ONNX Runtime latency tables (§10.4) |
| Monitoring | Non-blocking JSON-lines prediction log + schedulable PSI drift check (§10.5) |
| Deployment trade-offs | Cloud vs edge vs on-prem for Uganda → `reports/deployment_tradeoffs.md` (§10.6) |

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
- **Accuracy and the confusion matrix** are also computed
  (`evaluation/metrics.py`) and reported in the full metrics tables
  (`reports/baseline_results.md`, `reports/ablation_study_report.md`) —
  but, like ROC-AUC, accuracy is *not* the headline metric: at this
  dataset's ~14% positive rate, always predicting "No" scores ~86%
  accuracy while catching zero true positives, so it's included for
  completeness rather than as something to optimize. The confusion matrix
  is reported as the sum across the 5 outer-validation folds (each row a
  disjoint slice of the data, so summing double-counts nothing).
- **All metrics reported as mean ± std across the 5 outer folds**, not a
  single number — a model at 0.65 ± 0.02 PR-AUC is a materially different
  (and better) result than 0.65 ± 0.15, even with an identical headline
  number.

**Core results table** — Phase 4, nested-CV tuned (full methodology,
per-country breakdown, statistical test, and recommendation:
`reports/ablation_study_report.md`; untuned Phase 3 floor:
`reports/baseline_results.md`):

| Model | Imbalance strategy | PR-AUC | F1 | ROC-AUC |
|---|---|---|---|---|
| XGBoost | class-weight | **0.590 ± 0.019** | 0.547 ± 0.016 | 0.865 ± 0.003 |
| XGBoost | resampling (SMOTE) | 0.577 ± 0.017 | 0.537 ± 0.018 | 0.859 ± 0.004 |
| PyTorch (2-layer) | class-weight | 0.584 ± 0.019 | 0.546 ± 0.019 | 0.864 ± 0.006 |
| PyTorch (2-layer) | resampling (SMOTE) | 0.568 ± 0.018 | 0.525 ± 0.018 | 0.852 ± 0.008 |
| PyTorch (3-layer) | best strategy (class-weight) | 0.581 ± 0.019 | 0.542 ± 0.014 | 0.862 ± 0.006 |

**Shipped: XGBoost, class-weight.** Best mean PR-AUC *and* tied-tightest
std, though not statistically distinguishable from PyTorch-best at n=5
folds (paired t-test p=0.087, Wilcoxon p=0.125) — chosen on the balance of
PR-AUC stability, Uganda-specific per-country fairness, and
interpretability, not a bare PR-AUC-highest-wins rule (full reasoning:
`reports/ablation_study_report.md` §8). SMOTE underperforms class-weighting
for *both* model families, consistent with caveats the post-SMOTE EDA
checkpoint flagged before any ablation ran
(`notebooks/02_smote_distribution_eda.ipynb`). Logistic Regression
(untuned sanity floor, not part of this grid): PR-AUC 0.559 ± 0.019 — both
tuned XGBoost and PyTorch now clear it.

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
fixtures — none of it depends on `data/raw/` being present. The exceptions
are the serving and ONNX tests (`test_serving_api.py`, `test_onnx_export.py`),
which exercise the real `models/production/` artifact and need the serving
environment (pydantic v2, onnxmltools); they **skip visibly**, with the
reason, when either is missing. The deployment steps are in §10.7.

---

## 9. Repository Structure

```text
financial-inclusion-africa/
├── README.md
├── IMPLEMENTATION_PLAN.md
├── pyproject.toml
├── requirements.txt             # training/research environment
├── requirements-serving.txt     # serving-only deps (FastAPI, pydantic v2, numpy, xgboost-cpu)
├── requirements-onnx.txt        # ONNX export + benchmark tooling
├── Dockerfile                   # multi-stage slim image for the API
├── docker-compose.yml           # api service + on-demand drift-check job
├── .dockerignore                # allow-list: raw survey data can never enter the image
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
├── models/
│   └── production/              # deployable artifact (scripts/train_production_model.py)
│       ├── model.json           # XGBoost booster, native JSON
│       ├── metadata.json        # feature order, vocabulary, threshold, checksums, version
│       ├── model.onnx           # parity-verified ONNX export (scripts/export_onnx.py)
│       ├── onnx_export_report.json
│       └── reference_profile.json   # PSI reference distributions (pooled + per country)
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
│       ├── interpretability/
│       │   ├── __init__.py
│       │   └── shap_analysis.py      # SHAP importance + dependence plots on XGBoost
│       │
│       ├── serving/                  # online inference -- numpy + xgboost only, no pandas/sklearn
│       │   ├── app.py                # FastAPI app: lifespan loading, /predict, /health, error mapping
│       │   ├── schemas.py            # Pydantic request/response contract
│       │   ├── encoder.py            # numpy re-implementation of the fitted XGBoost preprocessing
│       │   ├── predictor.py          # artifact integrity checks + inference
│       │   ├── prediction_log.py     # non-blocking JSON-lines prediction log
│       │   ├── raw_records.py        # stdlib CSV reader for offline tooling
│       │   └── config.py             # environment-variable configuration
│       │
│       └── monitoring/
│           ├── psi.py                # PSI + binning
│           ├── reference_profile.py  # builds reference_profile.json from training data
│           └── drift_check.py        # schedulable drift CLI (exit codes 0/10/20/30/1)
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
│   ├── inference_benchmark.md/.csv       # native vs ONNX latency (Windows host)
│   ├── inference_benchmark_container.md/.csv  # same, inside the 1-CPU Linux container
│   ├── deployment_tradeoffs.md           # cloud vs edge vs on-prem for Uganda
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
│   ├── test_reporting.py
│   ├── test_serving_api.py       # every status-code path, NaN/Inf, logging, integrity checks
│   ├── test_onnx_export.py       # verifier passes a faithful export, rejects a broken one
│   ├── test_prediction_log.py
│   ├── test_psi.py
│   └── test_drift_check.py
│
└── scripts/
    ├── run_eda.py                    # re-executes notebooks/01_eda.ipynb in place
    ├── run_preprocessing.py
    ├── run_baseline_training.py
    ├── run_tuning_and_ablation.py
    ├── train_production_model.py     # refit shipped model + parity gates -> models/production/
    ├── export_onnx.py                # ONNX export + numerical verification
    ├── benchmark_inference.py        # native vs ONNX Runtime latency tables
    └── simulate_traffic.py           # baseline / rural-outreach traffic for the drift demo
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

## 10. Serving, Deployment & Monitoring

The Phase 4 winner (XGBoost, class-weight) taken from a research pipeline to
a deployable, monitored service. The guiding rule throughout: **no silent
failures** — every artifact is verified before it is trusted, and every
failure mode surfaces as a specific HTTP status, log line, or exit code.

### 10.1 Producing the deployable artifact

Phase 4 refit the winning model on the full training set only in memory (for
SHAP) — nothing was persisted. `scripts/train_production_model.py` reproduces
that exact refit (`production_xgb_params`, seed 42, no re-tuning, ~20 s) and
writes `models/production/`:

- `model.json` — XGBoost booster in its native JSON format.
- `metadata.json` — feature order, category vocabulary, recode map, outlier
  threshold, observed numeric ranges, the **decision threshold (0.4551 = the
  median of the five per-fold F1-optimal thresholds**, per the ablation
  report's §3.4), SHA-256 of the training data and of the model, and a
  content-addressed version: **`xgb-cw-5ec2d564be13`**.

Nothing is written unless two parity gates pass: the serving encoder must
reproduce the sklearn preprocessing pipeline **bit-for-bit on all 23,524
training rows**, and the raw booster on that output must match the
training-time `predict_proba`.

The service uses its own numpy encoder (`serving/encoder.py`) instead of an
unpickled sklearn `Pipeline`. That keeps pandas and sklearn out of the image
and the request path (~12 µs per record versus milliseconds), avoids
unpickling (which executes code), and lets the ONNX model consume the same
input vector.

### 10.2 FastAPI service (`src/fin_inclusion/serving/`)

**Input contract:** the 10 raw survey fields. `uniqueid` and `year` are not
inputs; `year` is determined by `country`.

| Field | Validation |
|---|---|
| 8 categoricals (`country`, `location_type`, `cellphone_access`, `gender_of_respondent`, `relationship_with_head`, `marital_status`, `education_level`, `job_type`) | `Literal` enums spelled exactly as in the raw CSVs, including the "Dont know"-style answers (a valid, informative response per EDA 1.3) |
| `household_size` | strict int, 1–30 (training max 21; values 22–30 are served with an out-of-distribution warning) |
| `age_of_respondent` | strict int, 16–100 |

Strict ints reject `3.5`, `"5"`, `true`, `NaN` and `Infinity`. Python's JSON
parser accepts the last two tokens, so this matters. Unknown fields are
rejected (`extra="forbid"`). An unseen category like `"uganda"` gets a 422
instead of the training pipeline's all-zero "unknown" encoding, which would
quietly produce a wrong score.

**Response:** `probability_bank_account`, `predicted_bank_account` (at the
0.4551 threshold), `confidence` (probability of the predicted class),
`decision_threshold`, `model_version`, `request_id`, `warnings`. The score
is a **ranking score, not a calibrated probability**: training with
`scale_pos_weight` = 3.57 means 20.6% of training rows score above the
threshold, against a true positive rate of 14.1%.

**Status codes:**

| Code | Meaning |
|---|---|
| 422 | Invalid client input (one entry per offending field; body never echoes the raw value, so a rejected `NaN` can't crash the error response) |
| 503 | Not ready: model missing or failing integrity checks, or prediction log not writable. Retryable, sends `Retry-After` |
| 500 | A valid request hit a server-side failure (booster error, non-finite score, encoder inconsistency). Logged with the request ID |

**Load-time integrity checks** (`Predictor.load`, run once in FastAPI's
`lifespan`):

- both files present;
- the model's SHA-256 matches the metadata;
- booster feature order matches the metadata;
- the API schema's allowed values equal the training vocabulary;
- a smoke-test prediction succeeds.

A failed load doesn't crash the process. It's logged with a traceback, and
`/health` reports 503 with the reason.

**Efficiency:**

- The model loads once per process.
- Scoring runs on the threadpool; XGBoost's C API releases the GIL, so
  concurrent requests score in parallel.
- `inplace_predict(validate_features=False)` skips a per-call name check
  that load time already did.
- `INFERENCE_THREADS=1` avoids thread fan-out that costs more than it saves
  on single rows and oversubscribes a container's CPU quota.

### 10.3 Container (`Dockerfile`, `docker-compose.yml`)

- **Base image `python:3.13-slim`, two stages.** Serving needs only prebuilt
  wheels, so there's no reason for a CUDA or PyTorch base. Alpine's musl
  libc would force numpy and XGBoost to compile from source. The runtime
  stage receives only the finished virtualenv.
- **`xgboost-cpu`** (5.7 MB) instead of `xgboost` (99 MB with CUDA kernels):
  same version and model format. scipy (139 MB) stays, because XGBoost
  imports it at module load.
- **Layer order:** dependencies → model artifact → source. A code edit
  doesn't reinstall packages (verified: the install layer stays `CACHED`).
- **Hardening:**
  - non-root user (UID 10001) and a read-only root filesystem with a
    temporary `/tmp`;
  - `cap_drop: ALL` and `no-new-privileges`;
  - a `HEALTHCHECK` against `/health` (unhealthy when the model can't load);
  - port bound to `127.0.0.1` only (there's no auth yet);
  - 1 CPU / 512 MB limits (measured ~76 MiB in use).
- **Prediction logs** live on the named volume `prediction-logs`
  (`/app/logs`). It survives `down`/rebuilds and inherits the non-root
  user's ownership.
- **`.dockerignore` is an allow-list,** so raw survey CSVs (personal data)
  can never enter the image.

### 10.4 ONNX export and benchmark

`scripts/export_onnx.py` converts the booster (opset 15, dynamic batch axis
`features: float32[None, 40]`). It writes `model.onnx` (0.81 MB) only if
every check passes:

- `onnx.checker`'s full check;
- batch sizes 1, 7 and 1000 all run;
- **max |Δp| = 2.1e-7 against native XGBoost across 33,654 rows** (the
  whole raw train and test sets plus edge cases);
- **zero Yes/No flips** at the production threshold.

Two converter quirks are handled explicitly:

- **Feature names:** it only accepts `f0…fN`. The real names are stored in
  the ONNX metadata instead.
- **Label output:** it emits a `label` output hard-wired to a 0.5 cut-off.
  That output is removed from the graph, so no consumer can use the wrong
  decision rule.

`scripts/benchmark_inference.py` times the model call only: 200 warmup
calls, 2,000 timed runs per cell, both runtimes on 1 thread, alternating in
ABBA blocks of 50 calls, and a fresh batch each run. It reports mean ± 95%
CI, p50/p95/p99/max. Results inside the 1-CPU deployment container
(`reports/inference_benchmark_container.md`):

| Batch | Native XGBoost p50 | ONNX Runtime p50 | ONNX speed-up |
|---:|---:|---:|---:|
| 1 | 519 µs | 33 µs | **15.7×** |
| 32 | 679 µs | 399 µs | 1.7× |
| 1024 | 5.18 ms | 11.38 ms | **0.46×** (slower) |

ONNX Runtime wins on single requests, where most of native XGBoost's time is
fixed per-call overhead. Native XGBoost wins on large single-threaded
batches. Both are under 1 ms per request, so the API keeps native XGBoost,
and ONNX is ready as an optional backend (and for edge devices, §10.6).

### 10.5 Monitoring

**Prediction log** (`serving/prediction_log.py`):

- Each successful prediction is one JSON line: timestamp, request ID, model
  version, input features, score, prediction, confidence, threshold,
  `latency_ms`, `inference_ms`, warnings. That's ~616 bytes per line.
- The request only does a non-blocking put onto a bounded queue. A
  background thread encodes and appends in batches to
  `predictions-<UTC date>-<host>.jsonl`.
- If the disk stalls and the queue fills, events are **dropped, counted,
  shown in `/health` and logged at WARNING**. Latency is protected, and the
  gap is never silent.
- An unwritable log directory makes the service return 503. Predictions that
  affect financial access are not served without an audit trail.

**Reference profile** (`python -m fin_inclusion.monitoring.reference_profile`):

- Built from the exact training file the model was fit on (checked against
  the SHA-256 in `metadata.json`).
- Decile bins for numerics (with observed-value edges, so there are no
  empty bins between integers), one bin per level for categoricals, plus
  the distribution of the model's **own score**.
- Stored for the pooled data **and per country**. A Uganda-only deployment
  compared against the four-country pool would falsely report drift on day
  one.

**Drift check** (`python -m fin_inclusion.monitoring.drift_check`, or
`docker compose run --rm drift-check`):

- PSI per feature and on the score, read with the standard thresholds
  (< 0.10 stable, 0.10–0.25 moderate, > 0.25 significant). Each result names
  the bin that moved most.
- Options: `--hours` or `--last-n`, `--country`, `--json-out`.
- **Exit codes for scheduling and alerting:** 0 stable, 10 moderate, 20
  significant, 30 insufficient data (< 500 predictions by default), 1 check
  failed (no logs, reference mismatch, > 1% malformed lines).
- Only events from the reference's model version are counted.

**Verified against real API traffic** (`scripts/simulate_traffic.py`):

| Scenario | Verdict | Evidence |
|---|---|---|
| 2,000 held-out `Test_v2.csv` respondents | stable (exit 0) | largest PSI 0.009; predicted-Yes rate 20.6% → 20.4% |
| 2,000 rural, low-education respondents (a financial-outreach population) | **significant (exit 20)** | `location_type` PSI 3.42, `education_level` 1.93, model score 0.40; predicted-Yes 20.6% → 5.3% |
| 600 Ugandan rural-outreach respondents vs the Uganda reference | **significant (exit 20)** | predicted-Yes 10.9% → 0.2% |

### 10.6 Deployment trade-offs (Uganda)

`reports/deployment_tradeoffs.md` compares cloud, edge and on-prem for this
model in the Ugandan context: rural connectivity, the Data Protection and
Privacy Act 2019 (no major cloud provider has a region in Uganda), capex vs
USD-billed opex, and who carries the maintenance burden.

**Recommendation: a hybrid.**

- **Hub:** an in-country server or Kampala colocation runs the container.
  It is the system of record, holds the prediction log and runs a nightly
  `drift-check --country Uganda`.
- **Edge:** offline ONNX scoring on field devices for rural outreach. Logs
  sync back to the hub for monitoring.
- **Cloud:** only for work that involves no personal data.

### 10.7 Running it

```bash
# 1. Build the artifacts (training environment: requirements.txt)
python scripts/train_production_model.py

# 2. Serving environment (pydantic v2; kept separate from the training env)
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-serving.txt -r requirements-onnx.txt pytest httpx
.venv/Scripts/python -m pip install --no-deps -e .
.venv/Scripts/python scripts/export_onnx.py                           # model.onnx + parity report
.venv/Scripts/python -m fin_inclusion.monitoring.reference_profile   # reference_profile.json
.venv/Scripts/python scripts/benchmark_inference.py                  # reports/inference_benchmark.*

# 3. Serve locally...
.venv/Scripts/python -m uvicorn fin_inclusion.serving.app:app --port 8000
# ...or in Docker
docker compose up -d --build            # http://localhost:8000/docs

# 4. Predict
curl -s -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "country": "Uganda", "location_type": "Rural", "cellphone_access": "Yes",
  "household_size": 5, "age_of_respondent": 34, "gender_of_respondent": "Female",
  "relationship_with_head": "Spouse", "marital_status": "Married/Living together",
  "education_level": "Primary education", "job_type": "Self employed"}'
# -> {"probability_bank_account": 0.180, "predicted_bank_account": "No", ...}

# 5. Monitor
python scripts/simulate_traffic.py --n 2000 --scenario baseline
docker compose run --rm drift-check                         # pooled, last 7 days
docker compose run --rm drift-check --country Uganda --hours 24

# Tests for the serving stack
.venv/Scripts/python -m pytest tests/test_serving_api.py tests/test_onnx_export.py \
    tests/test_prediction_log.py tests/test_psi.py tests/test_drift_check.py
```

Use Git Bash, or `curl.exe` in Windows PowerShell 5.1, where plain `curl` is
an alias for `Invoke-WebRequest`. After retraining, run the export and
reference-profile steps again, then rebuild the image. The model version
changes automatically, and the drift check ignores events from other model
versions.

---

## 11. Known Limitations / Honesty Notes

- Four national surveys pooled into one dataset is a domain-shift risk, not
  fully resolved by country-stratified CV alone — flagged rather than hidden.
- SHAP explanations describe the XGBoost model's learned associations, not
  causal drivers of financial inclusion.
- Survey years differ by country (2016–2018), so the pooled model implicitly
  assumes reasonable temporal stability across that window; not separately
  validated.
- **The deployed model is a deterministic refit**, not the original Phase 4
  in-memory object (which was never saved): same params, seed and data,
  verified by the parity gates in §10.1.
- **The API has no authentication or TLS yet.** It's only safe because it's
  bound to localhost. A reverse proxy with auth is required before any
  network exposure.
- **Scores are not calibrated probabilities** (see §10.2). Calibration
  (isotonic/Platt) would be needed before the score is read as a
  likelihood.
- **PSI catches input and score drift, not concept drift.** No ground-truth
  outcomes flow back from deployment, so accuracy decay itself isn't
  measured. Uganda, the weakest and least stable segment (PR-AUC
  0.553 ± 0.047, 2,101 training rows), most needs retraining on current
  local data.
- **The benchmark ran on a laptop CPU** (Ryzen 7 5800U). Absolute latencies
  will differ on servers; the native-vs-ONNX ratios are the part that
  carries over.
