# XAI decisions log

Every non-obvious choice in the Week 3 explainability work, with the reason.
Entries are appended as each phase lands. Attributions throughout describe
what the **model** learned to rely on; none of them is a causal claim about
what drives bank account ownership.

## Phase 0 — Reconnaissance (findings that constrain later choices)

**D0.1 — No trained MLP exists; it must be retrained.**
Phase 4 never persisted the PyTorch model (only XGBoost was later refit and
saved). All five `pt2_cw` outer folds in `reports/ablation_raw_results.json`
selected the *same* hyperparameters (`hidden_dims=(63, 103)`,
`embedding_dim=6`, `dropout=0.2571`, `lr=0.00153`, `weight_decay=1.53e-6`),
so there is no choice to make among per-fold configs.

**D0.2 — `household_size_outlier` is folded into `household_size`.**
Both models take 11 inputs for 10 raw features: the outlier flag is a
deterministic function of `household_size` (> 15). Its attribution is summed
into `household_size` (`interpretability/grouping.py`,
`DERIVED_FEATURE_PARENTS`). Consequence for baselines: whenever
`household_size` is replaced by a reference value, the flag must be
recomputed from that value, never left at the instance's value.

**D0.3 — The "unknown" bucket is not a trained state in either model.**
For the MLP, embedding index 0 is reserved for unseen categories and no
training row uses it: the three "Dont know"-style answers are recoded to the
string `unknown`, which receives an ordinary non-zero index. Row 0 of every
embedding therefore gets no data gradient. For XGBoost, only three features
have an `_unknown` column; for the other five, "unknown" is an all-zero
one-hot block that never occurs in training. An "unknown" baseline is
therefore an off-distribution reference for both models. It is still run (as
specified), alongside the most-frequent-category baseline, and the
difference between the two is reported as a result.

**D0.4 — The existing SHAP module is left untouched and not reused for results.**
`interpretability/shap_analysis.py` calls `TreeExplainer` without a
background (tree-path-dependent value function), on the 40 one-hot columns,
on an in-memory refit. The Week 3 work needs the interventional value
function against an explicit background, on the persisted artifact,
aggregated to raw features. The `shap_top10` already in
`ablation_raw_results.json` is therefore not comparable with the new results
and is not quoted alongside them.

**D0.5 — The shipped threshold (0.4551) applies to XGBoost only.**
It is the median of XGBoost's per-fold F1-optimal thresholds. The MLP's
per-fold thresholds are 0.61–0.74 (median 0.6824). Any TP/FN/FP/TN labelling
of MLP predictions uses the MLP's own median threshold.

## Phase 1 — Scaffolding

**D1.1 — A dedicated `.venv-xai` environment (`requirements-xai.txt`).**
Neither existing environment can run the whole assignment. The training
interpreter has pydantic 1.10 (the serving schemas the dashboard must reuse
need v2) and a `streamlit` that fails at import: `fastapi==0.115.4` holds
`starlette` at 0.41.3, while `streamlit==1.59.2` declares `starlette>=0.40`
but needs a newer one at runtime. pip resolves that pair without complaint,
so the conflict is avoided structurally: `requirements-xai.txt` is
`requirements.txt` + `captum`, `streamlit`, `pydantic` v2, and deliberately
no `fastapi`. `requirements-serving.txt` and the Dockerfile are unchanged.

**D1.2 — New modules live beside `shap_analysis.py` in `interpretability/`.**
Same responsibility (explaining models), so the same package; the existing
module and its test are not edited.

**D1.3 — One result type, `Attribution`, that refuses to exist without its reference.**
`method`, `model_name`, `target_space` and `baseline_description` are
required and validated non-empty, and are saved inside the same `.npz` as
the values. A number detached from its baseline and output space cannot be
interpreted, and the report must trace every number to an artifact.

**D1.4 — Target space is the logit (margin), everywhere.**
Attributions are additive in the logit, not in the probability, and the
probability is an uncalibrated ranking score (class-weighted training), so
"probability points" would mislead.

**D1.5 — Raw-feature order: the 8 categoricals, then `household_size`, `age_of_respondent`.**
`CATEGORICAL_COLUMNS + NUMERIC_COLUMNS` from `preprocessing/pipeline_factory.py`,
the same order `scripts/train_production_model.py` builds raw records in.

**D1.6 — Grouping by summation preserves efficiency but is not assumed to equal raw-feature Shapley values.**
Summing one-hot columns or embedding dimensions per raw feature keeps the
decomposition additive, so efficiency/completeness carries over exactly
(tested). It does *not* follow that grouped 40-column TreeSHAP equals the
Shapley values of the 10-player game on raw features: the 40-player game
averages over coalitions that mix one-hot columns of the same categorical
between the instance and the background (two levels hot, or none), which the
10-player game never evaluates. The two coincide only if the model has no
interactions between columns of one categorical. Phase 3's oracle is defined
on the raw features, so it measures this gap; whatever it finds is reported
as found.

**D1.7 — Importance is compared as share of each method's own total.**
`importance_share` rescales mean |attribution| to sum to 1. Methods measured
against different references live on different scales, so raw magnitudes are
not comparable across methods; shares and ranks are.

**D1.8 — Efficiency tolerance has no default.**
`check_efficiency(..., atol=...)` requires the caller to state the tolerance.
What counts as "equal" differs by method (float precision for exact Shapley,
regression error for KernelSHAP, step count for Integrated Gradients), and a
silent default would hide which claim is being made.

**D1.9 — Config defaults (`config/config.yaml`, `xai:`).**
`background_size: 200` (top of the 100–200 range: 4 countries × 2 classes
gives 8 strata, and Uganda-positive is the smallest at 181 training rows, so
a larger background represents it with more than a handful of rows while
exact Shapley stays tractable at 1,024 coalitions × 200 rows per instance).
`explanation_sample_size: 1000` plus all Uganda rows. `oracle_instances: 50`.
`ig_n_steps: 100` is a starting value only; Phase 4 raises it until the
reported convergence delta is small. `bootstrap_reps: 1000`.

**D1.10 — `reports/figures/xai/*.png` is tracked by git.**
The existing ignore rule covers only `reports/figures/*.png` (top level).
The XAI report embeds its figures, so they are committed with it.
`models/xai/` (retrained explanation-only weights) is gitignored.
