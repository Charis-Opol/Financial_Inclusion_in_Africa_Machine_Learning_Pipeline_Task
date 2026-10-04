# XAI decisions log

Every non-obvious choice in the Week 3 explainability work, with the reason.
Entries are appended as each phase lands. Attributions throughout describe
what the **model** learned to rely on; none of them is a causal claim about
what drives bank account ownership.

New to this code? Read `src/fin_inclusion/interpretability/README.md` first:
it explains how the pieces fit together and defines the terms used below.

## At a glance

Each line is the decision in plain words; the full reasoning is under the
same ID further down.

| ID | Decision |
| --- | --- |
| D0.1 | The MLP was never saved, so it is retrained. All five folds had picked the same settings, so there was nothing to choose. |
| D0.2 | The "large household" flag counts as part of `household_size`, not as an 11th feature. |
| D0.3 | "Unknown" is a value neither model ever saw in training, so it is a weak reference point. It is still run, next to a second one. |
| D0.4 | The older SHAP code is kept but not used: it works differently and its numbers are not comparable. |
| D0.5 | The 0.4551 cut-off belongs to XGBoost only. |
| D1.1 | The XAI work has its own Python environment, `.venv-xai`. |
| D1.2 | New code goes in the existing `interpretability/` folder. |
| D1.3 | Every saved explanation carries its model, method and reference with it. |
| D1.4 | Explanations are in logits, not probabilities. |
| D1.5 | Feature order: the 8 categorical fields, then household size, then age. |
| D1.6 | Adding encoded columns together is valid, but may not exactly equal Shapley values computed on the 10 fields directly. The gap is measured in Phase 3. |
| D1.7 | Methods are compared by each feature's share of importance, not by raw size. |
| D1.8 | Every "does it add up?" check must state how close counts as equal. |
| D1.9 | Sizes: 200 reference rows, 1,000 explained rows plus all of Uganda, 50 rows checked exactly, 1,000 bootstrap resamples. |
| D1.10 | XAI figures are committed; the retrained MLP file is not. |
| D2.1 | Both models were re-checked on held-out folds and perform as reported (MLP 0.584 exactly; XGBoost 0.588 vs 0.590). |
| D2.2 | The MLP is trained on all labelled rows, the same as the shipped XGBoost. |
| D2.3 | So every explained row is one both models trained on. No performance claims are made from them. |
| D2.4 | The adapters reject a category the model never saw, instead of quietly treating it as "unknown". |
| D2.5 | Reference set: 200 rows mirroring the training mix. Only 18 are from Uganda, so Uganda results are compared with a typical *pooled* respondent. |
| D2.6 | Pooled results use the 1,000-row sample; Uganda results use all 2,101 Uganda rows. |
| D2.7 | The MLP's "unknown" embeddings turned out to be almost all zeros. |
| D2.8 | Cut-offs: XGBoost 0.4551, MLP 0.6824. |

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
gives 8 strata, and the larger the background the less the smallest strata
are rounded away, while exact Shapley stays tractable at 1,024 coalitions ×
200 rows per instance; see D2.5 for what the smallest stratum actually got).
`explanation_sample_size: 1000` plus all Uganda rows. `oracle_instances: 50`.
`ig_n_steps: 100` is a starting value only; Phase 4 raises it until the
reported convergence delta is small. `bootstrap_reps: 1000`.

**D1.10 — `reports/figures/xai/*.png` is tracked by git.**
The existing ignore rule covers only `reports/figures/*.png` (top level).
The XAI report embeds its figures, so they are committed with it.
`models/xai/` (retrained explanation-only weights) is gitignored.

## Phase 2 — The models to explain

All numbers below are from `reports/xai/model_verification.json` and
`reports/xai/sample_sets.json`, written by `scripts/run_xai_models.py`.

**D2.1 — The verification gate is the report's outer 5-fold CV with final hyperparameters fixed.**
Neither final model has a labelled hold-out of its own: the shipped XGBoost
was fit on all 23,524 labelled rows and `Test_v2.csv` has no labels. The
gate re-runs the same folds (seed 42, country × target stratified) and
requires the mean PR-AUC to be within ±0.02 of the reported nested-CV mean.

- MLP: 0.5837 vs 0.5837 reported — every fold reproduces the reported value
  exactly (max per-fold difference 0.0), because all five outer folds had
  selected the same hyperparameters and the procedure is deterministic.
- XGBoost: 0.5876 vs 0.5901 reported (max per-fold difference 0.0070). This
  is not a like-for-like reproduction and is not presented as one: the
  shipped `production_xgb_params` were tuned on the full training set (so
  this estimate is mildly optimistic), whereas the reported figure used
  different, per-fold tuned hyperparameters.
- A full-data refit with the production params scores within 7.4e-8 of the
  booster on disk, so the CV evaluates the same procedure that produced the
  shipped artifact.
- Uganda-only held-out PR-AUC: XGBoost 0.544, MLP 0.542 (five folds each,
  wide spread: 0.476–0.640 and 0.469–0.639).

**D2.2 — The MLP is refit on all labelled rows, like the shipped XGBoost.**
Same hyperparameters, `ClassWeightStrategy`, seed 42, 20 epochs (Phase 4's
final-fit epoch count). Experiment B compares what the two candidate models
rely on, so they should have seen the same data. Saved to `models/xai/`
(gitignored) as `mlp.pt` + `mlp_metadata.json`; the metadata carries the
weights' SHA-256, which is checked before the checkpoint is unpickled.

**D2.3 — Consequence: every explained row is a training row of both models.**
There is no labelled data the final models have not seen. In-sample PR-AUC
is 0.642 (XGBoost) and 0.609 (MLP) against ~0.59/0.58 held out, so TP/FN/FP/
TN labels assigned from these models' own predictions are somewhat
optimistic. Explanations are of the models' behaviour on these rows, which
is what an explanation is; no performance claim is made from them.

**D2.4 — Preprocessing state is stored as JSON, and adapters refuse unseen categories.**
`MLPRawModel` keeps the category → index maps and scaler mean/scale as plain
dicts instead of a pickled sklearn pipeline, and both adapters' encodings
are checked bit-for-bit against the training pipelines on all 23,524 rows
before anything is saved. Where the training pipeline routes an unseen
category to "unknown" with a warning, the adapters raise: an explanation of
a silently mis-encoded row would explain the wrong input.

**D2.5 — Background set: 200 rows, proportional within country × target, seed 42.**
Proportional (not equal) allocation, so the base value approximates the
model's mean output over the training population; equal allocation would
over-represent account holders and small countries and shift the reference.
Resulting composition: Kenya 52, Rwanda 74, Tanzania 56, Uganda 18 (16 No,
2 Yes); positive rate 14.0% vs 14.08% in training. Mean logit over the
background vs over all training rows: XGBoost −1.247 vs −1.210, MLP −1.251
vs −1.162, i.e. the base value sits 0.04 (XGBoost) and 0.09 (MLP) logits
below the population mean. Uganda-only results use this same pooled
background: they answer "why does this Ugandan respondent score differently
from a typical respondent in the pool", not from a typical Ugandan. A
Uganda-only reference would have to be built from many more than 18 rows.

**D2.6 — Explanation set: 1,000 stratified rows (pooled) + all 2,101 Uganda rows = 3,012.**
The pooled sample is disjoint from the background and is the only set used
for pooled statistics (it contains 89 Uganda rows, their population share);
adding all Uganda rows to pooled statistics would over-weight Uganda about
sixfold. Uganda-only statistics use all 2,101 Uganda rows, 18 of which are
also background rows.

**D2.7 — Measured: the MLP's "unknown" embedding rows are near-zero vectors, not random ones.**
Follow-up to D0.3. Index 0 receives no data gradient, but Adam applies the
(tiny) L2 weight-decay gradient at a normalised step size, which shrinks the
unused rows toward zero over training. Their norms are 0.00006–0.53 against
1.5–2.8 for trained rows. The "unknown" IG baseline is therefore
approximately a zero-embedding baseline: a well-defined point, but one that
corresponds to no respondent the model was trained on.

**D2.8 — Decision thresholds are per model.**
XGBoost 0.4551 (shipped), MLP 0.6824 (median of its per-fold F1-optimal
thresholds). At those thresholds 20.6% and 14.9% of training rows are
predicted "Yes".
