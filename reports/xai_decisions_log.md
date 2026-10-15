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
| D3.1 | Every SHAP method uses the same 200 reference rows and replaces a missing field independently of the others ("interventional"). |
| D3.2 | Result: grouped TreeSHAP is close to the exact values but not identical (average gap 2.1%). KernelSHAP with all coalitions is identical. |
| D3.3 | So exact KernelSHAP is also run on XGBoost for every row, to have ground truth everywhere. |
| D3.4 | The MLP's main SHAP result is exact KernelSHAP. GradientSHAP is kept as a second, approximate one whose contributions do not add up exactly. |
| D3.5 | TreeSHAP was silently using only 100 of the 200 reference rows; fixed by passing the full set explicitly. |
| D3.6 | The five local examples are picked by written rules applied to XGBoost, and the same five rows are used for both models. |
| D3.7 | Dependence plots show the top three features by importance, whichever they turn out to be. |
| D3.8 | Categories with no natural order are drawn grey in beeswarm plots instead of shaded. |
| D4.1 | Captum methods compare each respondent with one reference point. Two are used: "unknown" (never-seen categories) and "modal" (most common answers). Neither is a typical respondent. |
| D4.2 | Integrated Gradients moves through the MLP's embedding vectors, since category numbers cannot be moved smoothly. |
| D4.3 | Integration steps: 256, chosen by a rule set in advance. A few rows (0.4%) still miss the target slightly; reported, not retuned. |
| D4.4 | Feature Ablation removes one field at a time, on both models. Its results do not add up, by design. |
| D4.5 | Result: switching the reference barely changes which features matter overall, but changes the explanation for an individual respondent a lot. |
| D4.6 | Hidden units: none unused, computation fairly spread out. They describe how the network computes, not respondents. |
| D4.7 | Dropout is off and every Captum method is deterministic. |

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

## Phase 3 — SHAP (Task 1)

All numbers below are from `reports/xai/shap_summary.json`, written by
`scripts/run_xai_shap.py`.

**D3.1 — One reference and one value function for every SHAP method.**
Reference: the 200 background rows (D2.5); base value −1.247 logits for
XGBoost and −1.251 for the MLP. Value function: interventional — a feature
that is "absent" takes its value from a background row, independently of the
other features. This breaks correlations between features, so the model is
asked about some hybrid respondents that do not exist (for example a
background row's marital status with the instance's relationship to the
household head). It was chosen because it needs no model of the joint
feature distribution and is what TreeSHAP, KernelSHAP and the brute-force
oracle can all compute identically, which is what makes them comparable.

**D3.2 — The oracle result: grouped TreeSHAP is close to, but not equal to, exact raw-feature Shapley values.**
On the 50 oracle rows (seeded draw from the pooled sample), against
brute-force Shapley over all 1,024 coalitions of the 10 raw features:

- KernelSHAP with all 1,022 non-trivial coalitions enumerated matches the
  oracle to floating-point precision (max |difference| 2.4e-15 for XGBoost,
  4.8e-8 for the MLP, whose arithmetic is float32).
- Grouped TreeSHAP differs from it: mean |difference| 0.0042 logits, max
  0.043, which is 2.1% of the mean |attribution| (0.20). The sign agrees for
  99.2% of the 500 values and the top feature agrees on all 50 rows. Base
  values are identical.
- The gap is largest for `education_level` (0.0105) and `country` (0.0075),
  the multi-level categoricals the model leans on most, and smallest for
  `household_size` (0.0007).

This is the outcome D1.6 anticipated, and it is a difference between two
games, not an error in either computation: a unit test confirms the two
agree to 1e-4 when the trees have depth 1 (no interactions possible).
"Grouped TreeSHAP matches the oracle numerically" is therefore true only
approximately for the shipped depth-8 model, and is reported that way.

**D3.3 — Exact KernelSHAP is run on XGBoost for all 3,012 rows as well.**
Because TreeSHAP is only approximately the raw-feature Shapley value, and
KernelSHAP with full enumeration is exactly it (D3.2), the exact values are
computed for every explained row of both models. TreeSHAP remains the
headline SHAP result for XGBoost (it is the method the assignment names);
the exact values are the ground truth later phases compare against. At the
global level the two are almost indistinguishable: the pooled importance
ranking is identical and shares differ by at most 0.003.

**D3.4 — MLP: exact KernelSHAP is the SHAP result; GradientSHAP is reported as an approximation.**
`shap.GradientExplainer` works on the MLP after the embedding lookup (51
inputs, summed to raw features). Two separate things were measured on the
oracle rows as its sample count rises from 200 to 1,000 to 5,000:

- its efficiency error falls (mean 0.145 → 0.076 → 0.027 logits): that part
  is sampling noise;
- its distance from the oracle barely moves (mean 0.050 → 0.040 → 0.036):
  that part is not noise. Expected gradients integrates along straight paths
  in embedding space and is a different quantity from the Shapley value.

At the default 200 samples used for the full run, attributions fail the
0.05-logit efficiency tolerance on 2,360 of 3,012 rows (max error 0.66).
This is recorded as a failed check, not hidden; GradientSHAP is not used for
any conclusion that needs attributions to add up.

**D3.5 — TreeSHAP's background is passed as an explicit masker.**
Given a plain array, `shap.TreeExplainer` silently subsamples the background
to 100 rows. A first smoke test showed its base value 0.015 logits away from
the other methods for that reason. The background is now wrapped in
`shap.maskers.Independent(..., max_samples=200)`; a test with a 150-row
background guards it.

**D3.6 — Local examples are chosen by rule, on XGBoost, and reused for the MLP.**
Rules (`sampling.LOCAL_EXAMPLE_RULES`): highest-scoring true positive,
lowest-scoring true negative, lowest-scoring false negative, highest-scoring
false positive (all within the pooled sample, at XGBoost's 0.4551
threshold), and the Uganda row whose score is closest to Uganda's median
score. Ties go to the lowest row position. The rules use XGBoost because it
is the shipped model; the same five rows are explained for the MLP so the
two can be compared respondent by respondent. All five happen to get the
same predicted label from both models. Rows: 946, 17886, 18896, 16210,
22829.

**D3.7 — Dependence plots: the top three features by pooled mean |TreeSHAP|.**
Chosen by rank rather than by expectation. They turned out to be
`cellphone_access`, `education_level` and `country` — the three the project
plan expected. KernelSHAP's `nsamples=20` is kept in the convergence table
although its result is invalid (singular regression): with 10 features, 20
coalitions do not determine the fit.

**D3.8 — Beeswarm colouring.**
Numeric and two-level features are shaded by value on a single-hue ramp;
multi-level categoricals are grey. The usual SHAP beeswarm colours every
feature by its numeric code, which for `country` or `job_type` would display
an ordering that does not exist.

**Findings worth carrying forward (pooled sample, n = 1,000).**

- XGBoost importance shares: `cellphone_access` 0.27, `education_level`
  0.19, `country` 0.14, `age_of_respondent` 0.11, `job_type` 0.09.
- MLP importance shares: `cellphone_access` 0.22, `education_level` 0.17,
  `country` 0.17, `job_type` 0.13, `age_of_respondent` 0.10. Same top three;
  `job_type` and `age_of_respondent` swap places.
- Uganda only (n = 2,101): `country` rises for both models (XGBoost share
  0.14 → 0.19; MLP 0.17 → 0.22, where it becomes the top feature). This is
  partly built in: every Uganda row has the same country, which the pooled
  reference mostly does not, so its attribution never averages out.
- The MLP's attributions are larger in absolute terms (mean |attribution| of
  the top feature 0.77 vs 0.57 logits). Its logits are more spread out;
  shares, not magnitudes, are comparable across models (D1.7).

## Phase 4 — Captum (Task 2)

All numbers below are from `reports/xai/captum_summary.json`, written by
`scripts/run_xai_captum.py`.

**D4.1 — Two baselines, both run for every method; numerics at the training mean in both.**
- *Unknown*: every categorical set to a never-seen category (MLP: embedding
  index 0; XGBoost: an all-zero one-hot block — what each training pipeline
  does with an unseen value). Expressed as a special code, `UNKNOWN_CODE`,
  that real input can never produce.
- *Modal*: every categorical at its most frequent training level. This is a
  Rwandan, rural, female head of household, married, with a phone, primary
  education, self-employed, household size 3.80, age 38.81.

Neither is a neutral reference. Against a mean training logit of −1.21
(XGBoost) and −1.16 (MLP), the baselines score: unknown −0.69 (XGBoost) and
**+0.66 (MLP)**; modal −0.58 and −0.08. The MLP rates a respondent who is
"unknown" on every field above the average actual respondent, a
consequence of the near-zero unknown embeddings (D2.7) meeting the MLP's
biases, not of anything in the data. These differ from SHAP's reference (the
mean logit over 200 background rows, −1.25), so Captum and SHAP numbers
answer "compared with what?" differently; Phase 6 aligns references before
comparing methods.

**D4.2 — Integrated Gradients is LayerIntegratedGradients on the 8 embedding layers plus a named identity layer on the numeric inputs, integrated jointly.**
Category indices cannot be interpolated, so the straight-line path runs
between embedding vectors. The numeric inputs pass through an `nn.Identity`
layer added in a wrapper that shares the MLP's weights (verified to give
identical output), so all 11 inputs are interpolated together and
completeness holds over all of them. The 51 per-dimension attributions are
summed per raw feature. The outlier flag is interpolated from the
respondent's value to the baseline's (0) along with the rest of the numeric
inputs and is counted under `household_size`.

**D4.3 — Integration steps chosen by a rule fixed in advance: 256.**
Rule: the smallest step count on {32, 64, …, 1024} whose largest
|convergence delta| on the 50 oracle rows is ≤ 0.01 logits for both
baselines. Worst delta at 128 steps was 0.012 (unknown) / 0.022 (modal); at
256, 0.0064 / 0.0074. On the full 3,012 rows at 256 steps: unknown max
|delta| 0.0100 (every row within 0.01); modal max 0.0144, with 12 rows
(0.4%) above 0.01 and a mean of 0.0021. Reported as found; the step count
was not raised after the fact to make the full run pass.

**D4.4 — Feature Ablation removes one raw feature at a time, on both models.**
Through the raw-feature wrapper, so ablating `education_level` resets the
whole categorical and ablating `household_size` resets the outlier flag with
it. Each feature is removed *alone*, with all others at the respondent's
values — no averaging over coalitions — so the attributions are not meant to
add up. Measured: their sum misses f(x) − f(baseline) by 0.61–0.63 logits on
average for XGBoost and 0.81–1.16 for the MLP, i.e. both models are far from
additive in their raw features. This is the main value-function difference
from SHAP for Phase 6.

**D4.5 — Result: the baseline barely moves global rankings but changes local explanations a lot.**
Unknown vs modal baseline, pooled sample (n = 1,000):

| | Global Spearman [95% CI] | Same top feature per respondent | Per-respondent Spearman, median |
| --- | --- | --- | --- |
| IG, MLP | 0.77 [0.71, 0.78] | 49% | 0.27 |
| Ablation, MLP | 0.92 [0.89, 0.93] | 40% | 0.30 |
| Ablation, XGBoost | 0.90 [0.79, 0.94] | 47% | 0.18 |

For 73–82% of respondents the two baselines' feature orderings correlate
below 0.5. A local explanation from these methods is therefore mostly a
statement about the baseline. The global picture is steadier: the same
four features lead under both baselines for every method, in different
orders.

The clearest single case is `country` for Uganda: with the modal baseline it
takes 0.39 of IG importance (and 0.34 of MLP ablation, 0.29 of XGBoost
ablation); with the unknown baseline 0.25, 0.19 and 0.16. The modal baseline
is a Rwandan, so for a Ugandan respondent "country" measures Uganda versus
Rwanda; the unknown baseline measures Uganda versus no country at all.

**D4.6 — Layer attribution: hidden units, and embedding dimensions.**
LayerConductance (how much of f(x) − f(baseline) flows through each unit,
256 steps) and LayerGradientXActivation (activation × local gradient, no
baseline) on the ReLU outputs of both hidden layers, on the
embedded-input view of the network:

- No unit is dead on the explained rows. Layer 1: 63 units, the top 10
  carry 38–39% of mean |conductance|, the median unit is active for 24% of
  respondents. Layer 2: 103 units, top 10 carry 23–25%, median unit active
  for 57%.
- Conductance completeness holds per layer up to max |error| 0.013 (layer
  1) and 0.0005 (layer 2).
- The two methods rank units similarly (Spearman 0.76–0.92 in layer 1,
  0.93–0.97 in layer 2).
- Embedding dimensions: within each categorical, attribution is spread
  across the 6 dimensions (the largest dimension carries 26–33%), except
  `gender_of_respondent` (60%). No dimension corresponds to a nameable
  property.

Hidden units and embedding dimensions have no meaning of their own; these
results describe how the network computes, not anything about respondents.

**D4.7 — Determinism.**
The MLP is put in `eval()` (dropout off) by every explainer; Integrated
Gradients, Feature Ablation and conductance are deterministic given the
inputs (re-running gives identical numbers).
