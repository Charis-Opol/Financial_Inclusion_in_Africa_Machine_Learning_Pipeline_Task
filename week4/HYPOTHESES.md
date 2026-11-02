# Hypotheses and falsification rules

**Status: pre-registration, written before any Week 4 experiment ran.** The commit timestamp of this file is the evidence that it came first.

If a hypothesis, criterion or definition here changes after any result has been seen, the change is logged in `RESEARCH_LOG.md` as post-hoc and stated in the paper. The earlier version stays in git history.

Each RQ gives the question and falsification rule **unchanged from the brief**, then an operational definition. The brief's wording decides the verdict. The operational definitions turn terms such as "trivially" and "consistently" into rules that can be computed. They are my proposals, and they become binding when this file is committed.

Numbers below are design parameters, not results, except where tagged `[VERIFY]`.

## Definitions used by every RQ

**Data.** Labelled rows in `data/raw/Train_v2.csv` only. `Test_v2.csv` has no labels and is not used for evaluation.

**Evaluation design.** Nested CV with 5 outer folds and 3 inner folds, stratified by country and label. Preprocessing is fitted inside each training fold. All tuning happens in the inner loop.

**Seeds.** Seeds 0, 1, 2, 3 and 4 control the outer split, the inner splits and model initialisation. Every model sees the same splits for a given seed, so results are paired by seed. E1 uses the baseline's seed 42 separately, because its job is reproduction.

**Primary metric.** PR-AUC (average precision), computed on the out-of-fold (OOF) predictions of one seed, concatenated across its outer folds, then averaged over seeds. Fold-mean PR-AUC is also reported, for comparison with the baseline reports.

| Term | Definition |
|---|---|
| Pooled, P(m) | PR-AUC of model m over all rows |
| Per-country, P_c(m) | PR-AUC of model m over rows of country c |
| **Worst-country, W(m)** | min over c of P_c(m). The country is chosen separately for each model and each bootstrap replicate, so it floats. |
| Gap, G(m) | P(m) − W(m) |
| Excess, Ex_c(m) | P_c(m) − prevalence of c |
| Lift, L_c(m) | P_c(m) / prevalence of c |

**Why the worst country floats.** The framework expects Uganda to be the worst-off country. The existing reports do not support assuming that: Rwanda has the lowest raw PR-AUC in every per-country table, 0.473 for tuned XGBoost against 0.553 for Uganda `[VERIFY: reports/ablation_study_report.md §5]`, and also the lowest in the untuned logistic regression, XGBoost and MLP `[VERIFY: reports/baseline_results.md]`. So the primary quantity is W(m), and these are reported alongside it for every model:

- Uganda, as the named focus of the framework (lowest prevalence, smallest sample).
- Rwanda, as the expected raw minimum.
- W computed on Excess and on Lift, because raw PR-AUC depends on prevalence and the ranking of countries can change with the scale.

The minimum of noisy estimates is biased downward and is more likely to land on a small group. The paper states this when interpreting W(m).

**Uncertainty.**

- Bootstrap: 2,000 replicates, resampling rows within country-by-label strata, with the same resampled rows for every model and seed. A metric's value in a replicate is its mean over seeds. Intervals are 95% percentile intervals.
- **Marginal CI:** the interval for one model's metric.
- **Paired CI:** the interval for the difference between two models, computed within each replicate.
- Seed noise: the SD across the 5 seeds of the per-seed paired difference.
- The bootstrap captures uncertainty from the evaluation sample. Refitting variation is captured separately, by seeds.

**Smallest effect of interest: δ = 0.01 PR-AUC.** This is our own judgement of the smallest change worth reporting as meaningful. No paper in the folder supplies it.

**Multiple comparisons.** Holm correction within a family: all hybrid variants in RQ4, and all pairwise method comparisons within each held-out country in RQ5.

**Conditions that would change the reading.**

- The survey year is fully confounded with country: Kenya 2018, Rwanda 2016, Tanzania 2017, Uganda 2018 `[VERIFY: ad hoc count of Train_v2.csv; Phase 1 writes this to week4/results]`. A "country" effect cannot be separated from a year or survey-round effect.
- The `country` feature is used in within-country CV, as in the baseline, and dropped for every model in leave-one-country-out (LOCO).
- `bank_account` is not the same quantity as account ownership in `demirguckunt2022findex`, whose definition includes mobile money (PDF p.33).

---

## RQ1 Accuracy

**Brief.** Does TabPFN v2, out of the box, match or beat tuned XGBoost and the PyTorch MLP on pooled PR-AUC under the nested, country-stratified CV? *Falsified if* TabPFN is clearly below XGBoost with non-overlapping confidence intervals. A tie inside the noise counts as "matches".

**Experiment.** E2.

**Operational definition.** Compare marginal 95% CIs of P(TabPFN) and P(XGBoost).

| Result | Verdict |
|---|---|
| XGBoost CI entirely above TabPFN CI | **Falsified** |
| CIs overlap | Matches |
| TabPFN CI entirely above XGBoost CI | Beats |

- The MLP comparison uses the same rule and is reported, but only XGBoost decides falsification, as in the brief.
- Non-overlap is a conservative rule. Paired CIs are reported as well. If a paired CI excludes 0 while marginal CIs overlap, the verdict stays "matches" and the paper reports the paired difference and whether it reaches δ.
- "Out of the box" means default settings with no tuning, for the TabPFN v2 weights specifically. If the training folds exceed its supported size, the handling decided in Phase 1 is applied and is reported as a condition of this result. It is fixed before E2 runs.

## RQ2 Worst-off country

**Brief.** How much worse is each model in the lowest-performing country than on pooled data, and does TabPFN shrink or widen that gap compared with XGBoost? *Falsified if* TabPFN's gap is indistinguishable from XGBoost's once uncertainty is included.

**Experiment.** E2.

**Operational definition.** Compute G(m) for every model, and the paired CI of G(TabPFN) − G(XGBoost).

- **Falsified** if the paired CI includes 0.
- If it excludes 0, the sign gives the direction: negative means TabPFN shrinks the gap, positive means it widens it.
- A narrower gap reached by lowering pooled PR-AUC is not an improvement for the worst-off group. W(TabPFN) − W(XGBoost) is therefore always reported beside the gap.

## RQ3 Group DRO

**Brief.** Training the MLP with Group DRO (country as group), does worst-country PR-AUC improve, at what pooled cost, and does it beat simple inverse-frequency reweighting? *Falsified if* the worst-country gain is within seed-to-seed noise, exceeds the pooled cost only trivially, or is matched by reweighting.

**Experiment.** E3.

**Operational definition.** Let gain = W(DRO) − W(ERM), cost = P(ERM) − P(DRO), and rw = the inverse-frequency-reweighted MLP. **Any one** of these falsifies the hypothesis:

1. **Within noise:** the paired CI of the gain includes 0, or |gain| is no larger than the seed noise of the gain.
2. **Trivial against cost:** gain − max(cost, 0) < δ.
3. **Matched by reweighting:** the paired CI of W(DRO) − W(rw) includes 0 or lies below 0.

Further rules:

- **The same selection criterion for all three MLP arms.** ERM, reweighted and Group DRO MLPs are all selected on inner-loop worst-country PR-AUC, so the comparison does not confound the objective with the selection rule. As a sensitivity check, ERM selected on pooled inner PR-AUC is also reported.
- Step size and the optional C/√n_g adjustment are tuned in the inner loop for the primary arm. Every ablation setting is reported.
- Reweighted and ERM XGBoost are reported as comparators and do not decide falsification.
- Class-imbalance handling is identical across MLP arms. The exact form is fixed in Phase 3 and reviewed at Checkpoint 3.

**Stated in advance:** results in `koh2021wilds` (Tables 2 and 15; FMoW, p.97-99) and `stahlschmidt2024domain` give reason to expect Group DRO may not beat ERM or reweighting here. This does not change the rule.

## RQ4 Hybrid

**Brief.** Does combining TabPFN with group-robust training beat either approach alone on worst-country PR-AUC? *Falsified if* no hybrid variant beats the better of (best TabPFN-only, best DRO-only) on worst-country PR-AUC beyond confidence intervals.

**Experiments.** E5a, E5b, and E5c if approved and feasible.

**Operational definition.** For each hybrid H, compute the paired interval of W(H) − max(W(TabPFN default), W(DRO MLP)), taking the max within each bootstrap replicate.

- A hybrid passes if the lower bound, after Holm correction across all hybrid variants, is above 0. **Falsified** if none passes.
- Taking the better comparator within each replicate works against the hybrids, so choosing the comparator cannot favour them.
- Counted as hybrids: H1 (group-balanced context), H2 with a reweighted head, H2 with a Group DRO head, and H3 if run.
- H2 with an ERM head is a control for the representation alone. It is reported but is not counted as a hybrid, because its training is not group-robust.
- **Every variant run is reported**, including failures. Variants are not added after seeing results without a post-hoc log entry.

## RQ5 Shift across countries

**Brief.** Training on three countries and testing on the fourth, which approach holds up best: TabPFN, XGBoost, the Group DRO MLP, or a hybrid? *Falsified if* no method is consistently better across held-out countries.

**Experiment.** E4, with four held-out countries.

**Operational definition.**

- The `country` feature is dropped for every model. Preprocessing is fitted on the three training countries. Group DRO uses those three as its groups. Tuning uses inner folds stratified by country and label within the three training countries.
- PR-AUC is compared only **within** a held-out country, because prevalence differs between countries.
- A method is **consistently better** if both hold:
  - in at least 3 of 4 held-out countries it has the highest PR-AUC, with Holm-corrected paired CIs excluding 0 against every other method;
  - in no held-out country is it beaten by another method with a Holm-corrected paired CI excluding 0.
- **Falsified** if no method meets this.
- The drop from E2 to E4 in the same country is also reported. It is not read as pure shift: training size differs and a country may be intrinsically harder, as `koh2021wilds` warns (p.17).
- A hybrid enters RQ5 only if run in E5. With E4 priority A and E5 priority B, E4 is extended to hybrids after E5. The paper says if that did not happen.

## RQ6 Explanations

**Brief.** Do SHAP-measured drivers differ by country, beyond what resampling noise alone produces, and does Group DRO change which features matter for the worst-off group? *Falsified if* cross-country ranking differences are no larger than differences between random subsamples of one country.

**Experiment.** E6.

**Operational definition.**

- Features are ranked by mean absolute SHAP over **out-of-fold** rows, on the 9 features other than `country`. Attribution to `country` is constant within one country, so it is excluded from the ranking. Rankings are compared, not magnitudes.
- **Size matching.** Every ranking uses the same sample size, half the smallest country: about 1,050 rows `[VERIFY: 2,101 Uganda rows, ad hoc count]`. Otherwise the noise baseline and the cross-country comparison would not be comparable.
- **Cross-country difference.** For each of the 6 country pairs, draw one subsample per country and compute 1 − Spearman ρ. Repeat 200 times. Kendall τ and top-3 overlap are also reported.
- **Noise baseline.** For each country, draw two disjoint subsamples of that size and compute the same quantity. Repeat 200 times.
- Statistic: mean over 6 pairs of the cross-country difference, minus mean over 4 countries of the noise baseline, with a 95% percentile interval over the repeats. **Falsified** if the interval includes 0 or lies below 0.
- The primary model is XGBoost with TreeExplainer. The MLP and Group DRO MLP use permutation or kernel SHAP with a documented background set. TabPFN is included if feasible.
- **Group DRO part.** In the worst-off country for the ERM MLP, compare DRO and ERM rankings against the same noise baseline, using size-matched subsamples and the same seeds. A difference counts only if it exceeds that noise.
- SHAP describes what a model relies on. It does not establish why people have bank accounts, and the paper will not make causal claims from it.

---

## Reporting commitments

- No tuning or new variants to rescue a hypothesis. Anything added after seeing results is logged as post-hoc and labelled in the paper.
- Every number in the paper traces to a file in `week4/results/`.
- The paper states which experiments were and were not completed.
