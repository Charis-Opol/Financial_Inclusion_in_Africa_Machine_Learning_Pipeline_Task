# Status: Checkpoint 4, results reviewed and verdicts applied

Date: 2026-11-03. Results commit: `1e5f1d3` (run from `ddc0584` on the Alienware, RTX 4080 SUPER). Nothing committed or pushed since. The only code change is a fix to E6 (D4.2).

Every verdict below comes from the pre-registered rules in `HYPOTHESES.md`, applied to the tables in `results/metrics/` and `results/e6/`. No rule, arm or setting was changed after seeing results.

## The run

- **Complete.** All 32 arms finished every fold: 760 of 760 (`results/phase4_completion.md`).
- **Checks passed.** The preflight passed, all 41 tests passed, and the smoke run passed.
- **GPU against CPU agreement for TabPFN** (fp16 against float32, synthetic labels): maximum score difference 0.0038, mean 0.00056, AUC 0.91525 against 0.91526. That's well below anything that could change a verdict (D3.7).
- **Wall time: 3 h 48 min.** Checks took 43 min, the CPU lanes 1 h 46 min, the GPU lane 2 h 6 min, E6 50 min and the analysis 9 min. One TabPFN E2 fit took about 3 s on the GPU, against 786 s on the laptop CPU in Phase 1.
- **One problem, in E6:** the MLP refits did not reproduce their saved scores. See Problems.

## Verdicts

P is pooled PR-AUC, W is worst-country PR-AUC and G = P − W. Values are 5-seed means, with 95% bootstrap intervals in brackets.

| RQ | Verdict | Deciding numbers |
|---|---|---|
| RQ1 Accuracy | **Matches** (not falsified) | P: TabPFN 0.590 [0.575, 0.606], XGBoost 0.588 [0.572, 0.603]. The intervals overlap. Paired difference +0.003 [−0.000, +0.006], below δ = 0.01. |
| RQ2 Worst-off gap | **Falsified** | G(TabPFN) − G(XGBoost) = +0.002 [−0.002, +0.008], which includes 0. W difference +0.001 [−0.006, +0.006]. |
| RQ3 Group DRO | **Falsified** on criteria 2 and 3 | Gain W(DRO) − W(ERM) = **−0.017** [−0.036, −0.011]: Group DRO *lowers* worst-country PR-AUC. Pooled cost 0.040. Against reweighting −0.016 [−0.033, −0.010]. |
| RQ4 Hybrids | **Falsified** | Every hybrid is below the better parent: H1 −0.003, H2-reweighted −0.009, H2-DRO −0.013. Holm-adjusted p = 1.0 for all three. |
| RQ5 Country shift | **Falsified** | No method is consistently better. Every method is clearly beaten in at least one held-out country. The only clear win is H1 in Tanzania. |
| RQ6 Explanations | **Not falsified**, primary model. The Group DRO part shows no difference. Both MLP parts are **provisional** (D4.2). | XGBoost: cross-country disagreement exceeds within-country noise by 0.038 [0.033, 0.042] (1 − Spearman ρ). Group DRO against ERM in Rwanda: +0.007 [−0.017, +0.025], which includes 0. |

### RQ1 and RQ2: TabPFN

- **Tied with tuned XGBoost** on pooled data, worst-country (Rwanda) and Uganda.
- **Ahead of the ERM MLP** on P (+0.009 [0.005, 0.012]) and W (+0.008 [0.002, 0.016]). The paired intervals exclude 0, but both differences are below δ, and the marginal intervals overlap. By the rule, that's still "matches".
- **Against XGBoost with tuned `scale_pos_weight`** (`xgb_spw`, the baseline's arm), TabPFN is clearly ahead: +0.026 on P. Most of that is a scoring artefact, not a weaker model. See Other findings, point 3.
- **The gap is the same for every model**, about 0.12. TabPFN neither narrows nor widens it.

### RQ3: Group DRO

All three falsification criteria were checked:

1. **Within noise:** not met. The gain's interval excludes 0, and |gain| 0.017 is above its seed SD of 0.013. But the gain is negative.
2. **Trivial against cost:** met. gain − cost = −0.017 − 0.040 = −0.057, below δ.
3. **Matched by reweighting:** met. DRO lies entirely below reweighting.

**Why, from the logged group weights** (`results/metrics/q_history_*.csv`, final epoch, mean over folds and seeds):

| Arm | Kenya | Rwanda | Tanzania | Uganda |
|---|---|---|---|---|
| `mlp_dro` (pooled class weights) | **0.91** | 0.06 | 0.02 | 0.02 |
| `mlp_dro_within_cw` (D3.3) | 0.33 | **0.65** | 0.00 | 0.02 |
| `h2_dro` | **0.88** | 0.07 | 0.02 | 0.03 |

- **With pooled class weights, Group DRO pushes almost all its weight onto Kenya.** Kenya has the highest training loss because it has the highest prevalence (25%). The country with the worst PR-AUC is Rwanda. Group DRO maximises the worst group *loss*, which is not the worst group *PR-AUC*. This is the risk D3.3 anticipated.
- **The D3.3 arm targets Rwanda as intended** (0.65 of the weight), yet Rwanda's PR-AUC doesn't move: −0.001 [−0.041, +0.003] against its ERM twin, at a pooled cost of 0.051. So the failure isn't only the wrong target.
- **Ablations:** larger step sizes are monotonically worse. W is 0.454 at step 0, 0.417 at 0.1 and 0.401 at 1.0. The C/√n adjustment has no effect (W 0.442 to 0.446 for every C).
- **E7, country-by-label groups:** better than country groups on W (+0.015 [0.003, 0.027]), but still not above ERM (0.458 against 0.461), and it costs 0.067 pooled.
- **The sensitivity check gives the same answer:** ERM selected on pooled inner PR-AUC instead of worst-country.

This is the outcome `HYPOTHESES.md` said in advance was plausible (`koh2021wilds`, `stahlschmidt2024domain`).

### RQ4: hybrids

- **H1 (balanced context)** is closest, at −0.003, a cost inside δ.
- **H2 heads** sit below TabPFN alone. The reweighted and DRO heads are no better than the ERM-head control (−0.009).
- **H2-DRO** inherits Group DRO's tilt towards Kenya (0.88).

### RQ5: leave-one-country-out

Highest PR-AUC in each held-out country, with clear wins after Holm correction:

| Held out | Best (P) | Clear wins |
|---|---|---|
| Kenya | Group DRO MLP 0.622 | none |
| Rwanda | Group DRO MLP 0.415 | over 5 of 6 methods; H2-DRO not separated (Holm p = 0.14) |
| Tanzania | **H1 0.466** | over all 6 methods |
| Uganda | TabPFN 0.513 | none; Uganda's intervals are about ±0.07 wide |

- **Group DRO does best under shift in Kenya and Rwanda,** although it is the worst model within countries. This is a pattern seen after the fact, not a pre-registered claim. Part of it is mechanical: its within-country scores are lower, so it has less to lose.
- **Tanzania drops the most for every model:** 0.15 to 0.21 PR-AUC from within-country CV to held out. That fits the Phase 1 anomaly (Tanzania's `location_type` runs the other way), but it has not been tested.

### RQ6: SHAP

- **Rankings differ by country beyond noise,** for XGBoost and the ERM MLP alike. The effect is small. `cellphone_access` and `education_level` are the top two in every country for every model (`results/e6/mean_abs_shap_by_country.csv`).
- **For the Group DRO MLP the interval's lower bound is 0,** which the rule treats as "includes 0".
- **In Rwanda, Group DRO does not change which features matter** beyond noise.

## Other findings for the paper

1. **Rwanda is the worst-off country, not Uganda.** It is the lowest for every primary model in every seed. Uganda is the lowest in only one seed of one ablation (`worst_country.csv`). This confirms D0.2: fixing the analysis on Uganda would have measured the wrong group.
2. **The scale doesn't change the answer here.** W on Excess picks the same ordering of models as raw W.
3. **Pooled and fold-mean PR-AUC disagree for `xgb_spw`:** 0.565 pooled against 0.589 fold-mean. Tuning `scale_pos_weight` separately in each fold changes the score scale from fold to fold, which hurts concatenated out-of-fold scores but not per-fold scores. The baseline reported fold-means. The paper should report both and explain.
4. **H1 is a cheap hybrid with a clear LOCO win in Tanzania.** The win wasn't pre-registered as a claim, but every LOCO hybrid ran before any result was seen (D3.5).

## Problems

**E6's MLP refits did not reproduce their saved scores (D4.2).**

- **What happened.** Maximum score difference was 0.02 to 0.17 per fold, where 1e-7 was expected. XGBoost reproduced to 3e-8.
- **Cause, confirmed here.** The CPU lanes trained with 5 threads, and E6 refitted with 32 (`run_everything.sh:92`). At a fixed thread count training is deterministic: two 5-thread fits on this laptop differed by 0.0. Changing 5 threads to 16 moved scores by up to 0.049 (mean 0.009). Hardware matters too, so the rerun has to happen on the Alienware.
- **Effect.** The MLP SHAP values explain a sibling model: same data, hyperparameters and seed, but a different floating-point path. The XGBoost result, which is RQ6's primary model, is unaffected. The MLP rows of RQ6 are provisional until the rerun.
- **Fixed** in `src/e6_shap.py`: each fold now refits with its recorded `torch_threads`, and the check records threads, host and mean difference. Smoke test: refits match to 3e-8.
- **Not rerun, by your decision.** The paper reports the MLP rows of RQ6 with this caveat. A later rerun would take about 45 min on the Alienware: delete `results/e6/attributions_mlp_*.csv.gz`, then run `src/e6_shap.py`. XGBoost attributions are cached, and its numbers would not change.

**Still open from earlier checkpoints:**

- The Sagawa et al. (2020) PDF is still missing (D3.2).
- D0.1 to D0.8 and D1.2 are still marked Confirm.

## Review checklist

1. The verdict table and the RQ3 group-weight table above.
2. `results/metrics/summary.md`, `nested_paired.csv`, `loco_paired.csv`.
3. Figures: `fig_rq3_sweeps.png`, `fig_dro_group_weights.png`, `fig_rq5_loco.png`.
4. `DECISIONS.md` D4.1 and D4.2.

## Phase 5: paper drafted

`paper/paper.md` is the full draft: abstract, introduction, related work, data, methods, results by RQ, discussion, limitations and ethics, conclusion, the completed and not-completed table, reproducibility, and references. Every number gives its source file. Only papers in `applied_research/literature/` are cited, and Group DRO's update rule is presented as our own statement (D3.2).

**To check in the draft:**

- The framing in the Discussion: the limit lies in the data, not the model.
- The two exploratory observations, which are labelled as not pre-registered: H1 in held-out Tanzania, and Group DRO under shift.
- The Tanzania-collapse reading in section 5.2, which is labelled as untested.

## Suggested commit message

```text
Week 4 Checkpoint 4 and paper draft: verdicts from the GPU run, E6 refit thread fix
```

## Next

1. Review `paper/paper.md`.
2. Optional: the Sagawa et al. (2020) PDF, to cite the update rule instead of stating it as our own.
3. Optional, post hoc and labelled as such: flip Tanzania's `location_type` coding and rerun held-out Tanzania.
