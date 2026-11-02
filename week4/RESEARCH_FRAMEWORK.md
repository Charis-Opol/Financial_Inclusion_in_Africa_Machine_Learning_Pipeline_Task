# Research framework

Copied unchanged from the study brief. This is the source of truth for the study. Operational definitions and any later changes are in `HYPOTHESES.md` and `RESEARCH_LOG.md`, not here.

---

**Topic:** Do tabular foundation models help the worst-off group? Testing TabPFN and Group DRO, alone and combined, for predicting bank account ownership in Kenya, Rwanda, Tanzania and Uganda.

**Research direction.** Pooled accuracy hides who a model fails. In financial-inclusion prediction, the country with the lowest account ownership and the smallest sample (likely Uganda, about 8.6% ownership) is the group where a poor model does the most harm, because it is the group least likely to be correctly identified for outreach. There are two plausible routes to a fairer model: a *model-side* route (a pretrained tabular foundation model, TabPFN, that may generalise better from few examples) and a *training-side* route (Group DRO, which directly optimises the worst group's loss). This study tests each route, tests them **combined**, and checks whether the gains hold when a country is entirely unseen.

**Aim.** Determine whether, and how, TabPFN, Group DRO, and combinations of the two improve worst-country performance for bank-account prediction, at what cost to pooled performance, and whether the improvement survives country shift.

**Objectives**
- O1. Build a reproducible, leakage-safe benchmark and reproduce the Week 2 baseline.
- O2. Compare TabPFN v2 (out of the box) with tuned XGBoost and the MLP, pooled and per country.
- O3. Implement Group DRO from scratch and quantify its worst-country gain and pooled cost against ERM and against simple reweighting.
- O4. Design and test hybrids that combine TabPFN with group-robust training.
- O5. Test robustness with leave-one-country-out evaluation.
- O6. Explain model behaviour per country with SHAP.
- O7. Report everything, honestly, in a complete research paper.

**Research questions and hypotheses** (write each hypothesis and its falsification criterion into `week4/HYPOTHESES.md` **before** running experiments for it; I will review and commit it at the Phase 0 checkpoint, so the commit timestamp shows it came first. If a hypothesis or criterion is changed after any results are seen, log it in `RESEARCH_LOG.md` as a post-hoc change and say so in the paper.)

- **RQ1 Accuracy.** Does TabPFN v2, out of the box, match or beat tuned XGBoost and the PyTorch MLP on pooled PR-AUC under the nested, country-stratified CV? *Falsified if* TabPFN is clearly below XGBoost with non-overlapping confidence intervals. A tie inside the noise counts as "matches".
- **RQ2 Worst-off country.** How much worse is each model in the lowest-performing country than on pooled data, and does TabPFN shrink or widen that gap compared with XGBoost? *Falsified if* TabPFN's gap is indistinguishable from XGBoost's once uncertainty is included.
- **RQ3 Group DRO.** Training the MLP with Group DRO (country as group), does worst-country PR-AUC improve, at what pooled cost, and does it beat simple inverse-frequency reweighting? *Falsified if* the worst-country gain is within seed-to-seed noise, exceeds the pooled cost only trivially, or is matched by reweighting.
- **RQ4 Hybrid (TabPFN + group-robust training).** Does combining TabPFN with group-robust training beat either approach alone on worst-country PR-AUC? *Falsified if* no hybrid variant beats the better of (best TabPFN-only, best DRO-only) on worst-country PR-AUC beyond confidence intervals.
- **RQ5 Shift across countries.** Training on three countries and testing on the fourth, which approach holds up best: TabPFN, XGBoost, the Group DRO MLP, or a hybrid? *Falsified if* no method is consistently better across held-out countries.
- **RQ6 Explanations.** Do SHAP-measured drivers differ by country, beyond what resampling noise alone produces, and does Group DRO change which features matter for the worst-off group? *Falsified if* cross-country ranking differences are no larger than differences between random subsamples of one country.

**Experiments**

| ID | Experiment | Answers | Priority |
|---|---|---|---|
| E1 | Reproduce the Week 2 baseline | O1 | A (must) |
| E2 | Main benchmark: TabPFN v2, tuned XGBoost, ERM MLP, nested CV, per-country metrics | RQ1, RQ2 | A |
| E3 | Group DRO MLP vs ERM MLP vs inverse-frequency-reweighted MLP and XGBoost; step-size and group-adjustment ablations | RQ3 | A |
| E4 | Leave-one-country-out for all methods | RQ5 | A |
| E5a | Hybrid H1: TabPFN with group-balanced context | RQ4 | B |
| E5b | Hybrid H2: TabPFN embeddings plus a head trained with ERM vs reweighting vs Group DRO | RQ4 | B |
| E6 | SHAP per country, with a resampling-noise baseline | RQ6 | B |
| E5c | Hybrid H3 (stretch): fine-tune TabPFN with a Group DRO loss | RQ4 | C |
| E7 | Further ablations (group definition: country vs country-by-label) | RQ3, RQ4 | C |

Work in priority order. A is the minimum viable study. B completes the main story. C only if time clearly allows. The paper must state exactly what was and wasn't completed.
