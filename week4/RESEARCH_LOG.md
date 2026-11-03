# Research log

Dated decisions, problems and surprises. Newest entry last. Post-hoc changes to `HYPOTHESES.md` must be logged here and labelled as post-hoc.

## 2026-11-02: Phase 0

**Done.** Read the repo and all 31 PDFs in `applied_research/literature/`. Wrote the Phase 0 documents. No model was trained, no package installed, and nothing outside `week4/` changed.

**Reading method.** Extracted text with `pdftotext`, tagged with PDF page numbers. All 31 have a text layer. 27 pages of `world bank.pdf` have no text; I could not render them to confirm they are blank, because `pdftoppm` is not installed. Figures were read through captions only. Reference lists were not studied.

**Surprises.**

1. **Rwanda, not Uganda, is the lowest raw PR-AUC country in the existing reports.** Tuned XGBoost: Kenya 0.649, Rwanda 0.473, Tanzania 0.627, Uganda 0.553 `[VERIFY: reports/ablation_study_report.md §5]`. Rwanda is also lowest for untuned logistic regression (0.433), XGBoost (0.438) and MLP (0.461) `[VERIFY: reports/baseline_results.md]`. Rwanda has the most rows and Uganda the fewest `[VERIFY: ad hoc count]`, so scarcity does not obviously explain the shortfall. Handled by the floating minimum in `HYPOTHESES.md`.
2. **The choice of scale changes which country is worst.** Derived by hand from tuned XGBoost and prevalence: PR-AUC minus prevalence gives Rwanda lowest (about 0.358), and lift gives Kenya lowest (about 2.6) `[VERIFY: derived from reports; regenerate in Phase 4]`.
3. **Error in an existing report.** `reports/ablation_study_report.md:278-279` says tuning reversed the ordering of Uganda and Rwanda, "where Uganda was worse" at 0.483 against 0.438. But 0.483 is higher than 0.438, and `reports/baseline_results.md` agrees, so Uganda was already ahead. There was no reversal. Left unedited under rule 1. Flagged at Checkpoint 0.
4. **The Group DRO update rule is not in the folder.** Sagawa et al. (2020) is absent, and "exponentiated" occurs in none of the 31 papers. This conflicts with the brief's instruction to follow that paper under rule 9.
5. **Country and survey year are fully confounded.** One year per country: Kenya 2018, Rwanda 2016, Tanzania 2017, Uganda 2018 `[VERIFY: ad hoc count]`. A country effect cannot be separated from a survey-round effect.
6. **Findex does not cover Rwanda and defines "account" differently.** It reports Uganda at 66 percent of adults with an account, against 8.6 percent `bank_account` here `[VERIFY: reports/eda_summary.md]`. Its definition includes mobile money (`demirguckunt2022findex` p.33, p.196), and its survey year differs too. These are not the same quantity.
7. **Baseline preprocessing is fitted before CV.** `scripts/run_tuning_and_ablation.py:169` and `:172`. The size of the effect is unknown until Phase 2.
8. **Training folds exceed TabPFN's published limit.** `hollmann2025accurate` and `jiang2026representation` (p.14) give 10,000 samples for v2. Outer training folds will have about 18,800 rows `[VERIFY: 80% of 23,524]`. There is also no GPU here. Both go to Phase 1. `kurashkin2026current` Table 7 lists v2 as Apache-2.0 with attribution and later versions as non-commercial and gated, so the installed default may not be v2.

**Assumptions.** See `DECISIONS.md` D0.1 to D0.15.

**Tooling problems, for the record.** `pdfinfo`, `pdfimages` and `pdftoppm` did not produce usable output, so pages were counted from form feeds. One long Bash heredoc write was blocked, so the index was written in parts and assembled by script.

## 2026-11-02: Phase 1

**Done.** Data audit, new environment `week4/.venv`, TabPFN source read, toy checks. No study model was trained, and no TabPFN score was computed on the study's real labels. `HYPOTHESES.md` was committed by the user first (`4929a9b`) and has not changed.

**Surprises.**

1. **`TabPFNClassifier()` is not v2 in this package.** The default is v3.5. Using the default would have silently run a gated, non-commercial model, not the one the study specifies.
2. **The obvious embedding call leaks.** On pure-noise labels a head on `data_source="test"` embeddings scored AUC 0.955 when those rows were also in the context. Out-of-fold scored 0.520 on average (SD 0.030 over 8 draws).
3. **Tanzania's `location_type` runs the other way.** Positive rate is 0.207 rural and 0.048 urban, while Kenya, Rwanda and Uganda are all higher in urban areas. `demirguckunt2022findex` (p.47) reports Tanzania's urban adults about 13 points more likely to have an account, though its definition and year differ. A swapped coding is possible but unproven. Tanzania's marital status and household size also look different from the other three (`data_audit.md`). Not corrected. A swap would not necessarily affect within-country CV, but could affect LOCO.
4. **`job_type` is coded very differently across countries.** Uganda has no "Farming and Fishing", "Informally employed", "Remittance Dependent" or "Formally employed Government" rows. Tanzania has no farming or government rows. Total variation distance reaches 0.866 (Rwanda and Uganda). This looks like questionnaire differences. It matters for LOCO, and it lets models infer country even with `country` dropped.
5. **Uganda has 36 to 37 positives per outer test fold.** This confirms that per-fold country metrics would be too noisy, and supports concatenated out-of-fold metrics.
6. **Provenance is unresolved.** The README calls the data "Findex-derived", but the Findex 2021 report in the folder never mentions Rwanda. Not resolved; a different Findex year is possible, and the folder cannot settle it.

**My errors.**

- A stray `python -` process from a Phase 0 command of mine had kept running since about 14:03 and used about 148 CPU minutes. It competed with steps 1 to 6 and most of step 7 of the first runtime ladder. I stopped it and reran the decision-relevant sizes. Effect at 5,000 rows: 311.7 s against 306.6 s clean. It did not affect any result other than timing.
- I told the user step 8 would take about 17 minutes and E2 about 7 hours. It took 745 s, and E2 is estimated at 5.5 hours in the setting chosen.

**Pre-registration.** No hypothesis or criterion changed. D1.8 removes E5c from planned work, which `HYPOTHESES.md` already allows ("H3 if run").

## 2026-11-03: Phase 2

**Done.** E1 ran both baseline arms twice: once exactly as the baseline did, and once with preprocessing fitted inside each outer fold. Comparison by script in `results/baseline_reproduction_summary.md`.

**Result: exact reproduction.** With the baseline's own code path, every fold PR-AUC and every threshold matches the original saved values to full precision (maximum difference 0.000000). The tuned hyperparameters match in every fold, and every mean ± std in the report matches to three decimals.

**Pre-CV preprocessing barely matters.**
- **XGBoost:** fitting inside folds changes nothing. One-hot encoding learns only which levels exist, and every level appears in every training fold.
- **MLP:** the scaler is learned from fewer rows. Mean PR-AUC moves by −0.000919, and at most 0.002724 in any fold. That's well below δ = 0.01 and the fold SD of about 0.019.

So the leakage flagged in Phase 0 was real in principle, but negligible here.

**My error, fixed before reporting.** My first comparison flagged every per-country SD as mismatched, by a factor of √(5/4). I had used the sample SD, while the original `summarize_country_metrics` uses NumPy's population SD (`reporting.py:51`). After matching it, all per-country values agree. There was no mismatch in the reproduction itself.

**Also done during Phase 2, at the user's request.** I wrote the Phase 3 implementation, tests and `CODE_WALKTHROUGH.md`.
- **Tests:** all 36 pass. One caught a real numerical bug: Group DRO weights overflowed to NaN at high step size and loss. I fixed it by moving the update to log space.
- **Smoke test:** every experiment completed on a tiny subset. Its outputs are kept apart in `results/smoke/`, which is gitignored.
- **Checkpoint 3:** Phase 3 still needs the user's review before any experiment uses this code.

## 2026-11-03: Phase 3

**Done.** The Phase 3 code written during Phase 2 was committed by the user in `95c91da`, before review. Rerunning the fast tests today: 35 pass. The Checkpoint 3 report is in `STATUS.md`. No experiment has used this code.

**Compute plan.** The user proposed running the whole suite on this computer. Scaling measured times (E1 per-seed MLP and XGBoost runs, Phase 1's 786 s TabPFN fit), I estimate about 32 h of compute, about 19 h of it TabPFN. I recommended it (D3.1). This estimate has not been checked against a real stage. Stage 4A's actual time will be recorded at Checkpoint 4A, along with how far the estimate was off.

## 2026-11-03: Phase 4 prepared for the GPU machine

**Plan change.** The user moved the runs to a different computer with a GPU, reached by clone, and asked for everything to run in one go, figures included (D3.8). Nothing has run on real labels yet.

**Added:**
- `run_everything.sh`: preflight, tests, smoke, parallel lanes, E6, analysis.
- `setup_env.sh`: environment setup on the GPU machine.
- `src/preflight.py`: machine checks, GPU against CPU agreement, timing.
- `src/analysis.py`: the pre-registered metrics, bootstrap, paired comparisons, figures.
- `src/e6_shap.py`: E6.
- `src/completion.py`: lists finished and missing folds.
- `GPU_RUN.md`: the run guide.

The raw data is now tracked in git. Decisions D3.3 to D3.11.

**Verified here:**
- **Tests:** 41 pass, including new ones for the bootstrap PR-AUC (matches sklearn with weights and ties), stratified resampling, Holm, and within-country class weights.
- **Smoke:** every stage, E6 and the analysis ran end to end on CPU. E6 refits matched saved scores to 3e-8.
- **Preflight:** passed on CPU.
- **Not verified:** nothing has run on a GPU yet. The preflight and smoke steps on that machine are the first test.

**Measured, and acted on.** Exact Shapley values for one MLP row took 260 ms with 100 background rows on this CPU. With Week 3's 200 rows, the two MLPs over all 23,524 rows would take about 7 hours here, so E6 scores MLP coalitions on the GPU.

**Decided by me without the user's answer (D3.3).** The within-country class-weight pair was added before any result, so it is not post hoc. It is reported whatever it shows.

**Smoke numbers are not results.** They come from 1-epoch, 2-trial runs on 2,400 rows, live in the gitignored `results/smoke/`, and are never reported.
