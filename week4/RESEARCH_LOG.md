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
