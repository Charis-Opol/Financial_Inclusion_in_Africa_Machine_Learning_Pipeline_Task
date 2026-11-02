# Status: Checkpoint 0 (Phase 0)

Date: 2026-11-02. Base commit: `48462c6`. Nothing committed or pushed. No experiments run.

**Please decide before Phase 1:** the three questions under Next.

## Done

| File | Contents |
|---|---|
| `week4/ORIENTATION.md` | What exists in the repo, with paths, and how it is reused |
| `week4/RESEARCH_FRAMEWORK.md` | Framework section of the brief, unchanged |
| `week4/HYPOTHESES.md` | RQ1 to RQ6 with brief wording unchanged, plus operational rules. **For you to commit before any experiment.** |
| `week4/paper/literature_index.md` | Inventory and an entry for all 31 PDFs |
| `week4/paper/LITERATURE_GAPS.md` | Missing sources, thin themes, data facts the folder cannot support |
| `week4/DECISIONS.md`, `week4/RESEARCH_LOG.md` | Decisions D0.1 to D0.15, dated log |
| `week4/src, notebooks, results/runs, figures, tests` | Empty folders, kept with `.gitkeep` |

## Findings

No results exist in `week4/results/` yet, so every number here is tagged `[VERIFY]` with its source. Phases 1, 2 and 4 regenerate them.

1. **Rwanda, not Uganda, is the worst country on raw PR-AUC so far.** Rwanda has the most rows and Uganda the fewest, so small sample size alone does not explain it. The framework's "likely Uganda" is not supported as a default.

   | Country | Rows | Prevalence | Tuned XGBoost PR-AUC | PR-AUC minus prevalence | Lift |
   |---|---|---|---|---|---|
   | Kenya | 6,068 | 25.1% | 0.649 | 0.398 | 2.6 |
   | Rwanda | 8,735 | 11.5% | **0.473** | **0.358** | 4.1 |
   | Tanzania | 6,620 | 9.2% | 0.627 | 0.535 | 6.8 |
   | Uganda | 2,101 | 8.6% | 0.553 | 0.467 | 6.4 |

   `[VERIFY: PR-AUC from reports/ablation_study_report.md §5; prevalence from reports/eda_summary.md; rows from an ad hoc count; last two columns derived by hand]`

   The worst country also depends on the scale: on lift it is Kenya.

2. **The source for the Group DRO update is not in the folder.** Sagawa et al. (2020) is absent, and "exponentiated" occurs in none of the 31 papers. This is the one direct conflict in the brief: Phase 3 says to follow that paper, and rule 9 bars citing it. The folder does support the objective, the C/√n_g adjustment, group-balanced sampling and the 0.01 default step size.

3. **Nothing in the folder combines foundation-model representations with robust training,** so RQ4 has no direct precedent. The folder's evidence also leans against RQ3: in `koh2021wilds`, Group DRO rarely beats ERM and matches reweighting where it does help.

4. **Country and survey year are fully confounded:** Kenya 2018, Rwanda 2016, Tanzania 2017, Uganda 2018 `[VERIFY: ad hoc count]`. The paper cannot separate a country effect from a survey-round effect.

5. **Error in an existing report.** `reports/ablation_study_report.md:278-279` says Uganda "was worse" than Rwanda before tuning at 0.483 against 0.438, and that tuning reversed the order. 0.483 is the higher value, so there was no reversal. I did not edit it (rule 1). You may want to correct it yourself.

6. **The baseline fits preprocessing before CV** (`scripts/run_tuning_and_ablation.py:169`, `:172`), which rule 5 does not allow for new work. Phase 2 measures the effect.

7. **All 31 PDFs have a readable text layer.** None is scanned. Limits on the read: figures through captions only, reference lists not studied, and 27 text-less pages in `world bank.pdf` that I could not render to confirm are blank. That report never mentions Rwanda.

## Decisions and assumptions

**To confirm or overrule by committing `HYPOTHESES.md`** (details in `DECISIONS.md`):

- D0.2: worst-country is a floating minimum per model, with Uganda, Rwanda, and prevalence-adjusted versions always reported.
- D0.3: δ = 0.01 PR-AUC as the meaning of "trivially" in RQ3. This is my judgement, not from the literature.
- D0.4: all MLP arms in RQ3 are selected on worst-country inner PR-AUC, so selection does not favour Group DRO.
- D0.5, D0.6: each hybrid is compared with the better of TabPFN and DRO in each bootstrap replicate, with Holm correction. H2 with an ERM head is a control, not a hybrid.
- D0.7: "consistently better" in RQ5 means best in at least 3 of 4 held-out countries, and not significantly beaten in any.
- D0.8: RQ6 uses size-matched rankings on the 9 non-country features.
- D0.1: "Week 2 baseline" means `scripts/run_tuning_and_ablation.py`, because the repo's week labels differ from the brief.

**Assumptions I will proceed on:** D0.9 to D0.15.

## Problems and risks

- **TabPFN feasibility is the main risk.** Outer training folds have about 18,800 rows `[VERIFY: 80% of 23,524]`, above the 10,000-sample v2 limit in `hollmann2025accurate` and `jiang2026representation`. This machine has no GPU, and `johnson2026uncertainty` reports CPU as far slower. Nested CV across 5 seeds may take too long without subsampling or a GPU.
- **The installed TabPFN default may not be v2.** `kurashkin2026current` lists later versions as non-commercial and gated.
- **Worst-country estimates are noisy.** Uganda has 181 positives `[VERIFY: ad hoc count]`, and a minimum over noisy estimates is biased downward, so real gains may be hard to detect.
- **The label may understate inclusion.** Findex counts mobile money as an account, and reports Uganda at 66 percent, against 8.6 percent `bank_account` here `[VERIFY: demirguckunt2022findex p.196; reports/eda_summary.md]`.

## Review checklist

Read in this order:

1. `week4/HYPOTHESES.md`, in full. It becomes binding when you commit it.
2. `week4/paper/LITERATURE_GAPS.md`, section 1.
3. `week4/DECISIONS.md`.
4. `week4/ORIENTATION.md`, "Differences the Week 4 harness has to handle".
5. Spot-check a few entries in `week4/paper/literature_index.md` against the PDFs.

Reproduce the checks (Git Bash):

```bash
# Literature: 31 files, and no paper contains "exponentiated"
ls applied_research/literature | wc -l
for f in applied_research/literature/*.pdf; do pdftotext "$f" - | grep -il exponentiated >/dev/null && echo "HIT: $f"; done

# Rows, positives and survey year per country
awk -F, 'NR>1{n[$1" "$2]++; if($4=="Yes")p[$1" "$2]++} END{for(k in n)print k,n[k],p[k]}' data/raw/Train_v2.csv | sort

# Nothing outside week4/ changed
git status --short
```

## Suggested commit message

```text
Add Week 4 Phase 0: pre-registered hypotheses, framework, literature index and gaps
```

To keep the pre-registration timestamp clean, you could commit `week4/HYPOTHESES.md` on its own first.

## Next

**Phase 1** audits the data per country and tests TabPFN feasibility in a new isolated environment: version, licence, row limits, CPU runtime, embeddings and fine-tuning on toy data. It trains none of the study's models.

**Three things I need from you:**

1. **Group DRO source:** add the Sagawa et al. (2020) PDF to `applied_research/literature/` (my recommendation), or tell me to describe the update as our own implementation with no citation for it. This must be settled by Phase 3.
2. **`HYPOTHESES.md`:** commit it, with any changes you want first. No experiment runs before this.
3. **Go-ahead:** reply "continue to Phase 1". If TabPFN needs a Hugging Face login or licence acceptance, I will stop and ask, because that is your account.
