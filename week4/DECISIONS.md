# Design decisions

Each entry gives the choice and why. Entries marked **Confirm** are for you to confirm or overrule at the checkpoint named. The operational rules for hypotheses live in `HYPOTHESES.md`, and these entries explain them.

## TabPFN verification record

Rule 3 requires the installed version and what was verified to be recorded here. **Not installed yet.** Phase 1 fills in: version, weights used, limits, licence and login requirements, embeddings, fine-tuning. Nothing about TabPFN's API has been relied on so far.

## Phase 0

| ID | Decision | Why | Status |
|---|---|---|---|
| D0.1 | "Week 2 baseline" means the tuned nested-CV results from `scripts/run_tuning_and_ablation.py`. | The repo's week labels do not match the brief. This script holds the tuned XGBoost and MLP with nested CV. | **Confirm, Checkpoint 0** |
| D0.2 | Worst-country is a floating minimum per model. Uganda and Rwanda are always reported, plus Excess and Lift versions. | The framework expects Uganda, but existing reports show Rwanda lowest on raw PR-AUC. Fixing the country in advance would measure the wrong group. Choosing it after results would be selection. The framework text itself is unchanged. | **Confirm, Checkpoint 0** |
| D0.3 | Smallest effect of interest δ = 0.01 PR-AUC, used for "trivially" in RQ3. | "Trivially" needs a number fixed before results. It is our judgement, not from the literature. | **Confirm, Checkpoint 0** |
| D0.4 | All three RQ3 MLP arms select on worst-country inner PR-AUC, with pooled-selected ERM as a sensitivity check. | Otherwise DRO's advantage could come from the selection rule and not from training. | **Confirm, Checkpoint 0** |
| D0.5 | RQ4 compares each hybrid with the better of TabPFN default and DRO MLP within each bootstrap replicate, with Holm correction across hybrids. | Avoids choosing a parent using test data, and controls for trying several variants. Conservative against hybrids. | **Confirm, Checkpoint 0** |
| D0.6 | H2 with an ERM head is a control, not a counted hybrid. | Its training is not group-robust, which is what RQ4 asks about. It still shows what the representation alone does. | **Confirm, Checkpoint 0** |
| D0.7 | RQ5 "consistently better" means best in at least 3 of 4 held-out countries with Holm-corrected paired CIs, and not significantly beaten in any. | The brief leaves "consistently" undefined. | **Confirm, Checkpoint 0** |
| D0.8 | RQ6 compares size-matched rankings on 9 non-country features, using out-of-fold rows. | Differences between large and small samples would otherwise look like country differences. Country attribution is constant within a country. | **Confirm, Checkpoint 0** |
| D0.9 | Seeds 0 to 4 control outer split, inner splits and initialisation, paired across models. E1 alone uses seed 42. | Rule 8 and rule 5. E1's purpose is to reproduce the existing run. | Assumption |
| D0.10 | E1 reproduces the baseline's pre-CV preprocessing as it stands, and the Week 4 harness fits inside folds. Phase 2 reports the difference. | Rule 1 protects the baseline, rule 5 governs new work, and a reproduction has to match the original to be one. | Assumption |
| D0.11 | Week 4 adds a wrapper for OOF predictions instead of editing `run_nested_cv`. | Rule 1. `run_nested_cv` returns fold metrics only. | Assumption |
| D0.12 | Keep the `country` feature in within-country CV, drop it for every model in LOCO. | Matches the baseline, and the brief requires dropping it in LOCO. Phase 1 reviews this. | Assumption |
| D0.13 | Inner folds in LOCO are stratified by country and label within the three training countries, not leave-one-country-out. | Uses the same tuning harness everywhere. Limitation: selection is not itself shift-aware, which the paper will state. | Assumption |
| D0.14 | Reference lists were not studied in the literature read. | Rule 9 bars citing them. Disclosed in `literature_index.md`. | Assumption |
| D0.15 | `demirguckunt2022findex` is the key for `world bank.pdf`. Its printed citation gives 2022, while its cover shows 2021. | Keys follow the printed reference citation. | Assumption |
