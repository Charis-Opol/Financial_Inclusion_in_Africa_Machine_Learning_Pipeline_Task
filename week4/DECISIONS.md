# Design decisions

Each entry gives the choice and why. Entries marked **Confirm** are for you to confirm or overrule at the checkpoint named. The operational rules for hypotheses live in `HYPOTHESES.md`, and these entries explain them.

## TabPFN verification record

Required by rule 3. Verified by reading the installed source and running `week4/src/tabpfn_feasibility.py`. Evidence: `results/feasibility.md`.

| Item | Verified |
|---|---|
| Version | `tabpfn` 9.2.0, in `week4/.venv` (Python 3.13.7, torch 2.12.1 CPU) |
| Weights | v2, `tabpfn-v2-classifier-finetuned-zk73skhh.ckpt`, requested explicitly with `create_default_for_version(ModelVersion.V2)`. The package default is v3.5. |
| Login, licence | No login or acceptance needed for v2. Weights under Prior Labs License 1.1. |
| Limits | 10,000 samples, 500 features, 10 classes; 1,000 samples on CPU unless overridden |
| Context | Set exactly by subsetting before `fit`, or per estimator through `SUBSAMPLE_SAMPLES` lists |
| Embeddings | `get_embeddings(X, data_source=...)`, shape `(n_estimators, n_rows, 192)`. `"train"` and in-context `"test"` leak labels. |
| Fine-tuning | `FinetunedTabPFNClassifier` runs on CPU on v2. Custom loss only through a private method; batches carry no row groups. |

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

## Phase 1

| ID | Decision | Why | Status |
|---|---|---|---|
| D1.1 | Use TabPFN v2 weights by explicit request, and pin `tabpfn==9.2.0`. | The brief specifies v2. The package default is v3.5, which is gated and non-commercial. | Decided |
| D1.2 | **Row limit: each of the default 8 estimators gets its own stratified context of about 10,000 rows, stratified by country and label, using `SUBSAMPLE_SAMPLES` lists.** Rows are dealt so every training row is used by at least one estimator. | Stays inside v2's supported size, keeps each country's share of the context, and uses all the data. Built-in `stratified` preserves class only. It costs about the same as one 10,000-row context (786 s against 745 s). | **Confirm, Checkpoint 1** |
| D1.3 | Set `ignore_pretraining_limits=True`. | Needed to run above 1,000 rows on CPU, and to pass a full fold to `fit`. The contexts themselves stay within 10,000 rows, so the flag is not used to exceed the sample limit. | Decided |
| D1.4 | RQ1's "out of the box" means v2 defaults (8 estimators, no tuning, no `balance_probabilities`), with D1.2 as the only departure, reported as a condition of the result. | `HYPOTHESES.md` RQ1 requires the handling to be fixed before E2. No change to the pre-registration. | Decided |
| D1.5 | **H2 uses out-of-fold embeddings only.** Within each outer training fold, each row is embedded by a model whose context excludes it (5 internal folds, each context within 10,000 rows). Test rows use a context drawn from the whole training fold. `data_source="train"` is never used. | Noise-label AUC: 1.000 for train tokens, 0.955 for test tokens with the row in context, 0.520 mean out-of-fold. Limitation: training embeddings come from slightly smaller candidate pools than test embeddings. | Decided |
| D1.6 | H2 embeddings use 2 estimators, unless you approve the longer run. Head tuning uses saved embeddings in the inner loop. | Estimated 8 h against 32 h for 8 estimators on this CPU. H2 is compared with a parent using 8, and the paper says so. Final setting confirmed at Checkpoint 3. | **Confirm, Checkpoint 1** |
| D1.7 | H1 context: equal rows per country, capped by the smallest country's training rows (about 1,680 in an outer fold, so about 6,700 in total), keeping each country's own class ratio. | Group balance is what H1 tests. Country-by-label balance belongs to E7. | Decided |
| D1.8 | **H3 (E5c) is marked as future work.** | The check asked whether this is clean and quick, and it is not. The loss hook is a private method, and batches carry no row groups, so Group DRO would need changes to the package's data pipeline, on a CPU machine. Stated as not completed in the paper. | **Confirm, Checkpoint 1** |
| D1.9 | `country` is a feature in within-country CV for every model, and dropped for every model in LOCO, with preprocessing refitted on the three training countries. | Matches the baseline and the brief. Dropping it does not remove all country information: `job_type` levels differ strongly by country (total variation distance up to 0.866), and the paper states this. | Decided |
| D1.10 | TabPFN receives the same fold-fitted output as the MLP branch: 8 categorical indices declared through `categorical_features_indices`, 2 scaled numerics, and the outlier flag. | Rule 5: same preprocessing, fitted inside folds. | Decided |
| D1.11 | Unseen levels in LOCO go to the encoder's unknown bucket. Only `marital_status` "unknown" (8 Kenya rows) is affected. | Existing encoder behaviour; affects very few rows. | Decided |
| D1.12 | Run TabPFN on CPU as resumable overnight jobs, saving each fold as it finishes. | About 10 h for priority A. A GPU would help but is not required. | **Confirm, Checkpoint 1** |
| D1.13 | Leave data values uncorrected, including Tanzania's `location_type`, and report the anomaly. | See `RESEARCH_LOG.md`. Rule 1, and there is no way to confirm a coding error from the data itself. | **Confirm, Checkpoint 1** |
| D1.14 | Displaying "Built with PriorLabs-TabPFN" is needed only if weights or a product containing them are distributed. Acknowledge TabPFN v2 and its licence in the paper regardless. | Licence paragraph 10. The paper distributes neither. | Decided |
