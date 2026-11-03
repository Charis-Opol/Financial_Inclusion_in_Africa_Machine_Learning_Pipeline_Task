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

## Phase 2

| ID | Decision | Why | Status |
|---|---|---|---|
| D2.1 | E1 calls the baseline's own `run_xgboost_arm` and `run_pytorch_arm`, importing the script without calling its `main()`. | `main()` writes to `reports/`, which rule 1 protects. Importing the original functions means the reproduction runs the original code, not a copy. | Decided |
| D2.2 | E1 covers the two arms the brief names (XGBoost/class-weight and the 2-layer MLP/class-weight), not the SMOTE or 3-layer arms. | These are the baseline the study compares against. | Decided |
| D2.3 | `imbalanced-learn==0.14.2` and `seaborn==0.13.2` added to `week4/.venv`, at the baseline's pinned versions. | The baseline modules import them. All shared libraries match `requirements.txt` exactly. | Decided |
| D2.4 | Week 4 results use fold-fitted preprocessing (the `in_fold` variant's approach). The baseline's reported numbers stay as reported. | Rule 5. The measured effect is small, so comparisons with the reported baseline remain meaningful. | Decided |

## Phase 3

| ID | Decision | Why | Status |
|---|---|---|---|
| D3.1 | **Run the whole suite on this CPU (Ryzen 7 5800U), one stage at a time, from the user's own terminal.** Supersedes D1.12's "possibly a GPU". | The user proposed it, and I recommend it. Estimated about 32 h of compute (4A about 18 h, 4B about 13 h, 4C about 1 h), scaled from E1 run times and Phase 1's 786 s TabPFN fit. These are estimates, not measurements. Keeps every arm and seed on one device, and keeps the tested environment. Risk: sustained heat throttling on a laptop chip is untested. | **Superseded:** the user moved the runs to a GPU machine (D3.4, D3.7, D3.8) |
| D3.2 | Group DRO's exponentiated-gradient update is presented as our own implementation unless the Sagawa et al. (2020) PDF is added to `applied_research/literature/`. | Rule 9. No paper in the folder gives the update. | **Confirm, Checkpoint 3** |
| D3.3 | **Added a secondary E3 pair, decided before any result exists:** `mlp_erm_within_cw` and `mlp_dro_within_cw`, with class weights balanced inside each country. Both select on worst-country inner PR-AUC. `mlp_dro` stays the primary RQ3 arm. | Walkthrough section 11, point 1: with pooled class weights, Group DRO may weight Kenya rather than the worst country. Adding the ERM twin means the pair differs only in the objective. The user asked for everything to be prepared to run, so I made this call rather than leave it open. Reported in full either way. | **Decided by me; you can drop it** |
| D3.4 | H2 embeddings use 8 estimators on the GPU machine (`H2_ESTIMATORS`, default 8), matching the TabPFN parent. Supersedes D1.6's 2, which was set only for CPU run time. The estimator count is in the embedding cache path, so counts never mix. | Removes the mismatch D1.6 had to report. | Decided |
| D3.5 | E4 hybrids run every hybrid arm (`h1_balanced`, `h2_erm`, `h2_rw`, `h2_dro`) under leave-one-country-out, in stage 4B. | HYPOTHESES.md RQ5 says the "best hybrids" enter E4 after E5. Running all of them means nothing is chosen after seeing E5. `h2_erm` is still a control. | Decided |
| D3.6 | **E6 settings:** seed 0. XGBoost uses interventional TreeSHAP; `mlp_erm_wc` and `mlp_dro` use exact interventional Shapley values over the raw features, with Week 3's code. 200 background rows from each fold's training rows, as Week 3 D2.5. Every row is explained by its own outer fold's refitted model, and the refit is checked against the saved scores. Each repeat draws one subsample per country, used in all 6 pairs. The Group DRO part uses the ERM MLP's worst country (lowest 5-seed mean PR-AUC). Rows with a level the fold never trained on are skipped and counted. TabPFN is excluded: exact Shapley values would need 1,024 × 200 TabPFN predictions per row. | HYPOTHESES.md RQ6 left these open. Exact Shapley values are what KernelSHAP approximates. On smoke data the refits matched the saved scores to 3e-8. | Decided |
| D3.7 | **On the GPU, TabPFN keeps its default fp16 autocast** (`inference_precision="auto"`, verified in `tabpfn/base.py` and `tabpfn/utils.py`). Phase 1 ran on CPU in float32. The preflight measures CPU against GPU score agreement on synthetic labels and records it. XGBoost and the MLPs stay on CPU; E6 scores MLP coalitions on the GPU. | RQ1 is about TabPFN "out of the box", and fp16 is its default on GPU. The agreement check shows how much that changes the scores. | Decided |
| D3.8 | **All stages run unattended in one go, at the user's request.** Checkpoints 4A, 4B and 4C become reviews of finished results. Nothing is rerun, retuned or added after results are seen without a post-hoc log entry. | Rule 4 still holds. Every arm was fixed before the run. | Decided by the user |
| D3.9 | CPU arms run in parallel lanes, one per seed (on 12+ cores), each with cores ÷ lanes threads. The thread count and host are saved with every fold. | Saves wall time. A sweep arm always runs after its parent in the same lane. Thread count can change MLP results in the last bits only. | Decided |
| D3.10 | `data/raw/Train_v2.csv` and `Test_v2.csv` are tracked in git (root `.gitignore`), so a clone has the data. | The user moves the repo by push and clone. **The repository is public, so pushing publishes the data.** | Decided by the user |
| D3.11 | **Analysis definitions:** W = min over countries of the seed-mean P_c, chosen per replicate; per-seed W is used only for seed noise. Bootstrap seed 2026. Bootstrap p-values: two-sided 2·min(share ≤ 0, share ≥ 0); for RQ4, one-sided share ≤ 0. Holm uses these. The analysis writes quantities and figures; verdicts are written at the checkpoint. | Fills in what HYPOTHESES.md implies but doesn't spell out. | Decided |

## Phase 4 review and Phase 5

| ID | Decision | Why | Status |
|---|---|---|---|
| D4.1 | Verdicts applied exactly as pre-registered: RQ1 matches; RQ2, RQ3, RQ4 and RQ5 falsified; RQ6 not falsified for the primary model (XGBoost). No rule, arm or setting changed after results. | `HYPOTHESES.md` and rule 4. | Decided |
| D4.2 | E6 now refits each fold with its recorded `torch_threads`. The GPU run refitted MLPs with 32 threads against 5 in training, and the refits differed from the saved scores by up to 0.17 (XGBoost matched to 3e-8). Confirmed here: same thread count gives identical fits; 5 against 16 threads moves scores by up to 0.049. **E6 was not rerun, by the user's decision.** The paper states the caveat for the MLP rows of RQ6. | MLP training is deterministic only at a fixed thread count. The primary RQ6 model is unaffected. | Decided by the user |
| D4.3 | The paper is `paper/paper.md`. Exploratory patterns (H1 in held-out Tanzania, Group DRO under shift, the Tanzania collapse under Group DRO) are labelled as not pre-registered and untested. | Reporting commitments in `HYPOTHESES.md`. | Decided |
