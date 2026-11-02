# Literature gaps

The folder `applied_research/literature/` is the whole literature for this study (rule 9). This file records what it does not support. Where a gap applies, the text is written as our own statement or left out. No outside source fills it. Page numbers are PDF pages, as in `literature_index.md`.

## 1. Methods we must implement whose source is missing

| Method | Status | What the folder does support | Consequence |
|---|---|---|---|
| **Group DRO online update of q (exponentiated gradient)** | **Missing, and it blocks a citation.** The source the brief names, Sagawa et al. (2020), is not in the folder. "Exponentiated" occurs in none of the 31 papers. | The objective and C/√n_g adjustment: `ghosal2023distributionally` p.2-3 (its text layer drops the √ sign) and `stahlschmidt2024domain` p.5 (Eq. 3 and 4). Need for strong regularisation: `stahlschmidt2024domain` p.5. Uniform per-domain sampling, default step size 0.01, and non-vanishing training loss through early stopping: `koh2021wilds` p.66-67. Group DRO as a baseline: `sulaiman2025advancing`. | **Decision needed.** Either you add the Sagawa PDF to the folder (recommended), or the Methodology gives the update as our own implementation statement, with no citation for the update rule. |
| SHAP (original method) | Missing. No paper in the folder introduces it. | Secondary use only: `hollmann2025accurate` p.7, `shaktah2026established` (permutation SHAP, rank-biased overlap), `shan2026tabpfn` (`tabpfn_extensions.get_shap_values`), `borisov2024deep` (KernelSHAP, Spearman), `chen2026evaluating` (regional SHAP). | Describe SHAP through these papers' use of it. Give Shapley definitions as our own statement. |
| TreeExplainer, permutation and kernel explainers | Source papers missing. | As above. | Name the library and version used. No method citation unless a folder paper supports the statement. |
| TabPFN v1 and the original in-context training idea | Original paper missing. | v2 in `hollmann2025accurate`. Variant limits in `jiang2026representation` p.14. Generation comparison in `kurashkin2026current`. | Cite these only. |
| Spearman, Kendall, Holm correction, bootstrap percentile intervals | No source in the folder. | Some are used in folder papers, such as `borisov2024deep` for Spearman. | Standard statistics, described as our own statements. |
| Optuna / TPE, Adam | No source in the folder. | Adam appears in the `koh2021wilds` baseline descriptions (p.83, p.98) as a usage. | Name the library and version only. |

`chen2016xgboost` covers XGBoost, and `hollmann2025accurate` covers TabPFN v2. Inverse-frequency reweighting appears as a baseline in `koh2021wilds` (reweighting by label, Table 15, p.92; Table 23, p.108). Reweighting by **country** is our own specification.

## 2. Themes in the framework

| Theme | Coverage | Gap |
|---|---|---|
| TabPFN | **Good.** `hollmann2025accurate`, `kurashkin2026current`, `jiang2026representation`, `shaktah2026established`, `harijeyan2025evaluating`, `johnson2026uncertainty`, `tanna2026exploring`, `tanna2026tabtune`, `djilani2026robustness`, `guan2026stablepfn`, `liu2026causal`, `das2026bridging`, `pathan2026intelligent`, `shan2026tabpfn` | Nothing on TabPFN for survey-based financial inclusion. `shan2026tabpfn` and `pathan2026intelligent` are the nearest finance uses, on different tasks. |
| Group DRO | **Partial.** Objective and empirical record in `ghosal2023distributionally`, `koh2021wilds`, `stahlschmidt2024domain`, `sulaiman2025advancing` | No update rule (section 1). No Group DRO on tabular survey or financial data. |
| Worst-group evaluation | **Good in general.** `koh2021wilds`, `ghosal2023distributionally`, `stahlschmidt2024domain`, `barrainkua2025preserving`, `coston2019fair` | No paper uses worst-group **PR-AUC**, or addresses the fact that raw PR-AUC depends on group prevalence when comparing groups. The floating minimum, Excess and Lift approach in `HYPOTHESES.md` is therefore our own. |
| Financial-inclusion ML | **Partial.** `ismail2021using`, `balde2022efficient`, `chen2026evaluating`, `engelmann2018unbanked`, `coston2019fair`, `demirguckunt2022findex` | No per-country worst-group analysis for these four countries. No leave-one-country-out test on this data. `demirguckunt2022findex` does not mention Rwanda at all. |
| SHAP | **Secondary only** (section 1) | No paper tests cross-country SHAP ranking differences against a resampling-noise baseline. This is our own design. |
| Combining foundation-model representations with robust training | **Not covered.** | No paper does this, so RQ4 has no direct precedent. Nearest neighbours, none of which is this combination: `pathan2026intelligent` (ANN and TabPFN ensemble), `djilani2026robustness` (context optimisation), `das2026bridging` (instruction tuning), `guan2026stablepfn` (StablePFN), `liu2026causal` (interventions needed for fairness). State this as a gap the study addresses, without claiming to be the first anywhere: this folder is not an exhaustive search. |

## 3. Facts about our own data the folder cannot support

| Point | Status | Handling |
|---|---|---|
| Provenance of the dataset | Folder papers disagree: `ismail2021using` describes data from banks, while `balde2022efficient` describes Finscope surveys. | Do not state provenance from these papers without saying they disagree. Phase 1 looks for the repo's own data documentation. |
| Exact meaning of `bank_account` | No folder paper defines it for this dataset. `demirguckunt2022findex` p.33 counts mobile money as an account. | The ownership rates are not comparable. Phase 1 records what the repo documents; if nothing does, the paper says so. |
| Country and survey-year confounding | Not discussed in the folder for this data. | Our own observation, from data in `week4/results/` after Phase 1. |
| Consequences of outreach errors | The framework says a poor model does the most harm in the group least likely to be identified for outreach. No folder paper measures this for these countries. `demirguckunt2022findex` p.28 reports that free accounts alone had no welfare effect in Chile, Malawi and Uganda. | Write the harm argument as our own reasoning, with this caveat. |

## 4. Cautions when citing papers in the folder

Recorded in the individual index entries and repeated here so they are not lost at drafting time:

- `balde2022efficient`: possibly evaluated on a balanced test set, which could inflate results.
- `pathan2026intelligent`: its abstract and table disagree.
- `chen2026evaluating`: inconsistent scale in its figures.
- `fieggen2025navigating`: prose and table disagree.
- `ghosal2023distributionally`: the text layer drops the √ in the adjustment term, so check the PDF before quoting that equation.
