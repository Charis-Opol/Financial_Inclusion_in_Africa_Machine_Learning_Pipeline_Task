# Literature index

The whole literature for this study is the set of PDFs in `applied_research/literature/` (rule 9). Bibliographic details below are **as printed in each file**; where the file does not print something, the entry says "not printed". Page numbers are **PDF page numbers** (page 1 = first page of the file), which differ from journal page numbers. Text was extracted with `pdftotext` (poppler); no PDF was scanned or unreadable. Figures were read only through their captions and any embedded text. Every paper was read in full except reference lists and back matter, which were skimmed at most, because rule 9 bars citing them (for example `koh2021wilds` PDF p.30-60).

Feeds uses: RQ1 to RQ6 (see `week4/HYPOTHESES.md`), RW = Related Work, M = Methodology, D = Discussion, L = Limitations and Ethics.

Papers that only appear in another paper's reference list are **not** in this index and must not be cited.

## Inventory

All 31 files in `applied_research/literature/`. Each has a readable text layer; none is scanned.

| # | File | Citation key |
|---|---|---|
| 1 | `A Benchmark of in-the-Wild Distribution Shifts.pdf` | koh2021wilds |
| 2 | `A Scalable Tree Boosting System.pdf` | chen2016xgboost |
| 3 | `A TabPFN-Based Framework for Credit Risk Prediction in Automotive Green Supply Chain Finance.pdf` | shan2026tabpfn |
| 4 | `A Unified Library for Inference and Fine-Tuning Tabular.pdf` | tanna2026tabtune |
| 5 | `Accurate predictions on small data with a tabular foundation model.pdf` | hollmann2025accurate |
| 6 | `Advancing Equal Opportunity Fairness and.pdf` | sulaiman2025advancing |
| 7 | `An_Efficient_Approach_for_Predicting_the_state_of_Financial_Inclusion_by_Addressing_Class_Imbalance_using_the_Oversampling_Technique.pdf` | balde2022efficient |
| 8 | `Bridging ancestry gaps in genomic risk prediction with.pdf` | das2026bridging |
| 9 | `Calibration, Architecture, and Distribution Shift in Predictive Uncertainty Estimation.pdf` | sabanovic2026calibration |
| 10 | `Causal Pre-training Under the Fairness Lens.pdf` | liu2026causal |
| 11 | `DEEP LEARNING IS NOT ALL YOU NEED.pdf` | shwartzziv2021tabular |
| 12 | `Deep_Neural_Networks_and_Tabular_Data_A_Survey.pdf` | borisov2024deep |
| 13 | `Distributionally Robust Optimization with Probabilistic Group.pdf` | ghosal2023distributionally |
| 14 | `Domain_Generalization_of_Deep_Learning_Models_Under_Subgroup_Shift_in_Breast_Cancer_Prognosis.pdf` | stahlschmidt2024domain |
| 15 | `Established machine learning matches tabular foundation models in clinical predictions.pdf` | shaktah2026established |
| 16 | `Evaluating_Out-of-Distribution_Robustness_of_TabPFN_on_Tabular_Data.pdf` | harijeyan2025evaluating |
| 17 | `Evidence from Emerging Economies via Explainable Deep Learning.pdf` | chen2026evaluating |
| 18 | `Exploring Fine-Tuning for Tabular Foundation Models.pdf` | tanna2026exploring |
| 19 | `Fair Transfer Learning with Missing Protected Attributes.pdf` | coston2019fair |
| 20 | `High Performance, Low Reliability.pdf` | costa2026high |
| 21 | `Intelligent_Bank_Customer_Churn_Prediction_An_AI_Ensemble_Approach_Using_Neural_Networks_and_TabPFN_Foundation_Models.pdf` | pathan2026intelligent |
| 22 | `Navigating_Severe_Class_Imbalance_in_Population_Cohort_Data.pdf` | fieggen2025navigating |
| 23 | `On_the_Robustness_of_Tabular_Foundation_Models_Test-Time_Attacks_and_In-Context_Defenses.pdf` | djilani2026robustness |
| 24 | `On_the_Uncertainty_Quantification_Ability_of_Tabular_Foundation_Models.pdf` | johnson2026uncertainty |
| 25 | `Preserving the Fairness Guarantees of Classifiers in.pdf` | barrainkua2025preserving |
| 26 | `Representation_Learning_for_Tabular_Data_A_Comprehensive_Survey.pdf` | jiang2026representation |
| 27 | `Stable Prediction with Causal-Aware Tabular Foundation Model.pdf` | guan2026stablepfn |
| 28 | `The Current Generation of Tabular Foundation Models.pdf` | kurashkin2026current |
| 29 | `The_Unbanked_and_Poverty_Predicting_area-level_socio-economic_vulnerability_from_M-Money_transactions.pdf` | engelmann2018unbanked |
| 30 | `Using_Machine_Learning_Algorithms_to_Predict_the_State_of_Financial_Inclusion_in_Africa.pdf` | ismail2021using |
| 31 | `world bank.pdf` | demirguckunt2022findex |

---

## hollmann2025accurate
- **File:** `Accurate predictions on small data with a tabular foundation model.pdf`
- **Title:** Accurate predictions on small data with a tabular foundation model
- **Authors:** Noah Hollmann, Samuel Müller, Lennart Purucker, Arjun Krishnakumar, Max Körfer, Shi Bin Hoo, Robin Tibor Schirrmeister, Frank Hutter
- **Year / venue:** 2025 (published online 8 January 2025), Nature, Vol 637, 9 January 2025, pp. 319-326; DOI 10.1038/s41586-024-08328-6
- **Summary:** Presents TabPFN (the version this study calls v2), a transformer pre-trained on about 130 million synthetic datasets drawn from structural causal models, which predicts on a new dataset by in-context learning in a single forward pass. On benchmarks of up to 10,000 samples and 500 features, default TabPFN beats tuned gradient-boosted trees and AutoGluon on normalised ROC AUC at much lower time cost. The architecture applies attention over features and over samples, with test rows attending only to training rows. The paper shows proof-of-concept foundation-model abilities (density estimation, generation, embeddings, fine-tuning) and SHAP support. It states explicitly that its results are not evidence of scaling beyond 10,000 samples and 500 features.
- **Methods and claims (PDF page):**
  - In-context learning and Bayesian interpretation (posterior predictive approximation): p.2.
  - Two-way attention; test samples do not attend to each other and do not influence training representations: p.3, p.9.
  - Normalised ROC AUC 0.939 (TabPFN) vs 0.752 (CatBoost) default; 0.952 vs 0.822 tuned: p.5. Protocol: 10 repetitions, 90/10 splits: p.5, p.12.
  - Robust to uninformative features and outliers; with half the samples it still matches the next best method using all samples: p.6.
  - "should not be taken as evidence that TabPFN scales well beyond the 10,000 samples and 500 features": p.6. Limitations list and GPU recommendation: p.9.
  - Compute O(n^2 + m^2), memory O(n·m): p.9. Inference ensembling and softmax temperature (default T = 0.9): p.11. Default is a four-way ensemble for classification: p.12.
  - Natively limited to at most 10 classes: p.10.
  - **Embeddings are "the target-column representations of its final layer"**: p.12; proof of concept on mfeat-factors: p.7.
  - Fine-tuning proof of concept on sine-curve datasets: p.7, Extended Data Fig. 4 (p.17).
  - SHAP supported for TabPFN; Extended Data Fig. 3 comparison: p.7, p.16.
  - Does not address fairness or group robustness.
- **Feeds:** RQ1, RQ2, RQ4 (H2 embeddings, H3 fine-tuning), RQ6; RW (foundation models), M (TabPFN description, context limits), L (size limit).

## ghosal2023distributionally
- **File:** `Distributionally Robust Optimization with Probabilistic Group.pdf`
- **Title:** Distributionally Robust Optimization with Probabilistic Group
- **Authors:** Soumya Suvra Ghosal, Yixuan Li
- **Year / venue:** The Thirty-Seventh AAAI Conference on Artificial Intelligence (AAAI-23); copyright 2023
- **Summary:** Proposes PG-DRO, which generalises Group DRO from hard to soft (probabilistic) group membership, with group probabilities estimated by a classifier trained on a small group-labelled set. It restates the Group DRO objective and defines the worst-group risk with a C/sqrt(n_g) adjustment term. Evaluated on vision and NLP benchmarks using worst-group accuracy, PG-DRO matches or beats Group DRO with far less group annotation. ERM has the best average accuracy but the lowest worst-group accuracy in its tables. Models are selected by maximising worst-group accuracy on validation.
- **Methods and claims (PDF page):**
  - **Group DRO objective stated with equations** (expected loss per group, max over groups; empirical worst-group risk with 1/n_g): p.2. This is the fullest in-folder statement of the objective alongside `stahlschmidt2024domain`, and is attributed there to Sagawa et al. (2020a).
  - Eq. 3 PG-DRO risk with the capacity term C/sqrt(n_g), and the claim that the adjustment "helps reduce the generalization gap for each group": p.3. The PDF text layer drops the square-root sign, so check the page image before quoting.
  - Reduction to Group DRO when memberships are one-hot: p.3.
  - Model selection by worst-group validation accuracy: p.4. ERM vs G-DRO tables: p.4, p.6.
  - Describes G-DRO as "an online optimization algorithm ... that focuses on training updates over data points from higher-loss groups": p.7. **Does not give the exponentiated-gradient update.**
  - Mentions re-weighting or re-sampling approaches to worst-group performance: p.7.
- **Feeds:** RQ3, RQ4; RW (Group DRO), M (objective and adjustment term), E7 (groups formed as g = (y, e), p.2).

## stahlschmidt2024domain
- **File:** `Domain_Generalization_of_Deep_Learning_Models_Under_Subgroup_Shift_in_Breast_Cancer_Prognosis.pdf`
- **Title:** Domain Generalization of Deep Learning Models Under Subgroup Shift in Breast Cancer Prognosis
- **Authors:** Sören Richard Stahlschmidt, Göran Falkman, Benjamin Ulfenborg, Jane Synnergren
- **Year / venue:** 2024 IEEE Conference on Computational Intelligence in Bioinformatics and Computational Biology (CIBCB); DOI 10.1109/CIBCB58642.2024.10702166
- **Summary:** Evaluates group DRO, group-adjusted DRO, subsampling, multiple episode learning and stochastic weight averaging on gene-expression survival data, using one training cohort and four external cohorts. Subgroup-specific methods gave no clear gain over domain-generalisation methods on strongly shifted cohorts, though they improved slightly over ERM. No method was consistently best across external cohorts. Group DRO's worst-group improvement over ERM was small and noisy, with large standard deviations, and selecting models on worst-group loss made no apparent difference (data not shown).
- **Methods and claims (PDF page):**
  - Group DRO objective Eq. 3 and group-adjusted form Eq. 4 with C/sqrt(n_g), plus the statement that group DRO needs strong regularisation: p.5.
  - Subgroup shift as a mixture of group distributions with changed proportions: p.3.
  - Reports group gap (best minus worst group) and mean over 10 seeds: p.4, p.6-7.
  - Model selection on mean vs worst-group loss showed "no apparent difference (data not shown)": p.4.
  - Shared model outperforms subtype-specific models: p.5.
  - Worst-group gains sit within large SDs and are "to be taken with caution": p.5. No method consistently best: p.5-6.
- **Feeds:** RQ3, RQ5; RW (Group DRO on non-image data, a modest and noisy effect), M (objective and adjustment, regularisation), D.

## das2026bridging
- **File:** `Bridging ancestry gaps in genomic risk prediction with.pdf`
- **Title:** Bridging ancestry gaps in genomic risk prediction with tabular foundation models
- **Authors:** Anirban Das, Yan Cui
- **Year / venue:** Bioinformatics, 2026, 42, btag217 (ISMB proceedings supplement); DOI 10.1093/bioinformatics/btag217
- **Summary:** Tests whether in-context learning helps under-sampled ancestry groups in genomic risk prediction and finds off-the-shelf TabPFN degrades less than elastic net and random forest under sample imbalance on two biobanks. It also shows that off-the-shelf TabPFN fails when effect sizes vary across ancestry (non-stationarity), in simulation and in the dispersed META cohort. Instruction-tuning TabPFN on synthetic non-stationary tasks yields a prototype that is more stable for individuals far from the context. It is the closest analogue in the folder to our question, separating a sample-size effect from a changing-relationship effect across groups.
- **Methods and claims (PDF page):**
  - In-context procedure description for TabPFN: p.2.
  - Sample availability gap and relative performance gap (Eq. 1, 2): p.3. TabPFN has the lowest RPG in 5/11 AoU triplets and is never highest; 3/6 and 5/6 in eMERGE: p.6.
  - Non-stationarity gap in off-the-shelf TabPFN: p.7-8.
  - Instruction tuning updates all parameters on synthetic tasks and is not fine-tuning on real data: p.5. Performance vs distance from context (Mahalanobis): p.6, p.8.
  - Caveat that heritability may confound between-group differences: p.3.
  - Bootstrap on the test set with 60% and 85% CIs (narrower than our 95%): p.3, p.6.
- **Feeds:** RQ2, RQ5, RQ4; RW (TFMs with uneven groups, closest analogue), D (mechanisms for between-country differences), M (optional relative gap framing).

## liu2026causal
- **File:** `Causal Pre-training Under the Fairness Lens.pdf`
- **Title:** Causal Pre-training Under the Fairness Lens: An Empirical Study of TabPFN
- **Authors:** Qinyi Liu, Mohammad Khalil, Naman Goel
- **Year / venue:** 2026, Proceedings of the ACM Web Conference 2026 (WWW '26), Dubai; DOI 10.1145/3774904.3792961
- **Summary:** Evaluates TabPFN and a fine-tuned variant against logistic regression, random forest and an MLP on four fairness benchmarks, varying dataset size, spurious correlation and MNAR covariate shift. TabPFN achieves strong accuracy and flip consistency under spurious correlations, but fairness gains (demographic parity, equalised odds) are moderate and inconsistent, and limited under MNAR selection bias. The authors conclude causal pre-training is helpful but insufficient for fairness and that further interventions may be necessary. This directly motivates testing TabPFN combined with an explicit group-robust method.
- **Methods and claims (PDF page):**
  - Definitions of DP and EO; criteria cannot all be satisfied except in trivial cases: p.1.
  - TabPFN configuration: pre-training limits disabled, n_estimators=2, 5,000 subsampled context rows; FT-TabPFN fine-tuned for 10 epochs, Adam lr 1e-5, meta-batch 1, inner batch 5,000: p.2.
  - Accuracy and fairness vs dataset size: p.3. Flip consistency Table 2, MNAR Table 3: p.4.
  - Conclusion on causal pre-training and MNAR selection bias: p.4.
- **Feeds:** RQ2, RQ4, RQ5; RW (TFMs and fairness), M (precedent for context subsampling and fine-tuning settings), D.

## harijeyan2025evaluating
- **File:** `Evaluating_Out-of-Distribution_Robustness_of_TabPFN_on_Tabular_Data.pdf`
- **Title:** Evaluating Out-of-Distribution Robustness of TabPFN on Tabular Data
- **Authors:** Daksana Harijeyan
- **Year / venue:** 2025 International Conference on Advances in Technology and Computing (ICATC); DOI 10.1109/ICATC68823.2025.11407733
- **Summary:** Compares pre-trained TabPFN with Random Forest, XGBoost and LinearSVC on the Adult dataset, training on one demographic subgroup (by education, age or gender) and testing on another. XGBoost is strongest in-distribution while TabPFN has the highest or near-highest OOD accuracy and F1. Accuracy differences on the age split are not significant. It is a demographic hold-out design akin to leave-one-group-out, on a single dataset.
- **Methods and claims (PDF page):**
  - OOD split construction: p.2. Table IV results: p.4. Table V tests: p.5.
  - Age split accuracy differences not significant: p.5.
  - Author's own caveat about the asymmetry between zero-shot TabPFN and trained baselines: p.3, p.5.
  - Caution for our use: categoricals are label-encoded, and baselines appear untuned.
- **Feeds:** RQ5, RQ1; RW (TabPFN under group shift).

## tanna2026exploring
- **File:** `Exploring Fine-Tuning for Tabular Foundation Models.pdf`
- **Title:** Exploring Fine-Tuning for Tabular Foundation Models
- **Authors:** Aditya Tanna, Pratinav Seth, Mohamed Bouadi, Vinay Kumar Sankarapu
- **Year / venue:** 2026, Proceedings of the ACM Web Conference 2026 (WWW '26), Dubai; DOI 10.1145/3774904.3792923
- **Summary:** Large study of zero-shot, meta-learning, supervised (SFT) and parameter-efficient (PEFT) fine-tuning across six TFMs on TALENT, OpenML-CC18 and TabZilla. Zero-shot inference is already strong; meta-learning gives moderate gains; full SFT often harms accuracy and calibration, though TabPFN tolerates it best. Fine-tuning helps mostly on medium-sized or wide datasets and little on large ones. Fairness shows an accuracy-equity trade-off. This sets expectations for H3 and the "out of the box" framing.
- **Methods and claims (PDF page):**
  - Strategy definitions (SFT AdamW lr 1e-5 to 5e-5 up to 10 epochs; meta-learning 48-512 support samples for 5 epochs; PEFT LoRA r=8, α=16): p.1.
  - Leaderboards: p.2. Dataset-characteristic breakdown (Large >10K: limited benefit): p.2-3.
  - TabPFN stays well calibrated after SFT; others do not: p.2, Table 3 p.4.
  - Fairness Table 4 (SPD, EOD, EOpD): p.4. Practical guidelines: p.3.
  - PEFT not supported for some models (TabPFN rows are absent from PEFT tables): p.2, p.3.
- **Feeds:** RQ4 (H3 feasibility and priors), RQ1; RW (fine-tuning TFMs), D.

## tanna2026tabtune
- **File:** `A Unified Library for Inference and Fine-Tuning Tabular.pdf`
- **Title:** TabTune: A Unified Library for Inference and Fine-Tuning Tabular Foundation Models
- **Authors:** Aditya Tanna, Pratinav Seth, Mohamed Bouadi, Vinay Kumar Sankarapu
- **Year / venue:** 2026, Companion Proceedings of the ACM Web Conference 2026 (WWW Companion '26), Dubai; DOI 10.1145/3774905.3793137
- **Summary:** Describes TabTune, a library wrapping seven TFMs behind a scikit-learn-style API with model-aware preprocessing, fine-tuning modes (SFT, meta-learning, PEFT) and calibration and fairness evaluation. It is a software demo paper with no new comparative results. It is relevant only as one possible route for fine-tuning TabPFN, and any use must be verified against installed code under rule 3.
- **Methods and claims (PDF page):** Pipeline and tuning API: p.1-2. Evaluation metrics: p.2-3.
- **Feeds:** RQ4 (H3 feasibility only); RW (minor), M (only if used).

## ismail2021using
- **File:** `Using_Machine_Learning_Algorithms_to_Predict_the_State_of_Financial_Inclusion_in_Africa.pdf`
- **Title:** Using Machine Learning Algorithms to Predict the State of Financial Inclusion in Africa
- **Authors:** Qusai F. Ismail, Eslam S. Al-Sobh, Sarah S. Al-Omari, Tuqa M. Bani Yaseen, Malak A. Abdullah
- **Year / venue:** 2021 12th International Conference on Information and Communication Systems (ICICS); DOI 10.1109/ICICS52457.2021.9464590
- **Summary:** Applies twelve classifiers to what appears to be our dataset (Zindi; Kenya, Rwanda, Tanzania, Uganda; same columns) to predict bank-account ownership from a single holdout split. XGBoost is best on accuracy (89.23%), with F1 0.51. Accuracy and F1 are the headline metrics, there are no per-country results, and resampling was tried and abandoned because it reduced accuracy. It ranks education level as the top feature by random-forest importance.
- **Methods and claims (PDF page):**
  - Dataset description Table I: p.2. Random forest feature ranking Table II: p.4.
  - Models and parameters Table III: p.5. Results Table IV (XGBoost 89.23%, precision 0.72, recall 0.39, F1 0.51): p.5. Resampling comment and confusion matrix: p.6.
  - The paper says the data were "obtained from banks" (p.2), which conflicts with `balde2022efficient` (Finscope surveys). Do not repeat without checking.
- **Feeds:** RQ1, RQ6 (feature importance); RW (financial-inclusion ML on this dataset); gap statement (no per-country or worst-group analysis).

## balde2022efficient
- **File:** `An_Efficient_Approach_for_Predicting_the_state_of_Financial_Inclusion_by_Addressing_Class_Imbalance_using_the_Oversampling_Technique.pdf`
- **Title:** An Efficient Approach for Predicting the state of Financial Inclusion by Addressing Class Imbalance using the Oversampling Technique
- **Authors:** Abdoulaye Balde, Sabin Adhikari, Adarsh Singh, Nomaswati Princess Mabuela, Ankur Choudhary
- **Year / venue:** 2022 Second International Conference on Computer Science, Engineering and Applications (ICCSEA); DOI 10.1109/ICCSEA54677.2022.9936452
- **Summary:** Uses the same Zindi data, applying SMOTE and comparing nine classifiers on a 90/10 holdout, with Random Forest reporting about 91.5% on F1, precision, recall and accuracy. The paper describes the data as coming from Finscope surveys from 2016 to 2018 and gives per-country sample sizes. The near-identical precision, recall and accuracy suggest evaluation on a balanced (oversampled) test set, so these numbers may not be comparable with metrics at the natural prevalence, although the paper does not state the order of SMOTE and the split explicitly. There are no per-country or PR-AUC results.
- **Methods and claims (PDF page):**
  - Per-country counts and Finscope origin: p.2. Class ratio 20212:3312 (6.10:1): p.3.
  - SMOTE, models, 90/10 holdout: p.1, p.3. Table I results: p.4.
  - Caution for our use: the balanced-test-set reading is our inference. Inspect the flowchart on p.2 and hedge it if used in the text.
- **Feeds:** RQ1; RW (financial-inclusion ML, imbalance handling), M (data provenance), D (why PR-AUC at natural prevalence).

## pathan2026intelligent
- **File:** `Intelligent_Bank_Customer_Churn_Prediction_An_AI_Ensemble_Approach_Using_Neural_Networks_and_TabPFN_Foundation_Models.pdf`
- **Title:** Intelligent Bank Customer Churn Prediction: An AI Ensemble Approach Using Neural Networks and TabPFN Foundation Models
- **Authors:** Nauman Pathan, Vishwajeet Patil, Ayushi Pahadia, Smita Bharne
- **Year / venue:** 2026 IEEE International Conference on Interdisciplinary Approaches in Technology and Management for Social Innovation (IATMSI); DOI 10.1109/IATMSI68868.2026.11465999
- **Summary:** Combines an ANN and TabPFN by weighted averaging of probabilities (α = 0.3) to predict churn on a 10,000-row Kaggle bank dataset. It reports ensemble accuracy of 91.3% and ROC-AUC of 0.924 against 89.2% and 0.902 for TabPFN. It is an example of a TabPFN hybrid in a finance setting, but not a group-robust one. The abstract's single-model figure (68.8%) does not match Table I (87.8%, 89.2%), so its numbers should be treated cautiously.
- **Methods and claims (PDF page):** Ensemble Eq. 1: p.3. TabPFN on CPU with pre-training limits ignored: p.3. Table I: p.4. Abstract vs Table I inconsistency: p.1 vs p.4.
- **Feeds:** RQ4 (TabPFN hybrids exist, none group-robust); RW (minor, finance applications of TabPFN).

## costa2026high
- **File:** `High Performance, Low Reliability.pdf`
- **Title:** High Performance, Low Reliability: Uncertainty Benchmarking for Tabular Foundation Models
- **Authors:** José Lucas De Melo Costa, Fabrice Popineau, Arpad Rimmel, Bich-liên Doan
- **Year / venue:** ESANN 2026 proceedings, Bruges, 22-24 April 2026
- **Summary:** Benchmarks four TFMs, GBDTs, classical and deep baselines on 112 TALENT datasets, using conformal prediction to measure conditional coverage. TFMs reach the highest AUC but have lower size-stratified coverage scores than GBDTs, indicating overconfident errors. The gap widens with high noise and low separability on synthetic data. The authors see GBDTs as the most reliable choice for high-noise heterogeneous data and do not evaluate distribution shift.
- **Methods and claims (PDF page):** Conformal setup, LAC score, target coverage 0.90: p.3. SSC and SSCS definitions: p.3-4. Table 1 (AUC vs SSCS across families): p.5. Table 2 synthetic: p.5. Conclusions: p.5-6.
- **Feeds:** RQ1 (secondary metrics); RW (reliability of TFMs), M (motivates calibration metrics), D.

## fieggen2025navigating
- **File:** `Navigating_Severe_Class_Imbalance_in_Population_Cohort_Data.pdf`
- **Title:** Navigating Severe Class Imbalance in Population Cohort Data
- **Authors:** Joshua Fieggen, Bradley Segal, Emma C. Walker, Anshul Thakur, Christopher C. Butler, David A. Clifton, Lei Clifton
- **Year / venue:** 2025 47th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC); DOI 10.1109/EMBC58623.2025.11254293
- **Summary:** Compares XGBoost, logistic regression, SMOTE variants, anomaly detection and TabPFN for predicting a rare disease (0.3% prevalence) in UK Biobank. SMOTE gave no benefit and worsened XGBoost. TabPFN reached high test AUC with low F1 at the chosen threshold. TabPFN was applied to a subsample keeping all positives and 9,850 negatives because of its 10,000-row optimisation, a concrete precedent for prevalence-altering context subsampling. The paper argues the choice among models depends on the application's costs of false positives and false negatives.
- **Methods and claims (PDF page):**
  - TabPFN context: all 147 positives plus 9,850 sampled negatives: p.2.
  - Test results Table II: p.4. SMOTE degradation discussion: p.4. Application-dependent choice: p.4-5.
  - Prose and table disagree for TabPFN (AUC 0.88 and sensitivity 0.74 in prose on p.3; 0.87 and 0.81 in Table II on p.4). Note this if quoted.
- **Feeds:** RQ1, RQ4 (H1 context design: precedent for stratified context); RW (imbalance, TabPFN), M (context sampling), D, L.

## shaktah2026established
- **File:** `Established machine learning matches tabular foundation models in clinical predictions.pdf`
- **Title:** Established machine learning matches tabular foundation models in clinical predictions
- **Authors:** Lawrence A. Shaktah, Marco Gustav, Tim Lenz, Junhao Liang, Lars Hilgers, Zunamys I. Carrero, Jakob Nikolas Kather
- **Year / venue:** BMC Medical Informatics and Decision Making (2026) 26:364; DOI 10.1186/s12911-026-03654-3
- **Summary:** Benchmarks TabPFN v2.5 (default and HPO) against twelve established classifiers on twelve binary clinical tasks with up to 139,528 patients. TabPFN beats the best ML model in only 2 of 12 tasks, with most AUROC differences within ±0.02, and a median runtime 5.53 times longer. The authors recommend treating TabPFN as complementary rather than a replacement. It is the main counterpoint to the Nature paper's claims and supplies a precedent for permutation SHAP on TabPFN with rank-agreement metrics. XGBoost is not among its twelve listed baselines.
- **Methods and claims (PDF page):**
  - Result summary (2/12 tasks, within ±0.02, 5.53-fold runtime): p.1, p.5.
  - Software: Python 3.12.7, TabPFN v2.5, tabpfn-extensions v0.1.2: p.8. Baseline list: p.8. Search spaces Table 2: p.10.
  - Datasets over 100,000 samples randomly subsampled for TabPFN input constraints; 70/30 split; bootstrap of the training set with a fixed test set: p.9.
  - TabPFN on GPU, baselines on CPU; CPU feasible but with markedly longer runtimes: p.5, p.9.
  - No class rebalancing or oversampling: p.9.
  - Permutation SHAP with rank-biased overlap, Jaccard and top-20 overlap; TabPFN and boosting broadly agree: p.5, p.7, p.9.
  - Statistically clear but practically tiny differences (no overlapping intervals yet within ±0.02): p.5. Useful caution for our falsification rules.
- **Feeds:** RQ1, RQ6; RW (counterpoint on TFMs), M (SHAP agreement metrics, runtime reporting), D, L.

## shwartzziv2021tabular
- **File:** `DEEP LEARNING IS NOT ALL YOU NEED.pdf`
- **Title:** Tabular Data: Deep Learning is Not All You Need
- **Authors:** Ravid Shwartz-Ziv, Amitai Armon
- **Year / venue:** arXiv:2106.03253v2 [cs.LG], 23 Nov 2021 (the file is the arXiv preprint; journal venue not printed)
- **Summary:** Compares TabNet, NODE, DNF-Net, 1D-CNN and XGBoost on eleven tabular datasets using the same tuning protocol. Each deep model tends to do best only on datasets from its own paper, and XGBoost generally outperforms them on unseen datasets while needing less tuning. An ensemble of the deep models with XGBoost performs best overall. The paper warns that reported deep-model gains may reflect selection bias and unequal tuning.
- **Methods and claims (PDF page):** Deep model weaknesses on tabular data: p.1. Models described: p.2-3. Setup, 1,000-step HyperOpt, Friedman test: p.4-5. Test results Table 2: p.6. Average relative performance Table 3: p.7. Ensemble and tuning conclusions: p.8-9. XGBoost search space: p.12.
- **Feeds:** RQ1; RW (deep learning vs trees), M (fair tuning rationale).

## coston2019fair
- **File:** `Fair Transfer Learning with Missing Protected Attributes.pdf`
- **Title:** Fair Transfer Learning with Missing Protected Attributes
- **Authors:** Amanda Coston, Karthikeyan Natesan Ramamurthy, Dennis Wei, Kush R. Varshney, Skyler Speakman, Zairah Mustahsan, Supriyo Chakraborty
- **Year / venue:** AAAI/ACM Conference on AI, Ethics, and Society (AIES '19), January 27-28, 2019, Honolulu; DOI 10.1145/3306618.3314236
- **Summary:** Formulates fair transfer learning under covariate shift, motivated by health insurance markets and mobile-money loans expanding between East African countries, and proposes two weighting methods (PCCS and TFCS) for when protected attributes are missing in the target or source. The methods reduce score-parity disparity with little AUC change. It evaluates a country-to-country transfer on a dataset "modeled after" a financial institution's data. It also observes that transfer learning can either worsen or improve fairness, depending on the application.
- **Methods and claims (PDF page):**
  - Covariate shift assumption and importance weighting: p.2. Empirical risk minimisation stated (Eq. 1): p.2.
  - Mean and thresholded score parity (Eq. 3, 4); prevalence as a measure of dataset bias: p.2-3.
  - Weighting as compatible with many classifiers: p.1, p.3.
  - Mobile money setup (country 1 to country 2; age and gender; t = 0.75): p.5. Table 4 results (Native AUC 0.658, Unadapted 0.642): p.6.
  - Transfer can exacerbate or improve fairness: p.4.
- **Feeds:** RQ3 (reweighting), RQ5; RW (fairness, East African mobile money, cross-country transfer), M (reweighting rationale), D, L.

## engelmann2018unbanked
- **File:** `The_Unbanked_and_Poverty_Predicting_area-level_socio-economic_vulnerability_from_M-Money_transactions.pdf`
- **Title:** The Unbanked and Poverty: Predicting area-level socio-economic vulnerability from M-Money transactions
- **Authors:** Gregor Engelmann, Gavin Smith, James Goulding
- **Year / venue:** 2018 IEEE International Conference on Big Data (Big Data)
- **Summary:** Predicts area-level socio-economic status in Dar es Salaam, Tanzania, using random forests on mobile-money and call detail records for 517 areas. Mobile-money features improve accuracy from 65.9% to 71.3%, and adding call records barely helps further. It highlights that mobile-money services stand in for bank accounts for many people. Its unit of analysis and data differ from ours, but it provides East African context and privacy considerations for financial data.
- **Methods and claims (PDF page):**
  - About 35% of households with an M-Money user vs 2% with active bank accounts, as cited from other sources: p.1. M-Money as an alternative for those without bank accounts: p.1.
  - Results Table III: p.7. Permutation importance and partial dependence: p.6-8. Higher M-Money use is associated with lower affluence: p.8.
  - Limitations (selection, validation data, data access and privacy): p.8-9.
- **Feeds:** RW (East African financial access), D (bank account as an incomplete measure of inclusion), L (label meaning, privacy).

## chen2016xgboost
- **File:** `A Scalable Tree Boosting System.pdf`
- **Title:** XGBoost: A Scalable Tree Boosting System
- **Authors:** Tianqi Chen, Carlos Guestrin
- **Year / venue:** KDD '16, August 13-17, 2016, San Francisco; DOI 10.1145/2939672.2939785
- **Summary:** Describes XGBoost, a scalable gradient tree boosting system with a regularised objective, second-order approximation, shrinkage and column subsampling. It introduces sparsity-aware split finding and a weighted quantile sketch, along with cache-aware, compressed and sharded out-of-core system design. It reports use in many competition-winning solutions and strong speed and scalability against other boosting systems. This is the source for describing the XGBoost baseline.
- **Methods and claims (PDF page):**
  - Regularised objective Eq. 2: p.2. Additive training and second-order approximation Eq. 3: p.2. Leaf weights and structure score Eq. 5-7: p.3.
  - Shrinkage and column subsampling: p.3. Sparsity-aware split finding Alg. 3: p.4-5.
  - Hyperparameter framing relevant to tuning (max depth, shrinkage): p.8.
- **Feeds:** RQ1; RW (GBDT), M (XGBoost description).

## kurashkin2026current
- **File:** `The Current Generation of Tabular Foundation Models.pdf`
- **Title:** The Current Generation of Tabular Foundation Models: A Critical Review
- **Authors:** Sergei O. Kurashkin, Vadim S. Tynchenko, Aleksei S. Borodulin, Vladimir A. Nelyub, Nikolay O. Kalutsky, Tee Connie
- **Year / venue:** Mach. Learn. Knowl. Extr. 2026, 8, 244; published 13 August 2026; DOI 10.3390/make8080244
- **Summary:** Narrative critical review of the 2024-2026 tabular foundation models (98 retained studies from 961 screened records), covering architectures, capabilities, language-model approaches, deployment and independent evidence. Its central reading is that in-context prediction works but independent evidence has not matched benchmark claims. It reports TFMs strongest on tiny-to-medium IID data, with tree-based and deep models ahead on non-IID, large and high-dimensional data, weaker conditional coverage than GBDTs, and group disparities that survive pretraining. It says calibration, feature-shift robustness, fairness and interpretability are properties of the surrounding pipeline rather than the forward pass. The authors flag selection bias and a large share of preprints in the evidence base.
- **Methods and claims (PDF page):**
  - Definition of a TFM (pretrained across tasks, in-context forward pass, no task-specific weight updates): p.4-5.
  - TabPFN line versions; v2.5 envelope about 50,000 samples and 2,000 features; v3 one million rows: p.6. Table 3: p.10.
  - **Licences: TabPFN v2 "Prior Labs licence (Apache-2.0 with attribution)", TabPFN-2.5 and -3 non-commercial, gated weights** (verified by those authors 13-14 July 2026): p.10, Table 7 p.21. Must be re-verified against the installed package in Phase 1.
  - Context construction as a design choice; the labelled context's size, class balance and representativeness carry the weight hyperparameters once carried: p.19. Context-length effects (plateau, gain, degradation): p.12.
  - Attention-weighted vote readout for TabPFNv2: p.13. Full finetuning study summary (gains grow with target size; often not significant on small data): p.15.
  - Decision guide: non-IID, temporal or grouped shift defaults to tuned GBDT or deep models; fairness needs an explicit intervention: p.22-23 (Fig. 7, Table 8).
  - Independent evidence on generalisation p.25, calibration p.25, feature shift p.26-27, fairness p.27. Benchmark provenance p.27-28. Evidence weight by axis Table 11 p.29. Integrated trustworthiness reading p.29-30. Research agenda p.30.
  - It is a review: when citing it for a finding it attributes to another study, cite it as "reviewed by"; the primary study may not be in our folder.
- **Feeds:** All RQs as framing; RW (TFM overview, the gap), M (context as design choice), D, L.

## guan2026stablepfn
- **File:** `Stable Prediction with Causal-Aware Tabular Foundation Model.pdf`
- **Title:** StablePFN: Stable Prediction with Causal-Aware Tabular Foundation Model
- **Authors:** Zhengkang Guan, Yikang Chen, Haoyuan Qian, Kairong Han, Peng Cui, Fei Wu, Kun Kuang
- **Year / venue:** 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2 (KDD '26), Jeju Island, 2026; DOI 10.1145/3770855.3818030
- **Summary:** Proposes StablePFN, a TFM trained with three branches (sample decoupling, Markov boundary discovery, prediction) that masks feature attention to the discovered boundary at inference. On regression tasks with injected spurious features and inverted test correlations, it beats TabPFN v2.5, LimiX, LightGBM and LASSO on R², while the others collapse near or below zero. Removing the mask or the decoupling branch loses most of the gain. It shows that a structural change, rather than pretraining alone, can improve cross-environment stability, but under an assumption that generative mechanisms stay invariant across environments.
- **Methods and claims (PDF page):**
  - PFN objective and posterior predictive formulation (Eq. 1, 2): p.3. Stable prediction problem and invariance assumption: p.3.
  - Architecture and masks: p.3-5. Biased-sampling protocol: p.6. Results Table 2: p.7; 21 OpenML datasets: p.8. Ablations: p.7-8, p.11.
  - Limits: mechanism invariance, error propagation from boundary discovery: p.9.
  - Note: regression only; requires retraining a new model, so it is not directly applicable to our H1 to H3.
- **Feeds:** RQ5, RQ4; RW (TFMs under environment shift, model-side vs training-side routes), D.

## johnson2026uncertainty
- **File:** `On_the_Uncertainty_Quantification_Ability_of_Tabular_Foundation_Models.pdf`
- **Title:** On the Uncertainty Quantification Ability of Tabular Foundation Models
- **Authors:** Tyler R. Johnson, Kian Ben-Jacob, Nima Negarandeh, Oriol Vendrell-Gallart, Ramin Bostanabad
- **Year / venue:** Computing in Science & Engineering, July-September 2026; DOI 10.1109/MCSE.2026.3701626
- **Summary:** Compares TabPFN v2.5 with default-configured Gaussian processes on regression benchmarks using relative RMSE, a negatively oriented interval score and wall-clock cost. Neither method wins consistently: TabPFN is competitive in complex, higher-dimensional problems with enough data, while GPs often win in data-scarce problems when the kernel is a good prior. Attempts to tune TabPFN beyond its defaults gave no meaningful gain. Regression-only, so it bears on our study only through TabPFN's default behaviour and cost.
- **Methods and claims (PDF page):**
  - PFN background and uncertainty tied to the prior: p.2. Cost O(Nc(Nc + Nq)), with context caching: p.2.
  - Default-tuning attempt gave no meaningful improvement: p.4, p.10.
  - GPU vs CPU times (for example 1.36 s vs 21.98 s at small sizes; almost identical metrics on CPU at much greater wall-clock time): p.4-7.
- **Feeds:** RQ1 (rationale for out-of-the-box TabPFN), M (cost), L (CPU runtime).

## sabanovic2026calibration
- **File:** `Calibration, Architecture, and Distribution Shift in Predictive Uncertainty Estimation.pdf`
- **Title:** Calibration, Architecture, and Distribution Shift in Predictive Uncertainty Estimation
- **Authors:** Dina Šabanović, Tea Krčmar, Zdravko Krpić, Ivica Lukić
- **Year / venue:** Mach. Learn. Knowl. Extr. 2026, 8, 179; published 28 June 2026; DOI 10.3390/make8070179
- **Summary:** Matched-protocol benchmark on 36 OpenML-CC18 datasets comparing GBDTs, MLPs, MC-Dropout and deep ensembles under five post-hoc calibrators, with feature-shift, label-noise, conformal and TabPFN v2 sub-analyses. Calibrator choice mattered more than GBDT vs five-member ensemble, and preferred calibrators change with model family, imbalance and shift. TabPFN v2's raw probabilities were already well calibrated on its compatible subset, and post-hoc calibrators did not help. The design is a useful statistical template: pre-specified hypothesis families, paired Wilcoxon tests and Holm correction, with effect sizes and an explicit absence-of-evidence caveat.
- **Methods and claims (PDF page):**
  - Hypothesis families, Wilcoxon, Holm within family, rank-biserial effect sizes: p.5, p.8. Note: their unit of replication is the dataset, and ours is not.
  - ECE averaged over bins B ∈ {10, 15, 20}; NLL, Brier definitions: p.3, p.7.
  - "absence of evidence ... rather than as a formal equivalence result": p.12. Relevant to our "tie counts as matches" rule.
  - Oracle tuning caveat: needs nested CV for honest estimates: p.13-14.
  - Imbalance stratified results: flexible calibrators degrade when minority classes are small in the calibration set: p.16-17, p.21.
  - TabPFN v2 Table 11 (raw NLL 0.099, ECE 0.030, Brier 0.049), limits ≤10,000 training samples, ≤500 features, ≤10 classes: p.6, p.18. Package versions: p.6.
  - Fixed split keeps the early-stopping and calibration partitions disjoint: p.6.
- **Feeds:** RQ1 and RQ2 (secondary metrics), all RQs (statistical testing); M (ECE, Brier, Holm, reporting), L.

## djilani2026robustness
- **File:** `On_the_Robustness_of_Tabular_Foundation_Models_Test-Time_Attacks_and_In-Context_Defenses.pdf`
- **Title:** On the Robustness of Tabular Foundation Models: Test-Time Attacks and In-Context Defenses
- **Authors:** Mohamed Djilani, Thibault Simonetto, Karim Tit, Florian Tambon, Salah Ghamizi, Maxime Cordy, Mike Papadakis
- **Year / venue:** 2026 IEEE Conference on Secure and Trustworthy Machine Learning (SaTML); DOI 10.1109/SaTML68715.2026.00028
- **Summary:** Shows TabPFN v2 and TabICL are vulnerable to constrained evasion attacks on credit, phishing and clinical data, and can serve as surrogates for transferable attacks. It compares adversarial fine-tuning (weights) with a new adversarial in-context learning method that changes only the context. Modifying the context improved robustness more than updating the weights. For our purposes the key contribution is not the attacks but evidence that context composition is itself a lever on behaviour, and its description of subsampling large datasets to a 10,000-row context.
- **Methods and claims (PDF page):**
  - Context limit: 10k used in their setup, with sampling chosen across 10 seeds with and without rebalancing, by validation MCC: p.4-5.
  - TabPFN v2 with 1 estimator instead of 4, for control: p.4.
  - Context size effect (1,000, 5,000, 10,000) on robust accuracy: p.5, Fig. 4 p.8.
  - TabFMs underperform on recall on unbalanced data, compared with balanced data: p.5.
  - AFT and AICL objectives (Eq. 3, 4) and algorithms: p.7-10. AICL beats AFT in Table IV: p.10-11. Clean performance Table V: p.11.
  - Weakness: they rely on `hollmann2025accurate` version limits and their threat model is unrelated to ours.
- **Feeds:** RQ4 (H1: context as a lever; H3: fine-tuning split procedure), RW (TFM robustness), M (context subsampling precedent), L.

## sulaiman2025advancing
- **File:** `Advancing Equal Opportunity Fairness and.pdf`
- **Title:** Advancing Equal Opportunity Fairness and Group Robustness through Group-Level Cost-Sensitive Deep Learning
- **Authors:** Modar Sulaiman, Nesma Talaat Abbas Mahmoud, Kallol Roy
- **Year / venue:** Baltic J. Modern Computing, Vol. 13 (2025), No. 1, pp. 96-127; DOI 10.22364/bjmc.2025.13.1.06
- **Summary:** Proposes GLCS, a constrained-optimisation training objective that separates positive and negative classes globally and within each protected group, solved with an augmented Lagrangian. It targets group-level class imbalance, arguing that intersectional imbalance drives unfairness. On CelebA and UTKFace it reduces equal-opportunity gaps at some discrimination cost compared with ERM, and on CivilComments-WILDS it reports the best worst-group accuracy among baselines including Group-DRO (0.707 vs 0.699). Threshold-agnostic metrics, including PR-AUC, and Brier score are reported, and calibration is treated as a separate post-processing step.
- **Methods and claims (PDF page):**
  - Definitions of equal opportunity, equalised odds, group robustness: p.5-6. Group-level imbalance partitioning: p.6-7.
  - Objective (Eq. 3) and augmented Lagrangian (Eq. 4): p.7-8.
  - Group-DRO described as "optimizes a soft version of the worst-group loss": p.4. **No update rule given.**
  - Threshold-agnostic metrics rationale (PR-AUC under imbalance, ROC-AUC, Brier): p.11. Worst-group accuracy definition: p.12.
  - Early-stopping trade-off in fair ML: p.13. Threshold selection under imbalance: p.13-14. Temperature scaling preserves AUC: p.14.
  - CivilComments-WILDS Tables 14-15, with the groups chosen after inspecting ERM: p.15, p.28-29. The Christian group was picked using test-like ERM evaluation, so read the 0.707 with care.
- **Feeds:** RQ3, E7 (group by country-by-label, group-level class imbalance); RW (worst-group evaluation), M (threshold-agnostic metrics, calibration), D.

## chen2026evaluating
- **File:** `Evidence from Emerging Economies via Explainable Deep Learning.pdf`
- **Title:** Evaluating the Non-linear Impact of Digital Literacy on Financial Inclusion: Evidence from Emerging Economies via Explainable Deep Learning
- **Authors:** Sihan Chen
- **Year / venue:** Int J Comput Intell Syst (2026) 19:296; DOI 10.1007/s44196-026-01415-8
- **Summary:** Proposes FERENet, combining polynomial features, mutual-information selection, Random Forest, LightGBM and a residual DNN, to predict a financial-inclusion score across 56 emerging economies using Findex-derived data, and uses SHAP and LIME to explain predictions. It reports R² of about 0.94 and region-specific SHAP signs for digital literacy (positive in Africa and Latin America, marginally negative in parts of Asia). The paper states its findings are correlational and not causal. It is the folder's only Findex-based study with region-level SHAP, so it matters for RQ6, but its figures need caution.
- **Methods and claims (PDF page):**
  - Data: Global Findex 2021 described, 56 economies, 64,917 final observations: p.11-12.
  - In-fold feature engineering to avoid leakage: p.8.
  - Regional Table 7 (SHAP signs by region): p.19. Reliability Table 8: p.20. Correlational caveat: p.21-22.
  - Fairness defined as a difference in conditional expected prediction (Eq. 12): p.14.
  - Quality caution: 5-fold results (MAE 0.217, R² 0.888, Table 5, p.17) are on a different scale from the headline results (MAE 6.1, R² 0.9445, Table 6, p.19), and epoch curves are described with layer types the model does not have (p.16-17). Use only for the qualitative point that drivers can differ by region, and hedge it.
- **Feeds:** RQ6; RW (financial-inclusion ML, SHAP by region), D, L (correlation vs causation).

## shan2026tabpfn
- **File:** `A TabPFN-Based Framework for Credit Risk Prediction in Automotive Green Supply Chain Finance.pdf`
- **Title:** A TabPFN-Based Framework for Credit Risk Prediction in Automotive Green Supply Chain Finance
- **Authors:** Wenjie Shan, Xiuyu Kang, Benhe Gao
- **Year / venue:** Sustainability 2026, 18, 6305; published 18 June 2026; DOI 10.3390/su18126305
- **Summary:** Combines DE-LightGBM feature selection, TabPFGen minority-class augmentation and TabPFN v2 to predict credit risk for 210 Chinese automotive SMEs (1,050 observations, 112 risky). TabPFN has the best point estimates against XGBoost, CatBoost, LightGBM, logistic regression and random forest on a single small test set. Confidence intervals on the risk class are wide (22 positives in test). It uses `get_shap_values` from `tabpfn_extensions` to explain TabPFN and warns that SHAP is post hoc and not causal.
- **Methods and claims (PDF page):**
  - Preprocessing fitted on training data only: p.6. Augmentation only in training folds: p.8.
  - Baselines tuned by Bayesian optimisation, TabPFN v2 out of the box, threshold selected on training folds: p.8-9.
  - Table 3 (test results): p.13. Wilson and bootstrap CIs Table A5: p.24. Wilcoxon on 10 folds Table 4: p.15.
  - SHAP via `tabpfn_extensions.get_shap_values`: p.9; not causal: p.16. Verify this function in Phase 1 before relying on it.
- **Feeds:** RQ1, RQ6 (SHAP on TabPFN); RW (TabPFN in credit and finance), M (CI reporting on small positive counts).

## borisov2024deep
- **File:** `Deep_Neural_Networks_and_Tabular_Data_A_Survey.pdf`
- **Title:** Deep Neural Networks and Tabular Data: A Survey
- **Authors:** Vadim Borisov, Tobias Leemann, Kathrin Seßler, Johannes Haug, Martin Pawelczyk, Gjergji Kasneci
- **Year / venue:** IEEE Transactions on Neural Networks and Learning Systems, Vol. 35, No. 6, June 2024, pp. 7499-7519; DOI 10.1109/TNNLS.2022.3229161
- **Summary:** Surveys deep learning for tabular data with a taxonomy of data transformations, specialised architectures and regularisation models, plus tabular generation and explanation methods. Its benchmark of 11 deep methods against classical ones on five datasets finds gradient-boosted tree ensembles still mostly win on small and medium data, and that deep models lag except on a very large mostly numeric dataset. It explains why: low data quality, missing spatial structure, dependence on preprocessing and the importance of single features. It also compares feature-attribution methods against KernelSHAP using Spearman rank correlation. Pre-dates TabPFN v2 and tabular foundation models.
- **Methods and claims (PDF page):**
  - Four challenges of deep learning on tabular data: p.4. Encoding categorical variables, including one-hot and ordinal: p.5.
  - Benchmark protocol (Optuna with 100 iterations, five-fold CV, same preprocessing for all models): p.12-13. Table V and summary: p.13, p.15.
  - KernelSHAP as a comparator with Spearman agreement Table VI: p.14-15. Spearman as a measure of attribution agreement is a precedent for RQ6.
  - Tree ensembles remain state of the art on small and medium data (less than about 1M samples): p.15, p.17.
- **Feeds:** RQ1, RQ6; RW (deep learning vs trees), M (uniform preprocessing, rank correlation), D.

## barrainkua2025preserving
- **File:** `Preserving the Fairness Guarantees of Classifiers in.pdf`
- **Title:** Preserving the Fairness Guarantees of Classifiers in Changing Environments: A Survey
- **Authors:** Ainhize Barrainkua, Paula Gordaliza, Jose A. Lozano, Novi Quadrianto
- **Year / venue:** ACM Computing Surveys, Vol. 57, No. 6, Article 137, February 2025; DOI 10.1145/3637438
- **Summary:** Surveys methods that keep fairness guarantees when training and deployment distributions differ, with a taxonomy of adaptive methods (data-based, distribution-based, causal) and robust methods (DRO and environment-agnostic). It defines group, individual and mini-max fairness, and names types of shift: demographic, covariate, label, concept and compound. It argues that treating fairness and shift separately can be sub-optimal, and notes that fairness methods have been shown not to transfer automatically to a new environment. It identifies multi-source settings as an open direction, which is close to our leave-one-country-out situation.
- **Methods and claims (PDF page):**
  - **Mini-max fairness: train to minimise the error of the worst group (Rawlsian notion): p.4-5.** This is the fairness framing for "worst-off country".
  - Group metrics and their controversy when base rates differ between groups: p.5. Relevant to raw PR-AUC not being comparable across prevalences.
  - Shift types (demographic, covariate, label, concept, compound): p.6-7.
  - Reweighing methods, their compatibility with many classifiers, and failure when supports differ: p.12-13.
  - DRO formulation (Eq. 9) and ambiguity sets: p.16-18. Note this survey covers ambiguity-set DRO and does not describe Group DRO's update rule.
  - Label shift by geography (disease prevalence differs across regions) as an example: p.7.
  - Multi-source approaches as a future direction: p.25.
  - Reports WILDS, including POVERTYMAP-WILDS, as benchmarks containing sensitive information: p.21.
- **Feeds:** RQ2, RQ3, RQ5; RW (worst-group fairness, shift taxonomy), M (mini-max definition, reweighting), D, L.

## jiang2026representation
- **File:** `Representation_Learning_for_Tabular_Data_A_Comprehensive_Survey.pdf`
- **Title:** Representation Learning for Tabular Data: A Comprehensive Survey
- **Authors:** Jun-Peng Jiang, Si-Yang Liu, Hao-Run Cai, Qi-Le Zhou, Han-Jia Ye
- **Year / venue:** IEEE Transactions on Pattern Analysis and Machine Intelligence, Vol. 48, No. 6, June 2026, pp. 6488-; DOI 10.1109/TPAMI.2026.3657217
- **Summary:** Surveys tabular representation learning across specialised, transferable and general (tabular foundation) models, with a taxonomy of sample, objective and feature aspects, plus ensembles and extensions. It records the debates between trees and DNNs and explains why GBDTs often remain strong baselines. It describes TabPFN's variants, including limits (v1 N<1000, d<100, C<10; v2 N<10000, d<500; v2.5 up to 50000 samples and 2000 features) and strategies for scaling and adapting them. It states that fine-tuning a general model typically improves on zero-shot inference, in contrast to `tanna2026exploring`'s mixed findings. It notes benchmarks that find standard models degrade under distribution shift.
- **Methods and claims (PDF page):**
  - Challenges including class imbalance and domain constraints: p.4. Single-task evaluation metrics and ECE: p.4. Protocols, including multiple seeds and cross-validation; holdout-based tuning can be unstable: p.5.
  - Trees vs DNNs: reasons trees retain an advantage (mixed types, noise, computation, interpretability): p.6. "rigorous benchmarking against these established non-DNN approaches remains a critical standard": p.6.
  - Taxonomy: Table I p.7. Encoding: p.8-9.
  - General models definition (Eq. 6, no Adapt step): p.13. TabPFN v1, v2, v2.5 limits: p.14. Fine-tuning and adaptive context: p.14.
  - Post-hoc ensembles and TabPFN preprocessing variation: p.15. Open-environment shift: p.15.
- **Feeds:** RQ1, RQ4; RW (foundation models, trees vs DNNs), M (metrics and evaluation protocol).

## koh2021wilds
- **File:** `A Benchmark of in-the-Wild Distribution Shifts.pdf`
- **Title:** Wilds: A Benchmark of in-the-Wild Distribution Shifts
- **Authors:** Pang Wei Koh, Shiori Sagawa, Henrik Marklund, Sang Michael Xie, Marvin Zhang, Akshay Balsubramani, Weihua Hu, Michihiro Yasunaga, Richard Lanas Phillips, Irena Gao, Tony Lee, Etienne David, Ian Stavness, Wei Guo, Berton A. Earnshaw, Imran S. Haque, Sara Beery, Jure Leskovec, Anshul Kundaje, Emma Pierson, Sergey Levine, Chelsea Finn, Percy Liang
- **Year / venue:** Proceedings of the 38th International Conference on Machine Learning, PMLR 139, 2021 (file is arXiv:2012.07421v3, 16 Jul 2021)
- **Summary:** Introduces a benchmark of 10 datasets with realistic distribution shifts, separating domain generalisation (unseen test domains) from subpopulation shift (worst-case seen domains). Standard training has substantially lower out-of-distribution than in-distribution performance on every dataset, and CORAL, IRM and Group DRO generally do not improve over ERM. Group DRO's clear win is only in subpopulation shift (CivilComments, 70.0% vs 56.0% worst-group), and there it matches simple reweighting (69.2%). Its PovertyMap dataset, predicting asset wealth across unseen African countries, is close to our setting. Its SQF appendix shows that a worst-group gap can come from intrinsic noise rather than small group size, because training only on the worst group did not help.
- **Methods and claims (PDF page):**
  - Domain generalisation and subpopulation shift definitions; "minimize the maximum error over all test domains": p.7-8. Maps directly to our nested-CV (RQ2, RQ3) and LOCO (RQ5) settings.
  - Group DRO described as "uses distributionally robust optimization to explicitly minimize the loss on the worst-case domain during training": p.21. **No update equations in this paper.**
  - Table 1 ID vs OOD: p.20. Table 2 baselines: p.20. Group DRO results reading: p.22.
  - Implementation detail: hyperparameters selected with ERM and reused; **Group DRO step size fixed at default 0.01**; samples uniformly from each domain regardless of domain size: p.66. "The Group DRO objective is unchanged, as it still optimizes for the domain with the worst loss, but the uniform sampling improves optimization stability"; reweighting methods including Group DRO are effective only when the training loss is non-vanishing, achieved through early stopping: p.66-67. The word "exponentiated" does not occur in this paper.
  - FMoW: worst-region OOD accuracy 32.3 ERM vs 30.8 Group DRO, Africa the worst region, and the region with the fewest examples (Oceania) was not the worst, so the shortfall "is not merely a function of the number of training examples": p.96-99. Also Camelyon17, RxRx1, OGB-MolPCBA, GlobalWheat, Amazon, Py150 tables, where Group DRO does not beat ERM on the reported OOD metric: p.72-89, p.108-112.
  - In-distribution gap should be measured on P_test (the test domain) rather than P_train, or it conflates shift with intrinsic difficulty: p.17. Directly relevant to reading our LOCO drop.
  - Average vs worst-case comparisons; test-to-test and mixed-to-test: p.18-19.
  - Group DRO (label) and reweighted (label) nearly identical on CivilComments, Table 15: p.92. Choice of groups matters: p.93.
  - PovertyMap: 23 African countries; country as domain; worst urban/rural Pearson r; Group DRO 0.39 vs ERM 0.45 on OOD test; countries differ so much that transfer is hard; Kenya and Uganda roof-type example: p.14-15, p.102-105.
  - SQF: worst-group gap not fixed by training on that group; the group may be intrinsically noisier: p.24, p.113-115. Supports asking whether the worst-off country is hard because of scarcity or noise.
  - Compounding of subpopulation and domain shifts: p.23. Ethical issues of remote sensing, surveys: p.105.
- **Feeds:** RQ3, RQ5, RQ2; RW (worst-group evaluation, Group DRO empirical record), M (settings, sampling, step-size default), D, L.

## demirguckunt2022findex
- **File:** `world bank.pdf`
- **Title:** The Global Findex Database 2021: Financial Inclusion, Digital Payments, and Resilience in the Age of COVID-19
- **Authors:** Asli Demirgüç-Kunt, Leora Klapper, Dorothe Singer, Saniya Ansar
- **Year / venue:** World Bank, Washington, DC, 2022 (printed reference citation; the cover shows "2021"); DOI 10.1596/978-1-4648-1897-4
- **Summary:** Reports nationally representative surveys of about 128,000 adults in 123 economies, collected during the COVID-19 pandemic, on account ownership, digital payments, saving, borrowing and financial resilience. It calls account ownership "the fundamental measure of financial inclusion" and reports ownership growing to 76 percent of adults worldwide and 55 percent in Sub-Saharan Africa, largely through mobile money in that region. It reports wide differences within income groups and persistent gaps by gender, income, education and rural residence. It gives figures for Kenya, Tanzania and Uganda, but **Rwanda is not mentioned anywhere in the extracted text**, including the survey table and the ownership table. Its "account" includes mobile money, which is why its ownership rates cannot be compared directly with our `bank_account` prevalence.
- **Methods and claims (PDF page; printed page in brackets):**
  - Account ownership as "the fundamental measure of financial inclusion", and why it matters: p.27 [9].
  - **Definition of account ownership, including mobile money and an additional 3 percent identified through other responses:** p.33 [15]. Glossary definitions of account, financial institution account and mobile money account: p.197-199 [179-181].
  - Table B.1 adults with an account in 2021: Kenya 79, Tanzania 52, Uganda 66 percent: p.194, p.196 [176, 178]. Uganda rose from 20 to 66 percent between 2011 and 2021: p.34 [16]. Tanzania is among economies where more adults have only a mobile money account than have a financial institution account: p.38 [20].
  - Sub-Saharan Africa: 55 percent of adults with an account, 33 percent with a mobile money account: p.20 [2].
  - Kenya, Tanzania and Uganda are among 15 economies where 20 percent or more of adults have only a mobile money account; Uganda is the one with no significant overall gender gap: p.43 [25]. Income gaps in Kenya and Uganda: p.43 [25].
  - Rural-urban gaps: Kenya 14, Tanzania 13, Uganda 12 percentage points: p.47 [29]. Rural definitions are hard to compare across economies: p.46 [28]. Relevant to reading `location_type` across countries.
  - Barriers: documentation cited by 50 percent of the unbanked in Tanzania and Uganda: p.54 [36]; distance cited by 41 percent in Uganda: p.53 [35]; unbanked adults with no ID, Tanzania 8 million: p.56 [38].
  - Giving free bank accounts in Chile, Malawi and Uganda had no impact on savings or welfare, so access alone is not enough: p.28 [10]. A caution for any claim that a better classifier produces inclusion.
  - Mobile money saving in Kenya and Uganda: p.99 [81]. Mobile money borrowing in Kenya 30 and Uganda 16 percent: p.110 [92].
  - Consumer risk: a mystery-shopper audit of microfinance firms in Uganda: p.150 [132]; unexpected fees for wage recipients in Uganda and Kenya: p.172 [154]. Relevant to what outreach could lead to.
  - Survey method (Gallup World Poll, sampling, weights, design effect, margin of error): p.179-181 [161-163]; Table A.1 rows for Kenya, p.186 [168], Tanzania and Uganda, p.190 [172].
- **Reading note:** 27 pages have no text layer (p.2, 4, 16, 18, 24, 26, 30, 32, 48, 50, 66, 72, 94, 96, 104, 106, 114, 126, 128, 138, 148, 152, 162, 168, 170, 192, 203). They sit where blank versos and section-divider backs would fall, but I could not render them because `pdftoppm` is not installed, so this is an inference. Maps and charts were read only through their captions and surrounding text, so country values shown only inside figures are not available to cite.
- **Feeds:** background for the Introduction; RW (financial inclusion); D, L (label definition, mobile money, limits of access as an outcome, outreach harm). Not a source for any Rwanda statement.

