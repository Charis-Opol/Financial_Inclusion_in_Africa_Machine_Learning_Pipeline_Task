# Papers to download for Related Work

Source: Scopus export `search-results/export_6dfd1039-5455-4e9c-b282-5278999c5f01_2026-10-24T141433.644764813.csv` (1026 records), screened 2026-10-24.

**Total: 41 must and 40 should (81 papers).** The `#` column is the row index in the CSV, counted from 0 after the header.

**Screening rules.** A paper is kept if it adds to one of the three Related Work areas: (1) TabPFN and tabular foundation models, (2) Group DRO and robustness for the worst-off group, (3) ML for financial inclusion in Africa.

- **Must** means it is a core method paper, a direct comparison, or uses the same kind of data. Download these first.
- **Should** means it is supporting or contrasting work. Download if time allows.
- Excluded: TabPFN papers that only apply it to an unrelated field (about 600 records: concrete, medicine, soil, power grids and similar). Also excluded: DRO for control, reinforcement learning, recommender systems or power systems; vision-only work on spurious correlations; fraud and smishing detection; conference front-matter records; general fintech or business commentary.
- OA = the record is open access in Scopus. Most of the non-OA ML papers are also free on arXiv or OpenReview.

## 1. TabPFN and tabular foundation models

### Must (20)

| # | Title | Year | Venue | Cited | OA | DOI | Why |
|---|---|---|---|---|---|---|---|
| 861 | Accurate predictions on small data with a tabular foundation model | 2025 | Nature | 1041 | yes | [10.1038/s41586-024-08328-6](https://doi.org/10.1038/s41586-024-08328-6) | TabPFN v2 (Nature). The main model paper |
| 683 | TABPFN: A TRANSFORMER THAT SOLVES SMALL TABULAR CLASSIFICATION PROBLEMS IN A SECOND | 2023 | 11th International Conference on Learning Representations, I | 230 |  | [Scopus](https://www.scopus.com/pages/publications/85192253065?origin=resultslist) | Original TabPFN (ICLR 2023) |
| 1005 | Why do tree-based models still outperform deep learning on typical tabular data? | 2022 | Advances in Neural Information Processing Systems | 1880 |  | [Scopus](https://www.scopus.com/pages/publications/85163175556?origin=resultslist) | Why trees still beat deep learning on tabular data. Baseline framing vs XGBoost |
| 943 | Position: Why Tabular Foundation Models Should Be a Research Priority | 2024 | Proceedings of Machine Learning Research | 32 |  | [Scopus](https://www.scopus.com/pages/publications/85203821276?origin=resultslist) | Position paper: why tabular foundation models matter |
| 214 | The Current Generation of Tabular Foundation Models: A Critical Review | 2026 | Machine Learning and Knowledge Extraction | 0 | yes | [10.3390/make8080244](https://doi.org/10.3390/make8080244) | Critical review of current tabular foundation models |
| 511 | A Closer Look at TabPFN v2: Understanding Its Strengths and Extending Its Capabilities | 2025 | Advances in Neural Information Processing Systems | 7 |  | [Scopus](https://www.scopus.com/pages/publications/105048526872?origin=resultslist) | Close look at TabPFN v2: where it is strong, where it is weak |
| 532 | TabArena: A Living Benchmark for Machine Learning on Tabular Data | 2025 | Advances in Neural Information Processing Systems | 8 |  | [Scopus](https://www.scopus.com/pages/publications/105048557121?origin=resultslist) | TabArena living benchmark. Where TabPFN ranks |
| 731 | TabICL: A Tabular Foundation Model for In-Context Learning on Large Data | 2025 | Proceedings of Machine Learning Research | 11 |  | [Scopus](https://www.scopus.com/pages/publications/105023636763?origin=resultslist) | TabICL: in-context learning on large tables (alternative tabular foundation model) |
| 682 | TabDPT: Scaling Tabular Foundation Models on Real Data | 2025 | Advances in Neural Information Processing Systems | 7 |  | [Scopus](https://www.scopus.com/pages/publications/105048582839?origin=resultslist) | TabDPT: tabular foundation model pre-trained on real data |
| 852 | MITRA: Mixed Synthetic Priors for Enhancing Tabular Foundation Models | 2025 | Advances in Neural Information Processing Systems | 7 |  | [Scopus](https://www.scopus.com/pages/publications/105048579507?origin=resultslist) | Mitra: mixed synthetic priors |
| 817 | Retrieval & Fine-Tuning for In-Context Tabular Models | 2024 | Advances in Neural Information Processing Systems | 22 |  | [Scopus](https://www.scopus.com/pages/publications/105000477160?origin=resultslist) | Retrieval + fine-tuning for in-context models. Relevant because our train set has 23.5k rows |
| 809 | Drift-Resilient TabPFN: In-Context Learning Temporal Distribution Shifts on Tabular Data | 2024 | Advances in Neural Information Processing Systems | 21 |  | [Scopus](https://www.scopus.com/pages/publications/105000494217?origin=resultslist) | Drift-Resilient TabPFN: distribution shift |
| 743 | Evaluating Out-of-Distribution Robustness of TabPFN on Tabular Data | 2025 | 2025 International Conference on Advances in Technology and  | 0 |  | [10.1109/ICATC68823.2025.11407733](https://doi.org/10.1109/ICATC68823.2025.11407733) | Out-of-distribution robustness of TabPFN on Adult (subgroup splits) |
| 140 | Bridging ancestry gaps in genomic risk prediction with tabular foundation models | 2026 | Bioinformatics | 0 | yes | [10.1093/bioinformatics/btag217](https://doi.org/10.1093/bioinformatics/btag217) | Tabular foundation models with uneven groups (ancestry). Closest analogue to our worst-off-country question |
| 195 | Causal Pre-training Under the Fairness Lens: An Empirical Study of TabPFN | 2026 | WWW 2026 - Proceedings of the ACM Web Conference 2026 | 1 | yes | [10.1145/3774904.3792961](https://doi.org/10.1145/3774904.3792961) | TabPFN fairness and robustness. Only record that hits both area 1 and area 2 |
| 728 | Do Foundation Models Learn Fair Representations? A Critical Evaluation of TabPFN on Algorithmic Fairness Benchmarks | 2025 | CEUR Workshop Proceedings | 1 |  | [Scopus](https://www.scopus.com/pages/publications/105038743085?origin=resultslist) | Does TabPFN learn fair representations? |
| 993 | FairPFN: A Tabular Foundation Model for Causal Fairness | 2025 | Proceedings of Machine Learning Research | 1 |  | [Scopus](https://www.scopus.com/pages/publications/105023640350?origin=resultslist) | FairPFN: causal fairness with a tabular foundation model |
| 27 | Towards Fair In-Context Learning with Tabular Foundation Models | 2026 | Transactions on Machine Learning Research | 1 |  | [Scopus](https://www.scopus.com/pages/publications/105028328910?origin=resultslist) | Fair in-context learning with tabular foundation models |
| 851 | High Performance, Low Reliability: Uncertainty Benchmarking for Tabular Foundation Models | 2026 | ESANN 2026 Proceedings - 34th European Symposium on Artifici | 1 | yes | [10.14428/esann/2026.ES2026-261](https://doi.org/10.14428/esann/2026.ES2026-261) | Uncertainty benchmark: high performance, low reliability |
| 123 | Established machine learning matches tabular foundation models in clinical predictions | 2026 | BMC Medical Informatics and Decision Making | 0 | yes | [10.1186/s12911-026-03654-3](https://doi.org/10.1186/s12911-026-03654-3) | Counterpoint: established ML matches tabular foundation models in clinical prediction |

### Should (18)

| # | Title | Year | Venue | Cited | OA | DOI | Why |
|---|---|---|---|---|---|---|---|
| 87 | StablePFN: Stable Prediction with Causal-Aware Tabular Foundation Model | 2026 | Proceedings of the ACM SIGKDD International Conference on Kn | 0 | yes | [10.1145/3770855.3818030](https://doi.org/10.1145/3770855.3818030) | StablePFN: stable prediction across environments |
| 45 | Exploring Fine-Tuning for Tabular Foundation Models | 2026 | WWW 2026 - Proceedings of the ACM Web Conference 2026 | 0 | yes | [10.1145/3774904.3792923](https://doi.org/10.1145/3774904.3792923) | Fine-tuning tabular foundation models |
| 755 | TuneTables: Context Optimization for Scalable Prior-Data Fitted Networks | 2024 | Advances in Neural Information Processing Systems | 17 |  | [Scopus](https://www.scopus.com/pages/publications/105000490855?origin=resultslist) | TuneTables: context optimisation for scaling PFNs |
| 662 | TabPFN Unleashed: A Scalable and Effective Solution to Tabular Classification Problems | 2025 | Proceedings of Machine Learning Research | 5 |  | [Scopus](https://www.scopus.com/pages/publications/105023637896?origin=resultslist) | TabPFN Unleashed: scaling for classification |
| 768 | TABFLEX: Scaling Tabular Learning to Millions with Linear Attention | 2025 | Proceedings of Machine Learning Research | 0 |  | [Scopus](https://www.scopus.com/pages/publications/105023583881?origin=resultslist) | TabFlex: scaling to millions of rows |
| 933 | Statistical Foundations of Prior-Data Fitted Networks | 2023 | Proceedings of Machine Learning Research | 23 |  | [Scopus](https://www.scopus.com/pages/publications/85174400933?origin=resultslist) | Statistical foundations of PFNs (theory) |
| 880 | Interpretable Machine Learning for TabPFN | 2024 | Communications in Computer and Information Science | 15 | yes | [10.1007/978-3-031-63797-1_23](https://doi.org/10.1007/978-3-031-63797-1_23) | Interpretability for TabPFN. Links to the Week 3 XAI work |
| 61 | On the Uncertainty Quantification Ability of Tabular Foundation Models | 2026 | Computing in Science and Engineering | 0 |  | [10.1109/MCSE.2026.3701626](https://doi.org/10.1109/MCSE.2026.3701626) | Uncertainty quantification ability of tabular foundation models |
| 162 | Calibration, Architecture, and Distribution Shift in Predictive Uncertainty Estimation | 2026 | Machine Learning and Knowledge Extraction | 0 | yes | [10.3390/make8070179](https://doi.org/10.3390/make8070179) | Calibration and uncertainty under distribution shift (includes TabPFN) |
| 336 | On the Robustness of Tabular Foundation Models: Test-Time Attacks and In-Context Defenses | 2026 | Proceedings - 2026 IEEE Conference on Secure and Trustworthy | 1 | yes | [10.1109/SaTML68715.2026.00028](https://doi.org/10.1109/SaTML68715.2026.00028) | Robustness of tabular foundation models: test-time attacks |
| 826 | Navigating Severe Class Imbalance in Population Cohort Data | 2025 | Proceedings of the Annual International Conference of the IE | 3 |  | [10.1109/EMBC58623.2025.11254293](https://doi.org/10.1109/EMBC58623.2025.11254293) | Severe class imbalance in population cohort data |
| 427 | Class-Imbalance-Aware Adaptive Dataset Distillation for Scalable Pretrained Model in Credit Scoring | 2026 | Intelligent Systems in Accounting, Finance and Management | 0 |  | [10.1002/isaf.70042](https://doi.org/10.1002/isaf.70042) | Pretrained models for credit scoring under class imbalance |
| 132 | A TabPFN-Based Framework for Credit Risk Prediction in Automotive Green Supply Chain Finance | 2026 | Sustainability (Switzerland) | 0 | yes | [10.3390/su18126305](https://doi.org/10.3390/su18126305) | TabPFN for credit risk prediction |
| 495 | Intelligent Bank Customer Churn Prediction: An AI Ensemble Approach Using Neural Networks and TabPFN Foundation Models | 2026 | Proceedings of the 4th IEEE International Conference on Inte | 0 |  | [10.1109/IATMSI68868.2026.11465999](https://doi.org/10.1109/IATMSI68868.2026.11465999) | TabPFN for bank churn (finance domain) |
| 1009 | Revisiting Deep Learning Models for Tabular Data | 2021 | Advances in Neural Information Processing Systems | 1339 |  | [Scopus](https://www.scopus.com/pages/publications/85127045130?origin=resultslist) | Revisiting deep tabular models (FT-Transformer). Deep learning baseline context |
| 867 | Deep Neural Networks and Tabular Data: A Survey | 2024 | IEEE Transactions on Neural Networks and Learning Systems | 856 | yes | [10.1109/TNNLS.2022.3229161](https://doi.org/10.1109/TNNLS.2022.3229161) | Survey: deep neural networks for tabular data |
| 70 | Representation Learning for Tabular Data: A Comprehensive Survey | 2026 | IEEE Transactions on Pattern Analysis and Machine Intelligen | 16 |  | [10.1109/TPAMI.2026.3657217](https://doi.org/10.1109/TPAMI.2026.3657217) | Survey: representation learning for tabular data |
| 178 | TabTune: A Unified Library for Inference and Fine-Tuning Tabular Foundation Models | 2026 | WWW Companion 2026 - Companion Proceedings of the ACM Web Co | 0 | yes | [10.1145/3774905.3793137](https://doi.org/10.1145/3774905.3793137) | TabTune: library for tabular foundation model inference and fine-tuning |

## 2. Group DRO and worst-group robustness

### Must (12)

| # | Title | Year | Venue | Cited | OA | DOI | Why |
|---|---|---|---|---|---|---|---|
| 1008 | DISTRIBUTIONALLY ROBUST NEURAL NETWORKS FOR GROUP SHIFTS: ON THE IMPORTANCE OF REGULARIZATION FOR WORST-CASE GENERALIZATION | 2020 | 8th International Conference on Learning Representations, IC | 411 |  | [Scopus](https://www.scopus.com/pages/publications/85150613153?origin=resultslist) | Group DRO (Sagawa et al.). The core method paper |
| 1017 | Does Distributionally Robust Supervised Learning Give Robust Classifiers? | 2018 | Proceedings of Machine Learning Research | 124 |  | [Scopus](https://www.scopus.com/pages/publications/105020569458?origin=resultslist) | Hu et al.: does distributionally robust learning give robust classifiers? (#1023 is a duplicate; download one) |
| 1021 | Just Train Twice: Improving Group Robustness without Training Group Information | 2021 | Proceedings of Machine Learning Research | 416 |  | [Scopus](https://www.scopus.com/pages/publications/85161336604?origin=resultslist) | Just Train Twice: group robustness without group labels |
| 1001 | Environment Inference for Invariant Learning | 2021 | Proceedings of Machine Learning Research | 316 |  | [Scopus](https://www.scopus.com/pages/publications/85161305720?origin=resultslist) | Environment inference for invariant learning (EIIL) |
| 998 | Group Robust Classification Without Any Group Information | 2023 | Advances in Neural Information Processing Systems | 20 |  | [Scopus](https://www.scopus.com/pages/publications/85205560191?origin=resultslist) | Group-robust classification without any group information |
| 558 | UNDERSTANDING WHY GENERALIZED REWEIGHTING DOES NOT IMPROVE OVER ERM | 2023 | 11th International Conference on Learning Representations, I | 11 |  | [Scopus](https://www.scopus.com/pages/publications/85172243684?origin=resultslist) | Why generalised reweighting does not beat ERM. Key negative result |
| 922 | FOCUS ON THE COMMON GOOD: GROUP DISTRIBUTIONAL ROBUSTNESS FOLLOWS | 2022 | ICLR 2022 - 10th International Conference on Learning Repres | 13 |  | [Scopus](https://www.scopus.com/pages/publications/85150368314?origin=resultslist) | Common Gradient Descent: Group DRO variant |
| 724 | Distributionally Robust Optimization with Probabilistic Group | 2023 | Proceedings of the 37th AAAI Conference on Artificial Intell | 9 | yes | [10.1609/aaai.v37i10.26394](https://doi.org/10.1609/aaai.v37i10.26394) | DRO with probabilistic group membership |
| 1012 | An investigation of why overparameterization exacerbates spurious correlations | 2020 | 37th International Conference on Machine Learning, ICML 2020 | 189 |  | [Scopus](https://www.scopus.com/pages/publications/85105274786?origin=resultslist) | Why overparameterisation makes spurious correlations worse |
| 1024 | Fair transfer learning with missing protected attributes | 2019 | AIES 2019 - Proceedings of the 2019 AAAI/ACM Conference on A | 90 | yes | [10.1145/3306618.3314236](https://doi.org/10.1145/3306618.3314236) | Fair transfer learning under covariate shift, motivated by mobile money markets (bridges areas 2 and 3) |
| 966 | Mixed-feature Logistic Regression Robust to Distribution Shifts | 2025 | Proceedings of Machine Learning Research | 1 |  | [Scopus](https://www.scopus.com/pages/publications/105014320868?origin=resultslist) | Distribution-shift-robust logistic regression on mixed (tabular) features |
| 876 | Distributionally Robust Optimization and Invariant Representation Learning for Addressing Subgroup Underrepresentation: Mechanisms and Limitations | 2023 | Lecture Notes in Computer Science (including subseries Lectu | 1 |  | [10.1007/978-3-031-45249-9_18](https://doi.org/10.1007/978-3-031-45249-9_18) | DRO vs invariant learning for under-represented subgroups: mechanisms and limits |

### Should (13)

| # | Title | Year | Venue | Cited | OA | DOI | Why |
|---|---|---|---|---|---|---|---|
| 902 | BITRATE-CONSTRAINED DRO: BEYOND WORST CASE ROBUSTNESS TO UNKNOWN GROUP SHIFTS | 2023 | 11th International Conference on Learning Representations, I | 2 |  | [Scopus](https://www.scopus.com/pages/publications/85190345775?origin=resultslist) | Bitrate-constrained DRO for unknown group shifts |
| 611 | AGRO: ADVERSARIAL DISCOVERY OF ERROR-PRONE GROUPS FOR ROBUST OPTIMIZATION | 2023 | 11th International Conference on Learning Representations, I | 4 |  | [Scopus](https://www.scopus.com/pages/publications/85174788543?origin=resultslist) | AGRO: finds error-prone groups automatically |
| 706 | On Tilted Losses in Machine Learning: Theory and Applications | 2023 | Journal of Machine Learning Research | 31 |  | [Scopus](https://www.scopus.com/pages/publications/85176877477?origin=resultslist) | Tilted losses (TERM): an alternative to worst-case objectives |
| 508 | Minimax Regret Optimization for Robust Machine Learning under Distribution Shift | 2022 | Proceedings of Machine Learning Research | 26 |  | [Scopus](https://www.scopus.com/pages/publications/85164746457?origin=resultslist) | Minimax regret optimisation under distribution shift |
| 1003 | SELECTIVE CLASSIFICATION CAN MAGNIFY DISPARITIES ACROSS GROUPS | 2021 | ICLR 2021 - 9th International Conference on Learning Represe | 29 |  | [Scopus](https://www.scopus.com/pages/publications/85105103685?origin=resultslist) | Selective classification can widen gaps between groups |
| 761 | Towards Last-layer Retraining for Group Robustness with Fewer Annotations | 2023 | Advances in Neural Information Processing Systems | 38 |  | [Scopus](https://www.scopus.com/pages/publications/85177865256?origin=resultslist) | Last-layer retraining with fewer group labels |
| 106 | On the Unreasonable Effectiveness of Last-layer Retraining | 2026 | Transactions on Machine Learning Research | 0 |  | [Scopus](https://www.scopus.com/pages/publications/105039108484?origin=resultslist) | Why last-layer retraining works so well |
| 898 | Improving Out-of-Distribution Robustness via Selective Augmentation | 2022 | Proceedings of Machine Learning Research | 144 |  | [Scopus](https://www.scopus.com/pages/publications/85163108323?origin=resultslist) | LISA: selective augmentation for out-of-distribution robustness |
| 936 | Advancing Equal Opportunity Fairness and Group Robustness through Group-Level Cost-Sensitive Deep Learning | 2025 | Baltic Journal of Modern Computing | 2 | yes | [10.22364/bjmc.2025.13.1.06](https://doi.org/10.22364/bjmc.2025.13.1.06) | Group-level cost-sensitive learning for fairness and group robustness |
| 824 | f-FERM: A SCALABLE FRAMEWORK FOR ROBUST FAIR EMPIRICAL RISK MINIMIZATION | 2024 | 12th International Conference on Learning Representations, I | 5 |  | [Scopus](https://www.scopus.com/pages/publications/85200510124?origin=resultslist) | f-FERM: robust fair empirical risk minimisation |
| 536 | Preserving the Fairness Guarantees of Classifiers in Changing Environments: A Survey | 2025 | ACM Computing Surveys | 8 | yes | [10.1145/3637438](https://doi.org/10.1145/3637438) | Survey: keeping fairness guarantees when the environment changes |
| 615 | Domain Generalization of Deep Learning Models Under Subgroup Shift in Breast Cancer Prognosis | 2024 | 21st IEEE Conference on Computational Intelligence in Bioinf | 2 |  | [10.1109/CIBCB58642.2024.10702166](https://doi.org/10.1109/CIBCB58642.2024.10702166) | Group DRO applied to subgroup shift in non-image data (gene expression) |
| 929 | Evaluating Model Bias Requires Characterizing its Mistakes | 2024 | Proceedings of Machine Learning Research | 1 |  | [Scopus](https://www.scopus.com/pages/publications/85203848628?origin=resultslist) | Evaluating model bias means characterising its mistakes |

## 3. ML for financial inclusion in Africa

### Must (9)

| # | Title | Year | Venue | Cited | OA | DOI | Why |
|---|---|---|---|---|---|---|---|
| 1016 | Using Machine Learning Algorithms to Predict the State of Financial Inclusion in Africa | 2021 | 2021 12th International Conference on Information and Commun | 13 |  | [10.1109/ICICS52457.2021.9464590](https://doi.org/10.1109/ICICS52457.2021.9464590) | ML to predict financial inclusion in Africa. Likely uses the same Zindi dataset |
| 816 | An Efficient Approach for Predicting the state of Financial Inclusion by Addressing Class Imbalance using the Oversampling Technique | 2022 | 2022 2nd International Conference on Computer Science, Engin | 2 |  | [10.1109/ICCSEA54677.2022.9936452](https://doi.org/10.1109/ICCSEA54677.2022.9936452) | Same Zindi dataset, handles class imbalance with oversampling |
| 612 | Modeling drivers of fintech adoption in Sub-Saharan Africa using machine learning | 2025 | Digital Finance | 1 | yes | [10.1007/s42521-025-00161-4](https://doi.org/10.1007/s42521-025-00161-4) | Findex data for 36 Sub-Saharan African countries: drivers of fintech adoption (ML vs logit/probit) |
| 1020 | Financial inclusion: Measures and applications to Africa | 2021 | Economic Analysis and Policy | 54 | yes | [10.1016/j.eap.2021.03.008](https://doi.org/10.1016/j.eap.2021.03.008) | Measuring financial inclusion in Africa with unsupervised ML |
| 526 | Financial Inclusion in Sub-Saharan Emerging Markets: The Application of Deep Learning to Improve Determinants | 2023 | Communications in Computer and Information Science | 0 |  | [10.1007/978-3-031-49002-6_15](https://doi.org/10.1007/978-3-031-49002-6_15) | Deep learning on financial inclusion household surveys (Rwanda among others) |
| 47 | Evaluating the Non-linear Impact of Digital Literacy on Financial Inclusion: Evidence from Emerging Economies via Explainable Deep Learning | 2026 | International Journal of Computational Intelligence Systems | 0 | yes | [10.1007/s44196-026-01415-8](https://doi.org/10.1007/s44196-026-01415-8) | Explainable deep learning: digital literacy and financial inclusion |
| 760 | Integrated data-driven credit default prediction in Uganda using machine learning models | 2025 | Journal of the Nigerian Society of Physical Sciences | 1 | yes | [10.46481/jnsps.2026.2649](https://doi.org/10.46481/jnsps.2026.2649) | Credit default prediction for unbanked people in Uganda |
| 1002 | The Unbanked and Poverty: Predicting area-level socio-economic vulnerability from M-Money transactions | 2018 | Proceedings - 2018 IEEE International Conference on Big Data | 11 | yes | [10.1109/BigData.2018.8622268](https://doi.org/10.1109/BigData.2018.8622268) | Predicting poverty and unbanked areas from mobile money data |
| 1018 | (Dis)crediting claims of financial inclusion: The integration of artificial intelligence in consumer credit markets in the united states and kenya | 2021 | Journal of International and Comparative Law | 15 |  | [Scopus](https://www.scopus.com/pages/publications/85119605033?origin=resultslist) | Critique of AI-driven financial inclusion claims in Kenya. Useful for the ethics framing |

### Should (9)

| # | Title | Year | Venue | Cited | OA | DOI | Why |
|---|---|---|---|---|---|---|---|
| 1000 | A comparative analysis of machine learning models for the prediction of insurance uptake in kenya | 2021 | Data | 12 | yes | [10.3390/data6110116](https://doi.org/10.3390/data6110116) | ML for insurance uptake in Kenya |
| 767 | Fintech Adoption for Poverty Alleviation in African Countries: Application of Supervised Machine Learning Approach | 2022 | Contributions to Management Science | 5 | yes | [10.1007/978-3-031-17008-9_10](https://doi.org/10.1007/978-3-031-17008-9_10) | Supervised ML for fintech adoption and poverty in African countries |
| 517 | Predictive Machine Learning Model for the Role of Financial Inclusion in Poverty Reduction in Rwanda | 2025 | African Journal of Business and Economic Research | 0 |  | [10.31920/1750-4562/2025/v20n2a12](https://doi.org/10.31920/1750-4562/2025/v20n2a12) | Financial inclusion and poverty in Rwanda (ML) |
| 778 | Credit Scoring in Africa: Employing Logistic Regression on Alternative Data | 2024 | International Conference on Electrical, Computer, Communicat | 3 |  | [10.1109/ICECCME62383.2024.10796011](https://doi.org/10.1109/ICECCME62383.2024.10796011) | Credit scoring for the unbanked in Africa using alternative data |
| 727 | Empowering Africa’s Disfranchised SMEs: Machine Learning-Based Credit Scoring for Informal African Merchants | 2025 | Human-Centric Intelligent Systems | 2 | yes | [10.1007/s44230-025-00105-6](https://doi.org/10.1007/s44230-025-00105-6) | Credit scoring for informal African merchants |
| 793 | Tausi: A Holistic Artificial Intelligence Approach to Credit Scoring Using Informal Data for a Sustainable Micro-lending African Economy | 2024 | Lecture Notes in Networks and Systems | 2 |  | [10.1007/978-3-031-73110-5_5](https://doi.org/10.1007/978-3-031-73110-5_5) | Tausi: credit scoring from informal data |
| 964 | Prediction of SACCOS Failure in Tanzania using Machine Learning Models | 2024 | Engineering, Technology and Applied Science Research | 5 | yes | [10.48084/etasr.6696](https://doi.org/10.48084/etasr.6696) | SACCO failure prediction in Tanzania |
| 1006 | A user-generated data based approach to enhancing location prediction of financial services in sub-Saharan Africa | 2019 | Applied Geography | 9 | yes | [10.1016/j.apgeog.2019.02.005](https://doi.org/10.1016/j.apgeog.2019.02.005) | Locating financial services in Sub-Saharan Africa |
| 1014 | AI Opportunities in the African Financial Sector: Use Cases | 2021 | The AI Book: The Artificial Intelligence Handbook for Invest | 0 |  | [10.1002/9781119551966.ch18](https://doi.org/10.1002/9781119551966.ch18) | AI opportunities in the African financial sector |

## Gap worth stating in Related Work

Only one record (#195) covers both TabPFN and fairness/robustness. No record combines tabular foundation models with Group DRO, and none studies either one on financial inclusion data. That is the gap this project fills.
