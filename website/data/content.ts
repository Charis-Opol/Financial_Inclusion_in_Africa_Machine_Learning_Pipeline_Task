/**
 * Single source of truth for all page copy and numbers.
 * Edit text here; components only render what they are given.
 *
 * Phase 0 holds site metadata and the section skeleton only.
 * Later phases add their own typed blocks below.
 */

export type SiteMeta = {
  title: string;
  shortTitle: string;
  description: string;
  repoUrl: string;
  author: string;
};

export type SectionId =
  | "dataset"
  | "week-1"
  | "week-2"
  | "week-3"
  | "reproducibility"
  | "limitations"
  | "stack";

export type SectionMeta = {
  id: SectionId;
  /** Short label for the sticky nav. */
  navLabel: string;
  /** Small uppercase label above the heading. */
  eyebrow: string;
  title: string;
};

export const site: SiteMeta = {
  title: "Financial Inclusion in Africa: End-to-End ML Pipeline",
  shortTitle: "Financial Inclusion ML",
  description:
    "A reproducible, SOLID-principle ML pipeline predicting bank account ownership across Kenya, Rwanda, Tanzania and Uganda, taken through to a containerised FastAPI service with ONNX export and drift monitoring.",
  repoUrl:
    "https://github.com/Charis-Opol/Financial_Inclusion_in_Africa_Machine_Learning_Pipeline_Task",
  author: "Charis Opol",
};

export const sections: SectionMeta[] = [
  { id: "dataset", navLabel: "Dataset", eyebrow: "01 · Context", title: "The problem and dataset" },
  { id: "week-1", navLabel: "Week 1", eyebrow: "02 · Week 1", title: "Understanding the data and preprocessing" },
  { id: "week-2", navLabel: "Week 2", eyebrow: "03 · Week 2", title: "Modeling, evaluation and ablation" },
  { id: "week-3", navLabel: "Week 3", eyebrow: "04 · Week 3", title: "Serving, deployment and monitoring" },
  { id: "reproducibility", navLabel: "Reproduce", eyebrow: "05 · Reproducibility", title: "Reproducibility" },
  { id: "limitations", navLabel: "Limits", eyebrow: "06 · Honesty notes", title: "Known limitations" },
  { id: "stack", navLabel: "Stack", eyebrow: "07 · Tools", title: "Tech stack" },
];

/* ------------------------------------------------------------------ */
/* Hero                                                                */
/* ------------------------------------------------------------------ */

export type Stat = {
  value: string;
  label: string;
};

export type LinkButton = {
  label: string;
  href: string;
  external?: boolean;
};

export type Hero = {
  eyebrow: string;
  /** Title split into lines; the last line is set in the accent italic. */
  titleLines: [string, string];
  subtitle: string;
  primary: LinkButton;
  secondary: LinkButton;
  stats: Stat[];
};

export const hero: Hero = {
  eyebrow: "Kenya · Rwanda · Tanzania · Uganda",
  titleLines: ["Financial Inclusion in Africa:", "End-to-End ML Pipeline"],
  subtitle:
    "A reproducible, SOLID-principle ML pipeline predicting bank account ownership across Kenya, Rwanda, Tanzania and Uganda. Built for honest evaluation over leaderboard-chasing, and taken through to a containerised FastAPI service with ONNX export and drift monitoring.",
  primary: { label: "View on GitHub", href: site.repoUrl, external: true },
  secondary: { label: "Jump to results", href: "#results" },
  stats: [
    { value: "23,524", label: "training rows" },
    { value: "4", label: "countries" },
    { value: "14.1%", label: "positive class" },
    { value: "0.590", label: "PR-AUC" },
    { value: "<1 ms", label: "inference" },
  ],
};

/* ------------------------------------------------------------------ */
/* 01 · The problem and dataset                                        */
/* ------------------------------------------------------------------ */

export type CountryOwnership = {
  country: string;
  /** Share of respondents with a bank account, in percent. */
  ownership: number;
  surveyYear: number;
  /** Drawn in the accent colour; the rest are context. */
  highlight?: boolean;
};

export type DatasetSection = {
  brief: string;
  source: { name: string; detail: string };
  splits: { label: string; rows: string; columns: string; note?: string }[];
  target: {
    name: string;
    classes: { label: string; share: number }[];
    note: string;
  };
  whyTitle: string;
  why: string[];
  chart: {
    title: string;
    caption: string;
    data: CountryOwnership[];
  };
};

export const dataset: DatasetSection = {
  brief:
    "The brief: design a complete end-to-end ML pipeline on a Uganda/Africa-relevant dataset, with clean reproducible code, proper evaluation methodology and solid documentation.",
  source: {
    name: "Financial Inclusion in Africa",
    detail: "Zindi / World Bank Findex-derived household surveys",
  },
  splits: [
    { label: "Train", rows: "23,524", columns: "13" },
    { label: "Test", rows: "10,086", columns: "12", note: "target held out" },
  ],
  target: {
    name: "bank_account",
    classes: [
      { label: "No", share: 85.9 },
      { label: "Yes", share: 14.1 },
    ],
    note: "Moderate-severe imbalance.",
  },
  whyTitle: "Why this dataset",
  why: [
    "Uganda-relevant.",
    "Tabular, which suits both XGBoost and a PyTorch MLP.",
    "A binary target with realistic imbalance.",
    "Rich categorical structure.",
  ],
  chart: {
    title: "Account ownership by country",
    caption:
      "Share of survey respondents with a bank account. Survey years differ by country: Kenya 2018, Rwanda 2016, Tanzania 2017, Uganda 2018.",
    data: [
      { country: "Kenya", ownership: 25.1, surveyYear: 2018 },
      { country: "Rwanda", ownership: 11.5, surveyYear: 2016 },
      { country: "Tanzania", ownership: 9.2, surveyYear: 2017 },
      { country: "Uganda", ownership: 8.6, surveyYear: 2018, highlight: true },
    ],
  },
};

/* ------------------------------------------------------------------ */
/* 02–04 · Weeks 1–3                                                   */
/* Text in `backticks` is rendered as inline code.                     */
/* ------------------------------------------------------------------ */

/** Short descriptions for the side stepper, keyed by week section id. */
export const weekSteps: Record<"week-1" | "week-2" | "week-3", string> = {
  "week-1": "Data and preprocessing",
  "week-2": "Modeling and evaluation",
  "week-3": "Serving and monitoring",
};

export type Finding = {
  title: string;
  finding: string;
  decision: string;
};

/** Either one decision for both model families, or one per family. */
export type PreprocessingRow =
  | { concern: string; shared: string }
  | { concern: string; xgboost: string; pytorch: string };

export type SolidPrinciple = {
  letter: "S" | "O" | "L" | "I" | "D";
  name: string;
  line: string;
};

export type Week1Content = {
  intro: string;
  findingsTitle: string;
  findings: Finding[];
  preprocessingTitle: string;
  preprocessing: PreprocessingRow[];
  solidTitle: string;
  solid: SolidPrinciple[];
};

export const week1: Week1Content = {
  intro:
    "Every exploratory finding was turned into an explicit, documented preprocessing decision before any model was trained.",
  findingsTitle: "EDA findings → decisions",
  findings: [
    {
      title: "The ID is not a key",
      finding:
        "`uniqueid` is not globally unique; it repeats across country/year. The real key is `uniqueid + country + year` (0 duplicates).",
      decision: "Documented in the loader docstring so nobody joins on `uniqueid` alone.",
    },
    {
      title: "Missingness in disguise",
      finding:
        "No literal NaNs, but placeholder categories: `Dont know` in `marital_status` (8 rows), `Other/Dont know/RTA` in `education_level` (35 rows) and `Dont Know/Refuse to answer` in `job_type` (126 rows).",
      decision: "Kept as their own category level.",
    },
    {
      title: "A strong country effect",
      finding:
        "Four separate national surveys stacked together. Ownership varies ~3× by country, plausibly driven by Kenya’s mobile-money (M-Pesa) penetration.",
      decision:
        "One pooled model with `country` as a feature; CV stratified on country + target jointly; all metrics reported both pooled and per-country, since Uganda is the quantity of interest.",
    },
    {
      title: "Encoding by cardinality",
      finding: "`job_type` has 10 levels and `relationship_with_head` has 6.",
      decision: "One-hot for XGBoost; learned embeddings for the PyTorch MLP.",
    },
    {
      title: "Possible redundancy",
      finding: "`relationship_with_head` and `marital_status` might encode the same thing.",
      decision: "A crosstab showed partial, not total, overlap, so no combined feature was engineered.",
    },
    {
      title: "Household-size outliers",
      finding: "8 rows have `household_size` > 15.",
      decision: "Flagged rather than blindly capped.",
    },
    {
      title: "Train/test consistency",
      finding: "Category levels must line up between the training and test files.",
      decision: "Encoders fit on train only, with an explicit “unknown” bucket.",
    },
  ],
  preprocessingTitle: "Preprocessing decisions",
  preprocessing: [
    { concern: "Missing-in-disguise", shared: "Kept as its own category" },
    { concern: "Categorical encoding", xgboost: "One-hot", pytorch: "Learned embeddings" },
    { concern: "Numeric features", xgboost: "Raw", pytorch: "Scaled" },
    { concern: "Outliers", shared: "Flagged, not capped" },
    { concern: "Country", shared: "Kept as a pooled feature and as a CV stratifier" },
    {
      concern: "Class imbalance",
      shared: "Class-weighting vs SMOTE, tested as an ablation axis",
    },
  ],
  solidTitle: "SOLID mapping",
  solid: [
    {
      letter: "S",
      name: "Single responsibility",
      line: "Data loading, cleaning, encoding, modeling, tuning and evaluation are separate modules.",
    },
    {
      letter: "O",
      name: "Open/closed",
      line: "Models implement a shared `BaseModel` interface (`fit`, `predict_proba`, `get_params`).",
    },
    {
      letter: "L",
      name: "Liskov substitution",
      line: "Any `BaseModel` (XGBoost, LR, PyTorch MLP) is swappable inside the CV loop.",
    },
    {
      letter: "I",
      name: "Interface segregation",
      line: "Preprocessors expose only sklearn-style `fit`/`transform`.",
    },
    {
      letter: "D",
      name: "Dependency inversion",
      line: "Orchestration depends on abstractions (`BaseModel`, `BasePreprocessor`) injected via config.",
    },
  ],
};

/* ---- Week 2 ------------------------------------------------------- */

export type MeanStd = { mean: number; std: number };

export type ModelEntry = {
  name: string;
  role: string;
  detail: string;
};

export type ValidationRule = string;

export type MetricNote = {
  name: string;
  /** Marks the headline metric. */
  headline?: boolean;
  note: string;
};

export type ResultRow = {
  id: string;
  model: string;
  imbalance: string;
  prAuc: MeanStd;
  f1: MeanStd;
  rocAuc: MeanStd;
  /** The shipped model; highlighted in the table and chart. */
  winner?: boolean;
};

export type Week2Content = {
  intro: string;
  modelsTitle: string;
  models: ModelEntry[];
  imbalance: string;
  validationTitle: string;
  validation: ValidationRule[];
  diagram: { title: string; description: string; rows: string };
  metricsTitle: string;
  metrics: MetricNote[];
  metricsFootnote: string;
  accuracyTrap: string;
  resultsTitle: string;
  resultsIntro: string;
  results: ResultRow[];
  /** Reference line on the PR-AUC chart. */
  floor: { label: string; prAuc: MeanStd };
  chart: { title: string; caption: string };
  shipped: {
    title: string;
    body: string[];
    stats: { label: string; value: string; note?: string }[];
  };
};

export const week2: Week2Content = {
  intro:
    "Two model families, two ways of handling imbalance, and a validation scheme where no reported number has been seen by tuning.",
  modelsTitle: "Models and imbalance handling",
  models: [
    { name: "XGBoost", role: "Primary baseline", detail: "Gradient-boosted trees on one-hot features." },
    { name: "Logistic Regression", role: "Sanity floor", detail: "PR-AUC 0.559 ± 0.019." },
    {
      name: "PyTorch MLP",
      role: "Neural comparison",
      detail: "2-layer untuned → tuned 2-layer → 3-layer ablation.",
    },
  ],
  imbalance: "Class-weighting vs SMOTE, applied to both model families.",
  validationTitle: "Validation: nested stratified 5-fold CV",
  validation: [
    "Folds are stratified jointly on country and target.",
    "Outer folds are never touched by tuning.",
    "The inner loop uses Optuna to search hyperparameters.",
    "SMOTE is fit only inside training folds, to avoid leakage.",
  ],
  diagram: {
    title: "Nested cross-validation",
    description:
      "Five outer folds each hold out one fifth of the data as an untouched validation fold. Inside each outer fold, an inner stratified CV loop runs Optuna on the four training folds only, the model is refit with the best parameters, and it is scored once on the held-out fold. Repeating this for all five folds gives a mean and standard deviation.",
    rows: "23,524 training rows",
  },
  metricsTitle: "What is measured, and why",
  metrics: [
    {
      name: "PR-AUC",
      headline: true,
      note: "The headline metric. ROC-AUC is flattered when only ~14% of rows are positive; PR-AUC is not.",
    },
    { name: "F1", note: "Taken at a tuned decision threshold, not 0.5." },
    { name: "Precision and recall", note: "Reported separately, not only through F1." },
    { name: "ROC-AUC", note: "Included for comparability with other work." },
  ],
  metricsFootnote: "Every metric is reported as mean ± std over the 5 outer folds.",
  accuracyTrap:
    "Always predicting “No” gives ~86% accuracy with zero true positives, which is why accuracy is not optimised.",
  resultsTitle: "Results",
  resultsIntro: "Mean ± std over 5 outer folds. The shipped configuration is highlighted.",
  results: [
    {
      id: "xgb-cw",
      model: "XGBoost",
      imbalance: "class-weight",
      prAuc: { mean: 0.59, std: 0.019 },
      f1: { mean: 0.547, std: 0.016 },
      rocAuc: { mean: 0.865, std: 0.003 },
      winner: true,
    },
    {
      id: "xgb-smote",
      model: "XGBoost",
      imbalance: "SMOTE",
      prAuc: { mean: 0.577, std: 0.017 },
      f1: { mean: 0.537, std: 0.018 },
      rocAuc: { mean: 0.859, std: 0.004 },
    },
    {
      id: "mlp2-cw",
      model: "PyTorch 2-layer",
      imbalance: "class-weight",
      prAuc: { mean: 0.584, std: 0.019 },
      f1: { mean: 0.546, std: 0.019 },
      rocAuc: { mean: 0.864, std: 0.006 },
    },
    {
      id: "mlp2-smote",
      model: "PyTorch 2-layer",
      imbalance: "SMOTE",
      prAuc: { mean: 0.568, std: 0.018 },
      f1: { mean: 0.525, std: 0.018 },
      rocAuc: { mean: 0.852, std: 0.008 },
    },
    {
      id: "mlp3-cw",
      model: "PyTorch 3-layer",
      imbalance: "class-weight",
      prAuc: { mean: 0.581, std: 0.019 },
      f1: { mean: 0.542, std: 0.014 },
      rocAuc: { mean: 0.862, std: 0.006 },
    },
  ],
  floor: { label: "Logistic Regression floor", prAuc: { mean: 0.559, std: 0.019 } },
  chart: {
    title: "PR-AUC by configuration",
    caption:
      "Bars show the mean over 5 outer folds; whiskers show ± one standard deviation. The dashed line is the Logistic Regression floor.",
  },
  shipped: {
    title: "Shipped: XGBoost + class-weight",
    body: [
      "It had the best mean PR-AUC and the tied-tightest std, though it is not statistically distinguishable from the best PyTorch model at n=5 folds. It was chosen on PR-AUC stability, Uganda per-country fairness and interpretability, not a bare “highest wins” rule.",
      "SMOTE underperformed class-weighting for both model families. SHAP is used to explain the shipped model; the expected top features are `cellphone_access`, `education_level` and `country`.",
    ],
    stats: [
      { label: "Paired t-test vs PyTorch-best", value: "p = 0.087" },
      { label: "Wilcoxon vs PyTorch-best", value: "p = 0.125" },
      {
        label: "Uganda PR-AUC, the weakest segment",
        value: "0.553 ± 0.047",
        note: "2,101 training rows",
      },
    ],
  },
};
