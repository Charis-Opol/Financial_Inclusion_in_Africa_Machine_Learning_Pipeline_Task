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
  // Points at Week 2 until the results table gets its own #results anchor in Phase 3.
  secondary: { label: "Jump to results", href: "#week-2" },
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
