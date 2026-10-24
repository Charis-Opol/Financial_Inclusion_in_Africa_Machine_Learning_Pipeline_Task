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
  { id: "week-1", navLabel: "Week 1", eyebrow: "02 · Week 1", title: "Data processing and model evaluation" },
  { id: "week-2", navLabel: "Week 2", eyebrow: "03 · Week 2", title: "Deployment and MLOps" },
  { id: "week-3", navLabel: "Week 3", eyebrow: "04 · Week 3", title: "Interpretability and explainability" },
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
  "week-1": "Data and evaluation",
  "week-2": "Deployment and MLOps",
  "week-3": "Interpretability",
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
    "Every exploratory finding was turned into an explicit, documented preprocessing decision before any model was trained. Models were then compared under nested cross-validation, so no reported number has been seen by tuning.",
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

/* ---- Week 1, part 2: modeling and evaluation ---------------------- */

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

export type ModelingContent = {
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

export const modeling: ModelingContent = {
  intro:
    "With preprocessing fixed, two model families and two ways of handling imbalance were compared under one validation scheme.",
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

/* ---- Week 2: deployment and MLOps -------------------------------- */

export type KeyValue = { label: string; value: string; note?: string };

export type StatusCode = { code: string; name: string; meaning: string };

export type BenchmarkRow = {
  batch: string;
  native: string;
  onnx: string;
  speedup: string;
  /** Which runtime was faster at this batch size. */
  faster: "native" | "onnx";
};

export type DriftLevel = "stable" | "moderate" | "significant";

export type PsiBand = { level: DriftLevel; range: string; label: string };

export type DriftScenario = {
  scenario: string;
  level: DriftLevel;
  verdict: string;
  evidence: string;
};

export type Week2Content = {
  quote: { text: string; detail: string };
  artifact: {
    title: string;
    summary: string;
    files: string[];
    facts: KeyValue[];
    gate: string;
  };
  api: {
    title: string;
    endpoints: string[];
    contract: string[];
    statusCodes: StatusCode[];
    checksTitle: string;
    checks: string[];
    sampleLabel: string;
    sample: string;
    sampleNote: string;
  };
  docker: { title: string; items: { label: string; detail: string }[] };
  onnx: {
    title: string;
    facts: KeyValue[];
    benchmarkCaption: string;
    benchmark: BenchmarkRow[];
    takeaway: string;
  };
  monitoring: {
    title: string;
    log: string;
    psi: string;
    bands: PsiBand[];
    exitCodesCaption: string;
    exitCodes: { code: string; meaning: string }[];
    scenariosCaption: string;
    scenarios: DriftScenario[];
  };
  tradeoffs: {
    title: string;
    intro: string;
    factors: { label: string; detail: string }[];
    recommendation: { title: string; points: string[] };
    diagram: {
      title: string;
      description: string;
      boundary: string;
      cloud: { title: string; lines: string[] };
      hub: { title: string; lines: string[] };
      edge: { title: string; lines: string[] };
      cloudLink: string;
      edgeLink: string;
    };
  };
};

export const week2: Week2Content = {
  quote: {
    text: "No silent failures.",
    detail:
      "Every artifact is verified before it is trusted, and every failure surfaces as a specific HTTP status, log line or exit code.",
  },
  artifact: {
    title: "Production artifact",
    summary:
      "The winning configuration is refit once, deterministically, and written out with everything needed to verify it later.",
    files: ["model.json", "metadata.json"],
    facts: [
      { label: "Refit", value: "seed 42", note: "~20 s" },
      {
        label: "Decision threshold",
        value: "0.4551",
        note: "Median of the 5 per-fold F1-optimal thresholds",
      },
      { label: "Version", value: "xgb-cw-5ec2d564be13" },
      { label: "Integrity", value: "SHA-256", note: "Checksums recorded in the metadata" },
    ],
    gate:
      "Nothing is written unless two parity gates pass: the numpy serving encoder must match the sklearn pipeline bit-for-bit on all 23,524 rows.",
  },
  api: {
    title: "FastAPI service",
    endpoints: ["POST /predict", "GET /health"],
    contract: [
      "Strict Pydantic contract: 10 raw fields, strict ints, `extra=\"forbid\"`.",
      "Rejects NaN/Infinity and unseen categories.",
    ],
    statusCodes: [
      { code: "422", name: "Invalid input", meaning: "The request broke the contract." },
      { code: "503", name: "Not ready", meaning: "Retryable; the response sends `Retry-After`." },
      { code: "500", name: "Server-side failure", meaning: "A failure on the service’s side, not the caller’s." },
    ],
    checksTitle: "Load-time integrity checks",
    checks: ["Files present", "SHA-256 match", "Feature order", "Schema vs vocabulary", "Smoke prediction"],
    sampleLabel: "POST /predict",
    sample: `POST /predict
{"country":"Uganda","location_type":"Rural","cellphone_access":"Yes",
 "household_size":5,"age_of_respondent":34,"gender_of_respondent":"Female",
 "relationship_with_head":"Spouse","marital_status":"Married/Living together",
 "education_level":"Primary education","job_type":"Self employed"}
→ {"probability_bank_account":0.180,"predicted_bank_account":"No", ...}`,
    sampleNote: "The score is a ranking score, not a calibrated probability.",
  },
  docker: {
    title: "Docker hardening",
    items: [
      { label: "Multi-stage build", detail: "`python:3.13-slim` base" },
      { label: "Non-root user", detail: "UID 10001" },
      { label: "Read-only root filesystem", detail: "The container cannot modify its own files" },
      { label: "No capabilities", detail: "`cap_drop: ALL`" },
      { label: "Healthcheck", detail: "Built into the container definition" },
      { label: "Localhost only", detail: "Bound to 127.0.0.1" },
      { label: "Resource limits", detail: "1 CPU / 512 MB (~76 MiB measured)" },
      { label: "Slim XGBoost", detail: "`xgboost-cpu`: 5.7 MB vs 99 MB" },
      { label: "Allow-list .dockerignore", detail: "Raw survey data can never enter the image" },
    ],
  },
  onnx: {
    title: "ONNX export",
    facts: [
      { label: "Opset", value: "15" },
      { label: "Model size", value: "0.81 MB" },
      { label: "Max |Δp|", value: "2.1e-7", note: "across 33,654 rows" },
      { label: "Yes/No flips", value: "0" },
    ],
    benchmarkCaption: "p50 latency in the 1-CPU container",
    benchmark: [
      { batch: "1", native: "519 µs", onnx: "33 µs", speedup: "15.7×", faster: "onnx" },
      { batch: "32", native: "679 µs", onnx: "399 µs", speedup: "1.7×", faster: "onnx" },
      { batch: "1024", native: "5.18 ms", onnx: "11.38 ms", speedup: "0.46× (slower)", faster: "native" },
    ],
    takeaway:
      "ONNX wins on single requests and native wins on big batches. Both are under 1 ms, so the API keeps native XGBoost and ONNX is reserved as an optional backend and edge format.",
  },
  monitoring: {
    title: "Monitoring and drift",
    log: "A non-blocking JSON-lines prediction log. Dropped events are counted and shown in `/health`, never silent; an unwritable log means 503.",
    psi: "A PSI drift check runs per feature and on the model score, against per-country reference profiles.",
    bands: [
      { level: "stable", range: "< 0.10", label: "Stable" },
      { level: "moderate", range: "0.10–0.25", label: "Moderate" },
      { level: "significant", range: "> 0.25", label: "Significant" },
    ],
    exitCodesCaption: "Drift CLI exit codes",
    exitCodes: [
      { code: "0", meaning: "Stable" },
      { code: "10", meaning: "Moderate drift" },
      { code: "20", meaning: "Significant drift" },
      { code: "30", meaning: "Insufficient data" },
      { code: "1", meaning: "Check failed" },
    ],
    scenariosCaption: "Drift scenarios",
    scenarios: [
      {
        scenario: "2,000 held-out test respondents",
        level: "stable",
        verdict: "Stable (exit 0)",
        evidence: "Largest PSI 0.009",
      },
      {
        scenario: "2,000 rural, low-education respondents",
        level: "significant",
        verdict: "Significant (exit 20)",
        evidence: "`location_type` PSI 3.42; predicted-Yes 20.6% → 5.3%",
      },
      {
        scenario: "600 Ugandan rural-outreach respondents",
        level: "significant",
        verdict: "Significant (exit 20)",
        evidence: "Predicted-Yes 10.9% → 0.2%",
      },
    ],
  },
  tradeoffs: {
    title: "Deployment trade-offs for Uganda",
    intro: "Cloud, edge and on-prem options were weighed against four constraints.",
    factors: [
      { label: "Rural connectivity", detail: "Field use cannot assume a reliable connection." },
      {
        label: "Data Protection and Privacy Act 2019",
        detail: "No major cloud provider has a region in Uganda.",
      },
      { label: "Cost", detail: "Capex vs USD-billed opex." },
      { label: "Maintenance burden", detail: "Someone has to keep each option running." },
    ],
    recommendation: {
      title: "Recommendation: hybrid",
      points: [
        "An in-country server or Kampala colocation hub is the system of record and runs a nightly Uganda drift check.",
        "Offline ONNX scoring runs on field devices for rural outreach.",
        "Cloud is used only for work involving no personal data.",
      ],
    },
    diagram: {
      title: "Hybrid deployment: hub, edge and cloud",
      description:
        "Inside Uganda, a hub (in-country server or Kampala colocation) is the system of record and runs a nightly Uganda drift check; field devices score offline with the ONNX model for rural outreach. Outside Uganda, the cloud is used only for work involving no personal data.",
      boundary: "IN UGANDA · DATA PROTECTION AND PRIVACY ACT 2019",
      cloud: { title: "Cloud", lines: ["Only work involving", "no personal data"] },
      hub: {
        title: "Hub · system of record",
        lines: ["In-country server or Kampala colocation", "Nightly Uganda drift check"],
      },
      edge: { title: "Edge · field devices", lines: ["Offline ONNX scoring", "for rural outreach"] },
      cloudLink: "no personal data",
      edgeLink: "ONNX model",
    },
  },
};

/* ---- Week 3: interpretability and explainability ------------------- */
/* Source: README §11, “Explainability (Week 3: SHAP vs Captum)”.       */

export type MethodFamily = {
  name: string;
  methods: string[];
};

export type Week3Content = {
  intro: string;
  stats: { value: string; label: string; note?: string }[];
  methodsTitle: string;
  families: [MethodFamily, MethodFamily];
  rigour: string[];
  findingsTitle: string;
  findings: { title: string; text: string }[];
  deliverablesTitle: string;
  deliverables: { what: string; where: string }[];
  dashboard: {
    title: string;
    text: string;
    envNote: string;
    codeLabel: string;
    code: string;
    runtimeNote: string;
  };
};

export const week3: Week3Content = {
  intro:
    "Both candidate models, the shipped XGBoost and the PyTorch MLP, are explained with SHAP and Captum on the 10 raw survey fields, in logits, pooled and for Uganda.",
  stats: [
    { value: "2", label: "models explained", note: "XGBoost and PyTorch MLP" },
    { value: "10", label: "raw survey fields" },
    { value: "1,024", label: "coalitions", note: "Exact Shapley values by brute force" },
    { value: "9 in 10", label: "top fields agree", note: "SHAP vs Integrated Gradients, reference aligned" },
  ],
  methodsTitle: "Two families of methods",
  families: [
    { name: "SHAP", methods: ["TreeSHAP", "Exact KernelSHAP", "GradientSHAP"] },
    { name: "Captum", methods: ["Integrated Gradients", "Feature Ablation", "Layer attribution"] },
  ],
  rigour: [
    "Every SHAP method is checked against exact Shapley values computed by brute force over all 1,024 coalitions.",
    "The two families are compared with their references aligned, so method effects are not confused with reference or model effects.",
  ],
  findingsTitle: "What the explanations show",
  findings: [
    {
      title: "The same drivers in both models",
      text: "Both models rely mainly on phone access, education and country, plus the country × urban/rural interaction, and agree closely with each other when explained the same way.",
    },
    {
      title: "Agree overall, differ per person",
      text: "SHAP and Captum agree on what matters overall but often disagree about an individual. That disagreement comes mostly from the reference each compares against; with the reference aligned, SHAP and Integrated Gradients agree on nine in ten respondents’ top field.",
    },
    {
      title: "Models, not causes",
      text: "Attributions describe the models, not causes; the score is a ranking score, not a calibrated probability.",
    },
  ],
  deliverablesTitle: "Deliverables",
  deliverables: [
    { what: "Analysis notebook", where: "`notebooks/03_xai_analysis.ipynb`" },
    { what: "Comparison report (3–5 pages)", where: "`reports/xai_comparison_report.md`, `.pdf`" },
    { what: "Interactive dashboard", where: "`src/fin_inclusion/dashboard/`" },
    { what: "Every choice and why", where: "`reports/xai_decisions_log.md`" },
    { what: "Code guide", where: "`src/fin_inclusion/interpretability/README.md`" },
    { what: "Saved results", where: "`reports/xai/`, `reports/figures/xai/`" },
  ],
  dashboard: {
    title: "Interactive dashboard",
    text: "A Streamlit dashboard for exploring the attributions, served on localhost only. In Docker, the data and XAI artifacts are mounted read-only, and the production model in `models/production/` is never modified.",
    envNote:
      "The XAI work has its own environment: the training interpreter’s pydantic v1 cannot load the serving schemas the dashboard reuses, and Streamlit cannot share an environment with the serving stack’s pinned FastAPI.",
    codeLabel: "bash",
    code: `# Regenerate the analysis (--force recomputes cached attributions)
.venv-xai/Scripts/python scripts/run_xai_analysis.py --force

# Dashboard in Docker, then open http://localhost:8501
docker compose up --build dashboard`,
    runtimeNote:
      "The steps took about 85 minutes in total when first run on a laptop CPU; the final clean run took 507 minutes of wall-clock time.",
  },
};

/* ------------------------------------------------------------------ */
/* 05 · Reproducibility                                                */
/* ------------------------------------------------------------------ */

export type ReproducibilityContent = {
  intro: string;
  guarantees: string[];
  runLabel: string;
  run: string;
};

export const reproducibility: ReproducibilityContent = {
  intro: "Every number on this page can be regenerated from the repository.",
  guarantees: [
    "Seeds are centralised in `config/config.yaml` (seed 42).",
    "Requirements are pinned.",
    "Folds are deterministic.",
    "Per-fold tuned hyperparameters are persisted in `reports/ablation_raw_results.json`.",
    "No metric is computed on data that model or threshold selection has seen.",
    "The test suite runs on synthetic fixtures.",
  ],
  runLabel: "Run order",
  run: `python scripts/run_eda.py
python scripts/run_preprocessing.py
python scripts/run_baseline_training.py
python scripts/run_tuning_and_ablation.py`,
};

/* ------------------------------------------------------------------ */
/* 06 · Known limitations                                              */
/* ------------------------------------------------------------------ */

export type Limitation = { title: string; detail: string };

export const limitations: { intro: string; items: Limitation[] } = {
  intro: "What these results do not show, stated up front.",
  items: [
    { title: "Domain shift", detail: "One pooled four-country model carries domain-shift risk." },
    { title: "Associative, not causal", detail: "SHAP describes the model’s associations, not causes." },
    { title: "Survey years differ", detail: "The four surveys span 2016–2018." },
    { title: "Deterministic refit", detail: "The deployed model is a deterministic refit of the winning configuration." },
    { title: "No auth or TLS yet", detail: "The API runs on localhost only." },
    { title: "Uncalibrated scores", detail: "Scores rank respondents; they are not calibrated probabilities." },
    { title: "Input drift only", detail: "PSI catches input drift, not concept drift." },
    {
      title: "Laptop benchmark",
      detail: "The benchmark ran on a laptop CPU (Ryzen 7 5800U), so only the ratios carry over.",
    },
  ],
};

/* ------------------------------------------------------------------ */
/* 07 · Tech stack and footer                                          */
/* ------------------------------------------------------------------ */

export type StackGroup = { label: string; items: string[] };

export const techStack: StackGroup[] = [
  { label: "Data and modeling", items: ["Python 3.13", "scikit-learn", "XGBoost", "PyTorch", "Optuna", "SMOTE"] },
  { label: "Serving and MLOps", items: ["FastAPI", "Pydantic v2", "ONNX Runtime", "Docker"] },
  { label: "Explainability", items: ["SHAP", "Captum", "Streamlit"] },
  { label: "Testing", items: ["pytest"] },
];

export const footer = {
  note: "Built for honest evaluation over leaderboard-chasing.",
  repoLabel: "View the code on GitHub",
};
