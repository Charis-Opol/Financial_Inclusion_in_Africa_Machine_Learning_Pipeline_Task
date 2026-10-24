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
