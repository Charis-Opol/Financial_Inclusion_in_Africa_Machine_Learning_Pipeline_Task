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
