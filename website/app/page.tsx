import { DatasetSection } from "@/components/DatasetSection";
import { Hero } from "@/components/Hero";
import { Section } from "@/components/Section";
import { Week1 } from "@/components/Week1";
import { Week2 } from "@/components/Week2";
import { WeeksTimeline } from "@/components/WeeksTimeline";
import { dataset, hero, sections, week1, week2, weekSteps, type SectionMeta } from "@/data/content";

const byId = (id: SectionMeta["id"]) => sections.find((s) => s.id === id)!;

// Placeholder anchors, filled in by later phases.
const Placeholder = ({ meta }: { meta: SectionMeta }) => (
  <Section id={meta.id} eyebrow={meta.eyebrow} title={meta.title} />
);

export default function Home() {
  const weeks = (["week-1", "week-2", "week-3"] as const).map(byId);

  return (
    <main id="main" className="mx-auto max-w-6xl px-4 sm:px-6">
      <Hero content={hero} />
      <DatasetSection meta={byId("dataset")} content={dataset} />

      <WeeksTimeline
        steps={weeks.map((w) => ({
          id: w.id,
          label: w.navLabel,
          description: weekSteps[w.id as keyof typeof weekSteps],
        }))}
      >
        <Week1 meta={byId("week-1")} content={week1} />
        <Week2 meta={byId("week-2")} content={week2} />
        <Placeholder meta={byId("week-3")} />
      </WeeksTimeline>

      <Placeholder meta={byId("reproducibility")} />
      <Placeholder meta={byId("limitations")} />
      <Placeholder meta={byId("stack")} />
    </main>
  );
}
