import { DatasetSection } from "@/components/DatasetSection";
import { Hero } from "@/components/Hero";
import { Section } from "@/components/Section";
import { dataset, hero, sections } from "@/data/content";

export default function Home() {
  return (
    <main id="main" className="mx-auto max-w-6xl px-4 sm:px-6">
      <Hero content={hero} />

      {sections.map((s) =>
        s.id === "dataset" ? (
          <DatasetSection key={s.id} meta={s} content={dataset} />
        ) : (
          // Placeholder anchors, filled in by later phases.
          <Section key={s.id} id={s.id} eyebrow={s.eyebrow} title={s.title} />
        ),
      )}
    </main>
  );
}
