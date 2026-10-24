import { DatasetSection } from "@/components/DatasetSection";
import { Footer } from "@/components/Footer";
import { Hero } from "@/components/Hero";
import { Limitations } from "@/components/Limitations";
import { Reproducibility } from "@/components/Reproducibility";
import { TechStack } from "@/components/TechStack";
import { Week1 } from "@/components/Week1";
import { Week2 } from "@/components/Week2";
import { Week3 } from "@/components/Week3";
import { WeeksTimeline } from "@/components/WeeksTimeline";
import {
  dataset,
  footer,
  hero,
  limitations,
  modeling,
  reproducibility,
  sections,
  site,
  techStack,
  week1,
  week2,
  week3,
  weekSteps,
  type SectionMeta,
} from "@/data/content";

const byId = (id: SectionMeta["id"]) => sections.find((s) => s.id === id)!;

export default function Home() {
  const weeks = (["week-1", "week-2", "week-3"] as const).map(byId);

  return (
    <>
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
          <Week1 meta={byId("week-1")} content={week1} modeling={modeling} />
          <Week2 meta={byId("week-2")} content={week2} />
          <Week3 meta={byId("week-3")} content={week3} />
        </WeeksTimeline>

        <Reproducibility meta={byId("reproducibility")} content={reproducibility} />
        <Limitations meta={byId("limitations")} intro={limitations.intro} items={limitations.items} />
        <TechStack meta={byId("stack")} groups={techStack} />
      </main>

      <Footer
        title={site.title}
        author={site.author}
        repoUrl={site.repoUrl}
        repoLabel={footer.repoLabel}
        note={footer.note}
      />
    </>
  );
}
