import type { SectionMeta, Week1Content } from "@/data/content";
import { FindingCard } from "./FindingCard";
import { PreprocessingTable } from "./PreprocessingTable";
import { Reveal } from "./Reveal";
import { Section } from "./Section";
import { SolidList } from "./SolidList";

function SubHeading({ id, children }: { id: string; children: string }) {
  return (
    <h3 id={id} className="mb-6 font-sans text-xl font-semibold">
      {children}
    </h3>
  );
}

/** Week 1: EDA findings and decisions, preprocessing table, SOLID mapping. */
export function Week1({ meta, content }: { meta: SectionMeta; content: Week1Content }) {
  return (
    <Section id={meta.id} eyebrow={meta.eyebrow} title={meta.title} intro={<p>{content.intro}</p>}>
      <div className="space-y-16">
        <div>
          <Reveal>
            <SubHeading id="week-1-findings">{content.findingsTitle}</SubHeading>
          </Reveal>
          <ol className="grid gap-4 md:grid-cols-2">
            {content.findings.map((f, i) => (
              <li key={f.title}>
                <Reveal className="h-full">
                  <FindingCard index={i + 1} item={f} />
                </Reveal>
              </li>
            ))}
          </ol>
        </div>

        <Reveal>
          <SubHeading id="week-1-preprocessing">{content.preprocessingTitle}</SubHeading>
          <PreprocessingTable rows={content.preprocessing} labelledBy="week-1-preprocessing" />
        </Reveal>

        <Reveal>
          <SubHeading id="week-1-solid">{content.solidTitle}</SubHeading>
          <SolidList items={content.solid} />
        </Reveal>
      </div>
    </Section>
  );
}
