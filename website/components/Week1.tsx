import type { ModelingContent, SectionMeta, Week1Content } from "@/data/content";
import { FindingCard } from "./FindingCard";
import { ModelingBlocks } from "./Modeling";
import { PreprocessingTable } from "./PreprocessingTable";
import { Reveal } from "./Reveal";
import { Section } from "./Section";
import { SolidList } from "./SolidList";
import { SubHeading } from "./SubHeading";

/**
 * Week 1: EDA findings and decisions, preprocessing table and SOLID mapping,
 * then modeling and evaluation (results, shipping decision).
 */
export function Week1({
  meta,
  content,
  modeling,
}: {
  meta: SectionMeta;
  content: Week1Content;
  modeling: ModelingContent;
}) {
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

        <ModelingBlocks content={modeling} />
      </div>
    </Section>
  );
}
