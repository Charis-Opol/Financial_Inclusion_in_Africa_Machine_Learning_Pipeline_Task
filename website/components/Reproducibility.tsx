import type { ReproducibilityContent, SectionMeta } from "@/data/content";
import { CodeBlock } from "./CodeBlock";
import { Reveal } from "./Reveal";
import { RichText } from "./RichText";
import { Section } from "./Section";

/** Reproducibility guarantees beside the copyable run order. */
export function Reproducibility({ meta, content }: { meta: SectionMeta; content: ReproducibilityContent }) {
  return (
    <Section id={meta.id} eyebrow={meta.eyebrow} title={meta.title} intro={<p>{content.intro}</p>}>
      <Reveal className="grid grid-cols-1 items-start gap-10 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <ul className="divide-y divide-rule border-y border-rule">
          {content.guarantees.map((g) => (
            <li key={g} className="flex gap-3 py-3 text-ink">
              <span aria-hidden="true" className="font-mono text-sm text-accent">
                ✓
              </span>
              <span>
                <RichText text={g} />
              </span>
            </li>
          ))}
        </ul>
        <CodeBlock code={content.run} label={content.runLabel} />
      </Reveal>
    </Section>
  );
}
