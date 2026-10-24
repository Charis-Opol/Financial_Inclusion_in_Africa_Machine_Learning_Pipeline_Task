import type { Limitation, SectionMeta } from "@/data/content";
import { Reveal } from "./Reveal";
import { Section } from "./Section";

/** Numbered honesty notes in a two-column grid. */
export function Limitations({
  meta,
  intro,
  items,
}: {
  meta: SectionMeta;
  intro: string;
  items: Limitation[];
}) {
  return (
    <Section id={meta.id} eyebrow={meta.eyebrow} title={meta.title} intro={<p>{intro}</p>}>
      <Reveal>
        <ol className="grid grid-cols-1 gap-px overflow-hidden rounded-lg border border-rule bg-rule sm:grid-cols-2">
          {items.map((item, i) => (
            <li key={item.title} className="flex gap-4 bg-surface p-5">
              <span className="font-mono text-xs text-muted tabular-nums">{String(i + 1).padStart(2, "0")}</span>
              <div>
                <p className="font-semibold text-ink">{item.title}</p>
                <p className="mt-1 text-[0.95rem] text-muted">{item.detail}</p>
              </div>
            </li>
          ))}
        </ol>
      </Reveal>
    </Section>
  );
}
