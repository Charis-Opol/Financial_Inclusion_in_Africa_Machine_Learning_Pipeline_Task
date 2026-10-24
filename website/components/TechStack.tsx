import type { SectionMeta, StackGroup } from "@/data/content";
import { Reveal } from "./Reveal";
import { Section } from "./Section";

/** Tools grouped by stage, shown as quiet badges. */
export function TechStack({ meta, groups }: { meta: SectionMeta; groups: StackGroup[] }) {
  return (
    <Section id={meta.id} eyebrow={meta.eyebrow} title={meta.title}>
      <Reveal>
        <dl className="grid grid-cols-1 gap-8 sm:grid-cols-2 lg:grid-cols-4">
          {groups.map((g) => (
            <div key={g.label}>
              <dt className="mb-3 font-mono text-[11px] uppercase tracking-[0.14em] text-muted">{g.label}</dt>
              <dd>
                <ul className="flex flex-wrap gap-2">
                  {g.items.map((item) => (
                    <li
                      key={item}
                      className="rounded-sm border border-rule bg-surface px-2.5 py-1 font-mono text-sm text-ink"
                    >
                      {item}
                    </li>
                  ))}
                </ul>
              </dd>
            </div>
          ))}
        </dl>
      </Reveal>
    </Section>
  );
}
