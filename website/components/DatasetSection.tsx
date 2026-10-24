import type { DatasetSection as DatasetContent, SectionMeta } from "@/data/content";
import { CountryOwnershipChart } from "./charts/CountryOwnershipChart";
import { Reveal } from "./Reveal";
import { Section } from "./Section";
import { StatCard } from "./StatCard";

/** Thin two-segment bar for the target split; numbers are repeated in text. */
function ClassSplit({ classes }: { classes: DatasetContent["target"]["classes"] }) {
  const [major, minor] = classes;
  return (
    <div>
      <div aria-hidden="true" className="flex h-2.5 gap-0.5">
        <div className="rounded-l-sm bg-[var(--chart-context)]" style={{ width: `${major.share}%` }} />
        <div className="rounded-r-sm bg-[var(--chart-highlight)]" style={{ width: `${minor.share}%` }} />
      </div>
      <ul className="mt-3 flex justify-between font-mono text-sm tabular-nums">
        {classes.map((c) => (
          <li key={c.label}>
            <span className="text-muted">{c.label}</span> <span className="text-ink">{c.share}%</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

export function DatasetSection({ meta, content }: { meta: SectionMeta; content: DatasetContent }) {
  return (
    <Section id={meta.id} eyebrow={meta.eyebrow} title={meta.title} intro={<p>{content.brief}</p>}>
      <div className="grid gap-12 lg:grid-cols-12 lg:gap-10">
        <Reveal className="space-y-10 lg:col-span-5">
          <div>
            <p className="font-mono text-xs uppercase tracking-[0.14em] text-muted">Dataset</p>
            <p className="mt-2 font-display text-2xl font-semibold">{content.source.name}</p>
            <p className="mt-1 text-muted">{content.source.detail}</p>
          </div>

          <div className="grid grid-cols-2 gap-6">
            {content.splits.map((s) => (
              <StatCard
                key={s.label}
                value={s.rows}
                label={`${s.label} rows × ${s.columns} columns`}
                note={s.note}
              />
            ))}
          </div>

          <div>
            <p className="mb-3 text-sm text-muted">
              Target <code className="font-mono text-ink">{content.target.name}</code>
            </p>
            <ClassSplit classes={content.target.classes} />
            <p className="mt-2 text-sm text-muted">{content.target.note}</p>
          </div>

          <div>
            <h3 className="font-sans text-base font-semibold">{content.whyTitle}</h3>
            <ul className="mt-3 space-y-2">
              {content.why.map((item) => (
                <li key={item} className="flex gap-3 text-muted">
                  <span aria-hidden="true" className="mt-[0.7em] h-px w-3 shrink-0 bg-accent" />
                  <span>{item}</span>
                </li>
              ))}
            </ul>
          </div>
        </Reveal>

        <Reveal className="lg:col-span-7">
          <CountryOwnershipChart
            data={content.chart.data}
            title={content.chart.title}
            caption={content.chart.caption}
          />
        </Reveal>
      </div>
    </Section>
  );
}
