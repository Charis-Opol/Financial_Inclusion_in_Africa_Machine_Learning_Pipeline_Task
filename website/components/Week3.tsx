import type { MethodFamily, SectionMeta, Week3Content } from "@/data/content";
import { CodeBlock } from "./CodeBlock";
import { DataTable } from "./DataTable";
import { ReportFigure } from "./ReportFigure";
import { Reveal } from "./Reveal";
import { RichText } from "./RichText";
import { Section } from "./Section";
import { StatCard } from "./StatCard";
import { SubHeading } from "./SubHeading";

const eyebrow = "font-mono text-[11px] uppercase tracking-[0.14em]";

function FamilyCard({ family }: { family: MethodFamily }) {
  return (
    <div className="rounded-lg border border-rule bg-surface p-5">
      <p className={`${eyebrow} text-accent`}>Family</p>
      <h4 className="mt-1 text-xl font-semibold">{family.name}</h4>
      <ul className="mt-3 space-y-1.5">
        {family.methods.map((m) => (
          <li key={m} className="font-mono text-sm text-ink">
            {m}
          </li>
        ))}
      </ul>
    </div>
  );
}

/** Week 3: SHAP vs Captum on both models, findings, deliverables, dashboard. */
export function Week3({ meta, content }: { meta: SectionMeta; content: Week3Content }) {
  const [shap, captum] = content.families;
  const { dashboard } = content;
  return (
    <Section id={meta.id} eyebrow={meta.eyebrow} title={meta.title} intro={<p>{content.intro}</p>}>
      <div className="space-y-16">
        <Reveal>
          <div className="grid grid-cols-2 gap-x-6 gap-y-8 lg:grid-cols-4">
            {content.stats.map((s) => (
              <StatCard key={s.label} value={s.value} label={s.label} note={s.note} />
            ))}
          </div>
        </Reveal>

        {/* Methods */}
        <Reveal>
          <SubHeading id="xai-methods">{content.methodsTitle}</SubHeading>
          <div className="grid grid-cols-1 items-center gap-3 md:grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)]">
            <FamilyCard family={shap} />
            <p className="mx-auto rounded-full border border-accent bg-accent-soft px-3 py-1 text-center font-mono text-xs text-accent">
              <span aria-hidden="true">⇄ </span>references aligned
            </p>
            <FamilyCard family={captum} />
          </div>
          <ul className="mt-6 space-y-3">
            {content.rigour.map((r) => (
              <li key={r} className="flex gap-3 text-ink">
                <span aria-hidden="true" className="mt-[0.75em] h-px w-3 shrink-0 bg-accent" />
                <span>{r}</span>
              </li>
            ))}
          </ul>
        </Reveal>

        {/* Findings */}
        <div>
          <Reveal>
            <SubHeading id="xai-findings">{content.findingsTitle}</SubHeading>
          </Reveal>
          <ol className="grid gap-4 md:grid-cols-3">
            {content.findings.map((f, i) => (
              <li key={f.title}>
                <Reveal className="h-full">
                  <article className="h-full rounded-lg border border-rule bg-surface p-5 sm:p-6">
                    <span className="font-mono text-xs text-muted tabular-nums">{String(i + 1).padStart(2, "0")}</span>
                    <h4 className="mt-1 text-lg font-semibold">{f.title}</h4>
                    <p className="mt-3 text-[0.95rem] text-ink">{f.text}</p>
                  </article>
                </Reveal>
              </li>
            ))}
          </ol>
        </div>

        {/* Figures */}
        <div>
          <Reveal>
            <SubHeading id="xai-figures">{content.figuresTitle}</SubHeading>
          </Reveal>
          <ul className="space-y-8">
            {content.figures.map((f) => (
              <li key={f.src}>
                <Reveal>
                  <ReportFigure figure={f} sizes="(min-width: 1024px) 880px, 100vw" />
                </Reveal>
              </li>
            ))}
          </ul>
        </div>

        {/* Deliverables */}
        <Reveal>
          <SubHeading id="xai-deliverables">{content.deliverablesTitle}</SubHeading>
          <DataTable
            caption="Where each piece of the Week 3 work lives in the repository"
            rows={content.deliverables}
            rowKey={(r) => r.what}
            minWidth="32rem"
            columns={[
              { header: "What", cell: (r) => r.what },
              { header: "Where", cell: (r) => <RichText text={r.where} /> },
            ]}
          />
        </Reveal>

        {/* Dashboard */}
        <Reveal>
          <SubHeading id="xai-dashboard">{dashboard.title}</SubHeading>
          <div className="space-y-8">
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
              <p className="text-ink">
                <RichText text={dashboard.text} />
              </p>
              <aside className="border-l-2 border-accent pl-4">
                <p className={`${eyebrow} mb-1 text-accent`}>Separate environment</p>
                <p className="text-[0.95rem] text-ink">{dashboard.envNote}</p>
              </aside>
            </div>
            <div>
              <CodeBlock code={dashboard.code} label={dashboard.codeLabel} />
              <p className="mt-3 text-sm text-muted">{dashboard.runtimeNote}</p>
            </div>
          </div>
        </Reveal>
      </div>
    </Section>
  );
}
