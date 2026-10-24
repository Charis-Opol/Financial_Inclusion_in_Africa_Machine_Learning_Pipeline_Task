import type { SectionMeta, Week2Content } from "@/data/content";
import { PrAucChart } from "./charts/PrAucChart";
import { NestedCvDiagram } from "./NestedCvDiagram";
import { ResultsTable } from "./ResultsTable";
import { Reveal } from "./Reveal";
import { RichText } from "./RichText";
import { Section } from "./Section";
import { SubHeading } from "./SubHeading";

const eyebrow = "font-mono text-[11px] uppercase tracking-[0.14em]";

/** Week 2: models, nested CV, metrics, results and the shipping decision. */
export function Week2({ meta, content }: { meta: SectionMeta; content: Week2Content }) {
  const { shipped } = content;
  return (
    <Section id={meta.id} eyebrow={meta.eyebrow} title={meta.title} intro={<p>{content.intro}</p>}>
      <div className="space-y-16">
        {/* Models */}
        <Reveal>
          <SubHeading id="week-2-models">{content.modelsTitle}</SubHeading>
          <ul className="grid gap-4 sm:grid-cols-3">
            {content.models.map((m) => (
              <li key={m.name} className="rounded-lg border border-rule bg-surface p-5">
                <p className={`${eyebrow} text-muted`}>{m.role}</p>
                <p className="mt-2 font-semibold text-ink">{m.name}</p>
                <p className="mt-1 text-[0.95rem] text-muted">{m.detail}</p>
              </li>
            ))}
          </ul>
          <p className="mt-4 text-[0.95rem] text-ink">
            <span className={`${eyebrow} mr-2 text-accent`}>Imbalance</span>
            {content.imbalance}
          </p>
        </Reveal>

        {/* Validation */}
        <Reveal>
          <SubHeading id="week-2-validation">{content.validationTitle}</SubHeading>
          <div className="grid grid-cols-1 items-start gap-10 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
            <figure className="rounded-lg border border-rule bg-surface p-4 sm:p-6">
              <NestedCvDiagram {...content.diagram} />
            </figure>
            <ul className="space-y-4">
              {content.validation.map((rule) => (
                <li key={rule} className="flex gap-3 text-ink">
                  <span aria-hidden="true" className="mt-[0.75em] h-px w-3 shrink-0 bg-accent" />
                  <span>{rule}</span>
                </li>
              ))}
            </ul>
          </div>
        </Reveal>

        {/* Metrics */}
        <Reveal>
          <SubHeading id="week-2-metrics">{content.metricsTitle}</SubHeading>
          <dl className="grid gap-px overflow-hidden rounded-lg border border-rule bg-rule sm:grid-cols-2">
            {content.metrics.map((m) => (
              <div key={m.name} className="bg-surface p-5">
                <dt className="flex items-center gap-2 font-semibold text-ink">
                  {m.name}
                  {m.headline && (
                    <span className={`${eyebrow} rounded-sm bg-accent-soft px-1.5 py-0.5 text-accent`}>
                      Headline
                    </span>
                  )}
                </dt>
                <dd className="mt-1 text-[0.95rem] text-muted">{m.note}</dd>
              </div>
            ))}
          </dl>
          <p className="mt-3 text-sm text-muted">{content.metricsFootnote}</p>
          <aside className="mt-6 border-l-2 border-accent pl-4 text-ink">
            <p className={`${eyebrow} mb-1 text-accent`}>Why not accuracy</p>
            <p>{content.accuracyTrap}</p>
          </aside>
        </Reveal>

        {/* Results */}
        <div id="results" className="scroll-mt-24">
          <Reveal>
            <SubHeading id="week-2-results">{content.resultsTitle}</SubHeading>
            <p className="-mt-3 mb-6 text-muted">{content.resultsIntro}</p>
            <ResultsTable rows={content.results} labelledBy="week-2-results" />
          </Reveal>
          <Reveal className="mt-8">
            <PrAucChart
              rows={content.results}
              floor={content.floor}
              title={content.chart.title}
              caption={content.chart.caption}
            />
          </Reveal>
        </div>

        {/* Shipping decision */}
        <Reveal>
          <aside
            aria-labelledby="week-2-shipped"
            className="grid gap-8 rounded-lg border border-accent bg-accent-soft p-6 sm:p-8 lg:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)]"
          >
            <div>
              <p className={`${eyebrow} text-accent`}>Decision</p>
              <h3 id="week-2-shipped" className="mt-2 text-2xl font-semibold">
                {shipped.title}
              </h3>
              <div className="mt-4 space-y-3 text-ink">
                {shipped.body.map((p) => (
                  <p key={p}>
                    <RichText text={p} />
                  </p>
                ))}
              </div>
            </div>
            <dl className="space-y-5 border-t border-accent/30 pt-6 lg:border-t-0 lg:border-l lg:pt-0 lg:pl-8">
              {shipped.stats.map((s) => (
                <div key={s.label}>
                  <dt className="text-sm text-muted">{s.label}</dt>
                  <dd className="mt-0.5 font-mono text-2xl text-ink tabular-nums">
                    {s.value}
                    {s.note && <span className="block font-sans text-xs text-muted">{s.note}</span>}
                  </dd>
                </div>
              ))}
            </dl>
          </aside>
        </Reveal>
      </div>
    </Section>
  );
}
