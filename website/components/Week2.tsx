import type { BenchmarkRow, DriftScenario, SectionMeta, Week2Content } from "@/data/content";
import { CodeBlock } from "./CodeBlock";
import { DataTable } from "./DataTable";
import { DeploymentDiagram } from "./DeploymentDiagram";
import { PullQuote } from "./PullQuote";
import { Reveal } from "./Reveal";
import { RichText } from "./RichText";
import { Section } from "./Section";
import { StatCard } from "./StatCard";
import { StatusBadge } from "./StatusBadge";
import { SubHeading } from "./SubHeading";

const eyebrow = "font-mono text-[11px] uppercase tracking-[0.14em]";

function Bullets({ items }: { items: string[] }) {
  return (
    <ul className="space-y-3">
      {items.map((item) => (
        <li key={item} className="flex gap-3 text-ink">
          <span aria-hidden="true" className="mt-[0.75em] h-px w-3 shrink-0 bg-accent" />
          <span>
            <RichText text={item} />
          </span>
        </li>
      ))}
    </ul>
  );
}

/** Faster runtime in each benchmark row is set in bold. */
const benchCell = (row: BenchmarkRow, which: "native" | "onnx") => (
  <span className={row.faster === which ? "font-semibold" : "text-muted"}>{row[which]}</span>
);

/** Week 2: production artifact, API, container, ONNX, monitoring, deployment. */
export function Week2({ meta, content }: { meta: SectionMeta; content: Week2Content }) {
  const { artifact, api, docker, onnx, monitoring, tradeoffs } = content;
  return (
    <Section id={meta.id} eyebrow={meta.eyebrow} title={meta.title}>
      <div className="space-y-16">
        <Reveal>
          <PullQuote text={content.quote.text} detail={content.quote.detail} />
        </Reveal>

        {/* Production artifact */}
        <Reveal>
          <SubHeading id="mlops-artifact">{artifact.title}</SubHeading>
          <div className="grid grid-cols-1 gap-8 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
            <div className="space-y-5">
              <p className="text-ink">{artifact.summary}</p>
              <ul className="flex flex-wrap gap-2" aria-label="Artifact files">
                {artifact.files.map((f) => (
                  <li key={f} className="rounded-sm border border-rule bg-surface px-2.5 py-1 font-mono text-sm text-ink">
                    {f}
                  </li>
                ))}
              </ul>
              <aside className="border-l-2 border-accent pl-4">
                <p className={`${eyebrow} mb-1 text-accent`}>Parity gate</p>
                <p className="text-ink">{artifact.gate}</p>
              </aside>
            </div>
            <dl className="grid gap-px overflow-hidden rounded-lg border border-rule bg-rule sm:grid-cols-2">
              {artifact.facts.map((f) => (
                <div key={f.label} className="bg-surface p-4">
                  <dt className="text-sm text-muted">{f.label}</dt>
                  <dd className="mt-1 font-mono text-lg text-ink wrap-anywhere">
                    {f.value}
                    {f.note && <span className="mt-0.5 block font-sans text-xs text-muted">{f.note}</span>}
                  </dd>
                </div>
              ))}
            </dl>
          </div>
        </Reveal>

        {/* FastAPI */}
        <Reveal>
          <SubHeading id="mlops-api">{api.title}</SubHeading>
          <ul className="mb-5 flex flex-wrap gap-2" aria-label="Endpoints">
            {api.endpoints.map((e) => (
              <li key={e} className="rounded-sm bg-accent-soft px-2.5 py-1 font-mono text-sm text-accent">
                {e}
              </li>
            ))}
          </ul>
          <Bullets items={api.contract} />

          <ul className="mt-8 grid gap-4 sm:grid-cols-3" aria-label="Status codes">
            {api.statusCodes.map((s) => (
              <li key={s.code} className="rounded-lg border border-rule bg-surface p-5">
                <p className="font-mono text-3xl text-ink tabular-nums">{s.code}</p>
                <p className="mt-1 font-semibold text-ink">{s.name}</p>
                <p className="mt-1 text-[0.95rem] text-muted">
                  <RichText text={s.meaning} />
                </p>
              </li>
            ))}
          </ul>

          <h4 className="mt-10 mb-4 font-semibold text-ink">{api.checksTitle}</h4>
          <ol className="flex flex-wrap gap-2">
            {api.checks.map((c, i) => (
              <li
                key={c}
                className="inline-flex items-center gap-2 rounded-sm border border-rule bg-surface px-3 py-1.5 text-sm text-ink"
              >
                <span aria-hidden="true" className="font-mono text-xs text-muted">
                  {i + 1}
                </span>
                {c}
              </li>
            ))}
          </ol>

          <div className="mt-10">
            <CodeBlock code={api.sample} label={api.sampleLabel} />
            <p className="mt-3 text-sm text-muted">{api.sampleNote}</p>
          </div>
        </Reveal>

        {/* Docker */}
        <Reveal>
          <SubHeading id="mlops-docker">{docker.title}</SubHeading>
          <ul className="grid gap-px overflow-hidden rounded-lg border border-rule bg-rule sm:grid-cols-2 lg:grid-cols-3">
            {docker.items.map((item) => (
              <li key={item.label} className="flex gap-3 bg-surface p-4">
                <span aria-hidden="true" className="mt-0.5 font-mono text-sm text-accent">
                  ✓
                </span>
                <div>
                  <p className="font-medium text-ink">{item.label}</p>
                  <p className="mt-0.5 text-sm text-muted">
                    <RichText text={item.detail} />
                  </p>
                </div>
              </li>
            ))}
          </ul>
        </Reveal>

        {/* ONNX */}
        <Reveal>
          <SubHeading id="mlops-onnx">{onnx.title}</SubHeading>
          <div className="mb-8 grid grid-cols-2 gap-6 sm:grid-cols-4">
            {onnx.facts.map((f) => (
              <StatCard key={f.label} value={f.value} label={f.label} note={f.note} />
            ))}
          </div>
          <DataTable<BenchmarkRow>
            caption={onnx.benchmarkCaption}
            rows={onnx.benchmark}
            rowKey={(r) => r.batch}
            minWidth="30rem"
            columns={[
              { header: "Batch", cell: (r) => r.batch, mono: true },
              { header: "Native XGBoost", cell: (r) => benchCell(r, "native"), mono: true, align: "right" },
              { header: "ONNX Runtime", cell: (r) => benchCell(r, "onnx"), mono: true, align: "right" },
              { header: "Speed-up", cell: (r) => r.speedup, mono: true, align: "right" },
            ]}
          />
          <p className="mt-4 text-ink">{onnx.takeaway}</p>
        </Reveal>

        {/* Monitoring */}
        <Reveal>
          <SubHeading id="mlops-monitoring">{monitoring.title}</SubHeading>
          <Bullets items={[monitoring.log, monitoring.psi]} />

          <ul className="mt-6 grid grid-cols-1 gap-2 sm:grid-cols-3" aria-label="PSI thresholds">
            {monitoring.bands.map((b) => (
              <li key={b.level} className="rounded-md border border-rule bg-surface p-3">
                <StatusBadge level={b.level}>{b.label}</StatusBadge>
                <p className="mt-2 font-mono text-sm text-ink tabular-nums">PSI {b.range}</p>
              </li>
            ))}
          </ul>

          <div className="mt-8 grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,2.2fr)]">
            <DataTable
              caption={monitoring.exitCodesCaption}
              rows={monitoring.exitCodes}
              rowKey={(r) => r.code}
              minWidth="14rem"
              columns={[
                { header: "Exit", cell: (r) => r.code, mono: true },
                { header: "Meaning", cell: (r) => r.meaning },
              ]}
            />
            <DataTable<DriftScenario>
              caption={monitoring.scenariosCaption}
              rows={monitoring.scenarios}
              rowKey={(r) => r.scenario}
              minWidth="34rem"
              columns={[
                { header: "Scenario", cell: (r) => r.scenario },
                { header: "Verdict", cell: (r) => <StatusBadge level={r.level}>{r.verdict}</StatusBadge> },
                { header: "Evidence", cell: (r) => <RichText text={r.evidence} /> },
              ]}
            />
          </div>
        </Reveal>

        {/* Deployment trade-offs */}
        <Reveal>
          <SubHeading id="mlops-deployment">{tradeoffs.title}</SubHeading>
          <div className="grid grid-cols-1 items-start gap-10 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
            <div className="space-y-8">
              <div>
                <p className="mb-4 text-ink">{tradeoffs.intro}</p>
                <dl className="space-y-3">
                  {tradeoffs.factors.map((f) => (
                    <div key={f.label}>
                      <dt className="font-medium text-ink">{f.label}</dt>
                      <dd className="text-[0.95rem] text-muted">{f.detail}</dd>
                    </div>
                  ))}
                </dl>
              </div>
              <aside aria-labelledby="mlops-recommendation" className="rounded-lg border border-accent bg-accent-soft p-5">
                <h4 id="mlops-recommendation" className="font-semibold text-ink">
                  {tradeoffs.recommendation.title}
                </h4>
                <ul className="mt-3 space-y-2 text-[0.95rem] text-ink">
                  {tradeoffs.recommendation.points.map((p) => (
                    <li key={p} className="flex gap-3">
                      <span aria-hidden="true" className="mt-[0.7em] h-px w-3 shrink-0 bg-accent" />
                      <span>{p}</span>
                    </li>
                  ))}
                </ul>
              </aside>
            </div>
            <figure className="mx-auto w-full max-w-md rounded-lg border border-rule bg-surface p-4 sm:p-6">
              <DeploymentDiagram d={tradeoffs.diagram} />
            </figure>
          </div>
        </Reveal>
      </div>
    </Section>
  );
}
