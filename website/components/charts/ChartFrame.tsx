import type { ReactNode } from "react";

type ChartFrameProps = {
  title: string;
  caption: string;
  /** Plain-text description of the data for screen readers. */
  summary: string;
  /** Reserved plot height, so the lazily loaded chart causes no layout shift. */
  height: number;
  chart: ReactNode;
  /** Data table behind a "Show data table" toggle; also the no-JS fallback. */
  table: ReactNode;
  /** Keeps the document outline intact: 3 under a section, 4 under a subsection. */
  headingLevel?: 3 | 4;
};

/** Card wrapper shared by every chart: title, caption, plot, data table. */
export function ChartFrame({
  title,
  caption,
  summary,
  height,
  chart,
  table,
  headingLevel = 3,
}: ChartFrameProps) {
  const Heading = headingLevel === 3 ? "h3" : "h4";
  return (
    <figure className="rounded-lg border border-rule bg-surface p-4 sm:p-6">
      <Heading className="font-sans text-base font-semibold text-ink">{title}</Heading>
      <p className="mt-1 text-sm text-muted">{caption}</p>

      <div className="chart mt-6" role="img" aria-label={summary} style={{ minHeight: height }}>
        {chart}
      </div>

      <details className="mt-4 border-t border-rule pt-3 text-sm">
        <summary className="cursor-pointer text-muted hover:text-ink">Show data table</summary>
        <div className="mt-3 overflow-x-auto">{table}</div>
      </details>
    </figure>
  );
}
