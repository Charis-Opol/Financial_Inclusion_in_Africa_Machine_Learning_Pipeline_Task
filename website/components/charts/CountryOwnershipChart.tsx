"use client";

import dynamic from "next/dynamic";
import type { CountryOwnership } from "@/data/content";

// Must match CHART_HEIGHT in CountryOwnershipBars so the placeholder reserves
// exactly the chart's space (no layout shift when Recharts loads).
const HEIGHT = 240;

// Recharts is loaded on the client only, after the page is interactive.
const Bars = dynamic(() => import("./CountryOwnershipBars"), {
  ssr: false,
  loading: () => <div style={{ height: HEIGHT }} />,
});

type Props = {
  data: CountryOwnership[];
  title: string;
  caption: string;
};

/**
 * Account-ownership bar chart with a summary for screen readers and a
 * visible data table toggle (also the no-JS fallback).
 */
export function CountryOwnershipChart({ data, title, caption }: Props) {
  const summary = `${title}: ${data
    .map((d) => `${d.country} ${d.ownership}% (${d.surveyYear})`)
    .join(", ")}.`;

  return (
    <figure className="rounded-lg border border-rule bg-surface p-4 sm:p-6">
      <h3 className="font-sans text-base font-semibold text-ink">{title}</h3>
      <p className="mt-1 text-sm text-muted">{caption}</p>

      <div className="chart mt-6" role="img" aria-label={summary} style={{ minHeight: HEIGHT }}>
        <Bars data={data} />
      </div>

      <details className="mt-4 border-t border-rule pt-3 text-sm">
        <summary className="cursor-pointer text-muted hover:text-ink">Show data table</summary>
        <div className="mt-3 overflow-x-auto">
          <table className="w-full min-w-[18rem] text-left">
            <thead>
              <tr className="border-b border-rule text-muted">
                <th scope="col" className="py-2 pr-4 font-medium">Country</th>
                <th scope="col" className="py-2 pr-4 text-right font-medium">Ownership</th>
                <th scope="col" className="py-2 text-right font-medium">Survey year</th>
              </tr>
            </thead>
            <tbody className="font-mono tabular-nums">
              {data.map((d) => (
                <tr key={d.country} className="border-b border-rule last:border-0">
                  <th scope="row" className="py-2 pr-4 font-sans font-normal">{d.country}</th>
                  <td className="py-2 pr-4 text-right">{d.ownership}%</td>
                  <td className="py-2 text-right">{d.surveyYear}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </figure>
  );
}
