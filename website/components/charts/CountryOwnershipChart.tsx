"use client";

import dynamic from "next/dynamic";
import type { CountryOwnership } from "@/data/content";
import { ChartFrame } from "./ChartFrame";
import { LazyWhenVisible } from "./LazyWhenVisible";

// Must match CHART_HEIGHT in CountryOwnershipBars so the placeholder reserves
// exactly the chart's space (no layout shift when Recharts loads).
const HEIGHT = 240;

// Recharts is loaded on the client only, once the chart nears the viewport.
const Bars = dynamic(() => import("./CountryOwnershipBars"), {
  ssr: false,
  loading: () => <div style={{ height: HEIGHT }} />,
});

type Props = {
  data: CountryOwnership[];
  title: string;
  caption: string;
};

/** Account-ownership bar chart with a screen-reader summary and data table. */
export function CountryOwnershipChart({ data, title, caption }: Props) {
  const summary = `${title}: ${data
    .map((d) => `${d.country} ${d.ownership}% (${d.surveyYear})`)
    .join(", ")}.`;

  return (
    <ChartFrame
      title={title}
      caption={caption}
      summary={summary}
      height={HEIGHT}
      chart={
        <LazyWhenVisible height={HEIGHT}>
          <Bars data={data} />
        </LazyWhenVisible>
      }
      table={
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
      }
    />
  );
}
