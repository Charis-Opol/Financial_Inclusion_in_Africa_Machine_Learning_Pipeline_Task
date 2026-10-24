"use client";

import dynamic from "next/dynamic";
import type { MeanStd, ResultRow } from "@/data/content";
import { meanStd } from "@/lib/format";
import { ChartFrame } from "./ChartFrame";

// Must match CHART_HEIGHT in PrAucBars (reserved space, no layout shift).
const HEIGHT = 320;

const Bars = dynamic(() => import("./PrAucBars"), {
  ssr: false,
  loading: () => <div style={{ height: HEIGHT }} />,
});

type Props = {
  rows: ResultRow[];
  floor: { label: string; prAuc: MeanStd };
  title: string;
  caption: string;
};

/** PR-AUC by configuration, with error bars, summary and data table. */
export function PrAucChart({ rows, floor, title, caption }: Props) {
  const summary = `${title}: ${rows
    .map((r) => `${r.model} with ${r.imbalance} ${meanStd(r.prAuc)}${r.winner ? " (shipped)" : ""}`)
    .join("; ")}. ${floor.label} ${meanStd(floor.prAuc)}.`;

  return (
    <ChartFrame
      title={title}
      caption={caption}
      summary={summary}
      height={HEIGHT}
      headingLevel={4}
      chart={<Bars rows={rows} floor={floor} />}
      table={
        <table className="w-full min-w-[22rem] text-left">
          <thead>
            <tr className="border-b border-rule text-muted">
              <th scope="col" className="py-2 pr-4 font-medium">Configuration</th>
              <th scope="col" className="py-2 text-right font-medium">PR-AUC (mean ± std)</th>
            </tr>
          </thead>
          <tbody className="font-mono tabular-nums">
            {rows.map((r) => (
              <tr key={r.id} className="border-b border-rule">
                <th scope="row" className="py-2 pr-4 font-sans font-normal">
                  {r.model} · {r.imbalance}
                </th>
                <td className="py-2 text-right">{meanStd(r.prAuc)}</td>
              </tr>
            ))}
            <tr>
              <th scope="row" className="py-2 pr-4 font-sans font-normal text-muted">{floor.label}</th>
              <td className="py-2 text-right text-muted">{meanStd(floor.prAuc)}</td>
            </tr>
          </tbody>
        </table>
      }
    />
  );
}
