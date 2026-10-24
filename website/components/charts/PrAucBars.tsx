"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ErrorBar,
  LabelList,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  type TooltipContentProps,
} from "recharts";
import type { MeanStd, ResultRow } from "@/data/content";
import { fixed3, meanStd } from "@/lib/format";

export const CHART_HEIGHT = 320;
const DOMAIN_MAX = 0.7;

type Datum = {
  id: string;
  model: string;
  imbalance: string;
  mean: number;
  std: number;
  winner: boolean;
};

type TickProps = { x?: number | string; y?: number | string; payload?: { value: string } };

/** Model name with the imbalance strategy underneath. */
function ModelTick({ x = 0, y = 0, payload, data }: TickProps & { data: Datum[] }) {
  const row = data.find((d) => d.id === payload?.value);
  return (
    <g transform={`translate(${x},${y})`}>
      <text x={-10} y={-3} textAnchor="end" className={`chart-label text-[12.5px] ${row?.winner ? "font-semibold" : ""}`}>
        {row?.model}
      </text>
      <text x={-10} y={12} textAnchor="end" className="chart-label-muted font-mono text-[11px]">
        {row?.imbalance}
      </text>
    </g>
  );
}

type ValueLabelProps = {
  x?: number | string;
  y?: number | string;
  width?: number | string;
  height?: number | string;
  index?: number;
};

/** "0.590" placed just past the right whisker. */
function ValueLabel({ x = 0, y = 0, width = 0, height = 0, index = 0, data }: ValueLabelProps & { data: Datum[] }) {
  const d = data[index];
  if (!d) return null;
  const w = Number(width);
  const whisker = d.mean > 0 ? (d.std / d.mean) * w : 0;
  return (
    <text
      x={Number(x) + w + whisker + 8}
      y={Number(y) + Number(height) / 2}
      dominantBaseline="central"
      className={`chart-label font-mono text-[12.5px] tabular-nums ${d.winner ? "font-semibold" : ""}`}
    >
      {fixed3(d.mean)}
    </text>
  );
}

function PrAucTooltip({ active, payload }: TooltipContentProps<number, string>) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload as Datum;
  return (
    <div className="rounded-md border border-rule bg-surface px-3 py-2 text-sm shadow-sm">
      <p className="font-medium text-ink">
        {d.model} <span className="text-muted">· {d.imbalance}</span>
      </p>
      <p className="font-mono text-ink tabular-nums">PR-AUC {meanStd({ mean: d.mean, std: d.std })}</p>
      {d.winner && <p className="text-xs text-accent">Shipped configuration</p>}
    </div>
  );
}

type Props = { rows: ResultRow[]; floor: { label: string; prAuc: MeanStd } };

/** Horizontal PR-AUC bars from zero, ± 1 std whiskers, LR floor reference. */
export default function PrAucBars({ rows, floor }: Props) {
  const data: Datum[] = rows.map((r) => ({
    id: r.id,
    model: r.model,
    imbalance: r.imbalance,
    mean: r.prAuc.mean,
    std: r.prAuc.std,
    winner: Boolean(r.winner),
  }));

  return (
    <ResponsiveContainer width="100%" height={CHART_HEIGHT}>
      <BarChart
        data={data}
        layout="vertical"
        margin={{ top: 22, right: 52, bottom: 4, left: 8 }}
        barCategoryGap="30%"
        accessibilityLayer={false}
      >
        <CartesianGrid horizontal={false} />
        <XAxis
          type="number"
          domain={[0, DOMAIN_MAX]}
          ticks={[0, 0.2, 0.4, 0.6]}
          tickFormatter={(v: number) => v.toFixed(1)}
          tickLine={false}
          axisLine={false}
          tick={{ fontSize: 12 }}
        />
        <YAxis
          type="category"
          dataKey="id"
          width={124}
          tickLine={false}
          axisLine={false}
          tick={(props: TickProps) => <ModelTick {...props} data={data} />}
        />
        <Tooltip
          cursor={{ className: "recharts-tooltip-cursor" }}
          content={(props) => <PrAucTooltip {...(props as TooltipContentProps<number, string>)} />}
          isAnimationActive={false}
        />
        <ReferenceLine
          x={floor.prAuc.mean}
          className="chart-reference"
          strokeDasharray="4 4"
          label={{
            value: `LR floor ${fixed3(floor.prAuc.mean)}`,
            position: "top",
            className: "chart-label-muted font-mono text-[11px]",
          }}
        />
        <Bar dataKey="mean" radius={[0, 4, 4, 0]} isAnimationActive={false}>
          {data.map((d) => (
            <Cell key={d.id} className={d.winner ? "bar-highlight" : "bar-context"} />
          ))}
          <ErrorBar
            dataKey="std"
            direction="x"
            width={8}
            strokeWidth={1.5}
            className="chart-error"
            isAnimationActive={false}
          />
          <LabelList
            dataKey="mean"
            content={(props) => <ValueLabel {...(props as ValueLabelProps)} data={data} />}
          />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
