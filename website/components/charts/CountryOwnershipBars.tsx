"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  LabelList,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  type TooltipContentProps,
} from "recharts";
import type { CountryOwnership } from "@/data/content";

export const CHART_HEIGHT = 240;

type TickProps = { x?: number | string; y?: number | string; payload?: { value: string } };

/** Country name with its survey year underneath, right-aligned to the axis. */
function CountryTick({ x = 0, y = 0, payload, data }: TickProps & { data: CountryOwnership[] }) {
  const row = data.find((d) => d.country === payload?.value);
  return (
    <g transform={`translate(${x},${y})`}>
      <text
        x={-10}
        y={-3}
        textAnchor="end"
        className={`chart-label text-[13px] ${row?.highlight ? "font-semibold" : ""}`}
      >
        {payload?.value}
      </text>
      <text x={-10} y={12} textAnchor="end" className="chart-label-muted font-mono text-[11px]">
        {row?.surveyYear}
      </text>
    </g>
  );
}

type ValueLabelProps = { x?: number | string; y?: number | string; width?: number | string; height?: number | string; value?: unknown };

function ValueLabel({ x = 0, y = 0, width = 0, height = 0, value }: ValueLabelProps) {
  return (
    <text
      x={Number(x) + Number(width) + 8}
      y={Number(y) + Number(height) / 2}
      dominantBaseline="central"
      className="chart-label font-mono text-[13px] tabular-nums"
    >
      {String(value)}%
    </text>
  );
}

function OwnershipTooltip({ active, payload }: TooltipContentProps<number, string>) {
  if (!active || !payload?.length) return null;
  const row = payload[0].payload as CountryOwnership;
  return (
    <div className="rounded-md border border-rule bg-surface px-3 py-2 text-sm shadow-sm">
      <p className="font-medium text-ink">{row.country}</p>
      <p className="font-mono text-ink tabular-nums">{row.ownership}% own an account</p>
      <p className="text-xs text-muted">Survey year {row.surveyYear}</p>
    </div>
  );
}

/** Horizontal bars, one per country; the highlighted country uses the accent. */
export default function CountryOwnershipBars({ data }: { data: CountryOwnership[] }) {
  return (
    <ResponsiveContainer width="100%" height={CHART_HEIGHT}>
      <BarChart
        data={data}
        layout="vertical"
        margin={{ top: 4, right: 56, bottom: 4, left: 8 }}
        barCategoryGap="28%"
        accessibilityLayer={false}
      >
        <CartesianGrid horizontal={false} strokeDasharray="0" />
        <XAxis
          type="number"
          domain={[0, 30]}
          ticks={[0, 10, 20, 30]}
          tickFormatter={(v: number) => `${v}%`}
          tickLine={false}
          axisLine={false}
          tick={{ fontSize: 12 }}
        />
        <YAxis
          type="category"
          dataKey="country"
          width={84}
          tickLine={false}
          axisLine={false}
          tick={(props: TickProps) => <CountryTick {...props} data={data} />}
        />
        <Tooltip
          cursor={{ className: "recharts-tooltip-cursor" }}
          content={(props) => <OwnershipTooltip {...(props as TooltipContentProps<number, string>)} />}
          isAnimationActive={false}
        />
        <Bar dataKey="ownership" radius={[0, 4, 4, 0]} isAnimationActive={false}>
          {data.map((d) => (
            <Cell key={d.country} className={d.highlight ? "bar-highlight" : "bar-context"} />
          ))}
          <LabelList dataKey="ownership" content={(props) => <ValueLabel {...(props as ValueLabelProps)} />} />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
