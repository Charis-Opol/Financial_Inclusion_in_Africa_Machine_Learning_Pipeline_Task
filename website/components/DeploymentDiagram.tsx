import type { Week2Content } from "@/data/content";

type Diagram = Week2Content["tradeoffs"]["diagram"];

/**
 * Hand-drawn hybrid deployment diagram: cloud outside the Uganda boundary,
 * hub and edge inside it. Vertical so it stays legible at phone width.
 */
export function DeploymentDiagram({ d }: { d: Diagram }) {
  return (
    <svg viewBox="0 0 360 446" role="img" aria-labelledby="dep-title dep-desc" className="block h-auto w-full">
      <title id="dep-title">{d.title}</title>
      <desc id="dep-desc">{d.description}</desc>

      <defs>
        <marker id="dep-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">
          <path d="M0,0 L8,4 L0,8 z" className="fill-muted" />
        </marker>
      </defs>

      {/* Cloud: outside Uganda */}
      <rect x={60.5} y={8.5} width={239} height={74} rx={10} className="fill-surface stroke-rule" />
      <text x={180} y={34} textAnchor="middle" className="fill-ink text-[13px] font-semibold">
        {d.cloud.title}
      </text>
      {d.cloud.lines.map((line, i) => (
        <text key={line} x={180} y={53 + i * 16} textAnchor="middle" className="fill-muted text-[12px]">
          {line}
        </text>
      ))}

      {/* Uganda boundary */}
      <rect x={6.5} y={118.5} width={347} height={320} rx={12} strokeDasharray="5 4" className="fill-none stroke-accent" />
      <text x={22} y={426} className="fill-accent font-mono text-[9.5px] tracking-[0.08em]">
        {d.boundary}
      </text>

      {/* Hub: system of record */}
      <rect x={30.5} y={140.5} width={299} height={92} rx={10} strokeWidth={1.5} className="fill-accent-soft stroke-accent" />
      <text x={48} y={166} className="fill-ink text-[13px] font-semibold">
        {d.hub.title}
      </text>
      {d.hub.lines.map((line, i) => (
        <text key={line} x={48} y={188 + i * 18} className="fill-ink text-[12px]">
          {line}
        </text>
      ))}

      {/* Edge: field devices */}
      <rect x={30.5} y={316.5} width={299} height={84} rx={10} className="fill-surface stroke-rule" />
      <text x={48} y={342} className="fill-ink text-[13px] font-semibold">
        {d.edge.title}
      </text>
      {d.edge.lines.map((line, i) => (
        <text key={line} x={48} y={364 + i * 18} className="fill-muted text-[12px]">
          {line}
        </text>
      ))}

      {/* Hub ⇢ cloud: no personal data crosses the boundary */}
      <line x1={180} y1={138} x2={180} y2={86} strokeDasharray="4 4" className="stroke-muted" markerEnd="url(#dep-arrow)" />
      <rect x={190} y={92} width={112} height={20} rx={4} className="fill-surface" />
      <text x={196} y={106} className="fill-muted font-mono text-[11px]">
        {d.cloudLink}
      </text>

      {/* Hub → edge: the ONNX model */}
      <line x1={180} y1={234} x2={180} y2={313} className="stroke-muted" markerEnd="url(#dep-arrow)" />
      <text x={190} y={278} className="fill-muted font-mono text-[11px]">
        {d.edgeLink}
      </text>
    </svg>
  );
}
