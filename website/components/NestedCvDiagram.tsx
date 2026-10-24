/**
 * Hand-drawn nested-CV diagram: five outer folds (one held out per row),
 * then what happens inside each outer fold. Colours come from theme
 * tokens via Tailwind fill/stroke utilities, so it follows dark mode.
 * On very narrow screens it scrolls inside its container rather than
 * shrinking the text below legibility.
 */

const FOLDS = 5;
const ROW_Y0 = 30;
const ROW_STEP = 30;
const ROW_H = 22;
const BAR_X = 64;
const BAR_W = 376;
const GAP = 3;
const SEG_W = (BAR_W - GAP * (FOLDS - 1)) / FOLDS;

const segX = (j: number) => BAR_X + j * (SEG_W + GAP);

export function NestedCvDiagram({
  title,
  description,
  rows,
}: {
  title: string;
  description: string;
  rows: string;
}) {
  return (
    <div className="overflow-x-auto">
      <svg
        viewBox="0 0 440 524"
        role="img"
        aria-labelledby="ncv-title ncv-desc"
        className="block h-auto w-full min-w-[22rem]"
      >
        <title id="ncv-title">{title}</title>
        <desc id="ncv-desc">{description}</desc>

        <defs>
          <marker id="ncv-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">
            <path d="M0,0 L8,4 L0,8 z" className="fill-muted" />
          </marker>
          <marker id="ncv-arrow-accent" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">
            <path d="M0,0 L8,4 L0,8 z" className="fill-(--chart-highlight)" />
          </marker>
        </defs>

        {/* Header */}
        <text x={0} y={14} className="fill-muted font-mono text-[10.5px] tracking-[0.12em]">
          OUTER LOOP · 5 STRATIFIED FOLDS
        </text>
        <text x={440} y={14} textAnchor="end" className="fill-muted font-mono text-[10.5px]">
          {rows}
        </text>

        {/* Outer folds: one held-out segment per row */}
        {Array.from({ length: FOLDS }, (_, i) => {
          const y = ROW_Y0 + i * ROW_STEP;
          return (
            <g key={i}>
              <text x={0} y={y + 15} className="fill-ink font-mono text-[11px]">
                Fold {i + 1}
              </text>
              {Array.from({ length: FOLDS }, (_, j) =>
                j === i ? (
                  <rect key={j} x={segX(j)} y={y} width={SEG_W} height={ROW_H} rx={3} className="fill-(--chart-highlight)" />
                ) : (
                  <rect
                    key={j}
                    x={segX(j)}
                    y={y}
                    width={SEG_W}
                    height={ROW_H}
                    rx={3}
                    className="fill-surface-2 stroke-rule"
                  />
                ),
              )}
            </g>
          );
        })}

        {/* Legend */}
        <rect x={BAR_X} y={186} width={11} height={11} rx={2} className="fill-(--chart-highlight)" />
        <text x={BAR_X + 17} y={195.5} className="fill-ink text-[12px]">
          Untouched validation fold
        </text>
        <rect x={264} y={186} width={11} height={11} rx={2} className="fill-surface-2 stroke-rule" />
        <text x={281} y={195.5} className="fill-ink text-[12px]">
          Training folds
        </text>

        {/* Inside each outer fold */}
        <rect x={0.5} y={214.5} width={439} height={250} rx={10} className="fill-none stroke-rule" strokeDasharray="4 4" />
        <text x={16} y={238} className="fill-muted font-mono text-[10.5px] tracking-[0.12em]">
          INSIDE EACH OUTER FOLD
        </text>

        <rect x={16} y={250} width={300} height={30} rx={4} className="fill-surface-2 stroke-rule" />
        <text x={166} y={269.5} textAnchor="middle" className="fill-ink text-[12.5px]">
          4 training folds
        </text>
        <rect x={328} y={250} width={96} height={30} rx={4} className="fill-(--chart-highlight)" />
        <text x={376} y={269.5} textAnchor="middle" className="fill-accent-ink text-[12.5px] font-medium">
          Held out
        </text>

        <line x1={166} y1={281} x2={166} y2={298} className="stroke-muted" markerEnd="url(#ncv-arrow)" />

        <rect x={16} y={300} width={300} height={76} rx={6} className="fill-surface stroke-rule" />
        <text x={30} y={322} className="fill-ink text-[12.5px] font-semibold">
          Inner stratified CV + Optuna
        </text>
        <text x={30} y={342} className="fill-muted text-[12px]">
          Searches hyperparameters on these
        </text>
        <text x={30} y={358} className="fill-muted text-[12px]">
          folds only; SMOTE is fit inside them.
        </text>

        <line x1={166} y1={377} x2={166} y2={394} className="stroke-muted" markerEnd="url(#ncv-arrow)" />

        <rect x={16} y={396} width={300} height={52} rx={6} className="fill-surface stroke-rule" />
        <text x={30} y={417} className="fill-ink text-[12.5px] font-semibold">
          Refit with the best parameters
        </text>
        <text x={30} y={435} className="fill-muted text-[12px]">
          on all 4 outer training folds
        </text>

        {/* Held-out fold is only touched once, at scoring time */}
        <line
          x1={376}
          y1={281}
          x2={376}
          y2={394}
          strokeDasharray="3 3"
          className="stroke-(--chart-highlight)"
          markerEnd="url(#ncv-arrow-accent)"
        />
        <line x1={317} y1={422} x2={326} y2={422} className="stroke-muted" markerEnd="url(#ncv-arrow)" />
        <rect x={328} y={396} width={96} height={52} rx={6} className="fill-surface stroke-(--chart-highlight)" strokeWidth={1.5} />
        <text x={376} y={417} textAnchor="middle" className="fill-ink text-[12.5px] font-semibold">
          Score once
        </text>
        <text x={376} y={435} textAnchor="middle" className="fill-muted font-mono text-[10.5px]">
          PR-AUC, F1 …
        </text>

        {/* Aggregate */}
        <line x1={220} y1={465} x2={220} y2={482} className="stroke-muted" markerEnd="url(#ncv-arrow)" />
        <rect x={0} y={484} width={440} height={38} rx={6} className="fill-accent-soft" />
        <text x={220} y={507.5} textAnchor="middle" className="fill-ink text-[12.5px] font-semibold">
          Repeat for all 5 outer folds → mean ± std
        </text>
      </svg>
      <p className="mt-2 text-xs text-muted sm:hidden">Scroll sideways to see the full diagram.</p>
    </div>
  );
}
