import type { ResultRow } from "@/data/content";
import { meanStd } from "@/lib/format";

const METRICS = [
  { key: "prAuc", label: "PR-AUC" },
  { key: "f1", label: "F1" },
  { key: "rocAuc", label: "ROC-AUC" },
] as const;

/**
 * Cross-validated results, mean ± std. The winning row gets a tinted
 * background, an accent rule and a "Shipped" tag, so it is not marked by
 * colour alone. Scrolls inside its own container on narrow screens.
 */
export function ResultsTable({ rows, labelledBy }: { rows: ResultRow[]; labelledBy: string }) {
  return (
    <div className="overflow-x-auto rounded-lg border border-rule bg-surface">
      <table aria-labelledby={labelledBy} className="w-full min-w-[40rem] border-collapse text-left text-[0.95rem]">
        <thead>
          <tr className="border-b border-rule">
            <th scope="col" className="px-4 py-3 font-medium text-muted">Model</th>
            <th scope="col" className="px-4 py-3 font-medium text-muted">Imbalance</th>
            {METRICS.map((m) => (
              <th key={m.key} scope="col" className="px-4 py-3 text-right font-medium text-muted">
                {m.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr
              key={row.id}
              className={`border-b border-rule last:border-0 ${
                row.winner ? "bg-accent-soft shadow-[inset_3px_0_0_var(--accent)]" : ""
              }`}
            >
              <th scope="row" className="px-4 py-3 font-medium text-ink">
                {row.model}
                {row.winner && (
                  <span className="ml-2 rounded-sm bg-accent px-1.5 py-0.5 align-middle font-mono text-[10px] uppercase tracking-[0.12em] text-accent-ink">
                    Shipped
                  </span>
                )}
              </th>
              <td className="px-4 py-3 text-ink">{row.imbalance}</td>
              {METRICS.map((m) => (
                <td
                  key={m.key}
                  className={`whitespace-nowrap px-4 py-3 text-right font-mono tabular-nums ${
                    row.winner && m.key === "prAuc" ? "font-semibold text-ink" : "text-ink"
                  }`}
                >
                  {meanStd(row[m.key])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
