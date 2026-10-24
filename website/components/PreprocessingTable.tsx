import type { PreprocessingRow } from "@/data/content";
import { RichText } from "./RichText";

/**
 * Decision per concern for each model family. A shared decision spans both
 * columns. The table scrolls inside its own container on narrow screens.
 */
export function PreprocessingTable({ rows, labelledBy }: { rows: PreprocessingRow[]; labelledBy: string }) {
  return (
    <div className="overflow-x-auto rounded-lg border border-rule bg-surface">
      <table aria-labelledby={labelledBy} className="w-full min-w-[34rem] border-collapse text-left text-[0.95rem]">
        <thead>
          <tr className="border-b border-rule">
            <th scope="col" className="px-4 py-3 font-medium text-muted">Concern</th>
            <th scope="col" className="px-4 py-3 font-medium text-muted">XGBoost</th>
            <th scope="col" className="px-4 py-3 font-medium text-muted">PyTorch MLP</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.concern} className="border-b border-rule last:border-0">
              <th scope="row" className="w-[32%] px-4 py-3 align-top font-medium text-ink">
                {row.concern}
              </th>
              {"shared" in row ? (
                <td colSpan={2} className="px-4 py-3 align-top text-ink">
                  <RichText text={row.shared} />
                  <span className="ml-2 font-mono text-[11px] uppercase tracking-[0.12em] text-muted">
                    both
                  </span>
                </td>
              ) : (
                <>
                  <td className="px-4 py-3 align-top text-ink">
                    <RichText text={row.xgboost} />
                  </td>
                  <td className="px-4 py-3 align-top text-ink">
                    <RichText text={row.pytorch} />
                  </td>
                </>
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
