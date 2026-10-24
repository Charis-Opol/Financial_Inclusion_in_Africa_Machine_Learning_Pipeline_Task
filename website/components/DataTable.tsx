import type { ReactNode } from "react";

export type Column<T> = {
  header: string;
  cell: (row: T) => ReactNode;
  align?: "left" | "right";
  /** Monospace, tabular figures. */
  mono?: boolean;
};

type DataTableProps<T> = {
  columns: Column<T>[];
  rows: T[];
  rowKey: (row: T) => string;
  /** Visible caption above the table. */
  caption: string;
  /** Minimum width before the table scrolls inside its container. */
  minWidth?: string;
  rowClassName?: (row: T) => string;
};

/**
 * Generic bordered table. The first column is the row header. Scrolls
 * horizontally inside its own container, never the page.
 */
export function DataTable<T>({
  columns,
  rows,
  rowKey,
  caption,
  minWidth = "28rem",
  rowClassName,
}: DataTableProps<T>) {
  const align = (c: Column<T>) => (c.align === "right" ? "text-right" : "text-left");
  return (
    <div className="overflow-x-auto rounded-lg border border-rule bg-surface">
      <table className="w-full border-collapse text-[0.95rem]" style={{ minWidth }}>
        <caption className="border-b border-rule px-4 py-3 text-left text-sm font-medium text-muted">
          {caption}
        </caption>
        <thead>
          <tr className="border-b border-rule">
            {columns.map((c) => (
              <th key={c.header} scope="col" className={`px-4 py-3 font-medium text-muted ${align(c)}`}>
                {c.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={rowKey(row)} className={`border-b border-rule last:border-0 ${rowClassName?.(row) ?? ""}`}>
              {columns.map((c, i) => {
                const cls = `px-4 py-3 align-top text-ink ${align(c)} ${c.mono ? "font-mono tabular-nums" : ""}`;
                return i === 0 ? (
                  <th key={c.header} scope="row" className={`${cls} font-medium`}>
                    {c.cell(row)}
                  </th>
                ) : (
                  <td key={c.header} className={cls}>
                    {c.cell(row)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
