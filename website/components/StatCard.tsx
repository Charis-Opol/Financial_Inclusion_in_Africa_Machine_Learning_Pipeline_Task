type StatCardProps = {
  /** The headline figure, rendered large in monospace (e.g. "0.590"). */
  value: string;
  label: string;
  note?: string;
  className?: string;
};

/** A key number with a short label. Uses <dl> semantics for screen readers. */
export function StatCard({ value, label, note, className = "" }: StatCardProps) {
  return (
    // column-reverse + justify-end pins the value to the top, so values in a row
    // line up even when one card has an extra note.
    <dl className={`flex flex-col-reverse justify-end gap-1 border-l border-rule pl-4 ${className}`.trim()}>
      <dt className="text-sm text-muted">
        {label}
        {note && <span className="block text-xs">{note}</span>}
      </dt>
      <dd className="font-mono text-3xl font-medium tracking-tight text-ink tabular-nums sm:text-4xl">
        {value}
      </dd>
    </dl>
  );
}
