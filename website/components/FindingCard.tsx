import type { Finding } from "@/data/content";
import { RichText } from "./RichText";

/** One EDA finding and the preprocessing decision it led to. */
export function FindingCard({ index, item }: { index: number; item: Finding }) {
  return (
    <article className="flex h-full flex-col rounded-lg border border-rule bg-surface p-5 sm:p-6">
      <header className="flex items-baseline gap-3">
        <span className="font-mono text-xs text-muted tabular-nums">{String(index).padStart(2, "0")}</span>
        <h4 className="text-lg font-semibold">{item.title}</h4>
      </header>
      <dl className="mt-4 flex flex-1 flex-col gap-4 text-[0.95rem]">
        <div>
          <dt className="font-mono text-[11px] uppercase tracking-[0.14em] text-muted">Finding</dt>
          <dd className="mt-1 text-ink">
            <RichText text={item.finding} />
          </dd>
        </div>
        <div className="border-l-2 border-accent pl-3">
          <dt className="font-mono text-[11px] uppercase tracking-[0.14em] text-accent">Decision</dt>
          <dd className="mt-1 text-ink">
            <RichText text={item.decision} />
          </dd>
        </div>
      </dl>
    </article>
  );
}
