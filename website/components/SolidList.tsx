import type { SolidPrinciple } from "@/data/content";
import { RichText } from "./RichText";

/** The five SOLID principles, each with how the codebase applies it. */
export function SolidList({ items }: { items: SolidPrinciple[] }) {
  return (
    <ul className="grid gap-px overflow-hidden rounded-lg border border-rule bg-rule sm:grid-cols-2">
      {items.map((p) => (
        <li
          key={p.letter}
          className="grid grid-cols-[2.75rem_minmax(0,1fr)] gap-x-3 bg-surface p-5 last:sm:col-span-2"
        >
          <span aria-hidden="true" className="font-display text-4xl leading-none font-semibold text-accent">
            {p.letter}
          </span>
          <div>
            <p className="font-semibold text-ink">{p.name}</p>
            <p className="mt-1 text-[0.95rem] text-muted">
              <RichText text={p.line} />
            </p>
          </div>
        </li>
      ))}
    </ul>
  );
}
