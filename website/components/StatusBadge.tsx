import type { DriftLevel } from "@/data/content";

const STYLES: Record<DriftLevel, { cls: string; icon: string }> = {
  stable: { cls: "bg-accent-soft text-accent", icon: "✓" },
  moderate: { cls: "bg-warn-soft text-warn", icon: "!" },
  significant: { cls: "bg-danger-soft text-danger", icon: "▲" },
};

/** Drift verdict pill: colour, a distinct icon and text, never colour alone. */
export function StatusBadge({ level, children }: { level: DriftLevel; children: string }) {
  const { cls, icon } = STYLES[level];
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-sm px-2 py-0.5 text-sm font-medium whitespace-nowrap ${cls}`}>
      <span aria-hidden="true" className="text-[0.8em]">
        {icon}
      </span>
      {children}
    </span>
  );
}
