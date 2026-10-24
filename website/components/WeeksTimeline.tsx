import type { ReactNode } from "react";
import { WeekStepper, type Step } from "./WeekStepper";

/**
 * Two-column shell for the three weeks: a sticky side stepper on desktop,
 * plain stacked week sections on mobile (each week has its own heading).
 */
export function WeeksTimeline({ steps, children }: { steps: Step[]; children: ReactNode }) {
  return (
    <div className="lg:grid lg:grid-cols-[12rem_minmax(0,1fr)] lg:gap-12">
      <aside className="hidden border-t border-rule lg:block">
        <div className="sticky top-28 pt-24">
          <WeekStepper steps={steps} />
        </div>
      </aside>
      <div className="min-w-0">{children}</div>
    </div>
  );
}
