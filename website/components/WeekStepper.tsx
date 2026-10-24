"use client";

import { useRef, type KeyboardEvent } from "react";
import { useActiveSection } from "@/hooks/useActiveSection";

export type Step = {
  id: string;
  label: string;
  description: string;
};

/**
 * Vertical stepper linking to each week. Tab reaches it; arrow keys and
 * Home/End move between steps; Enter follows the link. The step in view
 * is marked with aria-current="step".
 */
export function WeekStepper({ steps }: { steps: Step[] }) {
  const active = useActiveSection(steps.map((s) => s.id));
  const links = useRef<(HTMLAnchorElement | null)[]>([]);

  function onKeyDown(e: KeyboardEvent<HTMLOListElement>) {
    const current = links.current.findIndex((el) => el === document.activeElement);
    if (current === -1) return;
    const last = steps.length - 1;
    const next =
      e.key === "ArrowDown" || e.key === "ArrowRight"
        ? Math.min(current + 1, last)
        : e.key === "ArrowUp" || e.key === "ArrowLeft"
          ? Math.max(current - 1, 0)
          : e.key === "Home"
            ? 0
            : e.key === "End"
              ? last
              : null;
    if (next === null) return;
    e.preventDefault();
    links.current[next]?.focus();
  }

  const activeIndex = steps.findIndex((s) => s.id === active);

  return (
    <nav aria-label="Weeks">
      <ol onKeyDown={onKeyDown} className="relative">
        {steps.map((step, i) => {
          const isActive = i === activeIndex;
          const isPast = activeIndex > i;
          return (
            <li key={step.id} className="relative pb-8 pl-9 last:pb-0">
              {i < steps.length - 1 && (
                <span
                  aria-hidden="true"
                  className={`absolute top-7 bottom-1 left-[0.6875rem] w-px transition-colors ${
                    isPast ? "bg-accent" : "bg-rule"
                  }`}
                />
              )}
              <span
                aria-hidden="true"
                className={`absolute top-0.5 left-0 grid size-6 place-items-center rounded-full border font-mono text-[11px] transition-colors ${
                  isActive
                    ? "border-accent bg-accent text-accent-ink"
                    : isPast
                      ? "border-accent text-accent"
                      : "border-rule text-muted"
                }`}
              >
                {i + 1}
              </span>
              <a
                ref={(el) => {
                  links.current[i] = el;
                }}
                href={`#${step.id}`}
                aria-current={isActive ? "step" : undefined}
                className="group block rounded-sm"
              >
                <span
                  className={`block font-mono text-xs uppercase tracking-[0.14em] ${
                    isActive ? "text-accent" : "text-muted group-hover:text-ink"
                  }`}
                >
                  {step.label}
                </span>
                <span className={`mt-0.5 block text-sm ${isActive ? "text-ink" : "text-muted group-hover:text-ink"}`}>
                  {step.description}
                </span>
              </a>
            </li>
          );
        })}
      </ol>
    </nav>
  );
}
