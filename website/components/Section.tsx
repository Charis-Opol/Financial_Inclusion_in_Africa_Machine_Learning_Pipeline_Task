import type { ReactNode } from "react";
import { Reveal } from "./Reveal";

type SectionProps = {
  id: string;
  eyebrow?: string;
  title: string;
  intro?: ReactNode;
  children?: ReactNode;
  className?: string;
};

/**
 * A top-level page section: anchor target, thin top rule, eyebrow,
 * heading and optional intro. `scroll-mt` clears the sticky nav.
 * `.section-lazy` (globals.css) skips layout until the section nears the
 * viewport, which keeps the long page's first render cheap on phones.
 */
export function Section({ id, eyebrow, title, intro, children, className = "" }: SectionProps) {
  const headingId = `${id}-heading`;
  return (
    <section
      id={id}
      aria-labelledby={headingId}
      className={`section-lazy scroll-mt-24 border-t border-rule py-16 sm:py-24 ${className}`.trim()}
    >
      <Reveal className="max-w-3xl">
        {eyebrow && (
          <p className="mb-3 font-mono text-xs uppercase tracking-[0.18em] text-accent">{eyebrow}</p>
        )}
        <h2 id={headingId} className="text-3xl font-semibold sm:text-4xl">
          {title}
        </h2>
        {intro && <div className="mt-5 text-lg text-muted">{intro}</div>}
      </Reveal>
      {children && <div className="mt-10">{children}</div>}
    </section>
  );
}
