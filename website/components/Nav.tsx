"use client";

import { useEffect, useRef, useState } from "react";
import type { SectionMeta } from "@/data/content";

type NavProps = {
  brand: string;
  repoUrl: string;
  sections: SectionMeta[];
};

/**
 * Sticky top bar with anchor links, a thin scroll-progress line and
 * active-section highlighting. On narrow screens the links become a
 * horizontally scrollable strip so nothing wraps or overflows the page.
 */
export function Nav({ brand, repoUrl, sections }: NavProps) {
  const [active, setActive] = useState<string | null>(null);
  const bar = useRef<HTMLDivElement>(null);

  // Scroll progress: write a transform directly to avoid re-rendering.
  useEffect(() => {
    let frame = 0;
    const update = () => {
      frame = 0;
      const max = document.documentElement.scrollHeight - window.innerHeight;
      const progress = max > 0 ? Math.min(1, window.scrollY / max) : 0;
      if (bar.current) bar.current.style.transform = `scaleX(${progress})`;
    };
    const onScroll = () => {
      if (!frame) frame = requestAnimationFrame(update);
    };
    update();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onScroll);
    };
  }, []);

  // Active section: whichever section crosses the upper-middle band of the viewport.
  useEffect(() => {
    const targets = sections
      .map((s) => document.getElementById(s.id))
      .filter((el): el is HTMLElement => el !== null);
    if (!targets.length || typeof IntersectionObserver === "undefined") return;

    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) setActive(entry.target.id);
        }
      },
      { rootMargin: "-30% 0px -60% 0px" },
    );
    targets.forEach((el) => observer.observe(el));
    return () => observer.disconnect();
  }, [sections]);

  return (
    <header className="sticky top-0 z-40 border-b border-rule bg-bg/90 backdrop-blur supports-[backdrop-filter]:bg-bg/75">
      <nav aria-label="Primary" className="mx-auto flex max-w-6xl items-center gap-6 px-4 sm:px-6">
        <a
          href="#top"
          className="shrink-0 py-4 font-display text-base font-semibold tracking-tight text-ink"
        >
          {brand}
        </a>
        <ul className="-mr-4 flex min-w-0 flex-1 gap-1 overflow-x-auto py-2 [scrollbar-width:none] sm:mr-0 md:justify-end [&::-webkit-scrollbar]:hidden">
          {sections.map((s) => {
            const isActive = active === s.id;
            return (
              <li key={s.id} className="shrink-0">
                <a
                  href={`#${s.id}`}
                  aria-current={isActive ? "location" : undefined}
                  className={`block rounded-sm px-2.5 py-2 text-sm transition-colors ${
                    isActive ? "text-accent" : "text-muted hover:text-ink"
                  }`}
                >
                  {s.navLabel}
                </a>
              </li>
            );
          })}
          <li className="shrink-0 pr-4 sm:pr-0">
            <a
              href={repoUrl}
              className="block rounded-sm px-2.5 py-2 text-sm font-medium text-accent hover:underline"
              rel="noopener noreferrer"
            >
              GitHub<span className="sr-only"> repository (opens external site)</span>
            </a>
          </li>
        </ul>
      </nav>
      <div aria-hidden="true" className="absolute inset-x-0 bottom-[-1px] h-0.5">
        <div
          ref={bar}
          className="h-full origin-left bg-accent"
          style={{ transform: "scaleX(0)" }}
        />
      </div>
    </header>
  );
}
