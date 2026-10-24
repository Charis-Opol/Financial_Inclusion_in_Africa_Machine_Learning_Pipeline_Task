"use client";

import { useEffect } from "react";

/**
 * Adds `is-visible` to every element matching `selector` the first time it
 * scrolls into view. One observer serves the whole page, so `.reveal`
 * wrappers can stay plain server-rendered markup. Under
 * prefers-reduced-motion the CSS never hides anything, so this is a no-op.
 */
export function useScrollReveal(
  selector = ".reveal",
  options: IntersectionObserverInit = { rootMargin: "0px 0px -10% 0px", threshold: 0.1 },
) {
  const { root, rootMargin, threshold } = options;

  useEffect(() => {
    const els = document.querySelectorAll<HTMLElement>(selector);
    if (typeof IntersectionObserver === "undefined") {
      els.forEach((el) => el.classList.add("is-visible"));
      return;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-visible");
            observer.unobserve(entry.target);
          }
        }
      },
      { root, rootMargin, threshold },
    );
    els.forEach((el) => observer.observe(el));
    return () => observer.disconnect();
  }, [selector, root, rootMargin, threshold]);
}
