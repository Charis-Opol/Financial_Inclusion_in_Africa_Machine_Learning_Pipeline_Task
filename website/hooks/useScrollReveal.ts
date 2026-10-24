"use client";

import { useEffect, useRef } from "react";

/**
 * Adds `is-visible` to the element the first time it scrolls into view.
 * Pair with the `.reveal` class from globals.css. Under
 * prefers-reduced-motion the CSS never hides anything, so this is a no-op.
 */
export function useScrollReveal<T extends HTMLElement>(
  options: IntersectionObserverInit = { rootMargin: "0px 0px -10% 0px", threshold: 0.1 },
) {
  const ref = useRef<T>(null);
  const { root, rootMargin, threshold } = options;

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    if (typeof IntersectionObserver === "undefined") {
      el.classList.add("is-visible");
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
    observer.observe(el);
    return () => observer.disconnect();
  }, [root, rootMargin, threshold]);

  return ref;
}
