"use client";

import { useEffect, useState } from "react";

/**
 * Returns the id of whichever of `ids` currently crosses the upper-middle
 * band of the viewport, or null before any has been reached.
 */
export function useActiveSection(ids: readonly string[], rootMargin = "-30% 0px -60% 0px") {
  const [active, setActive] = useState<string | null>(null);
  const key = ids.join("|");

  useEffect(() => {
    const targets = key
      .split("|")
      .map((id) => document.getElementById(id))
      .filter((el): el is HTMLElement => el !== null);
    if (!targets.length || typeof IntersectionObserver === "undefined") return;

    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) setActive(entry.target.id);
        }
      },
      { rootMargin },
    );
    targets.forEach((el) => observer.observe(el));
    return () => observer.disconnect();
  }, [key, rootMargin]);

  return active;
}
