"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";

/**
 * Renders a fixed-height placeholder until the slot nears the viewport,
 * then mounts `children`. Keeps chart code (Recharts) out of the initial
 * page load entirely; the reserved height means no layout shift.
 */
export function LazyWhenVisible({ height, children }: { height: number; children: ReactNode }) {
  const ref = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (typeof IntersectionObserver === "undefined") {
      setVisible(true);
      return;
    }
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setVisible(true);
          observer.disconnect();
        }
      },
      { rootMargin: "400px 0px" },
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  return (
    <div ref={ref} style={{ minHeight: height }}>
      {visible ? children : null}
    </div>
  );
}
