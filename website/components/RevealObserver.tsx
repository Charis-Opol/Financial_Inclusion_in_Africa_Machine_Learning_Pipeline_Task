"use client";

import { useScrollReveal } from "@/hooks/useScrollReveal";

/** Mounted once in the layout; drives every `.reveal` element on the page. */
export function RevealObserver() {
  useScrollReveal();
  return null;
}
