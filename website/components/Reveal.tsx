"use client";

import type { ElementType, ReactNode } from "react";
import { useScrollReveal } from "@/hooks/useScrollReveal";

type RevealProps = {
  children: ReactNode;
  as?: ElementType;
  className?: string;
};

/** Wraps children in an element that fades up once on first view. */
export function Reveal({ children, as: Tag = "div", className = "" }: RevealProps) {
  const ref = useScrollReveal<HTMLElement>();
  return (
    <Tag ref={ref} className={`reveal ${className}`.trim()}>
      {children}
    </Tag>
  );
}
