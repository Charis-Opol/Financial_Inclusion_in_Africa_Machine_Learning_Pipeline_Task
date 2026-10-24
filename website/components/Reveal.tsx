import type { ElementType, ReactNode } from "react";

type RevealProps = {
  children: ReactNode;
  as?: ElementType;
  className?: string;
};

/**
 * Wraps children in an element that fades up once on first view. Plain
 * markup: the single RevealObserver in the layout does the observing.
 */
export function Reveal({ children, as: Tag = "div", className = "" }: RevealProps) {
  return <Tag className={`reveal ${className}`.trim()}>{children}</Tag>;
}
