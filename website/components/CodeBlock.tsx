"use client";

import { useEffect, useRef, useState } from "react";

type CodeBlockProps = {
  code: string;
  /** Shown in the header bar, e.g. "bash" or "POST /predict". */
  label?: string;
  className?: string;
};

type CopyState = "idle" | "copied" | "failed";

/** Preformatted code with a header and a copy-to-clipboard button. */
export function CodeBlock({ code, label, className = "" }: CodeBlockProps) {
  const [state, setState] = useState<CopyState>("idle");
  const timer = useRef<ReturnType<typeof setTimeout>>(undefined);

  useEffect(() => () => clearTimeout(timer.current), []);

  async function copy() {
    try {
      await navigator.clipboard.writeText(code);
      setState("copied");
    } catch {
      setState("failed");
    }
    clearTimeout(timer.current);
    timer.current = setTimeout(() => setState("idle"), 2000);
  }

  const buttonText = state === "copied" ? "Copied" : state === "failed" ? "Copy failed" : "Copy";

  return (
    <figure className={`overflow-hidden rounded-lg border border-rule bg-surface ${className}`.trim()}>
      <div className="flex items-center justify-between gap-4 border-b border-rule px-4 py-2">
        <figcaption className="font-mono text-xs text-muted">{label ?? "code"}</figcaption>
        <button
          type="button"
          onClick={copy}
          className="rounded-sm px-2 py-1 font-mono text-xs text-accent transition-colors hover:bg-accent-soft"
          aria-label={label ? `Copy ${label} to clipboard` : "Copy code to clipboard"}
        >
          {buttonText}
        </button>
        <span className="sr-only" aria-live="polite">
          {state === "copied" ? "Copied to clipboard" : state === "failed" ? "Copy failed" : ""}
        </span>
      </div>
      <pre className="overflow-x-auto p-4 font-mono text-sm leading-relaxed text-ink">
        <code>{code}</code>
      </pre>
    </figure>
  );
}
