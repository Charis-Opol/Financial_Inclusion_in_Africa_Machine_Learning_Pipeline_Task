import type { Hero as HeroContent, LinkButton } from "@/data/content";
import { StatCard } from "./StatCard";

function ButtonLink({ button, variant }: { button: LinkButton; variant: "primary" | "secondary" }) {
  const base =
    "inline-flex items-center justify-center gap-2 rounded-md px-5 py-3 text-sm font-medium transition-colors";
  const styles =
    variant === "primary"
      ? "bg-accent text-accent-ink hover:bg-ink hover:text-bg"
      : "border border-rule text-ink hover:border-accent hover:text-accent";
  return (
    <a
      href={button.href}
      className={`${base} ${styles}`}
      {...(button.external ? { rel: "noopener noreferrer" } : {})}
    >
      {button.label}
      <span aria-hidden="true">{button.external ? "↗" : "↓"}</span>
    </a>
  );
}

/** Page header: title, subtitle, calls to action and the key-stat strip. */
export function Hero({ content }: { content: HeroContent }) {
  const [lead, tail] = content.titleLines;
  return (
    <div id="top" className="scroll-mt-24 pb-16 pt-16 sm:pb-24 sm:pt-24">
      <p className="mb-6 font-mono text-xs uppercase tracking-[0.18em] text-accent">{content.eyebrow}</p>
      <h1 className="max-w-4xl text-4xl font-semibold sm:text-5xl lg:text-6xl">
        <span className="block">{lead}</span>
        <span className="block font-italic font-normal italic text-accent">{tail}</span>
      </h1>
      <p className="mt-8 max-w-2xl text-lg text-muted sm:text-xl">{content.subtitle}</p>

      <div className="mt-10 flex flex-col gap-3 sm:flex-row">
        <ButtonLink button={content.primary} variant="primary" />
        <ButtonLink button={content.secondary} variant="secondary" />
      </div>

      <div className="mt-16 grid grid-cols-2 gap-x-6 gap-y-8 border-t border-rule pt-8 sm:grid-cols-3 lg:grid-cols-5">
        {content.stats.map((stat) => (
          <StatCard key={stat.label} value={stat.value} label={stat.label} />
        ))}
      </div>
    </div>
  );
}
