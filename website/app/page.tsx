import { Section } from "@/components/Section";
import { sections, site } from "@/data/content";

export default function Home() {
  return (
    <main id="main" className="mx-auto max-w-6xl px-4 sm:px-6">
      {/* Hero placeholder: built out in Phase 1. */}
      <div id="top" className="scroll-mt-24 py-20 sm:py-28">
        <h1 className="max-w-4xl text-4xl font-semibold sm:text-6xl">{site.title}</h1>
      </div>

      {sections.map((s) => (
        <Section key={s.id} id={s.id} eyebrow={s.eyebrow} title={s.title} />
      ))}
    </main>
  );
}
