/** Large editorial quote with a supporting line. */
export function PullQuote({ text, detail }: { text: string; detail: string }) {
  return (
    <figure className="border-l-4 border-accent py-1 pl-6">
      <blockquote>
        <p className="text-3xl font-semibold tracking-tight text-ink sm:text-4xl">{text}</p>
      </blockquote>
      <figcaption className="mt-3 max-w-2xl text-lg text-muted">{detail}</figcaption>
    </figure>
  );
}
