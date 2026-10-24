/** Renders a content string, turning `backticked` spans into inline code. */
export function RichText({ text }: { text: string }) {
  const parts = text.split("`");
  return (
    <>
      {parts.map((part, i) =>
        i % 2 === 1 ? (
          <code
            key={i}
            className="rounded-sm bg-surface-2 px-1 py-0.5 font-mono text-[0.86em] text-ink wrap-anywhere"
          >
            {part}
          </code>
        ) : (
          part
        ),
      )}
    </>
  );
}
