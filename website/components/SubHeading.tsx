/** Heading for a subsection inside a page section. */
export function SubHeading({ id, children }: { id: string; children: string }) {
  return (
    <h3 id={id} className="mb-6 scroll-mt-24 font-sans text-xl font-semibold">
      {children}
    </h3>
  );
}
