type FooterProps = {
  title: string;
  author: string;
  repoUrl: string;
  repoLabel: string;
  note: string;
};

/** Site footer: project, author, year, repository link and back-to-top. */
export function Footer({ title, author, repoUrl, repoLabel, note }: FooterProps) {
  // Set at build time; the site is static.
  const year = new Date().getFullYear();
  return (
    <footer className="border-t border-rule">
      <div className="mx-auto flex max-w-6xl flex-col gap-6 px-4 py-12 sm:flex-row sm:items-end sm:justify-between sm:px-6">
        <div>
          <p className="font-semibold text-ink">{title}</p>
          <p className="mt-1 text-sm text-muted">{note}</p>
          <p className="mt-4 text-sm text-muted">
            © {year} {author}
          </p>
        </div>
        <ul className="flex flex-wrap gap-x-6 gap-y-2 text-sm">
          <li>
            <a href={repoUrl} rel="noopener noreferrer" className="font-medium text-accent hover:underline">
              {repoLabel}
              <span aria-hidden="true"> ↗</span>
            </a>
          </li>
          <li>
            <a href="#top" className="text-muted hover:text-ink">
              Back to top<span aria-hidden="true"> ↑</span>
            </a>
          </li>
        </ul>
      </div>
    </footer>
  );
}
