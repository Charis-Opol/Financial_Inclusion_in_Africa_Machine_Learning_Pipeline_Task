# Project website

A one-page site summarising **Financial Inclusion in Africa: End-to-End ML Pipeline**:
the data and evaluation work (Week 1), deployment and MLOps (Week 2) and
interpretability and explainability (Week 3).

Built with Next.js (App Router, static export), TypeScript, Tailwind CSS v4 and
Recharts. There is no server, no API routes and no environment variables to set:
`npm run build` writes a plain static site to `out/`.

## Run locally

Requires Node 20.9 or newer (Netlify builds with Node 22).

```bash
cd website
npm install
npm run dev        # http://localhost:3000
```

```bash
npm run build      # static site written to website/out/
npm run start      # serve out/ locally to check the production build
npm run typecheck  # TypeScript only
```

## Editing content

All page copy, numbers, figure captions and alt text live in
[`data/content.ts`](data/content.ts). Components in `components/` only render what
that file gives them, so text can be changed without touching layout code.
Text in `backticks` inside a content string renders as inline code.

## Figures

The images in `public/figures/` are WebP exports of plots from
`reports/figures/`. To regenerate them after re-running the analysis, from the
repository root:

```bash
.venv-xai/Scripts/python website/scripts/export_figures.py .
```

If a figure's content changes, update its caption and alt text in `data/content.ts`.

## Structure

```text
website/
├── app/                 layout (fonts, metadata), page, theme tokens,
│                        favicon, Open Graph / Twitter card images
├── components/          sections (Hero, Week1–3, …) and shared pieces
│   └── charts/          Recharts charts, loaded only when scrolled near
├── hooks/               useScrollReveal, useActiveSection
├── data/content.ts      the single source of page content
├── public/figures/      figures exported from the notebooks
└── scripts/             export_figures.py
```

## Deploy

The site lives in the `website/` subfolder of the repository.

### Netlify

1. In Netlify, choose **Add new site → Import an existing project** and pick this repository.
2. Leave the build settings empty. The `netlify.toml` at the repository root sets
   the base directory (`website`), build command (`npm run build`) and publish
   directory (`out`).
3. Deploy.

### Vercel

1. In Vercel, choose **Add New → Project** and import this repository.
2. Set **Root Directory** to `website`. The framework preset is detected as Next.js.
3. Deploy. The static export in `out/` is served as-is.

### Social previews

Open Graph and Twitter tags are generated from `app/opengraph-image.png` and
`app/twitter-image.png`. Absolute image URLs use, in order: `site.url` in
`data/content.ts` if set, otherwise the URL Netlify (`URL`) or Vercel
(`VERCEL_PROJECT_PRODUCTION_URL`) provides at build time. Set `site.url` if you
use a custom domain.

## Quality notes

- Light and dark mode follow `prefers-color-scheme`; all colours are tokens in
  `app/globals.css`.
- Charts have a text summary for screen readers and a "Show data table" toggle;
  diagrams are inline SVG with a title and description.
- Scroll-reveal animation is skipped under `prefers-reduced-motion`, and content
  is never hidden when JavaScript is off.
- Tables and code blocks scroll inside their own containers from 360px up.
