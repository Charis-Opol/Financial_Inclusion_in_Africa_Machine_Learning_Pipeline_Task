# Project website

A one-page site summarising **Financial Inclusion in Africa: End-to-End ML Pipeline**.
Built with Next.js (App Router, static export), TypeScript and Tailwind CSS v4.

## Run locally

Requires Node 20.9 or newer.

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

All page copy and numbers live in [`data/content.ts`](data/content.ts).
Components in `components/` only render what that file gives them.

## Structure

```
website/
├── app/            layout, page, global styles and theme tokens, favicon
├── components/     Section, StatCard, CodeBlock, Nav, Reveal
├── hooks/          useScrollReveal (IntersectionObserver)
├── data/           content.ts, the single source of page content
└── netlify.toml    Netlify build settings
```

## Deploy

The site lives in the `website/` subfolder of the repo, so point your host at that folder.

- **Netlify:** import the repo and set **Base directory** to `website`. `netlify.toml`
  supplies the build command (`npm run build`) and publish directory (`out`).
- **Vercel:** import the repo and set **Root Directory** to `website`. The framework
  preset is detected as Next.js automatically.

Fuller deploy notes come in the final phase.
