import type { Metadata, Viewport } from "next";
import { IBM_Plex_Mono, Poppins } from "next/font/google";
import { Nav } from "@/components/Nav";
import { RevealObserver } from "@/components/RevealObserver";
import { sections, site } from "@/data/content";
import "./globals.css";

// Poppins for headings and body, monospace for metrics and code.
const sans = Poppins({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-sans-face",
  display: "swap",
});

// Italic is only used for the hero accent line: one file, not preloaded,
// so it does not compete with the upright faces on first load.
const sansItalic = Poppins({
  subsets: ["latin"],
  weight: "400",
  style: "italic",
  variable: "--font-italic-face",
  display: "swap",
  preload: false,
});

const mono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500"],
  variable: "--font-mono-face",
  display: "swap",
});

/**
 * Base for absolute Open Graph URLs: site.url if set, else the URL the host
 * provides at build time (Netlify: URL, Vercel: VERCEL_PROJECT_PRODUCTION_URL).
 */
function siteBase(): URL {
  const vercel = process.env.VERCEL_PROJECT_PRODUCTION_URL;
  const url = site.url || process.env.URL || (vercel ? `https://${vercel}` : "") || "http://localhost:3000";
  return new URL(url);
}

export const metadata: Metadata = {
  metadataBase: siteBase(),
  title: site.title,
  description: site.description,
  authors: [{ name: site.author }],
  openGraph: {
    type: "website",
    title: site.title,
    description: site.description,
    siteName: site.shortTitle,
    locale: "en",
  },
  twitter: {
    card: "summary_large_image",
    title: site.title,
    description: site.description,
  },
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#f6f2ea" },
    { media: "(prefers-color-scheme: dark)", color: "#13120f" },
  ],
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html
      lang="en"
      className={`${sans.variable} ${sansItalic.variable} ${mono.variable}`}
      suppressHydrationWarning
    >
      <head>
        {/* Enables scroll-reveal styles only when JS runs; see globals.css. */}
        <script dangerouslySetInnerHTML={{ __html: "document.documentElement.classList.add('js')" }} />
      </head>
      <body className="min-h-dvh">
        <a
          href="#main"
          className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-50 focus:rounded-md focus:bg-accent focus:px-4 focus:py-2 focus:text-accent-ink"
        >
          Skip to content
        </a>
        <Nav brand={site.shortTitle} repoUrl={site.repoUrl} sections={sections} />
        {children}
        <RevealObserver />
      </body>
    </html>
  );
}
