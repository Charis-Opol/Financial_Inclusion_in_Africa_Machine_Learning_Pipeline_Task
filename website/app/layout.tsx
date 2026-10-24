import type { Metadata, Viewport } from "next";
import { IBM_Plex_Mono, Poppins } from "next/font/google";
import { Nav } from "@/components/Nav";
import { sections, site } from "@/data/content";
import "./globals.css";

// Poppins for headings and body (italic for the hero accent line),
// monospace for metrics and code.
const sans = Poppins({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  style: ["normal", "italic"],
  variable: "--font-sans-face",
  display: "swap",
});

const mono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500"],
  variable: "--font-mono-face",
  display: "swap",
});

export const metadata: Metadata = {
  title: site.title,
  description: site.description,
  authors: [{ name: site.author }],
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
      className={`${sans.variable} ${mono.variable}`}
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
      </body>
    </html>
  );
}
