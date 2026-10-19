"""Builds reports/xai_comparison_report.pdf from reports/xai_comparison_report.md.

Markdown -> HTML (mistune, already installed with Jupyter) with a print
stylesheet and images inlined, then printed to PDF by a headless Chromium
browser (Microsoft Edge or Google Chrome, whichever is installed). No LaTeX
or extra Python packages needed. Fails loudly if no browser is found or the
PDF is not written.
"""
from __future__ import annotations

import base64
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import mistune

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "reports" / "xai_comparison_report.md"
TARGET = ROOT / "reports" / "xai_comparison_report.pdf"
BROWSERS = [
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
]
CSS = """
@page { size: A4; margin: 14mm 15mm 14mm 15mm; }
body { font-family: "Segoe UI", system-ui, sans-serif; font-size: 9.6pt; line-height: 1.38; color: #0b0b0b; }
h1 { font-size: 16pt; margin: 0 0 2pt 0; }
h2 { font-size: 11.5pt; margin: 10pt 0 3pt 0; border-bottom: 1px solid #e1e0d9; padding-bottom: 2pt; }
h3 { font-size: 10pt; margin: 7pt 0 2pt 0; }
p, li { margin: 2pt 0; }
ul, ol { margin: 2pt 0 2pt 0; padding-left: 16pt; }
table { border-collapse: collapse; margin: 4pt 0; font-size: 8.4pt; width: 100%; }
th, td { border-bottom: 1px solid #e1e0d9; padding: 2pt 5pt; text-align: left; vertical-align: top; }
th { color: #52514e; font-weight: 600; }
img { display: block; max-width: 100%; max-height: 72mm; margin: 4pt auto 0 auto; }
em { color: #52514e; }
.figure-caption, p > em:only-child { font-size: 8.2pt; }
code { font-family: Consolas, monospace; font-size: 8.4pt; }
blockquote { margin: 4pt 0; padding: 2pt 8pt; border-left: 3px solid #c3c2b7; color: #52514e; }
"""


def find_browser() -> Path:
    for path in BROWSERS:
        if path.exists():
            return path
    for name in ("msedge", "chrome", "chromium"):
        found = shutil.which(name)
        if found:
            return Path(found)
    sys.exit("No Edge/Chrome found to print the PDF; open reports/xai_comparison_report.md instead")


def inline_images(html: str, base: Path) -> str:
    """Embeds every <img src="relative.png"> as a data URI, so the HTML is self-contained."""
    def replace(match: re.Match) -> str:
        path = (base / match.group(1)).resolve()
        if not path.exists():
            sys.exit(f"Report references a missing figure: {path}")
        return f'src="data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}"'

    return re.sub(r'src="([^"]+\.png)"', replace, html)


def main() -> None:
    body = mistune.html(SOURCE.read_text(encoding="utf-8"))
    html = (f"<!doctype html><html><head><meta charset='utf-8'><title>XAI comparison report</title>"
            f"<style>{CSS}</style></head><body>{inline_images(body, SOURCE.parent)}</body></html>")
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "report.html"
        page.write_text(html, encoding="utf-8")
        if TARGET.exists():
            TARGET.unlink()
        subprocess.run([str(find_browser()), "--headless", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={TARGET}", page.as_uri()], check=True, capture_output=True, timeout=180)
    if not TARGET.exists() or TARGET.stat().st_size == 0:
        sys.exit(f"The browser did not write {TARGET}")
    pages = len(re.findall(rb"/Type\s*/Page[^s]", TARGET.read_bytes()))
    print(f"Wrote {TARGET} ({pages} pages)")


if __name__ == "__main__":
    main()
